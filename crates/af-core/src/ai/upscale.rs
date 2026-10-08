//! Tiled super-resolution and AI denoising (Real-ESRGAN family).

use fast_image_resize as fr;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::catalog::{self, Kind, ModelSpec};
use super::engine::Engine;
use super::runtime::Device;
use crate::cancel::Progress;
use crate::{CancelToken, Error, Result, Rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SrModel {
    #[default]
    General,
    Photo,
    Illustration,
}

/// Largest result we allow (pixels) to keep memory predictable.
pub const MAX_OUTPUT_PIXELS: u64 = 160_000_000;

pub struct SrResult {
    pub image: Rgba,
    pub device: Device,
    pub note: Option<String>,
}

fn pick(model: SrModel, scale: u32, denoise: f32) -> (&'static ModelSpec, &'static str) {
    match model {
        SrModel::General => {
            let spec = catalog::get("sr-general").unwrap();
            // Real-ESRGAN: x4v3 has strong denoising, the "wdn" variant keeps more texture/noise.
            let t = if denoise < 0.25 {
                catalog::SR_GENERAL_WEAK_DENOISE
            } else if denoise < 0.75 {
                catalog::SR_GENERAL_MID_DENOISE
            } else {
                spec.template
            };
            (spec, t)
        }
        SrModel::Photo if scale == 2 => {
            let s = catalog::get("sr-photo-x2").unwrap();
            (s, s.template)
        }
        SrModel::Photo => {
            let s = catalog::get("sr-photo").unwrap();
            (s, s.template)
        }
        SrModel::Illustration => {
            let s = catalog::get("sr-anime").unwrap();
            (s, s.template)
        }
    }
}

/// Upscale by 2 or 4.
pub fn upscale(engine: &Engine, model: SrModel, scale: u32, denoise: f32, img: &Rgba, cancel: &CancelToken, progress: Progress) -> Result<SrResult> {
    let scale = if scale >= 3 { 4 } else { 2 };
    let (w, h) = img.dimensions();
    let out_px = w as u64 * h as u64 * (scale * scale) as u64;
    if out_px > MAX_OUTPUT_PIXELS {
        return Err(Error::Limit(format!(
            "Upscaling would create a {} × {} px image ({} MP), above the 160 MP limit. Resize the image first or use 2×.",
            w * scale,
            h * scale,
            out_px / 1_000_000
        )));
    }
    let (spec, template) = pick(model, scale, denoise);
    run_with_fallback(engine, spec, template, img, scale, cancel, progress)
}

/// AI noise reduction at the original size (general model, output downsampled 4× per tile).
pub fn denoise(engine: &Engine, strength: f32, img: &Rgba, cancel: &CancelToken, progress: Progress) -> Result<SrResult> {
    let (spec, template) = pick(SrModel::General, 4, strength);
    run_with_fallback(engine, spec, template, img, 1, cancel, progress)
}

fn run_with_fallback(engine: &Engine, spec: &'static ModelSpec, template: &str, img: &Rgba, out_scale: u32, cancel: &CancelToken, progress: Progress) -> Result<SrResult> {
    let (device, mut note) = engine.pick_device(spec);
    let r = run(engine, template, img, out_scale, device, cancel, progress);
    let (image, device) = match r {
        Err(e @ (Error::GpuOom(_) | Error::Runtime(_))) if device == Device::Cuda => {
            log::warn!("GPU upscale failed, retrying on CPU: {e}");
            engine.unload(template);
            note = Some(match e {
                Error::GpuOom(_) => "GPU ran out of memory — this image was upscaled on the CPU.".into(),
                _ => "GPU processing failed — this image was upscaled on the CPU.".into(),
            });
            (run(engine, template, img, out_scale, Device::Cpu, cancel, progress)?, Device::Cpu)
        }
        other => (other?, device),
    };
    Ok(SrResult { image, device, note })
}

fn run(engine: &Engine, template: &str, img: &Rgba, out_scale: u32, device: Device, cancel: &CancelToken, progress: Progress) -> Result<Rgba> {
    let sess = engine.session(template, Kind::Upscale, device)?;
    let net = sess.manifest.scale.unwrap_or(4);
    let heavy = template.starts_with("RealESRGAN_x");
    let max_tile: u32 = match (device, heavy) {
        (Device::Cuda, false) => 640,
        (Device::Cuda, true) => 448,
        (Device::Cpu, _) => 256,
    };
    let pad: u32 = 16;
    let (w, h) = img.dimensions();
    let opaque = crate::imageio::is_opaque(img);
    let src = if opaque { img.clone() } else { bleed_rgb(img) };
    let (ow, oh) = (w * out_scale, h * out_scale);
    let mut out = vec![0u8; (ow as usize) * (oh as usize) * 3];
    // Balanced tiles: same count as with `max_tile`, but evenly sized so edge tiles are not
    // mostly padding.
    let tiles_x = w.div_ceil(max_tile);
    let tiles_y = h.div_ceil(max_tile);
    let tile_w = w.div_ceil(tiles_x);
    let tile_h = h.div_ceil(tiles_y);
    let total = (tiles_x * tiles_y) as f32;
    let mut done = 0f32;
    // Every inference uses the same input shape (tile + padding, edge-replicated at the image
    // border). Varying shapes make cuDNN re-plan kernels for each new size, which is several
    // times slower than the inference itself.
    let even = |v: u32| v + (v % 2);
    let rw = even(tile_w + 2 * pad);
    let rh = even(tile_h + 2 * pad);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            cancel.check()?;
            let (x0, y0) = (tx * tile_w, ty * tile_h);
            let (tw, th) = (tile_w.min(w - x0), tile_h.min(h - y0));
            // region origin (may be negative = replicated border)
            let rx0 = (x0 as i64 - pad as i64).min(w as i64 + pad as i64 - rw as i64);
            let ry0 = (y0 as i64 - pad as i64).min(h as i64 + pad as i64 - rh as i64);
            let input = region_tensor(&src, rx0, ry0, rw, rh);
            let (shape, y) = sess.run(input, [1, 3, rh as usize, rw as usize], cancel)?;
            let (oh_t, ow_t) = (shape[2], shape[3]);
            if oh_t != (rh * net) as usize || ow_t != (rw * net) as usize {
                return Err(Error::Runtime("unexpected upscaler output size".into()));
            }
            let mut reg = to_rgb8(&y, ow_t, oh_t);
            let mut regw = ow_t as u32;
            if out_scale != net {
                let (nw, nh) = (rw * out_scale, rh * out_scale);
                reg = resize_rgb8(&reg, regw, oh_t as u32, nw, nh)?;
                regw = nw;
            }
            // paste the core (without padding) into the output
            let cx = ((x0 as i64 - rx0) as u32) * out_scale;
            let cy = ((y0 as i64 - ry0) as u32) * out_scale;
            let (cw, ch) = (tw * out_scale, th * out_scale);
            for yy in 0..ch {
                let srow = ((cy + yy) * regw + cx) as usize * 3;
                let drow = ((y0 * out_scale + yy) * ow + x0 * out_scale) as usize * 3;
                out[drow..drow + cw as usize * 3].copy_from_slice(&reg[srow..srow + cw as usize * 3]);
            }
            done += 1.0;
            progress(done / total, "Upscaling");
        }
    }
    // Alpha: high-quality resampling of the original alpha.
    let mut rgba = vec![255u8; (ow as usize) * (oh as usize) * 4];
    rgba.par_chunks_mut(4).zip(out.par_chunks(3)).for_each(|(d, s)| d[..3].copy_from_slice(s));
    if !opaque {
        let a: Vec<u8> = img.as_raw().chunks_exact(4).map(|p| p[3]).collect();
        let s = fr::images::ImageRef::new(w, h, &a, fr::PixelType::U8).map_err(|e| Error::Runtime(e.to_string()))?;
        let mut d = fr::images::Image::new(ow, oh, fr::PixelType::U8);
        let opts = fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3));
        fr::Resizer::new().resize(&s, &mut d, &opts).map_err(|e| Error::Runtime(e.to_string()))?;
        rgba.par_chunks_mut(4).zip(d.buffer().par_iter()).for_each(|(p, a)| p[3] = *a);
    }
    Rgba::from_raw(ow, oh, rgba).ok_or_else(|| Error::Runtime("buffer".into()))
}

/// NCHW float tensor of a `rw` × `rh` region starting at (x0, y0); outside pixels are
/// edge-replicated.
fn region_tensor(img: &Rgba, x0: i64, y0: i64, rw: u32, rh: u32) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let (rw, rh) = (rw as usize, rh as usize);
    let n = rw * rh;
    let raw = img.as_raw();
    let mut t = vec![0f32; 3 * n];
    for y in 0..rh {
        let sy = (y0 + y as i64).clamp(0, h as i64 - 1) as usize;
        for x in 0..rw {
            let sx = (x0 + x as i64).clamp(0, w as i64 - 1) as usize;
            let p = (sy * w as usize + sx) * 4;
            let i = y * rw + x;
            t[i] = raw[p] as f32 / 255.0;
            t[n + i] = raw[p + 1] as f32 / 255.0;
            t[2 * n + i] = raw[p + 2] as f32 / 255.0;
        }
    }
    t
}

fn to_rgb8(y: &[f32], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let mut out = vec![0u8; n * 3];
    out.par_chunks_mut(3).enumerate().for_each(|(i, p)| {
        for c in 0..3 {
            p[c] = (y[c * n + i].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    });
    out
}

fn resize_rgb8(src: &[u8], sw: u32, sh: u32, w: u32, h: u32) -> Result<Vec<u8>> {
    let s = fr::images::ImageRef::new(sw, sh, src, fr::PixelType::U8x3).map_err(|e| Error::Runtime(e.to_string()))?;
    let mut d = fr::images::Image::new(w, h, fr::PixelType::U8x3);
    let opts = fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3));
    fr::Resizer::new().resize(&s, &mut d, &opts).map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(d.into_vec())
}

/// Fill the RGB of (semi-)transparent pixels with nearby visible colors so the network does
/// not see black/garbage around cut-outs (which would create dark halos after upscaling).
pub fn bleed_rgb(img: &Rgba) -> Rgba {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let a: Vec<f32> = img.as_raw().chunks_exact(4).map(|p| p[3] as f32 / 255.0).collect();
    let mut planes: Vec<Vec<f32>> = (0..3).map(|c| img.as_raw().chunks_exact(4).zip(&a).map(|(p, a)| p[c] as f32 * a).collect()).collect();
    let mut acc_a = a.clone();
    // Progressive blurs reach further into large transparent regions.
    let mut fill: Vec<Vec<f32>> = vec![vec![0f32; w * h]; 3];
    let mut filled = vec![0f32; w * h];
    for sigma in [2.0f32, 8.0, 32.0, 128.0] {
        let ba = crate::mask::gaussian_blur(&acc_a, w, h, sigma);
        let bp: Vec<Vec<f32>> = planes.par_iter().map(|p| crate::mask::gaussian_blur(p, w, h, sigma)).collect();
        for i in 0..w * h {
            if filled[i] < 1.0 && ba[i] > 1e-4 {
                let take = 1.0 - filled[i];
                for c in 0..3 {
                    fill[c][i] += take * bp[c][i] / ba[i];
                }
                filled[i] = 1.0;
            }
        }
        acc_a = ba;
        planes = bp;
    }
    let mut out = img.clone();
    out.as_mut().par_chunks_mut(4).enumerate().for_each(|(i, p)| {
        let al = p[3] as f32 / 255.0;
        if al >= 1.0 {
            return;
        }
        for c in 0..3 {
            let f = if filled[i] > 0.0 { fill[c][i] } else { 0.0 };
            p[c] = (p[c] as f32 * al + f * (1.0 - al)).round().clamp(0.0, 255.0) as u8;
        }
    });
    out
}
