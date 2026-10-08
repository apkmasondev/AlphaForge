//! Monocular depth estimation (Depth Anything V2 Small) for depth-aware background blur.

use fast_image_resize as fr;
use rayon::prelude::*;

use super::catalog::{self, Kind};
use super::engine::Engine;
use super::matting::Matte;
use super::runtime::Device;
use crate::{CancelToken, Error, Result, Rgba};

pub fn spec() -> &'static catalog::ModelSpec {
    catalog::get("depth").expect("depth model in catalog")
}

/// Relative depth at model resolution, normalized to 0 (far) ..= 1 (near). Returned as a
/// [`Matte`] so the stage cache and resizing helpers can be shared with background masks.
pub fn predict(engine: &Engine, img: &Rgba, cancel: &CancelToken) -> Result<Matte> {
    let spec = spec();
    let (device, mut note) = engine.pick_device(spec);
    let t0 = std::time::Instant::now();
    let r = match run_on(engine, img, device, cancel) {
        Err(e @ (Error::GpuOom(_) | Error::Runtime(_))) if device == Device::Cuda => {
            log::warn!("GPU depth failed, retrying on CPU: {e}");
            engine.unload(spec.template);
            note = Some("GPU processing failed — depth was estimated on the CPU.".into());
            run_on(engine, img, Device::Cpu, cancel).map(|v| (v, Device::Cpu))
        }
        other => other.map(|v| (v, device)),
    };
    let ((alpha, w, h), device) = r?;
    Ok(Matte { alpha, w, h, device, model_id: spec.id, ms: t0.elapsed().as_millis() as u64, note })
}

fn run_on(engine: &Engine, img: &Rgba, device: Device, cancel: &CancelToken) -> Result<(Vec<f32>, u32, u32)> {
    let sess = engine.session(spec().template, Kind::Depth, device)?;
    let m = &sess.manifest;
    let [mw, mh] = m.input.size.ok_or_else(|| Error::Runtime("depth model needs a fixed input size".into()))?;
    let input = preprocess(img, mw, mh, m.input.mean, m.input.std)?;
    cancel.check()?;
    let (shape, out) = sess.run(input, [1, 3, mh as usize, mw as usize], cancel)?;
    let n = (mw * mh) as usize;
    if out.len() < n || shape.iter().product::<usize>() < n {
        return Err(Error::Runtime("unexpected depth output".into()));
    }
    Ok((normalize(&out[..n]), mw, mh))
}

/// Min-max normalize with a little outlier clipping (1st / 99th percentile).
fn normalize(v: &[f32]) -> Vec<f32> {
    let mut s: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return vec![0.5; v.len()];
    }
    s.sort_by(|a, b| a.total_cmp(b));
    let lo = s[s.len() / 100];
    let hi = s[(s.len() * 99 / 100).min(s.len() - 1)];
    let span = (hi - lo).max(1e-6);
    v.iter().map(|x| if x.is_finite() { ((x - lo) / span).clamp(0.0, 1.0) } else { 0.0 }).collect()
}

/// Resize to the model input (stretching to the fixed square), normalize, NCHW.
fn preprocess(img: &Rgba, mw: u32, mh: u32, mean: [f32; 3], std: [f32; 3]) -> Result<Vec<f32>> {
    let (w, h) = img.dimensions();
    let rgb: Vec<u8> = img.as_raw().par_chunks(4).flat_map_iter(|p| [p[0], p[1], p[2]]).collect();
    let src = fr::images::ImageRef::new(w, h, &rgb, fr::PixelType::U8x3).map_err(|e| Error::Runtime(e.to_string()))?;
    let mut dst = fr::images::Image::new(mw, mh, fr::PixelType::U8x3);
    let opts = fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom));
    fr::Resizer::new().resize(&src, &mut dst, &opts).map_err(|e| Error::Runtime(e.to_string()))?;
    let px = dst.buffer();
    let n = (mw * mh) as usize;
    let mut out = vec![0f32; 3 * n];
    out.par_chunks_mut(n).enumerate().for_each(|(c, plane)| {
        for (i, v) in plane.iter_mut().enumerate() {
            *v = (px[i * 3 + c] as f32 / 255.0 - mean[c]) / std[c];
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalize_clips_outliers() {
        let mut v: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        v[0] = -1e6;
        v[999] = 1e6;
        let n = super::normalize(&v);
        assert!(n.iter().all(|x| (0.0..=1.0).contains(x)));
        assert_eq!(n[0], 0.0);
        assert_eq!(n[999], 1.0);
        assert!((n[500] - 0.5).abs() < 0.02);
    }
}
