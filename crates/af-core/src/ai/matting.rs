//! Background removal (salient object matting).

use std::time::Instant;

use fast_image_resize as fr;
use rayon::prelude::*;

use super::catalog::{Kind, ModelSpec};
use super::engine::Engine;
use super::runtime::Device;
use crate::{CancelToken, Error, Result, Rgba};

/// Model-resolution alpha prediction.
#[derive(Clone)]
pub struct Matte {
    pub alpha: Vec<f32>,
    pub w: u32,
    pub h: u32,
    pub device: Device,
    pub model_id: &'static str,
    pub ms: u64,
    pub note: Option<String>,
}

pub fn predict(engine: &Engine, spec: &'static ModelSpec, img: &Rgba, cancel: &CancelToken) -> Result<Matte> {
    let (device, mut note) = engine.pick_device(spec);
    let t0 = Instant::now();
    let res = match run_on(engine, spec, img, device, cancel) {
        Err(e @ (Error::GpuOom(_) | Error::Runtime(_))) if device == Device::Cuda => {
            log::warn!("GPU inference failed, retrying on CPU: {e}");
            engine.unload(spec.template);
            note = Some(match e {
                Error::GpuOom(_) => "Not enough free GPU memory — this image was processed on the CPU.".into(),
                _ => "GPU inference failed — this image was processed on the CPU.".into(),
            });
            run_on(engine, spec, img, Device::Cpu, cancel).map(|(a, w, h)| (a, w, h, Device::Cpu))
        }
        other => other.map(|(a, w, h)| (a, w, h, device)),
    };
    let (alpha, w, h, device) = res?;
    Ok(Matte { alpha, w, h, device, model_id: spec.id, ms: t0.elapsed().as_millis() as u64, note })
}

fn run_on(engine: &Engine, spec: &ModelSpec, img: &Rgba, device: Device, cancel: &CancelToken) -> Result<(Vec<f32>, u32, u32)> {
    let sess = engine.session(spec.template, Kind::Background, device)?;
    let m = &sess.manifest;
    let [mw, mh] = m.input.size.ok_or_else(|| Error::Runtime("matting model needs a fixed input size".into()))?;
    let input = preprocess(img, mw, mh, m.input.mean, m.input.std)?;
    cancel.check()?;
    let (shape, out) = sess.run(input, [1, 3, mh as usize, mw as usize], cancel)?;
    let n = (mw * mh) as usize;
    if out.len() < n || shape.iter().product::<usize>() < n {
        return Err(Error::Runtime("unexpected model output".into()));
    }
    Ok((out[..n].iter().map(|v| v.clamp(0.0, 1.0)).collect(), mw, mh))
}

/// Resize to the model input (antialiased bilinear, like torchvision), normalize, NCHW.
/// Transparent source pixels are composited on white first.
fn preprocess(img: &Rgba, mw: u32, mh: u32, mean: [f32; 3], std: [f32; 3]) -> Result<Vec<f32>> {
    let (w, h) = img.dimensions();
    let rgb: Vec<u8> = img
        .as_raw()
        .par_chunks(4)
        .flat_map_iter(|p| {
            let a = p[3] as u32;
            let mix = |c: u8| ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
            [mix(p[0]), mix(p[1]), mix(p[2])]
        })
        .collect();
    let src = fr::images::ImageRef::new(w, h, &rgb, fr::PixelType::U8x3).map_err(|e| Error::Runtime(e.to_string()))?;
    let mut dst = fr::images::Image::new(mw, mh, fr::PixelType::U8x3);
    let opts = fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    fr::Resizer::new().resize(&src, &mut dst, &opts).map_err(|e| Error::Runtime(e.to_string()))?;
    let px = dst.buffer();
    let n = (mw * mh) as usize;
    let mut out = vec![0f32; 3 * n];
    out.par_chunks_mut(n).enumerate().for_each(|(c, plane)| {
        let (m, s) = (mean[c], std[c]);
        for (i, v) in plane.iter_mut().enumerate() {
            *v = (px[i * 3 + c] as f32 / 255.0 - m) / s;
        }
    });
    Ok(out)
}
