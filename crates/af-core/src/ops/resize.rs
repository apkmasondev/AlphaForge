use fast_image_resize as fr;
use serde::{Deserialize, Serialize};

use crate::{Error, Result, Rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Filter {
    /// Sharp, high quality (default for photos).
    #[default]
    Lanczos,
    /// Bicubic Catmull-Rom.
    Bicubic,
    Bilinear,
    /// Hard pixel edges (pixel art, masks).
    Nearest,
}

impl Filter {
    fn alg(self) -> fr::ResizeAlg {
        match self {
            Filter::Lanczos => fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3),
            Filter::Bicubic => fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom),
            Filter::Bilinear => fr::ResizeAlg::Convolution(fr::FilterType::Bilinear),
            Filter::Nearest => fr::ResizeAlg::Nearest,
        }
    }
}

/// Resize RGBA8 with premultiplied-alpha filtering (no dark fringes around transparent edges).
pub fn resize_rgba(img: &Rgba, w: u32, h: u32, filter: Filter) -> Result<Rgba> {
    let (sw, sh) = img.dimensions();
    if (sw, sh) == (w, h) {
        return Ok(img.clone());
    }
    if w == 0 || h == 0 {
        return Err(Error::Invalid("target size must be at least 1 × 1 px".into()));
    }
    let src = fr::images::ImageRef::new(sw, sh, img.as_raw(), fr::PixelType::U8x4).map_err(|e| Error::Invalid(e.to_string()))?;
    let mut dst = fr::images::Image::new(w, h, fr::PixelType::U8x4);
    let opaque = crate::imageio::is_opaque(img);
    let opts = fr::ResizeOptions::new().resize_alg(filter.alg()).use_alpha(!opaque);
    fr::Resizer::new().resize(&src, &mut dst, &opts).map_err(|e| Error::Invalid(e.to_string()))?;
    Rgba::from_raw(w, h, dst.into_vec()).ok_or_else(|| Error::Invalid("resize buffer".into()))
}

/// Resize a single-channel f32 plane (used for masks).
pub fn resize_f32_plane(src: &[f32], sw: u32, sh: u32, w: u32, h: u32, filter: Filter) -> Vec<f32> {
    if (sw, sh) == (w, h) {
        return src.to_vec();
    }
    let bytes: &[u8] = bytemuck::cast_slice(src);
    let s = fr::images::ImageRef::new(sw, sh, bytes, fr::PixelType::F32).expect("valid plane");
    let mut d = fr::images::Image::new(w, h, fr::PixelType::F32);
    let opts = fr::ResizeOptions::new().resize_alg(filter.alg()).use_alpha(false);
    fr::Resizer::new().resize(&s, &mut d, &opts).expect("resize plane");
    let v = d.into_vec();
    let mut out: Vec<f32> = bytemuck::pod_collect_to_vec(&v);
    for x in out.iter_mut() {
        *x = x.clamp(0.0, 1.0);
    }
    out
}
