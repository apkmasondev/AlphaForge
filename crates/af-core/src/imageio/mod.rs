//! Image decoding / encoding.
//!
//! * Decoding normalizes everything to RGBA8 sRGB with EXIF orientation applied, so every
//!   later step (and every output format) works in one predictable space.
//! * Encoding writes clean files without EXIF/GPS metadata.

mod decode;
mod encode;

pub use decode::{decode_bytes, decode_file, probe_format, Decoded, SourceInfo, MAX_PIXELS};
pub use encode::{encode, encode_preview_png, Chroma, EncodeOptions, Format, PngLevel};

/// True when every pixel is fully opaque.
pub fn is_opaque(img: &crate::Rgba) -> bool {
    img.as_raw().chunks_exact(4).all(|p| p[3] == 255)
}

/// Blend the image onto a solid background color (result is fully opaque).
pub fn flatten(img: &crate::Rgba, bg: [u8; 3]) -> crate::Rgba {
    use rayon::prelude::*;
    let mut out = img.clone();
    out.as_mut().par_chunks_mut(4 * 1024).for_each(|chunk| {
        for p in chunk.chunks_exact_mut(4) {
            let a = p[3] as u32;
            if a == 255 {
                continue;
            }
            for c in 0..3 {
                p[c] = ((p[c] as u32 * a + bg[c] as u32 * (255 - a) + 127) / 255) as u8;
            }
            p[3] = 255;
        }
    });
    out
}
