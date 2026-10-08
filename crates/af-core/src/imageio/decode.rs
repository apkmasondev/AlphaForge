use std::io::Cursor;
use std::path::Path;

use image::{DynamicImage, ImageDecoder, ImageReader, Limits};
use serde::Serialize;

use crate::{Error, Result, Rgba};

/// Hard safety limit against decompression bombs and accidental huge allocations.
pub const MAX_PIXELS: u64 = 200_000_000;
const MAX_SIDE: u32 = 40_000;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceInfo {
    /// Human-readable container format ("JPEG", "PNG", ...).
    pub format: &'static str,
    pub width: u32,
    pub height: u32,
    /// The source has an alpha channel with at least one non-opaque pixel.
    pub has_alpha: bool,
    /// Bits per channel of the source (8 or 16). 16-bit sources are reduced to 8 bit.
    pub bit_depth: u8,
    /// An embedded ICC profile was converted to sRGB.
    pub color_converted: bool,
}

pub struct Decoded {
    pub image: Rgba,
    pub info: SourceInfo,
}

/// Detect the container format from magic bytes. Returns `None` for unknown data.
pub fn probe_format(bytes: &[u8]) -> Option<&'static str> {
    if is_avif(bytes) {
        return Some("AVIF");
    }
    image::guess_format(bytes).ok().map(format_name)
}

fn format_name(f: image::ImageFormat) -> &'static str {
    use image::ImageFormat::*;
    match f {
        Png => "PNG",
        Jpeg => "JPEG",
        WebP => "WebP",
        Gif => "GIF",
        Bmp => "BMP",
        Tiff => "TIFF",
        Avif => "AVIF",
        _ => "image",
    }
}

fn is_avif(b: &[u8]) -> bool {
    b.len() > 12 && &b[4..8] == b"ftyp" && (&b[8..12] == b"avif" || &b[8..12] == b"avis" || b[8..b.len().min(64)].windows(4).any(|w| w == b"avif"))
}

pub fn decode_file(path: &Path) -> Result<Decoded> {
    let meta = std::fs::metadata(path).map_err(|e| Error::Path(path.to_path_buf(), e.to_string()))?;
    if meta.len() > 2_000_000_000 {
        return Err(Error::decode("file is larger than 2 GB"));
    }
    let bytes = std::fs::read(path).map_err(|e| Error::Path(path.to_path_buf(), e.to_string()))?;
    decode_bytes(&bytes)
}

pub fn decode_bytes(bytes: &[u8]) -> Result<Decoded> {
    if bytes.len() < 16 {
        return Err(Error::Decode(Some("file is empty or truncated".into())));
    }
    if is_avif(bytes) {
        return decode_avif(bytes);
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| Error::decode(e.to_string()))?;
    let fmt = reader.format().ok_or(Error::Decode(None))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    limits.max_alloc = Some(MAX_PIXELS * 8 + (64 << 20));
    reader.limits(limits);

    let mut decoder = reader.into_decoder().map_err(map_img_err)?;
    let (w, h) = decoder.dimensions();
    check_size(w, h)?;
    let icc = decoder.icc_profile().ok().flatten();
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let color = decoder.color_type();
    let mut img = DynamicImage::from_decoder(decoder).map_err(map_img_err)?;
    img.apply_orientation(orientation);

    let bit_depth = if color.bytes_per_pixel() / color.channel_count().max(1) >= 2 { 16 } else { 8 };
    let mut rgba = img.into_rgba8();
    let color_converted = match icc {
        Some(ref p) if color.has_color() => convert_to_srgb(&mut rgba, p),
        _ => false,
    };
    let has_alpha = color.has_alpha() && !crate::imageio::is_opaque(&rgba);
    let info = SourceInfo { format: format_name(fmt), width: rgba.width(), height: rgba.height(), has_alpha, bit_depth, color_converted };
    Ok(Decoded { image: rgba, info })
}

fn decode_avif(bytes: &[u8]) -> Result<Decoded> {
    use avif_decode::{Decoder, Image};
    let dec = Decoder::from_avif(bytes).map_err(|e| Error::decode(format!("AVIF: {e}")))?;
    let img = dec.to_image().map_err(|e| Error::decode(format!("AVIF: {e}")))?;
    let (w, h, depth, data): (usize, usize, u8, Vec<u8>) = match img {
        Image::Rgb8(i) => (i.width(), i.height(), 8, i.pixels().flat_map(|p| [p.r, p.g, p.b, 255]).collect()),
        Image::Rgba8(i) => (i.width(), i.height(), 8, i.pixels().flat_map(|p| [p.r, p.g, p.b, p.a]).collect()),
        Image::Rgb16(i) => (i.width(), i.height(), 16, i.pixels().flat_map(|p| [to8(p.r), to8(p.g), to8(p.b), 255]).collect()),
        Image::Rgba16(i) => (i.width(), i.height(), 16, i.pixels().flat_map(|p| [to8(p.r), to8(p.g), to8(p.b), to8(p.a)]).collect()),
        Image::Gray8(i) => (i.width(), i.height(), 8, i.pixels().flat_map(|p| { let v = p.value(); [v, v, v, 255] }).collect()),
        Image::Gray16(i) => (i.width(), i.height(), 16, i.pixels().flat_map(|p| { let v = to8(p.value()); [v, v, v, 255] }).collect()),
    };
    check_size(w as u32, h as u32)?;
    let rgba = Rgba::from_raw(w as u32, h as u32, data).ok_or(Error::Decode(None))?;
    let has_alpha = !crate::imageio::is_opaque(&rgba);
    let info = SourceInfo { format: "AVIF", width: w as u32, height: h as u32, has_alpha, bit_depth: depth, color_converted: false };
    Ok(Decoded { image: rgba, info })
}

#[inline]
fn to8(v: u16) -> u8 {
    ((v as u32 * 255 + 32767) / 65535) as u8
}

fn check_size(w: u32, h: u32) -> Result<()> {
    if w == 0 || h == 0 {
        return Err(Error::decode("image has zero size"));
    }
    if w as u64 * h as u64 > MAX_PIXELS || w > MAX_SIDE || h > MAX_SIDE {
        return Err(Error::TooLarge(w, h, (MAX_PIXELS / 1_000_000) as u32));
    }
    Ok(())
}

fn map_img_err(e: image::ImageError) -> Error {
    match e {
        image::ImageError::Limits(_) => Error::decode("image exceeds the safety limits (too large)"),
        image::ImageError::Unsupported(u) => Error::decode(format!("unsupported variant: {u}")),
        other => Error::decode(other.to_string()),
    }
}

/// Convert pixels from the embedded ICC profile to sRGB. Returns true if a conversion ran.
fn convert_to_srgb(img: &mut Rgba, icc: &[u8]) -> bool {
    // Most files already carry an sRGB profile; skip the work for those.
    if icc_is_srgb(icc) {
        return false;
    }
    let Some(src) = qcms::Profile::new_from_slice(icc, false) else { return false };
    let dst = qcms::Profile::new_sRGB();
    let Some(xf) = qcms::Transform::new(&src, &dst, qcms::DataType::RGBA8, qcms::Intent::Perceptual) else { return false };
    use rayon::prelude::*;
    img.as_mut().par_chunks_mut(4 * 4096).for_each(|c| xf.apply(c));
    true
}

fn icc_is_srgb(icc: &[u8]) -> bool {
    // The profile description tag usually contains "sRGB" for sRGB-family profiles
    // (e.g. "sRGB IEC61966-2.1", "sRGB built-in"). This avoids re-quantizing such images.
    let hay = &icc[..icc.len().min(4096)];
    hay.windows(4).any(|w| w == b"sRGB") || hay.windows(8).any(|w| w == b"s\0R\0G\0B\0")
}
