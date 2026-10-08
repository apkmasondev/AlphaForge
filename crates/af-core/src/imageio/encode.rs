use serde::{Deserialize, Serialize};

use crate::imageio::{flatten, is_opaque};
use crate::{Error, Result, Rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    #[default]
    Png,
    Jpeg,
    Webp,
    Avif,
}

impl Format {
    pub fn ext(self) -> &'static str {
        match self {
            Format::Png => "png",
            Format::Jpeg => "jpg",
            Format::Webp => "webp",
            Format::Avif => "avif",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Format::Png => "PNG",
            Format::Jpeg => "JPG",
            Format::Webp => "WebP",
            Format::Avif => "AVIF",
        }
    }
    pub fn supports_alpha(self) -> bool {
        !matches!(self, Format::Jpeg)
    }
    pub fn max_side(self) -> u32 {
        match self {
            Format::Png => 1 << 30,
            Format::Jpeg => 65_500,
            Format::Webp => 16_383,
            Format::Avif => 65_536,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PngLevel {
    /// Plain deflate, fastest.
    Fast,
    /// oxipng preset 1: ~13% smaller than plain deflate (default).
    #[default]
    Balanced,
    /// oxipng preset 5: smallest lossless files, slow on big images.
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Chroma {
    /// 4:4:4 at quality >= 90, otherwise 4:2:0.
    #[default]
    Auto,
    #[serde(rename = "420")]
    S420,
    #[serde(rename = "444")]
    S444,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct EncodeOptions {
    /// Concrete format; not serialized because the pipeline `Output` carries the user's choice
    /// (which may be "same as source") and resolves it per image.
    #[serde(skip)]
    pub format: Format,
    /// 1..=100, used by JPEG / WebP (lossy) / AVIF.
    pub quality: u8,
    /// WebP only: lossless mode.
    pub lossless: bool,
    pub png_level: PngLevel,
    /// PNG only: reduce to an optimized palette of at most this many colors (lossy, like pngquant).
    pub png_colors: Option<u16>,
    pub jpeg_progressive: bool,
    pub chroma: Chroma,
    /// AVIF encoder speed 1 (slow, small) ..= 10 (fast).
    pub avif_speed: u8,
    /// Color used where transparency must be removed (JPG output).
    pub background: [u8; 3],
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            format: Format::Png,
            quality: 85,
            lossless: false,
            png_level: PngLevel::Balanced,
            png_colors: None,
            jpeg_progressive: true,
            chroma: Chroma::Auto,
            avif_speed: 7,
            background: [255, 255, 255],
        }
    }
}

/// Encode an image. Transparent pixels are flattened onto `background` for formats without alpha.
pub fn encode(img: &Rgba, o: &EncodeOptions) -> Result<Vec<u8>> {
    let (w, h) = img.dimensions();
    let max = o.format.max_side();
    if w > max || h > max {
        return Err(Error::Encode(o.format.label(), format!("{} supports at most {max} px per side (image is {w} × {h})", o.format.label())));
    }
    let q = o.quality.clamp(1, 100);
    match o.format {
        Format::Png => encode_png(img, o),
        Format::Jpeg => {
            let flat;
            let src = if is_opaque(img) { img } else { flat = flatten(img, o.background); &flat };
            let rgb: Vec<u8> = src.as_raw().chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            let mut out = Vec::with_capacity((w * h) as usize / 4);
            let mut enc = jpeg_encoder::Encoder::new(&mut out, q);
            enc.set_progressive(o.jpeg_progressive);
            enc.set_optimized_huffman_tables(true);
            let s444 = match o.chroma {
                Chroma::Auto => q >= 90,
                Chroma::S444 => true,
                Chroma::S420 => false,
            };
            enc.set_sampling_factor(if s444 { jpeg_encoder::SamplingFactor::R_4_4_4 } else { jpeg_encoder::SamplingFactor::R_4_2_0 });
            enc.encode(&rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb)
                .map_err(|e| Error::Encode("JPG", e.to_string()))?;
            Ok(out)
        }
        Format::Webp => {
            let opaque = is_opaque(img);
            let rgb: Vec<u8>;
            let enc = if opaque {
                rgb = img.as_raw().chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
                webp::Encoder::new(&rgb, webp::PixelLayout::Rgb, w, h)
            } else {
                webp::Encoder::new(img.as_raw(), webp::PixelLayout::Rgba, w, h)
            };
            let mut cfg = webp::WebPConfig::new().map_err(|_| Error::Encode("WebP", "libwebp init failed".into()))?;
            cfg.lossless = o.lossless as i32;
            cfg.quality = if o.lossless { 80.0 } else { q as f32 };
            cfg.method = if o.lossless { 4 } else { 5 };
            cfg.alpha_quality = 100;
            cfg.alpha_compression = 1;
            cfg.exact = 0;
            cfg.use_sharp_yuv = 1;
            cfg.thread_level = 1;
            let mem = enc.encode_advanced(&cfg).map_err(|e| Error::Encode("WebP", format!("{e:?}")))?;
            Ok(mem.to_vec())
        }
        Format::Avif => {
            let enc = ravif::Encoder::new()
                .with_quality(q as f32)
                .with_alpha_quality((q as f32 + 10.0).min(100.0))
                .with_speed(o.avif_speed.clamp(1, 10))
                .with_alpha_color_mode(ravif::AlphaColorMode::UnassociatedClean);
            let res = if is_opaque(img) {
                let px: Vec<rgb::RGB8> = img.as_raw().chunks_exact(4).map(|p| rgb::RGB8::new(p[0], p[1], p[2])).collect();
                enc.encode_rgb(imgref::Img::new(&px[..], w as usize, h as usize))
            } else {
                let px: Vec<rgb::RGBA8> = img.as_raw().chunks_exact(4).map(|p| rgb::RGBA8::new(p[0], p[1], p[2], p[3])).collect();
                enc.encode_rgba(imgref::Img::new(&px[..], w as usize, h as usize))
            };
            res.map(|r| r.avif_file).map_err(|e| Error::Encode("AVIF", e.to_string()))
        }
    }
}

fn encode_png(img: &Rgba, o: &EncodeOptions) -> Result<Vec<u8>> {
    let (w, h) = img.dimensions();
    // Optional lossy palette reduction (great for flat graphics / cut-outs).
    if let Some(n) = o.png_colors {
        let (palette, indices, trns) = crate::ops::quantize::quantize(img, n.clamp(2, 256) as usize);
        let mut out = Vec::new();
        {
            let mut e = png::Encoder::new(&mut out, w, h);
            e.set_color(png::ColorType::Indexed);
            e.set_depth(png::BitDepth::Eight);
            e.set_palette(palette);
            if let Some(t) = trns {
                e.set_trns(t);
            }
            e.set_compression(png::Compression::High);
            let mut wr = e.write_header().map_err(|e| Error::Encode("PNG", e.to_string()))?;
            wr.write_image_data(&indices).map_err(|e| Error::Encode("PNG", e.to_string()))?;
            wr.finish().map_err(|e| Error::Encode("PNG", e.to_string()))?;
        }
        return optimize_png(out, o.png_level);
    }
    let opaque = is_opaque(img);
    let mut out = Vec::new();
    {
        let mut e = png::Encoder::new(&mut out, w, h);
        let data: std::borrow::Cow<[u8]> = if opaque {
            e.set_color(png::ColorType::Rgb);
            img.as_raw().chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect::<Vec<u8>>().into()
        } else {
            e.set_color(png::ColorType::Rgba);
            // Zero the color of fully transparent pixels: invisible, compresses much better.
            let mut v = img.as_raw().clone();
            for p in v.chunks_exact_mut(4) {
                if p[3] == 0 {
                    p[0] = 0;
                    p[1] = 0;
                    p[2] = 0;
                }
            }
            v.into()
        };
        e.set_depth(png::BitDepth::Eight);
        e.set_compression(if o.png_level == PngLevel::Fast { png::Compression::Balanced } else { png::Compression::Fast });
        let mut wr = e.write_header().map_err(|e| Error::Encode("PNG", e.to_string()))?;
        wr.write_image_data(&data).map_err(|e| Error::Encode("PNG", e.to_string()))?;
        wr.finish().map_err(|e| Error::Encode("PNG", e.to_string()))?;
    }
    optimize_png(out, o.png_level)
}

fn optimize_png(data: Vec<u8>, level: PngLevel) -> Result<Vec<u8>> {
    let preset = match level {
        PngLevel::Fast => return Ok(data),
        PngLevel::Balanced => 1,
        PngLevel::Max => 5,
    };
    let mut opts = oxipng::Options::from_preset(preset);
    opts.strip = oxipng::StripChunks::Safe;
    match oxipng::optimize_from_memory(&data, &opts) {
        Ok(v) if v.len() < data.len() => Ok(v),
        _ => Ok(data),
    }
}

/// Very fast PNG used only to hand pixels to the UI (localhost transfer, size irrelevant).
pub fn encode_preview_png(img: &Rgba) -> Result<Vec<u8>> {
    let (w, h) = img.dimensions();
    let mut out = Vec::with_capacity((w * h * 2) as usize);
    {
        let mut e = png::Encoder::new(&mut out, w, h);
        e.set_color(png::ColorType::Rgba);
        e.set_depth(png::BitDepth::Eight);
        e.set_compression(png::Compression::Fastest);
        let mut wr = e.write_header().map_err(|e| Error::Encode("PNG", e.to_string()))?;
        wr.write_image_data(img.as_raw()).map_err(|e| Error::Encode("PNG", e.to_string()))?;
        wr.finish().map_err(|e| Error::Encode("PNG", e.to_string()))?;
    }
    Ok(out)
}
