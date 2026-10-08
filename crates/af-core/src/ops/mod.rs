//! Fast classic (non-AI) image operations on RGBA8 sRGB images.

pub mod quantize;
mod resize;

pub use resize::{resize_f32_plane, resize_rgba, Filter};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Rgba;

/// Axis-aligned rectangle in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Bounding box of pixels whose alpha is greater than `threshold`. `None` if fully transparent.
pub fn alpha_bbox(img: &Rgba, threshold: u8) -> Option<Rect> {
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    let stride = w as usize * 4;
    let rows: Vec<Option<(u32, u32)>> = (0..h as usize)
        .into_par_iter()
        .map(|y| {
            let row = &raw[y * stride..(y + 1) * stride];
            let first = row.chunks_exact(4).position(|p| p[3] > threshold)?;
            let last = row.chunks_exact(4).rposition(|p| p[3] > threshold)?;
            Some((first as u32, last as u32))
        })
        .collect();
    let y0 = rows.iter().position(|r| r.is_some())? as u32;
    let y1 = rows.iter().rposition(|r| r.is_some())? as u32;
    let (mut x0, mut x1) = (u32::MAX, 0);
    for (a, b) in rows.iter().flatten() {
        x0 = x0.min(*a);
        x1 = x1.max(*b);
    }
    Some(Rect { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 })
}

/// Bounding box of pixels that differ from the given solid color (for trimming opaque images).
pub fn color_bbox(img: &Rgba, color: [u8; 4], tolerance: u8) -> Option<Rect> {
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    let stride = w as usize * 4;
    let differs = |p: &[u8]| (0..4).any(|c| (p[c] as i16 - color[c] as i16).unsigned_abs() as u8 > tolerance);
    let rows: Vec<Option<(u32, u32)>> = (0..h as usize)
        .into_par_iter()
        .map(|y| {
            let row = &raw[y * stride..(y + 1) * stride];
            let first = row.chunks_exact(4).position(differs)?;
            let last = row.chunks_exact(4).rposition(differs)?;
            Some((first as u32, last as u32))
        })
        .collect();
    let y0 = rows.iter().position(|r| r.is_some())? as u32;
    let y1 = rows.iter().rposition(|r| r.is_some())? as u32;
    let (mut x0, mut x1) = (u32::MAX, 0);
    for (a, b) in rows.iter().flatten() {
        x0 = x0.min(*a);
        x1 = x1.max(*b);
    }
    Some(Rect { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 })
}

pub fn crop(img: &Rgba, r: Rect) -> Rgba {
    let (w, _) = img.dimensions();
    let mut out = Rgba::new(r.w, r.h);
    let src = img.as_raw();
    let stride = w as usize * 4;
    out.as_mut().par_chunks_mut(r.w as usize * 4).enumerate().for_each(|(y, row)| {
        let sy = r.y as usize + y;
        let start = sy * stride + r.x as usize * 4;
        row.copy_from_slice(&src[start..start + r.w as usize * 4]);
    });
    out
}

/// Add margins around the image filled with `fill` (transparent by default).
pub fn pad(img: &Rgba, top: u32, right: u32, bottom: u32, left: u32, fill: [u8; 4]) -> Rgba {
    let (w, h) = img.dimensions();
    let (nw, nh) = (w + left + right, h + top + bottom);
    let mut out = Rgba::from_pixel(nw, nh, image::Rgba(fill));
    blit(&mut out, img, left as i64, top as i64);
    out
}

/// Copy `src` into `dst` at (x, y), clipping to bounds (no blending).
pub fn blit(dst: &mut Rgba, src: &Rgba, x: i64, y: i64) {
    let (dw, dh) = (dst.width() as i64, dst.height() as i64);
    let (sw, sh) = (src.width() as i64, src.height() as i64);
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = (x + sw).min(dw);
    let y1 = (y + sh).min(dh);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let n = ((x1 - x0) * 4) as usize;
    let dstride = (dw * 4) as usize;
    let sstride = (sw * 4) as usize;
    let s = src.as_raw();
    let d = dst.as_mut();
    for yy in y0..y1 {
        let so = ((yy - y) as usize) * sstride + ((x0 - x) as usize) * 4;
        let doff = (yy as usize) * dstride + (x0 as usize) * 4;
        d[doff..doff + n].copy_from_slice(&s[so..so + n]);
    }
}

/// Alpha-composite `src` over a solid canvas of size (cw, ch) at (x, y).
pub fn place_on_canvas(src: &Rgba, cw: u32, ch: u32, x: i64, y: i64, bg: [u8; 4]) -> Rgba {
    let mut out = Rgba::from_pixel(cw, ch, image::Rgba(bg));
    if bg[3] == 0 {
        blit(&mut out, src, x, y);
        return out;
    }
    let (sw, sh) = (src.width() as i64, src.height() as i64);
    for sy in 0..sh {
        let dy = y + sy;
        if dy < 0 || dy >= ch as i64 {
            continue;
        }
        for sx in 0..sw {
            let dx = x + sx;
            if dx < 0 || dx >= cw as i64 {
                continue;
            }
            let s = src.get_pixel(sx as u32, sy as u32).0;
            let d = out.get_pixel_mut(dx as u32, dy as u32);
            *d = image::Rgba(over(s, d.0));
        }
    }
    out
}

/// Straight-alpha "source over" compositing of one pixel.
#[inline]
pub fn over(s: [u8; 4], d: [u8; 4]) -> [u8; 4] {
    let sa = s[3] as f32 / 255.0;
    let da = d[3] as f32 / 255.0;
    let oa = sa + da * (1.0 - sa);
    if oa <= 0.0 {
        return [0, 0, 0, 0];
    }
    let mut o = [0u8; 4];
    for c in 0..3 {
        let v = (s[c] as f32 * sa + d[c] as f32 * da * (1.0 - sa)) / oa;
        o[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    o[3] = (oa * 255.0).round() as u8;
    o
}

/// Unsharp mask on RGB (alpha untouched). `amount` 0..=2, `radius` in px.
pub fn sharpen(img: &Rgba, amount: f32, radius: f32) -> Rgba {
    if amount <= 0.0 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    let n = (w * h) as usize;
    let mut planes: Vec<Vec<f32>> = (0..3).map(|c| img.as_raw().chunks_exact(4).map(|p| p[c] as f32).collect()).collect();
    let r = radius.max(0.5);
    let blurred: Vec<Vec<f32>> = planes.par_iter().map(|p| crate::mask::gaussian_blur(p, w as usize, h as usize, r)).collect();
    for c in 0..3 {
        let b = &blurred[c];
        planes[c].par_iter_mut().zip(b.par_iter()).for_each(|(v, bl)| {
            let d = *v - *bl;
            // small threshold to avoid amplifying noise
            if d.abs() > 2.0 {
                *v += amount * d;
            }
        });
    }
    let mut out = img.clone();
    let o = out.as_mut();
    for i in 0..n {
        for c in 0..3 {
            o[i * 4 + c] = planes[c][i].round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Stretch levels so that the darkest/brightest 0.3% of (visible) pixels map to 0/255.
/// Uses a shared luminance-based curve so hues are preserved.
pub fn auto_levels(img: &Rgba) -> Rgba {
    let mut hist = [0u64; 256];
    let mut total = 0u64;
    for p in img.as_raw().chunks_exact(4) {
        if p[3] < 128 {
            continue;
        }
        let l = (p[0] as u32 * 54 + p[1] as u32 * 183 + p[2] as u32 * 19) >> 8;
        hist[l as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return img.clone();
    }
    let clip = (total as f64 * 0.003) as u64;
    let (mut lo, mut acc) = (0usize, 0u64);
    while lo < 255 && acc + hist[lo] <= clip {
        acc += hist[lo];
        lo += 1;
    }
    let (mut hi, mut acc2) = (255usize, 0u64);
    while hi > 0 && acc2 + hist[hi] <= clip {
        acc2 += hist[hi];
        hi -= 1;
    }
    if hi <= lo + 10 || (lo < 3 && hi > 252) {
        return img.clone();
    }
    let scale = 255.0 / (hi - lo) as f32;
    let lut: Vec<u8> = (0..256).map(|v| ((v as f32 - lo as f32) * scale).round().clamp(0.0, 255.0) as u8).collect();
    let mut out = img.clone();
    out.as_mut().par_chunks_mut(4).for_each(|p| {
        p[0] = lut[p[0] as usize];
        p[1] = lut[p[1] as usize];
        p[2] = lut[p[2] as usize];
    });
    out
}
