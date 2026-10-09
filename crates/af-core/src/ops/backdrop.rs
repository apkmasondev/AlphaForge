//! New backgrounds behind a cut-out: solid colour, gradient, an image, or the original
//! background blurred (optionally depth-aware, like a wide-aperture lens).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{resize_f32_plane, resize_rgba, Filter};
use crate::mask::{gaussian_blur, morph};
use crate::{Result, Rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BackdropMode {
    /// Solid colour (the classic "fill background").
    #[default]
    Color,
    Gradient,
    /// The original photo behind the subject, blurred.
    Blur,
    /// A picture chosen by the user.
    Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ImageFit {
    /// Fill the whole canvas, cropping the picture if needed.
    #[default]
    Cover,
    /// Show the whole picture; the rest is filled with the colour.
    Contain,
}

/// Straight-alpha `fg` over `bg` (same size).
pub fn composite(fg: &Rgba, bg: &Rgba) -> Rgba {
    let mut out = bg.clone();
    out.as_mut().par_chunks_mut(4).zip(fg.as_raw().par_chunks(4)).for_each(|(d, s)| {
        let o = super::over([s[0], s[1], s[2], s[3]], [d[0], d[1], d[2], d[3]]);
        d.copy_from_slice(&o);
    });
    out
}

/// Darken a background (0 = unchanged, 1 = strongly darker).
pub fn dim(img: &mut Rgba, amount: f32) {
    let k = 1.0 - amount.clamp(0.0, 1.0) * 0.7;
    if k >= 1.0 {
        return;
    }
    img.as_mut().par_chunks_mut(4).for_each(|p| {
        for c in &mut p[..3] {
            *c = (*c as f32 * k).round() as u8;
        }
    });
}

#[inline]
fn noise(x: u32, y: u32) -> f32 {
    // cheap hash -> [-0.5, 0.5): ordered-ish dithering that hides 8-bit banding
    let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65536.0 - 0.5
}

/// Linear (angle in degrees, 0 = left → right, 90 = top → bottom) or radial (centre → corners)
/// gradient between two RGBA colours, dithered.
pub fn gradient(w: u32, h: u32, c1: [u8; 4], c2: [u8; 4], angle: f32, radial: bool) -> Rgba {
    let mut out = Rgba::new(w, h);
    let (fw, fh) = (w as f32, h as f32);
    let (dx, dy) = (angle.to_radians().cos(), angle.to_radians().sin());
    // projection range over the four corners
    let proj = |x: f32, y: f32| x * dx + y * dy;
    let corners = [proj(0.0, 0.0), proj(fw, 0.0), proj(0.0, fh), proj(fw, fh)];
    let (pmin, pmax) = corners.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let (cx, cy) = (fw / 2.0, fh / 2.0);
    let rmax = (cx * cx + cy * cy).sqrt().max(1.0);
    out.as_mut().par_chunks_mut(w as usize * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w as usize {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let t = if radial { ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() / rmax } else { (proj(px, py) - pmin) / (pmax - pmin).max(1e-6) };
            let t = t.clamp(0.0, 1.0);
            let n = noise(x as u32, y as u32);
            for c in 0..4 {
                let v = c1[c] as f32 + (c2[c] as f32 - c1[c] as f32) * t + if c < 3 { n } else { 0.0 };
                row[x * 4 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

/// Scale a picture to the canvas: `Cover` crops to fill, `Contain` letterboxes on `fill`.
pub fn fit_image(pic: &Rgba, w: u32, h: u32, fit: ImageFit, fill: [u8; 4]) -> Result<Rgba> {
    let (pw, ph) = (pic.width() as f32, pic.height() as f32);
    let s = match fit {
        ImageFit::Cover => (w as f32 / pw).max(h as f32 / ph),
        ImageFit::Contain => (w as f32 / pw).min(h as f32 / ph),
    };
    let (sw, sh) = (((pw * s).round() as u32).max(1), ((ph * s).round() as u32).max(1));
    let scaled = resize_rgba(pic, sw, sh, Filter::Lanczos)?;
    let x = (w as i64 - sw as i64) / 2;
    let y = (h as i64 - sh as i64) / 2;
    // Cover hides the fill completely: a plain copy is much faster than compositing
    let fill = if fit == ImageFit::Cover { [0, 0, 0, 0] } else { fill };
    Ok(super::place_on_canvas(&scaled, w, h, x, y, fill))
}

// ---------------------------------------------------------------------------------------------
// Blurred original background
// ---------------------------------------------------------------------------------------------

/// Working resolution for hole filling and large blurs (they are smooth anyway).
const WORK: u32 = 1024;

fn planes(img: &Rgba) -> [Vec<f32>; 3] {
    let raw = img.as_raw();
    std::array::from_fn(|c| raw.par_chunks(4).map(|p| p[c] as f32 / 255.0).collect())
}

fn small_size(w: u32, h: u32, max: u32) -> (u32, u32) {
    let s = (max as f32 / w.max(h) as f32).min(1.0);
    (((w as f32 * s).round() as u32).max(1), ((h as f32 * s).round() as u32).max(1))
}

/// Weighted pull-push fill: estimates every plane where `valid` is low from the valid pixels
/// around it (computed at a small working size; the estimate is smooth anyway). Returns the
/// estimate at full size for every pixel; `fallback` is used where nothing valid is reachable.
fn pull_push(planes: &[Vec<f32>], valid: &[f32], w: u32, h: u32, fallback: &[f32]) -> Vec<Vec<f32>> {
    let (sw, sh) = small_size(w, h, WORK);
    let m = (sw * sh) as usize;
    let mut acc_w = resize_f32_plane(valid, w, h, sw, sh, Filter::Bilinear);
    let mut acc: Vec<Vec<f32>> = planes
        .par_iter()
        .map(|pl| {
            let pre: Vec<f32> = pl.par_iter().zip(valid.par_iter()).map(|(v, a)| v * a).collect();
            resize_f32_plane(&pre, w, h, sw, sh, Filter::Bilinear)
        })
        .collect();
    let mut fill: Vec<Vec<f32>> = planes.iter().map(|_| vec![0f32; m]).collect();
    let mut done = vec![false; m];
    let mut sigma = 1.5f32;
    while sigma < (sw.max(sh) as f32) * 2.0 {
        let bw = gaussian_blur(&acc_w, sw as usize, sh as usize, sigma);
        let bc: Vec<Vec<f32>> = acc.par_iter().map(|p| gaussian_blur(p, sw as usize, sh as usize, sigma)).collect();
        for i in 0..m {
            if !done[i] && bw[i] > 1e-3 {
                for (c, f) in fill.iter_mut().enumerate() {
                    f[i] = bc[c][i] / bw[i];
                }
                done[i] = true;
            }
        }
        acc_w = bw;
        acc = bc;
        sigma *= 3.0;
    }
    for i in 0..m {
        if !done[i] {
            for (c, f) in fill.iter_mut().enumerate() {
                f[i] = fallback[c];
            }
        }
    }
    fill.par_iter().map(|p| resize_f32_plane(p, sw, sh, w, h, Filter::Bilinear)).collect()
}

/// Mix real values with the estimate, handing over softly so no seam shows.
fn blend_with_ramp(real: &[f32], est: &[f32], valid: &[f32], ramp: &[f32]) -> Vec<f32> {
    real.par_iter().zip(est.par_iter()).zip(valid.par_iter().zip(ramp.par_iter())).map(|((r, e), (v, rp))| {
        let a = v * rp.clamp(0.0, 1.0);
        r * a + e * (1.0 - a)
    }).collect()
}

fn ramp_of(valid: &[f32], w: u32, h: u32) -> Vec<f32> {
    gaussian_blur(valid, w as usize, h as usize, ((w.max(h) as f32) * 0.006).max(2.0))
}

/// Replace the subject (and any invalid area) of `behind` with plausible surrounding colours so
/// that blurring never smears the subject into the background. `subject` is the cut-out (its
/// alpha marks the hole); `behind`'s own alpha marks valid pixels (0 in added padding).
pub fn fill_background(behind: &Rgba, subject: &Rgba) -> Rgba {
    let (w, h) = behind.dimensions();
    // hole = subject grown a little (its edge pixels in the original still carry its colours)
    let a_sub: Vec<f32> = subject.as_raw().par_chunks(4).map(|p| if p[3] > 8 { 1.0 } else { 0.0 }).collect();
    let grow = ((w.max(h) as f32) * 0.004).round().max(2.0) as i32;
    let hole = morph(&a_sub, w as usize, h as usize, grow);
    let valid: Vec<f32> = behind.as_raw().par_chunks(4).zip(hole.par_iter()).map(|(p, hl)| (p[3] as f32 / 255.0) * (1.0 - hl)).collect();
    let ch = planes(behind);
    // anything unreachable (nothing valid at all): mean colour, or white
    let total: f32 = valid.iter().sum();
    let mean: Vec<f32> = (0..3).map(|c| if total > 1.0 { ch[c].iter().zip(&valid).map(|(v, a)| v * a).sum::<f32>() / total } else { 1.0 }).collect();
    let est = pull_push(&ch, &valid, w, h, &mean);
    let ramp = ramp_of(&valid, w, h);
    let mixed: Vec<Vec<f32>> = (0..3).map(|c| blend_with_ramp(&ch[c], &est[c], &valid, &ramp)).collect();
    let mut out = Rgba::new(w, h);
    out.as_mut().par_chunks_mut(4).enumerate().for_each(|(i, p)| {
        for c in 0..3 {
            p[c] = (mixed[c][i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        p[3] = 255;
    });
    out
}

/// Prepare the model depth (already at image size) for blurring: inside added padding the model
/// only saw invented pixels, so depth there is continued from the real pixels around it; then
/// the map is smoothed so the blur changes gradually, like a real lens.
pub fn prepare_depth(depth: &[f32], behind: &Rgba) -> Vec<f32> {
    let (w, h) = behind.dimensions();
    let valid: Vec<f32> = behind.as_raw().par_chunks(4).map(|p| p[3] as f32 / 255.0).collect();
    let d = if valid.iter().all(|v| *v >= 1.0) {
        depth.to_vec()
    } else {
        let est = pull_push(&[depth.to_vec()], &valid, w, h, &[0.0]);
        blend_with_ramp(depth, &est[0], &valid, &ramp_of(&valid, w, h))
    };
    gaussian_blur(&d, w as usize, h as usize, ((w.max(h) as f32) * 0.006).max(1.0))
}

/// Gaussian blur of an opaque image; large radii are computed at a reduced size.
fn blur_rgb(img: &Rgba, sigma: f32) -> [Vec<f32>; 3] {
    let (w, h) = img.dimensions();
    let p = planes(img);
    if sigma < 0.5 {
        return p;
    }
    // keep the blur kernel around 6 px at the working size
    let factor = (sigma / 6.0).max(1.0);
    let (sw, sh) = (((w as f32 / factor).round() as u32).max(1), ((h as f32 / factor).round() as u32).max(1));
    let s = sigma / (w as f32 / sw as f32);
    let out: Vec<Vec<f32>> = p
        .par_iter()
        .map(|pl| {
            let small = if (sw, sh) == (w, h) { pl.clone() } else { resize_f32_plane(pl, w, h, sw, sh, Filter::Bilinear) };
            let b = gaussian_blur(&small, sw as usize, sh as usize, s);
            if (sw, sh) == (w, h) {
                b
            } else {
                resize_f32_plane(&b, sw, sh, w, h, Filter::Bicubic)
            }
        })
        .collect();
    [out[0].clone(), out[1].clone(), out[2].clone()]
}

/// A blurred copy kept at a reduced size (blurred images are smooth, so they are sampled back
/// with bilinear interpolation instead of being stored at full resolution).
struct Small {
    p: [Vec<f32>; 3],
    w: usize,
    h: usize,
    sx: f32,
    sy: f32,
}

impl Small {
    fn new(full: &[Vec<f32>; 3], w: u32, h: u32, sigma: f32) -> Small {
        let factor = (sigma / 6.0).max(1.0);
        let (sw, sh) = (((w as f32 / factor).round() as u32).max(1), ((h as f32 / factor).round() as u32).max(1));
        let s = sigma / (w as f32 / sw as f32);
        let p: Vec<Vec<f32>> = full
            .par_iter()
            .map(|pl| {
                let small = if (sw, sh) == (w, h) { pl.clone() } else { resize_f32_plane(pl, w, h, sw, sh, Filter::Bilinear) };
                gaussian_blur(&small, sw as usize, sh as usize, s)
            })
            .collect();
        Small { p: [p[0].clone(), p[1].clone(), p[2].clone()], w: sw as usize, h: sh as usize, sx: sw as f32 / w as f32, sy: sh as f32 / h as f32 }
    }

    /// Bilinear sample of channel `c` at full-resolution pixel (x, y).
    #[inline]
    fn at(&self, c: usize, x: usize, y: usize) -> f32 {
        let fx = ((x as f32 + 0.5) * self.sx - 0.5).clamp(0.0, (self.w - 1) as f32);
        let fy = ((y as f32 + 0.5) * self.sy - 0.5).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (u, v) = (fx - x0 as f32, fy - y0 as f32);
        let p = &self.p[c];
        let top = p[y0 * self.w + x0] * (1.0 - u) + p[y0 * self.w + x1] * u;
        let bot = p[y1 * self.w + x0] * (1.0 - u) + p[y1 * self.w + x1] * u;
        top * (1.0 - v) + bot * v
    }
}

/// The picture the depth model should see: the original where it exists (subject included, so
/// its distance is known), the filled-in estimate in added padding (no black frame that would
/// read as a depth edge).
pub fn depth_source(behind: &Rgba, filled: &Rgba) -> Rgba {
    let mut out = filled.clone();
    out.as_mut().par_chunks_mut(4).zip(behind.as_raw().par_chunks(4)).for_each(|(o, b)| {
        let a = b[3] as u32;
        for c in 0..3 {
            o[c] = ((b[c] as u32 * a + o[c] as u32 * (255 - a) + 127) / 255) as u8;
        }
        o[3] = 255;
    });
    out
}

/// Blur strength (0..=1) → Gaussian sigma in pixels for an image of this size.
pub fn blur_sigma(w: u32, h: u32, strength: f32) -> f32 {
    strength.clamp(0.0, 1.0) * 0.022 * w.max(h) as f32
}

/// How far (in normalized depth) behind the subject the blur reaches its maximum, for a focus
/// range 0 (only the subject's plane stays sharp) ..= 1 (a deep zone stays sharp). Things in
/// front of the subject blur a little faster, as with a real lens.
pub fn focus_reach(focus: f32) -> (f32, f32) {
    // never thinner than ~0.12: a hair-thin sharp band only shows the noise of the depth map
    let behind = 0.12 + focus.clamp(0.0, 1.0) * 0.46;
    (behind, behind * 0.72)
}

/// Blur a filled background. With `depth` = (depth map 0 = far … 1 = near at image resolution,
/// subject depth, focus range 0..=1) the blur grows with the distance from the subject's depth,
/// like a real lens: the ground at the subject's feet stays sharp, the far background is the most
/// blurred.
pub fn blur_background(filled: &Rgba, strength: f32, depth: Option<(&[f32], f32, f32)>) -> Rgba {
    let (w, h) = filled.dimensions();
    let smax = blur_sigma(w, h, strength);
    let out_planes: [Vec<f32>; 3] = match depth {
        None => blur_rgb(filled, smax),
        Some((d, ds, focus)) => {
            let (reach_back, reach_front) = focus_reach(focus);
            const LEVELS: usize = 5;
            // level 0 is the sharp image itself; the blurred levels live at reduced size
            let sharp = planes(filled);
            let blurred: Vec<Small> = (1..LEVELS).map(|k| Small::new(&sharp, w, h, smax * k as f32 / (LEVELS - 1) as f32)).collect();
            let level = |k: usize, c: usize, i: usize| if k == 0 { sharp[c][i] } else { blurred[k - 1].at(c, i % w as usize, i / w as usize) };
            let n = (w * h) as usize;
            let mut o: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; n]);
            let [o0, o1, o2] = &mut o;
            o0.par_iter_mut().zip(o1.par_iter_mut()).zip(o2.par_iter_mut()).enumerate().for_each(|(i, ((a, b), c))| {
                let dd = ds - d[i];
                let t = if dd >= 0.0 { dd / reach_back } else { -dd / reach_front };
                let f = t.clamp(0.0, 1.0) * (LEVELS - 1) as f32;
                let k = (f.floor() as usize).min(LEVELS - 2);
                let u = f - k as f32;
                *a = level(k, 0, i) * (1.0 - u) + level(k + 1, 0, i) * u;
                *b = level(k, 1, i) * (1.0 - u) + level(k + 1, 1, i) * u;
                *c = level(k, 2, i) * (1.0 - u) + level(k + 1, 2, i) * u;
            });
            o
        }
    };
    let mut out = Rgba::new(w, h);
    out.as_mut().par_chunks_mut(4).enumerate().for_each(|(i, p)| {
        for c in 0..3 {
            p[c] = (out_planes[c][i] * 255.0 + noise(i as u32 % w, i as u32 / w)).round().clamp(0.0, 255.0) as u8;
        }
        p[3] = 255;
    });
    out
}

/// Invented areas (added padding) have no real depth: treat them as far behind the subject so
/// they are fully blurred, with a wide, gradual transition from the real pixels.
pub fn push_invented_back(depth: &mut [f32], behind: &Rgba, subject_depth: f32) {
    let (w, h) = behind.dimensions();
    let valid: Vec<f32> = behind.as_raw().par_chunks(4).map(|p| p[3] as f32 / 255.0).collect();
    if valid.iter().all(|v| *v >= 1.0) {
        return;
    }
    let ramp = gaussian_blur(&valid, w as usize, h as usize, (w.max(h) as f32) * 0.025);
    let far = subject_depth - 1.0;
    depth.par_iter_mut().zip(valid.par_iter().zip(ramp.par_iter())).for_each(|(d, (v, r))| {
        let inv = 1.0 - (v * r.clamp(0.0, 1.0));
        *d = *d * (1.0 - inv) + far * inv;
    });
}

/// Depth of the subject: median depth under the opaque part of the cut-out.
pub fn subject_depth(depth: &[f32], subject: &Rgba) -> f32 {
    let mut v: Vec<f32> = depth.iter().zip(subject.as_raw().chunks_exact(4)).filter(|(_, p)| p[3] > 200).map(|(d, _)| *d).collect();
    if v.is_empty() {
        return 1.0;
    }
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(w: u32, h: u32) -> Rgba {
        Rgba::from_fn(w, h, |x, y| if (x / 8 + y / 8) % 2 == 0 { image::Rgba([20, 120, 220, 255]) } else { image::Rgba([240, 200, 40, 255]) })
    }

    #[test]
    fn gradient_ends_and_banding() {
        let g = gradient(256, 16, [0, 0, 0, 255], [255, 255, 255, 255], 0.0, false);
        assert!(g.get_pixel(0, 8)[0] <= 2 && g.get_pixel(255, 8)[0] >= 253);
        let r = gradient(64, 64, [255, 0, 0, 255], [0, 0, 255, 255], 0.0, true);
        assert!(r.get_pixel(32, 32)[0] > 240 && r.get_pixel(0, 0)[2] > 240);
    }

    #[test]
    fn fit_cover_and_contain() {
        let pic = Rgba::from_pixel(200, 100, image::Rgba([10, 20, 30, 255]));
        let c = fit_image(&pic, 100, 100, ImageFit::Cover, [0, 0, 0, 0]).unwrap();
        assert_eq!(c.dimensions(), (100, 100));
        assert_eq!(c.get_pixel(0, 0)[3], 255);
        let t = fit_image(&pic, 100, 100, ImageFit::Contain, [255, 255, 255, 255]).unwrap();
        assert_eq!(t.get_pixel(50, 5).0, [255, 255, 255, 255]);
        assert_eq!(t.get_pixel(50, 50).0, [10, 20, 30, 255]);
    }

    #[test]
    fn fill_removes_the_subject_colour() {
        // grey background, a pure red subject in the middle
        let (w, h) = (200, 160);
        let behind = Rgba::from_fn(w, h, |x, y| if (60..140).contains(&x) && (40..120).contains(&y) { image::Rgba([255, 0, 0, 255]) } else { image::Rgba([90, 90, 90, 255]) });
        let subject = Rgba::from_fn(w, h, |x, y| if (60..140).contains(&x) && (40..120).contains(&y) { image::Rgba([255, 0, 0, 255]) } else { image::Rgba([0, 0, 0, 0]) });
        let f = fill_background(&behind, &subject);
        for p in f.pixels() {
            assert!((p[0] as i32 - 90).abs() <= 3 && (p[1] as i32 - 90).abs() <= 3, "{p:?}");
        }
        // blurring the filled background never brings red back
        let b = blur_background(&f, 1.0, None);
        assert!(b.pixels().all(|p| (p[0] as i32 - p[1] as i32).abs() <= 3));
    }

    #[test]
    fn padding_area_is_filled_too() {
        let mut behind = checker(120, 80);
        for (x, _, p) in behind.enumerate_pixels_mut() {
            if x >= 100 {
                *p = image::Rgba([0, 0, 0, 0]); // added padding
            }
        }
        let subject = Rgba::new(120, 80);
        let f = fill_background(&behind, &subject);
        assert!(f.pixels().all(|p| p[3] == 255));
        // the padding is not black: it continues the colours next to it
        assert!(f.get_pixel(115, 40).0[..3].iter().map(|v| *v as u32).sum::<u32>() > 150);
    }

    #[test]
    fn depth_blur_keeps_the_subject_plane_sharp() {
        // fine 2 px checker: a strong blur averages it out completely
        let img = Rgba::from_fn(160, 120, |x, y| if (x / 2 + y / 2) % 2 == 0 { image::Rgba([20, 120, 220, 255]) } else { image::Rgba([240, 200, 40, 255]) });
        let near = vec![1.0f32; 160 * 120];
        let far = vec![0.0f32; 160 * 120];
        let sharp = blur_background(&img, 1.0, Some((&near, 1.0, 0.5)));
        let soft = blur_background(&img, 1.0, Some((&far, 1.0, 0.5)));
        let contrast = |i: &Rgba| i.pixels().map(|p| p[2] as f32).fold((255f32, 0f32), |(a, b), v| (a.min(v), b.max(v)));
        let (a, b) = contrast(&sharp);
        let (c, d) = contrast(&soft);
        assert!(b - a > 150.0, "same depth as the subject stays sharp");
        assert!(d - c < 60.0, "far background gets blurred");
    }

    #[test]
    fn focus_range_controls_the_sharp_zone() {
        // a pixel a little behind the subject: sharp with a wide focus range, blurred with a narrow one
        let img = Rgba::from_fn(160, 120, |x, y| if (x / 2 + y / 2) % 2 == 0 { image::Rgba([20, 120, 220, 255]) } else { image::Rgba([240, 200, 40, 255]) });
        let d = vec![0.85f32; 160 * 120];
        let contrast = |i: &Rgba| i.pixels().map(|p| p[2] as f32).fold((255f32, 0f32), |(a, b), v| (a.min(v), b.max(v)));
        let (a, b) = contrast(&blur_background(&img, 1.0, Some((&d, 1.0, 1.0))));
        let (c, e) = contrast(&blur_background(&img, 1.0, Some((&d, 1.0, 0.0))));
        assert!(b - a > (e - c) + 40.0, "wide focus keeps more contrast: {} vs {}", b - a, e - c);
        assert_eq!(focus_reach(0.5).0, 0.35);
    }
}
