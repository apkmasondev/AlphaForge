//! Alpha-matte post-processing: refinement controls, brush edits and color decontamination.
//!
//! All planes are row-major `f32` in `0..=1`.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ops::{resize_f32_plane, Filter};
use crate::Rgba;

/// User-adjustable matte refinement. Defaults are tuned to be safe for most photos.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Refine {
    /// Grow (+) or shrink (−) the matte, in output pixels.
    pub edge_shift: f32,
    /// Soften the edge with a blur of this radius (output pixels).
    pub feather: f32,
    /// 0 = keep the soft alpha from the model, 1 = hard cut-out.
    pub hardness: f32,
    /// Remove small disconnected blobs (keeps the main subject and other large parts).
    pub remove_islands: bool,
    /// Re-estimate edge colors so background color does not bleed into hair/fur.
    pub decontaminate: bool,
    /// Snap the up-scaled matte to image edges (guided filter) for large photos.
    pub edge_snap: bool,
}

impl Default for Refine {
    fn default() -> Self {
        Self { edge_shift: 0.0, feather: 0.0, hardness: 0.0, remove_islands: true, decontaminate: true, edge_snap: true }
    }
}

/// Brush stroke in image pixel coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Stroke {
    pub mode: StrokeMode,
    pub radius: f32,
    /// 0 = very soft brush, 1 = hard brush.
    pub hardness: f32,
    pub points: Vec<[f32; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StrokeMode {
    /// Paint subject back in (alpha → 1).
    Keep,
    /// Paint background away (alpha → 0).
    Erase,
    /// Undo manual edits under the brush (back to the AI result).
    Restore,
}

// ---------------------------------------------------------------------------------------------
// Basic filters
// ---------------------------------------------------------------------------------------------

/// Horizontal running-sum box blur of radius `r` (window 2r+1), edge-clamped.
fn box_rows(src: &[f32], w: usize, h: usize, r: usize, dst: &mut [f32]) {
    if r == 0 {
        dst.copy_from_slice(src);
        return;
    }
    let inv = 1.0 / (2 * r + 1) as f32;
    dst.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(out, row)| {
        let at = |i: isize| row[i.clamp(0, w as isize - 1) as usize];
        let mut acc: f32 = (-(r as isize)..=r as isize).map(at).sum();
        for x in 0..w {
            out[x] = acc * inv;
            acc += at(x as isize + r as isize + 1) - at(x as isize - r as isize);
        }
    });
    let _ = h;
}

fn transpose(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    const B: usize = 32;
    out.par_chunks_mut(h * B).enumerate().for_each(|(bx, chunk)| {
        let x0 = bx * B;
        let x1 = (x0 + B).min(w);
        for y in 0..h {
            for x in x0..x1 {
                chunk[(x - x0) * h + y] = src[y * w + x];
            }
        }
    });
    out
}

/// Box blur (window 2r+1) in both directions.
pub fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0f32; w * h];
    box_rows(src, w, h, r, &mut tmp);
    let t = transpose(&tmp, w, h);
    let mut t2 = vec![0f32; w * h];
    box_rows(&t, h, w, r, &mut t2);
    transpose(&t2, h, w)
}

/// Gaussian approximation with three box passes.
pub fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.3 {
        return src.to_vec();
    }
    // Box radius for 3 passes (Kovesi): w_ideal = sqrt(12 s^2 / n + 1)
    let wi = ((12.0 * sigma * sigma / 3.0) + 1.0).sqrt();
    let r = (((wi - 1.0) / 2.0).round() as usize).max(1);
    let a = box_blur(src, w, h, r);
    let b = box_blur(&a, w, h, r);
    box_blur(&b, w, h, r)
}

/// Van Herk / Gil-Werman 1-D running min or max over window 2r+1.
fn minmax_rows(src: &[f32], w: usize, r: usize, take_max: bool, dst: &mut [f32]) {
    let k = 2 * r + 1;
    dst.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(out, row)| {
        let pad_val = if take_max { f32::NEG_INFINITY } else { f32::INFINITY };
        let n = w + 2 * r;
        let padded: Vec<f32> = (0..n).map(|i| if i < r || i >= r + w { pad_val } else { row[i - r] }).collect();
        let f = |a: f32, b: f32| if take_max { a.max(b) } else { a.min(b) };
        let mut g = vec![0f32; n];
        let mut hh = vec![0f32; n];
        for i in 0..n {
            g[i] = if i % k == 0 { padded[i] } else { f(g[i - 1], padded[i]) };
        }
        for i in (0..n).rev() {
            hh[i] = if i == n - 1 || (i + 1) % k == 0 { padded[i] } else { f(hh[i + 1], padded[i]) };
        }
        for x in 0..w {
            // window [x, x + 2r] in padded coordinates
            out[x] = f(hh[x], g[x + 2 * r]);
        }
    });
}

/// Grayscale dilation (`grow > 0`) or erosion (`grow < 0`) with a square element.
pub fn morph(src: &[f32], w: usize, h: usize, grow: i32) -> Vec<f32> {
    if grow == 0 {
        return src.to_vec();
    }
    let r = grow.unsigned_abs() as usize;
    let take_max = grow > 0;
    let mut tmp = vec![0f32; w * h];
    minmax_rows(src, w, r, take_max, &mut tmp);
    let t = transpose(&tmp, w, h);
    let mut t2 = vec![0f32; w * h];
    minmax_rows(&t, h, r, take_max, &mut t2);
    transpose(&t2, h, w)
}

// ---------------------------------------------------------------------------------------------
// Matte refinement
// ---------------------------------------------------------------------------------------------

/// Remove connected blobs that are tiny compared to the image and to the main subject.
/// Works on the (small) model-resolution matte.
pub fn remove_islands(a: &mut [f32], w: usize, h: usize) {
    let n = w * h;
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let on = |v: f32| v > 0.06;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !on(a[i]) {
                continue;
            }
            let mut unite = |j: usize| {
                if on(a[j]) {
                    let (ra, rb) = (find(&mut parent, i as u32), find(&mut parent, j as u32));
                    if ra != rb {
                        parent[ra.max(rb) as usize] = ra.min(rb);
                    }
                }
            };
            if x > 0 {
                unite(i - 1);
            }
            if y > 0 {
                unite(i - w);
                if x > 0 {
                    unite(i - w - 1);
                }
                if x + 1 < w {
                    unite(i - w + 1);
                }
            }
        }
    }
    // Mass-weighted area per component (soft pixels count partially).
    let mut mass = std::collections::HashMap::<u32, f32>::new();
    for i in 0..n {
        if on(a[i]) {
            let r = find(&mut parent, i as u32);
            *mass.entry(r).or_default() += a[i];
        }
    }
    let largest = mass.values().cloned().fold(0.0, f32::max);
    if largest <= 0.0 {
        return;
    }
    let min_keep = (0.10 * largest).min(0.004 * n as f32).max(0.0005 * n as f32);
    let mut kept = vec![0f32; n];
    for i in 0..n {
        if on(a[i]) {
            let r = find(&mut parent, i as u32);
            if mass[&r] < min_keep {
                a[i] = 0.0;
            } else {
                kept[i] = 1.0;
            }
        }
    }
    // Faint haze is only cleared when it is far from everything we kept, so wispy hair
    // tips next to the subject survive.
    let near = morph(&kept, w, h, ((w.max(h) as f32) * 0.02).ceil() as i32);
    for i in 0..n {
        if !on(a[i]) && a[i] > 0.0 && near[i] < 0.5 {
            a[i] = 0.0;
        }
    }
}

/// Fast guided filter (He et al.) with a grayscale guide; returns the refined matte.
/// `r` and `eps` are in full-resolution pixels / intensity^2. Computed at 1/s resolution.
pub fn guided_filter(guide: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32, s: usize) -> Vec<f32> {
    let s = s.max(1);
    let (sw, sh) = ((w / s).max(1), (h / s).max(1));
    let gi = resize_f32_plane(guide, w as u32, h as u32, sw as u32, sh as u32, Filter::Bilinear);
    let pi = resize_f32_plane(p, w as u32, h as u32, sw as u32, sh as u32, Filter::Bilinear);
    let rs = (r / s).max(1);
    let mean_i = box_blur(&gi, sw, sh, rs);
    let mean_p = box_blur(&pi, sw, sh, rs);
    let ii: Vec<f32> = gi.iter().map(|v| v * v).collect();
    let ip: Vec<f32> = gi.iter().zip(&pi).map(|(a, b)| a * b).collect();
    let corr_i = box_blur(&ii, sw, sh, rs);
    let corr_ip = box_blur(&ip, sw, sh, rs);
    let mut a = vec![0f32; sw * sh];
    let mut b = vec![0f32; sw * sh];
    for k in 0..sw * sh {
        let var = corr_i[k] - mean_i[k] * mean_i[k];
        let cov = corr_ip[k] - mean_i[k] * mean_p[k];
        a[k] = cov / (var + eps);
        b[k] = mean_p[k] - a[k] * mean_i[k];
    }
    let ma = box_blur(&a, sw, sh, rs);
    let mb = box_blur(&b, sw, sh, rs);
    let fa = resize_f32_plane_unclamped(&ma, sw, sh, w, h);
    let fb = resize_f32_plane_unclamped(&mb, sw, sh, w, h);
    (0..w * h).into_par_iter().map(|k| (fa[k] * guide[k] + fb[k]).clamp(0.0, 1.0)).collect()
}

fn resize_f32_plane_unclamped(src: &[f32], sw: usize, sh: usize, w: usize, h: usize) -> Vec<f32> {
    use fast_image_resize as fr;
    if (sw, sh) == (w, h) {
        return src.to_vec();
    }
    let s = fr::images::ImageRef::new(sw as u32, sh as u32, bytemuck::cast_slice(src), fr::PixelType::F32).expect("plane");
    let mut d = fr::images::Image::new(w as u32, h as u32, fr::PixelType::F32);
    let opts = fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    fr::Resizer::new().resize(&s, &mut d, &opts).expect("resize");
    bytemuck::pod_collect_to_vec(&d.into_vec())
}

/// Luma plane (0..1) of an RGBA image.
pub fn luma(img: &Rgba) -> Vec<f32> {
    img.as_raw().par_chunks(4).map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0).collect()
}

/// Up-sample a model-resolution matte to the image size and apply all refinements
/// except color decontamination. Returns the final alpha plane at image resolution.
pub fn build_alpha(model_alpha: &[f32], mw: u32, mh: u32, img: &Rgba, refine: &Refine, strokes: &[Stroke]) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut small = model_alpha.to_vec();
    if refine.remove_islands {
        remove_islands(&mut small, mw as usize, mh as usize);
    }
    let mut a = resize_f32_plane(&small, mw, mh, w as u32, h as u32, Filter::Bicubic);

    // Guided snapping only helps when the image is noticeably larger than the model input.
    let scale = (w.max(h) as f32) / (mw.max(mh) as f32);
    if refine.edge_snap && scale > 1.4 {
        let guide = luma(img);
        let r = (scale * 2.0).round().clamp(2.0, 24.0) as usize;
        let s = ((r / 4).max(1)).min(4);
        let g = guided_filter(&guide, &a, w, h, r, 1e-4, s);
        // Only let the guided result act in the uncertain edge band, keep confident areas intact.
        a.par_iter_mut().zip(g.par_iter()).for_each(|(v, gv)| {
            let band = 1.0 - (2.0 * *v - 1.0).abs(); // 0 at 0/1, 1 at 0.5
            let t = (band * 1.6).min(1.0);
            *v = *v * (1.0 - t) + gv * t;
        });
    }

    if refine.hardness > 0.0 {
        // Levels around 0.5: hardness 1 => nearly binary.
        let half = 0.5 * (1.0 - refine.hardness.clamp(0.0, 1.0) * 0.96);
        let lo = 0.5 - half;
        let hi = 0.5 + half;
        a.par_iter_mut().for_each(|v| *v = ((*v - lo) / (hi - lo)).clamp(0.0, 1.0));
    }
    let shift = refine.edge_shift.round() as i32;
    if shift != 0 {
        a = morph(&a, w, h, shift.clamp(-50, 50));
    }
    if refine.feather > 0.05 {
        a = gaussian_blur(&a, w, h, refine.feather.min(50.0) * 0.5);
    }
    if !strokes.is_empty() {
        apply_strokes(&mut a, w, h, strokes);
    }
    a
}

/// Rasterize brush strokes into an override layer and blend it over the matte.
pub fn apply_strokes(a: &mut [f32], w: usize, h: usize, strokes: &[Stroke]) {
    let mut val = vec![0f32; w * h];
    let mut wt = vec![0f32; w * h];
    for s in strokes {
        let r = s.radius.max(0.5);
        let hard = s.hardness.clamp(0.0, 1.0);
        let inner = r * (0.15 + 0.85 * hard);
        let target = match s.mode {
            StrokeMode::Keep => 1.0,
            StrokeMode::Erase => 0.0,
            StrokeMode::Restore => -1.0,
        };
        let pts: Vec<[f32; 2]> = if s.points.len() == 1 { vec![s.points[0], s.points[0]] } else { s.points.clone() };
        // coverage of this stroke (max over its segments) so overlapping segments don't accumulate
        let (mut bx0, mut by0, mut bx1, mut by1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &pts {
            bx0 = bx0.min(p[0] - r);
            by0 = by0.min(p[1] - r);
            bx1 = bx1.max(p[0] + r);
            by1 = by1.max(p[1] + r);
        }
        let x0 = bx0.floor().max(0.0) as usize;
        let y0 = by0.floor().max(0.0) as usize;
        let x1 = (bx1.ceil() as isize).min(w as isize - 1);
        let y1 = (by1.ceil() as isize).min(h as isize - 1);
        if x1 < 0 || y1 < 0 || x0 as isize > x1 || y0 as isize > y1 {
            continue;
        }
        let (x1, y1) = (x1 as usize, y1 as usize);
        let bw = x1 - x0 + 1;
        let mut cov = vec![0f32; bw * (y1 - y0 + 1)];
        for seg in pts.windows(2) {
            let (ax, ay, bxp, byp) = (seg[0][0], seg[0][1], seg[1][0], seg[1][1]);
            let (dx, dy) = (bxp - ax, byp - ay);
            let len2 = dx * dx + dy * dy;
            let sx0 = (ax.min(bxp) - r).floor().max(x0 as f32) as usize;
            let sx1 = ((ax.max(bxp) + r).ceil() as usize).min(x1);
            let sy0 = (ay.min(byp) - r).floor().max(y0 as f32) as usize;
            let sy1 = ((ay.max(byp) + r).ceil() as usize).min(y1);
            for y in sy0..=sy1 {
                for x in sx0..=sx1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let t = if len2 > 0.0 { (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
                    let (qx, qy) = (ax + t * dx - px, ay + t * dy - py);
                    let d = (qx * qx + qy * qy).sqrt();
                    let c = if d <= inner { 1.0 } else if d >= r { 0.0 } else { let u = 1.0 - (d - inner) / (r - inner); u * u * (3.0 - 2.0 * u) };
                    let k = (y - y0) * bw + (x - x0);
                    if c > cov[k] {
                        cov[k] = c;
                    }
                }
            }
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                let c = cov[(y - y0) * bw + (x - x0)];
                if c <= 0.0 {
                    continue;
                }
                let i = y * w + x;
                if target < 0.0 {
                    wt[i] *= 1.0 - c;
                } else {
                    let nw = c + wt[i] * (1.0 - c);
                    val[i] = (target * c + val[i] * wt[i] * (1.0 - c)) / nw.max(1e-6);
                    wt[i] = nw;
                }
            }
        }
    }
    a.par_iter_mut().zip(val.par_iter().zip(wt.par_iter())).for_each(|(a, (v, t))| {
        if *t > 0.0 {
            *a = *a * (1.0 - t) + v * t;
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Foreground color estimation (decontamination)
// ---------------------------------------------------------------------------------------------

/// Combine the image with an alpha plane into the final RGBA cut-out.
///
/// With `decontaminate`, edge colors are re-estimated with the two-pass blur-fusion
/// foreground estimator (Forte & Pitié, 2021, as used by BiRefNet's `refine_foreground`):
/// semi-transparent hair/fur no longer carries the old background color.
pub fn compose(img: &Rgba, alpha: &[f32], decontaminate: bool) -> Rgba {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = img.clone();
    if !decontaminate {
        out.as_mut().par_chunks_mut(4).zip(alpha.par_iter()).for_each(|(p, a)| {
            p[3] = (a * 255.0).round() as u8;
            if p[3] == 0 {
                p[0] = 0;
                p[1] = 0;
                p[2] = 0;
            }
        });
        return out;
    }
    let (bf, bb, ww, wh) = blur_fusion_fields(img, alpha);
    // Near-opaque pixels right at the boundary often still carry background color (the matte
    // says "foreground" but the pixel is a mix). Inside a thin band next to transparent areas
    // they are estimated as if slightly transparent, which removes that fringe.
    let band_r = ((w.max(h) as f32) / 900.0).round().clamp(1.0, 6.0) as i32;
    let outside: Vec<f32> = alpha.par_iter().map(|a| if *a < 0.5 { 1.0 } else { 0.0 }).collect();
    let near = morph(&outside, w, h, band_r);
    let est_alpha: Vec<f32> = alpha.par_iter().zip(near.par_iter()).map(|(a, n)| if *n > 0.5 && *a >= 0.5 { a.min(0.86) } else { *a }).collect();
    // Final per-channel evaluation at full resolution (one channel at a time to bound memory).
    for c in 0..3 {
        let f_up = resize_f32_plane_unclamped(&bf[c], ww, wh, w, h);
        let b_up = resize_f32_plane_unclamped(&bb[c], ww, wh, w, h);
        out.as_mut().par_chunks_mut(4).enumerate().for_each(|(i, p)| {
            let a = est_alpha[i];
            if alpha[i] <= 0.0 || a >= 0.999 {
                return;
            }
            let iv = img.as_raw()[i * 4 + c] as f32 / 255.0;
            let f = f_up[i] + a * (iv - a * f_up[i] - (1.0 - a) * b_up[i]);
            p[c] = (f.clamp(0.0, 1.0) * 255.0).round() as u8;
        });
    }
    out.as_mut().par_chunks_mut(4).zip(alpha.par_iter()).for_each(|(p, a)| {
        p[3] = (a * 255.0).round() as u8;
        if p[3] == 0 {
            p[0] = 0;
            p[1] = 0;
            p[2] = 0;
        }
    });
    out
}

/// Two blur-fusion passes at a reduced working resolution. Returns the smooth foreground and
/// background fields (bF, bB) per channel plus the working size.
fn blur_fusion_fields(img: &Rgba, alpha: &[f32]) -> ([Vec<f32>; 3], [Vec<f32>; 3], usize, usize) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let max_dim = w.max(h);
    let work = 1536usize.min(max_dim);
    let s = work as f32 / max_dim as f32;
    let (ww, wh) = (((w as f32 * s).round() as usize).max(1), ((h as f32 * s).round() as usize).max(1));
    let a = resize_f32_plane_unclamped(alpha, w, h, ww, wh);
    let a: Vec<f32> = a.into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let chan: Vec<Vec<f32>> = (0..3)
        .into_par_iter()
        .map(|c| {
            let full: Vec<f32> = img.as_raw().chunks_exact(4).map(|p| p[c] as f32 / 255.0).collect();
            resize_f32_plane_unclamped(&full, w, h, ww, wh)
        })
        .collect();
    // Radii relative to the working image (BiRefNet uses windows 90 and 6 at ~1-2K px).
    let r1 = ((90.0 * work as f32 / 1536.0) / 2.0).round().max(4.0) as usize;
    let r2 = ((6.0 * work as f32 / 1536.0) / 2.0).round().max(1.0) as usize;
    let eps = 1e-5f32;
    let ba1 = box_blur(&a, ww, wh, r1);
    let ba2 = box_blur(&a, ww, wh, r2);
    let results: Vec<(Vec<f32>, Vec<f32>)> = (0..3)
        .into_par_iter()
        .map(|c| {
            let i = &chan[c];
            // pass 1: F = B = I
            let fa: Vec<f32> = i.iter().zip(&a).map(|(v, a)| v * a).collect();
            let b1a: Vec<f32> = i.iter().zip(&a).map(|(v, a)| v * (1.0 - a)).collect();
            let bfa = box_blur(&fa, ww, wh, r1);
            let bb1a = box_blur(&b1a, ww, wh, r1);
            let bf1: Vec<f32> = bfa.iter().zip(&ba1).map(|(x, ba)| x / (ba + eps)).collect();
            let bb1: Vec<f32> = bb1a.iter().zip(&ba1).map(|(x, ba)| x / ((1.0 - ba) + eps)).collect();
            let f1: Vec<f32> = (0..ww * wh)
                .map(|k| (bf1[k] + a[k] * (i[k] - a[k] * bf1[k] - (1.0 - a[k]) * bb1[k])).clamp(0.0, 1.0))
                .collect();
            // pass 2: F = F1, B = bB1
            let fa2: Vec<f32> = f1.iter().zip(&a).map(|(v, a)| v * a).collect();
            let b2a: Vec<f32> = bb1.iter().zip(&a).map(|(v, a)| v * (1.0 - a)).collect();
            let bfa2 = box_blur(&fa2, ww, wh, r2);
            let bb2a = box_blur(&b2a, ww, wh, r2);
            let bf2: Vec<f32> = bfa2.iter().zip(&ba2).map(|(x, ba)| x / (ba + eps)).collect();
            let bb2: Vec<f32> = bb2a.iter().zip(&ba2).map(|(x, ba)| x / ((1.0 - ba) + eps)).collect();
            (bf2, bb2)
        })
        .collect();
    let mut it = results.into_iter();
    let (f0, b0) = it.next().unwrap();
    let (f1, b1) = it.next().unwrap();
    let (f2, b2) = it.next().unwrap();
    ([f0, f1, f2], [b0, b1, b2], ww, wh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_blur_preserves_constant() {
        let v = vec![0.5f32; 37 * 23];
        let b = box_blur(&v, 37, 23, 5);
        assert!(b.iter().all(|x| (x - 0.5).abs() < 1e-5));
    }

    #[test]
    fn morph_grows_and_shrinks() {
        let (w, h) = (20, 20);
        let mut v = vec![0f32; w * h];
        v[10 * w + 10] = 1.0;
        let g = morph(&v, w, h, 2);
        assert_eq!(g.iter().filter(|x| **x > 0.5).count(), 25);
        let s = morph(&g, w, h, -2);
        assert_eq!(s.iter().filter(|x| **x > 0.5).count(), 1);
    }

    #[test]
    fn islands_removed() {
        let (w, h) = (100, 100);
        let mut v = vec![0f32; w * h];
        for y in 20..80 {
            for x in 20..80 {
                v[y * w + x] = 1.0;
            }
        }
        v[5 * w + 5] = 1.0; // speck
        remove_islands(&mut v, w, h);
        assert_eq!(v[5 * w + 5], 0.0);
        assert_eq!(v[50 * w + 50], 1.0);
    }

    #[test]
    fn strokes_keep_and_restore() {
        let (w, h) = (50, 50);
        let mut a = vec![0f32; w * h];
        let s = Stroke { mode: StrokeMode::Keep, radius: 5.0, hardness: 1.0, points: vec![[25.0, 25.0]] };
        apply_strokes(&mut a, w, h, &[s.clone()]);
        assert!(a[25 * w + 25] > 0.99);
        let mut b = vec![0f32; w * h];
        let r = Stroke { mode: StrokeMode::Restore, ..s.clone() };
        apply_strokes(&mut b, w, h, &[s, r]);
        assert!(b[25 * w + 25] < 0.01);
    }
}
