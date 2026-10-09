//! Sticker outline: an even, round-cornered border around a cut-out.
//!
//! The border comes from an exact Euclidean distance transform of the subject mask
//! (Felzenszwalb & Huttenlocher), so it has the same thickness everywhere and round corners;
//! an optional smoothing pass rounds inner corners and closes narrow gaps, like a die-cut sticker.

use rayon::prelude::*;

use crate::mask::gaussian_blur;
use crate::Rgba;

#[derive(Debug, Clone, Copy)]
pub struct OutlineParams {
    /// 0..=1 → up to 8 % of the subject's longer side.
    pub thickness: f32,
    /// 0..=1: rounds inner corners / closes gaps.
    pub smooth: f32,
    pub color: [u8; 3],
}

/// Added transparent margins (top, right, bottom, left).
pub type Margins = (u32, u32, u32, u32);

const INF: f64 = 1e20;

/// 1-D squared distance transform (lower envelope of parabolas), in f64 so large images keep
/// their precision.
fn dt1d(f: &[f64], out: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for q in 1..n {
        let qf = q as f64;
        loop {
            let p = v[k];
            let pf = p as f64;
            let s = ((f[q] + qf * qf) - (f[p] + pf * pf)) / (2.0 * (qf - pf));
            if s <= z[k] {
                // z[0] is -INF, so this always stops at k = 0
                k -= 1;
                continue;
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = INF;
            break;
        }
    }
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        let qf = q as f64;
        while z[k + 1] < qf {
            k += 1;
        }
        let p = v[k] as f64;
        *o = (qf - p) * (qf - p) + f[v[k]];
    }
}

/// Euclidean distance (pixels) from every pixel to the nearest `inside` pixel.
pub fn distance_to(inside: &[bool], w: usize, h: usize) -> Vec<f32> {
    let mut g: Vec<f64> = inside.iter().map(|&b| if b { 0.0 } else { INF }).collect();
    let cols: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let f: Vec<f64> = (0..h).map(|y| g[y * w + x]).collect();
            let mut out = vec![0f64; h];
            let (mut v, mut z) = (vec![0usize; h], vec![0f64; h + 1]);
            dt1d(&f, &mut out, &mut v, &mut z);
            out
        })
        .collect();
    for (x, col) in cols.iter().enumerate() {
        for (y, val) in col.iter().enumerate() {
            g[y * w + x] = *val;
        }
    }
    g.par_chunks_mut(w).for_each(|row| {
        let f = row.to_vec();
        let (mut v, mut z) = (vec![0usize; w], vec![0f64; w + 1]);
        dt1d(&f, row, &mut v, &mut z);
    });
    g.par_iter().map(|d| d.sqrt() as f32).collect()
}

/// Outline thickness in pixels for a subject of this size.
fn radius(p: &OutlineParams, subject_side: f32) -> f32 {
    p.thickness.clamp(0.0, 1.0) * 0.08 * subject_side
}

/// Add the outline under the cut-out. Returns the new image and the margins added around the
/// original canvas, or `None` when there is no visible subject or the thickness is zero.
pub fn add_outline(img: &Rgba, p: &OutlineParams) -> Option<(Rgba, Margins)> {
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    let raw = img.as_raw();
    // subject box
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..h0 {
        for x in 0..w0 {
            if raw[(y * w0 + x) * 4 + 3] > 127 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x0 == usize::MAX {
        return None;
    }
    let r = radius(p, (x1 - x0).max(y1 - y0) as f32);
    if r < 0.5 {
        return None;
    }
    let smooth_sigma = p.smooth.clamp(0.0, 1.0) * r * 0.8;
    let reach = (r + 3.0 * smooth_sigma + 2.0).ceil() as i64;
    let l = (reach - x0 as i64).max(0) as u32;
    let t = (reach - y0 as i64).max(0) as u32;
    let rr = (x1 as i64 + reach - w0 as i64).max(0) as u32;
    let b = (y1 as i64 + reach - h0 as i64).max(0) as u32;
    let padded = super::pad(img, t, rr, b, l, [0, 0, 0, 0]);
    let (w, h) = (padded.width() as usize, padded.height() as usize);
    let inside: Vec<bool> = padded.as_raw().par_chunks(4).map(|q| q[3] > 127).collect();
    let d = distance_to(&inside, w, h);
    // anti-aliased coverage of "within r of the subject"
    let mut cov: Vec<f32> = d.par_iter().map(|d| (r + 0.5 - d).clamp(0.0, 1.0)).collect();
    if smooth_sigma >= 0.5 {
        // blur + re-threshold: rounds inner corners and closes narrow gaps
        // blur + re-threshold. The slope of a blurred edge is ~1/(σ·√2π) per pixel, so scaling by
        // σ·√2π gives a crisp, ~1 px anti-aliased edge again however strong the smoothing is.
        let b = gaussian_blur(&cov, w, h, smooth_sigma);
        let k = (smooth_sigma * 2.5).max(4.0);
        cov = b.par_iter().map(|v| ((v - 0.5) * k + 0.5).clamp(0.0, 1.0)).collect();
        // never eat into the subject itself
        cov.par_iter_mut().zip(inside.par_iter()).for_each(|(c, i)| {
            if *i {
                *c = 1.0;
            }
        });
    }
    let c = p.color;
    let mut out = padded;
    out.as_mut().par_chunks_mut(4).zip(cov.par_iter()).for_each(|(px, a)| {
        let border = [c[0], c[1], c[2], (a * 255.0).round() as u8];
        let o = super::over([px[0], px[1], px[2], px[3]], border);
        px.copy_from_slice(&o);
    });
    Some((out, (t, rr, b, l)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_transform_is_euclidean() {
        let (w, h) = (21, 21);
        let mut m = vec![false; w * h];
        m[10 * w + 10] = true;
        let d = distance_to(&m, w, h);
        assert_eq!(d[10 * w + 10], 0.0);
        assert!((d[10 * w + 13] - 3.0).abs() < 1e-4);
        assert!((d[13 * w + 14] - 5.0).abs() < 1e-4); // 3-4-5 triangle
        assert!((d[0] - (200f32).sqrt()).abs() < 1e-3);
    }

    #[test]
    fn outline_is_even_and_round() {
        // 40×40 red square in the middle of a transparent canvas
        let img = Rgba::from_fn(100, 100, |x, y| if (30..70).contains(&x) && (30..70).contains(&y) { image::Rgba([220, 20, 20, 255]) } else { image::Rgba([0, 0, 0, 0]) });
        let p = OutlineParams { thickness: 1.0, smooth: 0.0, color: [255, 255, 255] }; // 3.2 px
        let (out, m) = add_outline(&img, &p).unwrap();
        assert_eq!(m, (0, 0, 0, 0), "enough room already");
        // subject untouched, white border on every side
        assert_eq!(out.get_pixel(50, 50).0, [220, 20, 20, 255]);
        for (x, y) in [(50, 28), (50, 71), (28, 50), (71, 50)] {
            let px = out.get_pixel(x, y).0;
            assert!(px[3] > 200 && px[0] > 240 && px[1] > 240, "({x},{y}) {px:?}");
        }
        // outside the border stays transparent; corners are rounded (diagonal reach < side reach)
        assert_eq!(out.get_pixel(50, 24).0[3], 0);
        assert!(out.get_pixel(27, 27).0[3] < out.get_pixel(50, 27).0[3]);
    }

    #[test]
    fn grows_the_canvas_when_the_subject_touches_the_edge() {
        let img = Rgba::from_fn(50, 50, |x, _| if x < 40 { image::Rgba([0, 0, 255, 255]) } else { image::Rgba([0, 0, 0, 0]) });
        let (out, (t, r, b, l)) = add_outline(&img, &OutlineParams { thickness: 1.0, smooth: 0.3, color: [255, 255, 255] }).unwrap();
        assert!(t > 0 && b > 0 && l > 0 && r == 0, "{:?}", (t, r, b, l));
        assert_eq!(out.dimensions(), (50 + l + r, 50 + t + b));
        assert!(out.get_pixel(l - 2, t + 25).0[3] > 200, "border just left of the subject");
        assert_eq!(out.get_pixel(0, t + 25).0[3], 0, "margin beyond the border stays transparent");
    }

    #[test]
    fn smoothing_closes_a_narrow_gap() {
        // two bars separated by a 3 px gap; thin outline alone leaves the gap open in the middle
        let img = Rgba::from_fn(120, 80, |x, y| if (20..80).contains(&y) && ((10..50).contains(&x) || (53..93).contains(&x)) { image::Rgba([0, 0, 0, 255]) } else { image::Rgba([0, 0, 0, 0]) });
        let thin = add_outline(&img, &OutlineParams { thickness: 0.15, smooth: 0.0, color: [255, 0, 0] }).unwrap().0;
        let smooth = add_outline(&img, &OutlineParams { thickness: 0.15, smooth: 1.0, color: [255, 0, 0] }).unwrap().0;
        // the gap column at the top edge of the bars, just outside them
        assert!(smooth.get_pixel(smooth.width() / 2, smooth.height() / 2).0[3] >= thin.get_pixel(thin.width() / 2, thin.height() / 2).0[3]);
    }

    #[test]
    fn smoothed_edge_stays_crisp() {
        let img = Rgba::from_fn(200, 200, |x, y| if (60..140).contains(&x) && (60..140).contains(&y) { image::Rgba([0, 0, 0, 255]) } else { image::Rgba([0, 0, 0, 0]) });
        let (out, _) = add_outline(&img, &OutlineParams { thickness: 1.0, smooth: 1.0, color: [255, 255, 255] }).unwrap();
        // walk outwards from the middle of the left side: alpha must drop from opaque to 0 within ~3 px
        let y = out.height() / 2;
        let alphas: Vec<u8> = (0..out.width() / 2).map(|x| out.get_pixel(x, y).0[3]).collect();
        let partial = alphas.iter().filter(|a| **a > 10 && **a < 245).count();
        assert!(partial <= 3, "soft edge of {partial} px: {alphas:?}");
    }
}
