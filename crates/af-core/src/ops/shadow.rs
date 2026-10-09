//! Shadows for cut-outs: a soft "ground" shadow (the object stands on a surface) or a classic
//! drop shadow. Purely classic image processing — fast and predictable.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::mask::gaussian_blur;
use crate::Rgba;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ShadowMode {
    /// Contact shadow plus a soft elliptical shadow on the floor under the object.
    #[default]
    Ground,
    /// The silhouette, offset and blurred.
    Drop,
}

#[derive(Debug, Clone, Copy)]
pub struct ShadowParams {
    pub mode: ShadowMode,
    /// 0..=1
    pub opacity: f32,
    /// 0..=1 (blur, relative to the object size)
    pub softness: f32,
    /// Ground: width of the floor shadow, 0..=1.
    pub size: f32,
    /// Drop: direction in degrees (0 = right, 90 = down).
    pub angle: f32,
    /// Drop: offset 0..=1 (up to a quarter of the object size).
    pub distance: f32,
    pub color: [u8; 3],
}

/// Added transparent margins (top, right, bottom, left) so the shadow is never cut off.
pub type Margins = (u32, u32, u32, u32);

/// Bounding box (x0, y0, x1, y1), exclusive end, of pixels with alpha above `thr`.
fn bbox(a: &[f32], w: usize, h: usize, thr: f32) -> Option<(usize, usize, usize, usize)> {
    let rows: Vec<Option<(usize, usize)>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let r = &a[y * w..(y + 1) * w];
            let f = r.iter().position(|v| *v > thr)?;
            let l = r.iter().rposition(|v| *v > thr)?;
            Some((f, l + 1))
        })
        .collect();
    let y0 = rows.iter().position(|r| r.is_some())?;
    let y1 = rows.iter().rposition(|r| r.is_some())? + 1;
    let (mut x0, mut x1) = (usize::MAX, 0);
    for (a, b) in rows.iter().flatten() {
        x0 = x0.min(*a);
        x1 = x1.max(*b);
    }
    Some((x0, y0, x1, y1))
}

/// Shadow alpha (0..=1) for the cut-out alpha `a` on a canvas of `w` × `h`.
fn shadow_plane(a: &[f32], w: usize, h: usize, p: &ShadowParams, obj: (usize, usize, usize, usize)) -> Vec<f32> {
    let (x0, y0, x1, y1) = obj;
    let (wo, ho) = ((x1 - x0) as f32, (y1 - y0) as f32);
    let s = wo.max(ho);
    let soft = p.softness.clamp(0.0, 1.0);
    let n = w * h;
    let mut out = match p.mode {
        ShadowMode::Drop => {
            let dist = p.distance.clamp(0.0, 1.0) * 0.25 * s;
            let (dx, dy) = ((p.angle.to_radians().cos() * dist).round() as i64, (p.angle.to_radians().sin() * dist).round() as i64);
            let mut shifted = vec![0f32; n];
            shifted.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let sy = y as i64 - dy;
                if sy < 0 || sy >= h as i64 {
                    return;
                }
                let src = &a[sy as usize * w..(sy as usize + 1) * w];
                for (x, v) in row.iter_mut().enumerate() {
                    let sx = x as i64 - dx;
                    if sx >= 0 && sx < w as i64 {
                        *v = src[sx as usize];
                    }
                }
            });
            gaussian_blur(&shifted, w, h, (0.004 + soft * 0.05) * s + 0.5)
        }
        ShadowMode::Ground => {
            // contact: the lowest band of the silhouette, nudged down and blurred a little
            let band = (ho * 0.035).max(2.0) as usize;
            let nudge = (ho * 0.006).max(1.0) as usize;
            let mut contact = vec![0f32; n];
            for y in y1.saturating_sub(band)..y1 {
                let ty = (y + nudge).min(h - 1);
                for x in x0..x1 {
                    let v = a[y * w + x];
                    if v > contact[ty * w + x] {
                        contact[ty * w + x] = v;
                    }
                }
            }
            let contact = gaussian_blur(&contact, w, h, (0.003 + soft * 0.012) * wo + 1.0);
            // footprint: where the bottom band actually touches the floor
            let (mut fx0, mut fx1) = (usize::MAX, 0usize);
            for y in y1.saturating_sub(band)..y1 {
                for x in x0..x1 {
                    if a[y * w + x] > 0.5 {
                        fx0 = fx0.min(x);
                        fx1 = fx1.max(x + 1);
                    }
                }
            }
            if fx0 == usize::MAX {
                (fx0, fx1) = (x0, x1);
            }
            let cx = (fx0 + fx1) as f32 / 2.0;
            let half = ((fx1 - fx0) as f32 / 2.0 * 1.1).max(wo * 0.2);
            let rx = half * (0.6 + p.size.clamp(0.0, 1.0) * 0.8);
            let ry = (rx * 0.13).max(2.0);
            // centred a little above the base: the far half hides behind the object, the near
            // half shows in front of it, so the object sits *in* its shadow
            let yb = y1 as f32 - ry * 0.25;
            let mut ell = vec![0f32; n];
            ell.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let dy = (y as f32 + 0.5 - yb) / ry;
                if dy.abs() > 3.0 {
                    return;
                }
                for (x, v) in row.iter_mut().enumerate() {
                    let dx = (x as f32 + 0.5 - cx) / rx;
                    *v = (-2.0 * (dx * dx + dy * dy)).exp();
                }
            });
            let ell = gaussian_blur(&ell, w, h, (0.002 + soft * 0.025) * wo + 0.5);
            contact.par_iter().zip(ell.par_iter()).map(|(c, e)| 1.0 - (1.0 - c.min(1.0)) * (1.0 - e * 0.85)).collect()
        }
    };
    let k = p.opacity.clamp(0.0, 1.0);
    out.par_iter_mut().for_each(|v| *v = (*v * k).clamp(0.0, 1.0));
    out
}

/// How far the shadow can reach outside the object's box (conservative), as
/// (left, top, right, bottom) extents relative to the canvas edges.
fn reach(p: &ShadowParams, obj: (usize, usize, usize, usize), w: usize, h: usize) -> Margins {
    let (x0, y0, x1, y1) = obj;
    let (wo, ho) = ((x1 - x0) as f32, (y1 - y0) as f32);
    let s = wo.max(ho);
    let soft = p.softness.clamp(0.0, 1.0);
    let (minx, miny, maxx, maxy) = match p.mode {
        ShadowMode::Drop => {
            let dist = p.distance.clamp(0.0, 1.0) * 0.25 * s;
            let (dx, dy) = (p.angle.to_radians().cos() * dist, p.angle.to_radians().sin() * dist);
            let r = 3.0 * ((0.004 + soft * 0.05) * s + 0.5);
            (x0 as f32 + dx - r, y0 as f32 + dy - r, x1 as f32 + dx + r, y1 as f32 + dy + r)
        }
        ShadowMode::Ground => {
            let rx = wo * 0.55 * (0.6 + p.size.clamp(0.0, 1.0) * 0.8) * 1.4;
            let ry = (rx * 0.13).max(2.0);
            let r = 3.0 * ((0.002 + soft * 0.025) * wo + 0.5) + 3.0 * ((0.003 + soft * 0.012) * wo + 1.0);
            let cx = (x0 + x1) as f32 / 2.0;
            (cx - rx * 1.5 - r, y1 as f32 - ry * 3.0 - r, cx + rx * 1.5 + r, y1 as f32 + ry * 3.5 + r + (ho * 0.006).max(1.0))
        }
    };
    let l = (-minx).max(0.0).ceil() as u32;
    let t = (-miny).max(0.0).ceil() as u32;
    let r = (maxx - w as f32).max(0.0).ceil() as u32;
    let b = (maxy - h as f32).max(0.0).ceil() as u32;
    (t, r, b, l)
}

/// Put a shadow under the cut-out. Returns the new image and the transparent margins that were
/// added around the original canvas (so other layers can follow), or `None` when the image has
/// no visible subject.
pub fn add_shadow(img: &Rgba, p: &ShadowParams) -> Option<(Rgba, Margins)> {
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    let a0: Vec<f32> = img.as_raw().par_chunks(4).map(|q| q[3] as f32 / 255.0).collect();
    let obj0 = bbox(&a0, w0, h0, 0.5)?;
    let (t, r, b, l) = reach(p, obj0, w0, h0);
    let padded = super::pad(img, t, r, b, l, [0, 0, 0, 0]);
    let (w, h) = (padded.width() as usize, padded.height() as usize);
    let a: Vec<f32> = padded.as_raw().par_chunks(4).map(|q| q[3] as f32 / 255.0).collect();
    let obj = (obj0.0 + l as usize, obj0.1 + t as usize, obj0.2 + l as usize, obj0.3 + t as usize);
    let sh = shadow_plane(&a, w, h, p, obj);
    let c = p.color;
    let mut out = padded;
    out.as_mut().par_chunks_mut(4).zip(sh.par_iter()).for_each(|(px, s)| {
        let shadow = [c[0], c[1], c[2], (s * 255.0).round() as u8];
        let o = super::over([px[0], px[1], px[2], px[3]], shadow);
        px.copy_from_slice(&o);
    });
    // drop the unused part of the conservative margins, never cutting into the original canvas
    let vis: Vec<f32> = out.as_raw().par_chunks(4).map(|q| if q[3] > 1 { 1.0 } else { 0.0 }).collect();
    let (vx0, vy0, vx1, vy1) = bbox(&vis, w, h, 0.5).unwrap_or((l as usize, t as usize, l as usize + w0, t as usize + h0));
    let cx0 = vx0.min(l as usize);
    let cy0 = vy0.min(t as usize);
    let cx1 = vx1.max(l as usize + w0);
    let cy1 = vy1.max(t as usize + h0);
    let cropped = super::crop(&out, super::Rect { x: cx0 as u32, y: cy0 as u32, w: (cx1 - cx0) as u32, h: (cy1 - cy0) as u32 });
    let margins = (t - cy0 as u32, (cx1 - (l as usize + w0)) as u32, (cy1 - (t as usize + h0)) as u32, l - cx0 as u32);
    Some((cropped, margins))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bottle() -> Rgba {
        // an opaque "bottle" standing at the bottom of a tight canvas (like after Trim)
        Rgba::from_fn(60, 160, |x, y| if (10..50).contains(&x) && y >= 20 { image::Rgba([200, 30, 30, 255]) } else { image::Rgba([0, 0, 0, 0]) })
    }

    fn params(mode: ShadowMode) -> ShadowParams {
        ShadowParams { mode, opacity: 0.6, softness: 0.5, size: 0.5, angle: 45.0, distance: 0.3, color: [0, 0, 0] }
    }

    #[test]
    fn ground_shadow_extends_the_canvas_below_and_keeps_the_subject() {
        let img = bottle();
        let (out, (t, r, b, l)) = add_shadow(&img, &params(ShadowMode::Ground)).unwrap();
        assert!(b > 0, "room below for the floor shadow");
        assert_eq!(out.dimensions(), (60 + l + r, 160 + t + b));
        // the subject itself is untouched
        assert_eq!(out.get_pixel(30 + l, 100 + t).0, [200, 30, 30, 255]);
        // a dark, semi-transparent shadow right under the base, fading out sideways
        let under = out.get_pixel(30 + l, 160 + t + 1).0;
        assert!(under[3] > 60 && under[0] < 30, "{under:?}");
        let far = out.get_pixel(0, 160 + t + 1).0;
        assert!(far[3] < under[3]);
        // nothing above the subject's top
        assert!(out.get_pixel(30 + l, 5 + t).0[3] == 0);
    }

    #[test]
    fn drop_shadow_follows_the_angle() {
        let img = bottle();
        let mut p = params(ShadowMode::Drop);
        p.angle = 0.0; // to the right
        let (out, (t, r, _b, l)) = add_shadow(&img, &p).unwrap();
        assert!(r > 0 && l == 0, "extends only to the right: l={l} r={r}");
        let right = out.get_pixel(50 + l + 8, 90 + t).0[3];
        let left = out.get_pixel(l + 4, 90 + t).0[3];
        assert!(right > left + 40, "right {right} left {left}");
    }

    #[test]
    fn opacity_zero_adds_nothing_visible_and_empty_image_is_skipped() {
        let mut p = params(ShadowMode::Ground);
        p.opacity = 0.0;
        let (out, m) = add_shadow(&bottle(), &p).unwrap();
        assert_eq!(m, (0, 0, 0, 0));
        assert_eq!(out.dimensions(), (60, 160));
        assert!(add_shadow(&Rgba::new(10, 10), &p).is_none());
    }
}
