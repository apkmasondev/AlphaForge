//! Palette reduction for lossy PNG (pngquant-style) using NeuQuant + Floyd–Steinberg dithering.

use color_quant::NeuQuant;

use crate::Rgba;

/// Returns (RGB palette, palette indices, tRNS alpha table if any palette entry is not opaque).
pub fn quantize(img: &Rgba, colors: usize) -> (Vec<u8>, Vec<u8>, Option<Vec<u8>>) {
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    // Sample factor: 1 = best quality; use coarser sampling on large images for speed.
    let px = (w as usize) * (h as usize);
    let sample = if px > 8_000_000 { 10 } else if px > 2_000_000 { 5 } else { 2 };
    let nq = NeuQuant::new(sample, colors, raw);
    let map = nq.color_map_rgba();
    let n = map.len() / 4;

    // Alpha is never dithered: fully transparent pixels get the most transparent entry, fully
    // opaque pixels only opaque entries (otherwise the subject gets see-through speckles).
    let transparent_idx = (0..n).min_by_key(|&i| map[i * 4 + 3]).unwrap_or(0);
    let opaque_entries: Vec<usize> = (0..n).filter(|&i| map[i * 4 + 3] == 255).collect();
    let mut opaque_cache = std::collections::HashMap::<[u8; 3], u8>::new();
    let mut nearest_opaque = |c: [u8; 3]| -> u8 {
        *opaque_cache.entry(c).or_insert_with(|| {
            let d = |i: usize| (0..3).map(|k| (map[i * 4 + k] as i32 - c[k] as i32).pow(2)).sum::<i32>();
            opaque_entries.iter().copied().min_by_key(|&i| d(i)).unwrap_or(0) as u8
        })
    };

    let mut indices = vec![0u8; px];
    // Error diffusion of RGB in f32, serpentine scan.
    let mut err_cur = vec![[0f32; 3]; w as usize + 2];
    let mut err_next = vec![[0f32; 3]; w as usize + 2];
    for y in 0..h as usize {
        let ltr = y % 2 == 0;
        for i in 0..w as usize {
            let x = if ltr { i } else { w as usize - 1 - i };
            let o = (y * w as usize + x) * 4;
            let alpha = raw[o + 3];
            if alpha == 0 {
                indices[y * w as usize + x] = transparent_idx as u8;
                continue;
            }
            let mut v = [0f32; 3];
            for c in 0..3 {
                v[c] = (raw[o + c] as f32 + err_cur[x + 1][c] * 0.85).clamp(0.0, 255.0);
            }
            let q = [v[0] as u8, v[1] as u8, v[2] as u8, alpha];
            let mut idx = nq.index_of(&q);
            if alpha == 255 && map[idx * 4 + 3] != 255 && !opaque_entries.is_empty() {
                idx = nearest_opaque([q[0], q[1], q[2]]) as usize;
            }
            indices[y * w as usize + x] = idx as u8;
            let p = &map[idx * 4..idx * 4 + 4];
            let e = [v[0] - p[0] as f32, v[1] - p[1] as f32, v[2] - p[2] as f32];
            let (fwd, back) = if ltr { (x + 2, x) } else { (x, x + 2) };
            for c in 0..3 {
                err_cur[fwd][c] += e[c] * 7.0 / 16.0;
                err_next[back][c] += e[c] * 3.0 / 16.0;
                err_next[x + 1][c] += e[c] * 5.0 / 16.0;
                err_next[fwd][c] += e[c] * 1.0 / 16.0;
            }
        }
        std::mem::swap(&mut err_cur, &mut err_next);
        err_next.iter_mut().for_each(|e| *e = [0.0; 3]);
    }
    let mut palette = Vec::with_capacity(n * 3);
    let mut trns = Vec::with_capacity(n);
    for i in 0..n {
        palette.extend_from_slice(&map[i * 4..i * 4 + 3]);
        trns.push(map[i * 4 + 3]);
    }
    let any_alpha = trns.iter().any(|&a| a < 255);
    (palette, indices, if any_alpha { Some(trns) } else { None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_images_and_alpha_classes() {
        for (w, h) in [(1, 1), (2, 1), (3, 3)] {
            let img = Rgba::from_pixel(w, h, image::Rgba([200, 10, 10, 255]));
            let (pal, idx, _) = quantize(&img, 16);
            assert_eq!(idx.len(), (w * h) as usize);
            assert!(!pal.is_empty());
        }
        // half transparent, half a gradient: opaque pixels must stay opaque, empty ones empty
        let mut img = Rgba::new(64, 64);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = if x < 32 { image::Rgba([0, 0, 0, 0]) } else { image::Rgba([(x * 4) as u8, (y * 4) as u8, 128, 255]) };
        }
        let (_, idx, trns) = quantize(&img, 8);
        let trns = trns.expect("palette has a transparent entry");
        for (i, p) in img.pixels().enumerate() {
            let a = trns.get(idx[i] as usize).copied().unwrap_or(255);
            if p[3] == 255 {
                assert_eq!(a, 255, "opaque pixel became translucent");
            } else {
                assert_eq!(a, 0, "transparent pixel became visible");
            }
        }
    }
}
