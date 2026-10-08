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

    let mut indices = vec![0u8; px];
    // Error diffusion in f32 (RGBA), serpentine scan.
    let mut err_cur = vec![[0f32; 4]; w as usize + 2];
    let mut err_next = vec![[0f32; 4]; w as usize + 2];
    for y in 0..h as usize {
        let ltr = y % 2 == 0;
        for i in 0..w as usize {
            let x = if ltr { i } else { w as usize - 1 - i };
            let o = (y * w as usize + x) * 4;
            let mut v = [0f32; 4];
            for c in 0..4 {
                v[c] = (raw[o + c] as f32 + err_cur[x + 1][c] * 0.85).clamp(0.0, 255.0);
            }
            let q = [v[0] as u8, v[1] as u8, v[2] as u8, v[3] as u8];
            // fully transparent pixels map to the most transparent palette entry, no diffusion
            let idx = nq.index_of(&q);
            indices[y * w as usize + x] = idx as u8;
            let p = &map[idx * 4..idx * 4 + 4];
            let e = [v[0] - p[0] as f32, v[1] - p[1] as f32, v[2] - p[2] as f32, v[3] - p[3] as f32];
            let (fwd, back) = if ltr { (x + 2, x) } else { (x, x + 2) };
            for c in 0..4 {
                err_cur[fwd][c] += e[c] * 7.0 / 16.0;
                err_next[back][c] += e[c] * 3.0 / 16.0;
                err_next[x + 1][c] += e[c] * 5.0 / 16.0;
                err_next[fwd][c] += e[c] * 1.0 / 16.0;
            }
        }
        std::mem::swap(&mut err_cur, &mut err_next);
        err_next.iter_mut().for_each(|e| *e = [0.0; 4]);
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
