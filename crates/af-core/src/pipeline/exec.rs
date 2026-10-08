//! Pipeline executor.

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use super::{BgModel, PadUnit, Pipeline, ResizeMode, Step, StageCache, TrimMode};
use crate::ops::backdrop::{self, BackdropMode, ImageFit};
use crate::ai::catalog::{self, ModelSpec};
use crate::ai::runtime::{self, Device};
use crate::ai::{matting, upscale, Engine};
use crate::mask::{self, Stroke};
use crate::ops::{self, Rect};
use crate::{CancelToken, Error, Result, Rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Run every enabled step.
    Final,
    /// Stop right after the first "Remove background" step (used by the mask editor, whose
    /// brush coordinates live in that step's image space).
    MaskEdit,
}

pub struct ExecContext<'a> {
    pub engine: &'a Engine,
    pub cache: &'a StageCache,
    pub cancel: &'a CancelToken,
    /// Identity of the source pixels (changes when the file changes).
    pub item_key: u64,
    /// Manual brush edits for the first "Remove background" step.
    pub strokes: &'a [Stroke],
    /// (step index, fraction of that step, label)
    pub progress: &'a (dyn Fn(usize, f32, &str) + Sync),
    pub stage: Stage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StepReport {
    pub id: String,
    pub kind: &'static str,
    pub ms: u64,
    pub device: Option<Device>,
    pub model: Option<&'static str>,
    pub note: Option<String>,
    pub cached: bool,
}

pub struct RunResult {
    pub image: Arc<Rgba>,
    pub reports: Vec<StepReport>,
    /// The pipeline stopped early (mask-edit stage) — `image` is the matte stage.
    pub partial: bool,
}

fn h64<T: Hash + ?Sized>(seed: u64, v: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h);
    v.hash(&mut h);
    h.finish()
}

fn step_hash(seed: u64, s: &Step) -> u64 {
    h64(seed, serde_json::to_string(s).unwrap_or_default().as_str())
}

/// Pick the background model for "Auto": best installed model the hardware runs comfortably.
pub fn resolve_bg_model(engine: &Engine, m: BgModel) -> Result<&'static ModelSpec> {
    if let Some(id) = m.catalog_id() {
        let spec = catalog::get(id).unwrap();
        engine.weights_path(spec.template)?;
        return Ok(spec);
    }
    let fast = catalog::get("bg-fast").unwrap();
    let quality = catalog::get("bg-quality").unwrap();
    let gpu_ok = runtime::get().map(|r| r.cuda_ready).unwrap_or(false) && engine.pick_device(quality).0 == Device::Cuda;
    if gpu_ok && engine.weights_path(quality.template).is_ok() {
        Ok(quality)
    } else {
        engine.weights_path(fast.template)?;
        Ok(fast)
    }
}

pub fn run(p: &Pipeline, input: Arc<Rgba>, ctx: &ExecContext) -> Result<RunResult> {
    let mut img = input;
    let mut chain = ctx.item_key;
    let mut reports = Vec::new();
    let mut first_bg = true;
    // The original pixels behind the subject of the last "Remove background", kept in step with
    // every geometric change (trim, padding, resize, upscale) so a later "Background: blur" can
    // show the real background. Its alpha marks valid pixels (0 in added padding).
    let mut behind: Option<Arc<Rgba>> = None;
    for (i, entry) in p.active().enumerate() {
        ctx.cancel.check()?;
        let t0 = Instant::now();
        let mut rep = StepReport { id: entry.id.clone(), kind: entry.step.kind(), ms: 0, device: None, model: None, note: None, cached: false };
        let progress = |f: f32, label: &str| (ctx.progress)(i, f, label);
        progress(0.0, entry.step.label());
        let mut sh = step_hash(chain, &entry.step);
        match &entry.step {
            Step::RemoveBackground { model, refine } => {
                let spec = resolve_bg_model(ctx.engine, *model)?;
                rep.model = Some(spec.id);
                let mkey = h64(chain, &("matte", spec.id));
                let matte = match ctx.cache.get_matte(mkey) {
                    Some(m) => {
                        rep.cached = true;
                        m
                    }
                    None => {
                        let dev = ctx.engine.pick_device(spec).0;
                        progress(0.05, if ctx.engine.is_loaded(spec.template, dev) { "Detecting subject" } else { "Loading AI model (first use)…" });
                        let m = Arc::new(matting::predict(ctx.engine, spec, &img, ctx.cancel)?);
                        ctx.cache.put_matte(mkey, ctx.item_key, Arc::clone(&m));
                        m
                    }
                };
                rep.device = Some(matte.device);
                rep.note = matte.note.clone();
                progress(0.7, "Refining edges");
                let strokes: &[Stroke] = if first_bg { ctx.strokes } else { &[] };
                let tb = Instant::now();
                let mut alpha = mask::build_alpha(&matte.alpha, matte.w, matte.h, &img, refine, strokes);
                let t_alpha = tb.elapsed().as_millis();
                ctx.cancel.check()?;
                if !crate::imageio::is_opaque(&img) {
                    // keep transparency that already existed in the source
                    for (a, p) in alpha.iter_mut().zip(img.as_raw().chunks_exact(4)) {
                        *a *= p[3] as f32 / 255.0;
                    }
                }
                let tc = Instant::now();
                behind = Some(Arc::clone(&img));
                img = Arc::new(mask::compose(&img, &alpha, refine.decontaminate));
                log::debug!("matte {} ms (cached={}), alpha {} ms, compose {} ms", matte.ms, rep.cached, t_alpha, tc.elapsed().as_millis());
                if first_bg && !strokes.is_empty() {
                    sh = h64(sh, serde_json::to_string(strokes).unwrap_or_default().as_str());
                }
                first_bg = false;
                if ctx.stage == Stage::MaskEdit {
                    rep.ms = t0.elapsed().as_millis() as u64;
                    reports.push(rep);
                    return Ok(RunResult { image: img, reports, partial: true });
                }
            }
            Step::Trim { mode, threshold } => {
                let opaque = crate::imageio::is_opaque(&img);
                let bbox = match (mode, opaque) {
                    (TrimMode::Alpha, _) | (TrimMode::Auto, false) => ops::alpha_bbox(&img, *threshold),
                    (TrimMode::Color, _) | (TrimMode::Auto, true) => {
                        let c = img.get_pixel(0, 0).0;
                        ops::color_bbox(&img, c, (*threshold).max(1))
                    }
                };
                match bbox {
                    Some(r) if r != (Rect { x: 0, y: 0, w: img.width(), h: img.height() }) => {
                        img = Arc::new(ops::crop(&img, r));
                        behind = behind.map(|b| Arc::new(ops::crop(&b, r)));
                    }
                    Some(_) => {}
                    None => rep.note = Some("Nothing to trim — the image is empty.".into()),
                }
            }
            Step::Padding { top, right, bottom, left, unit, color } => {
                let base = img.width().max(img.height()) as f32;
                let px = |v: f32| -> u32 {
                    let v = v.max(0.0);
                    match unit {
                        PadUnit::Px => v.round() as u32,
                        PadUnit::Percent => (v / 100.0 * base).round() as u32,
                    }
                };
                let (t, r, b, l) = (px(*top), px(*right), px(*bottom), px(*left));
                if t + r + b + l > 0 {
                    check_dims(img.width() + l + r, img.height() + t + b)?;
                    img = Arc::new(ops::pad(&img, t, r, b, l, *color));
                    behind = behind.map(|bh| Arc::new(ops::pad(&bh, t, r, b, l, [0, 0, 0, 0])));
                }
            }
            Step::Resize { mode, width, height, percent, filter, enlarge, background } => {
                if let Some(out) = resize_step(&img, *mode, *width, *height, *percent, *filter, *enlarge, *background)? {
                    img = Arc::new(out);
                    if let Some(b) = &behind {
                        behind = resize_step(b, *mode, *width, *height, *percent, *filter, *enlarge, [0, 0, 0, 0])?.map(Arc::new);
                    }
                }
            }
            Step::Upscale { model, scale, denoise } => {
                match ctx.cache.get_image(sh) {
                    Some(c) => {
                        rep.cached = true;
                        img = c;
                    }
                    None => {
                        progress(0.0, "Loading AI model…");
                        let r = upscale::upscale(ctx.engine, *model, *scale, *denoise, &img, ctx.cancel, &|f, l| progress(f, l))?;
                        rep.device = Some(r.device);
                        rep.note = r.note;
                        let out = Arc::new(r.image);
                        ctx.cache.put_image(sh, ctx.item_key, Arc::clone(&out));
                        img = out;
                    }
                }
            }
            Step::Enhance { denoise, sharpen, auto_levels } => {
                if *denoise > 0.0 {
                    let dkey = h64(chain, &("denoise", (denoise * 100.0) as u32));
                    match ctx.cache.get_image(dkey) {
                        Some(c) => {
                            rep.cached = true;
                            img = c;
                        }
                        None => {
                            let r = upscale::denoise(ctx.engine, *denoise, &img, ctx.cancel, &|f, l| progress(f * 0.9, if l == "Upscaling" { "Reducing noise" } else { l }))?;
                            rep.device = Some(r.device);
                            rep.note = r.note;
                            let out = Arc::new(r.image);
                            ctx.cache.put_image(dkey, ctx.item_key, Arc::clone(&out));
                            img = out;
                        }
                    }
                }
                if *auto_levels {
                    img = Arc::new(ops::auto_levels(&img));
                }
                if *sharpen > 0.0 {
                    img = Arc::new(ops::sharpen(&img, sharpen.clamp(0.0, 1.0) * 1.5, 1.2));
                }
            }
            Step::Background { color, mode, color2, angle, radial, blur, depth, dim, image, fit } => {
                let args = BackdropArgs { color: *color, mode: *mode, color2: *color2, angle: *angle, radial: *radial, blur: *blur, depth: *depth, dim: *dim, image: image.as_deref(), fit: *fit };
                img = apply_backdrop(&img, behind.as_deref(), ctx, chain, args, &mut rep, &progress)?;
                if *mode != BackdropMode::Color || color[3] == 255 {
                    behind = None;
                }
            }
        }
        if let Some(b) = &behind {
            if b.dimensions() != img.dimensions() {
                // AI upscale keeps the subject aligned: follow it with a classic resize
                behind = if matches!(entry.step, Step::Upscale { .. }) {
                    Some(Arc::new(ops::resize_rgba(b, img.width(), img.height(), ops::Filter::Lanczos)?))
                } else {
                    None
                };
            }
        }
        chain = sh;
        progress(1.0, "");
        rep.ms = t0.elapsed().as_millis() as u64;
        reports.push(rep);
    }
    Ok(RunResult { image: img, reports, partial: false })
}

struct BackdropArgs<'a> {
    color: [u8; 4],
    mode: BackdropMode,
    color2: [u8; 4],
    angle: f32,
    radial: bool,
    blur: f32,
    depth: bool,
    dim: f32,
    image: Option<&'a str>,
    fit: ImageFit,
}

fn apply_backdrop(img: &Arc<Rgba>, behind: Option<&Rgba>, ctx: &ExecContext, chain: u64, a: BackdropArgs, rep: &mut StepReport, progress: &dyn Fn(f32, &str)) -> Result<Arc<Rgba>> {
    let (w, h) = img.dimensions();
    let mut bg = match a.mode {
        BackdropMode::Color => {
            if a.color[3] == 0 || crate::imageio::is_opaque(img) {
                return Ok(Arc::clone(img));
            }
            if a.color[3] == 255 && a.dim <= 0.0 {
                return Ok(Arc::new(crate::imageio::flatten(img, [a.color[0], a.color[1], a.color[2]])));
            }
            Rgba::from_pixel(w, h, image::Rgba(a.color))
        }
        BackdropMode::Gradient => backdrop::gradient(w, h, a.color, a.color2, a.angle, a.radial),
        BackdropMode::Image => {
            let path = a.image.filter(|p| !p.is_empty()).ok_or_else(|| Error::Limit("Choose a background picture".into()))?;
            let path = std::path::Path::new(path);
            let md = std::fs::metadata(path).map_err(|e| Error::Path(path.to_path_buf(), e.to_string()))?;
            let mtime = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as u64).unwrap_or(0);
            let key = h64(0, &("bgimage", path.to_string_lossy().as_ref(), md.len(), mtime));
            let pic = match ctx.cache.get_image(key) {
                Some(p) => p,
                None => {
                    progress(0.1, "Loading background picture");
                    let p = Arc::new(crate::imageio::decode_file(path)?.image);
                    ctx.cache.put_image(key, 0, Arc::clone(&p));
                    p
                }
            };
            backdrop::fit_image(&pic, w, h, a.fit, a.color)?
        }
        BackdropMode::Blur => {
            let behind = behind
                .filter(|b| b.dimensions() == (w, h))
                .ok_or_else(|| Error::Limit("A blurred original background needs \"Remove background\" earlier in the pipeline.".into()))?;
            let fkey = h64(chain, &"bgfill");
            let filled = match ctx.cache.get_image(fkey) {
                Some(f) => f,
                None => {
                    progress(0.1, "Filling in behind the subject");
                    let f = Arc::new(backdrop::fill_background(behind, img));
                    ctx.cache.put_image(fkey, ctx.item_key, Arc::clone(&f));
                    f
                }
            };
            ctx.cancel.check()?;
            let depth = if a.depth {
                let dkey = h64(chain, &"depth");
                let d = match ctx.cache.get_matte(dkey) {
                    Some(d) => {
                        rep.cached = true;
                        d
                    }
                    None => {
                        let spec = crate::ai::depth::spec();
                        ctx.engine.weights_path(spec.template)?;
                        let dev = ctx.engine.pick_device(spec).0;
                        progress(0.3, if ctx.engine.is_loaded(spec.template, dev) { "Estimating depth" } else { "Loading AI model (first use)…" });
                        let src = backdrop::depth_source(behind, &filled);
                        let d = Arc::new(crate::ai::depth::predict(ctx.engine, &src, ctx.cancel)?);
                        ctx.cache.put_matte(dkey, ctx.item_key, Arc::clone(&d));
                        d
                    }
                };
                rep.device = Some(d.device);
                rep.model = Some(d.model_id);
                rep.note = d.note.clone();
                Some(backdrop::prepare_depth(&crate::ops::resize_f32_plane(&d.alpha, d.w, d.h, w, h, crate::ops::Filter::Bicubic), behind))
            } else {
                None
            };
            ctx.cancel.check()?;
            progress(0.7, "Blurring background");
            let ds = depth.as_ref().map(|d| backdrop::subject_depth(d, img));
            let mut depth = depth;
            if let (Some(d), Some(s)) = (depth.as_mut(), ds) {
                backdrop::push_invented_back(d, behind, s);
            }
            backdrop::blur_background(&filled, a.blur, depth.as_deref().zip(ds))
        }
    };
    if a.dim > 0.0 {
        backdrop::dim(&mut bg, a.dim);
    }
    ctx.cancel.check()?;
    Ok(Arc::new(backdrop::composite(img, &bg)))
}

fn check_dims(w: u32, h: u32) -> Result<()> {
    if w as u64 * h as u64 > crate::imageio::MAX_PIXELS || w > 65_000 || h > 65_000 {
        return Err(Error::TooLarge(w, h, (crate::imageio::MAX_PIXELS / 1_000_000) as u32));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resize_step(img: &Rgba, mode: ResizeMode, tw: u32, th: u32, percent: f32, filter: ops::Filter, enlarge: bool, bg: [u8; 4]) -> Result<Option<Rgba>> {
    let (w, h) = (img.width() as f64, img.height() as f64);
    let r = |v: f64| (v.round() as u32).max(1);
    let limit = |s: f64| if enlarge { s } else { s.min(1.0) };
    let (nw, nh) = match mode {
        ResizeMode::Percent => {
            let s = (percent as f64 / 100.0).max(0.001);
            (r(w * s), r(h * s))
        }
        ResizeMode::Width => {
            if tw == 0 {
                return Ok(None);
            }
            let s = limit(tw as f64 / w);
            (r(w * s), r(h * s))
        }
        ResizeMode::Height => {
            if th == 0 {
                return Ok(None);
            }
            let s = limit(th as f64 / h);
            (r(w * s), r(h * s))
        }
        ResizeMode::Fit | ResizeMode::Pad => {
            if tw == 0 && th == 0 {
                return Ok(None);
            }
            let sx = if tw > 0 { tw as f64 / w } else { f64::INFINITY };
            let sy = if th > 0 { th as f64 / h } else { f64::INFINITY };
            let s = limit(sx.min(sy));
            let (fw, fh) = (r(w * s), r(h * s));
            if mode == ResizeMode::Pad {
                let (cw, ch) = (if tw > 0 { tw } else { fw }, if th > 0 { th } else { fh });
                check_dims(cw, ch)?;
                let fitted = ops::resize_rgba(img, fw.min(cw), fh.min(ch), filter)?;
                let x = (cw as i64 - fitted.width() as i64) / 2;
                let y = (ch as i64 - fitted.height() as i64) / 2;
                return Ok(Some(ops::place_on_canvas(&fitted, cw, ch, x, y, bg)));
            }
            (fw, fh)
        }
        ResizeMode::Fill => {
            if tw == 0 || th == 0 {
                return Ok(None);
            }
            check_dims(tw, th)?;
            let s = (tw as f64 / w).max(th as f64 / h);
            let (fw, fh) = (r(w * s).max(tw), r(h * s).max(th));
            let scaled = ops::resize_rgba(img, fw, fh, filter)?;
            let x = (fw - tw) / 2;
            let y = (fh - th) / 2;
            return Ok(Some(ops::crop(&scaled, Rect { x, y, w: tw, h: th })));
        }
        ResizeMode::Exact => {
            if tw == 0 || th == 0 {
                return Ok(None);
            }
            (tw, th)
        }
    };
    if (nw, nh) == (img.width(), img.height()) {
        return Ok(None);
    }
    check_dims(nw, nh)?;
    Ok(Some(ops::resize_rgba(img, nw, nh, filter)?))
}
