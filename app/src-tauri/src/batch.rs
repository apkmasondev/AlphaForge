//! Batch export: run the pipeline on many files and write the results.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use af_core::export::{write_atomic, ExportSettings, Planned, Planner, SourceRef};
use af_core::imageio;
use af_core::pipeline::{self, ExecContext, OutFormat, Pipeline, Stage};
use af_core::{CancelToken, Error, Result};
use parking_lot::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::state::{AppState, ItemStatus, OutInfo, Source};

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ProgressEvt {
    done: usize,
    total: usize,
    current: Option<u64>,
    label: String,
    fraction: f32,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Summary {
    pub ok: usize,
    pub failed: usize,
    pub skipped: usize,
    pub cancelled: bool,
    pub elapsed_ms: u64,
    pub in_bytes: u64,
    pub out_bytes: u64,
    pub folder: Option<String>,
    pub errors: Vec<(String, String)>,
}

pub fn start(app: &AppHandle, ids: Vec<u64>, pipe: Pipeline, settings: ExportSettings) -> Result<()> {
    let state = app.state::<AppState>();
    if ids.is_empty() {
        return Err(Error::Invalid("no images to export".into()));
    }
    if settings.location == af_core::export::Location::Custom {
        let f = settings.folder.clone().ok_or_else(|| Error::Invalid("choose an output folder first".into()))?;
        // Reject drive-relative ("D:foo") or relative paths: they resolve against an unpredictable
        // working directory.
        if !f.is_absolute() {
            return Err(Error::Invalid(format!("the output folder must be a full path (got “{}”)", f.display())));
        }
        std::fs::create_dir_all(&f).map_err(|e| Error::Path(f.clone(), e.to_string()))?;
    }
    let cancel = {
        let mut slot = state.export_cancel.lock();
        if slot.is_some() {
            return Err(Error::Invalid("an export is already running".into()));
        }
        let c = CancelToken::new();
        *slot = Some(c.clone());
        c
    };
    {
        let mut items = state.items.write();
        for id in &ids {
            if let Some(i) = items.map.get_mut(id) {
                if i.status != ItemStatus::Loading || i.info.is_some() {
                    i.status = ItemStatus::Queued;
                    i.error = None;
                }
            }
        }
    }
    let _ = app.emit("items-changed", ());
    let app = app.clone();
    std::thread::Builder::new()
        .name("export".into())
        .spawn(move || {
            let summary = run(&app, ids, pipe, settings, &cancel);
            *app.state::<AppState>().export_cancel.lock() = None;
            let _ = app.emit("export-done", summary);
        })
        .map_err(|e| Error::Runtime(e.to_string()))?;
    Ok(())
}

pub fn cancel(app: &AppHandle) {
    if let Some(c) = app.state::<AppState>().export_cancel.lock().as_ref() {
        c.cancel();
    }
}

fn run(app: &AppHandle, ids: Vec<u64>, pipe: Pipeline, settings: ExportSettings, cancel: &CancelToken) -> Summary {
    let state = app.state::<AppState>();
    let t0 = Instant::now();
    let total = ids.len();
    let planner = Mutex::new(Planner::new(&settings, state.paths.pictures.clone()));
    let done = AtomicUsize::new(0);
    let summary = Mutex::new(Summary { ok: 0, failed: 0, skipped: 0, cancelled: false, elapsed_ms: 0, in_bytes: 0, out_bytes: 0, folder: None, errors: vec![] });
    // GPU: overlap decode/encode of one image with inference of another. CPU: one at a time
    // (ONNX Runtime and the codecs already use every core).
    let gpu = af_core::ai::runtime::get().map(|r| r.cuda_ready).unwrap_or(false) && pipe.has_ai();
    let workers = if gpu || !pipe.has_ai() { 2 } else { 1 };
    let queue = Mutex::new(ids.into_iter());
    let emit_progress = |current: Option<u64>, label: &str, fraction: f32| {
        let _ = app.emit("export-progress", ProgressEvt { done: done.load(Ordering::Relaxed), total, current, label: label.to_string(), fraction });
    };
    emit_progress(None, "Starting", 0.0);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                if cancel.is_cancelled() {
                    break;
                }
                let Some(id) = queue.lock().next() else { break };
                set_status(app, id, ItemStatus::Processing, None, None);
                let r = af_core::guard(|| process_one(app, id, &pipe, &planner, cancel, &|label, f| emit_progress(Some(id), label, f)));
                match r {
                    Ok(Some((out, in_bytes))) => {
                        let mut sm = summary.lock();
                        sm.ok += 1;
                        sm.in_bytes += in_bytes;
                        sm.out_bytes += out.bytes;
                        if sm.folder.is_none() {
                            sm.folder = out.path.as_ref().and_then(|p| PathBuf::from(p).parent().map(|d| d.to_string_lossy().into_owned()));
                        }
                        drop(sm);
                        set_status(app, id, ItemStatus::Done, None, Some(out));
                    }
                    Ok(None) => {
                        summary.lock().skipped += 1;
                        set_status(app, id, ItemStatus::Skipped, Some("Output exists — skipped".into()), None);
                    }
                    Err(Error::Cancelled) => {
                        set_status(app, id, ItemStatus::Ready, None, None);
                        break;
                    }
                    Err(e) => {
                        let name = state.items.read().map.get(&id).map(|i| i.name.clone()).unwrap_or_default();
                        let mut sm = summary.lock();
                        sm.failed += 1;
                        sm.errors.push((name, e.to_string()));
                        drop(sm);
                        set_status(app, id, ItemStatus::Error, Some(e.to_string()), None);
                    }
                }
                done.fetch_add(1, Ordering::Relaxed);
                emit_progress(None, "", 1.0);
            });
        }
    });
    // Items that never started go back to "ready".
    if cancel.is_cancelled() {
        let mut items = state.items.write();
        for it in items.map.values_mut() {
            if it.status == ItemStatus::Queued || it.status == ItemStatus::Processing {
                it.status = ItemStatus::Ready;
            }
        }
        drop(items);
        let _ = app.emit("items-changed", ());
    }
    let mut sm = summary.into_inner();
    sm.cancelled = cancel.is_cancelled();
    sm.elapsed_ms = t0.elapsed().as_millis() as u64;
    sm
}

fn set_status(app: &AppHandle, id: u64, status: ItemStatus, error: Option<String>, out: Option<OutInfo>) {
    let state = app.state::<AppState>();
    let dto = {
        let mut items = state.items.write();
        let Some(it) = items.map.get_mut(&id) else { return };
        it.status = status;
        it.error = error;
        if out.is_some() {
            it.out = out;
        }
        it.dto()
    };
    let _ = app.emit("item-updated", dto);
}

fn process_one(app: &AppHandle, id: u64, pipe: &Pipeline, planner: &Mutex<Planner>, cancel: &CancelToken, progress: &(dyn Fn(&str, f32) + Sync)) -> Result<Option<(OutInfo, u64)>> {
    let state = app.state::<AppState>();
    let t0 = Instant::now();
    let (key, strokes, source, name, root, src_format, in_bytes) = {
        let items = state.items.read();
        // Removed from the list while the export was running: skip it quietly.
        let Some(it) = items.map.get(&id) else { return Ok(None) };
        (it.key(), it.strokes.clone(), it.source.clone(), it.name.clone(), it.root.clone(), it.info.as_ref().map(|i| i.format).unwrap_or("PNG"), it.size)
    };
    let path_buf = match &source {
        Source::File(p) => Some(p.clone()),
        Source::Memory(_) => None,
    };
    let src_ref = SourceRef { path: path_buf.as_deref(), name: &name, root: root.as_deref() };
    // With a fixed output format the target is known up front: with "Skip" an existing output
    // is skipped before any (possibly slow AI) processing.
    let mut early_target = None;
    if pipe.output.format != OutFormat::Same {
        match planner.lock().plan(&src_ref, pipe.output.resolve(src_format, false).ext())? {
            Planned::Write(p) => early_target = Some(p),
            Planned::Skip(_) => return Ok(None),
        }
    }
    progress("Loading", 0.0);
    let original = state.original(id)?;
    let n_steps = pipe.active().count().max(1);
    let step_progress = |i: usize, f: f32, label: &str| progress(label, (i as f32 + f) / (n_steps as f32 + 0.5));
    let ctx = ExecContext { engine: &state.engine, cache: &state.cache, cancel, item_key: key, strokes: &strokes, progress: &step_progress, stage: Stage::Final };
    let res = pipeline::run(pipe, original, &ctx)?;
    cancel.check()?;
    progress("Saving", 0.95);
    let has_alpha = !imageio::is_opaque(&res.image);
    let fmt = pipe.output.resolve(src_format, has_alpha);
    let bytes = imageio::encode(&res.image, &pipe.output.options_for(fmt))?;
    let target = match early_target {
        Some(t) => t,
        None => match planner.lock().plan(&src_ref, fmt.ext())? {
            Planned::Write(p) => p,
            Planned::Skip(_) => return Ok(None),
        },
    };
    write_atomic(&target, &bytes)?;
    let (w, h) = res.image.dimensions();
    Ok(Some((OutInfo { path: Some(target.to_string_lossy().into_owned()), bytes: bytes.len() as u64, width: w, height: h, format: fmt.label(), ms: t0.elapsed().as_millis() as u64 }, in_bytes)))
}
