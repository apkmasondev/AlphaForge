//! Live preview of the selected image. Requests are "latest wins": a new request cancels the
//! one in flight, so dragging a slider never queues up work.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use af_core::ai::catalog;
use af_core::imageio;
use af_core::pipeline::{self, ExecContext, Pipeline, Stage, StepReport};
use af_core::{CancelToken, Error};
use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::state::{AppState, PreviewOut};

pub struct Request {
    pub id: u64,
    pub pipeline: Pipeline,
    pub stage: Stage,
    pub seq: u64,
}

/// Encoding the result only to measure its file size can take seconds (PNG max, AVIF). It runs
/// on its own thread so the next preview never waits for it; stale jobs are dropped.
struct SizeJob {
    id: u64,
    seq: u64,
    image: Arc<af_core::Rgba>,
    options: imageio::EncodeOptions,
    format: &'static str,
    cancel: CancelToken,
}

fn size_worker(app: AppHandle, rx: Receiver<SizeJob>) {
    while let Ok(mut job) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            job = newer;
        }
        if job.cancel.is_cancelled() {
            continue;
        }
        let t1 = Instant::now();
        let r = af_core::guard(|| imageio::encode(&job.image, &job.options));
        if job.cancel.is_cancelled() {
            continue;
        }
        match r {
            Ok(bytes) => {
                let _ = app.emit("preview-size", SizeEvt { id: job.id, seq: job.seq, bytes: bytes.len() as u64, format: job.format, ms: t1.elapsed().as_millis() as u64 });
            }
            Err(e) => {
                let _ = app.emit("preview-error", ErrorEvt { id: job.id, seq: job.seq, message: e.to_string(), missing_model: None });
            }
        }
    }
}

pub struct PreviewWorker {
    tx: Sender<Request>,
    current: Arc<Mutex<Option<CancelToken>>>,
    seq: AtomicU64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ProgressEvt<'a> {
    id: u64,
    seq: u64,
    step: usize,
    fraction: f32,
    label: &'a str,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DoneEvt {
    pub id: u64,
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub partial: bool,
    pub has_alpha: bool,
    pub format: &'static str,
    pub ms: u64,
    pub reports: Vec<StepReport>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SizeEvt {
    id: u64,
    seq: u64,
    bytes: u64,
    format: &'static str,
    ms: u64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ErrorEvt {
    pub id: u64,
    pub seq: u64,
    pub message: String,
    /// Catalog id of a model that must be downloaded first.
    pub missing_model: Option<&'static str>,
}

pub fn missing_model(e: &Error) -> Option<&'static str> {
    match e {
        Error::ModelMissing(t) => catalog::MODELS.iter().find(|m| m.template == t).map(|m| m.id),
        _ => None,
    }
}

impl PreviewWorker {
    pub fn start(app: AppHandle) -> Self {
        let (tx, rx) = unbounded::<Request>();
        let (size_tx, size_rx) = unbounded::<SizeJob>();
        let current = Arc::new(Mutex::new(None));
        let cur2 = Arc::clone(&current);
        let app2 = app.clone();
        std::thread::Builder::new().name("preview-size".into()).spawn(move || size_worker(app2, size_rx)).expect("preview size thread");
        std::thread::Builder::new().name("preview".into()).spawn(move || worker(app, rx, cur2, size_tx)).expect("preview thread");
        Self { tx, current, seq: AtomicU64::new(1) }
    }

    pub fn request(&self, id: u64, pipeline: Pipeline, stage: Stage) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        if let Some(c) = self.current.lock().as_ref() {
            c.cancel();
        }
        let _ = self.tx.send(Request { id, pipeline, stage, seq });
        seq
    }

    pub fn cancel(&self) {
        self.seq.fetch_add(1, Ordering::SeqCst);
        if let Some(c) = self.current.lock().as_ref() {
            c.cancel();
        }
    }
}

fn worker(app: AppHandle, rx: Receiver<Request>, current: Arc<Mutex<Option<CancelToken>>>, size_tx: Sender<SizeJob>) {
    while let Ok(mut req) = rx.recv() {
        // Only the newest request matters.
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }
        let token = CancelToken::new();
        *current.lock() = Some(token.clone());
        let r = af_core::guard(|| {
            run_one(&app, &req, &token, &size_tx);
            Ok(())
        });
        if let Err(e) = r {
            log::error!("preview failed: {e}");
            let _ = app.emit("preview-error", ErrorEvt { id: req.id, seq: req.seq, message: e.to_string(), missing_model: None });
        }
        // The token stays registered until the next request cancels it; that also marks this
        // request's pending size job as stale.
    }
}

fn run_one(app: &AppHandle, req: &Request, cancel: &CancelToken, size_tx: &Sender<SizeJob>) {
    let state = app.state::<AppState>();
    let t0 = Instant::now();
    let fail = |e: Error| {
        if e.is_cancelled() {
            return;
        }
        let _ = app.emit("preview-error", ErrorEvt { id: req.id, seq: req.seq, message: e.to_string(), missing_model: missing_model(&e) });
    };
    let original = match state.original(req.id) {
        Ok(o) => o,
        Err(e) => return fail(e),
    };
    let (key, strokes, src_format) = {
        let items = state.items.read();
        match items.map.get(&req.id) {
            Some(i) => (i.key(), i.strokes.clone(), i.info.as_ref().map(|x| x.format).unwrap_or("PNG")),
            None => return,
        }
    };
    let last_emit = Mutex::new(Instant::now() - Duration::from_secs(1));
    let progress = |step: usize, fraction: f32, label: &str| {
        let mut le = last_emit.lock();
        if le.elapsed() > Duration::from_millis(60) || fraction >= 1.0 {
            *le = Instant::now();
            let _ = app.emit("preview-progress", ProgressEvt { id: req.id, seq: req.seq, step, fraction, label });
        }
    };
    let ctx = ExecContext { engine: &state.engine, cache: &state.cache, cancel, item_key: key, strokes: &strokes, progress: &progress, stage: req.stage };
    let res = match pipeline::run(&req.pipeline, original, &ctx) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    if cancel.is_cancelled() {
        return;
    }
    let has_alpha = !imageio::is_opaque(&res.image);
    let fmt = req.pipeline.output.resolve(src_format, has_alpha);
    let (w, h) = res.image.dimensions();
    {
        let mut p = state.previews.lock();
        // Keep only a few full-resolution previews in memory: at most 5 and ~1 GB in total
        // (upscaled results can be hundreds of MB each). The current one is always kept.
        const MAX_COUNT: usize = 5;
        const MAX_BYTES: usize = 1 << 30;
        p.insert(req.id, PreviewOut { seq: req.seq, image: Arc::clone(&res.image) });
        let mut older: Vec<u64> = p.keys().copied().filter(|k| *k != req.id).collect();
        older.sort_by_key(|k| p[k].seq);
        let mut bytes: usize = p.values().map(|v| v.image.as_raw().len()).sum();
        for k in older {
            if p.len() <= MAX_COUNT && bytes <= MAX_BYTES {
                break;
            }
            if let Some(old) = p.remove(&k) {
                bytes -= old.image.as_raw().len();
            }
        }
    }
    let _ = app.emit(
        "preview-done",
        DoneEvt { id: req.id, seq: req.seq, width: w, height: h, partial: res.partial, has_alpha, format: fmt.label(), ms: t0.elapsed().as_millis() as u64, reports: res.reports },
    );
    if res.partial || cancel.is_cancelled() {
        return;
    }
    // Real output size (encode with the actual settings) for the "Output size / Saved %" display.
    let _ = size_tx.send(SizeJob { id: req.id, seq: req.seq, image: res.image, options: req.pipeline.output.options_for(fmt), format: fmt.label(), cancel: cancel.clone() });
}
