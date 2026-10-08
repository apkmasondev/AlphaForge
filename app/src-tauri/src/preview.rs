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
        let current = Arc::new(Mutex::new(None));
        let cur2 = Arc::clone(&current);
        std::thread::Builder::new().name("preview".into()).spawn(move || worker(app, rx, cur2)).expect("preview thread");
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

fn worker(app: AppHandle, rx: Receiver<Request>, current: Arc<Mutex<Option<CancelToken>>>) {
    while let Ok(mut req) = rx.recv() {
        // Only the newest request matters.
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }
        let token = CancelToken::new();
        *current.lock() = Some(token.clone());
        run_one(&app, &req, &token);
        *current.lock() = None;
    }
}

fn run_one(app: &AppHandle, req: &Request, cancel: &CancelToken) {
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
        // keep only a few full-resolution previews in memory
        if p.len() > 4 {
            let keep = req.id;
            let mut ids: Vec<u64> = p.keys().copied().filter(|k| *k != keep).collect();
            ids.sort_by_key(|k| p[k].seq);
            for k in ids.into_iter().take(p.len() - 4) {
                p.remove(&k);
            }
        }
        p.insert(req.id, PreviewOut { seq: req.seq, image: Arc::clone(&res.image) });
    }
    let _ = app.emit(
        "preview-done",
        DoneEvt { id: req.id, seq: req.seq, width: w, height: h, partial: res.partial, has_alpha, format: fmt.label(), ms: t0.elapsed().as_millis() as u64, reports: res.reports },
    );
    if res.partial || cancel.is_cancelled() {
        return;
    }
    // Real output size (encode with the actual settings) for the "Output size / Saved %" display.
    let t1 = Instant::now();
    match imageio::encode(&res.image, &req.pipeline.output.options_for(fmt)) {
        Ok(bytes) if !cancel.is_cancelled() => {
            let _ = app.emit("preview-size", SizeEvt { id: req.id, seq: req.seq, bytes: bytes.len() as u64, format: fmt.label(), ms: t1.elapsed().as_millis() as u64 });
        }
        Ok(_) => {}
        Err(e) => fail(e),
    }
}
