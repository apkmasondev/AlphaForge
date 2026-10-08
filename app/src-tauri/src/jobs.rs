//! Long-running downloads (model weights, GPU pack) with progress and cancel.

use std::time::{Duration, Instant};

use af_core::ai::{catalog, gpupack};
use af_core::{CancelToken, Error, Result};
use parking_lot::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::state::AppState;

pub const GPU_PACK_KEY: &str = "gpu-pack";

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Progress<'a> {
    key: &'a str,
    done: u64,
    total: u64,
    label: &'a str,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Done<'a> {
    key: &'a str,
    ok: bool,
    cancelled: bool,
    error: Option<String>,
}

fn begin(app: &AppHandle, key: &str) -> Result<CancelToken> {
    let state = app.state::<AppState>();
    let mut d = state.downloads.lock();
    if d.contains_key(key) {
        return Err(Error::Invalid("this download is already running".into()));
    }
    let c = CancelToken::new();
    d.insert(key.to_string(), c.clone());
    Ok(c)
}

fn finish(app: &AppHandle, key: &str, r: Result<()>) {
    app.state::<AppState>().downloads.lock().remove(key);
    let (ok, cancelled, error) = match r {
        Ok(()) => (true, false, None),
        Err(Error::Cancelled) => (false, true, None),
        Err(e) => (false, false, Some(e.to_string())),
    };
    let _ = app.emit("download-done", Done { key, ok, cancelled, error });
}

pub fn cancel(app: &AppHandle, key: &str) {
    if let Some(c) = app.state::<AppState>().downloads.lock().get(key) {
        c.cancel();
    }
}

fn throttled(app: AppHandle, key: String) -> impl Fn(u64, u64, &str) + Sync {
    let last = Mutex::new(Instant::now() - Duration::from_secs(1));
    move |done, total, label| {
        let mut l = last.lock();
        if l.elapsed() > Duration::from_millis(120) || done >= total {
            *l = Instant::now();
            let _ = app.emit("download-progress", Progress { key: &key, done, total, label });
        }
    }
}

pub fn install_model(app: &AppHandle, id: &str) -> Result<()> {
    let spec = catalog::get(id).ok_or_else(|| Error::Invalid(format!("unknown model {id}")))?;
    let cancel = begin(app, id)?;
    let app2 = app.clone();
    let key = id.to_string();
    std::thread::spawn(move || {
        let state = app2.state::<AppState>();
        let p = throttled(app2.clone(), key.clone());
        let r = af_core::guard(|| state.engine.install(spec, &cancel, &|d, t| p(d, t, spec.name)));
        finish(&app2, &key, r);
    });
    Ok(())
}

pub fn install_gpu_pack(app: &AppHandle) -> Result<()> {
    let cancel = begin(app, GPU_PACK_KEY)?;
    let app2 = app.clone();
    std::thread::spawn(move || {
        let dir = app2.state::<AppState>().paths.data.join("runtime").join("cuda12");
        let p = throttled(app2.clone(), GPU_PACK_KEY.to_string());
        let r = af_core::guard(|| gpupack::install(&dir, &cancel, &p));
        finish(&app2, GPU_PACK_KEY, r);
    });
    Ok(())
}
