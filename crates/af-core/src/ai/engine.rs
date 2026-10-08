//! Lazily loaded ONNX Runtime sessions with explicit memory management.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ort::ep::{ArenaExtendStrategy, CUDA};
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, Session};
use parking_lot::{Mutex, RwLock};
use serde::Serialize;

use super::catalog::{self, Kind, ModelSpec};
use super::runtime::{self, Device, DevicePref};
use super::template::{self, Manifest};
use crate::{CancelToken, Error, Result};

pub struct ModelSession {
    pub template: String,
    pub kind: Kind,
    pub device: Device,
    pub manifest: Arc<Manifest>,
    session: Mutex<Session>,
    last_used: Mutex<Instant>,
    pub load_ms: u64,
}

impl ModelSession {
    /// Run the model on one NCHW float32 tensor. Cancelling the token terminates the run.
    pub fn run(&self, data: Vec<f32>, shape: [usize; 4], cancel: &CancelToken) -> Result<(Vec<usize>, Vec<f32>)> {
        cancel.check()?;
        let tensor = ort::value::Tensor::from_array((shape, data))?;
        let opts = Arc::new(RunOptions::new()?);
        let o2 = Arc::clone(&opts);
        let _guard = cancel.on_cancel(move || {
            let _ = o2.terminate();
        });
        let mut s = self.session.lock();
        let outputs = s.run_with_options(ort::inputs![self.manifest.input.name.as_str() => tensor], &*opts)?;
        let (shape, data) = outputs[self.manifest.output.name.as_str()].try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let v = data.to_vec();
        drop(outputs);
        drop(s);
        *self.last_used.lock() = Instant::now();
        cancel.check()?;
        Ok((shape, v))
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadedModel {
    pub template: String,
    pub device: Device,
    pub idle_secs: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    #[serde(flatten)]
    pub spec: ModelSpec,
    pub installed: bool,
    pub path: Option<PathBuf>,
    pub disk_bytes: u64,
}

pub struct Engine {
    /// `resources/models`: graph templates, manifests and bundled weights.
    templates_dir: PathBuf,
    /// `%LOCALAPPDATA%/AlphaForge/models`: downloaded weights.
    models_dir: PathBuf,
    pref: RwLock<DevicePref>,
    sessions: Mutex<Vec<Arc<ModelSession>>>,
    /// Serializes session creation so two threads never load the same model twice.
    load_lock: Mutex<()>,
}

impl Engine {
    pub fn new(templates_dir: PathBuf, models_dir: PathBuf, pref: DevicePref) -> Self {
        Self { templates_dir, models_dir, pref: RwLock::new(pref), sessions: Mutex::new(Vec::new()), load_lock: Mutex::new(()) }
    }

    pub fn set_pref(&self, p: DevicePref) {
        *self.pref.write() = p;
        self.unload_all();
    }

    pub fn pref(&self) -> DevicePref {
        *self.pref.read()
    }

    pub fn templates_dir(&self) -> &Path {
        &self.templates_dir
    }
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn manifest(&self, template: &str) -> Result<Manifest> {
        Manifest::load(&self.templates_dir.join(format!("{template}.manifest.json")))
    }

    /// Where the weights for a template live (bundled copy first, then downloaded copy).
    pub fn weights_path(&self, template: &str) -> Result<PathBuf> {
        let m = self.manifest(template)?;
        if let Some(b) = &m.source.bundled {
            let p = self.templates_dir.join(b);
            if p.exists() {
                return Ok(p);
            }
        }
        let p = self.downloaded_path(template);
        if p.exists() {
            return Ok(p);
        }
        Err(Error::ModelMissing(template.to_string()))
    }

    pub fn downloaded_path(&self, template: &str) -> PathBuf {
        self.models_dir.join(format!("{template}.safetensors"))
    }

    pub fn status(&self) -> Vec<ModelStatus> {
        catalog::MODELS
            .iter()
            .map(|s| {
                let path = self.weights_path(s.template).ok();
                let disk_bytes = if s.bundled { 0 } else { path.as_ref().and_then(|p| p.metadata().ok()).map(|m| m.len()).unwrap_or(0) };
                ModelStatus { spec: s.clone(), installed: path.is_some(), path, disk_bytes }
            })
            .collect()
    }

    /// Download the official weights for a catalog model (verified, resumable).
    pub fn install(&self, spec: &ModelSpec, cancel: &CancelToken, progress: &dyn Fn(u64, u64)) -> Result<()> {
        if self.weights_path(spec.template).is_ok() {
            return Ok(());
        }
        let m = self.manifest(spec.template)?;
        std::fs::create_dir_all(&self.models_dir)?;
        crate::download::download_file(&m.source.url, &self.downloaded_path(spec.template), &m.source.sha256, m.source.size, cancel, progress)
    }

    pub fn remove(&self, spec: &ModelSpec) -> Result<()> {
        if spec.bundled {
            return Err(Error::Invalid("built-in models cannot be removed".into()));
        }
        self.sessions.lock().retain(|s| s.template != spec.template);
        let p = self.downloaded_path(spec.template);
        if p.exists() {
            std::fs::remove_file(&p)?;
        }
        let _ = std::fs::remove_file(crate::download::part_path(&p));
        Ok(())
    }

    /// Choose CPU or GPU for a model. Returns a note when the GPU is skipped for a reason the
    /// user should know about.
    pub fn pick_device(&self, spec: &ModelSpec) -> (Device, Option<String>) {
        let Ok(rt) = runtime::wait() else { return (Device::Cpu, None) };
        if self.pref() == DevicePref::Cpu || !rt.cuda_ready {
            return (Device::Cpu, None);
        }
        if let Some(g) = &rt.gpu {
            if g.vram_total_mb < spec.min_vram_mb {
                return (
                    Device::Cpu,
                    Some(format!("{} needs about {} GB of VRAM; this GPU has {:.1} GB, so it runs on the CPU.", spec.name, spec.min_vram_mb / 1024 + 1, g.vram_total_mb as f32 / 1024.0)),
                );
            }
        }
        (Device::Cuda, None)
    }

    /// Get (loading if necessary) a session for `template` on `device`.
    pub fn session(&self, template: &str, kind: Kind, device: Device) -> Result<Arc<ModelSession>> {
        if let Some(s) = self.find(template, device) {
            return Ok(s);
        }
        let _l = self.load_lock.lock();
        if let Some(s) = self.find(template, device) {
            return Ok(s);
        }
        // Keep at most one model of each kind in memory (they can be several GB).
        self.sessions.lock().retain(|s| s.kind != kind);
        let s = Arc::new(self.load(template, kind, device)?);
        self.sessions.lock().push(Arc::clone(&s));
        Ok(s)
    }

    /// True when the model is already resident (no load delay).
    pub fn is_loaded(&self, template: &str, device: Device) -> bool {
        self.find(template, device).is_some()
    }

    fn find(&self, template: &str, device: Device) -> Option<Arc<ModelSession>> {
        self.sessions.lock().iter().find(|s| s.template == template && s.device == device).cloned()
    }

    fn load(&self, template: &str, kind: Kind, device: Device) -> Result<ModelSession> {
        runtime::wait()?;
        let t0 = Instant::now();
        let manifest = self.manifest(template)?;
        let prec = if device == Device::Cuda && manifest.variants.contains_key("fp16") { "fp16" } else { "fp32" };
        let variant = manifest.variants.get(prec).ok_or_else(|| Error::Runtime(format!("{template}: no {prec} graph")))?;
        let graph = self.templates_dir.join(format!("{template}.{prec}.onnx"));
        let weights = self.weights_path(template)?;
        let blob = template::assemble(variant, &weights)?;
        let t_blob = t0.elapsed().as_millis();

        let threads = runtime::get().map(|r| r.cpu_threads).unwrap_or(4);
        let mut b = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_memory_pattern(false)?
            .with_log_level(LogLevel::Error)?
            .with_intra_threads(threads)?
            .with_external_initializer_file_in_memory(&variant.blob, Cow::Owned(blob))?;
        if device == Device::Cuda {
            let cuda = CUDA::default()
                .with_device_id(0)
                .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                .build()
                .error_on_failure();
            b = b.with_execution_providers([cuda])?;
        }
        let session = b.commit_from_file(&graph)?;
        log::info!("loaded {template} ({prec}) on {:?} in {} ms (weights {} ms)", device, t0.elapsed().as_millis(), t_blob);
        Ok(ModelSession {
            template: template.to_string(),
            kind,
            device,
            manifest: Arc::new(manifest),
            session: Mutex::new(session),
            last_used: Mutex::new(Instant::now()),
            load_ms: t0.elapsed().as_millis() as u64,
        })
    }

    pub fn unload(&self, template: &str) {
        self.sessions.lock().retain(|s| s.template != template);
    }

    pub fn unload_all(&self) {
        self.sessions.lock().clear();
    }

    /// Free models that have not been used for `idle` (called periodically by the app).
    /// Sessions currently running are kept alive by their `Arc` until the run finishes.
    pub fn unload_idle(&self, idle: Duration) -> usize {
        let mut s = self.sessions.lock();
        let before = s.len();
        s.retain(|m| m.last_used.lock().elapsed() < idle);
        before - s.len()
    }

    pub fn loaded(&self) -> Vec<LoadedModel> {
        self.sessions
            .lock()
            .iter()
            .map(|s| LoadedModel { template: s.template.clone(), device: s.device, idle_secs: s.last_used.lock().elapsed().as_secs() })
            .collect()
    }
}
