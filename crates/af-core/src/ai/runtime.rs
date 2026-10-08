//! One-time ONNX Runtime initialization and device capability detection.
//!
//! Two ONNX Runtime builds exist:
//! * the CPU build bundled with the app (always works), and
//! * the CUDA build inside the optional GPU pack.
//!
//! ONNX Runtime can only be loaded once per process, so the choice is made at start-up. The
//! CUDA build also runs on the CPU, so a failing GPU never blocks processing.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Serialize;

use super::gpupack;
use crate::hw::{self, GpuInfo};
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DevicePref {
    /// GPU when available and suitable, otherwise CPU.
    #[default]
    Auto,
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    pub fn label(self) -> &'static str {
        match self {
            Device::Cpu => "CPU",
            Device::Cuda => "GPU (CUDA)",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInfo {
    /// ONNX Runtime library in use.
    pub ort_path: PathBuf,
    /// The CUDA build is loaded and the CUDA libraries were found.
    pub cuda_ready: bool,
    /// Human readable explanation of the GPU state (shown in Settings / status bar).
    pub gpu_status: String,
    pub gpu: Option<GpuInfo>,
    pub gpu_pack_installed: bool,
    /// GPU usable in principle (NVIDIA + driver ok), independent of the pack.
    pub gpu_supported: bool,
    pub cpu_threads: usize,
}

static RUNTIME: OnceLock<std::result::Result<RuntimeInfo, String>> = OnceLock::new();

pub struct RuntimePaths {
    /// Directory with the bundled CPU `onnxruntime.dll`.
    pub cpu_ort_dir: PathBuf,
    /// Directory of the GPU pack.
    pub gpu_pack_dir: PathBuf,
}

/// Initialize ONNX Runtime (idempotent). Later calls return the first result.
pub fn init(paths: &RuntimePaths, pref: DevicePref) -> Result<&'static RuntimeInfo> {
    let r = RUNTIME.get_or_init(|| init_inner(paths, pref).map_err(|e| e.to_string()));
    r.as_ref().map_err(|e| Error::Runtime(e.clone()))
}

pub fn get() -> Option<&'static RuntimeInfo> {
    RUNTIME.get().and_then(|r| r.as_ref().ok())
}

/// Block until [`init`] has completed (it runs on a background thread at app start-up).
pub fn wait() -> Result<&'static RuntimeInfo> {
    RUNTIME.wait().as_ref().map_err(|e| Error::Runtime(e.clone()))
}

fn init_inner(paths: &RuntimePaths, pref: DevicePref) -> Result<RuntimeInfo> {
    let gpu = hw::nvidia_gpus().into_iter().next();
    let pack_installed = gpupack::is_installed(&paths.gpu_pack_dir);
    let cpu_threads = hw::physical_cores();
    let (gpu_supported, mut gpu_status) = match &gpu {
        None => (false, "No NVIDIA GPU detected — AI runs on the CPU.".to_string()),
        Some(g) => match g.cuda12_compatible() {
            Ok(()) => (true, String::new()),
            Err(why) => (false, why),
        },
    };

    let mut cuda_ready = false;
    let mut ort_path = paths.cpu_ort_dir.join("onnxruntime.dll");
    if gpu_supported && pack_installed && pref != DevicePref::Cpu {
        match gpupack::preload(&paths.gpu_pack_dir) {
            Ok(()) => {
                ort_path = paths.gpu_pack_dir.join("onnxruntime.dll");
                cuda_ready = true;
            }
            Err(e) => gpu_status = format!("GPU pack could not be loaded ({e}) — using the CPU."),
        }
    } else if gpu_supported && !pack_installed {
        gpu_status = "NVIDIA GPU found. Install the GPU acceleration pack to use it.".into();
    } else if gpu_supported && pref == DevicePref::Cpu {
        gpu_status = "GPU disabled in settings — AI runs on the CPU.".into();
    }

    load_ort(&ort_path).or_else(|e| {
        if cuda_ready {
            // Fall back to the bundled CPU build.
            cuda_ready = false;
            gpu_status = format!("CUDA runtime failed to start ({e}) — using the CPU.");
            ort_path = paths.cpu_ort_dir.join("onnxruntime.dll");
            load_ort(&ort_path)
        } else {
            Err(e)
        }
    })?;

    if cuda_ready {
        if let Some(g) = &gpu {
            gpu_status = format!("{} · {} GB VRAM · CUDA", g.name.trim_start_matches("NVIDIA ").trim(), (g.vram_total_mb as f32 / 1024.0).round());
        }
    }
    Ok(RuntimeInfo { ort_path, cuda_ready, gpu_status, gpu, gpu_pack_installed: pack_installed, gpu_supported, cpu_threads })
}

fn load_ort(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(Error::Runtime(format!("ONNX Runtime not found at {}", path.display())));
    }
    let builder = ort::init_from(path).map_err(|e| Error::Runtime(format!("failed to load ONNX Runtime: {e}")))?;
    builder.with_name("AlphaForge").with_telemetry(false).commit();
    Ok(())
}
