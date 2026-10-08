//! Hardware detection (CPU, RAM, NVIDIA GPU via NVML, disk space).

use std::path::Path;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub driver_version: String,
    /// Highest CUDA version the driver supports, e.g. (12, 8).
    pub cuda_driver: (u32, u32),
    pub compute_capability: (u32, u32),
}

impl GpuInfo {
    /// The bundled CUDA 12.x runtime needs a driver that supports CUDA 12 and a GPU of
    /// compute capability 5.0 (Maxwell) or newer.
    pub fn cuda12_compatible(&self) -> Result<(), String> {
        if self.cuda_driver.0 < 12 {
            return Err(format!(
                "NVIDIA driver {} supports CUDA {}.{} only; update the driver to use GPU acceleration",
                self.driver_version, self.cuda_driver.0, self.cuda_driver.1
            ));
        }
        if self.compute_capability.0 < 5 {
            return Err(format!("{} is too old for CUDA 12 (compute capability {}.{})", self.name, self.compute_capability.0, self.compute_capability.1));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemInfo {
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub gpus: Vec<GpuInfo>,
    pub os: String,
}

/// Query NVIDIA GPUs through NVML (ships with the NVIDIA driver). Empty when no NVIDIA GPU/driver.
pub fn nvidia_gpus() -> Vec<GpuInfo> {
    let Ok(nvml) = nvml_wrapper::Nvml::init() else { return vec![] };
    let driver = nvml.sys_driver_version().unwrap_or_default();
    let cuda = nvml.sys_cuda_driver_version().unwrap_or(0);
    let cuda_driver = ((cuda / 1000) as u32, ((cuda % 1000) / 10) as u32);
    let n = nvml.device_count().unwrap_or(0);
    (0..n)
        .filter_map(|i| {
            let d = nvml.device_by_index(i).ok()?;
            let mem = d.memory_info().ok()?;
            let cc = d.cuda_compute_capability().ok()?;
            Some(GpuInfo {
                name: d.name().unwrap_or_else(|_| "NVIDIA GPU".into()),
                vram_total_mb: mem.total / (1 << 20),
                vram_free_mb: mem.free / (1 << 20),
                driver_version: driver.clone(),
                cuda_driver,
                compute_capability: (cc.major as u32, cc.minor as u32),
            })
        })
        .collect()
}

/// Current free VRAM of GPU 0 in MB (None without NVML).
pub fn vram_free_mb() -> Option<u64> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let d = nvml.device_by_index(0).ok()?;
    Some(d.memory_info().ok()?.free / (1 << 20))
}

pub fn system_info() -> SystemInfo {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
    let sys = System::new_with_specifics(RefreshKind::nothing().with_cpu(CpuRefreshKind::nothing()).with_memory(MemoryRefreshKind::everything()));
    let cpu_name = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
    SystemInfo {
        cpu_name,
        cpu_cores: System::physical_core_count().unwrap_or(1),
        cpu_threads: sys.cpus().len().max(1),
        ram_total_mb: sys.total_memory() / (1 << 20),
        ram_available_mb: sys.available_memory() / (1 << 20),
        gpus: nvidia_gpus(),
        os: System::long_os_version().unwrap_or_else(|| "Windows".into()),
    }
}

pub fn physical_cores() -> usize {
    sysinfo::System::physical_core_count().unwrap_or(4).max(1)
}

pub fn available_ram_mb() -> u64 {
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};
    let sys = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()));
    sys.available_memory() / (1 << 20)
}

/// Free space (bytes) on the volume containing `path` (walks up to an existing ancestor).
pub fn free_space(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let mut p = path.to_path_buf();
    while !p.exists() {
        if !p.pop() {
            return None;
        }
    }
    let wide: Vec<u16> = p.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let mut free: u64 = 0;
    // SAFETY: valid NUL-terminated wide string and out-pointer.
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) };
    if ok != 0 {
        Some(free)
    } else {
        None
    }
}

#[link(name = "kernel32")]
extern "system" {
    fn GetDiskFreeSpaceExW(dir: *const u16, free_avail: *mut u64, total: *mut u64, total_free: *mut u64) -> i32;
}

/// Ensure `needed` bytes (+ 200 MB headroom) are free next to `path`.
pub fn ensure_space(path: &Path, needed: u64) -> crate::Result<()> {
    if let Some(free) = free_space(path) {
        let headroom = 200 << 20;
        if free < needed + headroom {
            return Err(crate::Error::DiskSpace { needed_mb: (needed + headroom) >> 20, free_mb: free >> 20 });
        }
    }
    Ok(())
}
