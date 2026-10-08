//! Optional "GPU acceleration pack": ONNX Runtime CUDA build + NVIDIA CUDA 12 / cuDNN 9 runtime.
//!
//! The files are fetched from their official distribution points (ONNX Runtime GitHub release,
//! NVIDIA wheels on PyPI) by extracting single entries with HTTP range requests, then verified
//! against SHA-256 digests pinned in [`super::gpupack_data`].

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use super::gpupack_data::{ARCHIVES, PACK_ID, PRELOAD_ORDER};
use crate::download;
use crate::{CancelToken, Error, Result};

pub struct PackArchive {
    pub label: &'static str,
    pub url: &'static str,
    pub files: &'static [PackFile],
}

pub struct PackFile {
    pub entry: &'static str,
    pub name: &'static str,
    pub size: u64,
    pub compressed: u64,
    pub sha256: &'static str,
}

/// Files that are part of the official archives but not needed for AlphaForge's models
/// (cuDNN RNN / attention library). Keeping them out saves ~260 MB of download and disk.
pub const OPTIONAL: &[&str] = &["cudnn_adv64_9.dll"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackMarker {
    pub id: String,
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PackStatus {
    pub installed: bool,
    pub dir: PathBuf,
    pub download_bytes: u64,
    pub disk_bytes: u64,
    pub installed_bytes: u64,
}

pub fn wanted() -> impl Iterator<Item = (&'static PackArchive, &'static PackFile)> {
    ARCHIVES.iter().flat_map(|a| a.files.iter().map(move |f| (a, f))).filter(|(_, f)| !OPTIONAL.contains(&f.name))
}

pub fn status(dir: &Path) -> PackStatus {
    let download_bytes = wanted().map(|(_, f)| f.compressed).sum();
    let disk_bytes = wanted().map(|(_, f)| f.size).sum();
    let installed = is_installed(dir);
    let installed_bytes = if installed { wanted().filter_map(|(_, f)| dir.join(f.name).metadata().ok()).map(|m| m.len()).sum() } else { 0 };
    PackStatus { installed, dir: dir.to_path_buf(), download_bytes, disk_bytes, installed_bytes }
}

/// Installed = marker matches this build's pack id and every file exists with the right size.
pub fn is_installed(dir: &Path) -> bool {
    let Ok(s) = std::fs::read_to_string(dir.join("pack.json")) else { return false };
    let Ok(m) = serde_json::from_str::<PackMarker>(&s) else { return false };
    m.id == PACK_ID && wanted().all(|(_, f)| dir.join(f.name).metadata().map(|md| md.len() == f.size).unwrap_or(false))
}

/// Download + verify the pack. `progress(done, total, label)` in compressed bytes.
pub fn install(dir: &Path, cancel: &CancelToken, progress: &(dyn Fn(u64, u64, &str) + Sync)) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let total: u64 = wanted().map(|(_, f)| f.compressed).sum();
    let need: u64 = wanted().map(|(_, f)| f.size).sum();
    let present: u64 = wanted().filter(|(_, f)| file_ok_fast(dir, f)).map(|(_, f)| f.size).sum();
    crate::hw::ensure_space(dir, need.saturating_sub(present))?;
    let _ = std::fs::remove_file(dir.join("pack.json"));
    let mut done_before = 0u64;
    for arch in ARCHIVES {
        let files: Vec<&PackFile> = arch.files.iter().filter(|f| !OPTIONAL.contains(&f.name)).collect();
        if files.iter().all(|f| file_ok_fast(dir, f)) {
            done_before += files.iter().map(|f| f.compressed).sum::<u64>();
            progress(done_before, total, arch.label);
            continue;
        }
        cancel.check()?;
        let index = download::remote_zip_index(arch.url)?;
        for f in files {
            if file_ok_fast(dir, f) {
                done_before += f.compressed;
                continue;
            }
            let entry = index.iter().find(|e| e.name == f.entry).ok_or_else(|| Error::Download(format!("{} not found in {}", f.name, arch.label)))?;
            let base = done_before;
            let mut attempt = 0;
            loop {
                let r = download::extract_remote_entry(arch.url, entry, &dir.join(f.name), f.sha256, cancel, &|n| progress(base + n, total, f.name));
                match r {
                    Ok(()) => break,
                    Err(Error::Cancelled) => return Err(Error::Cancelled),
                    Err(e @ Error::Checksum(_)) => return Err(e),
                    Err(e) => {
                        attempt += 1;
                        if attempt >= 4 {
                            return Err(e);
                        }
                        log::warn!("retrying {}: {e}", f.name);
                        std::thread::sleep(std::time::Duration::from_secs(attempt as u64));
                    }
                }
            }
            done_before += f.compressed;
            progress(done_before, total, f.name);
        }
    }
    let marker = PackMarker { id: PACK_ID.into(), files: wanted().map(|(_, f)| f.name.to_string()).collect() };
    std::fs::write(dir.join("pack.json"), serde_json::to_vec_pretty(&marker).unwrap())?;
    Ok(())
}

/// Cheap check used to skip already-downloaded files while resuming an install.
fn file_ok_fast(dir: &Path, f: &PackFile) -> bool {
    dir.join(f.name).metadata().map(|m| m.len() == f.size).unwrap_or(false)
}

/// Full verification (hash every file); used by "Repair" in the UI.
pub fn verify(dir: &Path) -> Result<()> {
    for (_, f) in wanted() {
        let p = dir.join(f.name);
        let got = super::template::sha256_file(&p)?;
        if !got.eq_ignore_ascii_case(f.sha256) {
            return Err(Error::Checksum(f.name.into()));
        }
    }
    Ok(())
}

pub fn uninstall(dir: &Path) -> Result<()> {
    if dir.exists() {
        // only remove files we own
        for (_, f) in ARCHIVES.iter().flat_map(|a| a.files.iter().map(move |f| (a, f))) {
            let _ = std::fs::remove_file(dir.join(f.name));
            let _ = std::fs::remove_file(download::part_path(&dir.join(f.name)));
        }
        let _ = std::fs::remove_file(dir.join("pack.json"));
        let _ = std::fs::remove_dir(dir);
    }
    Ok(())
}

/// Load the CUDA / cuDNN DLLs from the pack directory (dependencies first). Each DLL's own
/// directory is used to resolve its imports (`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`), so nothing
/// from `PATH` or the current directory can be picked up instead.
pub fn preload(dir: &Path) -> Result<()> {
    use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR};
    for name in PRELOAD_ORDER {
        let p = dir.join(name);
        if !p.exists() {
            if OPTIONAL.contains(name) {
                continue;
            }
            return Err(Error::Runtime(format!("GPU pack is missing {name}")));
        }
        // SAFETY: loading verified NVIDIA runtime libraries; they are intentionally never unloaded.
        let lib = unsafe { Library::load_with_flags(&p, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
            .map_err(|e| Error::Runtime(format!("could not load {name}: {e}")))?;
        std::mem::forget(lib);
    }
    Ok(())
}
