//! Adding files / folders / pasted images to the list, thumbnails.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use af_core::imageio;
use tauri::{AppHandle, Emitter, Manager};

use crate::state::{AppState, Item, ItemStatus, Source};

pub const EXTENSIONS: &[&str] = &["jpg", "jpeg", "jfif", "png", "webp", "avif", "bmp", "gif", "tif", "tiff"];
const MAX_FILES: usize = 5000;
const MAX_DEPTH: usize = 12;
const THUMB: u32 = 120;

pub fn is_image_path(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).map(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str())).unwrap_or(false)
}

/// Expand dropped paths: files are taken as-is (if they look like images), folders are scanned
/// recursively. Returns (file, root folder if it came from a folder).
/// The third value is true when the file limit cut the list short.
pub fn expand(paths: &[PathBuf]) -> (Vec<(PathBuf, Option<PathBuf>)>, usize, bool) {
    let mut out = Vec::new();
    let mut skipped = 0;
    for p in paths {
        if p.is_dir() {
            scan(p, p, 0, &mut out, &mut skipped);
        } else if p.is_file() {
            if is_image_path(p) {
                out.push((p.clone(), None));
            } else {
                skipped += 1;
            }
        }
        if out.len() >= MAX_FILES {
            break;
        }
    }
    let limited = out.len() >= MAX_FILES;
    out.truncate(MAX_FILES);
    (out, skipped, limited)
}

fn scan(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(PathBuf, Option<PathBuf>)>, skipped: &mut usize) {
    if depth > MAX_DEPTH || out.len() >= MAX_FILES {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name().to_ascii_lowercase());
    for e in entries {
        let p = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_symlink() {
            continue; // avoid loops / surprises
        }
        if ft.is_dir() {
            let name = e.file_name();
            let n = name.to_string_lossy();
            if n.starts_with('.') || n.eq_ignore_ascii_case("AlphaForge") {
                continue; // skip hidden folders and our own output folders
            }
            scan(root, &p, depth + 1, out, skipped);
        } else if is_image_path(&p) {
            out.push((p, Some(root.to_path_buf())));
        } else {
            *skipped += 1;
        }
    }
}

fn file_version(p: &Path) -> (u64, u64) {
    let md = std::fs::metadata(p).ok();
    let size = md.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = md.and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as u64).unwrap_or(0);
    (size, size ^ mtime.rotate_left(17))
}

/// Add files; duplicates (same path) are skipped. Returns ids of new items.
pub fn add_files(app: &AppHandle, files: Vec<(PathBuf, Option<PathBuf>)>) -> Vec<u64> {
    let state = app.state::<AppState>();
    let mut ids = Vec::new();
    {
        let mut items = state.items.write();
        let mut existing: std::collections::HashSet<String> = items
            .map
            .values()
            .filter_map(|i| match &i.source {
                Source::File(p) => Some(p.to_string_lossy().to_lowercase()),
                _ => None,
            })
            .collect();
        for (p, root) in files {
            // also catches the same file twice in one drop (a folder plus a file inside it)
            if !existing.insert(p.to_string_lossy().to_lowercase()) {
                continue;
            }
            let (size, version) = file_version(&p);
            let id = state.new_id();
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
            items.map.insert(
                id,
                Item { id, name, source: Source::File(p), root, size, version, info: None, thumb: None, status: ItemStatus::Loading, error: None, out: None, strokes: vec![], redo: vec![] },
            );
            items.order.push(id);
            ids.push(id);
        }
    }
    load_info(app, &ids);
    ids
}

pub fn add_memory(app: &AppHandle, name: String, bytes: Vec<u8>) -> u64 {
    let state = app.state::<AppState>();
    let id = state.new_id();
    let size = bytes.len() as u64;
    let version = af_core::ai::template::sha256_bytes(&bytes[..bytes.len().min(1 << 16)]).bytes().fold(size, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
    {
        let mut items = state.items.write();
        items.map.insert(
            id,
            Item { id, name, source: Source::Memory(Arc::new(bytes)), root: None, size, version, info: None, thumb: None, status: ItemStatus::Loading, error: None, out: None, strokes: vec![], redo: vec![] },
        );
        items.order.push(id);
    }
    load_info(app, &[id]);
    id
}

/// Decode in the background to fill dimensions/format and the thumbnail.
fn load_info(app: &AppHandle, ids: &[u64]) {
    for &id in ids {
        let app = app.clone();
        rayon::spawn(move || {
            let state = app.state::<AppState>();
            let source = match state.items.read().map.get(&id) {
                Some(i) => i.source.clone(),
                None => return,
            };
            // Heavy work (thumbnail resize/encode uses rayon) happens before taking any lock:
            // rayon may run other queued jobs on this thread while it waits, and those jobs
            // lock the item list too. `guard`: a panicking decoder would abort the whole app.
            let prepared = af_core::guard(|| {
                let dec = match &source {
                    Source::File(p) => imageio::decode_file(p),
                    Source::Memory(b) => imageio::decode_bytes(b),
                }?;
                let (w, h) = dec.image.dimensions();
                let s = (THUMB as f32 / w.max(h) as f32).min(1.0);
                let (tw, th) = (((w as f32 * s).round() as u32).max(1), ((h as f32 * s).round() as u32).max(1));
                let thumb = af_core::ops::resize_rgba(&dec.image, tw, th, af_core::ops::Filter::Bilinear).ok().and_then(|t| imageio::encode_preview_png(&t).ok());
                Ok((dec, thumb))
            });
            let (dto, keep) = {
                let mut items = state.items.write();
                let Some(item) = items.map.get_mut(&id) else { return };
                match prepared {
                    Ok((dec, thumb)) => {
                        item.thumb = thumb.map(Arc::new);
                        item.status = ItemStatus::Ready;
                        let (w, h) = dec.image.dimensions();
                        item.info = Some(dec.info);
                        // keep small images decoded; large ones are re-decoded on demand
                        let keep = if (w as u64 * h as u64) < 12_000_000 { Some((item.key(), dec.image)) } else { None };
                        (item.dto(), keep)
                    }
                    Err(e) => {
                        item.status = ItemStatus::Error;
                        item.error = Some(e.to_string());
                        (item.dto(), None)
                    }
                }
            };
            if let Some((key, img)) = keep {
                state.originals.lock().put(key, Arc::new(img));
            }
            let _ = app.emit("item-updated", dto);
        });
    }
}

pub fn remove(app: &AppHandle, ids: &[u64]) {
    let state = app.state::<AppState>();
    let mut items = state.items.write();
    for id in ids {
        if let Some(it) = items.map.remove(id) {
            state.originals.lock().remove(it.key());
            state.cache.invalidate_item(it.key());
            state.previews.lock().remove(id);
        }
    }
    items.order.retain(|i| !ids.contains(i));
}

/// Re-read a file from disk (used by "Reprocess" when the user edited the file elsewhere).
pub fn refresh(app: &AppHandle, id: u64) {
    let state = app.state::<AppState>();
    let mut items = state.items.write();
    if let Some(it) = items.map.get_mut(&id) {
        state.originals.lock().remove(it.key());
        state.cache.invalidate_item(it.key());
        if let Source::File(p) = &it.source {
            let (size, version) = file_version(p);
            it.size = size;
            it.version = version;
        }
        it.out = None;
        it.error = None;
        if it.status != ItemStatus::Loading {
            it.status = ItemStatus::Ready;
        }
    }
}
