//! Application state: the file list, caches and shared services.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use af_core::ai::Engine;
use af_core::imageio::{self, SourceInfo};
use af_core::mask::Stroke;
use af_core::pipeline::presets::Preset;
use af_core::pipeline::StageCache;
use af_core::{CancelToken, Error, Result, Rgba};
use parking_lot::{Mutex, RwLock};
use serde::Serialize;

use crate::settings::{Settings, Store};

pub struct AppPaths {
    pub resources: PathBuf,
    pub data: PathBuf,
    /// Output folder for images without a source folder (pasted images).
    pub pictures: PathBuf,
}

#[derive(Debug, Clone)]
pub enum Source {
    File(PathBuf),
    /// Pasted / dropped bytes (kept in memory, never written anywhere unless exported).
    Memory(Arc<Vec<u8>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemStatus {
    Loading,
    Ready,
    Queued,
    Processing,
    Done,
    Skipped,
    Error,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OutInfo {
    pub path: Option<String>,
    pub bytes: u64,
    pub width: u32,
    pub height: u32,
    pub format: &'static str,
    pub ms: u64,
}

pub struct Item {
    pub id: u64,
    pub name: String,
    pub source: Source,
    /// Folder the user dropped (for "keep folder structure").
    pub root: Option<PathBuf>,
    pub size: u64,
    /// Changes whenever the source pixels may have changed (path + size + mtime).
    pub version: u64,
    pub info: Option<SourceInfo>,
    pub thumb: Option<Arc<Vec<u8>>>,
    pub status: ItemStatus,
    pub error: Option<String>,
    pub out: Option<OutInfo>,
    pub strokes: Vec<Stroke>,
    /// Undo stack for brush edits (previous stroke lists).
    pub redo: Vec<Vec<Stroke>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemDto {
    pub id: u64,
    pub name: String,
    pub path: Option<String>,
    pub rel_dir: Option<String>,
    pub size: u64,
    pub info: Option<SourceInfo>,
    pub has_thumb: bool,
    pub status: ItemStatus,
    pub error: Option<String>,
    pub out: Option<OutInfo>,
    pub edits: usize,
    pub can_redo: bool,
}

impl Item {
    pub fn dto(&self) -> ItemDto {
        let (path, rel_dir) = match &self.source {
            Source::File(p) => {
                let rel = self.root.as_ref().and_then(|r| af_core::paths::relative_dir(r, p)).map(|d| d.to_string_lossy().into_owned()).filter(|s| !s.is_empty());
                (Some(p.to_string_lossy().into_owned()), rel)
            }
            Source::Memory(_) => (None, None),
        };
        ItemDto {
            id: self.id,
            name: self.name.clone(),
            path,
            rel_dir,
            size: self.size,
            info: self.info.clone(),
            has_thumb: self.thumb.is_some(),
            status: self.status,
            error: self.error.clone(),
            out: self.out.clone(),
            edits: self.strokes.len(),
            can_redo: !self.redo.is_empty(),
        }
    }

    /// Cache identity of the source pixels.
    pub fn key(&self) -> u64 {
        self.id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.version
    }
}

#[derive(Default)]
pub struct Items {
    pub map: HashMap<u64, Item>,
    pub order: Vec<u64>,
}

/// Byte-bounded LRU of decoded originals.
pub struct Originals {
    lru: lru::LruCache<u64, Arc<Rgba>>,
    used: usize,
    budget: usize,
}

impl Originals {
    pub fn new(budget: usize) -> Self {
        Self { lru: lru::LruCache::unbounded(), used: 0, budget }
    }
    pub fn get(&mut self, k: u64) -> Option<Arc<Rgba>> {
        self.lru.get(&k).cloned()
    }
    pub fn put(&mut self, k: u64, v: Arc<Rgba>) {
        let b = v.as_raw().len();
        if let Some(old) = self.lru.put(k, v) {
            self.used -= old.as_raw().len();
        }
        self.used += b;
        while self.used > self.budget && self.lru.len() > 1 {
            if let Some((_, old)) = self.lru.pop_lru() {
                self.used -= old.as_raw().len();
            }
        }
    }
    pub fn remove(&mut self, k: u64) {
        if let Some(old) = self.lru.pop(&k) {
            self.used -= old.as_raw().len();
        }
    }
    pub fn clear(&mut self) {
        self.lru.clear();
        self.used = 0;
    }
    pub fn set_budget(&mut self, b: usize) {
        self.budget = b;
    }
}

/// Last rendered preview for an item (full resolution, kept for the viewer / clipboard).
pub struct PreviewOut {
    pub seq: u64,
    pub image: Arc<Rgba>,
}

pub struct AppState {
    pub paths: AppPaths,
    pub store: Store,
    pub engine: Arc<Engine>,
    pub cache: Arc<StageCache>,
    pub originals: Mutex<Originals>,
    pub items: RwLock<Items>,
    pub next_id: AtomicU64,
    pub settings: RwLock<Settings>,
    pub presets: RwLock<Vec<Preset>>,
    pub previews: Mutex<HashMap<u64, PreviewOut>>,
    pub export_cancel: Mutex<Option<CancelToken>>,
    pub downloads: Mutex<HashMap<String, CancelToken>>,
    pub paste_counter: AtomicU64,
}

impl AppState {
    pub fn new_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Decoded original pixels (cached).
    pub fn original(&self, id: u64) -> Result<Arc<Rgba>> {
        let (key, source) = {
            let items = self.items.read();
            let it = items.map.get(&id).ok_or_else(|| Error::Invalid("file was removed from the list".into()))?;
            (it.key(), it.source.clone())
        };
        if let Some(img) = self.originals.lock().get(key) {
            return Ok(img);
        }
        let dec = match &source {
            Source::File(p) => imageio::decode_file(p)?,
            Source::Memory(b) => imageio::decode_bytes(b)?,
        };
        let img = Arc::new(dec.image);
        self.originals.lock().put(key, Arc::clone(&img));
        Ok(img)
    }

    pub fn item_dto(&self, id: u64) -> Option<ItemDto> {
        self.items.read().map.get(&id).map(|i| i.dto())
    }
}
