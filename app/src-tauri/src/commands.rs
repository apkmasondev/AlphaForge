//! Tauri commands (the only API surface exposed to the web view).

use std::path::PathBuf;

use af_core::ai::{self, gpupack, ModelStatus};
use af_core::hw::SystemInfo;
use af_core::mask::Stroke;
use af_core::pipeline::presets::{self, Preset};
use af_core::pipeline::{Pipeline, Stage};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

use crate::settings::Settings;
use crate::state::{AppState, ItemDto, ItemStatus};
use crate::{batch, clipboard, items, jobs};

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bootstrap {
    pub version: &'static str,
    pub settings: Settings,
    pub presets: Vec<Preset>,
    pub items: Vec<ItemDto>,
    pub system: SystemInfo,
    pub models: Vec<ModelStatus>,
    pub gpu_pack: gpupack::PackStatus,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeDto {
    pub info: Option<ai::RuntimeInfo>,
    pub error: Option<String>,
    pub loaded: Vec<ai::engine::LoadedModel>,
}

fn all_presets(state: &AppState) -> Vec<Preset> {
    let mut v = presets::builtin();
    v.extend(state.presets.read().iter().cloned());
    v
}

#[tauri::command]
pub fn bootstrap(state: State<AppState>) -> Bootstrap {
    let items = state.items.read();
    Bootstrap {
        version: env!("CARGO_PKG_VERSION"),
        settings: state.settings.read().clone(),
        presets: all_presets(&state),
        items: items.order.iter().filter_map(|id| items.map.get(id)).map(|i| i.dto()).collect(),
        system: af_core::hw::system_info(),
        models: state.engine.status(),
        gpu_pack: gpupack::status(&state.paths.data.join("runtime").join("cuda12")),
    }
}

/// Blocks until ONNX Runtime finished initializing (normally instant after start-up).
#[tauri::command]
pub async fn runtime_info(state: State<'_, AppState>) -> CmdResult<RuntimeDto> {
    let r = tauri::async_runtime::spawn_blocking(ai::runtime::wait).await.map_err(err)?;
    Ok(RuntimeDto { info: r.as_ref().ok().map(|i| (*i).clone()), error: r.err().map(|e| e.to_string()), loaded: state.engine.loaded() })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AddResult {
    pub added: Vec<ItemDto>,
    pub skipped: usize,
    /// More images than the per-add limit were found; only the first ones were added.
    pub limited: bool,
}

fn add_and_report(app: &AppHandle, paths: Vec<PathBuf>) -> AddResult {
    let (files, skipped, limited) = items::expand(&paths);
    let ids = items::add_files(app, files);
    let state = app.state::<AppState>();
    AddResult { added: ids.iter().filter_map(|id| state.item_dto(*id)).collect(), skipped, limited }
}

#[tauri::command]
pub fn add_paths(app: AppHandle, paths: Vec<String>) -> AddResult {
    add_and_report(&app, paths.into_iter().map(|p| PathBuf::from(p.replace('/', "\\"))).collect())
}

#[tauri::command]
pub async fn open_files_dialog(app: AppHandle) -> CmdResult<AddResult> {
    let a2 = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        a2.dialog().file().set_title("Add images").add_filter("Images", items::EXTENSIONS).blocking_pick_files()
    })
    .await
    .map_err(err)?;
    let paths: Vec<PathBuf> = picked.unwrap_or_default().into_iter().filter_map(|p| p.into_path().ok()).collect();
    Ok(add_and_report(&app, paths))
}

#[tauri::command]
pub async fn open_folder_dialog(app: AppHandle) -> CmdResult<AddResult> {
    let a2 = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || a2.dialog().file().set_title("Add a folder of images").blocking_pick_folder()).await.map_err(err)?;
    let paths: Vec<PathBuf> = picked.and_then(|p| p.into_path().ok()).into_iter().collect();
    Ok(add_and_report(&app, paths))
}

#[tauri::command]
pub async fn choose_folder(app: AppHandle, title: String) -> CmdResult<Option<String>> {
    let a2 = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || a2.dialog().file().set_title(title).blocking_pick_folder()).await.map_err(err)?;
    Ok(picked.and_then(|p| p.into_path().ok()).map(|p| p.to_string_lossy().into_owned()))
}

#[tauri::command]
pub fn list_items(state: State<AppState>) -> Vec<ItemDto> {
    let items = state.items.read();
    items.order.iter().filter_map(|id| items.map.get(id)).map(|i| i.dto()).collect()
}

#[tauri::command]
pub fn remove_items(app: AppHandle, ids: Vec<u64>) {
    items::remove(&app, &ids);
}

#[tauri::command]
pub fn clear_items(app: AppHandle) {
    let ids: Vec<u64> = app.state::<AppState>().items.read().order.clone();
    items::remove(&app, &ids);
}

#[tauri::command]
pub fn reprocess(app: AppHandle, id: u64) -> Option<ItemDto> {
    items::refresh(&app, id);
    app.state::<AppState>().item_dto(id)
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum StageArg {
    Final,
    Mask,
}

#[tauri::command]
pub fn preview(state: State<AppState>, preview: State<crate::preview::PreviewWorker>, id: u64, pipeline: Pipeline, stage: StageArg) -> u64 {
    let _ = &state;
    preview.request(id, pipeline, if matches!(stage, StageArg::Mask) { Stage::MaskEdit } else { Stage::Final })
}

#[tauri::command]
pub fn cancel_preview(preview: State<crate::preview::PreviewWorker>) {
    preview.cancel();
}

#[tauri::command]
pub fn set_strokes(state: State<AppState>, id: u64, strokes: Vec<Stroke>) -> Option<ItemDto> {
    let mut items = state.items.write();
    let it = items.map.get_mut(&id)?;
    it.strokes = strokes;
    if matches!(it.status, ItemStatus::Done) {
        it.status = ItemStatus::Ready;
    }
    Some(it.dto())
}

#[tauri::command]
pub fn pipeline_warnings(pipeline: Pipeline) -> Vec<String> {
    pipeline.warnings()
}

#[tauri::command]
pub fn start_export(app: AppHandle, ids: Vec<u64>, pipeline: Pipeline) -> CmdResult<()> {
    let settings = app.state::<AppState>().settings.read().export.clone();
    batch::start(&app, ids, pipeline, settings).map_err(err)
}

#[tauri::command]
pub fn cancel_export(app: AppHandle) {
    batch::cancel(&app);
}

#[tauri::command]
pub fn paste_clipboard(app: AppHandle) -> CmdResult<clipboard::PasteResult> {
    clipboard::paste(&app).map_err(err)
}

#[tauri::command]
pub fn add_image_bytes(app: AppHandle, name: String, bytes: Vec<u8>) -> CmdResult<Option<ItemDto>> {
    if bytes.len() > 512 << 20 {
        return Err("file is too large".into());
    }
    if af_core::imageio::probe_format(&bytes).is_none() {
        return Err("not a supported image".into());
    }
    let id = items::add_memory(&app, af_core::paths::sanitize_component(&name), bytes);
    Ok(app.state::<AppState>().item_dto(id))
}

#[tauri::command]
pub fn copy_result(app: AppHandle, id: u64) -> CmdResult<()> {
    clipboard::copy(&app, id).map_err(err)
}

/// Show a file in Explorer (never executes it).
#[tauri::command]
pub fn reveal_path(path: String) -> CmdResult<()> {
    let p = PathBuf::from(&path);
    if !p.exists() {
        return Err("the file no longer exists".into());
    }
    tauri_plugin_opener::reveal_item_in_dir(&p).map_err(err)
}

/// Open a folder in Explorer. Only directories are accepted, so nothing can be launched.
#[tauri::command]
pub fn open_folder(path: String) -> CmdResult<()> {
    let p = PathBuf::from(&path);
    if !p.is_dir() {
        return Err("folder not found".into());
    }
    tauri_plugin_opener::open_path(&p, None::<&str>).map_err(err)
}

#[tauri::command]
pub fn open_licenses(state: State<AppState>) -> CmdResult<()> {
    let p = state.paths.resources.join("licenses");
    tauri_plugin_opener::open_path(&p, None::<&str>).map_err(err)
}

#[tauri::command]
pub fn open_data_folder(state: State<AppState>) -> CmdResult<()> {
    let p = state.paths.data.clone();
    std::fs::create_dir_all(&p).map_err(err)?;
    tauri_plugin_opener::open_path(&p, None::<&str>).map_err(err)
}

#[tauri::command]
pub fn save_settings(app: AppHandle, mut settings: Settings) {
    let state = app.state::<AppState>();
    settings.cache_mb = settings.cache_mb.clamp(256, 65_536);
    let prev = state.settings.read().clone();
    if prev.device != settings.device {
        state.engine.set_pref(settings.device);
    }
    if prev.cache_mb != settings.cache_mb {
        let b = settings.cache_mb as usize * (1 << 20);
        state.cache.set_budget(b / 2);
        state.originals.lock().set_budget(b / 2);
    }
    state.store.save_settings(&settings);
    *state.settings.write() = settings;
}

#[tauri::command]
pub fn save_preset(state: State<AppState>, id: Option<String>, name: String, pipeline: Pipeline) -> Vec<Preset> {
    let name = name.trim().chars().take(60).collect::<String>();
    let name = if name.is_empty() { "My preset".to_string() } else { name };
    {
        let mut user = state.presets.write();
        match id.as_ref().and_then(|id| user.iter_mut().find(|p| &p.id == id)) {
            Some(p) => {
                p.name = name;
                p.pipeline = pipeline;
            }
            None => {
                let id = format!("user-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0));
                user.push(Preset { id, name, description: "Custom preset".into(), builtin: false, pipeline });
            }
        }
        state.store.save_presets(&user);
    }
    all_presets(&state)
}

#[tauri::command]
pub fn delete_preset(state: State<AppState>, id: String) -> Vec<Preset> {
    {
        let mut user = state.presets.write();
        user.retain(|p| p.id != id);
        state.store.save_presets(&user);
    }
    all_presets(&state)
}

#[tauri::command]
pub fn models(state: State<AppState>) -> Vec<ModelStatus> {
    state.engine.status()
}

#[tauri::command]
pub fn install_model(app: AppHandle, id: String) -> CmdResult<()> {
    jobs::install_model(&app, &id).map_err(err)
}

#[tauri::command]
pub fn remove_model(state: State<AppState>, id: String) -> CmdResult<Vec<ModelStatus>> {
    let spec = ai::catalog::get(&id).ok_or("unknown model")?;
    state.engine.remove(spec).map_err(err)?;
    Ok(state.engine.status())
}

#[tauri::command]
pub fn gpu_pack_status(state: State<AppState>) -> gpupack::PackStatus {
    gpupack::status(&state.paths.data.join("runtime").join("cuda12"))
}

#[tauri::command]
pub fn install_gpu_pack(app: AppHandle) -> CmdResult<()> {
    jobs::install_gpu_pack(&app).map_err(err)
}

#[tauri::command]
pub fn remove_gpu_pack(state: State<AppState>) -> CmdResult<gpupack::PackStatus> {
    let dir = state.paths.data.join("runtime").join("cuda12");
    if ai::runtime::get().map(|r| r.cuda_ready).unwrap_or(false) {
        // The DLLs are loaded in this process; delete on next start instead.
        std::fs::write(state.paths.data.join("runtime").join("remove-gpu-pack"), b"1").map_err(err)?;
        return Err("The GPU pack is in use. It will be removed when AlphaForge restarts.".into());
    }
    gpupack::uninstall(&dir).map_err(err)?;
    Ok(gpupack::status(&dir))
}

#[tauri::command]
pub fn cancel_download(app: AppHandle, key: String) {
    jobs::cancel(&app, &key);
}

#[tauri::command]
pub fn unload_models(state: State<AppState>) {
    state.engine.unload_all();
    state.cache.clear();
    state.originals.lock().clear();
}

#[tauri::command]
pub fn restart_app(app: AppHandle) {
    app.restart();
}

#[tauri::command]
pub fn frontend_ready(app: AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.set_focus();
    }
    let _ = app.emit("backend-ready", ());
}
