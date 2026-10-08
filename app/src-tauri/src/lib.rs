//! AlphaForge desktop application (Tauri shell around `af-core`).

mod batch;
mod clipboard;
mod commands;
mod items;
mod jobs;
mod preview;
mod protocol;
mod settings;
mod state;

use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use af_core::ai::{runtime, Engine};
use af_core::pipeline::StageCache;
use parking_lot::{Mutex, RwLock};
use tauri::{Emitter, Manager};

use crate::settings::Store;
use crate::state::{AppPaths, AppState, Items, Originals};

fn init_logging(dir: &std::path::Path) {
    let _ = std::fs::create_dir_all(dir);
    let path = dir.join("alphaforge.log");
    // keep the log small: start fresh when it grows beyond 2 MB
    if path.metadata().map(|m| m.len() > 2 << 20).unwrap_or(false) {
        let _ = std::fs::rename(&path, dir.join("alphaforge.old.log"));
    }
    let file = std::fs::OpenOptions::new().create(true).append(true).open(&path);
    let mut b = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,ort=warn,tao=warn,wry=warn,oxipng=warn"));
    if let Ok(f) = file {
        b.target(env_logger::Target::Pipe(Box::new(f)));
    }
    let _ = b.try_init();
}

fn paths_from_args(args: &[String]) -> Vec<PathBuf> {
    args.iter().skip(1).filter(|a| !a.starts_with('-')).map(PathBuf::from).filter(|p| p.exists()).collect()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
            let paths = paths_from_args(&argv);
            if !paths.is_empty() {
                let (files, _) = items::expand(&paths);
                items::add_files(app, files);
                let _ = app.emit("items-changed", ());
            }
        }))
        .plugin(tauri_plugin_window_state::Builder::default().with_state_flags(tauri_plugin_window_state::StateFlags::SIZE | tauri_plugin_window_state::StateFlags::POSITION | tauri_plugin_window_state::StateFlags::MAXIMIZED).build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .register_asynchronous_uri_scheme_protocol("afimg", |ctx, req, responder| {
            let app = ctx.app_handle().clone();
            std::thread::spawn(move || responder.respond(protocol::handle(&app, &req)));
        })
        .setup(|app| {
            let resources = app.path().resource_dir()?;
            // Stable, human-readable locations: %LOCALAPPDATA%\AlphaForge (models, GPU pack, logs)
            // and %APPDATA%\AlphaForge (settings, presets).
            let data = app.path().local_data_dir()?.join("AlphaForge");
            let config = app.path().config_dir()?.join("AlphaForge");
            let pictures = app.path().picture_dir().unwrap_or_else(|_| data.clone()).join("AlphaForge");
            init_logging(&data.join("logs"));
            log::info!("AlphaForge {} starting; resources={}", env!("CARGO_PKG_VERSION"), resources.display());

            // Deferred removal of the GPU pack (its DLLs cannot be deleted while loaded).
            let marker = data.join("runtime").join("remove-gpu-pack");
            if marker.exists() {
                let _ = af_core::ai::gpupack::uninstall(&data.join("runtime").join("cuda12"));
                let _ = std::fs::remove_file(&marker);
            }

            let store = Store::new(&config);
            let settings = store.load_settings();
            let user_presets = store.load_presets();
            let cache_bytes = settings.cache_mb as usize * (1 << 20);
            let engine = Arc::new(Engine::new(resources.join("models"), data.join("models"), settings.device));

            // ONNX Runtime + CUDA libraries load in the background so the window appears instantly.
            let rt_paths = runtime::RuntimePaths { cpu_ort_dir: resources.join("onnxruntime"), gpu_pack_dir: data.join("runtime").join("cuda12") };
            let pref = settings.device;
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let r = runtime::init(&rt_paths, pref);
                match &r {
                    Ok(info) => log::info!("runtime ready: cuda={} ({})", info.cuda_ready, info.gpu_status),
                    Err(e) => log::error!("runtime init failed: {e}"),
                }
                let _ = handle.emit("runtime-ready", ());
            });

            let state = AppState {
                paths: AppPaths { resources, data, pictures },
                store,
                engine: Arc::clone(&engine),
                cache: Arc::new(StageCache::new(cache_bytes / 2)),
                originals: Mutex::new(Originals::new(cache_bytes / 2)),
                items: RwLock::new(Items::default()),
                next_id: AtomicU64::new(1),
                settings: RwLock::new(settings),
                presets: RwLock::new(user_presets),
                previews: Mutex::new(Default::default()),
                export_cancel: Mutex::new(None),
                downloads: Mutex::new(Default::default()),
                paste_counter: AtomicU64::new(0),
            };
            app.manage(state);
            app.manage(preview::PreviewWorker::start(app.handle().clone()));

            // Free AI models (RAM/VRAM) after a period of inactivity.
            let handle = app.handle().clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(Duration::from_secs(30));
                let state = handle.state::<AppState>();
                let mins = state.settings.read().unload_after_min;
                if mins > 0 {
                    let n = state.engine.unload_idle(Duration::from_secs(mins as u64 * 60));
                    if n > 0 {
                        log::info!("unloaded {n} idle model(s)");
                        let _ = handle.emit("models-unloaded", n);
                    }
                }
            });

            // Files passed on the command line ("Open with AlphaForge").
            let args: Vec<String> = std::env::args().collect();
            let paths = paths_from_args(&args);
            if !paths.is_empty() {
                let (files, _) = items::expand(&paths);
                items::add_files(app.handle(), files);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::bootstrap,
            commands::runtime_info,
            commands::add_paths,
            commands::open_files_dialog,
            commands::open_folder_dialog,
            commands::choose_folder,
            commands::list_items,
            commands::remove_items,
            commands::clear_items,
            commands::reorder_items,
            commands::reprocess,
            commands::preview,
            commands::cancel_preview,
            commands::set_strokes,
            commands::pipeline_warnings,
            commands::start_export,
            commands::cancel_export,
            commands::paste_clipboard,
            commands::add_image_bytes,
            commands::copy_result,
            commands::reveal_path,
            commands::open_folder,
            commands::open_licenses,
            commands::open_data_folder,
            commands::save_settings,
            commands::save_preset,
            commands::delete_preset,
            commands::models,
            commands::install_model,
            commands::remove_model,
            commands::gpu_pack_status,
            commands::install_gpu_pack,
            commands::remove_gpu_pack,
            commands::cancel_download,
            commands::unload_models,
            commands::restart_app,
            commands::frontend_ready,
        ])
        .run(tauri::generate_context!())
        .expect("error while running AlphaForge");
}
