//! Persistent user settings and user presets (JSON in the app config directory).

use std::path::{Path, PathBuf};

use af_core::ai::DevicePref;
use af_core::export::ExportSettings;
use af_core::pipeline::presets::Preset;
use af_core::pipeline::Pipeline;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Theme {
    #[default]
    System,
    Light,
    Dark,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub theme: Theme,
    /// "system", "en" or "pl".
    pub language: String,
    pub device: DevicePref,
    /// Unload AI models after this many idle minutes (0 = keep loaded).
    pub unload_after_min: u32,
    /// In-memory cache budget for decoded images and AI results (MB).
    pub cache_mb: u32,
    pub export: ExportSettings,
    pub pipeline: Option<Pipeline>,
    pub preset_id: Option<String>,
    /// Re-run the preview automatically when settings change.
    pub auto_preview: bool,
    /// Viewer backdrop: "checker", "white", "black", "gray" or "#rrggbb".
    pub viewer_bg: String,
    pub gpu_prompt_dismissed: bool,
    pub left_panel_width: u32,
    pub right_panel_width: u32,
}

impl Default for Settings {
    fn default() -> Self {
        let ram = af_core::hw::system_info().ram_total_mb;
        Self {
            theme: Theme::System,
            language: "system".into(),
            device: DevicePref::Auto,
            unload_after_min: 10,
            // one of the values offered in Settings (512 MB … 8 GB)
            cache_mb: if ram >= 24_000 { 4096 } else if ram >= 12_000 { 2048 } else if ram >= 6_000 { 1024 } else { 512 },
            export: ExportSettings::default(),
            pipeline: None,
            preset_id: None,
            auto_preview: true,
            viewer_bg: "checker".into(),
            gpu_prompt_dismissed: false,
            left_panel_width: 272,
            right_panel_width: 340,
        }
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let s = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str(&s) {
        Ok(v) => Some(v),
        Err(e) => {
            log::warn!("ignoring unreadable {}: {e}", path.display());
            // keep a copy of the broken file for the user instead of silently losing it
            let _ = std::fs::copy(path, path.with_extension("json.bak"));
            None
        }
    }
}

fn write_json<T: Serialize>(path: &Path, v: &T) -> std::io::Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(v).unwrap())?;
    std::fs::rename(&tmp, path)
}

pub struct Store {
    pub settings_path: PathBuf,
    pub presets_path: PathBuf,
}

impl Store {
    pub fn new(config_dir: &Path) -> Self {
        Self { settings_path: config_dir.join("settings.json"), presets_path: config_dir.join("presets.json") }
    }

    pub fn load_settings(&self) -> Settings {
        read_json(&self.settings_path).unwrap_or_default()
    }

    pub fn save_settings(&self, s: &Settings) {
        if let Err(e) = write_json(&self.settings_path, s) {
            log::error!("saving settings failed: {e}");
        }
    }

    pub fn load_presets(&self) -> Vec<Preset> {
        read_json::<Vec<Preset>>(&self.presets_path).unwrap_or_default().into_iter().map(|mut p| {
            p.builtin = false;
            p
        }).collect()
    }

    pub fn save_presets(&self, p: &[Preset]) {
        if let Err(e) = write_json(&self.presets_path, &p) {
            log::error!("saving presets failed: {e}");
        }
    }
}
