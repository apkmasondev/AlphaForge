//! Clipboard: paste screenshots / copied images / files copied in Explorer; copy results.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use af_core::{Error, Result};
use serde::Serialize;
use tauri::{AppHandle, Manager};

use crate::items;
use crate::state::AppState;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PasteResult {
    pub added: Vec<u64>,
    /// "files", "image" or "none"
    pub kind: &'static str,
}

pub fn paste(app: &AppHandle) -> Result<PasteResult> {
    // 1) Files copied in Explorer (CF_HDROP)
    if let Ok(list) = clipboard_win::get_clipboard::<Vec<String>, _>(clipboard_win::formats::FileList) {
        let paths: Vec<PathBuf> = list.into_iter().map(PathBuf::from).collect();
        if !paths.is_empty() {
            let (files, _, _) = items::expand(&paths);
            let added = items::add_files(app, files);
            return Ok(PasteResult { added, kind: "files" });
        }
    }
    // 2) Bitmap / PNG data (screenshots, "Copy image" from browsers and editors)
    let mut cb = arboard::Clipboard::new().map_err(|e| Error::Runtime(format!("clipboard unavailable: {e}")))?;
    match cb.get_image() {
        Ok(img) => {
            let (w, h) = (img.width as u32, img.height as u32);
            let rgba = af_core::Rgba::from_raw(w, h, img.bytes.into_owned()).ok_or_else(|| Error::decode("clipboard image"))?;
            let png = af_core::imageio::encode(&rgba, &af_core::imageio::EncodeOptions { png_level: af_core::imageio::PngLevel::Fast, ..Default::default() })?;
            let state = app.state::<AppState>();
            let n = state.paste_counter.fetch_add(1, Ordering::Relaxed) + 1;
            let name = if n == 1 { "Pasted image.png".to_string() } else { format!("Pasted image {n}.png") };
            let id = items::add_memory(app, name, png);
            Ok(PasteResult { added: vec![id], kind: "image" })
        }
        Err(_) => Ok(PasteResult { added: vec![], kind: "none" }),
    }
}

/// Copy the current preview result (or the original) to the clipboard with alpha.
pub fn copy(app: &AppHandle, id: u64) -> Result<()> {
    let state = app.state::<AppState>();
    let cached = state.previews.lock().get(&id).map(|p| p.image.clone());
    let img = match cached {
        Some(i) => i,
        None => state.original(id)?,
    };
    let (w, h) = img.dimensions();
    let mut cb = arboard::Clipboard::new().map_err(|e| Error::Runtime(format!("clipboard unavailable: {e}")))?;
    cb.set_image(arboard::ImageData { width: w as usize, height: h as usize, bytes: std::borrow::Cow::Borrowed(img.as_raw()) })
        .map_err(|e| Error::Runtime(format!("could not copy: {e}")))
}
