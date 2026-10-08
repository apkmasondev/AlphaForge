//! `afimg://` — serves thumbnails, originals and preview results to the web view from memory.
//! Nothing is written to disk and nothing is reachable from outside the app process.

use std::sync::Arc;

use af_core::imageio;
use tauri::http::{Request, Response, StatusCode};
use tauri::{AppHandle, Manager};

use crate::state::AppState;

fn respond(status: StatusCode, mime: &str, body: Vec<u8>) -> Response<Vec<u8>> {
    Response::builder()
        .status(status)
        .header("Content-Type", mime)
        .header("Cache-Control", "no-store")
        .header("X-Content-Type-Options", "nosniff")
        .body(body)
        .unwrap()
}

fn not_found() -> Response<Vec<u8>> {
    respond(StatusCode::NOT_FOUND, "text/plain", b"not found".to_vec())
}

/// Encode for display: JPEG (fast, small) when opaque, PNG when there is transparency.
fn encode_for_view(img: &af_core::Rgba) -> Option<(Vec<u8>, &'static str)> {
    if imageio::is_opaque(img) {
        let o = imageio::EncodeOptions { format: imageio::Format::Jpeg, quality: 94, jpeg_progressive: false, chroma: imageio::Chroma::S444, ..Default::default() };
        imageio::encode(img, &o).ok().map(|b| (b, "image/jpeg"))
    } else {
        imageio::encode_preview_png(img).ok().map(|b| (b, "image/png"))
    }
}

pub fn handle(app: &AppHandle, req: &Request<Vec<u8>>) -> Response<Vec<u8>> {
    let path = req.uri().path().trim_start_matches('/');
    let mut parts = path.split('/');
    let kind = parts.next().unwrap_or("");
    let Some(id) = parts.next().and_then(|s| s.parse::<u64>().ok()) else { return not_found() };
    let state = app.state::<AppState>();
    match kind {
        "thumb" => match state.items.read().map.get(&id).and_then(|i| i.thumb.clone()) {
            Some(t) => respond(StatusCode::OK, "image/png", (*t).clone()),
            None => not_found(),
        },
        "original" => match state.original(id) {
            Ok(img) => match encode_for_view(&img) {
                Some((b, mime)) => respond(StatusCode::OK, mime, b),
                None => not_found(),
            },
            Err(e) => respond(StatusCode::UNPROCESSABLE_ENTITY, "text/plain", e.to_string().into_bytes()),
        },
        "result" => {
            let img = state.previews.lock().get(&id).map(|p| Arc::clone(&p.image));
            match img.and_then(|i| encode_for_view(&i)) {
                Some((b, mime)) => respond(StatusCode::OK, mime, b),
                None => not_found(),
            }
        }
        _ => not_found(),
    }
}
