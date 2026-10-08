//! AlphaForge processing core.
//!
//! Everything here runs locally: image decoding/encoding, classic image operations,
//! ONNX Runtime inference (CPU or CUDA) and the pipeline executor. No module in this
//! crate sends image data anywhere; the only network access is the explicit,
//! allow-listed model / runtime downloader in [`download`].

pub mod ai;
pub mod cancel;
pub mod download;
pub mod error;
pub mod export;
pub mod hw;
pub mod imageio;
pub mod mask;
pub mod ops;
pub mod paths;
pub mod pipeline;

pub use cancel::CancelToken;
pub use error::{Error, Result};

/// RGBA8 image used as the common currency between pipeline steps (sRGB, straight alpha).
pub type Rgba = image::RgbaImage;

/// Run `f`, turning a panic (e.g. inside a third-party decoder) into an ordinary error so a
/// background thread can report it instead of dying silently or aborting the process.
pub fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => {
            let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "unknown".into());
            Err(Error::Runtime(format!("internal error: {msg}")))
        }
    }
}

#[cfg(test)]
mod guard_tests {
    #[test]
    fn panic_becomes_error() {
        let r: crate::Result<()> = crate::guard(|| panic!("boom"));
        assert!(matches!(r, Err(crate::Error::Runtime(m)) if m.contains("boom")));
        assert_eq!(crate::guard(|| Ok(5)).unwrap(), 5);
    }
}
