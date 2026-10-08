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
