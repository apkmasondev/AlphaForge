//! Local AI: ONNX Runtime sessions, model catalog, background matting and super-resolution.

pub mod catalog;
pub mod depth;
pub mod engine;
pub mod gpupack;
mod gpupack_data;
pub mod matting;
pub mod runtime;
pub mod template;
pub mod upscale;

pub use catalog::{Kind, ModelSpec, MODELS};
pub use engine::{Engine, ModelStatus};
pub use runtime::{Device, DevicePref, RuntimeInfo};
