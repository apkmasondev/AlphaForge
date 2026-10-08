//! The model catalog. Only permissively licensed models (MIT / BSD-3-Clause) are listed.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Background,
    Upscale,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSpec {
    pub id: &'static str,
    pub kind: Kind,
    /// Short UI name.
    pub name: &'static str,
    /// One-line description for the picker.
    pub tagline: &'static str,
    /// Architecture / checkpoint shown in Settings.
    pub family: &'static str,
    /// Template base name (graph + manifest in resources/models).
    pub template: &'static str,
    pub license: &'static str,
    pub homepage: &'static str,
    /// Weights ship with the installer.
    pub bundled: bool,
    /// Download size in bytes (0 when bundled).
    pub download_bytes: u64,
    /// Below this total VRAM the model runs on the CPU even when a GPU is available.
    pub min_vram_mb: u64,
    /// Typical seconds per image (background) or per input megapixel (upscale), measured on
    /// an RTX 3060 Laptop GPU and a Ryzen 7 5800H.
    pub gpu_secs: f32,
    pub cpu_secs: f32,
    /// Super-resolution factor of the network (1 for background models).
    pub scale: u32,
}

pub static MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "bg-fast",
        kind: Kind::Background,
        name: "Fast",
        tagline: "Quick, solid cut-outs. Works well on any computer.",
        family: "BiRefNet lite (Swin-T, 1024 px)",
        template: "birefnet-lite",
        license: "MIT",
        homepage: "https://github.com/ZhengPeng7/BiRefNet",
        bundled: true,
        download_bytes: 0,
        min_vram_mb: 2500,
        gpu_secs: 0.25,
        cpu_secs: 3.6,
        scale: 1,
    },
    ModelSpec {
        id: "bg-quality",
        kind: Kind::Background,
        name: "Best quality",
        tagline: "Cleanest edges on products, people, animals and cars.",
        family: "BiRefNet general (Swin-L, 1024 px)",
        template: "birefnet-general",
        license: "MIT",
        homepage: "https://huggingface.co/ZhengPeng7/BiRefNet",
        bundled: false,
        download_bytes: 444_473_596,
        min_vram_mb: 4000,
        gpu_secs: 0.5,
        cpu_secs: 11.0,
        scale: 1,
    },
    ModelSpec {
        id: "bg-hair",
        kind: Kind::Background,
        name: "Hair & fur",
        tagline: "Soft, natural alpha for hair, fur, smoke and translucent edges.",
        family: "BiRefNet matting (Swin-L, 1024 px)",
        template: "birefnet-matting",
        license: "MIT",
        homepage: "https://huggingface.co/ZhengPeng7/BiRefNet-matting",
        bundled: false,
        download_bytes: 884_907_548,
        min_vram_mb: 4000,
        gpu_secs: 0.5,
        cpu_secs: 11.0,
        scale: 1,
    },
    ModelSpec {
        id: "sr-general",
        kind: Kind::Upscale,
        name: "General",
        tagline: "Fast upscaler for photos and graphics, with noise control.",
        family: "Real-ESRGAN general v3 (SRVGG)",
        template: "realesr-general-x4v3",
        license: "BSD-3-Clause",
        homepage: "https://github.com/xinntao/Real-ESRGAN",
        bundled: true,
        download_bytes: 0,
        min_vram_mb: 1500,
        gpu_secs: 0.35,
        cpu_secs: 13.0,
        scale: 4,
    },
    ModelSpec {
        id: "sr-photo",
        kind: Kind::Upscale,
        name: "Photo (max quality)",
        tagline: "Highest detail on photos. Slow without an NVIDIA GPU.",
        family: "Real-ESRGAN x4plus (RRDB)",
        template: "RealESRGAN_x4plus",
        license: "BSD-3-Clause",
        homepage: "https://github.com/xinntao/Real-ESRGAN",
        bundled: true,
        download_bytes: 0,
        min_vram_mb: 2500,
        gpu_secs: 5.5,
        cpu_secs: 210.0,
        scale: 4,
    },
    ModelSpec {
        id: "sr-photo-x2",
        kind: Kind::Upscale,
        name: "Photo 2× (max quality)",
        tagline: "Native 2× version of the photo model.",
        family: "Real-ESRGAN x2plus (RRDB)",
        template: "RealESRGAN_x2plus",
        license: "BSD-3-Clause",
        homepage: "https://github.com/xinntao/Real-ESRGAN",
        bundled: true,
        download_bytes: 0,
        min_vram_mb: 2500,
        gpu_secs: 1.5,
        cpu_secs: 36.0,
        scale: 2,
    },
    ModelSpec {
        id: "sr-anime",
        kind: Kind::Upscale,
        name: "Illustration",
        tagline: "Line art, anime, drawings and flat graphics.",
        family: "Real-ESRGAN x4plus anime 6B",
        template: "RealESRGAN_x4plus_anime_6B",
        license: "BSD-3-Clause",
        homepage: "https://github.com/xinntao/Real-ESRGAN",
        bundled: true,
        download_bytes: 0,
        min_vram_mb: 1500,
        gpu_secs: 1.5,
        cpu_secs: 50.0,
        scale: 4,
    },
];

/// Internal-only variants of the general upscaler used for the denoise control.
pub const SR_GENERAL_WEAK_DENOISE: &str = "realesr-general-wdn-x4v3";
pub const SR_GENERAL_MID_DENOISE: &str = "realesr-general-x4v3-dn50";

pub fn get(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}
