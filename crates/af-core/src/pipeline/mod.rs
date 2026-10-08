//! Pipelines: an ordered list of steps + output encoding, applied identically to every file.

mod cache;
mod exec;
pub mod presets;

pub use cache::StageCache;
pub use exec::{run, ExecContext, RunResult, Stage, StepReport};

use serde::{Deserialize, Serialize};

use crate::ai::upscale::SrModel;
use crate::imageio::{EncodeOptions, Format};
use crate::mask::Refine;
use crate::ops::Filter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BgModel {
    /// Best installed model the hardware handles comfortably.
    #[default]
    Auto,
    Fast,
    Quality,
    Hair,
}

impl BgModel {
    pub fn catalog_id(self) -> Option<&'static str> {
        match self {
            BgModel::Auto => None,
            BgModel::Fast => Some("bg-fast"),
            BgModel::Quality => Some("bg-quality"),
            BgModel::Hair => Some("bg-hair"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TrimMode {
    /// Transparent margins when the image has transparency, otherwise uniform-color borders.
    #[default]
    Auto,
    Alpha,
    Color,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ResizeMode {
    /// Scale by a percentage.
    Percent,
    /// Set the width, height follows the aspect ratio.
    Width,
    /// Set the height, width follows the aspect ratio.
    Height,
    /// Fit inside width × height, keep aspect ratio.
    #[default]
    Fit,
    /// Fill width × height exactly, keep aspect ratio, crop the overflow (centered).
    Fill,
    /// Fit inside width × height, then pad to exactly width × height.
    Pad,
    /// Stretch to exactly width × height (aspect ratio not kept).
    Exact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PadUnit {
    #[default]
    Px,
    /// Percent of the longer image side.
    Percent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum Step {
    RemoveBackground {
        #[serde(default)]
        model: BgModel,
        #[serde(default)]
        refine: Refine,
    },
    Trim {
        #[serde(default)]
        mode: TrimMode,
        /// Alpha threshold (alpha mode) or color tolerance (color mode), 0..=254.
        #[serde(default = "default_trim_threshold")]
        threshold: u8,
    },
    Padding {
        #[serde(default)]
        top: f32,
        #[serde(default)]
        right: f32,
        #[serde(default)]
        bottom: f32,
        #[serde(default)]
        left: f32,
        #[serde(default)]
        unit: PadUnit,
        /// RGBA fill; transparent by default.
        #[serde(default)]
        color: [u8; 4],
    },
    Resize {
        #[serde(default)]
        mode: ResizeMode,
        #[serde(default)]
        width: u32,
        #[serde(default)]
        height: u32,
        #[serde(default = "default_percent")]
        percent: f32,
        #[serde(default)]
        filter: Filter,
        /// Allow making images larger (Fit/Width/Height/Pad). Fill/Exact always reach the size.
        #[serde(default)]
        enlarge: bool,
        /// Fill color for Pad mode.
        #[serde(default)]
        background: [u8; 4],
    },
    Upscale {
        #[serde(default)]
        model: SrModel,
        #[serde(default = "default_scale")]
        scale: u32,
        /// General model only: 0 keeps texture, 1 removes noise strongly.
        #[serde(default = "default_denoise")]
        denoise: f32,
    },
    Enhance {
        /// AI noise reduction strength (0 = off).
        #[serde(default)]
        denoise: f32,
        /// Unsharp-mask amount (0 = off, 1 = strong).
        #[serde(default)]
        sharpen: f32,
        #[serde(default)]
        auto_levels: bool,
    },
    Background {
        /// Replace transparency with this color.
        color: [u8; 4],
    },
}

fn default_trim_threshold() -> u8 {
    8
}
fn default_percent() -> f32 {
    50.0
}
fn default_scale() -> u32 {
    2
}
fn default_denoise() -> f32 {
    0.5
}

impl Step {
    pub fn kind(&self) -> &'static str {
        match self {
            Step::RemoveBackground { .. } => "removeBackground",
            Step::Trim { .. } => "trim",
            Step::Padding { .. } => "padding",
            Step::Resize { .. } => "resize",
            Step::Upscale { .. } => "upscale",
            Step::Enhance { .. } => "enhance",
            Step::Background { .. } => "background",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Step::RemoveBackground { .. } => "Remove background",
            Step::Trim { .. } => "Trim",
            Step::Padding { .. } => "Padding",
            Step::Resize { .. } => "Resize",
            Step::Upscale { .. } => "AI upscale",
            Step::Enhance { .. } => "Enhance",
            Step::Background { .. } => "Fill background",
        }
    }

    pub fn is_ai(&self) -> bool {
        matches!(self, Step::RemoveBackground { .. } | Step::Upscale { .. }) || matches!(self, Step::Enhance { denoise, .. } if *denoise > 0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepEntry {
    pub id: String,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(flatten)]
    pub step: Step,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutFormat {
    /// Same as the source when possible (PNG when transparency must be kept).
    Same,
    #[default]
    Png,
    Jpeg,
    Webp,
    Avif,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Output {
    pub format: OutFormat,
    #[serde(flatten)]
    pub encode: EncodeOptions,
}

impl Default for Output {
    fn default() -> Self {
        Self { format: OutFormat::Png, encode: EncodeOptions::default() }
    }
}

impl Output {
    /// Resolve the concrete format for a given source format / transparency.
    pub fn resolve(&self, source_format: &str, has_alpha: bool) -> Format {
        match self.format {
            OutFormat::Png => Format::Png,
            OutFormat::Jpeg => Format::Jpeg,
            OutFormat::Webp => Format::Webp,
            OutFormat::Avif => Format::Avif,
            OutFormat::Same => {
                let f = match source_format {
                    "JPEG" => Format::Jpeg,
                    "WebP" => Format::Webp,
                    "AVIF" => Format::Avif,
                    _ => Format::Png,
                };
                if has_alpha && !f.supports_alpha() {
                    Format::Png
                } else {
                    f
                }
            }
        }
    }

    pub fn options_for(&self, f: Format) -> EncodeOptions {
        EncodeOptions { format: f, ..self.encode.clone() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pipeline {
    pub steps: Vec<StepEntry>,
    pub output: Output,
}

impl Pipeline {
    pub fn active(&self) -> impl Iterator<Item = &StepEntry> {
        self.steps.iter().filter(|s| s.enabled)
    }

    /// Static warnings shown under the pipeline editor.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        let mut makes_alpha = false;
        let mut filled = false;
        for s in self.active() {
            match &s.step {
                Step::RemoveBackground { .. } => {
                    makes_alpha = true;
                    filled = false;
                }
                Step::Padding { color, .. } if color[3] < 255 => makes_alpha = true,
                Step::Resize { mode: ResizeMode::Pad, background, .. } if background[3] < 255 => makes_alpha = true,
                Step::Background { color } if color[3] == 255 => filled = true,
                _ => {}
            }
        }
        if makes_alpha && !filled && self.output.format == OutFormat::Jpeg {
            let c = self.output.encode.background;
            w.push(format!(
                "JPG does not support transparency. Transparent areas will be filled with #{:02X}{:02X}{:02X}. Use PNG, WebP or AVIF to keep them.",
                c[0], c[1], c[2]
            ));
        }
        let bg_idx = self.active().position(|s| matches!(s.step, Step::RemoveBackground { .. }));
        let up_idx = self.active().position(|s| matches!(s.step, Step::Upscale { .. }));
        if let (Some(b), Some(u)) = (bg_idx, up_idx) {
            if u < b {
                w.push("Removing the background after upscaling is slower and not more accurate. Consider moving \"AI upscale\" below \"Remove background\".".into());
            }
        }
        if self.active().filter(|s| matches!(s.step, Step::RemoveBackground { .. })).count() > 1 {
            w.push("The pipeline removes the background more than once.".into());
        }
        w
    }

    pub fn has_ai(&self) -> bool {
        self.active().any(|s| s.step.is_ai())
    }
}
