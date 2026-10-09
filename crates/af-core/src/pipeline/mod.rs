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
use crate::ops::backdrop::{BackdropMode, ImageFit};
use crate::ops::shadow::ShadowMode;
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
    /// An even, round-cornered border around the cut-out (sticker look).
    Outline {
        #[serde(default = "default_half")]
        thickness: f32,
        #[serde(default = "default_outline_smooth")]
        smooth: f32,
        /// RGBA; alpha is ignored.
        #[serde(default = "default_outline_color")]
        color: [u8; 4],
    },
    /// A shadow under the cut-out: on the ground, or a drop shadow.
    Shadow {
        #[serde(default)]
        mode: ShadowMode,
        #[serde(default = "default_shadow_opacity")]
        opacity: f32,
        #[serde(default = "default_half")]
        softness: f32,
        #[serde(default = "default_half")]
        size: f32,
        #[serde(default = "default_shadow_angle")]
        angle: f32,
        #[serde(default = "default_shadow_distance")]
        distance: f32,
        /// RGBA; alpha is ignored (opacity has its own control).
        #[serde(default = "default_shadow_color")]
        color: [u8; 4],
    },
    /// Put something behind the (cut-out) image: a colour, a gradient, a picture, or the
    /// original background blurred.
    Background {
        /// Colour mode: the colour. Gradient: start colour. Picture (fit = contain): border colour.
        color: [u8; 4],
        #[serde(default)]
        mode: BackdropMode,
        /// Gradient end colour.
        #[serde(default = "default_color2")]
        color2: [u8; 4],
        /// Gradient direction in degrees (0 = left to right, 90 = top to bottom).
        #[serde(default = "default_angle")]
        angle: f32,
        #[serde(default)]
        radial: bool,
        /// Blur strength 0..=1 (mode = blur).
        #[serde(default = "default_blur")]
        blur: f32,
        /// Depth-aware blur with the AI depth model (mode = blur).
        #[serde(default)]
        depth: bool,
        /// How deep the sharp zone around the subject is, 0..=1 (depth-aware blur).
        #[serde(default = "default_focus")]
        focus: f32,
        /// Darken the new background 0..=1.
        #[serde(default)]
        dim: f32,
        /// Picture file (mode = image).
        #[serde(default)]
        image: Option<String>,
        #[serde(default)]
        fit: ImageFit,
    },
}

fn default_color2() -> [u8; 4] {
    [32, 34, 40, 255]
}
fn default_angle() -> f32 {
    90.0
}
fn default_outline_smooth() -> f32 {
    0.3
}
fn default_outline_color() -> [u8; 4] {
    [255, 255, 255, 255]
}
fn default_shadow_opacity() -> f32 {
    0.6
}
fn default_half() -> f32 {
    0.5
}
fn default_shadow_angle() -> f32 {
    60.0
}
fn default_shadow_distance() -> f32 {
    0.25
}
fn default_shadow_color() -> [u8; 4] {
    [0, 0, 0, 255]
}
fn default_focus() -> f32 {
    0.5
}
fn default_blur() -> f32 {
    0.5
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
            Step::Outline { .. } => "outline",
            Step::Shadow { .. } => "shadow",
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
            Step::Outline { .. } => "Outline",
            Step::Shadow { .. } => "Shadow",
            Step::Background { .. } => "Background",
        }
    }

    pub fn is_ai(&self) -> bool {
        matches!(self, Step::RemoveBackground { .. } | Step::Upscale { .. })
            || matches!(self, Step::Enhance { denoise, .. } if *denoise > 0.0)
            || matches!(self, Step::Background { mode: BackdropMode::Blur, depth: true, .. })
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
                // transparency added after a fill is not covered by that fill
                Step::Padding { color, .. } if color[3] < 255 => {
                    makes_alpha = true;
                    filled = false;
                }
                Step::Resize { mode: ResizeMode::Pad, background, .. } if background[3] < 255 => {
                    makes_alpha = true;
                    filled = false;
                }
                Step::Background { mode, color, color2, .. } => {
                    filled = match mode {
                        BackdropMode::Color => color[3] == 255,
                        BackdropMode::Gradient => color[3] == 255 && color2[3] == 255,
                        BackdropMode::Blur | BackdropMode::Image => true,
                    } || filled;
                }
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
        // a shadow needs transparency to fall on: warn when an earlier step already filled it
        let mut canvas_filled = false;
        for s in self.active() {
            match &s.step {
                Step::RemoveBackground { .. } => canvas_filled = false,
                Step::Resize { mode: ResizeMode::Pad, background, .. } if background[3] == 255 => canvas_filled = true,
                Step::Background { mode, color, .. } if *mode != BackdropMode::Color || color[3] == 255 => canvas_filled = true,
                Step::Shadow { .. } if canvas_filled => {
                    w.push("The shadow comes after a step that fills the background, so it has nothing to fall on. Move \"Shadow\" above that step.".into());
                    break;
                }
                Step::Outline { .. } if canvas_filled => {
                    w.push("The outline comes after a step that fills the background, so there is no edge to outline. Move \"Outline\" above that step.".into());
                    break;
                }
                _ => {}
            }
        }
        let blur_idx = self.active().position(|s| matches!(s.step, Step::Background { mode: BackdropMode::Blur, .. }));
        if let Some(b) = blur_idx {
            if bg_idx.is_none_or(|r| r > b) {
                w.push("A blurred original background needs \"Remove background\" earlier in the pipeline.".into());
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

#[cfg(test)]
mod warning_tests {
    use super::Pipeline;

    fn pipe(steps: &str, format: &str) -> Pipeline {
        serde_json::from_str(&format!(r#"{{"steps":[{steps}],"output":{{"format":"{format}"}}}}"#)).unwrap()
    }

    #[test]
    fn jpg_warning_for_transparency_added_after_a_fill() {
        let bg = r#"{"id":"b","type":"removeBackground"}"#;
        let fill = r#"{"id":"f","type":"background","color":[255,255,255,255]}"#;
        let pad = r#"{"id":"p","type":"padding","top":10,"right":10,"bottom":10,"left":10}"#;
        assert!(pipe(&format!("{bg},{fill}"), "jpeg").warnings().is_empty());
        assert_eq!(pipe(&format!("{bg},{fill},{pad}"), "jpeg").warnings().len(), 1);
        assert_eq!(pipe(bg, "jpeg").warnings().len(), 1);
        assert!(pipe(bg, "png").warnings().is_empty());
    }

    #[test]
    fn shadow_after_a_fill_is_flagged() {
        let bg = r#"{"id":"b","type":"removeBackground"}"#;
        let pad_white = r#"{"id":"r","type":"resize","mode":"pad","width":100,"height":100,"background":[255,255,255,255]}"#;
        let shadow = r#"{"id":"s","type":"shadow"}"#;
        assert_eq!(pipe(&format!("{bg},{shadow},{pad_white}"), "png").warnings().len(), 0);
        assert_eq!(pipe(&format!("{bg},{pad_white},{shadow}"), "png").warnings().len(), 1);
    }
}
