//! Built-in pipeline presets.

use serde::{Deserialize, Serialize};

use super::{BgModel, OutFormat, Output, PadUnit, Pipeline, ResizeMode, Step, StepEntry, TrimMode};
use crate::ai::upscale::SrModel;
use crate::imageio::{EncodeOptions, PngLevel};
use crate::mask::Refine;
use crate::ops::backdrop::{BackdropMode, ImageFit};
use crate::ops::shadow::ShadowMode;
use crate::ops::Filter;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Preset {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub builtin: bool,
    pub pipeline: Pipeline,
}

fn e(id: &str, step: Step) -> StepEntry {
    StepEntry { id: id.into(), enabled: true, step }
}

fn bg() -> Step {
    Step::RemoveBackground { model: BgModel::Auto, refine: Refine::default() }
}
fn backdrop(mode: BackdropMode, color: [u8; 4]) -> Step {
    Step::Background { color, mode, color2: [32, 34, 40, 255], angle: 90.0, radial: false, blur: 0.5, depth: false, focus: 0.5, dim: 0.0, image: None, fit: ImageFit::Cover }
}
fn fill(color: [u8; 4]) -> Step {
    backdrop(BackdropMode::Color, color)
}
fn trim() -> Step {
    Step::Trim { mode: TrimMode::Auto, threshold: 8 }
}
fn pad(px: f32) -> Step {
    Step::Padding { top: px, right: px, bottom: px, left: px, unit: PadUnit::Px, color: [0, 0, 0, 0] }
}
fn out(format: OutFormat, quality: u8) -> Output {
    Output { format, encode: EncodeOptions { quality, ..Default::default() } }
}

pub fn builtin() -> Vec<Preset> {
    let p = |id: &str, name: &str, desc: &str, steps: Vec<StepEntry>, output: Output| Preset {
        id: id.into(),
        name: name.into(),
        description: desc.into(),
        builtin: true,
        pipeline: Pipeline { steps, output },
    };
    vec![
        p(
            "transparent-asset",
            "Transparent Asset",
            "Cut-out with tight margins, PNG with alpha.",
            vec![e("bg", bg()), e("trim", trim()), e("pad", pad(32.0))],
            out(OutFormat::Png, 90),
        ),
        p(
            "web-asset",
            "Web Asset",
            "Cut-out centered on a 1024 × 1024 transparent canvas, WebP.",
            vec![
                e("bg", bg()),
                e("trim", trim()),
                e("pad", pad(32.0)),
                e(
                    "resize",
                    Step::Resize { mode: ResizeMode::Pad, width: 1024, height: 1024, percent: 100.0, filter: Filter::Lanczos, enlarge: true, background: [0, 0, 0, 0] },
                ),
            ],
            out(OutFormat::Webp, 85),
        ),
        p(
            "website-hero",
            "Website Hero",
            "Full-width banner: max 1920 px wide, efficient WebP.",
            vec![e(
                "resize",
                Step::Resize { mode: ResizeMode::Width, width: 1920, height: 0, percent: 100.0, filter: Filter::Lanczos, enlarge: false, background: [0, 0, 0, 0] },
            )],
            out(OutFormat::Webp, 80),
        ),
        p(
            "thumbnail",
            "Thumbnail",
            "400 × 400 center crop, small WebP.",
            vec![e(
                "resize",
                Step::Resize { mode: ResizeMode::Fill, width: 400, height: 400, percent: 100.0, filter: Filter::Lanczos, enlarge: true, background: [0, 0, 0, 0] },
            )],
            out(OutFormat::Webp, 78),
        ),
        p(
            "product-white",
            "Product Photo",
            "E-commerce: subject on pure white, 2000 × 2000, JPG.",
            vec![
                e("bg", bg()),
                e("trim", trim()),
                e("pad", Step::Padding { top: 6.0, right: 6.0, bottom: 6.0, left: 6.0, unit: PadUnit::Percent, color: [0, 0, 0, 0] }),
                e(
                    "resize",
                    Step::Resize { mode: ResizeMode::Pad, width: 2000, height: 2000, percent: 100.0, filter: Filter::Lanczos, enlarge: true, background: [255, 255, 255, 255] },
                ),
                e("fill", fill([255, 255, 255, 255])),
            ],
            out(OutFormat::Jpeg, 90),
        ),
        p(
            "blurred-background",
            "Blurred Background",
            "Subject in focus, original background softly blurred by distance (AI depth), JPG.",
            vec![
                e("bg", bg()),
                e("blur", Step::Background { color: [255, 255, 255, 255], mode: BackdropMode::Blur, color2: [32, 34, 40, 255], angle: 90.0, radial: false, blur: 0.55, depth: true, focus: 0.5, dim: 0.0, image: None, fit: ImageFit::Cover }),
            ],
            out(OutFormat::Jpeg, 92),
        ),
        p(
            "sticker",
            "Sticker",
            "Cut-out with a white border and a soft shadow, PNG with alpha.",
            vec![
                e("bg", bg()),
                e("trim", trim()),
                e("outline", Step::Outline { thickness: 0.5, smooth: 0.3, color: [255, 255, 255, 255] }),
                e("shadow", Step::Shadow { mode: ShadowMode::Drop, opacity: 0.35, softness: 0.35, size: 0.5, angle: 90.0, distance: 0.08, color: [0, 0, 0, 255] }),
                e("pad", pad(24.0)),
            ],
            out(OutFormat::Png, 90),
        ),
        p(
            "game-texture",
            "Game Texture",
            "Cut-out on a 1024 × 1024 power-of-two canvas, lossless PNG.",
            vec![
                e("bg", bg()),
                e("trim", trim()),
                e(
                    "resize",
                    Step::Resize { mode: ResizeMode::Pad, width: 1024, height: 1024, percent: 100.0, filter: Filter::Lanczos, enlarge: true, background: [0, 0, 0, 0] },
                ),
            ],
            Output { format: OutFormat::Png, encode: EncodeOptions { png_level: PngLevel::Max, ..Default::default() } },
        ),
        p("png-alpha", "PNG Alpha", "Remove the background, keep the original size.", vec![e("bg", bg())], Output { format: OutFormat::Png, encode: EncodeOptions { png_level: PngLevel::Max, ..Default::default() } }),
        p(
            "small-webp",
            "Small WebP",
            "Fit within 1600 px, compact WebP for sharing.",
            vec![e(
                "resize",
                Step::Resize { mode: ResizeMode::Fit, width: 1600, height: 1600, percent: 100.0, filter: Filter::Lanczos, enlarge: false, background: [0, 0, 0, 0] },
            )],
            out(OutFormat::Webp, 75),
        ),
        p(
            "upscale-4x",
            "Upscale 4×",
            "AI super-resolution, saved as high-quality JPG.",
            vec![e("up", Step::Upscale { model: SrModel::General, scale: 4, denoise: 0.5 })],
            out(OutFormat::Jpeg, 92),
        ),
        p("compress", "Compress", "Same format, smaller file (quality 80).", vec![], out(OutFormat::Same, 80)),
    ]
}

pub fn default_preset() -> Preset {
    builtin().into_iter().next().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_roundtrip_json_without_duplicate_keys() {
        for p in builtin() {
            let s = serde_json::to_string(&p.pipeline.output).unwrap();
            assert_eq!(s.matches("\"format\"").count(), 1, "{s}");
            let back: Output = serde_json::from_str(&s).unwrap();
            assert_eq!(back.format, p.pipeline.output.format);
            let full = serde_json::to_string(&p.pipeline).unwrap();
            let again: super::Pipeline = serde_json::from_str(&full).unwrap();
            assert_eq!(again, p.pipeline);
        }
    }
}
