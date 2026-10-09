# AlphaForge

**Local image toolkit for Windows** — remove backgrounds with a real alpha channel, trim, pad, resize,
upscale with AI, convert and compress to PNG / JPG / WebP / AVIF. One pipeline, any number of files.

> Processing happens locally on this computer. Images are not uploaded.

**Website:** https://apkmason.dev/alphaforge-site/ · **Download:** [latest release](../../releases/latest)

![AlphaForge](assets/app-icon.png)

## Highlights

* **Background removal** with BiRefNet (MIT): *Fast* (built in, works offline), *Best quality* and
  *Hair & fur* (soft alpha) — optional downloads from the author's official repository. Edge controls
  (shift, feather, hardness), colour decontamination for hair/fur, stray-speck removal, and a brush to
  keep / erase / restore areas.
* **New background** — solid colour, gradient, your own picture, or the original background blurred.
  The blur can be *depth-aware* (Depth Anything V2 Small, Apache-2.0, optional 95 MB download): things
  near the subject stay sharp and the far background melts away, like a fast lens.
* **Shadows and stickers** — a soft shadow on the ground or a drop shadow, and an even, round-cornered
  sticker outline (optionally smoothed like a die-cut sticker). The canvas grows only as much as needed.
* **Pipelines** — chain Remove background → Trim → Padding → Outline → Shadow → Resize → Upscale → Enhance →
  Background → output format. Reorder by drag, toggle steps, save as presets. Twelve built-in presets
  (Transparent Asset, Web Asset, Website Hero, Thumbnail, Product Photo, Blurred Background, Sticker, Game
  Texture, PNG Alpha, Small WebP, Upscale 4×, Compress).
* **Batch** — drop files or whole folders (folder structure can be kept), export everything with one
  click; per-file errors never stop the batch; cancel anytime.
* **AI upscale 2× / 4×** with Real-ESRGAN (BSD-3): General (fast, noise control), Photo (max detail),
  Illustration.
* **Formats** — PNG (lossless optimisation or palette reduction), JPG, WebP (lossy with alpha / lossless),
  AVIF. Live output size and savings for the selected image. Warns when a format can't keep
  transparency.
* **Fast on NVIDIA, works everywhere** — optional one-click CUDA pack (RTX 3060 Laptop: 0.2–0.5 s per
  cut-out); without an NVIDIA GPU everything runs on the CPU.
* **Desktop niceties** — Ctrl+V screenshots / copied files, copy result to clipboard, zoom/pan/fit/100 %,
  before/after slider, dark & light theme, open output folder, reprocess.
* **Private by design** — no accounts, no telemetry, no uploads, no metadata leaks in exports.
* **English and Polish UI** — follows the system language, switchable in Settings.

## Install

Download the latest `AlphaForge_<version>_x64-setup.exe` from [Releases](../../releases/latest) and run it (per-user install, no admin rights needed; the installer is not code-signed, so SmartScreen asks once — *More info → Run anyway*). Requires Windows 10
21H2+ or Windows 11 (64-bit). WebView2 is installed automatically if missing.

For NVIDIA GPUs: *Settings → AI & GPU → GPU acceleration pack → Install* (one-time 1.46 GB download,
needs driver 528+), then restart.

## Performance (reference: Ryzen 7 5800H, RTX 3060 Laptop 6 GB)

| Task | GPU | CPU |
|---|---|---|
| Remove background, Fast | 0.25 s | 3.6 s |
| Remove background, Best quality / Hair & fur | 0.5 s | 11 s |
| Upscale 4× General (per input megapixel) | 0.4 s | 13 s |
| Batch: 24 photos → transparent PNG (Best quality) | 29 s | — |
| Batch: 24 photos → Web Asset WebP | 21 s | — |

## Documentation

* [docs/RESEARCH.md](docs/RESEARCH.md) — model & technology research, benchmarks, licences, decisions
* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how it works
* [docs/SECURITY.md](docs/SECURITY.md) — privacy & security review
* [docs/TESTING.md](docs/TESTING.md) — test matrix and results
* [BUILDING.md](BUILDING.md) — building from source

## Licence

AlphaForge is MIT licensed. Bundled and downloadable models are MIT (BiRefNet), BSD-3-Clause
(Real-ESRGAN) and Apache-2.0 (Depth Anything V2 Small) — usable commercially. See `licenses/` in the install folder for all third-party notices.
