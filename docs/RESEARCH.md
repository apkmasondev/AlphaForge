# AlphaForge — research, model selection and technology decisions

Research date: October 2026. All measurements below were taken on the reference machine:
**AMD Ryzen 7 5800H (8C/16T), 16 GB RAM, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), driver 616.64, Windows 11**,
ONNX Runtime 1.28, 1024 × 1024 model input unless noted. "Warm" = steady state after the first run.

---

## 1. Background removal / matting

### 1.1 Candidates

| Model | Licence | Weights | Notes |
|---|---|---|---|
| **BiRefNet** (general, lite, matting, HR, dynamic) — ZhengPeng7, CAAI AIR'24 | **MIT** (code + weights, HF model cards) | 85 MB – 885 MB safetensors | Actively maintained (last update Sep 2026), many task-specific checkpoints, state of the art on DIS5K / HRSOD / P3M |
| **BEN2 Base** — PramaLLC | MIT (base model only; the stronger model is commercial/API) | 213 MB ONNX | Good hair results; only the base model is open |
| **RMBG-2.0** — BRIA AI | **CC BY-NC 4.0** (commercial use needs a paid agreement) | 1 GB | BiRefNet architecture trained on BRIA's data. **Excluded: non-commercial.** |
| RMBG-1.4 | Non-commercial (BRIA licence) | 176 MB | **Excluded.** |
| **ISNet** (DIS) — xuebinqin | Apache-2.0 | 176 MB | Fast, older; halos on fur, keeps background objects |
| **U²-Net** — xuebinqin | Apache-2.0 | 176 MB | 320 px input, fast, clearly the weakest edges |
| InSPyReNet (transparent-background) | MIT | 360 MB | Good, but slower and less maintained than BiRefNet |
| SAM / SAM2 | Apache-2.0 | — | Promptable segmentation, not automatic salient-object matting; not a fit for one-click batch work |

### 1.2 Measured speed and memory

Upstream ONNX exports (GitHub release of BiRefNet, rembg, onnx-community), ORT 1.28:

| Model / export | CPU warm | GPU warm | Peak VRAM | Notes |
|---|---|---|---|---|
| U²-Net (rembg) | 0.47 s | 0.05 s | 1.1 GB | |
| ISNet general (rembg) | 1.33 s | 0.09 s | 1.1 GB | |
| BiRefNet-lite fp32 (official ONNX) | 9.6 s | **10.2 s** | 5.9 GB (spills to shared RAM) | deformable conv emulated with ~16 000 Gather/Scatter nodes |
| BiRefNet-lite fp16 (onnx-community) | 11.7 s | 0.54 s | 5.8 GB | |
| BiRefNet general fp32 (official ONNX) | — | **21.8 s** | full | unusable on 6 GB |
| BiRefNet general fp16 (onnx-community) | — | 0.69 s (2.8 s with default ORT settings) | 5.6 GB | |
| BEN2 Base (official ONNX) | 14.0 s | 0.42 s | 3.0 GB | |

Two findings drove the final design:

1. **Native `DeformConv`.** ONNX Runtime ≥ 1.25 ships CPU and CUDA kernels for the ONNX `DeformConv`
   operator. All public BiRefNet ONNX files predate this and emulate deformable convolution with
   thousands of gather/scatter nodes that allocate ~0.8 GB scratch tensors. Re-exporting BiRefNet with a
   native `DeformConv` (see `tools/build_bg_template.py`) gives:

   | AlphaForge export | CPU warm | GPU warm | Max abs. error vs PyTorch |
   |---|---|---|---|
   | BiRefNet-lite fp32 graph | **3.5 s** (2.7× faster) | 0.24 s | 1.5e-7 |
   | BiRefNet-lite fp16 graph | 4.3 s | **0.18 s** | 2.6e-5 |
   | BiRefNet general (Swin-L) fp16 | — | **0.49 s** | 0 (weights are fp16) |
   | BiRefNet general fp32 | 10.9 s | 1.37 s | 0 |
   | BiRefNet-matting (Swin-L) fp16 | — | ~0.5 s | 4.0e-3 (mean 4e-4) |
   | BiRefNet_HR-matting 2048 px fp16 | — | 7.5 s | — |

2. **`enable_mem_pattern = false`.** With ORT's default memory-pattern planning, Swin-L models
   allocate the whole 6 GB after the first run and then spill into shared system memory
   (1.4 s → 13 s per image). Disabling it keeps steady-state memory stable. AlphaForge always sets it.

### 1.3 Quality (visual comparison on the test set)

Test images: portraits with long/curly/back-lit hair, white dog on grey (similar colours), white dog on
green, long-haired cat, lion mane, chameleon on branches (camouflage), dandelion seed head
(semi-transparent), glassware, bottle, sneakers, two cars, food on dark background, 6000 × 4000 photo,
600 × 400 photo, anime illustration, plants.

* **U²-Net** — misses thin structures (chameleon tail), soft blurry edges. Not competitive.
* **ISNet** — keeps background objects (branches, leash), grey halos on fur.
* **BiRefNet-lite** — clean subject selection, slight colour fringe on fur against strong colours.
* **BiRefNet general (Swin-L)** — best object accuracy (cars, products, glass, chameleon), cleanest
  fur edges; hair is crisp but sometimes too hard.
* **BiRefNet-matting (Swin-L)** — best soft alpha on hair, fur, dandelion; also good on objects
  (cars, shoes). The lite matting checkpoint failed on objects (lost glassware, smeared car) — matting
  checkpoints trained only on people/animals must not be used as a general model.
* **BEN2 Base** — close to BiRefNet general, but no advantage that justifies a fourth architecture.

Colour decontamination matters as much as the model: every model leaves background colour in
semi-transparent hair. AlphaForge applies the two-pass *blur-fusion foreground estimation* (Forte & Pitié
2021, the method used by BiRefNet's `refine_foreground`) plus an edge-band pass, at a bounded working
resolution so 24 MP images still finish in < 1 s.

### 1.4 Decision

| Mode in the UI | Model | Why | Ships |
|---|---|---|---|
| **Fast** | BiRefNet-lite (Swin-T) | Good quality, 0.25 s GPU / 3.6 s CPU, small | **Bundled** (85 MB fp16) |
| **Best quality** | BiRefNet general (Swin-L) | Best overall accuracy | Download 424 MB |
| **Hair & fur** | BiRefNet-matting (Swin-L) | Real soft alpha for hair/fur/smoke/glass | Download 844 MB |
| **Auto** (default) | Best quality if installed and a CUDA GPU with ≥ 4 GB is active, else Fast | | — |

"Balanced" was dropped: there is no checkpoint between Swin-T and Swin-L that is meaningfully better than
lite and meaningfully faster than general. HR (2048 px) variants were measured (7.5 s/image on 6 GB) and
excluded; they only make sense on ≥ 12 GB GPUs.

---

## 2. Upscaling and enhancement

| Model | Licence | Size (fp16) | GPU (per input MP) | CPU (per input MP) | Use |
|---|---|---|---|---|---|
| **Real-ESRGAN general x4v3** (SRVGG) + weak-denoise variant + 50 % blend | **BSD-3-Clause** | 2.3 MB each | **0.35–0.5 s** | ~13 s | Default "General", denoise control, 1× AI denoise |
| **RealESRGAN_x4plus / x2plus** (RRDB) | BSD-3-Clause | 32 MB each | ~5 s | ~210 s | "Photo (max quality)" |
| **RealESRGAN_x4plus_anime_6B** | BSD-3-Clause | 8.6 MB | ~1.5 s | ~50 s | "Illustration" |
| 4x-UltraSharp | **CC BY-NC-SA** | — | — | — | **Excluded: non-commercial** |
| Nomos / SPAN community models | CC BY 4.0 | small | — | — | Viable, attribution required; Real-ESRGAN family preferred for provenance |
| SwinIR / HAT / Swin2SR | Apache-2.0 | — | slow | very slow | Too slow / too much VRAM for a 6 GB laptop GPU |
| SUPIR / diffusion upscalers | various | GBs | — | — | Out of scope (generative, slow, hallucinates) |

The upscaler runs tiled with fixed-shape, evenly sized tiles (cuDNN re-plans kernels for every new shape;
fixed shapes made the photo model 1.8× faster), 16 px overlap, edge replication, separate Lanczos alpha
upscaling and colour bleeding under transparent pixels (no dark halos on cut-outs).

"Enhance" stays modest by design: AI denoise (general model at 1×), unsharp mask and auto levels.

---

## 3. Classic image processing and codecs

| Need | Choice | Licence |
|---|---|---|
| Decode JPEG/PNG/WebP/GIF/BMP/TIFF | `image` (zune-jpeg, png, image-webp) | MIT/Apache |
| Decode AVIF | `avif-decode` 3 (pure-Rust `rav1d`) | BSD |
| Colour management | `qcms` (Firefox) – ICC → sRGB on import | MIT |
| Resize | `fast_image_resize` (SIMD, premultiplied alpha) | MIT/Apache |
| PNG | `png` + `oxipng` (lossless optimisation), NeuQuant palette mode | MIT |
| JPEG | `jpeg-encoder` (progressive, optimised Huffman) | MIT/Apache + IJG |
| WebP | libwebp via `webp` (lossy with alpha, lossless, sharp YUV) | BSD (libwebp) |
| AVIF | `ravif` / rav1e | BSD |

Not used: libimagequant/pngquant (**GPL-3**), ImageMagick (large), OpenCV (large, not needed).

Measured PNG optimisation trade-off (2866 × 2521 cut-out): deflate 3.66 MB / 0.9 s; oxipng preset 1
3.23 MB / 4.1 s; preset 2 3.18 MB / 6.2 s; preset 5 3.04 MB / 11 s → *Balanced* = preset 1.

---

## 4. Application stack

| Option | Verdict |
|---|---|
| Electron + React + Python backend | Two runtimes (Chromium + Python) → 300–500 MB before models; Python + CUDA packaging (PyInstaller) is fragile and often flagged by antivirus. |
| Tauri + web UI + Python sidecar | Small shell, but the Python sidecar keeps all packaging problems and adds IPC for every image. |
| Python + Qt (PySide6) | One runtime, but Qt styling for a modern UI is laborious, and packaging Python + onnxruntime-gpu + CUDA is the same problem. |
| **Tauri 2 + React + Rust core (chosen)** | Native Rust binary (~20 MB), system WebView2, no Python at all. ONNX Runtime is loaded dynamically (`ort` crate), image codecs are native Rust/C. Fast, single process, easy NSIS installer, updater plugin available. |

Measured: release binary 20 MB, CPU ONNX Runtime 16 MB, bundled models 190 MB.

## 5. GPU strategy

* NVIDIA detection through NVML (driver, CUDA version, compute capability, VRAM).
* The installer ships the **CPU** ONNX Runtime. NVIDIA users install the optional **GPU pack** in one
  click: ONNX Runtime 1.28.3 CUDA 12 build (GitHub release) + cuDNN 9.27 and CUDA 12.9 runtime libraries
  (NVIDIA wheels on PyPI). Only the needed DLLs are extracted with HTTP range requests (1.46 GB download,
  2.26 GB on disk, ~50 s on a fast connection) and verified file by file against pinned SHA-256.
  CUDA 12 (not 13) was chosen for broad driver (≥ 528) and GPU (Maxwell+) support.
* The CUDA build also runs on CPU, so any GPU failure (OOM, unsupported card) falls back per image
  with a clear note. Models are only placed on the GPU when its VRAM is large enough.
* Without an NVIDIA GPU everything works on the CPU (Fast model ~4 s per image).

## 6. Model distribution

The app never downloads re-uploaded or converted weights. A *template* (optimised ONNX graph whose
weights are external, ~3–4 MB, shipped in the installer) plus a *manifest* describes how to assemble
the weight blob from the **author's original `.safetensors`** at a pinned Hugging Face revision. The
download is verified with the SHA-256 of the upstream LFS object; assembly happens in memory at load
time and is byte-identical to the build-time reference (integration test `tests/templates.rs`).
safetensors / ONNX are data formats — no pickle, no code execution.

Sources: [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) · [BiRefNet HF](https://huggingface.co/ZhengPeng7) ·
[RMBG-2.0 licence](https://huggingface.co/briaai/RMBG-2.0) · [BEN2](https://huggingface.co/PramaLLC/BEN2) ·
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) · [4x-UltraSharp](https://openmodeldb.info/models/4x-UltraSharp) ·
[ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) ·
[ORT DeformConv PR #27393](https://github.com/microsoft/onnxruntime/pull/27393) ·
[Pillow/libavif notes](https://pillow.readthedocs.io/en/stable/releasenotes/11.3.0.html) · [rembg](https://github.com/danielgatis/rembg)
