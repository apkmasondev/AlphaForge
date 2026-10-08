# Architecture

```
┌──────────────────────────── AlphaForge.exe (one process) ────────────────────────────┐
│  WebView2 UI (React + TypeScript, app/src)                                            │
│    file list · viewer (zoom/pan/compare/brush) · pipeline editor · dialogs            │
│        │  Tauri commands (JSON)          ▲ events (progress, results)                 │
│        ▼                                 │        afimg:// (in-memory images)         │
│  Tauri shell (Rust, app/src-tauri)                                                     │
│    items · preview worker (latest-wins) · batch exporter · clipboard · downloads      │
│        │                                                                              │
│        ▼                                                                              │
│  af-core (Rust library, crates/af-core)                                               │
│    imageio   decode (EXIF orientation, ICC→sRGB, limits) / encode PNG JPG WebP AVIF   │
│    ops       trim · pad · resize (premultiplied) · sharpen · levels · quantize        │
│    mask      matte refine · brush strokes · islands · guided filter · decontamination │
│    ai        runtime (ORT CPU/CUDA) · engine (sessions, VRAM policy) · templates      │
│              matting · tiled upscaling · GPU pack · model catalog                     │
│    pipeline  steps · executor · stage cache · presets                                 │
│    export    naming · conflicts · folder structure · atomic writes                    │
│    download  allow-listed HTTPS, resumable, SHA-256 verified, range-zip extraction    │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │ loads at start-up (once)
        ▼
  onnxruntime.dll  — bundled CPU build, or the CUDA build from the GPU pack
```

## Processing model

* **Preview** — the selected image is re-rendered whenever the pipeline changes (debounced 220 ms). A new
  request cancels the one in flight (ONNX Runtime runs are terminated via `RunOptions::terminate`).
  Expensive stages are cached in a byte-bounded LRU: the AI matte (model resolution, keyed by image +
  preceding steps + model) and upscale results. Moving a refine slider therefore costs only
  post-processing (tens of ms). The real output size is computed by encoding with the actual settings.
* **Export** — a background thread pool (2 workers when the GPU does the AI work, 1 when the CPU does,
  since ONNX Runtime and the codecs already use all cores). Per-file errors never stop the batch.
  Cancel stops within one inference; finished files are kept, the rest return to "ready".
* **Memory** — at most one background and one upscale model are resident; idle models are unloaded
  after a configurable time. Decoded originals and AI results share a configurable cache budget
  (default RAM/6). Images over 200 MP are refused before decoding; upscales over 160 MP are refused.
* **Threads** — UI never blocks: commands that can take time run on worker threads; heavy loops use
  rayon; ONNX Runtime uses its own intra-op pool sized to the physical cores.

## Pipeline

`Pipeline = { steps: [StepEntry], output }`, serialised as JSON (presets are plain files).
Steps: `removeBackground`, `trim`, `padding`, `resize`, `upscale`, `enhance`, `background`.
Each step has an id and an enabled flag; order is free (the UI warns about odd orders, e.g. upscaling
before background removal). Output: `same | png | jpeg | webp | avif` + quality, PNG level / palette,
WebP lossless, JPEG flatten colour.

Brush edits are stored per image as vector strokes in the coordinate space of the first
"Remove background" step's input; the mask editor shows that stage, so coordinates are exact
regardless of later trim / resize steps.

## Data locations

| What | Where |
|---|---|
| Application, CPU runtime, bundled models, licences | install dir (`%LOCALAPPDATA%\Programs\AlphaForge`) |
| Downloaded models, GPU pack, logs | `%LOCALAPPDATA%\AlphaForge` |
| Settings, user presets | `%APPDATA%\AlphaForge` |
| Default output | `<source folder>\AlphaForge\` (pasted images: `Pictures\AlphaForge`) |

## Model templates

`resources/models/<name>.{fp32,fp16}.onnx` are optimised graphs whose initializers point into a virtual
blob `<name>.<prec>.bin`. `<name>.manifest.json` lists, for every initializer, the source tensor in the
author's safetensors file, the target dtype and its offset. At load time the engine memory-maps the
safetensors file, builds the blob (with f32↔f16 conversion), and hands it to ONNX Runtime through
`AddExternalInitializersFromFilesInMemory`. CPU sessions use the fp32 graph, CUDA sessions the fp16 graph.

Rebuilding templates: see `BUILDING.md` (`tools/build_bg_template.py`, `tools/build_sr_templates.py`,
`tools/bundle_weights.py`).
