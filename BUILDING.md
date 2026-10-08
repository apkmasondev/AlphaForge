# Building AlphaForge

## Requirements

* Windows 10/11 x64
* Visual Studio 2022 Build Tools (MSVC, Windows SDK)
* Rust — the toolchain is pinned in `rust-toolchain.toml` (rustup installs it automatically)
* Node.js 20+ and npm
* NASM 2.15+ on `PATH` (used by the AV1 codecs) — https://www.nasm.us (a portable copy can live in `.tools/nasm-*/`)
* Git LFS — the bundled model weights are stored with LFS (`git lfs install` before cloning)

## Build

```powershell
git clone https://github.com/apkmasondev/AlphaForge.git
cd AlphaForge
powershell -ExecutionPolicy Bypass -File tools\fetch_runtime.ps1   # ONNX Runtime + VC++ runtime DLLs
$env:PATH = "$PWD\.tools\nasm-3.01;$env:PATH"
cd app
npm ci
npm run tauri build          # → target\release\bundle\nsis\AlphaForge_<version>_x64-setup.exe
```

Development with hot reload: `npm run tauri dev`.

Core tests: `cargo test -p af-core` (add `-- --include-ignored` for the network tests).
Headless harness: `target\release\af-cli.exe run --preset web-asset --out out <files>` (see
`crates/af-cli/src/main.rs`).

## Bundled resources (`app/src-tauri/resources`)

| Folder | Content | How to regenerate |
|---|---|---|
| `onnxruntime/` | `onnxruntime.dll`, `onnxruntime_providers_shared.dll` from `onnxruntime-win-x64-1.28.3.zip` (GitHub release) | download & copy |
| `vcruntime/` | `msvcp140*.dll`, `vcruntime140*.dll` from `VC\Redist\MSVC\<ver>\x64\Microsoft.VC143.CRT` | copy |
| `models/` | graph templates, manifests, bundled fp16 weights | see below |
| `licenses/` | third-party licences | `rust-crates.txt` from `cargo metadata` |

### Model templates

Python 3.11 with `torch` (CPU), `torchvision`, `timm`, `einops`, `kornia`, `onnx`, `onnxruntime`,
`safetensors`, plus a checkout of https://github.com/ZhengPeng7/BiRefNet:

```bash
# BiRefNet (repeat for BiRefNet / BiRefNet-matting with --backbone swin_v1_l)
python tools/build_bg_template.py --src BiRefNet --weights BiRefNet_lite/model.safetensors \
  --repo ZhengPeng7/BiRefNet_lite --revision aa62cd87eafb9cc43056d08ef3615a14628b831d \
  --backbone swin_v1_t --name birefnet-lite --out templates
python tools/bundle_weights.py templates birefnet-lite BiRefNet_lite/model.safetensors

# Real-ESRGAN (downloads the official .pth files, loads them with weights_only=True)
python tools/build_sr_templates.py templates pth_cache

# Depth Anything V2 Small (needs transformers==4.46.x for the checkpoint's tensor names)
python tools/build_depth_template.py templates
```

Copy `*.onnx`, `*.manifest.json` and the bundled `*.safetensors` (not the `*.bin` blobs) into
`resources/models`. `cargo test -p af-core --test templates` verifies that the Rust assembler reproduces
the reference blobs byte for byte.

### GPU pack table

`tools/gen_gpupack.py <archives-dir>` regenerates `crates/af-core/src/ai/gpupack_data.rs` (URLs, entry
names, sizes, SHA-256) from the official archives.
