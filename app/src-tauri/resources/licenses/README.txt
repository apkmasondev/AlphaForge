AlphaForge — open-source components
====================================

AlphaForge itself: MIT License.

AI models (weights are used unmodified except for precision conversion / ONNX graph export)
- BiRefNet (lite, general, matting) — Copyright (c) 2024 ZhengPeng — MIT License
  https://github.com/ZhengPeng7/BiRefNet  · BiRefNet-LICENSE.txt
  Peng Zheng et al., "Bilateral Reference for High-Resolution Dichotomous Image Segmentation", CAAI AIR 2024.
- Real-ESRGAN (general x4v3, wdn x4v3, x4plus, x2plus, x4plus anime 6B) — Copyright (c) 2021 Xintao Wang — BSD 3-Clause
  https://github.com/xinntao/Real-ESRGAN  · Real-ESRGAN-LICENSE.txt

Runtime
- ONNX Runtime 1.28.3 — Copyright (c) Microsoft Corporation — MIT License
  onnxruntime-LICENSE.txt, onnxruntime-ThirdPartyNotices.txt
- Microsoft Visual C++ runtime (msvcp140*.dll, vcruntime140*.dll) — redistributed under the
  Microsoft Visual Studio redistribution terms.
- Microsoft Edge WebView2 (system component, not bundled).

Optional GPU acceleration pack (downloaded on request from the official sources, not bundled)
- ONNX Runtime GPU (CUDA 12) — MIT License (Microsoft, GitHub release v1.28.3)
- NVIDIA CUDA runtime, cuBLAS, NVRTC 12.9 and cuDNN 9.27 — NVIDIA Software License Agreement /
  CUDA EULA, obtained from NVIDIA's packages on PyPI (nvidia-*-cu12).

Rust crates compiled into AlphaForge: see rust-crates.txt (all permissive: MIT, Apache-2.0, BSD,
ISC, Zlib, Unicode, MPL-2.0 for a few unmodified CSS parsing crates used by Tauri).

Web UI libraries: React, React DOM (MIT), Zustand (MIT), Lucide icons (ISC), @tauri-apps/api (MIT/Apache-2.0).
