"""Build AlphaForge templates for the Real-ESRGAN upscalers (BSD-3-Clause).

For every model:
  * the official .pth (GitHub release, loaded with weights_only=True) is converted to an fp16
    .safetensors file that ships with the app (``bundled``),
  * an fp32 graph (CPU) and an fp16 graph (GPU) with dynamic H/W are exported, all weights external,
  * the graphs are validated with ONNX Runtime against PyTorch.

Usage: python tools/build_sr_templates.py <out_dir> <pth_cache_dir>
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bg_template import assemble, externalize, sha256_file  # noqa: E402
from export_realesrgan import WEIGHTS, build, fetch, load_state  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, net, dtype):
        super().__init__()
        self.net = net
        self.dtype = dtype

    def forward(self, x):
        return self.net(x.to(self.dtype)).float()


MODELS = [
    # name, arch name, scale, source weights (list of (name, factor)) for DNI blends
    ("realesr-general-x4v3", "realesr-general-x4v3", 4, [("realesr-general-x4v3", 1.0)]),
    ("realesr-general-wdn-x4v3", "realesr-general-x4v3", 4, [("realesr-general-wdn-x4v3", 1.0)]),
    ("realesr-general-x4v3-dn50", "realesr-general-x4v3", 4, [("realesr-general-x4v3", 0.5), ("realesr-general-wdn-x4v3", 0.5)]),
    ("RealESRGAN_x4plus", "RealESRGAN_x4plus", 4, [("RealESRGAN_x4plus", 1.0)]),
    ("RealESRGAN_x2plus", "RealESRGAN_x2plus", 2, [("RealESRGAN_x2plus", 1.0)]),
    ("RealESRGAN_x4plus_anime_6B", "RealESRGAN_x4plus_anime_6B", 4, [("RealESRGAN_x4plus_anime_6B", 1.0)]),
]


def main():
    out, cache = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    import onnxruntime as ort

    for name, arch, scale, blend in MODELS:
        states = [(load_state(fetch(n, cache)), f) for n, f in blend]
        sd = {k: sum(s[k].float() * f for s, f in states) for k in states[0][0]}
        sd16 = {k: v.half().contiguous() for k, v in sd.items()}
        bundled = f"{name}.fp16.safetensors"
        save_file(sd16, os.path.join(out, bundled))
        src = {k: v for k, v in sd16.items()}

        net = build(arch)
        net.load_state_dict({k: v.float() for k, v in sd16.items()}, strict=True)
        net.eval()
        x = torch.rand(1, 3, 48, 64, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            ref = Wrapper(net, torch.float32)(x).numpy()

        manifest = {
            "format": 1, "name": name, "arch": "realesrgan", "scale": scale,
            "input": {"name": "input", "size": None, "mean": [0, 0, 0], "std": [1, 1, 1]},
            "output": {"name": "output"},
            "source": {
                "url": "", "sha256": sha256_file(os.path.join(out, bundled)),
                "size": os.path.getsize(os.path.join(out, bundled)),
                "bundled": bundled, "bundled_sha256": sha256_file(os.path.join(out, bundled)),
                "provenance": [WEIGHTS[n] for n, _ in blend],
            },
            "variants": {},
        }
        for prec in ("fp32", "fp16"):
            dtype = torch.float16 if prec == "fp16" else torch.float32
            model = Wrapper(net.to(dtype), dtype).eval()
            raw = os.path.join(out, f"_{name}.{prec}.raw.onnx")
            with torch.no_grad():
                torch.onnx.export(model, x, raw, input_names=["input"], output_names=["output"], opset_version=17,
                                  dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "oh", 3: "ow"}},
                                  dynamo=False, do_constant_folding=False)
            net.to(torch.float32)
            graph = os.path.join(out, f"{name}.{prec}.onnx")
            blob = f"{name}.{prec}.bin"
            v = externalize(raw, graph, blob, src, prec)
            os.remove(raw)
            blob_path = os.path.join(out, blob)
            v["blob_sha256"] = assemble(v, src, blob_path)
            v["graph_sha256"] = sha256_file(graph)
            s = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
            y = s.run(None, {"input": x.numpy()})[0]
            v["max_abs_err_vs_torch"] = float(np.abs(y - ref).max())
            os.remove(blob_path)
            manifest["variants"][prec] = v
            print(f"{name} {prec}: graph {os.path.getsize(graph)/1024:.0f} KB, tensors {len(v['tensors'])}, "
                  f"inline {v['inline_initializers']}, err {v['max_abs_err_vs_torch']:.5f}", flush=True)
        with open(os.path.join(out, f"{name}.manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
