"""Build the AlphaForge template for Depth Anything V2 Small (Apache-2.0).

The official weights (``model.safetensors`` of depth-anything/Depth-Anything-V2-Small-hf, pinned
revision) are downloaded by the app on first use. This script exports fixed-size (518 x 518) fp32
(CPU) and fp16 (GPU) graphs whose weights all live in the external blob assembled from that file,
and checks both graphs with ONNX Runtime against PyTorch.

Only the *Small* checkpoint is Apache-2.0; Base/Large/Giant are CC-BY-NC-4.0 and must not be used.

Usage: python tools/build_depth_template.py <out_dir>
"""
import json
import os
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bg_template import assemble, externalize, sha256_file  # noqa: E402

REPO = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
NAME = "depth-anything-v2-small"
SIZE = 518  # multiple of the ViT patch size (14)


class Wrapper(torch.nn.Module):
    """float32 NCHW (ImageNet-normalized) in -> float32 relative inverse depth (N,1,H,W) out."""

    def __init__(self, net, dtype):
        super().__init__()
        self.net = net
        self.dtype = dtype

    def forward(self, x):
        return self.net(pixel_values=x.to(self.dtype)).predicted_depth.float().unsqueeze(1)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
    import onnxruntime as ort

    cfg_path = hf_hub_download(REPO, "config.json", revision=REVISION)
    weights = hf_hub_download(REPO, "model.safetensors", revision=REVISION)
    src = load_file(weights)
    net = DepthAnythingForDepthEstimation(DepthAnythingConfig.from_json_file(cfg_path))
    missing, unexpected = net.load_state_dict(src, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    net.eval()

    x = torch.randn(1, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = Wrapper(net, torch.float32)(x).numpy()

    manifest = {
        "format": 1, "name": NAME, "arch": "depth-anything-v2",
        "input": {"name": "input", "size": [SIZE, SIZE], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": {"name": "output"},
        "source": {
            "url": f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors",
            "sha256": sha256_file(weights), "size": os.path.getsize(weights),
            "provenance": {"repo": REPO, "revision": REVISION, "license": "Apache-2.0"},
        },
        "variants": {},
    }
    for prec in ("fp32", "fp16"):
        dtype = torch.float16 if prec == "fp16" else torch.float32
        model = Wrapper(net.to(dtype), dtype).eval()
        raw = os.path.join(out, f"_{NAME}.{prec}.raw.onnx")
        with torch.no_grad():
            torch.onnx.export(model, x, raw, input_names=["input"], output_names=["output"], opset_version=17,
                              dynamo=False, do_constant_folding=False)
        net.to(torch.float32)
        graph = os.path.join(out, f"{NAME}.{prec}.onnx")
        blob = f"{NAME}.{prec}.bin"
        v = externalize(raw, graph, blob, src, prec)
        os.remove(raw)
        blob_path = os.path.join(out, blob)
        v["blob_sha256"] = assemble(v, src, blob_path)
        v["graph_sha256"] = sha256_file(graph)
        s = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
        y = s.run(None, {"input": x.numpy()})[0]
        v["max_abs_err_vs_torch"] = float(np.abs(y - ref).max())
        v["ref_range"] = [float(ref.min()), float(ref.max())]
        os.remove(blob_path)
        manifest["variants"][prec] = v
        print(f"{NAME} {prec}: graph {os.path.getsize(graph) / 1024:.0f} KB, tensors {len(v['tensors'])}, "
              f"inline {v['inline_initializers']} ({v['inline_bytes'] / 1024:.0f} KB), err {v['max_abs_err_vs_torch']:.5f} "
              f"(depth range {ref.min():.2f}..{ref.max():.2f})", flush=True)
    with open(os.path.join(out, f"{NAME}.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
