"""Build an AlphaForge model *template* for a BiRefNet checkpoint.

A template is an optimized ONNX graph whose weights are NOT embedded. Every weight is an
external initializer that points into a virtual ``<name>.<prec>.bin`` blob, and a manifest
describes how to assemble that blob from the *official, unmodified* safetensors file
published by the model author (ZhengPeng7 on Hugging Face, MIT licence).

The app therefore never downloads third-party re-uploads: it fetches the author's file at a
pinned revision, verifies its SHA-256, and builds the weight blob in memory.

Graph improvements over upstream ONNX exports:
  * native ONNX ``DeformConv`` (ORT >= 1.25 CPU/CUDA kernels) instead of Gather/Scatter emulation
  * sigmoid fused into the graph, output = alpha in [0, 1]
  * fp32 graph for CPU, fp16 graph (DeformConv kept in fp32) for GPU

Usage:
  python tools/build_bg_template.py --src <BiRefNet git checkout> --weights <model.safetensors>
         --repo ZhengPeng7/BiRefNet_lite --revision <commit> --backbone swin_v1_t --name birefnet-lite
         --out <dir> [--size 1024] [--precisions fp32,fp16]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors import safe_open
from torch.onnx import symbolic_helper as sh

ALIGN = 64
DT = {torch.float32: "f32", torch.float16: "f16", torch.int64: "i64", torch.bfloat16: "bf16"}
ONNX_DT = {onnx.TensorProto.FLOAT: "f32", onnx.TensorProto.FLOAT16: "f16", onnx.TensorProto.INT64: "i64"}
NP_DT = {"f32": np.float32, "f16": np.float16, "i64": np.int64}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_source(src, backbone):
    tmp = tempfile.mkdtemp(prefix="birefnet_")
    for item in ("config.py", "utils.py", "dataset.py", "image_proc.py", "models"):
        s = os.path.join(src, item)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, os.path.join(tmp, item))
    cfg_path = os.path.join(tmp, "config.py")
    cfg = open(cfg_path, encoding="utf-8").read()
    idx = {"swin_v1_l": 3, "swin_v1_t": 6}[backbone]
    marker = "'dino_v3_b', 'dino_v3_s_plus', 'dino_v3_s',\n        ][3]"
    assert marker in cfg, "upstream config layout changed"
    cfg = cfg.replace(marker, marker[:-3] + f"[{idx}]")
    cfg = cfg.replace("self.sys_home_dir = [os.path.expanduser('~'), '/workspace'][1]", "self.sys_home_dir = '.'")
    open(cfg_path, "w", encoding="utf-8").write(cfg)
    # Keep the deformable conv in fp32 even in the fp16 graph (accuracy + kernel coverage).
    dc_path = os.path.join(tmp, "models", "modules", "deform_conv.py")
    dc = open(dc_path, encoding="utf-8").read()
    old = """        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x"""
    new = """        dt = x.dtype
        x = deform_conv2d(
            input=x.float(),
            offset=offset.float(),
            weight=self.regular_conv.weight.float(),
            bias=None if self.regular_conv.bias is None else self.regular_conv.bias.float(),
            padding=self.padding,
            mask=modulator.float(),
            stride=self.stride,
        )
        return x.to(dt)"""
    assert old in dc, "upstream deform_conv layout changed"
    open(dc_path, "w", encoding="utf-8").write(dc.replace(old, new))
    return tmp


@sh.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def deform_conv2d_symbolic(g, inp, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w,
                           dil_h, dil_w, groups, offset_groups, use_mask):
    kshape = sh._get_tensor_sizes(weight)[2:]
    args = [inp, weight, offset, bias] + ([mask] if use_mask else [])
    return g.op("DeformConv", *args, dilations_i=[dil_h, dil_w], group_i=groups, kernel_shape_i=kshape,
                offset_group_i=offset_groups, pads_i=[pad_h, pad_w, pad_h, pad_w], strides_i=[stride_h, stride_w])


class Wrapper(torch.nn.Module):
    """float32 NCHW (ImageNet-normalized) in -> float32 alpha (N,1,H,W) out."""

    def __init__(self, net, dtype):
        super().__init__()
        self.net = net
        self.dtype = dtype

    def forward(self, x):
        out = self.net(x.to(self.dtype))
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return torch.sigmoid(out.float())


def load_net(tmp, weights_path):
    cwd = os.getcwd()
    os.chdir(tmp)
    sys.path.insert(0, tmp)
    try:
        from models.birefnet import BiRefNet
        net = BiRefNet(bb_pretrained=False)
    finally:
        os.chdir(cwd)
    src = {}
    with safe_open(weights_path, framework="pt") as f:
        for k in f.keys():
            src[k] = f.get_tensor(k)
    missing, unexpected = net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in src.items()}, strict=False)
    missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    if missing:
        raise SystemExit(f"weights missing for: {missing[:10]}")
    return net.eval(), src


def externalize(model_path, out_graph, blob_name, src, prec):
    """Move every initializer that equals an official tensor into the external blob."""
    m = onnx.load(model_path)
    entries, offset, inline = [], 0, 0
    keep_inline = []
    for init in m.graph.initializer:
        name = init.name
        key = name[4:] if name.startswith("net.") else name
        if key not in src:
            keep_inline.append(name)
            inline += len(init.raw_data) if init.raw_data else 0
            continue
        arr = numpy_helper.to_array(init)
        tdt = ONNX_DT[init.data_type]
        s = src[key]
        ref = s.float().numpy() if s.is_floating_point() else s.numpy()
        conv = ref.astype(NP_DT[tdt])
        if conv.shape != arr.shape or not np.array_equal(conv, arr):
            raise SystemExit(f"initializer {name} does not match the official tensor")
        nbytes = arr.nbytes
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        entries.append({"name": name, "src": key, "src_dtype": DT[s.dtype], "dtype": tdt,
                        "shape": list(arr.shape), "offset": offset, "nbytes": nbytes})
        init.ClearField("raw_data")
        for f in ("float_data", "int64_data", "int32_data", "double_data"):
            init.ClearField(f)
        init.data_location = onnx.TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", blob_name), ("offset", str(offset)), ("length", str(nbytes))):
            e = init.external_data.add(); e.key = k; e.value = v
        offset += nbytes
    onnx.save(m, out_graph)
    return {"blob": blob_name, "blob_size": offset, "tensors": entries,
            "inline_initializers": len(keep_inline), "inline_bytes": inline}


def assemble(manifest_variant, src, out_path):
    """Reference implementation of the blob assembly done by the Rust app."""
    buf = bytearray(manifest_variant["blob_size"])
    for t in manifest_variant["tensors"]:
        s = src[t["src"]]
        a = (s.float().numpy() if s.is_floating_point() else s.numpy()).astype(NP_DT[t["dtype"]])
        b = a.tobytes()
        assert len(b) == t["nbytes"]
        buf[t["offset"]:t["offset"] + len(b)] = b
    with open(out_path, "wb") as f:
        f.write(buf)
    return hashlib.sha256(buf).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--precisions", default="fp32,fp16")
    ap.add_argument("--opset", type=int, default=19)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tmp = prepare_source(a.src, a.backbone)
    net, src = load_net(tmp, a.weights)
    torch.onnx.register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic, a.opset)

    x = torch.randn(1, 3, a.size, a.size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = Wrapper(net, torch.float32).eval()(x).numpy()

    manifest = {
        "format": 1, "name": a.name, "arch": "birefnet", "backbone": a.backbone,
        "input": {"name": "input_image", "size": [a.size, a.size], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": {"name": "alpha", "activation": "none"},
        "source": {"repo": a.repo, "revision": a.revision, "file": "model.safetensors",
                   "url": f"https://huggingface.co/{a.repo}/resolve/{a.revision}/model.safetensors",
                   "sha256": sha256_file(a.weights), "size": os.path.getsize(a.weights)},
        "variants": {},
    }
    import onnxruntime as ort
    for prec in a.precisions.split(","):
        dtype = torch.float16 if prec == "fp16" else torch.float32
        model = Wrapper(net.to(dtype), dtype).eval()
        raw = os.path.join(tmp, f"{a.name}.{prec}.raw.onnx")
        with torch.no_grad():
            torch.onnx.export(model, x, raw, input_names=["input_image"], output_names=["alpha"],
                              opset_version=a.opset, dynamo=False, do_constant_folding=False,
                              keep_initializers_as_inputs=False)
        net.to(torch.float32)
        graph = os.path.join(a.out, f"{a.name}.{prec}.onnx")
        blob = f"{a.name}.{prec}.bin"
        v = externalize(raw, graph, blob, src, prec)
        blob_path = os.path.join(a.out, blob)
        v["blob_sha256"] = assemble(v, src, blob_path)
        v["graph_sha256"] = sha256_file(graph)
        v["graph_size"] = os.path.getsize(graph)
        # validate with ONNX Runtime (CPU) against the PyTorch fp32 reference
        so = ort.SessionOptions()
        sess = ort.InferenceSession(graph, so, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"input_image": x.numpy()})[0]
        v["max_abs_err_vs_torch_fp32"] = float(np.abs(y - ref).max())
        v["mean_abs_err_vs_torch_fp32"] = float(np.abs(y - ref).mean())
        manifest["variants"][prec] = v
        print(f"{prec}: graph {v['graph_size']/2**20:.1f} MB, blob {v['blob_size']/2**20:.1f} MB, "
              f"tensors {len(v['tensors'])}, inline {v['inline_initializers']} ({v['inline_bytes']} B), "
              f"err max {v['max_abs_err_vs_torch_fp32']:.5f} mean {v['mean_abs_err_vs_torch_fp32']:.6f}")
        os.remove(raw)
    with open(os.path.join(a.out, f"{a.name}.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
