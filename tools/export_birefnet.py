"""Export official BiRefNet weights (MIT, ZhengPeng7/BiRefNet) to an optimized ONNX graph.

Differences to the upstream ONNX files:
  * deformable convolutions are exported as the native ONNX ``DeformConv`` op (opset 19+),
    which ONNX Runtime >= 1.25 implements on CPU and CUDA. Upstream emulates it with
    thousands of Gather/Scatter nodes that need several GB of scratch memory.
  * the final sigmoid is part of the graph (output = alpha in [0, 1]).
  * optional fp16 weights (``--fp16``), graph I/O stays float32.

Usage:
    python tools/export_birefnet.py <birefnet-src-dir> <weights.pth> <backbone> <out.onnx> [--fp16] [--size 1024]
    backbone: swin_v1_t | swin_v1_l
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile

import torch
from torch.onnx import symbolic_helper as sh


def prepare_source(src, backbone):
    """Copy upstream source to a temp dir and select the backbone in config.py."""
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
    # never try to load ImageNet backbone weights / touch dataset dirs
    cfg = cfg.replace("self.sys_home_dir = [os.path.expanduser('~'), '/workspace'][1]", "self.sys_home_dir = '.'")
    open(cfg_path, "w", encoding="utf-8").write(cfg)
    return tmp


@sh.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def deform_conv2d_symbolic(g, inp, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w,
                           dil_h, dil_w, groups, offset_groups, use_mask):
    kshape = sh._get_tensor_sizes(weight)[2:]
    args = [inp, weight, offset, bias] + ([mask] if use_mask else [])
    return g.op("DeformConv", *args, dilations_i=[dil_h, dil_w], group_i=groups, kernel_shape_i=kshape,
                offset_group_i=offset_groups, pads_i=[pad_h, pad_w, pad_h, pad_w], strides_i=[stride_h, stride_w])


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return torch.sigmoid(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("weights"); ap.add_argument("backbone"); ap.add_argument("out")
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=19)
    a = ap.parse_args()

    tmp = prepare_source(a.src, a.backbone)
    cwd = os.getcwd()
    os.chdir(tmp)
    sys.path.insert(0, tmp)
    try:
        from utils import check_state_dict
        from models.birefnet import BiRefNet
        net = BiRefNet(bb_pretrained=False)
        sd = torch.load(os.path.join(cwd, a.weights), map_location="cpu", weights_only=True)
        net.load_state_dict(check_state_dict(sd), strict=True)
    finally:
        os.chdir(cwd)
    model = Wrapper(net).eval()

    torch.onnx.register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic, a.opset)
    x = torch.randn(1, 3, a.size, a.size)
    with torch.no_grad():
        ref = model(x)
        torch.onnx.export(model, x, a.out, input_names=["input_image"], output_names=["alpha"],
                          opset_version=a.opset, dynamo=False, do_constant_folding=True)

    import onnx
    m = onnx.load(a.out)
    if a.fp16:
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=["DeformConv"] + float16.DEFAULT_OP_BLOCK_LIST)
    onnx.checker.check_model(m)
    onnx.save(m, a.out)

    import numpy as np
    import onnxruntime as ort
    s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    y = s.run(None, {"input_image": x.numpy()})[0]
    err = float(np.abs(y - ref.numpy()).max())
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"{a.out}: {os.path.getsize(a.out)/2**20:.1f} MB nodes={len(m.graph.node)} max|onnx-torch|={err:.5f} sha256={h}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
