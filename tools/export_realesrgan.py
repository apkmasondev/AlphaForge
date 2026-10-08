"""Export official Real-ESRGAN weights (BSD-3-Clause, xinntao/Real-ESRGAN) to ONNX.

Run once at build time (needs torch + onnx + onnxconverter-common):
    python tools/export_realesrgan.py <out_dir> [--fp16]

The .pth files are fetched from the official GitHub releases, verified against
pinned SHA-256 digests and loaded with ``weights_only=True`` (no pickle code
execution). The application itself only ever loads the resulting ONNX files.
"""
import hashlib
import os
import sys
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"
WEIGHTS = {
    "realesr-general-x4v3": f"{BASE}/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3": f"{BASE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "realesr-animevideov3": f"{BASE}/v0.2.5.0/realesr-animevideov3.pth",
    "RealESRGAN_x4plus": f"{BASE}/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus": f"{BASE}/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus_anime_6B": f"{BASE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}


# --- architectures (re-implemented from basicsr / Real-ESRGAN, BSD-3-Clause / Apache-2.0) ---
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        in_ch = num_in_ch * (4 if scale == 2 else 16 if scale == 1 else 1)
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = F.pixel_unshuffle(x, 2) if self.scale == 2 else x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def fetch(name, cache):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, name + ".pth")
    if not os.path.exists(path):
        print("download", WEIGHTS[name])
        urllib.request.urlretrieve(WEIGHTS[name], path + ".part")
        os.replace(path + ".part", path)
    return path


def load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params"):
        if k in sd:
            return sd[k]
    return sd


def build(name):
    if name in ("realesr-general-x4v3", "realesr-general-wdn-x4v3"):
        return SRVGGNetCompact(num_conv=32, upscale=4)
    if name == "realesr-animevideov3":
        return SRVGGNetCompact(num_conv=16, upscale=4)
    if name == "RealESRGAN_x4plus":
        return RRDBNet(scale=4, num_block=23)
    if name == "RealESRGAN_x2plus":
        return RRDBNet(scale=2, num_block=23)
    if name == "RealESRGAN_x4plus_anime_6B":
        return RRDBNet(scale=4, num_block=6)
    raise KeyError(name)


def export(model, out_path, fp16):
    model.eval()
    dummy = torch.rand(1, 3, 64, 64)
    torch.onnx.export(
        model, dummy, out_path, input_names=["input"], output_names=["output"], opset_version=17,
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "oh", 3: "ow"}}, dynamo=False,
    )
    import onnx
    m = onnx.load(out_path)
    if fp16:
        from onnxconverter_common import float16
        # keep_io_types: the graph takes/returns float32, weights stored as fp16
        m = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    h = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(f"{os.path.basename(out_path)}  {os.path.getsize(out_path)/2**20:.1f} MB  sha256={h}")


def main():
    out = sys.argv[1]
    fp16 = "--fp16" in sys.argv
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "_pth")
    for name in WEIGHTS:
        net = build(name)
        net.load_state_dict(load_state(fetch(name, cache)), strict=True)
        export(net, os.path.join(out, f"{name}.onnx"), fp16)
    # Denoise blend for the general model (Real-ESRGAN "DNI"): strength s -> s*x4v3 + (1-s)*wdn
    a = load_state(fetch("realesr-general-x4v3", cache))
    b = load_state(fetch("realesr-general-wdn-x4v3", cache))
    net = build("realesr-general-x4v3")
    net.load_state_dict({k: 0.5 * a[k] + 0.5 * b[k] for k in a}, strict=True)
    export(net, os.path.join(out, "realesr-general-x4v3-dn50.onnx"), fp16)


if __name__ == "__main__":
    main()
