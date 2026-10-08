"""Create the fp16 weights file that ships with the installer for a template and record it
in the manifest (``source.bundled``). Integer tensors are kept as-is.

Usage: python tools/bundle_weights.py <templates_dir> <name> <official.safetensors>
"""
import hashlib
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    tdir, name, src_path = sys.argv[1], sys.argv[2], sys.argv[3]
    mpath = os.path.join(tdir, f"{name}.manifest.json")
    m = json.load(open(mpath, encoding="utf-8"))
    assert sha256_file(src_path) == m["source"]["sha256"], "source file does not match the manifest"
    keys = sorted({t["src"] for v in m["variants"].values() for t in v["tensors"]})
    out = {}
    with safe_open(src_path, framework="pt") as f:
        for k in keys:
            t = f.get_tensor(k)
            out[k] = t.half().contiguous() if t.is_floating_point() else t.contiguous()
    bundled = f"{name}.fp16.safetensors"
    save_file(out, os.path.join(tdir, bundled))
    m["source"]["bundled"] = bundled
    m["source"]["bundled_sha256"] = sha256_file(os.path.join(tdir, bundled))
    json.dump(m, open(mpath, "w", encoding="utf-8"), indent=1)
    print(bundled, os.path.getsize(os.path.join(tdir, bundled)) / 2**20, "MB")


if __name__ == "__main__":
    main()
