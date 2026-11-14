"""Is the image VAE's encoder the video VAE's encoder? Compares every tensor
the two files share by name (image file is F16, video file F32 -> rounded to F16).
usage: vae_encoder_compare.py <image vae.safetensors> <video vae dir or file>"""
import json, struct, sys, re, collections
from pathlib import Path
import numpy as np
DT = {"F16": "<f2", "F32": "<f4", "BF16": "<u2"}
def open_st(p):
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); h.pop("__metadata__", None)
    return f, h, 8 + n
def read(f, h, base, k):
    a, b = h[k]["data_offsets"]; f.seek(base + a)
    return np.frombuffer(f.read(b - a), DT[h[k]["dtype"]]).reshape(h[k]["shape"])
fi, hi, bi = open_st(sys.argv[1])
video = Path(sys.argv[2]); files = sorted(video.glob("*.safetensors")) if video.is_dir() else [video]
hv = {}
for p in files:
    f, h, b = open_st(p)
    for k in h: hv[k] = (f, h, b)
def group(k): return k.split(".")[0]
gi = collections.Counter(group(k) for k in hi); gv = collections.Counter(group(k) for k in hv)
print("image file groups:", dict(gi)); print("video file groups:", dict(gv))
print("image dtypes:", dict(collections.Counter(v["dtype"] for v in hi.values())), "video dtypes:", dict(collections.Counter(h[k]["dtype"] for k, (f, h, b) in hv.items())))
stats = collections.defaultdict(lambda: [0, 0, 0, 0])  # shared, bit-equal after rounding, shape mismatch, elements
only_image = collections.Counter(); only_video = collections.Counter(group(k) for k in hv if k not in hi)
for k in hi:
    if k not in hv: only_image[group(k)] += 1; continue
    f, h, b = hv[k]; s = stats[group(k)]; s[0] += 1
    if h[k]["shape"] != hi[k]["shape"]: s[2] += 1; continue
    a = read(fi, hi, bi, k); v = read(f, h, b, k)
    if a.dtype != v.dtype: v = v.astype(a.dtype)
    s[3] += a.size
    if np.array_equal(a.view(np.uint16) if a.dtype == np.float16 else a, v.view(np.uint16) if v.dtype == np.float16 else v): s[1] += 1
for g, (shared, equal, shape, elements) in stats.items():
    print("%-16s shared %4d, bit-equal (video rounded to the image dtype) %4d, shape differs %4d, %d elements" % (g, shared, equal, shape, elements))
print("only in image:", dict(only_image)); print("only in video:", dict(only_video))
