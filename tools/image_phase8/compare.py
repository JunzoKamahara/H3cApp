# Compare one tensor of two files: relative L2 error, cosine and per-row
# figures. Files are safetensors (any dtype) or .npy.
#   python -I compare.py <reference> <other> [name] [other name]
import sys
import numpy as np
import torch
from safetensors.torch import load_file

def load(path, name):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    tensors = load_file(path)
    return tensors[name].to(torch.float64).numpy()

name = sys.argv[3] if len(sys.argv) > 3 else "embedding"
other_name = sys.argv[4] if len(sys.argv) > 4 else name
ref, other = load(sys.argv[1], name), load(sys.argv[2], other_name)
print("reference", ref.shape, "| other", other.shape)
if ref.shape != other.shape:
    sys.exit("shapes differ")
diff = other - ref
norm = lambda a: np.sqrt((a ** 2).sum(axis=-1))
print("relative L2 error: %.5f   cosine: %.6f   max abs difference %.4f" % (
    np.linalg.norm(diff) / np.linalg.norm(ref),
    (ref * other).sum() / (np.linalg.norm(ref) * np.linalg.norm(other)),
    np.abs(diff).max()))
if ref.ndim == 2:
    rows = norm(diff) / np.maximum(norm(ref), 1e-30)
    cos = (ref * other).sum(-1) / np.maximum(norm(ref) * norm(other), 1e-30)
    print("per row: relative L2 max %.5f (row %d), median %.5f; cosine min %.6f" % (
        rows.max(), rows.argmax(), np.median(rows), cos.min()))
if ref.ndim == 3 and ref.shape[-1] == 3:
    mse = (diff ** 2).mean()
    print("PSNR %.2f dB (values in [0, 1])" % (10 * np.log10(1.0 / max(mse, 1e-20))))
