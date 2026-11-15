"""Compare two 8-bit RGB images of the same size (PNG, PPM, ...).
usage: image_compare.py <reference> <other> [<reference> <other> ...]
Prints the PSNR, the largest difference of a value, and how many values are
equal and within 1 of each other."""
import sys
import numpy as np
from PIL import Image

for reference, other in zip(sys.argv[1::2], sys.argv[2::2]):
    a = np.asarray(Image.open(reference).convert("RGB"), dtype=np.int32)
    b = np.asarray(Image.open(other).convert("RGB"), dtype=np.int32)
    if a.shape != b.shape:
        print(other, "size differs:", a.shape, b.shape)
        continue
    diff = np.abs(a - b)
    mse = float((diff.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    worst = np.unravel_index(diff.argmax(), diff.shape)
    print("%s: %dx%d, PSNR %.2f dB, max difference %d (at x=%d y=%d), equal %.2f%%, within 1 %.2f%%" % (
        other.rsplit("/", 1)[-1], a.shape[1], a.shape[0], psnr, diff.max(), worst[1], worst[0],
        100 * (diff == 0).mean(), 100 * (diff <= 1).mean()))
