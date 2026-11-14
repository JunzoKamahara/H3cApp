"""Compare two sets of runs of the same prompt, seed and sizes that differ in
the text embedding they were driven by.
usage: run_compare.py <phase0 dir> <prompt name> <reference source> <other source>
For each size present in both: relative L2 error of the refined text, of the
latent rows after each step and of the final latent, and the PSNR of the
decoded image, all against the reference source."""
import sys
from pathlib import Path
import mlx.core as mx
import numpy as np
from PIL import Image

root, name, ref_source, other_source = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
f64 = lambda a: np.array(a.astype(mx.float32)).astype(np.float64)
rel = lambda a, b: np.sqrt(((b - a) ** 2).sum() / (a ** 2).sum())
print("| prompt | size | refined text | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | final latent | image PSNR dB |")
print("| - | - | - | - | - | - | - | - | - | - | - |")
for ref in sorted((root / name / ref_source).iterdir(), key=lambda p: (len(p.name), p.name)):
    other = root / name / other_source / ref.name
    if not (ref / "latent.safetensors").exists() or not (other / "latent.safetensors").exists():
        continue
    a, b = mx.load(str(ref / "trace.safetensors")), mx.load(str(other / "trace.safetensors"))
    assert np.array_equal(np.array(a["noise"]), np.array(b["noise"])), "initial noise differs"
    steps = [rel(f64(a["patches_after_step_%d" % s]), f64(b["patches_after_step_%d" % s])) for s in range(1, 7)]
    latent = rel(f64(mx.load(str(ref / "latent.safetensors"))["latent"]), f64(mx.load(str(other / "latent.safetensors"))["latent"]))
    image = lambda p: np.asarray(Image.open(p / "image.png").convert("RGB"), dtype=np.float64)
    mse = ((image(ref) - image(other)) ** 2).mean()
    print("| %s | %s | %.4f | %s | %.4f | %.1f |" % (
        name, ref.name, rel(f64(a["refined_text"]), f64(b["refined_text"])),
        " | ".join("%.4f" % s for s in steps), latent, 10 * np.log10(255 ** 2 / mse) if mse else float("inf")))
