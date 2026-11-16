# Fits the linear map the app draws an image's in-progress preview with
# (native/H3Spike/Sources/H3cApp/ImageDenoisePreview.swift): each latent
# pixel's 3x3 neighbourhood (24 channels each, edges repeated, plus a 1)
# to the 16x16 RGB patch the VAE decodes it to. Ridge least squares over
# pictures the engine generated, with the latents it saved beside them
# (h3_image_cli writes <name>.ppm and <name>.ppm.latent.f32, F32 [24, h, w]).
#
#   python -I fit_preview.py <out.safetensors> <model name> <picture.ppm>...
#
# Writes "weight" F16 [217, 768]: row (dy * 3 + dx) * 24 + c, last row the
# bias; column (py * 16 + px) * 3 + rgb, values in 0...1.
import json, struct, sys
import numpy as np
from PIL import Image

K, PATCH, CHANNELS = 3, 16, 24

def load(path):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = image.shape[0] // PATCH, image.shape[1] // PATCH
    latent = np.fromfile(path + ".latent.f32", dtype=np.float32).reshape(CHANNELS, h, w)
    return image, latent

def features(latent):
    c, h, w = latent.shape
    r = K // 2
    padded = np.pad(latent, ((0, 0), (r, r), (r, r)), mode="edge")
    columns = [padded[:, dy:dy + h, dx:dx + w] for dy in range(K) for dx in range(K)]
    x = np.concatenate(columns, 0).reshape(c * K * K, h * w).T
    return np.concatenate([x, np.ones((h * w, 1), np.float32)], 1)

def targets(image):
    h, w = image.shape[0] // PATCH, image.shape[1] // PATCH
    patches = image[:h * PATCH, :w * PATCH].reshape(h, PATCH, w, PATCH, 3)
    return patches.transpose(0, 2, 1, 3, 4).reshape(h * w, PATCH * PATCH * 3)

out, name, pictures = sys.argv[1], sys.argv[2], sys.argv[3:]
xtx = np.zeros((CHANNELS * K * K + 1,) * 2, np.float64)
xty = np.zeros((CHANNELS * K * K + 1, PATCH * PATCH * 3), np.float64)
rows = 0
for path in pictures:
    image, latent = load(path)
    x, y = features(latent).astype(np.float64), targets(image).astype(np.float64)
    xtx += x.T @ x
    xty += x.T @ y
    rows += len(x)
ridge = 1e-2 * rows * np.eye(len(xtx))
ridge[-1, -1] = 0
weight = np.linalg.solve(xtx + ridge, xty)
for path in pictures:
    image, latent = load(path)
    error = np.mean((np.clip(features(latent) @ weight, 0, 1) - targets(image)) ** 2)
    print("%s: PSNR %.2f dB" % (path, 10 * np.log10(1 / error)))
data = weight.astype(np.float16).tobytes()
header = {"__metadata__": {"model": name, "neighbourhood": str(K), "patch": str(PATCH),
                           "pictures": str(len(pictures))},
          "weight": {"dtype": "F16", "shape": list(weight.shape), "data_offsets": [0, len(data)]}}
encoded = json.dumps(header, separators=(",", ":")).encode()
encoded += b" " * (-len(encoded) % 8)
with open(out, "wb") as f:
    f.write(struct.pack("<Q", len(encoded)) + encoded + data)
print("wrote", out, weight.shape)
