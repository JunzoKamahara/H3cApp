"""Compare a weight as H3cApp's store restores it (raw BF16 dump from
h3_image_dit_probe ... dump) with the reference reading of the same bytes:
W = (int8 * row scale), each 256-wide input group multiplied by H256.
usage: int8_compare.py <checkpoint.safetensors> <tensor name> <dump.bf16>"""
import json, struct, sys
import numpy as np
path, name, dump = sys.argv[1:4]
f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
def raw(key, dtype):
    a, b = h[key]["data_offsets"]; f.seek(base + a)
    return np.frombuffer(f.read(b - a), dtype).reshape(h[key]["shape"])
H4 = np.ones((4, 4), np.float32); np.fill_diagonal(H4[:, ::-1], -1)
H256 = np.kron(np.kron(H4, H4), np.kron(H4, H4)) / 16
spec = h[name]; rows, cols = spec["shape"]
if spec["dtype"] == "I8":
    w = raw(name, "i1").astype(np.float32) * raw(name + "_scale", "<f4").reshape(-1, 1)
    w = (w.reshape(rows, -1, 256) @ H256).reshape(rows, cols)
else:
    w = (raw(name, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
if name.endswith("qkv_proj.weight"):
    # ComfyUI keeps [all Q; all K; all V]; the engine loads per-head [q; k; v] (128 rows each).
    inner, head = rows // 3, 128
    w = w.reshape(3, inner // head, head, cols).transpose(1, 0, 2, 3).reshape(rows, cols)
got = (np.fromfile(dump, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
diff = got - w
rel = np.linalg.norm(diff) / np.linalg.norm(w)
bf = (w.view(np.uint32) + 0x8000 >> 16 << 16).view(np.float32)  # reference rounded to BF16
exact = float(np.mean(got == bf))
print("%-34s %s [%d,%d]  rel L2 %.3e, max abs %.3e (max |w| %.3f), equal to BF16-rounded reference: %.2f%%"
      % (name, spec["dtype"], rows, cols, rel, np.abs(diff).max(), np.abs(w).max(), 100 * exact))
