# Compare ComfyUI's final_layer.video_out [576,5376] with the released six
# PDD heads [6,96,5376]: which combination of the official heads each
# 96-row chunk holds.
import array, json, struct, sys, math
def st(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n)); return f, h, 8 + n
def f32(f, h, base, k):
    a, b = h[k]["data_offsets"]; f.seek(base + a)
    v = array.array("f"); v.frombytes(f.read(b - a)); return v
fo, ho, bo = st(sys.argv[1]); fc, hc, bc = st(sys.argv[2])
print({k: (v["dtype"], v["shape"]) for k, v in ho.items() if k != "__metadata__"}, ho.get("__metadata__"))
for part, rows in (("weight", 96 * 5376), ("bias", 96)):
    off = f32(fo, ho, bo, "video_out." + part); com = f32(fc, hc, bc, "final_layer.video_out." + part)
    assert len(off) == len(com) == 6 * rows
    for k in range(6):
        o = off[k * rows:(k + 1) * rows]; c = com[k * rows:(k + 1) * rows]; o0 = off[:rows]
        cand = {"head[k]": o, "head[k]-head[0]": [x - y for x, y in zip(o, o0)],
                "head[k]-head[k-1]": [x - y for x, y in zip(o, off[(k - 1) * rows:k * rows])] if k else o}
        norm = math.sqrt(sum(x * x for x in c))
        res = {name: math.sqrt(sum((x - y) ** 2 for x, y in zip(c, v))) / (norm or 1) for name, v in cand.items()}
        print(part, "chunk", k, "L2 %.4g" % norm, " rel err:", ", ".join("%s %.3g" % kv for kv in res.items()))
