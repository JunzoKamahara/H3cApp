# Fetch named tensors of a remote safetensors file by byte range and save
# them as .npy (BF16 and F16 widened to F32).
#   python3 fetch_tensor.py URL HEADER.json OUTDIR NAME [NAME ...]
# HEADER.json is the file's JSON header (without the 8-byte length prefix).
import json, os, sys, urllib.request
import numpy as np

url, header_path, out_dir = sys.argv[1:4]
names = sys.argv[4:]
raw = open(header_path, "rb").read()
header = json.loads(raw)
base = 8 + len(raw)
os.makedirs(out_dir, exist_ok=True)

def get(a, b):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b - 1)})
    with urllib.request.urlopen(req) as r:
        return r.read()

for name in names:
    t = header[name]
    a, b = t["data_offsets"]
    data = get(base + a, base + b)
    dt = t["dtype"]
    if dt == "BF16":
        v = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
        arr = v.view(np.float32)
    elif dt == "F16":
        arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
    elif dt == "F32":
        arr = np.frombuffer(data, dtype=np.float32)
    elif dt == "I8":
        arr = np.frombuffer(data, dtype=np.int8)
    elif dt == "U8":
        arr = np.frombuffer(data, dtype=np.uint8)
    else:
        raise SystemExit("unsupported dtype " + dt)
    arr = arr.reshape(t["shape"]) if t["shape"] else arr.reshape(())
    np.save(os.path.join(out_dir, name + ".npy"), arr)
    print(name, dt, t["shape"], len(data))
