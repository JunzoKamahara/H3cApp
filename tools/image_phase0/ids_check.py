import json, struct, sys
up = {l.split()[0]: [int(v) for v in l.split(":")[1].split()] for l in open(sys.argv[1])}
for name, ids in up.items():
    f = open("%s/%s/embedding.h3c.safetensors" % (sys.argv[2], name), "rb")
    n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    a, b = h["ids"]["data_offsets"]; f.seek(8 + n + a)
    mine = list(struct.unpack("<%di" % ((b - a) // 4), f.read(b - a)))
    print(name, "upstream", len(ids), "engine", len(mine), "identical" if ids == mine else "DIFFERENT", h["embedding"]["shape"])
