# Plan which byte ranges of a safetensors file to fetch: every tensor except
# the large block / refiner matrices. Prints merged ranges (data-relative).
import json, sys
hdr = json.load(open(sys.argv[1])); hdr.pop("__metadata__", None)
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 16 << 20
items = sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
end = max(v["data_offsets"][1] for _, v in items)
keep, skipped = [], 0
for k, v in items:
    a, b = v["data_offsets"]
    big = (b - a) > limit and (k.startswith("blocks.") or k.startswith("token_refiner."))
    if big: skipped += b - a
    else: keep.append((a, b, k))
runs = []
for a, b, k in keep:
    if runs and a <= runs[-1][1]: runs[-1][1] = max(runs[-1][1], b)
    else: runs.append([a, b])
total = sum(b - a for a, b in runs)
print("data_bytes", end, "skipped", skipped, "fetch", total, "runs", len(runs), file=sys.stderr)
for k, v in items[:6]: print("  first", k, v["data_offsets"], file=sys.stderr)
big_kept = [(b - a, k) for a, b, k in keep if b - a > (4 << 20)]
print("  kept >4MB:", sorted(big_kept, reverse=True)[:8], file=sys.stderr)
for a, b in runs: print(a, b)
