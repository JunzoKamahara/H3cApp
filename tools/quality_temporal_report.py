#!/usr/bin/env python3
"""Collects set-v2/*/metrics-*.json (tools/quality_temporal.py) into Markdown
tables and composes crop sheets around a window boundary. Development only.

    python tools/quality_temporal_report.py SET_DIR [--variants a,b,c] [--sheets]

With --sheets, for every clip with crops.txt ("X,Y,W,H" lines, the first is
the face/fur crop) and frames under out/<variant>/ (PNG) or out/<variant>.raw,
writes SET_DIR/<clip>/sheet-<frame>.png: one row per variant, the crop at
frames f-1, f, f+1 around the first boundary pair of the 17-frame window
(frame 16/17), plus the ground truth row.
"""
import glob, json, os, re, sys
import numpy as np
from PIL import Image, ImageDraw

COLS = [("warp", "warp", 3), ("warp_boundary", "warp bnd", 3), ("warp_interior", "warp int", 3),
        ("face_warp", "face warp", 3), ("flicker", "flicker", 2), ("psnr", "PSNR", 2), ("lpips", "LPIPS↓", 4),
        ("musiq", "MUSIQ", 1), ("maniqa", "MANIQA", 4), ("face_lpips", "face LPIPS↓", 4),
        ("face_musiq", "face MUSIQ", 1), ("face_maniqa", "face MANIQA", 4),
        ("seconds_per_frame", "s/frame", 2), ("peak_gib", "peak GiB", 1), ("footprint_gb", "footprint GB", 1)]


def fmt(r, key, digits):
    v = r.get(key)
    if v is None:
        return "—"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def read_info(d):
    t = open(os.path.join(d, "info.txt")).read()
    g = re.search(r"gt (\d+)x(\d+)", t)
    return int(g.group(1)), int(g.group(2))


def read_frame(path, w, h, t):
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.lower().endswith(".png"))
        return np.asarray(Image.open(os.path.join(path, names[t])).convert("RGB"))
    a = np.fromfile(path, dtype=np.uint8, count=w * h * 3, offset=t * w * h * 3)
    return a.reshape(h, w, 3)


def sheet(root, clip, rows, frame):
    d = os.path.join(root, clip)
    if not os.path.exists(os.path.join(d, "crops.txt")):
        return
    words = open(os.path.join(d, "crops.txt")).read().split()
    x, y, cw, ch = [int(v) for v in words[0].split(",")]
    if "frame" in words:  # "frame N": the sheet is centred on N instead
        frame = int(words[words.index("frame") + 1])
    w, h = read_info(d)
    frames = [frame - 1, frame, frame + 1]
    sources = [("gt", os.path.join(d, "gt.raw"))]
    for r in rows:
        v = r["variant"]
        p = os.path.join(d, "out", v)
        if not os.path.isdir(p):
            p = p + ".raw"
        if os.path.exists(p):
            sources.append((v, p))
    scale = max(1, 256 // max(cw, ch))
    tw, th = cw * scale, ch * scale
    label = 24
    im = Image.new("RGB", (len(frames) * tw + 120, len(sources) * (th + label)), "black")
    dr = ImageDraw.Draw(im)
    for j, (name, p) in enumerate(sources):
        dr.text((4, j * (th + label) + th // 2), name, fill="white")
        for i, t in enumerate(frames):
            fr = read_frame(p, w, h, t)[y:y + ch, x:x + cw]
            crop = Image.fromarray(fr).resize((tw, th), Image.NEAREST)
            im.paste(crop, (120 + i * tw, j * (th + label)))
            dr.text((120 + i * tw + 4, j * (th + label) + th + 4), f"frame {t}", fill="white")
    out = os.path.join(d, f"sheet-{frame}.png")
    im.save(out)
    print(f"wrote {out}", file=sys.stderr)


def main():
    root = sys.argv[1]
    variants = None
    sheets = "--sheets" in sys.argv
    if "--variants" in sys.argv:
        variants = sys.argv[sys.argv.index("--variants") + 1].split(",")
    clips = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for clip in clips:
        files = sorted(glob.glob(os.path.join(root, clip, "metrics-*.json")))
        rows = [json.load(open(f)) for f in files]
        if variants:
            rows = sorted((r for r in rows if r["variant"] in variants), key=lambda r: variants.index(r["variant"]))
        if not rows:
            continue
        info = open(os.path.join(root, clip, "info.txt")).read().strip()
        print(f"\n### {clip} ({info.split(' src')[0]})\n")
        print("| variant | " + " | ".join(c[1] for c in COLS) + " |")
        print("|---|" + "---:|" * len(COLS))
        gt = rows[0]
        print("| ground truth | " + " | ".join(fmt(gt, "gt_" + c[0], c[2]) if ("gt_" + c[0]) in gt else "—" for c in COLS) + " |")
        for r in rows:
            print(f"| {r['variant']} | " + " | ".join(fmt(r, c[0], c[2]) for c in COLS) + " |")
        if "warp_cut" in rows[0]:
            print("\n| variant | warp at cut | warp near cut (±2) | gt near cut |")
            print("|---|---:|---:|---:|")
            for r in rows:
                print(f"| {r['variant']} | {fmt(r, 'warp_cut', 2)} | {fmt(r, 'warp_near_cut', 3)} | {fmt(r, 'gt_warp_near_cut', 3)} |")
        if sheets:
            sheet(root, clip, rows, 17)


if __name__ == "__main__":
    main()
