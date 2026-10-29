#!/usr/bin/env python3
"""Collects misc/quality/set-v1/*/metrics-*.json into Markdown tables and
composes the face-crop A/B sheets (UPSCALE_PLAN §9.5). Development only.

    python tools/quality_report.py misc/quality/set-v1 [--variants a,b,c]
"""
import glob, json, os, sys
from PIL import Image, ImageDraw


def main():
    root = sys.argv[1]
    variants = None
    if len(sys.argv) > 3 and sys.argv[2] == "--variants":
        variants = sys.argv[3].split(",")
    clips = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    cols = [("psnr", "PSNR", 2), ("ssim", "SSIM", 4), ("lpips", "LPIPS↓", 4), ("musiq", "MUSIQ", 1), ("maniqa", "MANIQA", 4),
            ("flicker", "flicker", 3), ("face_psnr", "face PSNR", 2), ("face_lpips", "face LPIPS↓", 4),
            ("face_musiq", "face MUSIQ", 1), ("face_maniqa", "face MANIQA", 4)]
    for clip in clips:
        files = sorted(glob.glob(os.path.join(root, clip, "metrics-*.json")))
        rows = [json.load(open(f)) for f in files]
        if variants:
            rows = [r for r in rows if r["variant"] in variants]
        if not rows:
            continue
        print(f"\n### {clip}\n")
        print("| variant | " + " | ".join(c[1] for c in cols) + " |")
        print("|---|" + "---|" * len(cols))
        gt = rows[0]
        print("| ground truth | " + " | ".join(
            (f"{gt['gt_' + c[0]]:.{c[2]}f}" if ("gt_" + c[0]) in gt else "—") for c in cols) + " |")
        for r in rows:
            print(f"| {r['variant']} | " + " | ".join((f"{r[c[0]]:.{c[2]}f}" if c[0] in r else "—") for c in cols) + " |")
        # A/B sheet of the face crops
        crops = [("gt", os.path.join(root, clip, "crop16-gt.png"))] + \
                [(r["variant"], os.path.join(root, clip, f"crop16-{r['variant']}.png")) for r in rows]
        crops = [(n, p) for n, p in crops if os.path.exists(p)]
        if crops:
            ims = [Image.open(p) for _, p in crops]
            w, h = ims[0].size
            sheet = Image.new("RGB", (len(ims) * (w + 6), h + 18), "white")
            d = ImageDraw.Draw(sheet)
            for i, ((n, _), im) in enumerate(zip(crops, ims)):
                sheet.paste(im, (i * (w + 6), 18)); d.text((i * (w + 6) + 2, 2), n, fill="black")
            sheet.save(os.path.join(root, clip, "ab-crops.png"))


if __name__ == "__main__":
    main()
