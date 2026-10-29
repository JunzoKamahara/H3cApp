#!/usr/bin/env python3
"""Quality metrics for one upscaled variant of an evaluation clip
(UPSCALE_PLAN §9.5). Development only.

    python tools/quality_metrics.py --clip DIR --variant NAME --input out.raw|out.mp4
        [--crop X,Y,W,H] [--frames 8] [--delete-raw] [--no-nr]

Reads DIR/gt.raw (uint8 [T, H, W, 3]) and the variant's frames, and writes
DIR/metrics-NAME.json with: PSNR / SSIM / LPIPS(alex) against the ground
truth, MUSIQ and MANIQA (no-reference, pyiqa, on --frames evenly spaced
frames), a flicker figure (mean absolute difference between consecutive
frames, also for the ground truth) and, with --crop, the same metrics on the
crop (a face). Saves DIR/frame16-NAME.png and DIR/crop16-NAME.png.
"""
import argparse, json, os, re, sys
import numpy as np
import torch
from PIL import Image


def read_info(d):
    t = open(os.path.join(d, "info.txt")).read()
    g = re.search(r"gt (\d+)x(\d+)", t)
    return int(g.group(1)), int(g.group(2))


def read_frames(path, w, h, count):
    if path.endswith(".raw"):
        a = np.fromfile(path, dtype=np.uint8)
        return a.reshape(-1, h, w, 3)[:count]
    import av
    frames = []
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= count:
                break
    return np.stack(frames)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--crop", default=None, help="X,Y,W,H in output pixels")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--delete-raw", action="store_true")
    ap.add_argument("--no-nr", action="store_true")
    args = ap.parse_args()
    w, h = read_info(args.clip)
    gt = np.fromfile(os.path.join(args.clip, "gt.raw"), dtype=np.uint8).reshape(-1, h, w, 3)
    T = gt.shape[0]
    out = read_frames(args.input, w, h, T)
    if out.shape != gt.shape:
        sys.exit(f"{args.variant}: frames {out.shape} vs gt {gt.shape}")
    from skimage.metrics import structural_similarity
    import lpips
    loss = lpips.LPIPS(net="alex", verbose=False)
    idx = np.linspace(0, T - 1, args.frames).round().astype(int)
    res = {"variant": args.variant, "frames": T, "sampled": idx.tolist()}

    def full_reference(a, b, tag):
        ps = [psnr(a[i], b[i]) for i in range(T)]
        ss = [structural_similarity(a[i], b[i], channel_axis=2, data_range=255) for i in idx]
        lp = []
        for i in idx:
            x = torch.from_numpy(a[i]).permute(2, 0, 1)[None].float() / 127.5 - 1
            y = torch.from_numpy(b[i]).permute(2, 0, 1)[None].float() / 127.5 - 1
            lp.append(float(loss(x, y)))
        res[f"{tag}psnr"] = float(np.mean(ps)); res[f"{tag}ssim"] = float(np.mean(ss)); res[f"{tag}lpips"] = float(np.mean(lp))

    def flicker(a, tag):
        d = [float(np.mean(np.abs(a[i].astype(np.float32) - a[i - 1].astype(np.float32)))) for i in range(1, T)]
        res[f"{tag}flicker"] = float(np.mean(d))

    full_reference(out, gt, ""); flicker(out, ""); flicker(gt, "gt_")
    crop = None
    if args.crop:
        x, y, cw, ch = [int(v) for v in args.crop.split(",")]
        crop = (x, y, cw, ch)
        full_reference(out[:, y:y + ch, x:x + cw], gt[:, y:y + ch, x:x + cw], "face_")
    if not args.no_nr:
        import pyiqa
        for name in ("musiq", "maniqa"):
            metric = pyiqa.create_metric(name, device="cpu")
            vals, gvals, cvals = [], [], []
            for i in idx:
                x = torch.from_numpy(out[i]).permute(2, 0, 1)[None].float() / 255
                g = torch.from_numpy(gt[i]).permute(2, 0, 1)[None].float() / 255
                vals.append(float(metric(x))); gvals.append(float(metric(g)))
                if crop:
                    xx, yy, cw, ch = crop
                    cvals.append(float(metric(x[:, :, yy:yy + ch, xx:xx + cw])))
            res[name] = float(np.mean(vals)); res[f"gt_{name}"] = float(np.mean(gvals))
            if crop: res[f"face_{name}"] = float(np.mean(cvals))
    Image.fromarray(out[16]).save(os.path.join(args.clip, f"frame16-{args.variant}.png"))
    if crop:
        x, y, cw, ch = crop
        Image.fromarray(out[16, y:y + ch, x:x + cw]).save(os.path.join(args.clip, f"crop16-{args.variant}.png"))
        if not os.path.exists(os.path.join(args.clip, "crop16-gt.png")):
            Image.fromarray(gt[16, y:y + ch, x:x + cw]).save(os.path.join(args.clip, "crop16-gt.png"))
            Image.fromarray(gt[16]).save(os.path.join(args.clip, "frame16-gt.png"))
    with open(os.path.join(args.clip, f"metrics-{args.variant}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "sampled"}))
    if args.delete_raw and args.input.endswith(".raw"):
        os.remove(args.input)


if __name__ == "__main__":
    main()
