#!/usr/bin/env python3
"""Temporal quality metrics for one upscaled variant of an evaluation clip
(SEEDVR2_TEMPORAL_PLAN phase 5). Development only.

    python tools/quality_temporal.py --clip DIR --variant NAME --input FRAMES_DIR|out.raw
        [--window W --overlap O --blend 0|1] [--crop X,Y,W,H] [--cut N]
        [--frames 8] [--no-nr] [--flow-scale 1.0]

DIR holds gt.raw (uint8 [T, H, W, 3]) and info.txt ("gt WxH ..."). The
variant is a folder of PNG frames (sorted by name) or a raw file of the
same layout as gt.raw.

Metrics written to DIR/metrics-NAME.json:
  warp         mean warping error (Lai et al. 2018 style): frame t+1 warped
               back to t with RAFT flow computed on the ground truth, mean
               absolute difference (0-255) over pixels that pass the
               forward-backward check. The same flow and mask for every
               variant of a clip, so the figure compares methods, not flows.
  gt_warp      the same figure for the ground truth itself (the floor).
  warp_boundary / warp_interior
               the mean over frame pairs that straddle a window boundary
               (--window/--overlap/--blend, the session's window rule) and
               over the other pairs.
  flicker      mean absolute difference between consecutive frames.
  psnr/ssim/lpips, musiq/maniqa, and face_* on --crop, as quality_metrics.py.
  pairs        per-pair warp errors (for plots).
--cut N excludes the pair (N-1, N) (a scene cut) from every warp figure
and reports warp_cut (that pair) and warp_near_cut (pairs within 2 of it).
The flow is cached in DIR/flow.npz (float16).
"""
import argparse, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def read_info(d):
    t = open(os.path.join(d, "info.txt")).read()
    g = re.search(r"gt (\d+)x(\d+)", t)
    return int(g.group(1)), int(g.group(2))


def read_frames(path, w, h, count):
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.lower().endswith(".png"))[:count]
        return np.stack([np.asarray(Image.open(os.path.join(path, n)).convert("RGB")) for n in names])
    a = np.fromfile(path, dtype=np.uint8)
    return a.reshape(-1, h, w, 3)[:count]


def window_starts(n, w, o):
    """The session's windows: starts at multiples of the stride, the last one
    moved back to end on the last frame; a clip shorter than the window is
    one (padded) window."""
    if w <= 1:
        return []
    if n <= w:
        return [(0, n - 1)]
    stride = w - o
    starts, s = [], 0
    while True:
        if s + w >= n:
            starts.append(n - w)
            break
        starts.append(s)
        s += stride
    return [(s, s + w - 1) for s in starts]


def boundary_pairs(n, w, o, blend):
    wins = window_starts(n, w, o)
    if len(wins) <= 1:
        return set()
    pairs = set()
    if blend == 0:
        owner = [None] * n
        for i, (a, b) in enumerate(wins):
            for t in range(a, b + 1):
                if owner[t] is None:
                    owner[t] = i
        for t in range(n - 1):
            if owner[t] != owner[t + 1]:
                pairs.add(t)
    else:
        cover = [0] * n
        for a, b in wins:
            for t in range(a, b + 1):
                cover[t] += 1
        for t in range(n - 1):
            if cover[t] > 1 or cover[t + 1] > 1:
                pairs.add(t)
    return pairs


def raft_flow(gt, cache, scale):
    """Forward flow t->t+1 and backward t+1->t on the ground truth, RAFT
    large, cached. Returns float32 arrays [T-1, 2, H, W] (pixels)."""
    if os.path.exists(cache):
        z = np.load(cache)
        if z["fwd"].shape[0] == gt.shape[0] - 1 and float(z["scale"]) == scale:
            return z["fwd"].astype(np.float32), z["bwd"].astype(np.float32)
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval()
    T, H, W = gt.shape[:3]
    h8, w8 = int(round(H * scale / 8)) * 8, int(round(W * scale / 8)) * 8
    fwd, bwd = [], []
    with torch.no_grad():
        for t in range(T - 1):
            a = torch.from_numpy(gt[t]).permute(2, 0, 1)[None].float() / 127.5 - 1
            b = torch.from_numpy(gt[t + 1]).permute(2, 0, 1)[None].float() / 127.5 - 1
            if (h8, w8) != (H, W):
                a = F.interpolate(a, (h8, w8), mode="bilinear", align_corners=False)
                b = F.interpolate(b, (h8, w8), mode="bilinear", align_corners=False)
            f = model(a, b)[-1]
            g = model(b, a)[-1]
            if (h8, w8) != (H, W):
                f = F.interpolate(f, (H, W), mode="bilinear", align_corners=False)
                g = F.interpolate(g, (H, W), mode="bilinear", align_corners=False)
                f[:, 0] *= W / w8; f[:, 1] *= H / h8
                g[:, 0] *= W / w8; g[:, 1] *= H / h8
            fwd.append(f[0].numpy()); bwd.append(g[0].numpy())
            print(f"  flow {t + 1}/{T - 1}", end="\r", file=sys.stderr)
    fwd, bwd = np.stack(fwd), np.stack(bwd)
    np.savez_compressed(cache, fwd=fwd.astype(np.float16), bwd=bwd.astype(np.float16), scale=scale)
    return fwd, bwd


def warp(img, flow):
    """img [H, W, 3] uint8 sampled at x + flow (flow [2, H, W]) -> float [H, W, 3]."""
    H, W = img.shape[:2]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    fx, fy = torch.from_numpy(flow[0]), torch.from_numpy(flow[1])
    gx = (xs + fx) / max(W - 1, 1) * 2 - 1
    gy = (ys + fy) / max(H - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    inside = (gx.abs() <= 1) & (gy.abs() <= 1)
    return out[0].permute(1, 2, 0).numpy(), inside.numpy()


def valid_mask(fwd, bwd):
    """Forward-backward consistency (Sundaram et al.): |f(x) + b(x + f(x))|^2
    < 0.01 (|f|^2 + |b'|^2) + 0.5."""
    H, W = fwd.shape[1:]
    # warp the backward flow to t with the forward flow
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    gx = (xs + torch.from_numpy(fwd[0])) / max(W - 1, 1) * 2 - 1
    gy = (ys + torch.from_numpy(fwd[1])) / max(H - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    b = F.grid_sample(torch.from_numpy(bwd)[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0].numpy()
    inside = ((gx.abs() <= 1) & (gy.abs() <= 1)).numpy()
    diff = np.sum((fwd + b) ** 2, axis=0)
    mag = np.sum(fwd ** 2, axis=0) + np.sum(b ** 2, axis=0)
    return (diff < 0.01 * mag + 0.5) & inside


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--blend", type=int, default=0)
    ap.add_argument("--crop", default=None, help="X,Y,W,H in output pixels")
    ap.add_argument("--cut", type=int, default=None, help="first frame of the second scene")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--no-nr", action="store_true")
    ap.add_argument("--flow-scale", type=float, default=1.0)
    ap.add_argument("--seconds-per-frame", type=float, default=None)
    ap.add_argument("--peak-gib", type=float, default=None)
    ap.add_argument("--footprint-gb", type=float, default=None)
    args = ap.parse_args()
    w, h = read_info(args.clip)
    gt = np.fromfile(os.path.join(args.clip, "gt.raw"), dtype=np.uint8).reshape(-1, h, w, 3)
    T = gt.shape[0]
    out = read_frames(args.input, w, h, T)
    if out.shape != gt.shape:
        sys.exit(f"{args.variant}: frames {out.shape} vs gt {gt.shape}")
    res = {"variant": args.variant, "frames": T, "window": args.window, "overlap": args.overlap, "blend": args.blend}
    for k in ("seconds_per_frame", "peak_gib", "footprint_gb"):
        if getattr(args, k) is not None:
            res[k] = getattr(args, k)
    crop = None
    if args.crop:
        x, y, cw, ch = [int(v) for v in args.crop.split(",")]
        crop = (x, y, cw, ch)

    # --- warping error (flow on the ground truth, cached per clip)
    fwd, bwd = raft_flow(gt, os.path.join(args.clip, "flow.npz"), args.flow_scale)
    bpairs = boundary_pairs(T, args.window, args.overlap, args.blend)
    cut_pair = args.cut - 1 if args.cut is not None else None
    pairs, gpairs, cpairs, valid_frac = [], [], [], []
    for t in range(T - 1):
        m = valid_mask(fwd[t], bwd[t])
        wo, _ = warp(out[t + 1], fwd[t])
        wg, _ = warp(gt[t + 1], fwd[t])
        d = np.abs(out[t].astype(np.float32) - wo).mean(axis=2)
        dg = np.abs(gt[t].astype(np.float32) - wg).mean(axis=2)
        pairs.append(float(d[m].mean()) if m.any() else float("nan"))
        gpairs.append(float(dg[m].mean()) if m.any() else float("nan"))
        valid_frac.append(float(m.mean()))
        if crop:
            x, y, cw, ch = crop
            mc = m[y:y + ch, x:x + cw]
            cpairs.append(float(d[y:y + ch, x:x + cw][mc].mean()) if mc.any() else float("nan"))
        print(f"  warp {t + 1}/{T - 1}", end="\r", file=sys.stderr)
    pairs, gpairs = np.array(pairs), np.array(gpairs)
    keep = np.ones(T - 1, dtype=bool)
    if cut_pair is not None and 0 <= cut_pair < T - 1:
        keep[cut_pair] = False
        res["warp_cut"] = float(pairs[cut_pair])
        near = [t for t in range(max(0, cut_pair - 2), min(T - 1, cut_pair + 3)) if t != cut_pair]
        res["warp_near_cut"] = float(np.nanmean(pairs[near]))
        res["gt_warp_near_cut"] = float(np.nanmean(gpairs[near]))
    bmask = np.array([t in bpairs for t in range(T - 1)]) & keep
    imask = ~np.array([t in bpairs for t in range(T - 1)]) & keep
    res["warp"] = float(np.nanmean(pairs[keep]))
    res["gt_warp"] = float(np.nanmean(gpairs[keep]))
    res["warp_boundary"] = float(np.nanmean(pairs[bmask])) if bmask.any() else None
    res["warp_interior"] = float(np.nanmean(pairs[imask])) if imask.any() else None
    res["gt_warp_boundary"] = float(np.nanmean(gpairs[bmask])) if bmask.any() else None
    res["boundary_pairs"] = sorted(int(t) for t in bpairs)
    res["valid_fraction"] = float(np.mean(valid_frac))
    res["pairs"] = [round(float(v), 3) for v in pairs]
    res["gt_pairs"] = [round(float(v), 3) for v in gpairs]
    if crop:
        cpairs = np.array(cpairs)
        res["face_warp"] = float(np.nanmean(cpairs[keep]))
        res["face_warp_boundary"] = float(np.nanmean(cpairs[bmask])) if bmask.any() else None
        res["face_warp_interior"] = float(np.nanmean(cpairs[imask])) if imask.any() else None

    # --- flicker, fidelity, no-reference (as quality_metrics.py)
    def flicker(a, tag):
        d = [float(np.mean(np.abs(a[i].astype(np.float32) - a[i - 1].astype(np.float32)))) for i in range(1, T)
             if cut_pair is None or i - 1 != cut_pair]
        res[f"{tag}flicker"] = float(np.mean(d))
    flicker(out, ""); flicker(gt, "gt_")
    from skimage.metrics import structural_similarity
    import lpips
    loss = lpips.LPIPS(net="alex", verbose=False)
    idx = np.linspace(0, T - 1, args.frames).round().astype(int)
    res["sampled"] = idx.tolist()

    def full_reference(a, b, tag):
        ps = [psnr(a[i], b[i]) for i in range(T)]
        ss = [structural_similarity(a[i], b[i], channel_axis=2, data_range=255) for i in idx]
        lp = []
        for i in idx:
            x = torch.from_numpy(a[i]).permute(2, 0, 1)[None].float() / 127.5 - 1
            y = torch.from_numpy(b[i]).permute(2, 0, 1)[None].float() / 127.5 - 1
            lp.append(float(loss(x, y)))
        res[f"{tag}psnr"] = float(np.mean(ps)); res[f"{tag}ssim"] = float(np.mean(ss)); res[f"{tag}lpips"] = float(np.mean(lp))
    full_reference(out, gt, "")
    if crop:
        x, y, cw, ch = crop
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
    with open(os.path.join(args.clip, f"metrics-{args.variant}.json"), "w") as f:
        json.dump(res, f, indent=1)
    short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
             if k not in ("pairs", "gt_pairs", "sampled", "boundary_pairs")}
    print(json.dumps(short))


if __name__ == "__main__":
    main()
