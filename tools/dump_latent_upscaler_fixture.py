#!/usr/bin/env python3
"""Reference run of the LBH latent upscaler (the author node, fp32) for
h3_latent_upscale's parity test. Development only (needs torch).

  dump_latent_upscaler_fixture.py --node-dir <dir with upscaler_node.py>
      --weights <safetensors> --h3lt <clip latent> --out-dir <dir>
      [--crop T,H,W] [--target H,W] [--device cpu|mps]

Writes crop.h3lt (the cropped input in the engine's clip-latent format, with
a zero audio latent), reference.f32 ([24,T,oh,ow] F32, the node's output =
de-normalized network output) and info.json. Compare with
  h3_latent_upscale crop.h3lt out.h3lt <ow*16> <oh*16>
  python3 tools/dump_latent_upscaler_fixture.py --compare out.h3lt reference.f32
"""
import argparse, importlib.util, json, os, struct, sys, types
import numpy as np

MAGIC = 0x544C3348


def read_h3lt(path):
    with open(path, "rb") as f:
        header = struct.unpack("<9i", f.read(36))
        assert header[0] == MAGIC and header[1] == 1 and header[2] == 24 and header[6] == 32, "not an H3 clip latent file"
        _, _, _, t, h, w, _, audio_t, frames = header
        video = np.frombuffer(f.read(24 * t * h * w * 4), dtype=np.float32).reshape(24, t, h, w)
        audio = np.frombuffer(f.read(32 * 2 * audio_t * 4), dtype=np.float32).reshape(32, 2, audio_t)
    return video, audio, frames


def write_h3lt(path, video, audio, frames):
    c, t, h, w = video.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<9i", MAGIC, 1, 24, t, h, w, 32, audio.shape[2], frames))
        f.write(np.ascontiguousarray(video, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(audio, dtype=np.float32).tobytes())


def load_node(node_dir):
    """The shim NVIDIA's open_h3/upscaler.py uses: only ComfyUI's registration is faked."""
    folder = types.ModuleType("folder_paths")
    folder.folder_names_and_paths = {"latent_upscale_models": ([], set())}
    sys.modules["folder_paths"] = folder
    sys.modules["comfy.model_management"] = None
    sys.modules["comfy_api.latest"] = None
    spec = importlib.util.spec_from_file_location("author_upscaler", os.path.join(node_dir, "upscaler_node.py"))
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    node.io.NodeOutput = lambda value: (value,)
    return node


def psnr(a, b):
    err = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    peak = float(np.max(np.abs(b)))
    return float("inf") if err == 0 else 10 * np.log10(peak * peak / err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-dir")
    ap.add_argument("--weights")
    ap.add_argument("--h3lt")
    ap.add_argument("--out-dir")
    ap.add_argument("--crop", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("OUT_H3LT", "REFERENCE_F32"))
    ap.add_argument("--dump-stages", action="store_true")
    ap.add_argument("--compare-stages", metavar="DIR")
    ap.add_argument("--debug-out-block", type=int)
    args = ap.parse_args()
    if args.compare_stages:
        import glob
        for ref in sorted(glob.glob(os.path.join(args.compare_stages, "ref_stage_*.f32"))) + sorted(glob.glob(os.path.join(args.compare_stages, "ref_sub_*.f32"))):
            i = int(os.path.basename(ref)[:-4].split("_")[-1])
            eng = os.path.join(args.compare_stages, os.path.basename(ref)[4:])
            if not os.path.exists(eng):
                print(i, "engine stage missing"); continue
            a = np.fromfile(eng, dtype=np.float32); b = np.fromfile(ref, dtype=np.float32)
            if a.shape != b.shape:
                print(i, "shape", a.shape, b.shape); continue
            print(f"stage {i:02d}: PSNR {psnr(a, b):6.2f} dB  rel-L2 {np.linalg.norm(a-b)/np.linalg.norm(b):.3e}  ref |max| {np.max(np.abs(b)):.3f}")
        return
    if args.compare:
        video, _, _ = read_h3lt(args.compare[0])
        ref = np.fromfile(args.compare[1], dtype=np.float32).reshape(video.shape)
        rel = np.linalg.norm(video - ref) / np.linalg.norm(ref)
        print(f"engine vs reference: PSNR {psnr(video, ref):.2f} dB, rel-L2 {rel:.3e}, max|diff| {np.max(np.abs(video - ref)):.3e}, ref range [{ref.min():.3f}, {ref.max():.3f}]")
        return
    import torch
    node = load_node(args.node_dir)
    folder = sys.modules["folder_paths"]
    folder.get_full_path_or_raise = lambda category, name: args.weights
    video, audio, frames = read_h3lt(args.h3lt)
    if args.crop:
        t, h, w = (int(v) for v in args.crop.split(","))
        video = video[:, :t, :h, :w]
        frames = 1 + 3 * (t - 1) if t > 1 else 1
        audio = np.zeros((32, 2, 1), dtype=np.float32)
    c, t, h, w = video.shape
    oh, ow = (int(v) for v in args.target.split(",")) if args.target else (2 * h, 2 * w)
    os.makedirs(args.out_dir, exist_ok=True)
    write_h3lt(os.path.join(args.out_dir, "crop.h3lt"), video, audio, frames)
    mode = {"mode": node.UpscaleMode.TARGET_DIMENSIONS, "width": ow * 16, "height": oh * 16}
    if args.dump_stages:
        # the same stage numbering as H3_LU_DUMP_DIR: conv_in, each in block,
        # the resize, each out block; saved channels-last [T,H,W,C]
        model = node.load_model("upscaler", torch.device(args.device), "fp32")
        stages = []
        def save(_, __, out):
            stages.append(out.detach().float()[0].permute(1, 2, 3, 0).contiguous().numpy())
        model.conv_in.register_forward_hook(save)
        for b in model.in_blocks: b.register_forward_hook(save)
        orig = torch.nn.functional.interpolate
        def hooked(x, *a, **k):
            y = orig(x, *a, **k); save(None, None, y); return y
        torch.nn.functional.interpolate = hooked
        for b in model.out_blocks: b.register_forward_hook(save)
        if args.debug_out_block is not None:
            blk = model.out_blocks[args.debug_out_block]
            subs = []
            def save_sub(_, __, out):
                subs.append(out.detach().float()[0].permute(1, 2, 3, 0).contiguous().numpy())
            blk.in_layers[1].register_forward_hook(save_sub)
            blk.in_layers.register_forward_hook(save_sub)
            blk.out_layers[0].register_forward_hook(save_sub)
            blk.out_layers.register_forward_hook(save_sub)
    samples = torch.from_numpy(np.ascontiguousarray(video))[None].float()
    with torch.inference_mode():
        result, = node.MinimaxH3LatentUpscaler3D.execute(
            {"samples": samples}, "upscaler", mode, align=16, enable_temporal_chunking=False,
            force_unload=False, device=args.device, precision="fp32")
    out = result["samples"].float()[0].cpu().numpy()
    if args.dump_stages:
        torch.nn.functional.interpolate = orig
        for i, s in enumerate(stages):
            s.astype(np.float32).tofile(os.path.join(args.out_dir, f"ref_stage_{i:02d}.f32"))
        print("stages", len(stages), [s.shape for s in stages[:2]])
        if args.debug_out_block is not None:
            for i, s in enumerate(subs):
                s.astype(np.float32).tofile(os.path.join(args.out_dir, f"ref_sub_{i}.f32"))
    assert out.shape == (24, t, oh, ow), out.shape
    out.astype(np.float32).tofile(os.path.join(args.out_dir, "reference.f32"))
    json.dump({"input": [c, t, h, w], "output": list(out.shape), "device": args.device,
               "weights": os.path.basename(args.weights), "node": "upscaler_node.py fp32",
               "latents_mean_equal_vae": True}, open(os.path.join(args.out_dir, "info.json"), "w"), indent=1)
    print("wrote", args.out_dir, "output", out.shape, "range", float(out.min()), float(out.max()))


if __name__ == "__main__":
    main()
