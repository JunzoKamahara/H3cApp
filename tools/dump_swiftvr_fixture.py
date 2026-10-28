#!/usr/bin/env python3
"""Dump SwiftVR reference checkpoints for tests/test_real_vsr.c (development only).

Runs the official SwiftVR code (github.com/H-oliday/SwiftVR, Apache-2.0) in
float32 on a short clip and writes every intermediate the engine is checked
against: the preprocessed input, the ReAE latents of each chunk, the DiT
constants, the first blocks' outputs, the denoised latents, the decoded frames
and the shifted-window index tables. The weights are the SwiftVR Research
License files the app downloads; the dump is a derivative of them and stays
local (misc/ is gitignored).

    python tools/dump_swiftvr_fixture.py \
        --swiftvr-repo /path/to/SwiftVR \
        --model-dir "$HOME/Library/Application Support/H3cApp/Upscalers/SwiftVR" \
        --video some_clip.mp4 --out misc/fixtures/swiftvr-768x544

Needs torch, safetensors, diffusers, pillow and av in the environment (a
scratch venv; nothing here ships with the app). The DiT is evaluated one block
at a time from the safetensors file so the 20 GB F32 weights never have to be
resident; the ReAE is small and is loaded whole.

The clip is the first --frames frames of --video, centre-cropped to the
output size and downscaled to the LQ size, so the crop doubles as ground
truth. Defaults give a 384x272 LQ clip restored to 768x544 (a 24x17 token
grid: more than one 16-token window on the wide axis, so both window
parities and the owner rule are exercised) over 57 frames (FIRST 28,
MIDDLE 24, LAST 5: all three chunk kinds).
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def stub_module(name, **attrs):
    """The official package imports decord at package import time; it is not
    needed here (frames come from PyAV)."""
    if name in sys.modules:
        return
    import importlib.machinery
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


def read_frames(path, count):
    import av
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= count:
                break
    if len(frames) < count:
        raise SystemExit(f"{path}: only {len(frames)} frames, need {count}")
    return np.stack(frames)


def make_clip(frames, out_w, out_h, lq_w, lq_h):
    """Centre crop to out_h x out_w (ground truth) and downscale to lq_h x lq_w."""
    n, h, w, _ = frames.shape
    if h < out_h or w < out_w:
        raise SystemExit(f"source {w}x{h} is smaller than the output {out_w}x{out_h}")
    y0, x0 = (h - out_h) // 2, (w - out_w) // 2
    hq = np.ascontiguousarray(frames[:, y0:y0 + out_h, x0:x0 + out_w])
    lq = np.stack([np.asarray(Image.fromarray(f).resize((lq_w, lq_h), Image.BICUBIC)) for f in hq])
    return hq, np.ascontiguousarray(lq)


def preprocess_clip_uint8(frames_uint8, out_h, out_w, pad_h, pad_w):
    """swiftvr.io.preprocess_clip_uint8 (bilinear, align_corners=False), F32."""
    frames = frames_uint8.permute(0, 3, 1, 2).contiguous().to(torch.float32)
    _, _, h, w = frames.shape
    if (h, w) != (out_h, out_w):
        frames = F.interpolate(frames, size=(out_h, out_w), mode="bilinear", align_corners=False)
    frames = frames / 255.0
    if pad_h > 0 or pad_w > 0:
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return frames.unsqueeze(0)


def aligned_pad(size, multiple=32):
    return (multiple - size % multiple) % multiple


class LazyDiT:
    """The official WanTransformer3DModel forward (swiftvr.streaming.dit
    _dit_forward_chunk) with every block loaded from the safetensors file just
    before it runs, in float32."""

    def __init__(self, model_dir, device, tf):
        self.tf = tf
        self.device = device
        path = os.path.join(model_dir, "transformer", "diffusion_pytorch_model.safetensors")
        with open(os.path.join(model_dir, "transformer", "config.json")) as f:
            self.config = json.load(f)
        cfg = self.config
        self.layers = int(cfg["num_layers"])
        self.heads = int(cfg["num_attention_heads"])
        self.head_dim = int(cfg["attention_head_dim"])
        self.dim = self.heads * self.head_dim
        self.ffn = int(cfg["ffn_dim"])
        self.patch = tuple(cfg["patch_size"])
        self.eps = float(cfg.get("eps", 1e-6))
        self.store = safe_open(path, "pt", device="cpu")
        self.keys = list(self.store.keys())

        self.cond = tf.WanTimeTextImageEmbedding(
            dim=self.dim, time_freq_dim=int(cfg["freq_dim"]), time_proj_dim=self.dim * 6,
            text_embed_dim=int(cfg["text_dim"]))
        self._load(self.cond, "condition_embedder.")
        self.patch_embedding = nn.Conv3d(int(cfg["in_channels"]), self.dim,
                                         kernel_size=self.patch, stride=self.patch)
        self._load(self.patch_embedding, "patch_embedding.")
        self.proj_out = nn.Linear(self.dim, int(cfg["out_channels"]) * int(np.prod(self.patch)))
        self._load(self.proj_out, "proj_out.")
        self.norm_out = tf.FP32LayerNorm(self.dim, self.eps, elementwise_affine=False)
        self.scale_shift_table = self.store.get_tensor("scale_shift_table").float().to(device)
        self.rope = tf.WanRotaryPosEmbed(self.head_dim, self.patch, int(cfg.get("rope_max_seq_len", 1024)))
        self.rope.to(device)
        self.window_hw = (16, 16)

    def _load(self, module, prefix):
        state = {k[len(prefix):]: self.store.get_tensor(k).float()
                 for k in self.keys if k.startswith(prefix)}
        module.load_state_dict(state, strict=True)
        module.to(self.device).eval()

    def load_block(self, i):
        tf = self.tf
        blk = tf.WanTransformerBlock(self.dim, self.ffn, self.heads, "rms_norm_across_heads",
                                     bool(self.config.get("cross_attn_norm", True)), self.eps, None)
        self._load(blk, f"blocks.{i}.")
        return blk

    def condition(self, prompt_emb):
        ts = torch.full((1,), self.tf.__dict__.get("INFERENCE_TIMESTEP", 1000.0),
                        device=self.device, dtype=torch.float32)
        pe = prompt_emb.to(self.device, torch.float32).unsqueeze(0)
        with torch.inference_mode():
            temb, tp, enc_hs, _ = self.cond(ts, pe, None, timestep_seq_len=None)
        return temb, tp.unflatten(1, (6, -1)), enc_hs

    @torch.inference_mode()
    def forward_chunk(self, chunk, temb, tp, enc_hs, t_off, full_attention=False, dump=None, log=None):
        """chunk: [1, C, F, H, W] latents. Returns the predicted velocity."""
        from swiftvr.streaming.dit import _rope_with_offset
        tf = self.tf
        p_t, p_h, p_w = self.patch
        B, C, Fr, H, W = chunk.shape
        ppf, pph, ppw = Fr // p_t, H // p_h, W // p_w
        rope = _rope_with_offset(self.rope, ppf, pph, ppw, t_off)
        hs = self.patch_embedding(chunk).flatten(2).transpose(1, 2).contiguous()
        if dump is not None:
            dump["rope_cos"] = rope[0][0, :, 0, :].clone()
            dump["rope_sin"] = rope[1][0, :, 0, :].clone()
            dump["hs_in"] = hs[0].clone()
        for i in range(self.layers):
            t0 = time.perf_counter()
            blk = self.load_block(i)
            if not full_attention:
                blk.attn1._thw = (ppf, pph, ppw)
                blk.attn1._do_shift = bool(i % 2 == 1)
                blk.attn1.set_processor(tf.WanShiftWindow2DInferProcessor(window_hw=self.window_hw))
            if dump is not None and i == 0:
                ctx = enc_hs
                dump["k2_0"] = blk.attn2.norm_k(blk.attn2.to_k(ctx))[0].clone()
                dump["v2_0"] = blk.attn2.to_v(ctx)[0].clone()
            hs = blk(hs, enc_hs, tp, rope)
            if dump is not None and i < 2:
                dump[f"block{i}_out"] = hs[0].clone()
            if log:
                log(f"    block {i:2d} {time.perf_counter() - t0:.2f}s")
            del blk
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hs = (self.norm_out(hs.float()) * (1 + scale) + shift)
        hs = self.proj_out(hs)
        hs = hs.reshape(B, ppf, pph, ppw, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swiftvr-repo", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lq-width", type=int, default=384)
    ap.add_argument("--lq-height", type=int, default=272)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--frames", type=int, default=57)
    ap.add_argument("--clip-len", type=int, default=24)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--stages", default="input,reae,dit,decode",
                    help="comma list of input,reae,dit,decode (later stages need the earlier dumps)")
    ap.add_argument("--no-full-attention", action="store_true",
                    help="skip the extra dense-attention pass over chunk 0")
    args = ap.parse_args()

    stub_module("decord", bridge=types.SimpleNamespace(set_bridge=lambda *a, **k: None))
    try:
        import imageio  # noqa: F401  (only imported by the package; not used here)
    except ImportError:
        stub_module("imageio", get_writer=None)
    sys.path.insert(0, os.path.abspath(args.swiftvr_repo))
    import swiftvr.models.transformer as tf
    from swiftvr.models.reae import ReAE
    from swiftvr.streaming.chunk import build_chunk_specs, ChunkType
    from swiftvr.streaming.tae import StreamingTAE
    from swiftvr.streaming import dit as sdit

    stages = set(args.stages.split(","))
    device = torch.device(args.device)
    model_dir = os.path.expanduser(args.model_dir)
    out = args.out
    os.makedirs(out, exist_ok=True)
    out_w, out_h = args.lq_width * args.scale, args.lq_height * args.scale
    pad_w, pad_h = aligned_pad(out_w), aligned_pad(out_h)
    total_frames = 4 * ((args.frames - 1) // 4) + 1
    if total_frames != args.frames:
        print(f"note: {args.frames} frames -> {total_frames} (4k+1), the same as restore_video")
    specs = build_chunk_specs(total_frames, args.clip_len)
    print(f"device {device}, LQ {args.lq_width}x{args.lq_height} -> {out_w}x{out_h} (pad {pad_w}x{pad_h}), "
          f"{total_frames} frames, chunks {[(s.ctype.value, s.frame_start, s.frame_count, s.b) for s in specs]}")

    meta_path = os.path.join(out, "fixture.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({
        "lq_width": args.lq_width, "lq_height": args.lq_height, "out_width": out_w, "out_height": out_h,
        "pad_width": pad_w, "pad_height": pad_h, "frames": total_frames, "clip_len": args.clip_len,
        "source_video": os.path.abspath(args.video), "torch": torch.__version__, "device": str(device),
        "chunks": [{"type": s.ctype.value, "start": s.frame_start, "frames": s.frame_count, "b": s.b,
                    "index": s.clip_idx} for s in specs],
    })

    def save(name, tensors):
        path = os.path.join(out, name)
        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, path)
        print(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB): {', '.join(tensors)}")

    # ---- input ---------------------------------------------------------
    if "input" in stages:
        t0 = time.perf_counter()
        frames = read_frames(args.video, total_frames)
        hq, lq = make_clip(frames, out_w, out_h, args.lq_width, args.lq_height)
        save("input.safetensors", {"lq_rgb": torch.from_numpy(lq), "hq_rgb": torch.from_numpy(hq)})
        print(f"input: {time.perf_counter() - t0:.1f}s")
    lq = load_file(os.path.join(out, "input.safetensors"))["lq_rgb"]

    # ---- ReAE encoder --------------------------------------------------
    if "reae" in stages:
        t0 = time.perf_counter()
        reae = ReAE(os.path.join(model_dir, "reae.safetensors")).to(device, torch.float32).eval()
        tae = StreamingTAE(reae)
        dump = {}
        for spec in specs:
            frames_u8 = lq[spec.frame_start:spec.frame_start + spec.frame_count].to(device)
            clip = preprocess_clip_uint8(frames_u8, out_h, out_w, pad_h, pad_w)
            if spec.clip_idx == 0:
                dump["pre_clip4"] = clip[0, :4].clone()
            z = tae.encode_chunk_fixed(clip, spec)       # [1, T', C, h, w]
            dump[f"z{spec.clip_idx}"] = z[0].clone()
            print(f"  chunk {spec.clip_idx} {spec.ctype.value}: {tuple(clip.shape)} -> z {tuple(z[0].shape)}")
        # The decoder on the raw LQ latents of chunk 0 (fresh state): checks
        # the engine's decoder before any DiT exists. First 4 frames after
        # the 3 trimmed ones, F32.
        tae_dec = StreamingTAE(reae)
        rgb = tae_dec.decode_chunk_fixed(dump["z0"].unsqueeze(0), specs[0])
        if pad_h > 0:
            rgb = rgb[:, :, :, :-pad_h, :]
        if pad_w > 0:
            rgb = rgb[:, :, :, :, :-pad_w]
        dump["dec_z0_4"] = rgb[0, :4].clone()
        save("reae.safetensors", dump)
        del reae, tae, tae_dec
        print(f"reae: {time.perf_counter() - t0:.1f}s")

    # ---- DiT -----------------------------------------------------------
    if "dit" in stages:
        t0 = time.perf_counter()
        zs = load_file(os.path.join(out, "reae.safetensors"))
        dit = LazyDiT(model_dir, device, tf)
        prompt_emb = load_file(os.path.join(model_dir, "prompt_embedding.safetensors"))["prompt_emb"][0]
        temb, tp, enc_hs = dit.condition(prompt_emb)
        dump = {"temb": temb[0].clone(), "tp": tp[0].clone(), "ctx": enc_hs[0].clone()}
        n_lat = args.clip_len // 4
        g_off = 0
        prev_lq = None
        rope_offsets = []
        for spec in specs:
            z = zs[f"z{spec.clip_idx}"].to(device)                 # [T', C, h, w]
            z_bcfhw = z.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            if spec.ctype == ChunkType.LAST:
                lat_count = spec.b + 1
                pad_count = (n_lat + 1) - lat_count
                if pad_count > 0:
                    if prev_lq is not None:
                        pad_z = prev_lq[:, :, -pad_count:]
                    else:
                        pad_z = torch.zeros(1, z_bcfhw.shape[1], pad_count, *z_bcfhw.shape[3:], device=device)
                    z_in = torch.cat([pad_z, z_bcfhw], dim=2)
                else:
                    z_in = z_bcfhw
                t_off = max(0, g_off - pad_count)
                dump[f"zin{spec.clip_idx}"] = z_in[0].permute(1, 0, 2, 3).clone()
                pred = dit.forward_chunk(z_in, temb, tp, enc_hs, t_off, log=print)
                z_den = (z_in - pred)[:, :, -lat_count:]
                g_off += lat_count
            else:
                t_off = g_off
                chunk_dump = {} if spec.clip_idx == 0 else None
                pred = dit.forward_chunk(z_bcfhw, temb, tp, enc_hs, t_off, dump=chunk_dump, log=print)
                if chunk_dump is not None:
                    for k, v in chunk_dump.items():
                        dump[k if k.endswith("_0") else f"{k}0"] = v
                    dump["v0"] = pred[0].permute(1, 0, 2, 3).clone()
                z_den = z_bcfhw - pred
                prev_lq = z_bcfhw[:, :, -n_lat:].clone()
                g_off += z_bcfhw.shape[2]
            rope_offsets.append(t_off)
            dump[f"zden{spec.clip_idx}"] = z_den[0].permute(1, 0, 2, 3).clone()
            print(f"  chunk {spec.clip_idx} {spec.ctype.value}: t_off {t_off}, zden {tuple(dump[f'zden{spec.clip_idx}'].shape)}  "
                  f"[{time.perf_counter() - t0:.0f}s]")
        if not args.no_full_attention:
            z = zs["z0"].to(device).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            pred = dit.forward_chunk(z, temb, tp, enc_hs, 0, full_attention=True)
            dump["v0_full"] = pred[0].permute(1, 0, 2, 3).clone()
            print(f"  chunk 0 dense attention done [{time.perf_counter() - t0:.0f}s]")
        # Window tables of every token grid the chunks use (both parities).
        grids = {}
        for spec in specs:
            latents = (n_lat + 1) if spec.ctype == ChunkType.LAST else zs[f"z{spec.clip_idx}"].shape[0]
            grids[latents] = (latents, zs["z0"].shape[2] // 2, zs["z0"].shape[3] // 2)
        for T, H, W in grids.values():
            for do_shift in (False, True):
                m = tf._WindowRuntimeMetaCache.get(T, H, W, min(16, H), min(16, W), do_shift=do_shift,
                                                   prefer_front=not do_shift, device=torch.device("cpu"))
                tag = f"{'odd' if do_shift else 'even'}_t{T}"
                dump[f"win_lin_{tag}"] = m.lin_flat.to(torch.int32).view(m.Nw, m.Lw).clone()
                dump[f"win_owner_{tag}"] = m.owner_pos.to(torch.int32).clone()
        meta["rope_offsets"] = rope_offsets
        meta["token_grids"] = sorted(grids.values())
        save("dit.safetensors", dump)
        del dit
        print(f"dit: {time.perf_counter() - t0:.1f}s")

    # ---- ReAE decoder --------------------------------------------------
    if "decode" in stages:
        t0 = time.perf_counter()
        zd = load_file(os.path.join(out, "dit.safetensors"))
        reae = ReAE(os.path.join(model_dir, "reae.safetensors")).to(device, torch.float32).eval()
        tae = StreamingTAE(reae)
        frames_out = []
        dump = {}
        for spec in specs:
            z = zd[f"zden{spec.clip_idx}"].to(device).unsqueeze(0)   # [1, T', C, h, w]
            rgb = tae.decode_chunk_fixed(z, spec)                      # [1, T, 3, H+pad, W+pad]
            if pad_h > 0:
                rgb = rgb[:, :, :, :-pad_h, :]
            if pad_w > 0:
                rgb = rgb[:, :, :, :, :-pad_w]
            if spec.clip_idx == 0:
                dump["rgb0_f32_4"] = rgb[0, :4].clone()
            frames_out.append((rgb[0] * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu())
            print(f"  chunk {spec.clip_idx} {spec.ctype.value}: {tuple(rgb[0].shape)} frames")
        rgb_out = torch.cat(frames_out, 0)
        assert rgb_out.shape[0] == total_frames, rgb_out.shape
        dump["rgb_out"] = rgb_out
        save("output.safetensors", dump)
        hq = load_file(os.path.join(out, "input.safetensors"))["hq_rgb"].float()
        mse = ((rgb_out.float() - hq) ** 2).mean().item()
        up = F.interpolate(lq.permute(0, 3, 1, 2).float(), size=(out_h, out_w), mode="bilinear",
                           align_corners=False).permute(0, 2, 3, 1)
        mse_bilinear = ((up - hq) ** 2).mean().item()
        meta["psnr_vs_ground_truth"] = 10 * np.log10(255.0 ** 2 / mse)
        meta["psnr_bilinear_vs_ground_truth"] = 10 * np.log10(255.0 ** 2 / mse_bilinear)
        print(f"  PSNR vs ground truth: SwiftVR {meta['psnr_vs_ground_truth']:.2f} dB, "
              f"bilinear {meta['psnr_bilinear_vs_ground_truth']:.2f} dB")
        print(f"decode: {time.perf_counter() - t0:.1f}s")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
