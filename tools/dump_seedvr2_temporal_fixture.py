"""Reference tensors for H3cApp's temporal SeedVR2 (tests/test_seedvr2_temporal.c),
from the official ByteDance-Seed/SeedVR code run on the CPU in float32 with
the very weights the engine uses (seedvr2_ref.py). Development only; needs
torch, diffusers, einops, omegaconf, rotary-embedding-torch, safetensors and
Pillow, the SeedVR checkout (SEEDVR_REPO) and mflux's weight table
(MFLUX_MAPPING):

    python -I dump_seedvr2_temporal_fixture.py --model DIR --frames DIR \
        --out OUT.safetensors --frames-count 17 [--no-dit] [--seed 42]

Tensors (float32, the reference's own [C, T, H, W] layout unless noted):
  x                 the input clip in [-1, 1], [3, T, H, W]
  enc_conv_in, enc_down0..3, enc_mid, enc_h     encoder stages (unsliced)
  enc_mean, enc_logvar                          posterior, [16, L, h, w]
  latent            enc_mean * 0.9152 (what the DiT is conditioned on)
  enc_mean_sliced   the same encode through the reference's causal slicing
                    (split 4: 1+4 frames, then 4 at a time, memory carried)
  noise             [16, L, h, w], torch.randn with --seed
  pred              the DiT's velocity at t = 1000 for [noise, latent, 1]
  block0_vid        block 0's video rows, [L*h/2*w/2, 2560]
  z                 noise - pred
  dec_conv_in, dec_mid, dec_up0..3, decoded     decoder stages on `latent`
  decoded_sliced    the decode of `latent` sliced (1+1 latent, then 1)
  decoded_z         the decode of `z` (the restored clip before colour)
The metadata records the commits, the weights' SHA-256 and torch's version."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seedvr2_ref  # noqa: E402

SCALING = 0.9152


def read_frames(folder, count):
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".png"))[:count]
    if len(names) < count:
        sys.exit(f"{folder} has {len(names)} PNG frames, {count} wanted")
    frames = [np.asarray(Image.open(os.path.join(folder, n)).convert("RGB"), dtype=np.float32) / 255.0
              for n in names]
    x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)   # [3, T, H, W]
    return x * 2.0 - 1.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--frames-count", type=int, default=17)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dit", action="store_true")
    parser.add_argument("--no-stages", action="store_true",
                        help="skip the per-stage tensors (large clips: only the latents and decodes)")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    from models.video_vae_v3.modules.types import MemoryState

    # the VAE first; the 13 GB float32 transformer only for the DiT step
    vae, _, pos, missing_v, _ = seedvr2_ref.load_models(args.model, want_dit=False)
    assert not missing_v
    out = {}
    x = read_frames(args.frames, args.frames_count)
    T, H, W = x.shape[1:]
    assert (T - 1) % 4 == 0 and H % 16 == 0 and W % 16 == 0
    out["x"] = x

    # encoder stages (unsliced, memory DISABLED: the reference's path when the
    # clip is no longer than split + 1)
    captured = {}

    def hook(name):
        def f(_module, _inputs, output):
            if args.no_stages and name != "block0_vid":
                return
            captured[name] = (output[0] if isinstance(output, (tuple, list)) else output).detach().clone()
        return f
    hooks = [vae.encoder.conv_in.register_forward_hook(hook("enc_conv_in"))]
    for i, block in enumerate(vae.encoder.down_blocks):
        hooks.append(block.register_forward_hook(hook(f"enc_down{i}")))
    hooks.append(vae.encoder.mid_block.register_forward_hook(hook("enc_mid")))
    hooks.append(vae.encoder.conv_out.register_forward_hook(hook("enc_h")))
    vae.disable_slicing()
    h = vae._encode(x[None], memory_state=MemoryState.DISABLED)
    for hk in hooks:
        hk.remove()
    for k, v in captured.items():
        out[k] = v[0]
    mean, logvar = h[0].chunk(2, dim=0)
    out["enc_mean"], out["enc_logvar"] = mean, logvar
    latent = mean * SCALING
    out["latent"] = latent
    L = latent.shape[1]

    # the same encode, sliced as the reference does by default (split 4)
    vae.set_causal_slicing(split_size=4, memory_device="same")
    h_sliced = vae.slicing_encode(x[None])
    out["enc_mean_sliced"] = h_sliced[0].chunk(2, dim=0)[0]

    # decoder stages on the reference latent (unsliced)
    captured.clear()
    hooks = [vae.decoder.conv_in.register_forward_hook(hook("dec_conv_in")),
             vae.decoder.mid_block.register_forward_hook(hook("dec_mid"))]
    for i, block in enumerate(vae.decoder.up_blocks):
        hooks.append(block.register_forward_hook(hook(f"dec_up{i}")))
    vae.disable_slicing()
    decoded = vae._decode((latent / SCALING)[None], memory_state=MemoryState.DISABLED)
    for hk in hooks:
        hk.remove()
    for k, v in captured.items():
        out[k] = v[0]
    out["decoded"] = decoded[0]
    vae.set_causal_slicing(split_size=4, memory_device="same")
    out["decoded_sliced"] = vae.slicing_decode((latent / SCALING)[None])[0]

    if not args.no_dit:
        # park the VAE results on disk and free them and the VAE: the float32
        # transformer alone needs about 18 GB
        import gc
        import tempfile
        from safetensors.torch import load_file
        parked = tempfile.NamedTemporaryFile(suffix=".safetensors", dir=os.path.dirname(os.path.abspath(args.out)),
                                             delete=False).name
        save_file({k: v.float().contiguous() for k, v in out.items()}, parked)
        out.clear()
        del vae, h, h_sliced, decoded, mean, logvar, captured
        gc.collect()
        _, dit, _, _, missing_t = seedvr2_ref.load_models(args.model, want_vae=False)
        assert not missing_t
        torch.manual_seed(args.seed)
        noise = torch.randn(latent.shape)
        lat_thwc = latent.permute(1, 2, 3, 0)                         # [L, h, w, 16]
        noise_thwc = noise.permute(1, 2, 3, 0)
        cond = torch.cat([lat_thwc, torch.ones(*lat_thwc.shape[:3], 1)], dim=-1)
        vid = torch.cat([noise_thwc, cond], dim=-1).reshape(-1, 33)
        vid_shape = torch.tensor([list(lat_thwc.shape[:3])])
        txt_shape = torch.tensor([[pos.shape[0]]])
        captured = {}
        hk = dit.blocks[0].register_forward_hook(hook("block0_vid"))
        pred = dit(vid=vid, txt=pos, vid_shape=vid_shape, txt_shape=txt_shape,
                   timestep=torch.tensor([1000.0])).vid_sample
        hk.remove()
        block0 = captured["block0_vid"]
        pred = pred.reshape(*lat_thwc.shape[:3], 16).permute(3, 0, 1, 2).contiguous()
        z = noise - pred
        del dit, captured
        gc.collect()
        vae, _, _, _, _ = seedvr2_ref.load_models(args.model, want_dit=False)
        vae.disable_slicing()
        decoded_z = vae._decode((z / SCALING)[None], memory_state=MemoryState.DISABLED)[0]
        del vae
        gc.collect()
        out = load_file(parked)
        os.unlink(parked)
        out.update({"noise": noise, "block0_vid": block0, "pred": pred, "z": z, "decoded_z": decoded_z})

    meta = {
        "seedvr_commit": os.environ.get("SEEDVR_COMMIT", ""),
        "mflux_mapping": os.environ.get("MFLUX_COMMIT", ""),
        "torch": torch.__version__,
        "seed": str(args.seed),
        "frames": str(T), "height": str(H), "width": str(W), "latent_frames": str(L),
        "sha256": json.dumps({n: sha256(os.path.join(args.model, n))
                              for n in ("vae.safetensors", "transformer.safetensors", "pos_emb.safetensors")}),
        "posterior": "mean (the engine's choice; the reference samples it)",
    }
    save_file({k: v.float().contiguous() for k, v in out.items()}, args.out, metadata=meta)
    print("wrote", args.out, {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
