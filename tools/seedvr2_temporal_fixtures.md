# Temporal SeedVR2 reference fixtures (manifest)

Reference tensors for `tests/test_seedvr2_temporal.c`, written by
`tools/dump_seedvr2_temporal_fixture.py`. Development only: none of this is
part of the app, which has no Python.

## What the reference is

- **Code:** the official ByteDance-Seed/SeedVR repository at
  `e4de8c24441a67e1b7df56abea10645059bb1185` (`models/video_vae_v3`,
  `models/dit_v2`, configs `configs_3b/main.yaml`), run on the CPU in float32.
- **Stand-ins** (`tools/seedvr2_ref_stubs/`): `flash_attn.flash_attn_varlen_func`
  as one float32 SDPA per `cu_seqlens` segment, and apex's `FusedRMSNorm` /
  `FusedLayerNorm` as float32 PyTorch modules. The official attention casts
  q/k/v to bf16 before flash attention; the stand-in computes in float32.
- **Weights:** the MLX int8 export the app installs
  (`~/Library/Application Support/H3cApp/Upscalers/SeedVR2`), dequantised
  (`w = scale * q + bias` per group of 64), renamed through mflux's
  `seedvr2_weight_mapping.py` (mflux `ab162f3b99bdf1897ab42574fc52f3bd8cba92d5`),
  the VAE's conv kernels transposed `(O, kt, kh, kw, I) -> (O, I, kt, kh, kw)`.
  The transformer's 2-D weights (other than `emb_in`) and the positive text
  embedding are rounded to bf16, as the engine holds them. Every parameter of
  both models is filled (`load_models` asserts that none is missing), so the
  reference and the engine run the same effective weights and differences are
  implementation differences, not quantisation.
  SHA-256 (recorded per file in the safetensors metadata):
  `transformer.safetensors` 4f0d2b37…, `vae.safetensors` a2efbfb3…,
  `pos_emb.safetensors` d2e06cad….
- **RoPE patch:** `NaMMRotaryEmbedding3d.get_freqs` builds its table with
  `get_axial_freqs(l + f, h, w)` sliced to `[l : l + f]` instead of the full
  cache (same values, far less memory on the CPU).
- **conv3d patch:** `F.conv3d` runs as one `conv2d` per temporal tap, summed
  (the CPU's conv3d unfolds the clip: 33 GB at 17 frames of 256²). Equal up
  to the order of the float32 sums (≤ 3e-5 on random tensors). Used for the
  T = 17 files; T = 1, 5, 9 were written with PyTorch's own conv3d.

## Inputs

`~/Documents/work/h3c/seedvr2-temporal/`:

- `frames-256/001.png …` 256×256 RGB, from an app generation
  (`20261104-134101_seed203.mp4`, an anime runner), scaled to 448×256 and
  centre-cropped. Fast motion, flat areas and hair.
- `frames-480x352/` 17 frames at 480×352 (latent 60×44, the shape of
  mlx-gen's period-4 corruption report 0115).

Input in `[-1, 1]` (`x * 2 - 1`), no resize: the fixtures test the model, not
the app's resize.

## Files (`fixtures/`)

| File | Frames | Size | DiT |
|---|---|---|---|
| `sv2t_T1_256.safetensors` | 1 | 256×256 | yes |
| `sv2t_T5_256.safetensors` | 5 | 256×256 | yes |
| `sv2t_T9_256.safetensors` | 9 | 256×256 | yes |
| `sv2t_T17_256.safetensors` | 17 | 256×256 | yes |
| `sv2t_T17_480x352_vae.safetensors` | 17 | 480×352 | no (`--no-dit --no-stages`: latents and decodes only) |

Tensors are float32 in the reference's `[C, T, H, W]` layout:

| Name | Shape (T=5, 256²) | Meaning |
|---|---|---|
| `x` | 3, 5, 256, 256 | input |
| `enc_conv_in`, `enc_down0..3`, `enc_mid`, `enc_h` | e.g. 256, 3, 64, 64 for `enc_down1` | encoder stages, whole clip, memory disabled |
| `enc_mean`, `enc_logvar` | 16, 2, 32, 32 | posterior |
| `latent` | 16, 2, 32, 32 | `enc_mean × 0.9152` (the engine takes the mean, the reference samples) |
| `enc_mean_sliced` | 16, 2, 32, 32 | the same encode through the reference's causal slicing (split 4: 1+4 frames, then 4) |
| `dec_conv_in`, `dec_mid`, `dec_up0..3` | e.g. 512, 3, 64, 64 for `dec_up0` | decoder stages on `latent`, whole |
| `decoded` | 3, 5, 256, 256 | decode of `latent`, whole |
| `decoded_sliced` | 3, 5, 256, 256 | decode of `latent`, sliced (1+1 latent frames, then 1) |
| `noise` | 16, 2, 32, 32 | `torch.randn`, seed 42 (re-seeded right before the draw) |
| `pred` | 16, 2, 32, 32 | the DiT's velocity at t = 1000 for `[noise, latent, 1]` |
| `block0_vid` | L·h/2·w/2, 2560 | block 0's video rows |
| `z` | 16, 2, 32, 32 | `noise − pred` |
| `decoded_z` | 3, 5, 256, 256 | decode of `z`, whole (before colour) |

Frame counts per stage for T = 4n+1: `enc_down1` 2n+1, `enc_down2..` n+1 = L;
`dec_up0` 2L−1, `dec_up1..` 4L−3 = T.

## Tolerances (`tests/test_seedvr2_temporal.c`)

The engine runs in bf16 (grid path) or F32 with bf16 storage, so it is
compared by relative RMS error: encoder stages before the output < 0.2,
latent / prediction / decode < 0.05, decode of the engine's own z < 0.1
(the same limits as `tests/test_seedvr2.c` against mflux). Exact equalities
(sliced vs whole in the engine itself) are tested separately.

## Memory measurements of the clip pipeline

`seedvr2_cli --video-frames` with `H3_SV2_TRACE=1` prints each phase's peak
live GPU bytes; `/usr/bin/time -l` the process footprint. The table that
`h3_seedvr2_video_check_requirements` is fitted to is in
`h3_seedvr2_video.c` (M5, 24 GB, 2026-11-06). Run one configuration at a
time and only ones the estimate allows: before the memory policy a
17-frame 1344x768 clip reached 36 GiB and swapped.

## Regenerating

**Run a dump on its own.** The CPU reference takes 15-20 GB (about 30 GB
for 480×352 with the stage tensors, hence `--no-stages` there); on a 24 GB
Mac a dump next to a GPU test ran out of swap and the machine hit a
watchdog panic (2026-11-05).

Needs Python 3.12 with torch, diffusers, einops, omegaconf,
rotary-embedding-torch, safetensors and Pillow, the SeedVR checkout and
mflux's mapping file; about 20 GB of memory for T = 17 at 256² (the VAE
and the 13 GB float32 transformer are loaded one after the other).

```bash
SEEDVR_REPO=~/src/SeedVR MFLUX_MAPPING=~/src/mflux/src/mflux/models/seedvr2/weights/seedvr2_weight_mapping.py \
SEEDVR_COMMIT=e4de8c2 MFLUX_COMMIT=ab162f3 \
python -I tools/dump_seedvr2_temporal_fixture.py \
    --model "$HOME/Library/Application Support/H3cApp/Upscalers/SeedVR2" \
    --frames ~/Documents/work/h3c/seedvr2-temporal/frames-256 --frames-count 17 \
    --out ~/Documents/work/h3c/seedvr2-temporal/fixtures/sv2t_T17_256.safetensors
```
