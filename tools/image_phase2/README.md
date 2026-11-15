# Phase-2 checks of the image engine

Phase 2 of `plans/IMAGE_PLAN.ja.md` is the engine side of still images:
the image VAE decoder (`src/model/h3_image_vae.c`), the image DiT (the
existing DiT, taught to run without audio rows and with one output head
per step) and the entry point (`include/h3_image.h`). Each part is checked
against a run of the reference implementation kept by phase 0
(`misc/image-phase0/<prompt>/<source>/<WxH>/`, see `tools/image_phase0`).

The programs are built by the Makefile and are not part of `make test`:
they need the 21 GB checkpoint. Run them from `build/bin` or pass the
shader file (`build/h3_shaders.metal`).

| Program | What it checks |
| - | - |
| `h3_image_vae_check <vae dir> <latent.safetensors> <out.ppm>` | Decodes a reference latent; `image_compare.py` compares the PPM with the reference's PNG |
| `h3_image_dit_check <transformer dir> <run dir> <out latent.safetensors> [shaders] [forced\|free\|both]` | `forced`: the velocity of each step from the reference's own latent before that step. `free`: six steps on from the reference's noise, the latent after each compared, the last written out |
| `h3_image_cli <model dir> <prompt> <out.ppm> [WxH] [seed]` | Text to image through `h3_image_generate`, as the app does |
| `image_compare.py <reference> <other> ...` | PSNR and largest difference of two 8-bit images (Pillow, numpy) |

A model folder for `h3_image_cli` and the app holds `transformer/`, `vae/`
(with `config.json`), `text_encoder/` and `tokenizer/`; links to an
h3-image-mlx download and to a video model's text encoder and tokenizer do.

What to expect (2026-11-15, M5 24 GB; plan appendix C):

- VAE: PSNR above 61 dB against the reference at every size, no value more
  than 3 of 255 apart (the reference computes in half precision).
- DiT, `forced`: relative L2 error of the velocity 1-6 % per step, cosine
  above 0.998. The reference with its activations left unquantized
  (`H3_FLOAT_ACTIVATIONS=1` in `tools/image_phase0/stage_trace.py`) is as
  far from the engine as the stock one, so the difference is the engine's
  own BF16 arithmetic, not the reference's int8 activations.
- DiT, `free`: the final latent 20-40 % from the reference's and a
  different picture of the same scene - the six-step sampler amplifies
  small differences (appendix A.3); this is not a pass/fail figure.
