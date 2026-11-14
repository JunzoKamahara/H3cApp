# Phase-0 scripts for the still-image plan

Helpers behind the results in `plans/IMAGE_PLAN.ja.md`, appendix A. They read
model files and write under `misc/` (not tracked); nothing here is part of
the build or of `make test`.

The reference implementation is [h3-image-mlx](https://github.com/Anionex/h3-image-mlx)
at `eccdf321` (0.1.0a5), installed into its own Python 3.12 environment:

    uv venv --python 3.12 <venv>
    uv pip install --python <venv>/bin/python <h3-image-mlx checkout>

| Script | What it does |
| - | - |
| `ranges.py`, `standin.py` | Build a sparse stand-in of a remote safetensors file: real header and small tensors, holes for the large block matrices (appendix A.1) |
| `heads.py` | Compare ComfyUI's stacked `final_layer.video_out` with the released six PDD heads |
| `int8_compare.py` | Compare a weight as the engine restores it (`h3_image_dit_probe … dump`) with the reference reading of the same int8 + ConvRot bytes |
| `vae_encoder_compare.py` | Compare the image VAE's tensors with the video VAE's, by name |
| `tok.py`, `ids_check.py` | Token ids of the prompts from the reference tokenizer, against the ids in an `h3_image_text_embed` output |
| `stage_trace.py` | Run one h3-image-mlx stage on a work directory; the DiT stage also saves `trace.safetensors` (initial noise, sigmas, RoPE angles, refined text, AdaLN values of blocks 0 and 49 and the final layer, the latent rows and the head output after each step). `H3_FLOAT_ACTIVATIONS=1` computes the int8 projections without quantizing the activations |
| `request.py` | Write the `request.json` a stage reads |
| `run_text.sh` | The reference text stage on each prompt: `embedding.upstream.safetensors` (or `embedding.upstream-float.safetensors` with `H3_FLOAT_ACTIVATIONS=1`) |
| `run_sizes.sh` | For one prompt, one embedding source and several sizes: DiT and VAE stages, keeping every stage output and its time and peak memory |
| `embed_compare.py` | Compare two text embeddings of one prompt: relative L2 error and cosine, whole and per token |
| `run_compare.py` | Compare two sets of runs that differ in the embedding source: refined text, latent after each step, final latent, image PSNR |
| `sheet.py`, `summary.py` | Contact sheets and the table of a set of runs (pass `<prompt>/<source>` as the name) |

Outputs are in `misc/image-phase0/<prompt>/`:

- `embedding.h3c.safetensors` from `h3_image_text_embed` (the engine's text
  path, Q4_K_M GGUF), `embedding.upstream.safetensors` from the reference
  text stage (int8 + ConvRot weights, int8 activations) and
  `embedding.upstream-float.safetensors` (same weights, float activations).
- `<source>/<WxH>/` for the sources `upstream` (2026-11-15, the baseline) and
  `h3c` (2026-11-14): `request.json`, `embedding.safetensors`,
  `trace.safetensors`, `latent.safetensors`, `image.png`, `profile.dit.json`,
  `time.*.txt`.

The model files (46 GB) are in `~/H3cApp-models/MiniMax-H3-Image/`, in the
layout h3-image-mlx expects (`diffusion_models/`, `text_encoders/`, `vae/`,
`tokenizer.json`). They are kept outside `~/Documents` on purpose: that
folder is mirrored to a cloud drive on the development machine. The
`request.json` files written before 2026-11-15 still name the earlier
location, `misc/models/MiniMax-H3-Image/`.
