# Phase-8 scripts (P-H3-Image)

Helpers behind the results in `plans/IMAGE_PLAN.ja.md`, appendix I. They are
tools, not part of the build or of `make test`; outputs go outside the
repository (`~/H3cApp-models/`).

The PyTorch references run in their own Python 3.12 environment with torch,
transformers (5.x, for Qwen3-VL), safetensors, numpy and pillow from PyPI:

    uv venv --python 3.12 <venv>
    uv pip install --python <venv>/bin/python torch transformers safetensors numpy pillow

Run them with `python -I`. Nothing from the diffusers or ComfyUI checkouts is
imported or executed; `ref_vae.py` re-implements the decoder from diffusers'
source (`43e30028`).

| Script | What it does |
| - | - |
| `fetch_tensor.py` | Fetch named tensors of a remote safetensors file by byte range (header from a local copy) and save them as `.npy`: how the P-H3 DiT was compared with FL2VA plus the LoRA, the text encoder with Qwen3-VL-4B, and the VAE with the video VAE (appendix I.1) |
| `ref_text.py` | P-H3-Image's text conditioning as diffusers computes it: transformers' Qwen3VL on the ComfyUI single-file text encoder, `output_hidden_states`, then the text connector |
| `ref_vae.py` | P-H3-Image's VAE decoder (tiled, F32 or `--bf16`) from the ComfyUI single-file VAE, with ComfyUI's weight layout mapped to diffusers' |
| `compare.py` | Relative L2, cosine and per-row figures of one tensor of two files (safetensors or `.npy`); PSNR for images |

`ref_text.py` reads a folder holding the diffusers `text_encoder/config.json`
as `config.json`, the ComfyUI `p-h3-image_text_encoder_bf16.safetensors` as
`model.safetensors`, and `tokenizer.json` / `tokenizer_config.json`.

The engine's sides of the comparisons are `h3_image_text_embed` and
`h3_image_vae_check` (built by `make`) on a model folder laid out as
`transformer/`, `text_encoder/`, `vae/` (with the diffusers `vae/config.json`)
and `tokenizer/tokenizer.json`.
