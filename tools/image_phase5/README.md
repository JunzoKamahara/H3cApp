# Phase-5 scripts: editing with reference images

Helpers behind appendix F of `plans/IMAGE_PLAN.ja.md`: the reference
implementation's edit as a baseline, and the comparisons of the engine with
it. They read model files and write outside the repository; nothing here is
part of the build or of `make test`.

The reference implementation is [h3-image-mlx](https://github.com/Anionex/h3-image-mlx)
at `eccdf321`, installed as described in `tools/image_phase0/README.md`. Its
text stage needs the reference text encoder with the vision tower
(`text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`, 27 GB).

| Script | What it does |
| - | - |
| `edit_request.py` | Write the `request.json` of an edit (prompt, size, seed, reference images in order) |
| `edit_stage.py` | Run one h3-image-mlx stage (`text`, `references`, `dit`, `vae`) on a work directory, saving what the engine is compared with: the vision tower's rows per reference and the text positions (`text_trace.safetensors`), the posterior mean and log-variance of each reference (`references_trace.safetensors`), and the DiT trace of `tools/image_phase0/stage_trace.py` plus the augmented reference rows and the segments (`trace.safetensors`) |
| `run_edit.sh` | One edit, stage by stage, keeping every output and each stage's time and peak memory |
| `compare_edit.py text` | The engine's edit text embedding (`h3_image_text_embed … <references>`) against the baseline's: tags row for row, error on vision and language rows apart |
| `compare_edit.py reference` | The engine's reference latent (`h3_image_reference_encode`) against the baseline's posterior mean and its sample |

The engine side:

    make h3_image_text_embed h3_image_reference_encode h3_image_dit_check h3_image_cli
    build/bin/h3_image_reference_encode <model>/vae <image> <out.safetensors> [max short edge] [shaders]
    build/bin/h3_image_text_embed <model> <prompt> <out.safetensors> <shaders> <reference image>...
    build/bin/h3_image_dit_check <model>/transformer <baseline run dir> <out latent> <shaders> forced
    build/bin/h3_image_cli <model> <prompt> <out.ppm> 1024x1024 <seed> <shaders> <reference image>...

`h3_image_dit_check` takes an edit when the baseline's trace has `ref_rows_1`:
it loads the baseline's embedding, tags and augmented reference rows, so the
comparison is of the DiT alone.

The runs of 2026-11-15 are in `~/H3cApp-models/image-phase5/` (outside
`~/Documents`, which is mirrored to a cloud drive on the development
machine): `refs/` (the two reference pictures, the 1024x1024 baselines of
phase 0), `edit1/` (one reference) and `edit2/` (two), each with `upstream/`
(the baseline) and `h3c/` (the engine's outputs and comparison logs).
