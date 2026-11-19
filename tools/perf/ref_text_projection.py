# The video text projection (H3_TEXT_PROJECTION) computed with PyTorch, for
# comparison with build/bin/h3_text_projection_embed: transformers' Qwen3-VL
# on a stock Qwen3-VL-4B, hidden_states[tap], then the ClipProj projection
# as ComfyUI-ClipProj applies it (clipproj_nodes.py):
#
#     rows = ((hidden - mean_in) / std_in) @ W * std_out + mean_out
#     rows[0] = sink_out
#
#   python -I ref_text_projection.py <ref dir> <projection.safetensors>
#        <engine output.safetensors> <out.safetensors> [--float32]
#
# <ref dir> holds config.json (a Qwen3-VL-4B's) and model.safetensors (the
# single-file qwen3vl_4b_bf16.safetensors). The token IDs are the engine's
# own ("ids" of its output), so the two sides read the same sequence.
# Writes "embedding" BF16 [tokens, 5120], "embedding_f32", "hidden" (the
# tapped hidden state, F32) and "ids". --float32 computes the layers in F32
# from the same BF16 weights: what is left between it and the BF16 result is
# BF16 arithmetic alone. Runs in the environment
# tools/image_phase8/README.md describes.
import sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3VLForConditionalGeneration

ref_dir, projection_path, engine_path, out_path = sys.argv[1:5]
dtype = torch.float32 if "--float32" in sys.argv else torch.bfloat16

with safe_open(projection_path, "pt") as f:
    tap = int((f.metadata() or {})["tap"])
    projection = {name: f.get_tensor(name).float() for name in
                  ("W", "mean_in", "std_in", "mean_out", "std_out", "sink_out")}
ids = load_file(engine_path)["ids"].long().tolist()
print(len(ids), "tokens, tap", tap)

model = Qwen3VLForConditionalGeneration.from_pretrained(ref_dir,
                                                        dtype=torch.bfloat16)
# hidden_states[tap] is the input of layer `tap`: the layers after it change
# nothing of it. Dropping them keeps --float32 within a 24 GB Mac. Layer
# `tap` itself stays, or hidden_states[tap] would be the last entry, which
# transformers gives after the final norm.
layers = model.model.language_model.layers
assert tap < len(layers), "tap %d of %d layers" % (tap, len(layers))
del layers[tap + 1:]
model.to(dtype)
model.eval()
with torch.no_grad():
    input_ids = torch.tensor([ids])
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    use_cache=False, output_hidden_states=True, logits_to_keep=1)
    hidden = outputs.hidden_states[tap].float()
    x = (hidden - projection["mean_in"]) / projection["std_in"]
    x = x @ projection["W"] * projection["std_out"] + projection["mean_out"]
    x[:, 0] = projection["sink_out"]
    x = x[0]

save_file({"embedding": x.to(torch.bfloat16).contiguous(),
           "embedding_f32": x.contiguous(),
           "hidden": hidden[0].contiguous(),
           "ids": torch.tensor(ids, dtype=torch.int32)}, out_path)
print("wrote", out_path, tuple(x.shape), "rms %.4f" % x.pow(2).mean().sqrt().item())
