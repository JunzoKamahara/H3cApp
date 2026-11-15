# P-H3-Image's text conditioning computed as diffusers' P-H3-Image pipeline
# computes it (encoders.get_p_h3_image_prompt_embeds and
# PH3ImageTextConnector), from the ComfyUI single-file text encoder.
#
#   python ref_text.py <ref dir> "<prompt>" <out.safetensors> [--float32]
#        [--image <picture> ...]
#
# With pictures it is an edit, as PH3ImageEditTextEncoderStep presents one:
# `<Picture i>: `, the vision tokens of each picture, then the prompt with
# "Edit <Picture 1>. " in front; the pictures go to the processor as they
# are (give them at their vision size, about a megapixel, so that neither
# side resizes). <ref dir> then also needs the diffusers
# processor/processor_config.json.
#
# <ref dir> holds config.json (the diffusers text_encoder config),
# model.safetensors (p-h3-image_text_encoder_bf16.safetensors, which carries
# the text connector too) and tokenizer.json / tokenizer_config.json.
# Writes "embedding" BF16 [tokens, 5120] (as the pipeline hands it to the
# transformer), "embedding_f32" and "ids".
import sys
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

ref_dir, prompt, out_path = sys.argv[1:4]
dtype = torch.float32 if "--float32" in sys.argv else torch.bfloat16
pictures = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--image"]
taps = (13, 19, 25, 31, 36)

tokenizer = AutoTokenizer.from_pretrained(ref_dir)
vision = {}
if pictures:
    processor = AutoProcessor.from_pretrained(ref_dir)
    images = [Image.open(p).convert("RGB") for p in pictures]
    vision = processor.image_processor(images=images, return_tensors="pt")
    merge = processor.image_processor.merge_size ** 2
    ids = []
    for index, grid in enumerate(vision["image_grid_thw"]):
        ids += tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        ids += ([tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge)
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")])
    if not prompt.lstrip().startswith(("Edit <Picture 1>.", "subject_definitions:")):
        prompt = f"Edit <Picture 1>. {prompt}"
    ids += tokenizer(prompt, add_special_tokens=False)["input_ids"]
    mm_types = torch.tensor(processor.create_mm_token_type_ids([ids]), dtype=torch.long)
    print("grid", vision["image_grid_thw"].tolist())
else:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
print(len(ids), "tokens")

model = Qwen3VLForConditionalGeneration.from_pretrained(ref_dir, dtype=dtype)
model.eval()
connector = {}
with safe_open(f"{ref_dir}/model.safetensors", "pt") as f:
    for name in ("input_mean", "input_std", "proj.weight", "output_mean",
                 "output_std", "sink_embedding"):
        connector[name] = f.get_tensor("text_connector." + name).float()

with torch.no_grad():
    input_ids = torch.tensor([ids])
    extra = {}
    if pictures:
        extra = {"mm_token_type_ids": mm_types,
                 "pixel_values": vision["pixel_values"].to(dtype),
                 "image_grid_thw": vision["image_grid_thw"]}
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    use_cache=False, output_hidden_states=True, logits_to_keep=1, **extra)
    hidden = torch.cat([outputs.hidden_states[k] for k in taps], dim=-1).float()
    x = (hidden - connector["input_mean"]) / connector["input_std"]
    x = x @ connector["proj.weight"].T * connector["output_std"] + connector["output_mean"]
    x = torch.cat([connector["sink_embedding"][None, None], x[:, 1:]], dim=1)[0]

save_file({"embedding": x.to(torch.bfloat16).contiguous(),
           "embedding_f32": x.contiguous(),
           "ids": torch.tensor(ids, dtype=torch.int32)}, out_path)
print("wrote", out_path, tuple(x.shape), "rms %.4f" % x.pow(2).mean().sqrt().item())
