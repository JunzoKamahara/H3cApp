# Downloading and importing GGUF (quantized) models

[English](GGUF.md) | [日本語](GGUF.ja.md)

From H3cApp v0.6.0 a quantized model (GGUF) can stand in for the official
BF16 checkpoint (about 134 GiB for FL2VA alone), cutting what you download
and store to about 45 GB with the recommended files, or about 35 GB at the
smallest.

This document covers downloading GGUF files and turning them into a model,
importing GGUF files you already have, and how it works and what to watch for.

- [What comes as GGUF and what doesn't](#what-comes-as-gguf-and-what-doesnt)
- [Way 1: the download wizard (recommended)](#way-1-the-download-wizard-recommended)
- [Way 2: importing GGUF files you have](#way-2-importing-gguf-files-you-have)
- [After importing](#after-importing)
- [Sources and supported formats](#sources-and-supported-formats)
- [Combinations checked](#combinations-checked)
- [HyperFlow H3 Flash](#hyperflow-h3-flash)
- [ComfyUI int8 (ConvRot) DiTs](#comfyui-int8-convrot-dits)
- [Deleting models and disk space](#deleting-models-and-disk-space)
- [How it works](#how-it-works)
- [API](#api)
- [License](#license)

## What comes as GGUF and what doesn't

A MiniMax H3 model has several parts. Only the parts with large weights are
published as GGUF; the rest come from the official model.

| Part | Where it comes from | Typical size |
|---|---|---|
| DiT (FL2VA: from text, first and last images) | GGUF | 21.4 GB at Q8_0, 11.4 GB at Q4_K |
| DiT (Ref2VA: from reference images, video, audio) | GGUF (optional) | about the same as FL2VA |
| Text encoder (Qwen3-VL-32B) | GGUF | 18.2 GB for the H3-specific Q4_K_M, 34.8 GB for Qwen's Q8_0 |
| Vision part (vision tower, mmproj) | GGUF (not needed when the text-encoder file has it) | 1.2 GB at F16 |
| Tokenizer and configs | the official model (no GGUF) | tens of MB |
| Video and audio VAEs | the official model, or unsloth's safetensors (no GGUF) | 11 GB official, 5.9 GB unsloth |

A tokenizer is a few MB of vocabulary, not weights, so there is no GGUF of it.
The VAEs aren't published as GGUF either, but unsloth and Comfy-Org publish them
as single safetensors files (the video VAE in FP16, half the size), which
H3cApp reads too.

## Way 1: the download wizard (recommended)

1. Use "Set Up the MiniMax-H3 Model", shown at first launch. Model Manager's
   "Download and Add…" opens it too.
2. Under "What to download", choose "GGUF (quantized) and needed parts".
3. Pick a source and a file for each part.

   | Part | Recommended | Smaller |
   |---|---|---|
   | DiT (FL2VA) | unsloth's Q8_0 (21.4 GB, tested) | Q4_K (11.4 GB) |
   | Text encoder | unsloth's H3-specific Q4_K_M (18.2 GB, 50 layers, with the image part) | — |
   | VAEs | unsloth's VAEs (5.9 GB, tested) | — |

   - For the most certain text-encoder quality, choose Qwen's own Q8_0 (34.8 GB,
     tested) instead. The import then cuts it to the 50 layers used (about 27 GB
     written), so that much more space is needed; it has no image part, so the
     mmproj (F16, 1.2 GB) is added automatically.
   - To generate with reference images, video or audio too, turn on
     "Include Ref2VA".
4. Choose the "Location" and "Folder name". This becomes the imported model's
   folder, and the downloads are kept inside it (`.downloads`) until imported.
   Even on an external disk, the import clones within that disk, so the space
   isn't doubled.
5. Read the license note and press "Download and Import (… GB)". Download,
   import and registration in Model Manager run on their own. An interrupted
   download resumes next time.

Before you press it, the wizard shows the download size, what the import will
write, and the free space on the destination disk; with too little room (2 GB is
kept in reserve) the button stays off.

### Changing the source

Each part's source (a Hugging Face repository) can be changed.

- The list button offers publishers known to carry that part.
- The download-arrow button (or Return in the field) reads the first bytes of
  every file in the source and lists that part's files by what is inside them,
  not by their names, so each publisher's naming doesn't matter. What was read
  is kept, so the next time is instant.
- List button → "Custom…" takes a repository name or a Hugging Face URL: a
  repository (`https://huggingface.co/owner/name`), a folder
  (`…/tree/main/vae`), one file (`…/blob/main/xxx.gguf`,
  `…/resolve/main/xxx.gguf`), on `main` or another branch. Custom places stay
  in that part's list and can be removed with "Remove Custom".
- The source chosen is remembered per part.

Files the app can't use are still listed, with "(unsupported: reason)", and
can't be picked; see [Supported formats](#supported-formats).

## Way 2: importing GGUF files you have

With GGUF files already on disk, use Model Manager's "Import GGUF…".

1. **Choose files**: "Add Model Files…" adds files; "Add from Folder…" adds a
   folder's (three levels down, following linked folders). Each file is
   inspected and shown as a DiT, text encoder or vision part. FL2VA and Ref2VA
   DiTs have the same tensors, so a file whose name contains `ref2va` is taken
   as Ref2VA; you can switch it. "Parts needed" / "Missing parts" list where to
   get the rest, with links and "Download…" buttons.
2. **Review**: which file serves each part and what is missing.
   - With several files for one part, pick the one to use. Links to the same
     file, and files of the same size and type, are pointed out.
   - Files that won't be used say why.
   - "Clone parts from" chooses where the tokenizer, VAEs and configs come
     from: a model in Model Manager, a folder picked with "Register Folder…",
     or the official shared parts fetched with "Download Shared Parts…"
     (about 6–11 GB).
   - What the import will write (such as the trimmed text encoder) and the
     free space are shown.
3. **Import**: creates the model folder in
   `~/Library/Application Support/H3cApp/Imported/` and registers it.

## After importing

- Model Manager's info button shows what the model holds (each part's format
  and size, AdaLN form, layers, caches, the import record) and what it can do.
- For the fast M5 mode, build the int8 cache (about 19 GB) with
  "Build Cache…" in Advanced Settings; GGUF models can have one too. M1–M4 run
  with SSD streaming, which reads GGUF weights still quantized and expands them
  on the GPU, so less is read.
- A GGUF DiT with curve-form AdaLN carries the conditioning approximation
  itself; no separate curve file is needed.
- From v0.6.1, a model imported without Ref2VA can get it later: Model Manager's "…" →
  "Add Ref2VA…" downloads a Ref2VA DiT GGUF from Hugging Face or takes a file
  you have. The text encoder, VAEs and configs are cloned from the model's
  FL2VA, so only the DiT takes space.

## Sources and supported formats

### Known sources

| Source | Parts | Notes |
|---|---|---|
| [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) | DiT (FL2VA, Ref2VA), H3-specific text encoder, VAEs | the default; DiTs have curve-form AdaLN |
| [Qwen/Qwen3-VL-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF) | text encoder (64 layers), mmproj | official; cut to 50 layers when imported |
| [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) | DiT | curve-form AdaLN |
| [Abiray/MiniMax-H3-GGUF](https://huggingface.co/Abiray/MiniMax-H3-GGUF) | DiT, text encoder | DiTs have dense AdaLN |
| [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF) | DiT, text encoder | both curve-form and dense |
| [molbal/MiniMax-H3-GGUF](https://huggingface.co/molbal/MiniMax-H3-GGUF) | DiT | Q8_CR and U16G unsupported |
| [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) | DiT, text encoder | DiTs have dense AdaLN (the same weights as Abiray's) |
| [joeygambino/MiniMax-H3-x-Z-Image-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-GGUF) | DiT | a merge with Z-Image; `curve` files are curve-form |
| [nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF](https://huggingface.co/nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF) | text encoder (50 layers), mmproj | IQ types unsupported |
| [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | VAEs | the int8 one is unsupported |
| [VideoRebirthAI/hyperflow-h3-flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) | DiT (`fl2va_gguf/`, `ref2va_gguf/`) | HyperFlow H3 Flash (section below); the int8_convrot files are unsupported |

These are sorted by reading the files' headers in the download screens. Every
source has been checked to make a short video with a typical file around Q4
(see "Combinations checked" below).

### Supported formats

- **Quantization**: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K–Q6_K, F16, BF16, F32.
  Mixes such as unsloth's "UD-…_XL" work when made only of these.
- **Unsupported** (listed with the reason):
  - files with IQ types (IQ4_XS, IQ3_XXS, …), I8 and the like
  - split GGUF files (`-00001-of-00002`, …)
  - safetensors DiTs and text encoders (only GGUF and
    [a DiT in ComfyUI's int8 format](#comfyui-int8-convrot-dits) can be imported)
  - VAEs in number formats the app can't read (F8, int8, …)
  - text encoders with fewer than 50 layers
- **AdaLN form**: DiT GGUFs come with curve-form AdaLN (ComfyUI's "pruned"
  form, about 24 GiB of weights replaced by a small curve) or dense AdaLN.
  Both work; dense files are about 8 GB larger. Tensors that ComfyUI-GGUF
  reshaped to fit whole quantization blocks (`comfy.gguf.orig_shape`) are read
  back in their original shape.
- **Text encoder**: the engine uses the first 50 layers of Qwen3-VL-32B. A
  64-layer file is cut to those 50 when imported; an H3-specific (50-layer) file
  is used as is.

## Combinations checked

| Combination | What was checked |
|---|---|
| DiT unsloth Q8_0 + text encoder Qwen Q8_0 (cut to 50 layers) + mmproj F16 | values against the official weights (errors at quantization level), generation |
| DiT unsloth Q4_K + unsloth H3-specific text encoder Q4_K_M + unsloth VAEs | download, import, generation |
| unsloth's VAEs (video FP16, audio) | the same latents decoded with the official VAEs: video PSNR 51.7 dB on average, audio SNR 72 dB (indistinguishable) |
| unsloth's H3-specific text encoder | tensor names and shapes match the official weights; relative error 7% at Q4_K for the language part, the image part bit-exact |

For each known source, a DiT and a text encoder from that source at the same
quantization (as far as it offers them) made a 3-second 512×512 video with the
same prompt and seed using SSD streaming, and the picture and the audio (speech)
were checked (October 19, 2026, Apple M5, 24 GB). Sources without a text encoder
were paired with unsloth's Q4_K_M; text-encoder-only sources with unsloth's Q4_K
DiT.

| Source | DiT | Text encoder |
|---|---|---|
| unsloth | `minimax_h3_fl2va_pruned-Q4_K` (curve-form) | `qwen3vl_32b_minimax_h3-Q4_K_M` |
| leejet (pruned) | the same file as unsloth's | the same file as unsloth's |
| leejet | `minimax_h3_fl2va-Q4_K_M` (dense) | the same file as unsloth's |
| Abiray | `MiniMax-H3-FL2VA-Q4_K_M` (dense) | `qwen3vl_32b_minimax_h3-Q4_K_M` |
| Abiray (Pruned) | `MiniMax-H3-FL2VA-Pruned-Q4_K_M` (curve-form) | Abiray's `Q4_K_M` |
| realrebelai | `MiniMax-H3-FL2VA-Q4_K_M` (dense) | `qwen3vl-32B-MiniMax-H3-Q4_K_M` |
| molbal | `minimax_h3_fl2va_pruned_fp8_Q4_0` (curve-form) | unsloth's `Q4_K_M` |
| joeygambino | `MiniMax-H3-fl2va-curve-zs05-Q4_0` (curve-form) | unsloth's `Q4_K_M` |
| Qwen | unsloth's `Q4_K` | `Qwen3VL-32B-Instruct-Q4_K_M` (cut to 50 layers) + mmproj F16 |
| nif0 | unsloth's `Q4_K` | `…-L0-49-Q4_K_M` + mmproj BF16 |

The image part (mmproj) is only used when generating with reference images, so
this check covers loading it.

A plain Q2_K DiT (unsloth's `minimax_h3_fl2va_pruned-Q2_K`) makes broken videos
(melted figures, a grid-like background, humming audio): its block matrices are
all Q2_K, 27.7% relative error against Q8_0 (15.6% for Q3_K, 7.4% for Q4_K).
Use Q3_K_M or better; H3cApp marks such files with a warning. Q2_K text encoders
work, and mixes such as "UD-Q2_K_XL" are untested.

## HyperFlow H3 Flash

[HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) (Video Rebirth)
is a derivative of MiniMax H3 whose DiT is distilled and compressed for few
steps: FL2VA has 13.7B parameters (47 blocks, 18 of them with low-rank
factored linears), Ref2VA 13.9B (49 blocks). The text encoder and the VAEs
are MiniMax H3's own. H3cApp supports the GGUF files (`fl2va_gguf/`,
`ref2va_gguf/`).

- **Importing**: like any other DiT, through the wizard or from GGUF files
  you have. The shared parts and the text encoder are cloned from an
  imported MiniMax-H3 model (the DiT alone is about 9 GB at Q4_K_M). Use
  HyperFlow for both FL2VA and Ref2VA (the import stops on a mix).
- **Generating**: 4, 6 or 8 steps only (6 by default), with the model's own
  sigma schedule and Euler, no CFG. The speed presets, denoiser reuse, the
  layer count, add-on models (LoRA), re-noise and Open-H3 regeneration do
  not apply (hidden in the advanced settings; the API answers `400` with the
  reason). They come back when a regular MiniMax-H3 model is active.
- **Compute mode**: no int8 cache. With the "int8 cache" choice it runs
  resident (int8) on an M5 with at least 24 GB, and with SSD streaming
  elsewhere.
- **Records**: the video's record (`.json`) has `modelFamily: "hyperflow"`
  and its XMP `AISystemUsed` says "HyperFlow H3 Flash (MiniMax H3 derivative)".

Measured on an M5 (24 GB), 672x384, 5 s, 6 steps (2026-11-09):

| Mode | Per step | Whole run | Peak memory |
|---|---|---|---|
| HyperFlow, resident (int8) | about 30 s | 256 s (309 s in the app) | 16.5 GiB |
| HyperFlow, SSD streaming | about 39 s (7.8 GiB read per step) | 303 s | 8.5 GiB |
| Regular H3 GGUF (Q4_K_M), int8 cache | about 44 s | 327 s | 15.7 GiB |
| Regular H3 GGUF (Q4_K_M), SSD streaming | about 52 s (10.1 GiB) | 378 s | 8.9 GiB |

Against the reference implementation (ComfyUI on CUDA), block by block,
the final output differs by 3-8 % resident (int8) and 2-5 % with SSD
streaming (BF16).

Picture quality (experimental support): with the same three prompts and
seeds, compared with the Turbo LoRA (4 steps), DMAD and regular 20-step runs
(M5, 672x384, 5 s, 2026-11-09). HyperFlow at 6 and 8 steps matches Turbo at
4 steps, follows the prompt slightly better (CLIPScore) and flickers about
half as much between frames. At 6 steps fine fur can come out a little
softer than with Turbo. It takes about as long as Turbo at 4 steps, about
1.2 times as long at 6 and about 1.45 times at 8.

| Configuration (mean of 3 prompts) | Time | MUSIQ | CLIPScore | Flicker |
|---|---|---|---|---|
| Regular, 20 steps | 645 s | 64.6 | 0.914 | 0.82 |
| Turbo LoRA, 4 steps | 182 s | 68.1 | 0.858 | 1.97 |
| DMAD LoRA, 4 steps | 182 s | 64.3 | 0.831 | 1.40 |
| HyperFlow, 4 steps | 170 s | 65.9 | 0.866 | 0.96 |
| HyperFlow, 6 steps | 216 s | 67.5 | 0.877 | 1.13 |
| HyperFlow, 8 steps | 264 s | 68.7 | 0.853 | 1.22 |

MUSIQ estimates picture quality (higher is better); flicker is the mean
brightness change between neighbouring frames. The differences between
prompts are larger than those between configurations, so read these as a
rough guide.

Ref2VA (Q4_K_M, 9.22 GB) was also checked: it generates from a reference
image, a reference video, and a reference image plus reference audio
(2026-11-10, M5 with 24 GB, 672x384, 5 s, one clip per kind of reference).

| Reference | Regular model, 20 steps | HyperFlow, 6 steps | HyperFlow peak memory |
|---|---|---|---|
| One image | 638 s | 225 s (4 steps 184 s, 8 steps 280 s) | 17-18 GiB |
| One image + audio | 698 s | 249 s | 17 GiB |
| One video | 2,285 s | 1,180 s | 19 GiB |

By eye, the person in the reference was kept by both models, and the pitch of the
reference voice was followed by both. HyperFlow's Ref2VA sound comes out
about 5 dB louder than the regular model's; listening to every clip, the
voice in the one with reference audio was slightly distorted (just about
acceptable). With a reference video, both models made the speech fast,
but not to the point of being a problem. Ref2VA also works with SSD
streaming for all three kinds of reference (measured on the M5 with the
compute mode set to SSD streaming, 6 steps): 305 s with an image, 324 s
with an image and audio, 1,426 s with a video (1.2-1.35 times resident),
with a peak of 6.8-9.3 GiB. Resident (int8) and SSD streaming (BF16)
compute at different precision, so with the same seed the background
details change; the person and the framing stay the same. Emulating an M1-M4 on the M5
(TensorOps and the newer GPU features turned off), all three kinds also
work with a peak of 10-11 GiB; denoising takes about as long as on the M5
and the VAE decode about three times as long (425 s in all with an image
reference; these are the M5 GPU's times, not a real M1-M4's). Actual
M1-M4 Macs have not been tried yet.

## ComfyUI int8 (ConvRot) DiTs

An int8-quantized DiT distributed for ComfyUI can be imported too: a single
`.safetensors` file that holds, for each linear layer, an int8 weight
(`…weight`), a scale per row (`…weight_scale`) and a JSON describing the
format (`…comfy_quant`). Files with "ConvRot" - the weight's columns rotated
256 at a time by a Hadamard matrix before quantizing - are read as well.
This is the only safetensors layout the import takes, and only for the DiT.

- **Importing**: the same steps as a GGUF. Pick the file with "Add Model
  Files…" or "Add from Folder…", or point the wizard's source at the
  repository. The format shows as "INT8 ConvRot" ("INT8" without the
  rotation). The text encoder, VAEs and configs come from other files or the
  base model, as with a GGUF.
- **One file for FL2VA and Ref2VA**: a merge whose metadata
  (`merge_strategy`) names both FL2VA and Ref2VA is placed as both
  transformers (a clone on the same disk, so no extra space). A file picked
  for Ref2VA takes precedence.
- **How it is read**: as it loads, the engine un-rotates the int8 values and
  multiplies each row by its scale, restoring the plain weight. From there it
  computes exactly as for an ordinary MiniMax H3: the int8 cache (M5) and
  SSD streaming were checked; add-on models (LoRAs) and the speed settings
  go through the same code but were not tried with this format. SSD
  streaming reads the int8 values as they are and restores them on the GPU,
  so a step reads half of what BF16 would (about 19 GB).
- **The step count does not change by itself.** For a model with few-step
  distillation merged in, set the steps it was made for in Advanced Settings.

**Checked** (M5, 24 GB, 672×384, 5 s): the int8 file of
[OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD](https://huggingface.co/OliviaRossi/Minimax-H3-Hybrid-DMAD-PDMD)
(`Minimax-H3-Hybrid-DMAD-PDMD_int8.safetensors`, 20.3 GB, revision
`474da26c`) - a merge that blends the FL2VA and Ref2VA AdaLN and folds the
DMAD (8-step) and PDMD LoRAs into the weights. Generated with no extra
LoRAs, 6 steps, Euler, audio shift 3.

| Compute mode | Text to video | Reference image (Ref2VA) | Peak memory |
|---|---|---|---|
| int8 cache (M5) | 251 s | 261 s | 4.5-6.1 GiB |
| SSD streaming | 400 s | 418 s | 6.7 GiB |

- Decoding: the engine's result matches a separately written numpy
  calculation (no difference on four matrices), and the GPU decoding
  used by SSD streaming matches the CPU decoding bit for bit (unit test). All
  208 matrices are within 7-8.4% of the FL2VA Q4_K weights (a wrong rotation
  would give about 140%).
- A dark Ref2VA scene was compared on two seeds against the ordinary model
  at 20 steps and the ordinary model with the DMAD and PDMD LoRAs stacked at
  6 steps. The merge rendered the face brighter and more evenly lit and
  closest to the person in the reference image; it also followed the
  prompt's "dim room lit only by a small lamp" the least (its background
  drifts toward the reference photo's room). What causes the difference was
  not isolated. The audio has not been checked by listening.
- Limits: HyperFlow's int8 (safetensors) files, other quantizations such as
  W6A8, and safetensors text encoders are not supported. A ConvRot group must
  be a power of 4 (up to 256 for SSD streaming). Not yet checked on a real
  M1-M4 Mac, at high resolution (1344×768) or at 8 steps.

## Deleting models and disk space

A GGUF import places files as APFS clones on the same disk, so importing takes
almost no space; in exchange, models can share data with each other and with
the downloaded originals.

Model Manager's "…" → "Delete…":

- Even when only unregistering, the model's generation caches (int8 cache and
  the like) are deleted right away (they can be rebuilt).
- With "Also delete the model files", the model folder goes to the Trash, after
  one more confirmation. The space emptying the Trash will free is computed
  from APFS's clone tracking; data shared with other models or originals stays,
  as those still use it.
- Deleting the last model also deletes the downloaded shared parts and GGUF
  files, and caches no model uses.

## How it works

- **Download**: straight from Hugging Face, resumable per file. Source listings
  read each file's header with HTTP ranges (a GGUF's key/values and tensor
  table, a safetensors file's JSON) and sort the files with the engine's own
  inspection.
- **Import**: builds a model folder laid out like the official one (`FL2VA/`,
  and `Ref2VA/` if any) from the DiT, text-encoder and vision GGUFs plus the
  base's tokenizer, VAEs and configs, cutting the text encoder to the 50 layers
  used. The import is recorded in `h3c-import.json`.
- **Generation**: quantized GGUF weights are expanded to BF16 on the GPU (Metal)
  when used and computed with the existing matrix kernels; matrices are not
  multiplied in the quantized form. SSD streaming reads them still quantized,
  so less is read. A ComfyUI int8 DiT is restored to plain weights as it
  loads, on the CPU (on the GPU for SSD streaming).

For the engine side see [curve-form AdaLN in the technical reference](TECHNICAL.md#curve-form-adaln-experimental-h3_adaln_curve)
and `h3_gguf.c`, `h3_quant.c`, `h3_shaders.metal`.

## API

The app's JSON API (`127.0.0.1:8420`) does the same; each request needs
the API token (see [API](API.md#api)).

| Endpoint | What it does |
|---|---|
| `GET /api/models/import/downloads?kind=dit&repo=…` | download choices for a part (`dit`, `ref2va`, `textEncoder`, `vision`, `shared`); `repo` takes a name or URL. `403` when a web page on another site sends it (a foreign `Origin`, or `Sec-Fetch-Site: cross-site` / `same-site`); scripts and `curl` are unaffected |
| `POST /api/models/import` | imports GGUF files (and a DiT in ComfyUI's int8 format): `files` or `folder`, `base_model` or `base_path`, `name`, `dry_run` |
| `GET /api/models/detail?id=…` | what a model holds (the details screen), plus its `family` (`h3` or `hyperflow`) and the `steps` it takes |
| `POST /api/models/remove` | unregisters (caches deleted); `delete_files` also moves the files to the Trash, `dry_run` only computes |
| `POST /api/models/add` | registers a folder as a model |
| `POST /api/models/add_ref2va` | adds Ref2VA to a model without it: `file` (the Ref2VA DiT) for a GGUF model, `dry_run` for the plan only (an official model's only as a plan) |

## License

- The MiniMax H3 model (the official one and GGUF conversions such as
  unsloth's) is under the
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
- HyperFlow H3 Flash (Video Rebirth) is distributed as a Model Derivative
  of MiniMax H3 under the same MiniMax H3 Community License Agreement. It
  is not an official MiniMax product.
- Merges and quantized conversions made from the MiniMax H3 weights
  (including ones in ComfyUI's int8 format) are Model Derivatives of
  MiniMax H3 as well. Check the license their source states and those of
  anything merged in, such as LoRAs.
- Qwen's text and image parts (Qwen3-VL-32B-Instruct) are under the
  [Apache License 2.0](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF).
- Check each source's own license as well. Read each license before
  downloading and using them, and follow their terms.
