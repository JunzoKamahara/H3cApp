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

1. **Choose files**: "Add GGUF Files…" adds files; "Add from Folder…" adds a
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
- A model imported without Ref2VA can get it later: Model Manager's "…" →
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

These are sorted by reading the files' headers in the download screens. Every
source has been checked to make a short video with a typical file around Q4
(see "Combinations checked" below).

### Supported formats

- **Quantization**: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K–Q6_K, F16, BF16, F32.
  Mixes such as unsloth's "UD-…_XL" work when made only of these.
- **Unsupported** (listed with the reason):
  - files with IQ types (IQ4_XS, IQ3_XXS, …), I8 and the like
  - split GGUF files (`-00001-of-00002`, …)
  - safetensors DiTs and text encoders (only GGUF can be imported)
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
  so less is read.

For the engine side see [curve-form AdaLN in the README](README.md#curve-form-adaln-experimental-h3_adaln_curve)
and `h3_gguf.c`, `h3_quant.c`, `h3_shaders.metal`.

## API

The app's JSON API (`127.0.0.1:8420`) does the same.

| Endpoint | What it does |
|---|---|
| `GET /api/models/import/downloads?kind=dit&repo=…` | download choices for a part (`dit`, `ref2va`, `textEncoder`, `vision`, `shared`); `repo` takes a name or URL |
| `POST /api/models/import` | imports GGUF files: `files` or `folder`, `base_model` or `base_path`, `name`, `dry_run` |
| `GET /api/models/detail?id=…` | what a model holds (the details screen) |
| `POST /api/models/remove` | unregisters (caches deleted); `delete_files` also moves the files to the Trash, `dry_run` only computes |
| `POST /api/models/add` | registers a folder as a model |
| `POST /api/models/add_ref2va` | adds Ref2VA to a model without it: `file` (the Ref2VA DiT) for a GGUF model, `dry_run` for the plan only (an official model's only as a plan) |

## License

- The MiniMax H3 model (the official one and GGUF conversions such as
  unsloth's) is under the
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
- Qwen's text and image parts (Qwen3-VL-32B-Instruct) are under the
  [Apache License 2.0](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-GGUF).
- Check each source's own license as well. Read each license before
  downloading and using them, and follow their terms.
