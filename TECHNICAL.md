# H3cApp technical reference

[English](TECHNICAL.md) | [日本語](TECHNICAL.ja.md)

Technical details and measurements moved from the v0.6.4 README. Times are examples, not performance guarantees. First-time users should start with the [README](README.md).

Related documentation: [Local API](API.md) · [Building and testing](BUILD.md) · [GGUF models](GGUF.md)

## Features

- **GGUF (quantized) models** (v0.6.0): download a quantized DiT and text
  encoder from Hugging Face (unsloth, Qwen and other publishers, or any
  repository or URL) or import GGUF files you have, and use them like the
  official model — see [GGUF.md](GGUF.md).
- **Text-to-video+audio** (T2VA): a prompt alone produces a synchronized
  HEVC (or H.264) + AAC clip.
- **First/last-frame conditioning** (FL2VA): anchor a generation's opening
  and/or closing frame (the app's first/last-image mode, or the API's
  `first_frame_path`/`last_frame_path`).
- **Reference conditioning** (Ref2VA), ordered and mixable: images, videos
  (with or without embedded audio), and standalone audio files as
  style/content references (the app's reference-image/video/audio mode, or
  the API's `reference_paths`). An audio file can't be the only reference —
  it needs at least one image or video reference alongside it. References
  appear to the model in order as `<Picture N>`/`<Video N>`. All reference
  reads and generated-video writes go through native AVFoundation/ImageIO.
- **Lip sync** (forced audio): choose a voice or song in the panel (Match
  a Voice or Song, with a start offset) and the picture is generated to
  match it - the audio is given to the model clean at every step and the
  video carries the file's own sound (API: `forced_audio_path`). Best at
  medium size or larger: at the small 448x256 size it makes colored light
  blobs more likely.
- **Continuing a video**: Continue This Video (in Project Videos, or for
  the video in the preview) makes a new clip that carries on from a
  project video - its last 22 frames (about 0.9 s) and their sound are
  held at the start of the new clip, the way ComfyUI's H3 Motion Context
  node does it (see [How joined generation works](#how-joined-generation-works)) - and Join Up to Here and Export cross-fades a chain of
  them into one video in the project, listed with its clips folded under
  it. With lip sync, each
  continuation takes the audio on from where the last one ended, for long
  lip-synced videos. Every generated video keeps its latents next to it
  (`.h3lt`, about 3.6 MB for 5 s at medium size) for this. A continuation
  takes about 20% longer than a new clip (446 s vs 364 s for 5 s at medium
  size on a 24 GB M5); lip sync adds almost nothing, and joining three
  clips takes under a second. The panel offers Continue once a video is
  in the preview (a video outside any project is saved as one first), and
  Join and Export when that video continues another.
- **Joined generation**: the length menu has 1-15 s in one generation, then
  15 s Joined and Custom Clips…. 15 s Joined is made of clips of about 7 s
  (or 5 s, in Advanced Settings), Custom Clips of clips whose length,
  directions, images and lip-sync audio are set one by one; each clip
  continues the one before, and they are joined automatically over their
  overlaps; the joined video is the one in the project's list, with
  its clips folded under it (kept in `chains/<name>/`), and it can be
  continued like any other. With lip sync the audio moves on clip by clip;
  clips that start past its end make their own sound. 15 s this way took
  about 23-25% less time than one 362-frame generation (the longest one
  generation can make) at medium size; a joined length costs about 105 s
  per second of video at medium size on a 24 GB M5. See
  [Joined generation: how to use it](GUIDE.md#joined-generation-how-to-use-it).
- **Memory-aware reference-video sizing**: a large reference video is
  automatically downscaled based on the machine's physical RAM
  (`h3_reference_max_pixels()` in [h3_host.c](h3_host.c)) before it reaches
  the model, rather than always targeting the same fixed cap regardless of
  what the machine can hold. See [Reference-conditioning cost](#reference-conditioning-cost-why-a-bigger-reference-is-not-free) below for why this matters.
- **Stackable LoRAs**: the model manager keeps a library of LoRA files;
  any of them (up to 8) can be switched on at once, each with its own
  strength, and they add up. Every published H3 layout loads as-is —
  diffusers/PEFT (`to_q`/`to_k`/`to_v`), ComfyUI
  (`diffusion_model.blocks.N...`), kohya (`lora_unet_...`) and plain native
  names, BF16/F16/F32, any rank, full or partial coverage — and the manager
  shows what each file touches (format, blocks, rank). Adapters work in all
  three compute modes with no fused cache files: the engine adds the
  stacked delta to each weight on the GPU as it is loaded or streamed in
  (see [LoRA](#lora)). A 4-step Turbo distillation LoRA (from
  [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo))
  works end to end; each LoRA can carry a recommended step count
  (auto-detected from names like `..._4step_...`), and switching such a
  Turbo LoRA on moves the generation to that many steps. Each LoRA also has
  a **step rule** (auto / Euler / re-noise) and an **audio shift**, for
  few-step adapters trained with another sampler, such as
  [DMAD](https://github.com/Yzmblog/DMAD) (see
  [Step rule and audio shift](#step-rule-and-audio-shift)).
- **Progress previews** (v0.6.2): with "Show progress previews while
  generating" on in General Settings > Generation, a frame of the finished video the model
  currently predicts is shown about five times per video while noise is
  removed. The heavy video VAE doesn't run: a small decoder (taeh3 from
  madebyollin/taehv, MIT, 22.7 MB, downloaded the first time) runs with
  MPSGraph, so generation time barely changes. API: `GET /api/preview` (PNG),
  `POST /api/preview/settings` (`{"enabled": true}`), and `denoise_preview` in
  `/api/status`.
- **Three compute modes**, see [Compute modes](#compute-modes): a fast int8
  attention cache, a resident mode for a Mac with memory to spare that
  needs no cache file, and a slow-but-low-memory SSD-streaming mode.
- **Speed modes**: standard / fast / fastest, bundling the engine's own
  approximations - transformer-core reuse across steps (`core_reuse`,
  scaled to the step count) and token reduction - about 2.7x / 2.9x faster
  in the DiT phase at 20 steps; see [Speed modes](#speed-modes).
- **Fast mode (experimental, M5)**: runs the DiT's attention on ccv's int8
  attention kernel - about 1.44x faster for a 15 s clip at 20 steps, little
  or no gain for short clips; off by default. See
  [Fast mode](#fast-mode-experimental).
- **Projects**: a project is a folder (by default `~/Movies/H3cApp/<name>`)
  holding `project.json` (the prompt and every setting), `references/`
  (copies of the images, videos and audio used) and each generated video,
  named `<date>-<time>_seed<seed>.mp4` with a `.json` record of its exact
  request. Videos stay until you delete them (to the Trash). Opening a
  project restores its form; Restore Settings on a video pins its seed to
  make it again. Generated videos can be used as references, and the
  composer's count makes several videos in a row with different seeds (with
  a fixed seed, only the first uses it; the rest are random) into the project - with none open, generating first
  asks for a new or existing project, and asks before replacing an existing
  project's different prompt and settings; after the first, each one
  reuses the encoded prompt and references (about 8 s saved for text,
  ~14 s with a reference image). Without a project the result
  is a temp file, as before; Save as Project names a project from the
  current form and moves that video into it with its record (its input
  files are copied in too). The same request and seed give the same
  frames, bit for bit; the MP4 file itself can still differ invisibly
  (around 57-60 dB PSNR) because the hardware encoder isn't bit-exact.
- **Sizes**: the composer picks a shape (landscape / square / portrait)
  and a size 小/中/大 (small / medium / large), each optionally upscaled 2x
  after generation (Accelerate's high-quality resampling; it adds seconds,
  not minutes). Generated canvases: square 256 / 512 / 768, landscape
  448x256 / 672x384 / 960x544, portrait the same turned upright - so the
  largest upscaled outputs are 1536x1536 and 1920x1088 / 1088x1920. The
  advanced settings take a custom finished size instead (multiples of 32;
  with the 2x option it is generated at half that size, so multiples of
  64) whose generated size must stay within 600,000 pixels (256-1344 per
  side), and
  show an estimate of the whole generation from this Mac's past runs,
  flagging one over an hour. Time grows about with the generated pixels:
  1 s at 20 steps took 107 s at 512x512, 207 s at 960x544 and 234 s at
  768x768 on a 24 GB M5.
- **Window**: a full-window preview with the prompt in a floating panel over
  it (Enter generates, Shift+Enter adds a line, Tab takes the suggested
  prompt; the panel collapses while generating, and clicking its prompt
  line opens it again), an advanced settings dialog (⌘,), named settings
  presets (the last one used is restored at launch), zoom by pinch or
  scroll wheel in the preview, and export that starts in Downloads and
  remembers the last folder. In full screen, the toolbar (project menu and
  title) hides so the preview takes the whole screen, and shows again when
  the pointer reaches the top edge.
- **int8 video VAE on M5**: the video VAE decoder's transformer linears run
  on the int8 TensorOps kernels, about 3x faster decode than F32 at 46 dB
  PSNR against it (`H3_VAE_INT8=0` restores F32).
- **Patch-edge smoothing**: the video VAE decoder turns each latent cell
  straight into its own 16x16-pixel patch, so the step between pixels is
  larger across patch edges (1.2-1.4x elsewhere's): a faint square grid,
  plain once sharpened. After decoding, only that excess jump at each
  edge is softened (a quarter of it), which makes the grid much fainter and brings
  decoded real video slightly closer to the original (+0.13 dB);
  `H3_VAE_DEBLOCK` sets the strength (0 turns it off).
- **Reference VAE tiles**: the decoder runs on the reference decoder's
  256-pixel tiles (with 64-pixel overlaps), and so matches MiniMax's
  reference decoder (diffusers) at about 85 dB in F32. Larger tiles (up to
  320 pixels, picked before for ~10% faster decoding) change the output
  (about 37 dB from the reference at 800x544) and show the 16-pixel patches
  as a staircase of blocks on dark edges. `H3_VAE_TILE_PIXELS` still picks
  another size.
- **Video compression**: every video the app keeps (generated, joined,
  enlarged, imported) is HEVC at a constant quality: General Settings > Saved Quality,
  Quality of Saved Videos, Standard (0.85), High (0.90, the default) or
  Highest (0.95). Bright, smooth skin needs High: at 0.85 a 768x768
  close-up got 2.2 Mbps and its 16-pixel blocks showed, at 0.90 6.4 Mbps
  (as much as the old H.264) with fewer than H.264 had. Files say their
  colours are BT.709; left unsaid, a decoder taking BT.601 shifted red and
  green by about 3 levels in 255. H.264 at an
  average bitrate (5 bits a pixel, at least 9.5 Mbps) gave dark scenes
  about a fifth of it - 2.0 Mbps for a dark 1920x1088 clip - and they broke
  into 8-pixel blocks; HEVC there took 2.4 Mbps and came 0.8 dB closer to
  the uncompressed frames, and on a bright 1600x1088 clip 5.7 Mbps instead
  of 8.0 at 0.5 dB closer (at 0.85). Export can make an H.264 copy for
  players without HEVC ("Export as H.264"), and re-encodes it, or a copy
  with the AI label, at its own Export quality: written at a constant
  quality too
  (3.4 Mbps for the dark clip, 0.5 dB below the HEVC master), as
  AVFoundation's H.264 preset starved it again (1.7 Mbps). Videos made
  before this stay H.264 and are exported as they are. The engine's own
  default for `libh3.a` callers is still H.264 (`h3_params.video_hevc`).
- **Model manager**: register and switch between several H3 checkpoint
  directories and LoRA files instead of one fixed path, and download
  FL2VA/Ref2VA directly from Hugging Face with no external dependency (no
  Python/huggingface_hub) — a plain resumable URLSession downloader,
  offered automatically on first launch. See
  [ModelDownloader.swift](native/H3Spike/Sources/H3cApp/ModelDownloader.swift).
- **Embedded local API**: a plain HTTP/JSON API on `127.0.0.1`, served by
  the app itself over raw POSIX sockets (no Network.framework, no
  third-party server), driving the exact same job/engine state as the
  window. See [API](API.md#api).

## How joined generation works

Joined generation repeats "continue the previous video" and joins the
results at their overlaps. Continuing is in the engine ([h3.c](h3.c)); the
chain and the joining are in the app
([Continuation.swift](native/H3Spike/Sources/H3cApp/Continuation.swift),
[ChainGeneration.swift](native/H3Spike/Sources/H3cApp/ChainGeneration.swift)).

1. **Keep the latents.** Every generation saves its latents, as they were
   before decoding, to a `.h3lt` file (`h3_params.save_latent_path`): a
   9-int32 header (`H3LT`, version, sizes, frame count) followed by the F32
   video latent `[24,T,H,W]` and the normalized audio latent `[32,2,T]`
   (about 3.6 MB for 5 s at landscape medium). Nothing is re-read from a
   rendered video, so joining doesn't cost picture quality.
2. **Pin the last 22 frames.** A continuation takes the previous clip's last
   22 frames (7 latent steps, about 0.9 s) and places them, one latent step
   at a time, as conditions at the first 22 frames of the new clip, held for
   the whole generation (`continue_latent_path` / `continue_frames`,
   `h3_continuation_plan`). This extends FL2VA's first/last-frame
   conditioning to frames in between: each step sits at its own frame's time
   (text length + frame x 5/3). The last frame keeps its original formula, so
   generations without a continuation are unchanged (a first-image video was
   checked bit-identical, picture and sound). A continuation is an FL2VA
   generation, so it needs FL2VA and its cache only, not Ref2VA (unless the
   first clip uses references).
3. **Carry the sound.** The previous clip's sound for those 22 frames (37
   audio latent steps, about 0.92 s) is placed on the new clip's timeline so
   that it ends where the pinned frames end, snapped to the new clip's 40 Hz
   audio grid. With lip sync, the sound is the audio file itself, so no sound
   is pinned.
4. **Join at the overlap.** A continuation's first 22 frames come out nearly
   the same as the previous clip's last 22. Joining lays the clips on two
   alternating video and audio tracks in AVFoundation and, over those 22
   frames, ramps the picture's opacity and the volume from one to the next
   (`JoinedVideoExporter`). Clips sit on a whole-frame grid, and the shift
   from snapping the carried sound to the audio grid (up to 8.3 ms, depending
   on the previous clip's length) is compensated; the joined-generation clip
   lengths (175 and 124 frames) are ones with no shift.
5. **Run the chain.** Joined generation picks the number of clips for the
   length (each adds its length minus 22 frames; the last is shortened to fit
   what is left) and repeats steps 2-4 with the seed going up by one per clip.
   With lip sync, the audio's start moves on by (frames - 22) / 24 s per clip.

Measured on a 24 GB M5 at landscape medium:

- Over the 22-frame overlaps the pictures differed by about 3/255 on
  average; the overlapping sound correlated at 0.86-0.93 for generated sound
  and above 0.99 with lip sync.
- A continuation takes about 20% longer than a new clip (446 s against 364 s
  for 5 s), for the pinned conditions.
- 15 s from three clips took 23-25% less time than one 362-frame
  generation.
- Each clip only sees the last 0.9 s or so before it, so over many clips the
  scene can drift slowly (for example, a sunset gradually coming into view).

### Implementations this follows

The way of continuing and joining follows the ideas of these ComfyUI
implementations. None of their code is used: the ideas were implemented
independently in this repository's engine (C) and app (Swift).

- [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
  (NikoDemon80, GPL-3.0): pin the previous clip's last frames, one per
  condition, at the head of the next clip, and place its tail sound at the
  same time. Its author reports that placing the sound on the timeline
  turned a similar-but-not-continuing sound into one that carries on.
- The "H3 Motion Context Clip Stitcher" in
  [ComfyUI-noEmbryo](https://github.com/noembryo/ComfyUI-noEmbryo)
  (noembryo, MIT): decode saved latents one clip at a time and crossfade the
  overlap (22 frames or another whole number of latent steps), picture and
  sound.
- The workflow combining them,
  [Endless MiniMax H3 (with Endless LipSync)](https://civitai.com/models/2909076)
  (NikoDemon80, Civitai): long videos together with lip sync.

## Compute modes

The DiT's ~37 GiB of BF16 weights have to be served to the GPU somehow every
generation. Three mutually exclusive modes exist, all selectable from the
app's compute-mode picker (or the API's `compute_mode` field):

| | Int8 attention cache | Resident | SSD streaming |
|---|---|---|---|
| App / API value | Fast, int8 cache / `attentionCache` | Resident, no cache needed / `resident` | Memory-saving, SSD streaming / `ssdStreaming` |
| Setup | A one-time cache build (the app offers this in-window when needed; ~28s per checkpoint, writes an ~18 GiB int8 cache file) | None to disk — quantizes in place at load time instead (same ~28s cost, paid again on every launch/model switch) | None — reads the checkpoint as-is |
| Speed | Fastest measured path | Comparable to the cache path (no per-step disk I/O either) | Slower; a 22-frame/512-square clip measured 141s vs. ~78s for the cache path (both 20 steps) |
| Memory | int8-quantized weights streamed in double-buffered slots | Every DiT block resident at once — ~18 GiB int8-quantized on a tensor-capable GPU, or the full ~37 GiB BF16 otherwise | Only 2 DiT blocks resident (~2 GiB tracked storage) at a time |
| Requires | M5-class GPU (Metal 4 TensorOps / int8 path) | Any Apple Silicon GPU (falls back to BF16 residency without tensor hardware) — meant for a Mac with memory to spare | Any Apple Silicon GPU |
| LoRA | Yes | Yes | Yes |

Full detail on the cache format — versioning, `model_kind`/`model_id`
mismatch guards — is in [h3_attention_cache.c](h3_attention_cache.c). A
cache can also be built as a standalone step with
`build_attention_cache <FL2VA/transformer dir> <output file>` (`make
build_attention_cache`), or pointed at a model root to build both FL2VA and
Ref2VA caches in one pass.

#### Curve-form AdaLN (experimental, `H3_ADALN_CURVE`)

About 24 GiB of each DiT checkpoint is AdaLN: per block, a [96768, 2688]
projection of `silu(time embedding(t))`. Its only input is the scalar
timestep, so the inputs lie on one smooth curve. `build_adaln_curve
<transformer dir> <output file>` (`make build_adaln_curve`, ~7 s) projects
that curve onto a 32-dimensional basis and stores each projection as
[96768, 32] plus a 4097-point table of the curve's coordinates — a 611 MiB
file, the same layout as ComfyUI's curve-form ("pruned") checkpoints. With
`H3_ADALN_CURVE=<file>` the engine builds the AdaLN schedule from it in
~0.01 s instead of reading the dense projections (~8.5 s).

On the released FL2VA checkpoint, across all 51 projections and the
schedules the app uses, the curve-form values are as close to an exact
float64 computation as the dense BF16 path is (relative L2 1.5–1.7e-3 for
both, which is BF16 rounding). They are not bit-identical, so the same seed
gives a slightly different video. A curve file records the fingerprint of
the transformer it was built from and is refused for any other (build one
each for FL2VA and Ref2VA). For now this saves time, not disk space: the
dense weights stay in the checkpoint until the planned import packs drop
them.

The app uses it by default in every compute mode (General Settings > Generation:
"Speed up the condition computation with an approximation"). The first generation
with a model builds `adaln_fl2va.curve` or `adaln_ref2va.curve` next to that
model's int8 cache (~7 s, shown as "Preparing the computation (first time
only)"), rebuilds it if the weights change, and falls back to the dense
projections if it cannot be built.

Both an unbounded generation duration *and* an oversized reference video push
against the same ceiling here — see the next section.

### Library-level configuration

The engine itself (`h3_dit.c`) reads a number of environment variables
directly — `H3_ATTENTION_CACHE`, `H3_ATTENTION_CACHE_DIR`, `H3_INT8_STREAM_MLP`,
`H3_QWEN_PREFETCH*`, `H3_ZERO_COPY_WEIGHTS`, `H3_VAE_TILE_PIXELS`,
`H3_VAE_INT8`, `H3_DIT_COMMAND_BLOCKS`, `H3_PROFILE`, and a long tail of
`H3_DISABLE_*`/`H3_USE_SLOWER_*`-style A/B diagnostic switches. These apply
to anyone linking `libh3.a` directly; the app itself doesn't rely on them
for normal use; it drives the same underlying options through its own UI
and `H3GenerationParams`. They're documented at their point of use in the
source (start from [h3_dit.c](h3_dit.c), [h3_gpu.m](h3_gpu.m), and
[h3_attention_cache.c](h3_attention_cache.c)).

## Speed modes

Independent of the compute mode, the advanced settings (or the API's
`speed_mode`) choose how much the engine approximates:

| Mode | Engine settings | DiT phase, 512x512 / 124 frames (5 s) / 20 steps on M5 |
|---|---|---|
| Standard (`quality`) | exact; whole-velocity `reuse` as set (default 2) | 557 s at reuse 1 |
| Fast (`fast`) | transformer core reused over 4 steps, `reuse` 1 | 204 s (~2.7x) |
| Fastest (`fastest`) | fast + token reduction | 189 s (~2.9x) |

Both faster modes change the sampling trajectory, so the same seed gives a
different composition than standard. Core reuse scales with the step count
(steps / 5, at most 4), so a 4-step Turbo LoRA run effectively keeps only the
token reduction. The engine won't combine core reuse with a whole-velocity
`reuse` above 1, so the presets run at `reuse` 1; the stored `reuse` setting
is kept and applies again under standard.

All modes use all 50 DiT blocks. Up to 0.2.0 the faster presets also skipped
5 gate-ranked blocks (45 of 50); that broke the audio (loud broadband noise
with tonal bands), so 0.3.0 dropped it. The block count is still available
on its own as the layer count in the advanced settings (35–50, default 50,
the API's `dit_layers`), with a warning that fewer blocks can break the
audio.

### Fast mode (experimental)

A separate fast-mode checkbox in the advanced settings (the API's
`fast_attention`) runs the DiT's attention through
[ccv](https://github.com/liuliu/ccv)'s int8 attention kernel with BF16 input
and output instead of the engine's own path. It is off by default and shown
only when available: the app must be built with ccv linked in (the release
`.dmg` is; see [Native app](BUILD.md#native-app)) and the GPU must have neural matrix
accelerators (M5). Requesting it elsewhere is an error, never a silent
fallback.

Measured on an M5 with 24 GB at 768x768, 15 s, 20 steps, seed 7 (command-line
driver): 13727 s → 9506 s (1.44x, 70 min saved), peak footprint 21.09 →
22.35 GiB. The gain grows with canvas size and clip length and is about nil
at 256x256; it also applies on top of the speed presets (see the table
below). The same seed gives a different video than standard. Side-by-side
checks on several prompts and seeds found no consistent quality loss, which
is not a proof of equal quality.

### Measured generation times

A grid run through the app's API on an M5 with 24 GB (2026-10-02/03): square,
int8 attention cache, 20 steps, the prompt "A cat playing with a ball of
yarn.", seed 7, one run each, wall time from request to finished file.
Standard ran at the default `reuse` 2; fast and fastest at `reuse` 1 with
core reuse 4. Each cell is fast mode off / on.

| Size | Length | Standard | Fast | Fastest |
|---|---|---|---|---|
| 256x256 | 5 s | 1:34 / 1:39 | 1:08 / 1:09 | 0:58 / 0:58 |
| 256x256 | 10 s | 2:54 / 2:54 | 1:48 / 1:48 | 1:28 / 1:28 |
| 256x256 | 15 s | 4:24 / 4:19 | 2:44 / 2:44 | 2:08 / 2:08 |
| 512x512 | 5 s | 5:59 / 5:39 | 3:44 / 3:29 | 2:44 / 2:38 |
| 512x512 | 10 s | 15:56 / 13:35 | 9:25 / 8:10 | 6:14 / 5:39 |
| 512x512 | 15 s | 30:08 / 24:27 | 17:16 / 14:10 | 11:05 / 9:25 |

- The presets help more the larger and longer the clip: against standard,
  fast is 1.4–1.75x and fastest 1.6–2.7x faster.
- Fast mode does nothing at 256x256 (5% slower at 5 s standard). At 512x512
  it saves 3–7% at 5 s, 9–15% at 10 s and 15–19% at 15 s, presets included.
  The fastest setting for a 15 s 512x512 clip, fastest + fast mode, took
  9:25 against 30:08 for standard.
- Time grows roughly in proportion to length at 256x256 (1 : 1.9 : 2.8 for
  5/10/15 s) and faster than that at 512x512 (1 : 2.7 : 5.0 for standard).
- Swap stayed between 559 and 715 MiB across all 36 runs: 512x512 at 15 s
  fits in 24 GB.

## LoRA

Adapters are never fused into a cache file. [h3_lora.c](h3_lora.c) reads
each file once per generation, maps its tensors onto the engine's own
weight layout (ComfyUI and diffusers store attention q/k/v contiguously,
the official checkpoint interleaves them per head; diffusers also swaps
the two halves of the SwiGLU `fc1`), and concatenates the stack along the
rank. Whenever a block's weights reach the GPU — once at load for resident
weights, every step for streamed ones (int8 cache, SSD) — one matmul per
projection adds `Σ strength_i · alpha_i/rank_i · B_i A_i` in place. The
delta is far below one int8 or BF16 step, so it is added with stochastic
rounding (round-to-nearest would silently drop most of it); the rounding
noise is deterministic, so the same seed reproduces the same video.

Covered: `qkv`, `out`, `fc1` and `fc2` of the 50 DiT blocks and the 2
token-refiner blocks. Anything else in a file (`adaln_proj`, `final_layer`,
…) is ignored, and the model manager shows how many such tensors there
were. Measured on M5 at 512x512 / 39 frames / 4 steps: the lightx2v Turbo
LoRA (rank 128, all 50 blocks) adds about 1.8 s per step with the int8
cache (7.5 → 9.3 s) and about 2 s with SSD streaming (11.9 → 13.9 s); stacking a
second, rank-16 LoRA on top costs nothing measurable. Its diffusers and
ComfyUI releases produce bit-identical videos.

### Step rule and audio shift

The engine normally steps with the official pipeline's Euler rule. Some
few-step distillations are trained with the **re-noise** rule instead — the
rule of ComfyUI's `lcm` sampler, which LCM/DMD-style adapters usually
recommend: at every step the model's velocity gives a clean-latent
prediction `x0 = x + σ·v`, and the next step starts from fresh noise at the
next sigma, `x = (1 − σ_next)·x0 + σ_next·ε`. The noise comes from a hash of
the seed, the step and the element, so the same seed still reproduces the
same video (`h3_params.sampler`; [h3_host.c](h3_host.c),
`h3_renoise_bf16` in [h3_shaders.metal](h3_shaders.metal)). Re-noise
evaluates every step, so denoiser reuse runs at 1 with it. Some adapters
were also trained with another audio sigma shift than the engine's 3
(`h3_params.audio_sigma_shift`).

Each LoRA in the model manager has both settings:

- **Step rule**: 自動 (auto), Euler or re-noise. Auto uses what the file's
  metadata declares and is Euler otherwise; nothing is guessed from file
  names. Today only DMAD's files declare it (their `sampling` metadata
  says "re-noise step rule" and "audio shift 2"); for any other adapter,
  set it from the publisher's instructions.
- **Audio shift**: empty means the metadata's value, else the default 3.

If any enabled LoRA asks for re-noise, the job uses it; the first audio
shift in stack order wins. The used-settings summary and each project
video's record show both.

Compared on M5 at 448x256 and 672x384, 5 s, 4 steps, 3 prompts x 2 seeds
(DMAD with re-noise and audio shift 2 against the lightx2v Turbo v1.2 LoRA
with Euler): both take the same time. Turbo gives clearly more detail; it
also showed colored flare blobs or garbled burned-in captions in 4 of 12
videos, where DMAD showed none, but DMAD's music came out 6-11 dB quieter.
Run with Euler instead, DMAD looked dull and soft — use the rule an
adapter was trained with. Turbo is an Euler adapter; with re-noise it
showed ghosting.

## Reference-conditioning cost: why a bigger reference is not free

A Ref2VA reference isn't encoded once and cached — the DiT re-attends to it
on every layer, every denoising step, alongside the main video/audio
latents. Since attention cost scales at least quadratically with total
token count, a larger or longer reference multiplies the cost of the
*entire* generation, not just a one-time encoding pass. Measured cases with
large/long references have taken 4+ hours to complete (correctly, but
slowly). The memory-aware sizing described above reduces the risk of an
extreme case but doesn't change this scaling.

**If a generation with references is unexpectedly slow**: use a
smaller/shorter reference, fewer denoising steps, or a higher `reuse`
value. An extreme case is refused outright with a clear error (a 512 MiB
attention-mask cap in the MPSGraph fallback path, see [h3_gpu.m](h3_gpu.m))
rather than risking an out-of-memory crash.

## AI video super-resolution (SwiftVR)

"動画を拡大" has, next to the resize, MetalFX and Real-ESRGAN upscalers, the
engine's own implementation of SwiftVR (arXiv 2606.09516): a one-step video
restoration model built on Wan2.2-TI2V-5B with a small causal autoencoder
(the Restoration-aware Autoencoder, ReAE). The design and every measurement
are in `UPSCALE_PLAN.md`; the short version:

- **Weights.** Not bundled (SwiftVR Research License, non-commercial). The
  app downloads the 20 GB float32 safetensors from Hugging Face into
  `~/Library/Application Support/H3cApp/Upscalers/SwiftVR` after the license
  notice, and converts them once into an `.h3vc` cache: int8 (per-row scales,
  4.6 GB) on a GPU with tensor hardware (M5), BF16 (8.9 GB) elsewhere. The
  constant conditioning (timestep 1000, the fixed prompt's cross-attention
  keys and values) is precomputed into the cache, so none of the text or
  time embedders ever run.
- **Pipeline.** The input is bilinearly resized to the target, padded to a
  multiple of 32, encoded by the ReAE (16x spatial, 4x temporal, 48
  channels) in chunks of 28 then 24 frames with the autoencoder's state
  carried across chunks, restored by the 30-block DiT in a single pass
  (`z_hq = z_lq - v`), and decoded. The DiT's self-attention is the
  mask-free shifted-window scheme (16x16 token windows, full temporal
  extent, alternating half-window shifts); only each window's owned rows run
  as queries. Chunks follow the reference protocol exactly (the last chunk
  pads to 4k+1 frames and prepends the previous chunk's latents), so the
  output matches the official implementation: 51 dB on the restored latents
  and 48 dB on the 8-bit frames (int8; 60 and 51 dB on BF16) against a
  float32 run of the authors' code.
- **Kernels.** The DiT runs on the same int8/BF16 TensorOps linears as the
  generator; the FFN's 14336-wide reduction runs as two K-contiguous halves
  with the GELU fused into the quantization. The ReAE's 3x3 convolutions and
  MemBlocks run as tap-accumulated `matmul2d` products on a padded BF16
  "grid" (`h3_vsr_tapconv_bf16_nax`), the rest of its layers on small grid
  kernels; on Macs without tensor hardware it falls back to F32 MPSGraph.
- **Cost (M5, 24 GB, int8).** 960x544 -> 1920x1088: about 8.2 s per 24-frame
  chunk (ReAE encode 0.5, DiT 6.2, ReAE decode 1.7), about 40 s for a 5 s
  clip, resident GPU memory about 10 GiB. 384x272 -> 768x544: about 4 s per
  chunk. Progress is reported per DiT block and 中止 lands at a block
  boundary. SwiftVR is refused while a generation, cache build or import is
  running, since it needs the whole GPU.
- **Tools.** `build_vsr_cache` converts weights from the command line;
  `tests/test_real_vsr.c` checks the engine against a dump of the official
  implementation (`tools/dump_swiftvr_fixture.py`); `h3_vsr_bench` times the
  stages of a chunk.

## License

MIT — see [LICENSE](LICENSE). Third-party notices (a Metal kernel design
adapted from ccv's FlashAttention implementation) are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The inference engine
originates from [antirez/h3.c](https://github.com/antirez/h3.c) by
Salvatore Sanfilippo.

**Publishing generated videos.** MiniMax H3's license (its Acceptable Use
Policy, item 12) requires a clear disclosure that a video is AI-generated.
The app's “Export Video…” can put “AI-generated with MiniMax H3” in the
lower right (on by default); without it, use the platform's AI label or the
description. Every MP4 the app writes also records it in the file's
metadata.

**What a video records about itself.** Generated, joined and exported videos
carry an XMP packet (an MP4 `uuid` box) with IPTC's standard fields (Photo
Metadata 2025.1 / Video Metadata Hub 1.7): `Iptc4xmpExt:AIPromptInformation`
(the prompt, and each clip's directions for a joined video), `AISystemUsed`
(MiniMax H3), `AISystemVersionUsed` (FL2VA / Ref2VA) and `DigitalSourceType`
(trainedAlgorithmicMedia), plus `xmp:CreatorTool` and the settings to make it
again (the same as its `.json` record, absolute paths cut to file names) as
JSON in `h3capp:GenerationSettings`. Turning off “Record the prompt and
generation settings in the file” in Export Video… records only that it is
AI-generated and which model made it. Tools such as `exiftool -XMP:all
video.mp4` read it.
