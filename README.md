# H3cApp

[English](README.md) | [日本語](README.ja.md)

**Powered by [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)** — an independent app, not an official MiniMax product.

H3cApp is a native macOS app that runs
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — a
text/image/video-to-video-with-audio diffusion transformer — entirely on
Apple Silicon. Everything runs in-process against Metal/MPSGraph: no
Python, no PyTorch, no cloud calls — media in and out goes through native
AVFoundation/ImageIO. The underlying
C/Objective-C inference engine (`libh3.a`) is a fork of Salvatore
Sanfilippo's (antirez) [h3.c](https://github.com/antirez/h3.c); full
attribution is in [License](#license).

While the app is running it also serves a local JSON API on `127.0.0.1`,
so it can be driven from a script as well as from its own window — see
[API](#api).

## Download

A Japanese guide for first-time users - requirements, installation, model
download and a first video - is at https://junzokamahara.github.io/H3cApp/.

The easiest way to install H3cApp is the prebuilt, signed and notarized
`.dmg` on the [Releases page](https://github.com/JunzoKamahara/H3cApp/releases/latest) —
no Xcode Command Line Tools or building from source needed. Download the
`.dmg`, open it, and drag `H3cApp` into `Applications`; Gatekeeper accepts
it on first launch with no "unidentified developer" warning.

To build from source instead, see [Building from source](#building-from-source)
below.

## Requirements

- Apple Silicon Mac, macOS 15 (Sequoia) or later: the engine uses
  MPSGraph's scaled dot-product attention (macOS 15) and BF16 (macOS 14).
  Releases up to 0.4.3 listed macOS 13, but their engine already needed
  these APIs. An M5-class GPU (Metal 4 TensorOps) unlocks the fastest int8
  paths; older Apple Silicon works but falls back to BF16/MPSGraph
  automatically.
- Xcode Command Line Tools (`clang`, `swift`, `ar`) — only needed when
  building from source; not required for the prebuilt `.dmg`.
- The MiniMax-H3 checkpoint from Hugging Face
  ([`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)) —
  `FL2VA/` alone is ~134 GiB (a ~37 GiB BF16 transformer plus the Qwen3-VL
  text encoder and video/audio VAEs); adding `Ref2VA/` for reference-
  conditioned generation brings it to ~268 GiB combined. The app's model
  manager can download either or both directly.
  Or, from v0.6.0, a quantized model (GGUF): about 35–45 GB, downloaded
  and imported from the same wizard — see
  [Downloading and importing GGUF models](GGUF.md).
- Nothing else at runtime.

## Quick start

```sh
make -j8 libh3.a
cd native/H3Spike
./package_app.sh
open .build/H3cApp.app
```

On first launch, if no model is registered yet, the app offers to download
MiniMax-H3 directly from Hugging Face (see [Features](#features)) — or use
the model manager to point it at a checkpoint you already have.

With a supported GPU, use the compute-mode picker to build the int8
attention cache (fast path) — or pick the memory-saving SSD-streaming mode
to skip that step and stream the original BF16 weights instead.

## Features

- **GGUF (quantized) models** (v0.6.0): download a quantized DiT and text
  encoder from Hugging Face (unsloth, Qwen and other publishers, or any
  repository or URL) or import GGUF files you have, and use them like the
  official model — see [GGUF.md](GGUF.md).
- **Text-to-video+audio** (T2VA): a prompt alone produces a synchronized
  H.264 + AAC clip.
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
  [Joined generation: how to use it](#joined-generation-how-to-use-it).
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
  generating" on in Advanced Settings, a frame of the finished video the model
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
  frames, bit for bit; the H.264 file itself can still differ invisibly
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
- **Model manager**: register and switch between several H3 checkpoint
  directories and LoRA files instead of one fixed path, and download
  FL2VA/Ref2VA directly from Hugging Face with no external dependency (no
  Python/huggingface_hub) — a plain resumable URLSession downloader,
  offered automatically on first launch. See
  [ModelDownloader.swift](native/H3Spike/Sources/H3cApp/ModelDownloader.swift).
- **Embedded local API**: a plain HTTP/JSON API on `127.0.0.1`, served by
  the app itself over raw POSIX sockets (no Network.framework, no
  third-party server), driving the exact same job/engine state as the
  window. See [API](#api).

## Joined generation: how to use it

One generation makes at most 15 s (362 frames). Longer videos come from
joined generation: short videos are made one after another,
each continuing the one before, and joined into one at their overlaps.

1. **Open a project.** The clips and the joined video are kept in one. With
   none open, pressing generate asks you to create one or pick an existing
   one.
2. **Pick the length.** In the length menu under the prompt, pick “15 s
   Joined” or “Custom Clips…” from the “Joined Generation” section.
   - **15 s Joined**: three clips of about 7 s (“Clip length for joined
     generation” in Advanced Settings can make them about 5 s).
   - **Custom Clips**: an editor opens with one clip; “Add a Clip” adds the
     next one below. Each clip gets a length (3-15 s) and what happens in
     it. From the second clip on, turning off “Use the same reference
     images” lets that clip use its own
     reference images, or a first and/or last frame. A clip with a first
     frame doesn't continue the one before: it starts from that image and is
     joined with a cut. With only a last frame it still continues and is
     made to end on that image. Each clip's lip-sync audio can carry on from
     the clip before, change to another file, or be off. Each clip's start
     time and where it is in its audio are listed, so scenes can be laid
     out along a song. The clips can be saved to and read from a JSON file.
   The summary under the panel shows the number of clips, e.g. “3 clips
   joined (15.1 s in all)”. The count menu is off for joined generation (one
   at a time).
3. **Write the common directions.** The prompt box is used for every clip
   (the people, their clothes, the style); each clip gets its own directions
   added after it.
4. **Generate.** The first clip is an ordinary generation; from the second
   on, each one carries over the previous clip's last 22 frames (about
   0.9 s) and their sound. While it runs, the preview shows which clip of
   how many and the whole run's elapsed time, time left and finish time.
5. **Done.** The clips are joined with a crossfade over each overlap into
   one video, shown in the preview. Project Videos lists it as one row,
   “Joined video (5 clips, 30.7 s)”, with its clips folded under it.

Good to know:

- **Directions for each clip.** Given the same prompt, every clip tends to
  start over from the prompt's first scene. Writing what happens in each
  clip in Custom Clips moves the story on (a clip left empty is made from
  the prompt alone). With 15 s Joined, “Edit the clips one by one…” in the
  panel switches to Custom Clips.
- **Make it longer.** With the joined video shown, “Continue” in the panel
  makes one more clip; “Join and Export” then joins the whole run again.
- **If you stop partway,** the clips made so far (two or more) are joined.
- **Make the same run again.** With Randomness set to Fixed, the first clip
  uses that seed and each later one the next seed up, so the same settings
  make the same run.
- **First image, references and lip sync.** Reference images are used by
  every clip, so the people look the same when they come back after leaving
  the frame; in Custom Clips, “Use the same reference images” can be turned
  off for a scene they don't appear in. A first or last frame is used by the first
  clip only. With lip sync, the audio's start moves on by
  itself for each clip, and clips that start after the audio ends make their
  own sound.
- **How long it takes.** On a 24 GB M5 at landscape medium, about 105 s per
  second of finished video (about 50 min for 30 s, about 2.5 h for 90 s).
  15 s joined took 23-25% less time than one 362-frame generation. The
  estimate for the current settings is under “Estimate on this Mac” in
  Advanced Settings.
- **Files.** The clips go in the project's `chains/<name>/` with their
  `.json` records and `.h3lt` latents. Deleting the joined video moves its
  clips to the Trash too.

Through the API, use `seconds` / `chained` / `chain_clip_seconds` (and
`clip_prompts` for directions for each clip) on `POST /api/generate` (see [API](#api)).

### How joined generation works

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

#### Implementations this follows

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

The app uses it by default in every compute mode (Advanced settings: "Speed
up the condition computation with an approximation"). The first generation
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
`.dmg` is; see [Native app](#native-app)) and the GPU must have neural matrix
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

## Repository layout

```
h3.c, h3_dit.c, h3_gpu.m, ...   Core inference engine (C + Objective-C/Metal), builds into libh3.a
h3.h                            Public C API surface (h3_load_dir, h3_generate, h3_build_attention_cache, ...)
h3_shaders.metal                All Metal compute kernels
h3_build_attention_cache.c      CLI wrapper around h3_build_attention_cache() -> build_attention_cache
h3_adaln_curve.c                Curve-form AdaLN files (reader + builder); h3_build_adaln_curve.c -> build_adaln_curve
h3_lora.c                       LoRA loading (format normalization) and GPU weight patching
tests/                          C test suite (make test / make parity)
native/H3Spike/                 Native macOS app (SwiftPM)
  Sources/CH3                   C shim exposing libh3.a's C API to Swift
  Sources/H3Engine              Swift async wrapper over the C API (AsyncThrowingStream-based progress/cancellation)
  Sources/H3cApp                The SwiftUI app itself (H3cApp.app), including ModelLibrary/ModelManagerView
                                 (registered models/LoRAs), ModelDownloader (Hugging Face downloads), and
                                 HTTPServer/GenerationViewModel+API (the embedded automation API)
  Sources/H3Spike                Minimal in-process spike/reference client for H3Engine, not the shipped app
  package_app.sh                Builds + bundles H3cApp.app; also signs/notarizes it, see below
  make_dmg.sh                   Packages the built H3cApp.app into a distributable .dmg
```

## Building from source

### Library

```sh
make -j8 libh3.a     # builds the C/Objective-C engine
make test             # deterministic host suite (+ Metal/MLX parity if fixtures are installed)
make parity            # just the Metal/MLX numerical checks
```

`make build_attention_cache` builds the standalone cache-preparation tool
mentioned above, and `make build_adaln_curve` the curve-form AdaLN tool;
neither is part of the default `make` target.

### Native app

```sh
cd native/H3Spike
./package_app.sh
```

`libh3.a` isn't rebuilt by this script — run `make libh3.a` at the repo root
first, and again after changing engine code. `package_app.sh` then runs
`swift build -c release` and assembles `H3cApp.app`. It builds to a
scratch directory outside the repo (`${TMPDIR}H3cApp-build-scratch`)
rather than SwiftPM's default in-tree `.build`, because a repo that lives
under a synced folder (Google Drive, iCloud Drive, Dropbox, ...) can cause
that daemon to hold locks on SwiftPM's `build.db`, intermittently failing
the build with a spurious "disk I/O error" — if you see that, check whether
your checkout is inside a synced directory before assuming it's flaky.

Fast mode needs ccv: build a patched, MPS-enabled ccv checkout as described
in [tools/ccv_eval/README.md](tools/ccv_eval/README.md), then pass its path to
both steps (the release `.dmg` is built this way):

```sh
make -j8 libh3.a CCV_DIR=/path/to/ccv
cd native/H3Spike
CCV_DIR=/path/to/ccv ./package_app.sh
```

Without `CCV_DIR` the app builds as before, without fast mode.

The UI is in English and Japanese, following the macOS language setting.
The Japanese strings in the Swift sources are the keys; the tables are
`native/H3Spike/Packaging/{en,ja}.lproj/Localizable.strings`, copied into the
app by `package_app.sh`. A string reaches the UI translated only as a SwiftUI
literal (`Text("…")`, `Button("…")`, …) or `String(localized: "…")`.
`./check_localizations.sh` (in `native/H3Spike`) extracts the keys with the
compiler and lists any missing from either table. To try the other
language: `open .build/H3cApp.app --args -AppleLanguages '(en)'`.

#### Code signing and notarization

Signing and notarization are opt-in via two environment variables, so a
plain `./package_app.sh` with neither set produces an unsigned development
build:

```sh
H3C_SIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" \
H3C_NOTARY_PROFILE="some-keychain-profile" \
./package_app.sh
```

- `H3C_SIGN_IDENTITY` — a `Developer ID Application` identity from
  `security find-identity -v -p codesigning`. Requires enrolling in the
  Apple Developer Program and creating that certificate (Xcode → Settings →
  Accounts → Manage Certificates → `+` → Developer ID Application — no
  Xcode project is needed for this, just the account manager). Signing
  alone (no `H3C_NOTARY_PROFILE`) is enough to run the app locally with
  Hardened Runtime enabled.
- `H3C_NOTARY_PROFILE` — a profile name saved once via
  `xcrun notarytool store-credentials <profile> --apple-id ... --team-id ...`
  (needs an [app-specific password](https://appleid.apple.com), not your
  regular Apple ID password). With both variables set, the script also
  zips, submits for notarization, waits, and staples the ticket, so the
  resulting `.app` passes Gatekeeper (`spctl -a -vvv -t exec`) on any Mac.

There are no nested frameworks or embedded dylibs to worry about here —
`H3cApp`'s only linked libraries are Apple system frameworks and the
statically-linked `libh3.a` (plus `libccv.a` with `CCV_DIR`), so a single
`codesign --deep` on the bundle is sufficient.

#### Distributing as a .dmg

```sh
./make_dmg.sh
```

Packages the already-built `H3cApp.app` into a compressed `.dmg` with a
drag-to-Applications shortcut, using only `hdiutil` (no third-party
dmg-building tool). Run `package_app.sh` first. If `H3C_SIGN_IDENTITY` and
`H3C_NOTARY_PROFILE` are set, `make_dmg.sh` also signs, notarizes, and
staples the `.dmg` file itself — optional, since Gatekeeper's real check on
launch is against the `.app`'s own signature/staple, but it makes the `.dmg`
itself pass a Gatekeeper check too for a fully clean download experience.

## Testing

```sh
make test
make parity
```

`make test` runs the deterministic host suite and, when the (git-ignored)
MLX fixtures are installed under `misc/fixtures/`, also compiles the Metal
source at runtime and checks a toy H3 block against named MLX outputs —
intentionally at runtime, matching Iris, so it needs no Xcode offline Metal
toolchain. `make parity` runs just those Metal/MLX checks.

Same-seed reproducibility: `make test` includes `h3_determinism_tests`
(repeat runs of the threadgroup-reduction kernels must match bit for bit).
With the released weights installed, `make h3_repro_check CCV_DIR=...` and
`./h3_repro_check` generate a short Ref2VA clip three times (~4 min) and
fail unless the RGB frames and the decoded audio handed to the encoder are
identical. `--run` overrides the request per run, and runs with the same
request (ignoring `cache`) are compared - for example `--run cache=0 --run
cache=conditioning --run cache=conditioning` checks cache off, a miss and a
hit against each other.

## API

While `H3cApp.app` is running it serves a plain JSON API on
`http://127.0.0.1:8420` — implemented over raw POSIX sockets (see
[HTTPServer.swift](native/H3Spike/Sources/H3cApp/HTTPServer.swift)), not
Network.framework or any third-party server, and with no separate process
to start or stop. It drives the app's primary `GenerationViewModel`/engine
instance — the one the first window shows (see
[AppModels.swift](native/H3Spike/Sources/H3cApp/AppModels.swift) and
[GenerationViewModel+API.swift](native/H3Spike/Sources/H3cApp/GenerationViewModel+API.swift)).
It keeps answering with every window closed, as long as the app runs;
closing a window closes its project, so open one with
`POST /api/project/open` when needed. There is only ever one job at a time,
shared with the window showing that instance — pressing its generate button
and a `POST /api/generate` compete for the same slot, and whichever loses
gets a clear `409`.

Because the client and server are always on the same Mac, media inputs are
plain filesystem paths, not uploads.

| Method | Path | Does |
|---|---|---|
| `POST` | `/api/generate` | Starts a job. JSON body fully replaces the current draft (see below); `202` once started, `400` on a bad request, `409` if a job is already running. |
| `GET` | `/api/status` | Engine/job state, progress fraction, phase/stage text, error message, and whether a result is ready. While several videos or a joined generation run, `batch` gives `index`/`total`/`chained` and the whole run's `elapsed_seconds` and `estimated_remaining_seconds` (the current video's live estimate plus the later ones, scaled by the pace seen so far); the top-level `estimated_remaining_seconds` stays the current video's. |
| `POST` | `/api/cancel` | Cancels the running job, if any. |
| `GET` | `/api/result/video` | Streams the current result as `video/mp4`; `404` if none, or once the next job's `generate()` call deletes it. |
| `GET` | `/api/models` | Registered H3 model directories (id, name, path, whether active). |
| `GET` | `/api/loras` | Registered LoRA files (id, name, path, strength, recommended steps, whether enabled, step rule `sampler` and `audio_shift` as set in the model manager, and `recommended_sampling` from the file's metadata - `null` while the file is still being read). |
| `GET` | `/api/project` | The open project: name, path, count, and its videos (file, seed, completion time, generation time, `continued_from`, `can_continue`, and for a joined video `joined_clips` / `planned_clips`). |
| `POST` | `/api/project/new` | `{"name", "directory"?}` creates a project from the current form (default directory `~/Movies/H3cApp`) and opens it. |
| `POST` | `/api/project/open` | `{"path"}` opens a project folder; the form takes its saved state. |
| `POST` | `/api/project/close` | Closes it; results are temp files again. |
| `POST` | `/api/project/restore` | `{"video"}` puts that video's request into the form with its seed fixed. |
| `POST` | `/api/project/use-as-reference` | `{"video"}` adds that video to the form as a reference video. |
| `POST` | `/api/project/delete-video` | `{"video"}` moves the video and its record to the Trash. |
| `POST` | `/api/project/join` | `{"video"}` joins the chain of continuations ending at that video into `<name>_joined.mp4` in the project, with a record, (cross-fading each carried-over stretch) and returns `{"path", "clips"}` when done. |

`POST /api/generate` body fields, all optional except `prompt`:

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | Required. |
| `size_profile` | `"square"` | A preset `<shape>-<size>` with shape `square`, `landscape` or `portrait` and size `small`, `medium` or `large`, plus `-x2` for the 2x upscaled output (e.g. `landscape-large-x2`: 960x544 upscaled to 1920x1088); or `custom-<W>x<H>` / `custom-<W>x<H>-x2` for a finished size of W x H (multiples of 32, or of 64 with `-x2`, which generates at half that size); the generated size must be at most 600,000 pixels with 256-1344 per side. The older names `smallSquare`, `square`, `landscapeNative`, `landscapeUpscaled`, `portraitNative` and `portraitUpscaled` still work and stand for the matching small/medium presets (see `SizeProfile` in [GenerationModels.swift](native/H3Spike/Sources/H3cApp/GenerationModels.swift)). |
| `seconds` | `5` | 1–15. |
| `steps` | `20`, or the first enabled Turbo LoRA's recommended steps | 3–40. |
| `reuse` | `2` | 1–3. The faster speed modes run at 1 whatever this is. |
| `compute_mode` | the app's own default for this GPU | `attentionCache`, `resident`, or `ssdStreaming`. |
| `speed_mode` | `"quality"` | `quality`, `fast`, or `fastest` (see [Speed modes](#speed-modes)). |
| `dit_layers` | `50` | 35–50. Fewer blocks can break the audio. |
| `fast_attention` | `false` | `true` for [fast mode](#fast-mode-experimental); `400` where it isn't available. |
| `seed` | random | 0–18446744073709551615, as a number or a decimal string (a string keeps a large seed exact for clients whose JSON numbers are doubles). |
| `first_frame_path` / `last_frame_path` | none | FL2VA anchors; cannot combine with `reference_paths`. |
| `seconds` / `chained` / `chain_clip_seconds` | `5` / `false` / `7` | 1-15 s are made in one generation; 16-90 s (or 15 with `"chained": true`) by joined generation, from clips of 5 or 7 s, into one joined video. A joined length needs an open project and `count` 1. |
| `clip_prompts` | none | Joined generation only: an array of strings, one per clip in order, each added after `prompt` for that clip (an empty string or a missing entry uses `prompt` alone). |
| `clip_references` | none | Joined generation with reference images only: an array of true/false, one per clip in order, whether that clip uses the reference images (missing entries do). |
| `clips` | none | Custom Clips: an array of objects, one per clip: `seconds` (3-15; default 7), `prompt` (added after `prompt`), and from the second clip `images` (`same` / `references` / `frames`), `reference_paths`, `first_frame_path` (the clip then starts from it, joined with a cut), `last_frame_path`, `audio` (`carry_on` / `file` / `none`), `audio_path`, `audio_offset`. The first clip uses the request's own images and lip sync. Needs an open project. |
| `continue_from` | none | A video of the open project (a file name from `GET /api/project`) to continue: its last 22 frames and their sound are carried over. Needs the same size; not with first/last frames (references are fine). |
| `forced_audio_path` / `forced_audio_offset` | none / `0` | Lip sync: this file's audio (an audio file or a video's soundtrack), from `forced_audio_offset` seconds on, is held as the soundtrack while the picture is generated to match it, and the output carries that audio itself. Works with every creation method. |
| `reference_paths` | `[]` | Ordered Ref2VA references; image/video/audio is auto-detected per path. At least one image or video is required if any audio path is included. |
| `loras` | `[]` | The LoRA stack for this job: names from `GET /api/loras`, or `{"name": ..., "strength": 0.8}` objects (strength multiplies the adapter's trained scale; omitted keeps the entry's saved strength). An object may also set `"sampler"` (`"auto"`, `"euler"`, `"renoise"`) and `"audio_shift"` (0 = auto); like strength, these are saved to the entry. Omitting `loras` means no LoRA, even if some are switched on in the window. |
| `sampler` / `audio_shift` | from the LoRAs | Override the [step rule and audio shift](#step-rule-and-audio-shift) for this one job only (`"auto"`, `"euler"`, `"renoise"`; 0 = auto), e.g. for A/B tests. |
| `lora_name` / `lora_scale` | none | Older single-LoRA form of `loras`; ignored when `loras` is given. |
| `count` | `1` | 1–20 videos in a row with different seeds; above 1 needs an open project. |
| `from_form` | `false` | `true` generates from the form as it is (e.g. after `/api/project/restore`); only `count` may accompany it. |

A value outside its range, or of the wrong type (`true`, `"5"` or `2.5`
where an integer is expected), is a `400` naming the allowed range; nothing
is clamped or rounded silently.

```sh
curl -X POST http://127.0.0.1:8420/api/generate -H "Content-Type: application/json" -d '{
  "prompt": "A red fox walks through fresh snow in a pine forest.",
  "size_profile": "square", "seconds": 5, "steps": 20
}'

curl http://127.0.0.1:8420/api/status
curl http://127.0.0.1:8420/api/result/video -o fox.mp4
```

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
