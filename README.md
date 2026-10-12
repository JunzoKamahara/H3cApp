# h3c-app

[English](README.md) | [日本語](README.ja.md)

H3cApp is a native macOS app that runs
[MiniMax-H3](https://huggingface.co/lightx2v/Minimax-h3-Turbo) — a
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
  Turbo LoRA on moves the generation to that many steps.
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
mentioned above; it isn't part of the default `make` target.

### Native app

```sh
cd native/H3Spike
./package_app.sh
```

`libh3.a` isn't rebuilt by this script — run `make libh3.a` at the repo root
first, and again after changing engine code. `package_app.sh` then runs
`swift build -c release` and assembles `H3cApp.app`. It builds to a
scratch directory outside the repo (`${TMPDIR}h3c-app-build-scratch`)
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
| `GET` | `/api/status` | Engine/job state, progress fraction, phase/stage text, error message, and whether a result is ready. |
| `POST` | `/api/cancel` | Cancels the running job, if any. |
| `GET` | `/api/result/video` | Streams the current result as `video/mp4`; `404` if none, or once the next job's `generate()` call deletes it. |
| `GET` | `/api/models` | Registered H3 model directories (id, name, path, whether active). |
| `GET` | `/api/loras` | Registered LoRA files (id, name, path, strength, recommended steps, whether enabled). |
| `GET` | `/api/project` | The open project: name, path, count, and its videos (file, seed, completion time, generation time). |
| `POST` | `/api/project/new` | `{"name", "directory"?}` creates a project from the current form (default directory `~/Movies/H3cApp`) and opens it. |
| `POST` | `/api/project/open` | `{"path"}` opens a project folder; the form takes its saved state. |
| `POST` | `/api/project/close` | Closes it; results are temp files again. |
| `POST` | `/api/project/restore` | `{"video"}` puts that video's request into the form with its seed fixed. |
| `POST` | `/api/project/use-as-reference` | `{"video"}` adds that video to the form as a reference video. |
| `POST` | `/api/project/delete-video` | `{"video"}` moves the video and its record to the Trash. |

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
| `reference_paths` | `[]` | Ordered Ref2VA references; image/video/audio is auto-detected per path. At least one image or video is required if any audio path is included. |
| `loras` | `[]` | The LoRA stack for this job: names from `GET /api/loras`, or `{"name": ..., "strength": 0.8}` objects (strength multiplies the adapter's trained scale; omitted keeps the entry's saved strength). Omitting it means no LoRA, even if some are switched on in the window. |
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
