# H3cApp development guide

[English](BUILD.md) | [日本語](BUILD.ja.md)

For developers who build, test, sign or distribute H3cApp. If you simply want to use it, install the signed [release .dmg](https://github.com/JunzoKamahara/H3cApp/releases/latest) instead. Building from source requires Xcode Command Line Tools.

## Quick start

```sh
make -j8 libh3.a
cd native/H3Spike
./package_app.sh
open .build/H3cApp.app
```

On first launch, if no model is registered yet, the app offers to download
MiniMax-H3 directly from Hugging Face (see [Features](TECHNICAL.md#features)) — or use
the model manager to point it at a checkpoint you already have.

With a supported GPU, use the compute-mode picker to build the int8
attention cache (fast path) — or pick the memory-saving SSD-streaming mode
to skip that step and stream the original BF16 weights instead.

## Repository layout

```
include/                        Public C API: h3.h (h3_load_dir, h3_generate, h3_build_attention_cache, ...),
                                 h3_vsr.h, h3_seedvr2.h, h3_latent_upscale.h, h3_lora_extract.h, h3_refmod.h
src/core/                       h3.c (h3_generate) with h3_params.c, h3_references.c, h3_conditioning.c and
                                 h3_plan_generation.c beside it; host math, memory, planning; h3_options.c, the
                                 table of every H3_ environment switch (`build/bin/h3_options` prints it)
src/io/                         Weights (safetensors, GGUF, quantized formats), tokenizer, video/audio files
src/model/                      The DiT (h3_dit.c with h3_dit_setup.c, h3_dit_weights.c, h3_dit_block.c) and its schedule,
                                 LoRA, HyperFlow, the text/vision encoders, the VAEs, RefMod; h3_model_family.c, the
                                 tables of model families, samplers and compute modes
src/gpu/                        The Metal layer (h3_gpu.m with h3_gpu_blocks.m, h3_gpu_linear.m, h3_gpu_attention.m,
                                 h3_gpu_conv.m) and the engine's Metal compute kernels (h3_shaders_core.metal,
                                 h3_shaders_tensorops.metal, h3_shaders_blocks.metal)
src/upscale/                    The upscale module, built as libh3upscale.a: SwiftVR (h3_vsr*), SeedVR2 (h3_seedvr2*),
                                 the latent upscaler, their GPU code (h3_gpu_vsr.m, h3_gpu_sv2.m) and kernels
tools/                          Command-line drivers (h3_build_attention_cache.c -> build_attention_cache,
                                 h3_build_adaln_curve.c -> build_adaln_curve, ...) and fixture scripts
build/                          Everything `make` writes: build/libh3.a, build/libh3upscale.a, build/h3_shaders.metal
                                 (the kernel sources joined), build/obj/, build/bin/ (not in git)
tests/                          C test suite (make test / make parity)
native/H3Spike/                 Native macOS app (SwiftPM)
  Sources/CH3                   C shim exposing libh3.a's C API to Swift
  Sources/H3Engine              Swift async wrapper over the C API (AsyncThrowingStream-based progress/cancellation)
  Sources/H3Upscale             Swift wrappers of the upscale module (SwiftVR, SeedVR2, the latent upscaler)
  Sources/H3cApp                The SwiftUI app itself (H3cApp.app), including ModelLibrary/ModelManagerView
                                 (registered models/LoRAs), ModelDownloader (Hugging Face downloads), and
                                 HTTPServer/GenerationViewModel+API (the embedded automation API);
                                 Upscale/ holds the enlargement feature: one type per method in Upscale/Methods,
                                 listed in UpscalerRegistry (UpscalerMethod.swift)
  Sources/H3Spike                Minimal in-process spike/reference client for H3Engine, not the shipped app
  package_app.sh                Builds + bundles H3cApp.app; also signs/notarizes it, see below
  make_dmg.sh                   Packages the built H3cApp.app into a distributable .dmg
```

## Building from source

Building from source needs the Xcode Command Line Tools (`clang`, `swift`,
`ar`); the prebuilt `.dmg` doesn't.

### Library

```sh
make -j8 libh3.a     # builds the C/Objective-C engine
make test             # deterministic host suite (+ Metal/MLX parity if fixtures are installed)
make parity            # just the Metal/MLX numerical checks
```

`make build_attention_cache` builds the standalone cache-preparation tool
mentioned above, and `make build_adaln_curve` the curve-form AdaLN tool;
neither is part of the default `make` target.

Everything `make` builds lands in `build/`: the generation engine at
`build/libh3.a`, the upscale module at `build/libh3upscale.a` (it uses the engine,
not the other way round; `make libh3.a` builds both and the joined shader source),
the tools and tests in `build/bin/` (run them from the repo root, e.g.
`build/bin/h3_gguf_tool`).

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
in [tools/ccv_eval/README.md](../tools/ccv_eval/README.md), then pass its path to
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
toolchain. `make parity` runs just those Metal/MLX checks. The SwiftVR
parity test (`h3_real_vsr_test`) likewise runs only when the downloaded
SwiftVR weights and the fixture of `tools/dump_swiftvr_fixture.py` are
present, and prints `skip:` otherwise.

Same-seed reproducibility: `make test` includes `h3_determinism_tests`
(repeat runs of the threadgroup-reduction kernels must match bit for bit).
With the released weights installed, `make h3_repro_check CCV_DIR=...` and
`./h3_repro_check` generate a short Ref2VA clip three times (~4 min) and
fail unless the RGB frames and the decoded audio handed to the encoder are
identical. `--run` overrides the request per run, and runs with the same
request (ignoring `cache`) are compared - for example `--run cache=0 --run
cache=conditioning --run cache=conditioning` checks cache off, a miss and a
hit against each other.
