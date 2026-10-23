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
