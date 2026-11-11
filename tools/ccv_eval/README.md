# ccv attention kernel evaluation tools

Reproducible tooling for comparing [liuliu/ccv](https://github.com/liuliu/ccv)'s
Metal (MFA) int8 attention kernels against this engine's own real production
attention output. Built while investigating item 5 (Sol-Attn-style block-sparse
attention) in [`plans/SPEEDUP_ROADMAP.md`](../../plans/SPEEDUP_ROADMAP.md) — read that file
for the full narrative, the numbers these tools produced, and an important
correction (a real capture-layout bug that invalidated an earlier round of
results before these tools existed).

Not built by this repo's own `Makefile` — ccv is a separate, large (BSD-3-Clause
licensed) project, not vendored here. These are standalone tools you build
against your own `ccv` checkout.

## 1. Get a patched, MPS-enabled ccv checkout

```bash
git clone https://github.com/liuliu/ccv.git
cd ccv
git checkout unstable
# Tools here were built and validated against this exact commit:
git checkout 6a611be1aab6470ae279115ae4eaf7e01bc87135
git apply /path/to/h3c-app/tools/ccv_eval/ccv-sol-metal27-addrspace.patch
git apply /path/to/h3c-app/tools/ccv_eval/ccv-na-int8-bf16-lse-store.patch
git apply /path/to/h3c-app/tools/ccv_eval/ccv-na-attention-msl4-options.patch
```

The third patch makes `NAInt8AttentionKernel` and `NAAttentionKernel`
compile their generated shaders with an explicit Metal 4.0 language
version instead of `nil` options. With `nil`, the runtime's default
language version follows the SDK version recorded in the host
executable's `LC_BUILD_VERSION`; SwiftPM links `native/H3Spike` (and the
app) with `sdk 13.0` for its macOS 13 deployment target, which predates
`MetalPerformancePrimitives`, so the shader failed with `use of undeclared
identifier 'mpp'` from Swift but not from `clang`-built tools (`sdk 27.0`).
This was the "Swift-only" failure; it reproduces from plain C linked with
`-Wl,-platform_version,macos,13.0,13.0` and disappears from Swift when
linked with SDK 27.0 recorded. h3's own shaders already set
`MTLLanguageVersion4_0` explicitly (`h3_gpu.m`).

The second patch is needed only for `H3_CCV_DIRECT=1` (BF16 I/O into the
dense `NAInt8AttentionKernel`): with BF16 I/O the kernel stores its
log-sum-exp `L` as `bfloat`, and the generated
`L[idx[0]] = cM[k] + fast::log2(cL[k]);` fails to compile (`assigning to
'bfloat' from incompatible type 'float'`). It adds an explicit
`({{L_MEMORY_NAME}})(...)` conversion at those two stores. `L` is only
written in the forward pass (it feeds the backward kernels), so the forward
output is unaffected. `h3_generate_cli` depends on `$(CCV_DIR)/lib/libccv.a`,
so rebuilding ccv relinks it.

The patch fixes a real Metal shader compile failure
(`no matching member function for call to 'get_destination_cooperative_tensor'`)
seen on this machine's SDK (27.0) — a Metal address-space qualifier surviving
into a `decltype(...)` used as a template argument in
`NAInt8SolAttentionKernel.cpp`'s runtime-generated shader source, matching a
reported [MLX issue](https://github.com/ml-explore/mlx/issues/4533) with the
same symptom. Fixed with `metal::remove_addrspace_t<...>` at 7 call sites —
no change to math, precision, or data layout. This patch is *only* needed for
the Sol (sparse) kernel; the plain dense `NAInt8AttentionKernel` this
repo currently uses in production evaluation doesn't hit it.

```bash
brew install wget   # the default `make libccv.a` fetches an unrelated sample model
cd lib
./configure --enable-mps   # opt-in flag - NOT auto-detected, easy to miss
cd ..
# Target the app's oldest macOS, like this repo's Makefile (MACOSX_MIN);
# otherwise clang targets the build machine's macOS and libccv.a's objects
# claim that as their minimum.
MACOSX_DEPLOYMENT_TARGET=15.0 make libccv.a
```

## 2. Capture real Q/K/V and the real production output from this app

Build `H3Spike` (`native/H3Spike`) against this repo's `libh3.a` as usual, then:

```bash
# Rebuild the attention cache first - a stale cache fingerprint only warns
# by default and can silently feed wrong int8-quantized weights into the
# very generation you're trying to capture from. Use the strict flag so a
# validation run refuses to proceed on a mismatch instead of warning:
./build_attention_cache ~/models/MiniMax-H3/FL2VA/transformer \
    ~/models/cache/dit_int8_v2.cache
export H3_ATTENTION_CACHE_STRICT=1

H3SPIKE_SIZE=512 H3SPIKE_FRAMES=39 H3SPIKE_STEPS=4 H3SPIKE_SEED=7 \
H3_DUMP_ATTENTION_QKV=/path/to/capture/block25 \
H3_DUMP_ATTENTION_BLOCK=25 H3_DUMP_ATTENTION_STEP=2 \
H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1 \
./native/H3Spike/.build/release/H3Spike ~/models/MiniMax-H3 /tmp/out.mp4 "prompt text"
```

This writes `block25.{q,k,v,out}.bin` (contiguous FP16, `[1, rows, HEADS,
HEAD_DIM]`, already transposed to row-major regardless of which internal
layout the engine used — see `block25.meta.txt`, which records the true
source layout and the written layout *separately* for every tensor) plus
`block25.meta.txt` itself.

`H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1` is required to get a *comparable*
row-major output capture — production runs normally use the head-major output
layout for the following projection's efficiency, which this capture
intentionally does not exercise. Don't confuse this forced-row-major capture
path with the production-optimized path when interpreting timing.

## 3. Run ccv's kernels on the captured Q/K/V, diff against the real output

```bash
cd ccv/bin/mfa
clang h3_real_attention_compare.cpp -o h3_real_attention_compare.o -c \
  -std=c++17 -O3 -I"../.." -I"../../lib" -fblocks -D HAVE_CBLAS -D HAVE_PTHREAD \
  -D HAVE_ACCELERATE_FRAMEWORK -D USE_DISPATCH -D HAVE_MPS -I/usr/local/include
clang -o h3_real_attention_compare h3_real_attention_compare.o ../../lib/libccv.a \
  -L/usr/local/lib -lm -lblas -lpthread -framework Accelerate \
  -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
  -framework Foundation -framework CoreVideo -framework CoreML -framework IOSurface \
  -framework Metal -lc++ -framework QuartzCore

./h3_real_attention_compare <T> 56 /path/to/capture/block25 470 0.08838834765
# writes block25.{ccv_dense,ccv_sparse}.bin - <T> must match rows in block25.meta.txt

clang -O2 -o compare_fp16 /path/to/h3c-app/tools/ccv_eval/compare_fp16.c -lm
./compare_fp16 block25.ccv_dense.bin  block25.out.bin   # dense int8 vs. real production output
./compare_fp16 block25.ccv_sparse.bin block25.out.bin   # Sol sparse vs. real production output
```

`compare_fp16` reports relative L2, max abs diff, RMSE, and PSNR between two
same-length contiguous-FP16 files.

## What this caught, and why it matters

An earlier round of "real data" validation compared ccv's kernels against
**ccv's own internal reference** (its Sol-exact-mode output vs. its own dense
kernel's output), not against this app's real production SDPA output. That
self-check reported a plausible-looking `sparse_relative_l2 = 0.1274` — but it
was measured on Q/K/V that had been captured with the wrong axis layout
(this app's default int8 QKV path produces `[heads, rows, head_dim]`, not
`[rows, heads, head_dim]`), silently scrambling the data before it ever
reached ccv's kernel.

Building the direct real-vs-real comparison in `h3_real_attention_compare.cpp`
surfaced this immediately: the first result was `relative_l2 = 1.37` — worse
than uncorrelated noise, an unmistakable layout-bug signature. **A
`relative_l2` above roughly 1.0 against a real reference means check your
layout assumptions before concluding anything about kernel quality.** After
fixing the capture (see `h3_gpu_head_major_sdpa_inputs()` in `h3_gpu.m`/`.h`)
and rebuilding the stale attention cache, the same comparison gave sane,
trustworthy numbers — see `plans/SPEEDUP_ROADMAP.md` for the current results table
and what they do and don't establish (per-block PSNR is not directly
comparable to final decoded-frame PSNR; end-to-end generation validation is
still the open item).

## Fast mode in the engine and app

`h3_params.fast_attention = 1` (the app's "高速モード（試験的）" checkbox,
`"fast_attention": true` in its HTTP API, `H3SPIKE_FAST_ATTENTION=1` in
H3Spike / `h3_generate_cli`) selects the direct path for one generation;
`h3_fast_attention_available()` says whether it can run.

**Diagnostic override:** when `fast_attention` is 0, the environment
variables used throughout this directory still select a ccv path —
`H3_ATTENTION_BACKEND=ccv_dense` (FP16 bridge, or direct with
`H3_CCV_DIRECT=1`) or `ccv_fp16`. So ccv can run even with the checkbox
off if one of these is set in the app's environment. `h3_result`'s
`ccv_attention_calls` / `ccv_attention_direct_calls` report what actually
ran; the app keys its history and timing calibration on those, not on the
checkbox.
