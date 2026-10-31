# H3cApp upscaling (AI video super-resolution) plan

[English](UPSCALE_PLAN.md) | [日本語](UPSCALE_PLAN.ja.md)

This document is the design and implementation plan for an "AI video super-resolution" feature that enlarges finished videos with generated detail.
It separates decisions, verified facts, estimates and open points so that an implementer (a person or an AI) can start without having to guess.
Written 2026-10-28 against main at h3c-app v0.7.1.

## Background (why this document exists)

- The next task is upscaling. The LightX2V review memo (2026-10-28) concluded:
  "do not replace the H3 engine; for quality, build **an independent upscaler that adopts SwiftVR's memory-efficient ReAE**;
  on the H3 side, compute the LoRA correction separately and improve memory management where needed."
- The existing enlargement (v0.6.8–0.6.9) stops at Lanczos / MetalFX / Real-ESRGAN (a CNN) and cannot give
  "the generative texture of SeedVR2". SeedVR2 itself is impractical on a 24 GB Mac (34 GB of memory, 539 s for 149 frames at 320x240).
- This document is written so that an implementer (a person or an AI) can **start implementing without hesitation**,
  with decisions, verified facts, estimates and open points kept apart.

## Decisions (2026-10-28)

| Question | Decision |
|---|---|
| Model | **Design for both.** Build one "one-step Wan-family VSR engine"; SwiftVR first, structured so FlashVSR-1.1 Tiny can be added later |
| Document | **New bilingual pair** (`UPSCALE_PLAN.ja.md` / `UPSCALE_PLAN.md`) |
| Alternative (re-noise "hires-fix" with H3 itself) | **Included as an experiment phase** |

---

## §0 How to read this, assumptions

- Target repository: `~/Documents/work/h3c-app` (MIT). Work goes on **one branch per task from main**,
  deleted after the merge (this repository's working rule). Check `git branch --show-current` before committing.
- Notation: **[verified]** = confirmed in source or official material, **[estimate]** = computed or inferred, **[open]** = to be measured or confirmed during implementation.
- Target machines: Apple Silicon with 24 GB is the baseline. **M5 (MTLGPUFamilyApple10, Metal 4 TensorOps)** takes the int8 NAX path;
  everything else (M1–M4) **must work** on the BF16 MPSGraph path (slower).
- No external programs (no ffmpeg etc.). Everything runs inside the app (the same constraint as the existing enlargement).
- Weights are **not bundled**. The user fetches them from Hugging Face on first use (with the license shown) and the app converts them into an on-device cache.

## §1 Background, purpose, goals

- Goal: enlarge finished videos (e.g. 960x544, 24 fps, 5–90 s) **2x (4x later)** with generative detail, in practical time on a 24 GB Mac.
  The user's interest is "skin, hair and texture at SeedVR2 level".
- Non-goals: replacing the H3 generation engine, training, porting SeedVR2.
- **Quality goal (first): get as close as possible to SeedVR2 (3B) output.** SeedVR2 itself does not run on a 24 GB Mac, so the goal is defined as
  "use SeedVR2 as the gold reference for quality and compare this implementation's output against it, narrowing the gap" (how to measure: §9.5).
  Targets: on the same input set, the mean of no-reference quality metrics (MUSIQ / MANIQA / CLIP-IQA, DOVER for video) is **at least 95% of SeedVR2-3B's**,
  the flicker metric is no worse than SeedVR2's, and in the user's blind A/B a majority of cases are "indistinguishable from or equal to SeedVR2" [target values].
  The difference between this Metal port and the official PyTorch SwiftVR must stay within the parity tolerances of §9 (the port must not lose quality).
- Success criteria (acceptance; repeated in §8):
  - Measure the quality goal with the §9.5 procedure and record the result (met / not met, with a breakdown of the gap) in the document.
  - 960x544→1920x1088, 124 frames (5.17 s) in **under 2 minutes on a 24 GB M5 [estimated target]**, peak ≤ 10 GiB.
  - Completes on a 24 GB M4 too (target ≤ 6 minutes, peak ≤ 14 GiB).
  - Numerical agreement with the PyTorch reference (tolerances in §9).
  - Added as a choice in the existing "Enlarge" UI; the existing four methods are unchanged.
- Priority between quality and speed: **always keep a quality-first mode (full attention, BF16, overlap on); make the speed mode (MFSWA, int8) the default only after measuring the quality difference.**

## §2 Candidate model survey (2026-10-28)

| Item | SwiftVR (first) | FlashVSR-1.1 Tiny (second) | (reference) SeedVR2 |
|---|---|---|---|
| Official | github H-oliday/SwiftVR, HF H-oliday/SwiftVR, arXiv 2606.09516 | github OpenImagingLab/FlashVSR, HF JunhaoZhuang/FlashVSR-v1.1 | ByteDance-Seed |
| Backbone | Wan2.2-TI2V-5B (30 layers, dim 3072, 24 heads×128, FFN 14336, in/out 48 ch, patch (1,2,2)) **[verified: transformer/config.json]** | Wan2.1-T2V-1.3B shape (30 layers, dim 1536, 12 heads×128, FFN 8960, in/out 16 ch) **[verified: Wan2.1-1.3B config; DMD weights 5.68 GB = 1.42B F32]** | 3B/7B DiT |
| Inference | **1 step**, t=1000 fixed, z_HQ = z_LQ − v **[verified]** | 1 step, t=1000, streaming KV cache, noise input + LQ feature addition **[verified]** | multi-step |
| Attention | MFSWA: 16x16 windows (latent-token grid), full temporal extent, even layers unshifted / odd layers half-window shift, windows clamped, mask-free dense SDPA **[verified: official transformer.py / LightX2V attention.py]** | 3D windows (2,8,8), causal block mask, **LCSA = top-k block selection from Q/K window means** (local range 9/11), needs a block-sparse kernel **[verified]** | full attention |
| AE | **ReAE** 40.95M params, 16x spatial, 4x temporal, 48 ch, TAEHV family (causal streaming with MemBlock state) **[verified: official reae.py]** | TCDecoder (TAEHV family [512,256,128,128], latent 16+768 with LQ pixels concatenated as conditioning) + LQ_proj_in (Causal_LQ4x_Proj, 576 MB) | 3D VAE |
| Weights | F32 20.0 GB + reae 164 MB + prompt_emb 4.2 MB | 5.68 GB + 0.58 GB + 0.19 GB (Wan2.1_VAE 0.5 GB not needed by Tiny) | — |
| License | **SwiftVR Research License (non-commercial, research / personal only; commercial use needs written permission)**. Code is Apache-2.0. An older HF page said Apache-2.0, but the current README says Research License **[verified]** | **Apache-2.0** (code and HF tag) | Apache-2.0 |
| Published performance | H100 bf16 1920x1080: Enc 26 ms / DiT 328 ms / Dec 85 ms, 24 frames, peak 29 GB (ReAE unbatched) **[verified: paper Table 5]** | A100 768x1408 ≈17 fps **[verified]** | H100 1440p 24 fr 17.3 s |
| Other than 4x | The paper only shows 4x. The input is bilinearly resized to the target size before restoration, so **any factor (including 2x) is mechanically possible; quality to be measured [open]** | "4x only", input center-cropped to multiples of 128 | — |
| Apple implementations | None (MPS untested) | WaifuFinder's Swift/MLX conversion (BF16 3.5 GB, ROI validated on a 16 GB Mac) | mlx-gen (needs ffmpeg) |

Additional candidate **[not surveyed]**: FlashVSR+ (released 2026-10, github OpenImagingLab/FlashVSR_Plus, arXiv 2610.17552, claims better quality with "Hybrid Attention + RY-DMD", pretrained models published). Check its license, backbone and attention scheme in P1 and decide whether it replaces FlashVSR Tiny as the second back-end candidate.

**Judgement**: SwiftVR is the strongest candidate for quality, speed and implementation simplicity (constant conditioning, mask-free attention, small AE).
The one obstacle is the license → **not bundled; user download with a non-commercial notice** (the same style as the existing MiniMax H3 Community License notice).
FlashVSR Tiny is safe license-wise but needs data-dependent block-sparse attention and a KV cache, so it is more work → second phase.
On the quality gap to SeedVR2, the SwiftVR paper claims "strong" no-reference quality, but **this survey did not obtain concrete comparison numbers [open]** → measure it ourselves in §9.5.

## §3 Map of the existing code (what the implementer reads first)

| Role | Where | Key points |
|---|---|---|
| Enlargement entry point, frame I/O | `native/H3Spike/Sources/H3cApp/VideoUpscale.swift` | `VideoUpscaler` enum (rawValue = API name and record value), `UpscaleFactor` 1.5/2/3/4, **private `FrameScaler`** (one BGRA8 frame → one, synchronous, no temporal state), AVAssetReader (32BGRA, Metal-compatible) → scale → AVAssetWriter (HEVC, quality value), audio and metadata passed through, written to a hidden file and swapped in on completion, progress every 4 frames, cancel between frames |
| CNN super-resolution | `RealESRGAN.swift` | MPSGraph F16, `graph.run` per tile (no executable), tiles = 1024 core + context, F16 safetensors loader |
| Existing TAEHV-family decoder | `TinyPreviewDecoder.swift` | **Same layer structure as ReAE** (Clamp, conv, MemBlock, nearest 2x, TGrow, pixel shuffle) in MPSGraph. `MPSGraphExecutable` cached by shape, GPU-only placement (the ANE rejects F32) |
| UI | `UpscaleSheet.swift`, `H3cApp.swift` (Video menu), `ProjectViews.swift`, `ContentView.swift` (progress banner) | Enumerates `VideoUpscaler.allCases`; disabled via `isAvailable` |
| HTTP API | `GenerationViewModel+API.swift` | `/api/project/upscale-video`, `/api/upscale-file`, `/api/upscale/cancel`; validates `upscaler`/`scale` |
| Records | `Projects.swift` (`upscaledFrom`, `upscaler`, `upscaledSize`, `upscaleFactor`), `Continuation.swift` (enlargement) | Unknown rawValues fall back to nil/standard |
| Existing on-demand downloads | `DenoisePreview.swift` (taeh3 22.7 MB, URLSession), `ModelDownloader.swift` (resumable, chunked, for huge checkpoints), `ModelImportDownload.swift` (`ModelLicenseNotice`) | Large downloads reuse `ModelDownloader` |
| Engine GPU base | `h3_gpu.h/.m`, `h3_shaders.metal` (compiled at runtime; new kernels must be added to the name list in `h3_gpu_create`), `h3_metal.m` | BF16/int8 linears (NAX `matmul2d`), MPSGraph SDPA (with a batch dim), LayerNorm/RMSNorm/GELU, conv3d (channels-last, padding separate), `stream_file_*` (F_NOCACHE), mmap zero-copy |
| Example of driving a transformer | `h3_dit.c` | QKV→SDPA→out, int8 paths, SSD double-buffer prefetch pthread, `h3_attention_cache.c` int8 cache with fingerprint |
| Video writing | `h3_av_writer.m` (HEVC/H.264, BT.709) | For writing directly from the engine |
| Conversion tools | `tools/convert_realesrgan.py` (numpy only, restricted unpickler), `h3_gguf.c`/`h3_quant.c` | Policy: conversion is done **in C on the device** (Python cannot be shipped) |
| Build | `Makefile` (`LIB_C`/`LIB_M`), `native/H3Spike/Package.swift`, `package_app.sh`, `BUILD.md` (`make test`, skip without fixtures) | |
| Documents | `TECHNICAL(.ja).md`, `GUIDE(.ja).md`, `API(.ja).md`, `THIRD_PARTY_NOTICES.md`, `SPEEDUP_ROADMAP.md`, `docs/` (site) | Bilingual pairs, matching heading counts, measurements state machine, settings and seed |

## §4 Model specifications (everything needed to implement)

### §4.1 SwiftVR DiT (Wan2.2-TI2V-5B shape) [verified: diffusers transformer_wan.py, LightX2V swiftvr/model.py, official streaming/dit.py]

- Input latent z: [48, T, H/16, W/16] (ReAE output, no scaling). Patch (1,2,2) → token count N = T·(H/32)·(W/32);
  a token is a 192-dimensional linear (Conv3d(48→3072, k=s=(1,2,2)) is a Linear 192→3072).
- Conditioning (**all constant**): timestep = 1000.0.
  `temb = Linear(SiLU(Linear(sinusoidal_256(1000))))`, `block_mod = Linear(SiLU(temb))` → 6×3072.
  Text: `prompt_emb[1,512,4096]` → Linear(4096→3072) → GELU(tanh) → Linear(3072→3072) = context[512,3072].
- Block (30 layers, each with `scale_shift_table[1,6,3072]`):
  1. `shift,scale,gate,c_shift,c_scale,c_gate = scale_shift_table + block_mod` (F32)
  2. `h = LN_noaffine(x)·(1+scale)+shift` → self-attention: q=RMSNorm_h(Wq h), k=RMSNorm_h(Wk h), v=Wv h (the RMSNorm is one 3072-wide norm across heads = "rms_norm_across_heads"), 3D RoPE (t:44, h:42, w:42 dims, θ=10000, interleaved real format), MFSWA, Wo → `x += attn·gate`
  3. `h = LN_affine(x)` (norm2, cross_attn_norm=true) → cross-attention: q=RMSNorm(Wq h), k=RMSNorm(Wk ctx), v=Wv ctx (**K,V are constant → precompute**), Wo → `x += attn`
  4. `h = LN_noaffine(x)·(1+c_scale)+c_shift` → FFN Linear(3072→14336) GELU(tanh) Linear(14336→3072) → `x += ff·c_gate`
- Output: `LN_noaffine(x)·(1+scale_out)+shift_out` (`scale_shift_table[1,2,3072]+temb`) → `proj_out` Linear(3072→192) → unpatchify → v [48,T,H/16,W/16].
- One-step inference: **z_HQ = z_LQ − v** (paper Eq. 5; LightX2V `latents = model_input - predict(...)`). No randomness, no seed (deterministic).
- Input preprocessing: the LQ video is **bilinearly resized to the target size (align_corners=False, no antialias)** → [0,1] → **zero-padded right and bottom to multiples of 32** → ReAE. The output is cropped to the target size (rounded to even).
- MFSWA (window (16,16) in latent tokens, full temporal extent):
  - Window starts per axis (LightX2V `axis_window_starts`): `{0}` if `size ≤ window`. Otherwise `starts = clamp(k·window − shift, 0, size−window)` for k=0..ceil(size/window)+1, deduplicated,
    and an interior start is kept only when `starts[i+1] > starts[i−1] + window`. shift = 0 (even layers) / window/2 (odd layers).
  - Example 1920x1088 → grid 34x60: unshifted H:{0,16,18} W:{0,16,32,44} → 12 windows; shifted H:{0,8,18} W:{0,8,24,40,44} → 15 windows. Window length = T·256 (1536 at T=6).
  - Owner: each token's output is taken from one window. LightX2V iterates windows in order and overwrites `owner[idx] = window*len+local` (descending order for unshifted layers, ascending for shifted ones).
    Equivalent to the official implementation (`WanShiftWindow2DInferProcessor`, `prefer_front = not do_shift`). **Confirm both implementations agree in the parity harness [open]**.
  - Paper Table 3 (1440p): full attention PSNR 25.86 / MFSWA 25.58 / masked SWA 25.34. **MFSWA is an inference-time approximation**, so keep full attention (dense SDPA) as a "high quality" option where memory allows.
- RoPE positions: temporal offset = `chunk.start // 4` (positions continue across chunks), spatial from 0. The table is extended as needed (LightX2V `ROPE_EXTENSION=256`).

### §4.2 ReAE [verified: official swiftvr/models/reae.py = LightX2V reae.py]

- Encoder (input RGB [0,1], NTCHW): `pixel_unshuffle(2)`→12 ch; conv3x3(12→64)+ReLU; TPool(64,2) [1x1 conv on 2 stacked frames]; conv3x3 s2 (no bias); MemBlock×3; TPool(64,2); conv s2; MemBlock×3; TPool(64,1); conv s2; MemBlock×3; conv3x3(64→48). 16x spatial, 4x temporal.
  The last chunk appends 3 copies of its final frame to reach 4k+1 (`is_last`).
- MemBlock(x, past): `ReLU(conv3(ReLU(conv3(ReLU(conv3(cat_ch[x, past]))))) + skip(x))`, skip is identity when channels match. past = **the previous frame's input to the same block** (zeros at first; carried as state across chunks).
- Decoder: Clamp(`tanh(x/3)·3`); conv(48→512)+ReLU; MemBlock×3(512); Upsample2x; TGrow(512,1)=1x1 conv; conv(512→256, no bias); MemBlock×3(256); Upsample2x; TGrow(256,2); conv(256→128); MemBlock×3(128); Upsample2x; TGrow(128,2); conv(128→64); ReLU; conv(64→12); `pixel_shuffle(2)`; clamp[0,1].
  **TGrow(stride 2) = temporal nearest ×2 → Conv3d(C→C, kernel (3,1,1), pad (1,0,0), no bias)** (different from the original TAEHV 1x1 conv; reusing `TinyPreviewDecoder`'s `taeh3` TGrow as-is would be wrong).
- `frames_to_trim = 3`: the first chunk's first 3 output frames are discarded.
- Memory savings (LightX2V, exact transformations): (a) TGrow is spatially pointwise, so run it **before the nearest spatial upsample** (1/4 of the pixels).
  (b) Everything other than MemBlock/TPool is frame-independent → run consecutive layers `reae_frame_batch_size` frames at a time, never holding all frames' intermediates.
  (c) The decoder's layers after the last TGrow (two full-resolution convs + pixel shuffle) are split off as "output layers" and streamed `output_batch_size` frames at a time.
- Chunk rule (LightX2V streaming.py): `clip_len=24` (multiple of 4); one chunk if frame_count ≤ clip_len+4;
  otherwise the first chunk is clip_len+4, then clip_len each, the last takes the rest. `latent_count=(frames−1)//4+1`.
  Output range `[max(0,start−3), start+frames−(last?0:3))`. Missing latents in the last chunk are front-filled with the previous chunk's latents.

### §4.3 FlashVSR-1.1 Tiny (second phase, essentials) [verified: diffsynth/pipelines/flashvsr_tiny.py, models/wan_video_dit.py, examples/WanVSR/utils]
- DiT: Wan2.1-1.3B shape. The input is **noise** (seeded); the LQ enters through `Causal_LQ4x_Proj` (PixelShuffle3d(4,8,8)→CausalConv3d(4,3,3) s(2,1,1)×2→Linear(→1536)×layer_num=1), whose output is added before block 0.
  Cross-attention K/V are precomputed from the fixed prompt (`posi_prompt.pth`, `init_cross_kv`). timestep 1000, 1 step, `x − v`.
- Streaming: 6 latent frames the first time, then 2 latent frames at a time; per block, `kv_ratio=3` window blocks of K/V are cached.
- Attention: (2,8,8) window partition, local block mask with `local_range` 9/11, plus **data-dependent top-k** (`topk_ratio = sparse_ratio·768·1280/(H·W)`, softmax over Q/K window means) → block-sparse attention. On Metal it can be implemented exactly as "gather the selected blocks → dense SDPA", but the selection differs per window and head, so the gather volume is large [estimate].
- Decoder: TCDecoder = TAEHV [512,256,128,128] with a "3x3 identity-initialized conv after every ReLU", latent 16+768 (LQ pixels PixelShuffle3d(4,8,8) concatenated). Wavelet/AdaIN color correction after decoding.
- Constraints: 4x, center-crop to multiples of 128, frames 8n+1.
- **Design implication**: the shared engine separates (i) the Wan block (dim/heads/ffn/in_ch parameterized), (ii) the TAEHV-family decoder (driven by a layer list), (iii) the chunk streamer (state carry), and (iv) the attention back-end (dense / window gather / block-selection gather).

## §5 Mathematical and algorithmic devices (effects and their basis)

1. **One-step algebra**: `z_HQ = z_LQ − v_θ(z_LQ, t=1)`. No scheduler, no randomness, one DiT pass per chunk.
2. **Folding the constant conditioning (compute ahead of time)** [verified: the conditioning is constant]:
   - Precompute the 30 blocks × 6 modulation vectors and keep `(1+scale)`, `shift`, `gate` as F32 constants (Wan2.2's time-embedding MLP and text MLP never run at inference time).
   - Precompute cross-attention K,V (RMSNorm applied) per layer: 30×512×3072×2×2 B ≈ **189 MB BF16**. Cross-attention becomes just the Q projection + a 512-key SDPA + Wo (two linears' worth, under 1/8 of the total).
   - LayerNorm → modulation fused into one kernel (in the style of the existing `adaln*_bf16`).
3. **MFSWA cost** (1920x1088, T=6, N=12,240 tokens, 24 heads) [estimate: from the formulas]:
   - Full attention: QKᵀ+PV = 2·N²·128·24 MAC ≈ 0.92 TFLOP/layer → 27.6 TFLOP/chunk. Materializing the scores naively would take N²·24·2 B ≈ 7.2 GB.
   - MFSWA: window length 1536, 12/15 windows → 0.35/0.43 TFLOP/layer → about 11.7 TFLOP/chunk (**about 2.4x less**, memory bounded by window length²).
     The gather copies Q/K/V 1.5–1.9x each (≤ 3×23,040×3072×2 B ≈ 425 MB).
   - Linears: QKV+O 4·3072² + FFN 2·3072·14336 + cross Q,O 2·3072² = 144.7 M MAC/token → **3.54 TFLOP/layer, 106 TFLOP/chunk**. Attention is about 1/9 of the linears (with MFSWA).
     → The main battleground for speed is **the linears (int8 NAX)**; for attention, running the windows in parallel through MPSGraph SDPA's batch dimension is enough.
4. **Per-row int8 quantization (NAX path)**: existing `h3_gpu_quantize_weight_int8` / `linear_int8_bf16` (activations quantized per row on the fly).
   Shape constraints in%128, out%128: 3072 ✓, 14336=112·128 ✓, `proj_out`'s 192 ✗ and patch_embed's in=192 ✗ → these two stay BF16 (NAX BF16 needs in%32, out%64, so 192 ✓).
   Existing measurements: int8 linears in the DiT are about 2x BF16; ccv int8 attention per-block PSNR 52–62 dB. For SwiftVR the pass line is **per-block PSNR ≥ 40 dB, final RGB ≥ 45 dB vs BF16** [open: to measure].
5. **Exact memory-saving transformations of ReAE**: commutation of TGrow (pointwise) with nearest spatial upsampling (strictly exact), frame-batched frame-independent layers, carrying only the MemBlock state (each block's last frame input) → O(1) memory in video length.
6. **Numerical agreement of preprocessing**: re-implement bilinear (align_corners=False, no antialias) on the GPU. MPSGraph `resize(bilinear, centerResult=YES, alignCorners=NO)` uses the same half-pixel centers as OpenCV/TF2 [verified: SDK header], so it is a candidate; if it does not match, a custom Metal kernel. Zero padding right and bottom.
7. **RoPE**: compute `freqs_cos/sin` in F32 per (t_off, T, H', W') into BF16 tables and apply in interleaved form (`x1·cos − x2·sin`, `x1·sin + x2·cos` on adjacent pairs). If the format differs from H3's `rope_text_bf16`, a new kernel.
8. **Zero-copy and SSD**: the weight cache reuses `h3_gpu_load_bf16` (mmap, M5) / `stream_file_*` (F_NOCACHE). With 5 GB of int8 resident, no re-reading (decided in §6).
9. **Determinism**: SwiftVR uses no randomness, so the same input gives the same output (testable in the style of `h3_determinism_tests`).

## §6 Engine design (C / Metal / NAX / ANE)

### §6.0 Facts about the existing engine the design rests on [verified: file:line]

| Fact | Where |
|---|---|
| Each `h3_gpu_create` has its own `MTLCommandQueue` and MPSGraph caches → **two contexts can run on two threads** | `h3_gpu.m:344-357` |
| Kernels must be listed in the `names` array (TensorOps ones inside the `tensorOpsEnabled` branch) | `h3_gpu.m:437-512` |
| **`h3_gpu_linear_bf16` takes the NAX path only when `tensorOpsMode==1` or for H3 shapes (7168/5376/21504)** → 3072/9216/14336 silently fall to MPSGraph. A new setter is needed | `h3_gpu.m:408-413, 3146-3160` |
| NAX BF16 linear: no bias, rows≥128, in%32, out%64, weight `[out,in]` | `h3_gpu.m:3161-3163`, `h3_shaders.metal:1165-1195` |
| int8 linear: rows≥128, in%128, out%128, per-row F32 scales, per-row dynamic activation quantization (rows padded to 128), BF16 output, no bias. The K loop accumulates 128-wide tiles with `multiply_accumulate` | `h3_gpu.m:3643-3706`, `h3_shaders.metal:2597-2650` |
| The MPSGraph SDPA graph has a `batch` dimension, but the BF16 wrapper fixes batch=1 and requires equal q/k/v element counts | `h3_gpu.m:1623-1700, 1704-1718, 1770-1776` |
| `h3_gpu_layer_norm_bf16` computes in F32 with `fma(norm, weight, bias)` → **identical to the modulated LayerNorm with weight=(1+scale), bias=shift** | `h3_shaders.metal:3159-3203` |
| `h3_gelu_bf16` has a tanh-approximation flag (no bias) | `h3_shaders.metal:3219-3243` |
| Existing RoPE is half-split (pairs d with d+half); Wan is interleaved (2i, 2i+1) → **new kernel** | `h3_shaders.metal:3769-3836` |
| Existing `mlp_*` are SwiGLU-specific → unusable for the GELU FFN | `h3_gpu.m:3297-3435` |
| conv3d is F32, channels-last, padding separate, MPSGraph | `h3_gpu.m:2380-2480` |
| `h3_gpu_dequantize` expands raw F32/F16/BF16/GGUF bytes to BF16/F32 (used by the builder) | `h3_gpu.h:169-173` |
| Fingerprint = FNV-1a over shard names, sizes and mtimes, 32 B | `h3_weights.c h3_weight_store_fingerprint` |
| Attention cache: 64 B header `H3AC` v3, `.partial` → rename, unlink on failure | `h3_attention_cache.c:7-23, 54-64, 135-220` |
| `h3_av_reader` decodes whole videos only (no incremental reader) → reading uses Swift's AVAssetReader | `h3_av_reader.h` |
| Upscaling while generating is not guarded (the reverse is) → VSR also guards `isGenerating` | `GenerationViewModel.swift:785`, `VideoUpscale.swift:546,600,745` |

### §6.1 File layout

Engine (added to `LIB_C`/`LIB_M` in the `Makefile`):

| File | Role |
|---|---|
| `h3_vsr.h` | Public C API (§6.2). `#include "h3_vsr.h"` at the end of `h3.h` (the CH3 module is a single symlink to `h3.h`, so nothing else changes) |
| `h3_vsr.c` | Model-agnostic driver: `h3_vsr` (weights, two `h3_gpu` contexts for the DiT and ReAE), `h3_vsr_session` (chunk plan, carried state, one-chunk-lag pipeline, progress/cancel), requirements, `h3_vsr_model_ops` vtable |
| `h3_vsr_chunk.c` | Chunk protocol (28 / 24 / 4b+1 padding / trim 3 / overlap). Pure C, unit-tested |
| `h3_vsr_window.c` | MFSWA geometry (axis starts, window list, gather indices `[Nw][Lw]`, owner map, **owner-only query classes**), RoPE tables (computed in double), patchify indices. Pure C, unit-tested |
| `h3_vsr_wan.c` | Wan block runner (`h3_vsr_wan_config` parameterizes dim/heads/ffn/text length). SwiftVR=MFSWA, FlashVSR=LCSA (second phase) through the same gather/scatter kernels |
| `h3_vsr_reae.c` | TAEHV-family CNN interpreter (table-driven layer list: CONV3x3, CONV3x3_S2, MEMBLOCK, TPOOL, TGROW, UP2_FOLDED_CONV, CLAMP, RELU, PIXEL_(UN)SHUFFLE), MemBlock/TPool state carry, frame batching. Tables for the ReAE enc/dec and (second phase) TCDecoder / LQ_proj_in |
| `h3_vsr_weights.c` | On-device cache builder (`.h3vc`), loader, fingerprint, constant precomputation |
| `h3_vsr_swiftvr.c` / `h3_vsr_flashvsr.c` | Per-model (file names, config.json, weight-name map, pre/post-processing, FlashVSR color correction) |
| `h3_gpu.m/.h` additions | `h3_gpu_set_nax_generic`, `h3_gpu_sdpa_batched_bf16`, `h3_gpu_conv2d_bf16_mps` (M4), dispatch for the new kernels |
| `h3_shaders.metal` additions | The `h3_vsr_*` kernels of §6.5 (NAX ones registered inside the tensorOps branch) |
| `tests/test_vsr.c` (always), `tests/test_real_vsr.c` (needs weights and fixtures, skip-guarded), `tests/bench_vsr.c` | §9. Makefile targets `h3_vsr_tests`, `h3_real_vsr_test`, `h3_vsr_bench`, added to `test:` with the existing `@if test -f` skip pattern |
| `tools/dump_swiftvr_fixture.py` | Dev-only fixture dumper (torch + official repo) |

Swift:

| File | Change |
|---|---|
| `Sources/H3Engine/H3VSR.swift` (new) | `H3VideoSuperResolver`: open/close, `requirements`, `processChunk`, `buildCache` (copy of the `buildAttentionCache` bridge at `H3Engine.swift:710-745`), reuse of `CancelFlag` |
| `VideoUpscale.swift` | Add `private protocol ChunkedFrameScaler` next to `FrameScaler`. `PerFrameChunkAdapter` (wraps existing scalers with chunk size 1) and `VSRChunkScaler`. `upscale()` becomes a 3-stage pipeline (§6.6) that **every method goes through** (existing ones via the adapter, output byte-identical). `VideoUpscaler` gains `.swiftVR` (second phase `.flashVSRTiny`). `UpscaleFactor.fits` gains the pixel budget from `h3_vsr_requirements` |
| `UpscaleSheet.swift` | Estimated time (`estimated_seconds_per_chunk × chunks`), a "research / non-commercial" badge, a "Prepare model" button when weights or cache are missing |
| `ModelManagerView.swift` / `ModelLibrary.swift` / `ModelDownloader.swift` | New "Upscaling models" section: download from HF (`ModelDownloader`, resumable), license acceptance (date stored in `ModelLibrary`), cache-build progress, precision choice (M5 = int8 default / others = BF16) |
| `GenerationViewModel+API.swift:68-107` | Accept `"upscaler":"swiftVR"` only when available and accepted; 409 while generating |
| `GenerationViewModel.swift` | VSR methods also guard `isGenerating` |
| `Packaging/{en,ja}.lproj/Localizable.strings` | New strings (`check_localizations.sh`) |
| `THIRD_PARTY_NOTICES.md`, `.gitignore` (`*.h3vc`) | SwiftVR (code Apache-2.0 / weights Research License, not bundled), TAEHV MIT, FlashVSR Apache-2.0 |

### §6.2 Public C API (`h3_vsr.h`)

```c
typedef struct h3_vsr h3_vsr;                  /* loaded model; reused across videos */
typedef struct h3_vsr_session h3_vsr_session;  /* one video: carries causal state */

typedef enum { H3_VSR_SWIFTVR = 1, H3_VSR_FLASHVSR_TINY = 2 } h3_vsr_kind;
typedef enum { H3_VSR_PRECISION_AUTO = 0, H3_VSR_PRECISION_BF16 = 1, H3_VSR_PRECISION_INT8 = 2 } h3_vsr_precision;
typedef enum { H3_VSR_BGRA8 = 0, H3_VSR_RGB24 = 1 } h3_vsr_pixel_format;

typedef struct {
    h3_vsr_kind kind; char name[64];
    int fixed_factor_x100;            /* 0 = any output size (SwiftVR), 400 = 4x only (FlashVSR) */
    int output_multiple;              /* 32 (SwiftVR: internal pad+crop), 128 (FlashVSR: crop) */
    int min_output_side, max_output_side;
    int first_chunk_frames, chunk_frames;   /* 28, 24 */
    uint64_t weight_bytes_bf16, weight_bytes_int8;
    int int8_supported;               /* h3_gpu_is_m5() */
    char license_id[64];              /* "SwiftVR-Research-License" / "Apache-2.0" */
    int commercial_use_allowed;
} h3_vsr_info;
int  h3_vsr_inspect_dir(const char *model_dir, h3_vsr_info *info, char *error, size_t error_size);

/* weight cache */
int  h3_vsr_cache_path(const char *model_dir, h3_vsr_precision p, char *path, size_t size);
int  h3_vsr_cache_matches(const char *cache_path, const char *model_dir);  /* header + fingerprint only */
int  h3_vsr_build_cache(const char *model_dir, const char *cache_path, h3_vsr_precision p,
                        const char *shader_source_path, h3_progress_callback on_progress,
                        void *opaque, char *error, size_t error_size);     /* phase "vsr cache", n/30 */

typedef struct {
    h3_vsr_precision precision;   /* AUTO: INT8 on tensor hardware, BF16 elsewhere */
    const char *cache_path;       /* required; raw F32 safetensors are never run directly */
    const char *shader_source_path;
    int dit_overlap;              /* latents prepended from the previous chunk; 0 = the official restore_video */
    int reae_frame_batch;         /* 0 = auto from memory */
    int reae_separate_queue;      /* 1: ReAE on a second h3_gpu + thread (default 1) */
    int keep_fc2_bf16;            /* quality fallback if int8 FC2 fails parity */
} h3_vsr_open_params;
h3_vsr *h3_vsr_open(const char *model_dir, const h3_vsr_open_params *params,
                    h3_progress_callback on_progress, void *opaque, char *error, size_t error_size);
void    h3_vsr_close(h3_vsr *vsr);

typedef struct {
    int supported; char reason[256];
    uint64_t gpu_bytes, host_bytes;
    double estimated_seconds_per_chunk;   /* bench calibration scaled by token count */
    int tokens_per_chunk, windows_even, windows_odd;
} h3_vsr_requirements;
int h3_vsr_check_requirements(const h3_vsr *vsr, int in_w, int in_h, int out_w, int out_h, h3_vsr_requirements *out);

h3_vsr_session *h3_vsr_session_open(h3_vsr *vsr, int in_w, int in_h, int out_w, int out_h,
                                    int total_frames /* 0 = unknown */, char *error, size_t error_size);
int  h3_vsr_session_next_chunk_frames(const h3_vsr_session *s);   /* 28, then 24 */

typedef struct { const uint8_t *pixels; int width, height, stride_bytes; h3_vsr_pixel_format format; } h3_vsr_image;
typedef struct { uint8_t *pixels; int width, height, stride_bytes; h3_vsr_pixel_format format; } h3_vsr_image_out;

/* Feeds one input chunk and writes 0..N finished output frames (in order, one-chunk lag;
 * is_last flushes everything so total out == total in). Progress phases: "vsr encode",
 * "vsr dit" (block/30), "vsr decode". A non-zero callback return or *cancel stops at the next
 * block/phase boundary (error "vsr cancelled"). */
int  h3_vsr_session_process(h3_vsr_session *s, const h3_vsr_image *in, int in_frames, int is_last,
                            h3_vsr_image_out *out, int out_capacity, int *out_frames,
                            h3_progress_callback on_progress, void *opaque,
                            const volatile int *cancel_requested, char *error, size_t error_size);
int  h3_vsr_session_pending_frames(const h3_vsr_session *s);
void h3_vsr_session_close(h3_vsr_session *s);
int  h3_vsr_get_stats(const h3_vsr *vsr, double *dit_seconds, double *reae_seconds, uint64_t *peak_live_bytes);
```

Output count rule (SwiftVR): call 1 (28 frames in) → 0 out; call 2 (24 in) → chunk 1's 25 frames; then 24 per call; `is_last` → the previous chunk's remainder plus the last chunk (padded internally to 4b+1, 3 encoder copies, 4b+4 decoded, cropped to the real count).

### §6.3 Weight pipeline (on-device conversion and cache)

- Inputs: `reae.safetensors` (**F32, 128 tensors** [verified: header]), `prompt_embedding.safetensors` (`prompt_emb` **BF16 [1,512,4096]** [verified]), `transformer/diffusion_pytorch_model.safetensors` (**F32, 825 tensors, every linear has a bias, `attn1/attn2.norm_q/norm_k.weight [3072]` present** [verified: header]), `transformer/config.json`.
- `h3_vsr_build_cache` has the same shape as `h3_build_attention_cache` (own `h3_gpu`, `h3_weight_store`, per-block progress, `.partial`→rename, unlink on failure). Per tensor: `stream_file_i8` (raw F32 with F_NOCACHE) → `h3_gpu_dequantize(F32→BF16)` (Q,K,V read into consecutive row ranges and fused into `[9216,3072]`) → for int8 `quantize_weight_int8` → read back and `fwrite`. Reading 20 GB dominates → **1–2 minutes [estimate]**, peak RSS a few hundred MB.
- Precision per tensor:

| Tensor | Stored as | Why |
|---|---|---|
| `attn1.to_{q,k,v}` fused `[9216,3072]`, `attn1.to_out.0`, `attn2.to_q`, `attn2.to_out.0`, `ffn.net.0.proj [14336,3072]`, `ffn.net.2 [3072,14336]` | int8 per-row + F32 scales (M5) / BF16 (M4, quality mode) | 3072=24·128, 9216=72·128, 14336=112·128 satisfy the int8 constraints |
| All linear biases | F32 | added in F32 epilogues |
| `attn1/attn2.norm_q/norm_k.weight` | F32 | RMSNorm across heads |
| `norm2.weight/bias` | F32 (+BF16 copy) | affine LN before cross-attention |
| `blocks.b.mod` = `scale_shift_table[6,3072] + time_proj` [6,D] (raw shift, scale, gate, c_shift, c_scale, c_gate; `(1+scale)` is formed at load) | F32 | t=1000 fixed |
| `attn2.to_k/to_v` not stored; instead **`K2_b = RMSNorm_k(ctx·Wk+bk)`, `V2_b = ctx·Wv+bv` `[512,3072]`** | BF16 | fixed prompt. 6.3 MB/layer, 189 MB total |
| `patch_embedding [3072,48,1,2,2]` → `[3072,192]` + bias | BF16 + F32 | |
| `norm_out` modulation (`scale_shift_table[2,3072]+temb`) | F32 | |
| `proj_out [192,3072]` + bias | BF16 + F32 (bias folded into unpatchify) | |
| Condition embedder (time 256→3072→3072, time_proj 3072→18432, text 4096→3072→3072) | **not stored**; evaluated once on the CPU in double/F32 at build time | its outputs are the constants above |
| ReAE 40.95M | **not in the cache**: read from `reae.safetensors` (F32, 164 MB) at load time and converted to the precision needed; only its fingerprint is recorded in the header | small enough to skip conversion; an F32 quality mode stays possible |

- Constant evaluation (CPU, build time): `temb = L2(SiLU(L1(sinusoidal_256(1000.0))))` (diffusers `get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0)`), `time_proj = L(SiLU(temb))`→6×3072, `mod_b = scale_shift_table_b + time_proj`, `ctx = L(GELU_tanh(L(prompt_emb)))`, `K2_b/V2_b`. **Settle whether attn2 applies norm_q/norm_k by matching block 0 in the harness.**
- Cache format `swiftvr-dit-{int8|bf16}.h3vc` (next to the model directory, gitignored like `.h3ac`): 128 B header `{magic "H3VC", version, model_kind, precision, blocks, dim, ffn, heads, head_dim, text_tokens, patch_in, latent_channels, fingerprint[32] (transformer/), reae_fingerprint[32] (reae+prompt), table_offset, table_count}`, a name table `{name[96], dtype, rows, cols, offset, bytes}` (64 B aligned so `stream_file_*` can pread directly), then payload in block order. The name table is what lets one format hold both models.
- Size: 144.7M params per block → int8 ≈ 30×(144.7+0.13+6.3+0.3) MB ≈ **4.55 GB**, BF16 ≈ **8.9 GB**, ReAE 82 MB. The 20.2 GB download can be deleted after a successful build ("delete original", default: keep).
- Resident vs SSD: **resident by default on every target**. M5 int8: 4.6 GB + DiT activations ≈1.8 GB + ReAE ≈2 GB + host 0.6 GB + MPS scratch ≈1 GB ≈ **10 GB**. M4 BF16 ≈ **14.3 GB**. SSD double slots (`h3_dit.c:894-980`) are a later addition for 16 GB machines (the API leaves room). Reading 4.6 GB by pread ≈ 2–3 s.

### §6.4 Per-chunk compute (960×544 → 1920×1088, 24 frames, T=6) [estimate: from the formulas]

- Geometry: output 1920×1088 (multiples of 32, no padding) → latent 120×68×48 → token grid W=60, H=34 → 2040 per latent frame → **N = 12,240** (first chunk T=7 → 14,280).
- Windows: H=34 even layers `{0,16,18}`, odd `{0,8,18}`; W=60 even `{0,16,32,44}`, odd `{0,8,24,40,44}` → **Nw = 12 / 15, Lw = 1536**. Gathered rows 18,432 (1.51N) / 23,040 (1.88N). Owner widths: even layers h 16/16/2, w 16/16/16/12; odd layers h 8/10/16, w 8/16/16/4/16.
- FLOPs per block: QKV 0.693 T, out 0.231 T, attn2 q 0.231 T, attn2 out 0.231 T, FFN 2.156 T → **linears 3.54 TFLOP**. Self-attention: full windows 0.348/0.435 T, **owner-only queries (exact) 0.231 T**, dense would be 1.84 T. Cross-attention 0.077 T. Gather/scatter ≈0.65 GB, elementwise ≈0.8 GB. Block ≈3.85 TFLOP, **chunk ≈116 TFLOP** (consistent with the H100's 328 ms ⇔ 353 TFLOP/s).
- ReAE per chunk: encoder ≈1.9 T, decoder naive 15.0 T → **12.4 T with the upsample fold**. Total ≈**14.5 TFLOP** (12.5% of the DiT).
- Time (assumptions: NAX BF16 14 TFLOP/s measured, int8 ≈ 2× [bench needed], MPSGraph SDPA 11 TFLOP/s, NAX tapconv ≈10 TFLOP/s, M4 Pro MPSGraph ≈6 TFLOP/s and F16 conv ≈5 TFLOP/s **[unmeasured]**):

| | block | DiT×30 | ReAE | **middle chunk** | 5 s / 124 fr (6 chunks: 28+4×24+1) | 60 s / 1440 fr (60 chunks) |
|---|---|---|---|---|---|---|
| M5 int8 | 127 lin + 21 attn + 7 cross + 10 ≈ 165 ms | 5.0 s | 1.45 s | **≈6.5 s** | ≈34 s | ≈6.5 min |
| M5 BF16 | ≈291 ms | 8.7 s | 1.45 s | ≈10.2 s | ≈53 s | ≈10 min |
| M4 Pro BF16 | ≈670 ms | 20 s | 2.9 s | ≈23 s | ≈2 min | ≈23 min |
| M4 base BF16 | ≈1.1 s | 34 s | 4.8 s | ≈39 s | ≈3.3 min | ≈40 min |

- Peak memory: weights 4.6/8.9 GB, DiT activations ≈1.8 GB (hidden 75 MB, qkv 226 MB, gather 3×142 MB, FFN intermediate 351 MB, int8 scratch 176 MB), ReAE ≈1.5–2 GB, host 0.6 GB, MPS scratch ≈1 GB → **≈10 GB (int8) / ≈14.3 GB (BF16)**. Coefficients for `h3_vsr_requirements`: ≈150 KB per token + ≈1 KB per output pixel + weights (calibrated from `h3_gpu_stats.peak_live_bytes`). On 24 GB machines refuse outputs above ≈4.5 M pixels (the UI greys the factor out).

### §6.5 Kernel plan

**Reused as-is**: `h3_gpu_linear_int8_bf16` (the block's 5 linears), `h3_gpu_linear_bf16` (BF16 path; patch_embed 192→3072 has a bias and in<256, so it takes the 16×16 tile kernel, 14 GFLOP, acceptable; `proj_out` on NAX), `h3_gpu_quantize_weight_int8`, `h3_gpu_layer_norm_bf16` (norm1/norm3 with weight=(1+scale), bias=shift; norm2 affine), `h3_gpu_copy_bf16` (MemBlock state), `h3_gpu_dequantize`, `stream_file_*`, `h3_weight_store_*`, `begin/continue/submit`, `h3_gpu_stats`.

**Small generalizations**:
- `void h3_gpu_set_nax_generic(h3_gpu*, int on)` — makes `naxShape` true for any shape meeting in%32/out%64/rows≥128/no-bias (`h3_gpu.m:3151`).
- `int h3_gpu_sdpa_batched_bf16(gpu, out, q, k, v, batch, q_len, kv_len, heads, head_dim, scale)` — add `kv_sequence` to the existing `h3_gpu_sdpa_graph` (K/V placeholders `[batch, kv_len, heads, head_dim]`, kv_len in the cache key). Shared by MFSWA (batch=Nw or owner class), cross-attention (batch 1, kv 512) and FlashVSR's top-k gather.
- `h3_gpu_conv2d_bf16_mps(...)` NHWC MPSGraph conv with explicit padding (M4 path). If MPSGraph rejects BF16 convolution, cast to F16 inside the graph (probe once in `h3_gpu_create`).

**New kernels** (all `h3_vsr_*`, F32 math, BF16 storage; each with a CPU double reference in `tests/test_vsr.c`):
1. `h3_vsr_bilinear_up_to_unshuffled_bf16` (BGRA8 → bilinear `src=(dst+0.5)·scale−0.5` clamped, no antialias → /255 → zero pad right/bottom to multiples of 32 → pixel_unshuffle(2) channel order (c,ph,pw) → `[T, Hp/2, Wp/2, 12]`). MPSGraph `resize(bilinear, centerResult=YES, alignCorners=NO)` uses the same half-pixel centers as OpenCV/TF2 [verified: SDK header] and is an alternative, but fusing the whole preprocessing into one kernel is faster.
2. `h3_vsr_shuffle_to_pixels` (clamp[0,1] → ×255 → truncate (reference) or round (option) → BGRA8/RGB24, crop).
3. **`h3_vsr_tapconv_bf16_nax` (M5) — a general tap-accumulated matmul2d kernel**: `out[rows,Cout] = epi(Σ_tap X[row+off_tap,:]·W_tapᵀ (+ Σ Past[...]·W2_tapᵀ) + bias)`, args `{rows, Cin, Cout, taps, offsets[16], relu, residual_ptr, k_split}`. Same `multiply_accumulate` as `h3_linear_int8_nax_r128` (`h3_shaders.metal:2597-2650`) but on bfloat. Input in a zero-padded channels-last `[(T), H+2, W+2, C]` layout (a tap = a constant row offset). Uses: 3×3 conv (9 taps), the MemBlock's first conv without materializing `cat[x,past]` (K split: 9 taps on x with `W[:, :C]`, 9 on past with `W[:, C:]`), 1×1, TPool (2 taps on frame pairs, weight `[C,2C]` split), TGrow conv3d(3,1,1) (3 taps at ±1 frame), stride-2 3×3 as space-to-depth(2) + 2×2 taps (K=16C, +0.2 T/chunk), **Upsample2×→conv3×3 folded** (4 phase kernels = 2×2 taps with `4·Cout` outputs + depth-to-space; exact because nearest upsampling commutes; phase weights are tap sums formed in F32 at load). Channel padding Cin 12→32, Cout 12/48→64.
4. `h3_vsr_s2d_bf16`/`h3_vsr_d2s_bf16`, `h3_vsr_clamp_tanh_bf16`, `h3_vsr_pad_copy_bf16`.
5. `h3_vsr_patchify_bf16` (`[T,68,120,48] → [N,192]` order (c,ph,pw)), `h3_vsr_unpatchify_sub_f32` (`z_hq = z_lq − (v + b_proj_out)` to an F32 latent).
6. `h3_vsr_qk_rms_inverse_bf16` (`qkv [N,9216]`, bq, bk → `inv[N,2]`) and `h3_vsr_window_gather_qkv_bf16` (fuses bias + RMSNorm across heads + **interleaved RoPE** (22 t pairs, 21 h, 21 w) + gather → `Qg[rows_q,24,128]`, `Kg/Vg[rows_kv,24,128]`).
7. `h3_vsr_window_scatter_bf16` (`Og[rows,3072]` → `O[N,3072]`, owner/permutation).
8. `h3_vsr_bias_gate_add_bf16` (`x += (y+b)⊙g`, g NULL means 1), `h3_vsr_gelu_bias_bf16`, `h3_vsr_bias_rms_bf16` (attn2 q).
9. Second-stage fusions: `h3_vsr_gelu_bias_quantize_int8`, `h3_vsr_layer_norm_mod_quantize_int8` (save one 351 MB / 75 MB pass each).

**Exact algorithmic device — owner-only queries**: each output token is read from exactly one window, so queries at non-owner positions are discarded. Gathering only the owned query rows per window (K/V still the full window) gives Σowned = N, attention FLOPs of 0.231 T/block for both parities (−33%/−47% vs full windows), and an output that is already contiguous (the scatter becomes a permutation). Windows are grouped into classes by owned-row count (even layers 4 classes {1536,1152,192,144}, odd layers 7 classes), one `h3_gpu_sdpa_batched_bf16` call per class. **Full-window version first for parity, owner-only second.**

**RoPE offset device**: with `dit_overlap=0`, self-attention is confined to the chunk and RoPE depends only on relative position, so the temporal offset cancels → `t_off=0` for every chunk (one 6.3 MB table per shape class). The `rope_max_seq_len=1024` concern for long videos disappears too. Verify in the harness that block 0 with t_off 0 vs 6 agrees to ≥60 dB.

**Per-block op sequence (SwiftVR)**: LN(norm1-mod) → int8 QKV → rms_inverse → gather(bias, norm, RoPE) → batched SDPA → scatter → int8 out → bias_gate_add(gate_msa) → LN(norm2 affine) → int8 q2 → bias_rms → cross SDPA(K2,V2) → int8 out2 → bias_gate_add(gate NULL) → LN(norm3-mod) → int8 fc1 → gelu_bias → int8 fc2 → bias_gate_add(c_gate). Final: LN(norm_out-mod) → NAX BF16 proj_out → unpatchify_sub. `h3_gpu_continue` every few blocks, cancel check between blocks.

### §6.6 Pipelining and cancellation

Swift (`VideoUpscale.upscale` becomes 3 stages; existing methods go through `PerFrameChunkAdapter` (chunk 1) on the same path):
1. **Reader** (`h3c.vsr.read`): AVAssetReader BGRA8 frames grouped `h3_vsr_session_next_chunk_frames` at a time with their PTS, into a bounded queue of depth 2.
2. **Engine** (`h3c.vsr.engine`, serial): lock the input `CVPixelBuffer`s and call `h3_vsr_session_process`; outputs are `CVPixelBuffer`s from the adaptor pool. Output + PTS go FIFO (1:1 with input) to the writer queue.
3. **Writer**: the existing `pump(videoInput)` pops finished frames and appends. Audio pump, staging, replace, metadata and transform are unchanged.

Inside the engine, thread A owns the DiT `h3_gpu`, thread B the ReAE one (`reae_separate_queue`). `process(k)`: B encodes k → A runs DiT k while B decodes k−1 → return chunk k−1's frames (one-chunk lag, `is_last` flushes). MemBlock/TPool state lives on B, so causality is trivially preserved. **Honest expectation**: the GPU is one shared resource, so running two queues hides CPU-side encode/upload latency and lets ReAE's small kernels fill DiT bubbles (10–30% of ReAE time [estimate]); the per-chunk floor is the GPU sum.
Cancellation: `cancel_requested` is checked between DiT blocks and phases (the `stop_if_cancelled` style of `h3_dit.c`), both threads read the same flag, `process` returns 0 with "vsr cancelled". Swift stops the writer and removes the staging file (as today). Progress = (frames_done + chunk_frames·block/30)/expected.

### §6.7 ReAE on the Neural Engine (Core ML) — go/no-go rule

- Expected effect: move ReAE's GPU time (≈1.45 s of 6.5 s per chunk on M5 int8 ⇒ up to −22%, −13% on M4 Pro) to the ANE, running alongside the DiT.
- Mechanism (if GO): the weights cannot be redistributed, so ship a **weightless `.mlpackage` template** per resolution class (built with coremltools at dev time from the Apache-2.0 architecture, weights referenced as `BLOBFILE`), write `weights/weight.bin` (coremltools `blob_writer` format) on the device from `reae.safetensors`, and compile with `MLModel.compileModel(at:)` [verified: SDK]. MemBlock/TPool states as model inputs/outputs (or `MLState` [verified: macOS 15 SDK]), F16, static shapes.
- Decision (measured before committing): build one stage-3 MemBlock (480×272×128, 12 frames, F16) with coremltools on the dev machine and measure on M5. **GO iff** the extrapolated full-ReAE ANE time per chunk ≤ 0.8 × DiT time (fully hidden) **and** its output PSNR ≥ 45 dB vs F32 **and** input/output copies take ≤20% of its runtime.
- Default for v1: **NO-GO** (ship with GPU NAX tapconv (M5) / MPSGraph (M4); ANE is a P7 experiment).

## §7 App integration

- Add **`.swiftVR`** to `VideoUpscaler` (display name proposal: "AI super-resolution (SwiftVR, non-commercial)"). `isAvailable` = converted cache present ∧ Metal available.
  Reserve `.flashVSRTiny` for FlashVSR Tiny (second phase).
- **New protocol `ChunkedFrameScaler`** (alongside `FrameScaler`, §6.1/§6.6): `func chunkFrames() -> Int`,
  `func process(_ frames: [CVPixelBuffer], isLast: Bool, into pool: CVPixelBufferPool) throws -> [CVPixelBuffer]`.
  The existing four methods go through `PerFrameChunkAdapter` (chunk 1) on the same 3-stage pipeline and keep byte-identical output.
  Output is in PTS order, one output per input (one-chunk lag, flushed by `isLast`). Progress per block from the engine callback; cancel at block boundaries plus `cancel_requested`.
- Factors: keep the current 1.5/2/3/4 contract. SwiftVR works at any factor because it bilinearly resizes the input to the target before restoring, but **the first release recommends 2x only** (measure time and quality before 4x and above; add a way for `UpscaleSheet` to disable unsupported factors).
- Weight acquisition UI: the SwiftVR row in the Enlarge sheet gets a "Download model (about 20.2 GB → about 5.3 GB [int8] / 10.3 GB [BF16] after conversion)" button,
  a notice in the style of `ModelLicenseNotice` (Research License, non-commercial, HF link). Download via `ModelDownloader` (resumable).
  Conversion runs in the engine's C code tensor by tensor (peak RSS < 2 GiB [target]); the F32 original is deleted afterwards (kept if the setting says so). Location `~/Library/Application Support/H3cApp/Upscalers/SwiftVR/`.
- Records: `upscaler="swiftVR"`, `upscaledSize`, new `upscalerModelVersion` (the cache fingerprint).
- API: add `swiftVR` to `upscaler` (update `API(.ja).md`). If not downloaded, 400 with the reason.
- Documents: one item in `GUIDE(.ja).md` "Enlarge afterwards", an "AI super-resolution" section in `TECHNICAL(.ja).md`, SwiftVR in `THIRD_PARTY_NOTICES.md` (attribution for the Apache-2.0 code; the weights are not bundled, so the license is shown at download), one row in the `guide.html` table on the site, `website-facts.md` updated.
- Continuation of enlarged copies keeps working (only the record's `upscaler` string gains a value).

## §8 Implementation phases and acceptance criteria

| Phase | Content | Done when (measure and record) |
|---|---|---|
| P0 Document | Add this plan as `UPSCALE_PLAN(.ja).md` (branch `upscale-plan`) | merged |
| P1 Reference data | Run the official SwiftVR on CPU/MPS in a scratch venv on a small input (8 fr 256x256 → 512x512) and dump the §9 checkpoints and constants as safetensors (`tools/dump_swiftvr_fixture.py`; Python is dev-only). The same script also writes the window starts and owner rule | dump set and README |
| P2 Weight conversion | `h3_vsr_weights.c` + CLI `build_vsr_cache`: F32 safetensors → `.h3vc` (BF16 / int8 per-row, F32 constants, cross K/V, fused QKV, fingerprint). `h3_gpu_set_nax_generic` here too | conversion time and size recorded, reload matches, host part of `test_vsr` |
| P3 ReAE | `h3_vsr_reae.c` + tapconv NAX kernel + MPSGraph path (state carry, frame batching, TGrow first, upsample fold) | ReAE latent ≥ 40 dB (BF16) vs reference, decoded RGB ≥ 38 dB; both paths agree |
| P4 DiT | `h3_vsr_wan.c` + `h3_vsr_window.c`: full-window MFSWA → owner-only, BF16 → int8, constant conditioning, cross-attention, RoPE (verify t_off=0) | block 0 ≥ 40 dB, final latent ≥ 36/33 dB, RGB ≥ 38/35 dB (BF16/int8); record the difference from dense attention. Real weights (Appendix A.1): v vs the official float32 run 60.0 dB bf16 / 51.3 dB int8 (MFSWA), 60.4 / 51.7 dB dense |
| P5 Streamer | `h3_vsr_chunk.c`/`h3_vsr.c`: chunking, state, last-chunk front fill, preprocessing and cropping, two contexts in parallel, cancel | 124-frame end-to-end ≥ 35 dB vs the reference's whole output; time and peak-memory table on M5/M4 (`bench_vsr`). Real weights: the 8-bit output of 57 frames 384x272 -> 768x544 matches the official run at 51.2 dB bf16 / 49.3 dB int8, counts match |
| P5.5 Quality evaluation | §9.5: evaluation set, SeedVR2 reference output (a ≥64 GB Mac or cloud), metric scripts, fill the table for every variant of this implementation, blind A/B | record the achieved ratio vs SeedVR2 (§1's 95%) and choose the defaults of the "quality" and "speed" modes |
| P6 App | `ChunkedFrameScaler` and the 3-stage pipeline, **quality / speed mode switch (full attention, precision, overlap)**, `.swiftVR`, download/acceptance/conversion UI, API, records, documents, localization, `isGenerating` guard | the existing four methods are byte-identical and unchanged; the new method works from both the project and file entry points and with continuation. Real weights: the rebuilt app enlarged a 512x512, 39-frame clip to 1024x1024 HEVC (sound kept) in about 30 s through `POST /api/upscale-file` (swiftVR, 2x) on 2026-10-29; the SwiftVR setup sheet itself was not clicked through |
| P7 Optimization | in progress (Appendix A.2: the ReAE convolutions moved to TensorOps, a 1080p middle chunk 17.6 s -> 9.6 s) | `h3_vsr_tapconv_bf16_nax`, `h3_vsr_reae.c`, `tests/bench_vsr.c` | `tests/test_vsr_tapconv.c`, the TensorOps part of `tests/test_vsr_reae.c`, real weights 56 to 60 dB |
| P8 FlashVSR Tiny | Second back-end on the shared engine (LQ_proj_in, KV cache, block-selection attention, TCDecoder, color correction) | same parity and speed tables |
| PX Gating experiment | Regeneration with H3 itself (Open-H3-Regenerate-2K, §11: starting-latent API → latent upscaler → existing LoRA stacking → A/B against SwiftVR at 1344×768) | the A/B table of §11.3 and the recorded decision |

Phases can be separate branches (P2/P3/P4 depend on P1's dumps).

## §9 Verification plan

- **Fixtures** (local, gitignored under `misc/fixtures/`; derived from licensed weights, so never distributed): `tools/dump_swiftvr_fixture.py` runs the official pipeline on 8 frames of 256×256 (bf16 and f32) and saves
  `x.lq_rgb`, `reae.z_lq`, `dit.block0_out` (also the t_off=6 variant), `dit.final_latent`, `z_hq`, `rgb_out`, plus the constants (`temb`, `mod[30,6,3072]`, `context`, `K2_0`, `V2_0`) as safetensors.
- **`tests/test_real_vsr.c`** (skip-guarded): PSNR / relative RMS at each checkpoint. Pass lines [targets, revised by measurement]: constants ≥ 60 dB vs f32; ReAE latent ≥ 40 dB (BF16); block 0 ≥ 40 dB (BF16 and int8); final latent ≥ 36 dB BF16 / ≥ 33 dB int8; decoded RGB ≥ 38 dB BF16 / ≥ 35 dB int8 (vs the bf16 reference), mean per-channel bias < 0.5/255; t_off 0 vs 6 ≥ 60 dB. If int8 FC2 fails, `keep_fc2_bf16` becomes the default.
- **`tests/test_vsr.c`** (always in `make test`): host — the axis starts reproduce `{0,16,18} {0,8,18} {0,16,32,44} {0,8,24,40,44}`, every token is owned exactly once by a window containing it, owner-class sums equal N, chunk plans sum to the total for 1…200 frames, RoPE table vs double, patchify roundtrip, fold-weight equivalence; GPU — every kernel in §6.5 vs CPU double, batched SDPA vs looped `h3_gpu_sdpa_bf16` (relative 1e-2), cross SDPA vs CPU, determinism (two runs bit-identical).
- **`tests/bench_vsr.c`**: synthetic weights at the real shapes, prints ms/block and ms/chunk, replacing the §6.4 estimates and the UI's `estimated_seconds_per_chunk`.
  Principle: never judge from numbers alone; compare both outputs' full-resolution frames side by side.
- **Quality (fidelity)**: add a SwiftVR column to the existing comparison (half-size copy → 2x → compare with the original; PSNR/SSIM/detail/flicker). Visual check on 3 generated scenes (live-action-like, anime-like, people).

### §9.5 Quality program for getting close to SeedVR2

1. **Evaluation set** (`misc/quality/set-v1/`, gitignored): 6 H3cApp videos (960x544 and 672x384; live-action-like people / anime-like / landscape / dark scenes / fur / text, 2–5 s each) + 2 real videos (1080p downscaled to 540p, so the originals are ground truth). Record seed, prompt and settings in a list.
2. **SeedVR2 reference output** (once, during development): it does not run on a 24 GB machine, so either **(a) mlx-gen (SeedVR2-3B, Apache-2.0) on a Mac with ≥64 GB**, or **(b) the official SeedVR2-3B on a rented cloud GPU (A100/H100 for a few hours)**, on the whole set at 2x, saved under `misc/quality/seedvr2/`. Record the settings (resolution, steps, seed, tiling, color correction).
   If neither is possible, use the SeedVR2 comparison values from the SwiftVR and FlashVSR papers as a provisional baseline, marked **[open]**.
3. **Compared systems**: SeedVR2-3B (baseline) / official SwiftVR (PyTorch, bf16) / this implementation (BF16 full attention, BF16 MFSWA, int8 MFSWA, overlap 0/1, 2x directly vs 4x→Lanczos 1/2) / Real-ESRGAN General / Lanczos.
4. **Metrics** (dev-only Python, `tools/quality/`): no-reference MUSIQ, MANIQA, CLIP-IQA (pyiqa), DOVER for video; temporal consistency = adjacent-frame warp error (RAFT optical flow) and the existing flicker metric; PSNR/SSIM/LPIPS vs ground truth for the 2 real videos; **distance to the SeedVR2 output** = LPIPS / DISTS (both are generative, so PSNR is a reference value only).
5. **Visual**: 3 full-resolution stills and a 2-second loop side by side, blind A/B by the user (SeedVR2 vs this implementation), scored on 5 items: skin, hair, fur, text, dark areas.
6. **Quality levers, in evaluation order** (what to move when a gap appears):
   1. **Full attention (dense SDPA)**: paper Table 3 shows full attention +0.28 dB PSNR and better LPIPS than MFSWA. +≈4.4 s per chunk on M5 (1.84 vs 0.23 TFLOP/layer); MPSGraph's fused SDPA does not materialize the score matrix [verified: existing use of h3_gpu_sdpa]. Default candidate for the "quality" mode.
   2. **Precision**: int8 → FC2 only in BF16 → all BF16. Judge by both per-block PSNR and the final metrics.
   3. **`dit_overlap=1`** (the official stream's default): less flicker at chunk boundaries.
   4. **How 2x is made**: directly at the 2x target vs generate at the training-like 4x and Lanczos down by 1/2 (the method Real-ESRGAN already uses here). Compare texture and time.
   5. Run ReAE in F32 (decoder tonal precision).
   6. Input preprocessing: try bicubic / Lanczos instead of bilinear, carefully, since it differs from the training distribution.
   7. Fill the same table for the second back-end (FlashVSR Tiny or FlashVSR+) and make the one closest to SeedVR2 the default.
7. **Records**: the result tables go in an appendix of `UPSCALE_PLAN(.ja).md`, in the style of `SPEEDUP_ROADMAP.md` (machine, settings, seed, date).

## §10 Performance and memory estimates (summary) [estimate]

Details in §6.4. In short: 960x544→1920x1088 with 24-frame chunks is **≈6.5 s per chunk on M5 int8, peak ≈10 GB** (5 s clip ≈34 s, 60 s video ≈6.5 min),
**M5 BF16 ≈10 s / ≈14.3 GB**, **M4 Pro BF16 ≈23 s** (5 s clip ≈2 min, 60 s ≈23 min), M4 base ≈39 s per chunk.
The assumptions (int8 = 2× BF16, M4's effective MPSGraph TFLOP/s) are unmeasured and are replaced by `tests/bench_vsr.c`. The §1 targets (M5 under 2 minutes / M4 under 6 minutes for a 5 s clip) are within this estimate.

## §11 Gating experiment: regeneration with H3 itself (hires-fix → Open-H3-Regenerate-2K)

Revised 2026-10-31. The earlier idea (re-noising a finished latent ad hoc, "hires-fix") is replaced by **Open-H3-Regenerate-2K** (NVlabs, released 2026-10-27), public weights trained for exactly that idea. The goal is a **decision**, not a feature port, and there is one decision point:

> **If even 1344×768 is not clearly better than SwiftVR 2x on faces, skin, hair and text, do not proceed to 1080p / 2K.**

### §11.1 Facts about the target [verified 2026-10-31: HF model card, NVlabs README / project page, two ComfyUI ports]

- Components: on the frozen H3 DiT and VAE, (a) a merged LoRA (FP32, rank 576 = refiner 384 + LightX2V FL2VA Turbo v1.2 128 + LMS Sharp v1.0 64 concatenated along the rank axis, 12.5 GB, **312 linears** = 50 blocks × 6 + 2 token-refiner blocks × 6), (b) the LMS Sharp AdaLN curve branches (50 of them, FP32 1.24 GB; `B·A·curve8(t)` added to each block's AdaLN projection, where the 8-dimensional curve is read from the 1025×8 `adaln_t_table` in Comfy-Org's pruned checkpoint), (c) a community 3D-conv latent upscaler (LBH-123-AI, BF16 691 MB, **345.3M parameters, 322 tensors**), (d) a fast VAE decoder (StreamRender-H3, the 36-layer decoder pruned to 24 layers and distilled, non-causal, 1.62B, FP32 6.47 GB, needs FlashAttention-4).
- Procedure: upscale the latent straight onto the target grid (no decode, no re-encode) → `x = (1−σ₀)·z_up + σ₀·ε`, σ₀ = 0.909375 → 3 Euler steps (σ = 0.909375, 0.857143, 0.666667, 0; CFG 1; model time = 1−σ) → decode. The 2-step variant uses σ = 0.7, 0.494118, 0 (README: "about 30% faster but visibly softer on fine texture (fur, moss)"). `--start-sigma` changes the start noise. Audio is "fixed encoded silence". **Every step is one DiT call over all tokens at the output resolution**; there is no tiling or windowing (sol_engine uses Sol-Attn sparse attention).
- Outputs: 768p (refine only, no upscale) / 1080p / 2K (default) / 4K (two 2x passes via 2K). The README documents two inputs, 1344×768 and 1024×576 (384P/512P appear only in the blog). 5 s (121 frames, 37 latent frames) is the standard, up to 15 s in one pass.
- Performance (one B200, 5 s, 768P → 2K, warm, median): dense BF16 + released VAE 87.26 s (refine 71.27 s, decode 14.99 s, upscale 1.01 s); sol_engine_fast + fast VAE 22.45 s (3.9x — **relative to their own dense implementation**; the headline "5x over the official API" has no timing behind it on the page). The CUDA-graph engine needs "about 150 GB on one GPU". A port on a 72 GB card (int8 base): refine 122 s (sparse) / 474 s (dense); the fast decoder alone peaks at 21 GB for 2K 5 s and 45 GB for 15 s.
- Quality claims: a blind study of 200 judgements (50 cases) prefers it 87% **over the official H3-Regenerate-2K API**. VBench average 0.814 vs 0.802, UniPercept 60.0 vs 53.4. No comparison against other upscalers (SwiftVR, SeedVR2). The README itself says "same-seed PSNR is not a usable metric". The fast decoder is 30.0 dB from the released decoder (37.8 dB per the porter).
- Known weakness: σ₀ 0.909 redraws a lot; the porter reports a fox face drifting toward a wolf (same with the curve branches off and via decode, so it is the method). start σ 0.8 mitigates.
- Licenses: merged LoRA, fast VAE and base_model under the MiniMax H3 Community License. The latent upscaler ships with an Apache-2.0 `latent_upsampler/LICENSE` in NVIDIA's bundle, while **the original LBH repository is MIT** (NVIDIA's wrapper also says "MIT"). The LightX2V and LMS parts are not stated. The model card itself says the bundle has no blanket license. Ref2VA is not mentioned.
- Mapping to H3cApp: the app's LoRA loader reads the diffusers layout, already swaps FC1 [value; gate] → [gate; value] (the so-called SwiGLU conversion) and handles the two token-refiner blocks (`h3_lora.c`). It reads F32/F16/BF16 and takes the rank from the shapes. Generated videos keep their latents in `.h3lt`, so app-made drafts need no re-encode.

### §11.2 Outlook on a 24 GB Mac [estimate: extrapolated from measurements]

Measurements used: at 768²/124 frames (21,728 tokens) **65.7% of one DiT block is attention** (micro-benchmark in `SPEEDUP_ROADMAP.md`, int8 linears). At 768²/362 frames (about 63k tokens) one step took 657 s with swapping (Stage ⑦). At 1344×768/15 s (about 110k tokens) **one step did not finish in 2.5 hours** (note in `GenerationViewModel.swift`). Counting video tokens only, with 5 s = 37 latent frames:

| Output | Video tokens | Outlook (M5 24 GB, int8) |
|---|---:|---|
| 768p refine only (1344×768) | 37,296 | **10–15 min** for 3 steps + VAE. Memory expected to fit. A hypothesis until measured |
| 1080p (1920×1088) | 75,480 | Beyond the 63k that already swapped. 15–20 min or more per step, over an hour for 3 steps. Requires token-chunked FFN and attention |
| 2K (2560×1440) | 133,200 | Larger than the configuration whose single step did not finish in 2.5 h. Out of scope without Sol-Attn-class reductions |

FLOPs agree: 2K, 5 s, 3 steps ≈ 7×10¹⁶ FLOP (linears 29 GFLOP/token, attention 4·N²·5376 per block). B200's dense 71 s is about 1 PFLOP/s effective; at the M5's effective 10–25 TFLOP/s that is 1.5 hours. **2K is impossible on 24 GB not because it does not fit in memory but because, even chunked to fit, the compute time is impractical.** So the core problem is attention, not the VAE. VAE decode in the app is already int8 and tiled, 15–20% of the total; a 24-layer decoder shrinks only about a third of that (5–7% overall) and is lossy. **The 24-layer VAE comes last** (after Regenerate is adopted and the DiT is sped up with sparse attention).

### §11.3 Implementation order (the minimum the gating experiment needs)

1. **A general starting-latent API** (not Open-H3-specific): add to `h3_params` an `initial_video_latent` (`.h3lt` or array), `video_sigmas[]` (arbitrary σ list), `audio_sigmas[]` or "audio pinned clean". Use the `h3_forced_trajectory_start` formula as is. Use the project's prompt (the README's inference loads the text encoder, so conditioning is live). 1344×768 is inside the engine cap (`h3.c`, "released 768*1344"). The same API serves hires-fix, 2/3-step refiners and regeneration from an intermediate latent.
2. **Port the latent upscaler to Metal** (as a part of the experiment, not a standalone feature): the structure is the author node `upscaler_node.py` (MIT), `LatentResizer3D(in_channels=24, in_blocks=12, out_blocks=12, channels=512, temporal_every=2, temporal_kernel=5, attn=False)`. `conv_in` Conv3d(24→512, 3³) → 12 in_blocks → **trilinear interpolation (align_corners=False) to the target grid** → 12 out_blocks → GroupNorm(32)+SiLU → `conv_out` Conv3d(512→24, 3³). A block is ResBlockEmb3D (GroupNorm(32) → SiLU → Conv3d 3³ → FiLM (scale, shift) from the embedding → SiLU → Conv3d 3³, residual), followed after every even-numbered block by TemporalConv (depthwise Conv3d (5,1,1) → 1×1×1, residual). The embedding is `Linear(1→64) → SiLU → Linear(64→64)` fed `scale − 1`. Input is the **normalized latent** (divided by the 24-channel mean/std; the same role as the app VAE's `latents_mean/std`, values from the author node's constants), output de-normalized. Target aligned to 32 px. The author node chunks in time above 32 latent frames (overlap 5) but NVIDIA's wrapper disables that, so run it whole too. On the GPU side `h3_gpu`'s conv3d (F32 MPSGraph) and the existing GroupNorm+SiLU kernels suffice. Check against a double-precision reference in the `h3_vsr_reae.c` style.
3. **LoRA without new code**: apply Kijai's refiner-only LoRA (rank 384, ranks 0–383 of NVIDIA's merge) + the user's LightX2V FL2VA Turbo v1.2 + LMS Sharp v1.0 r64 through the existing stacking (up to 8). If NVIDIA's merged FP32 12.5 GB file is read directly, check the loader's memory (rank-576 B·A held in F32). **No Sharp curve branches at first** (implement after the decision; precomputing the 3 timesteps × 50 blocks × 96768 F32 deltas at load is 58 MB; fetch Comfy-Org's 32 KB `adaln_t_table` by byte range for the 8-dimensional curve).
4. **A normalized A/B comparison**: fix the input to app-generated **1024×576, 5 s** latents (and 672×384). A: regenerate → 1344×768 (3-step standard / conservative 2-step σ 0.7, 0.494118, 0 / start σ 0.8). B: SwiftVR 2x → 2048×1152 → **high-quality downscale to 1344×768**. C: plain resize → 1344×768. D (optional): the ad-hoc hires-fix (old §11, σ₀ 0.3–0.5). Metrics are the `ltx-refine-eval` tools (`tools/quality_metrics.py`: LPIPS / face LPIPS / flicker / MUSIQ / MANIQA with face crops; view `frames-sheet.png` at 1:1). **Record "fidelity to the source" and "perceived quality" separately** (it regenerates, so LPIPS can worsen while the look improves). Also tabulate composition / identity changes, wall time and peak footprint (`/usr/bin/time -l`).
5. **Decision**: only if A is clearly better than B on faces, skin, hair and text (a blind visual majority, and better on both face LPIPS and the no-reference metrics) continue: Sharp curve branches → exact query-chunked attention (memory) → Sol-Attn-style block-sparse attention (time; NVIDIA itself runs step 0 dense and steps 1–2 sparse, accepting 5–12% relative error) → 1080p. 2K stays out of scope even then. Otherwise record "not adopted", as the old §11 would, and return to speeding up ordinary generation (VDN-H3 and the like).

### §11.4 License display

Per component, **show the license that ships with the weight / source actually fetched** (never a blanket "Apache-2.0"): the refiner LoRA under the MiniMax H3 Community License (the existing notice for the base model), the latent upscaler MIT (LBH GitHub) or Apache-2.0 (NVIDIA bundle) depending on the source, and LightX2V / LMS as LoRAs the user added themselves, with the library's own display. Nothing is bundled; the same download-and-consent flow as SwiftVR.

### §11.5 Risks and open points

- Identity drift at σ₀ 0.909 (faces). Measure start σ 0.8 and the 2-step variant at the same time.
- Audio: the README pins encoded silence. The app substitutes its forced clean-audio path and passes the original audio through.
- The model card does not mention Ref2VA. The LoRA was trained against base (= the FL2VA DiT). Applying it to Ref2VA is unverified.
- Even 768p runs the DiT three times over 37k tokens. M1–M4 (BF16 path) are 2–3x slower; decide on the M5 first.
- The author node's `precision` default is fp16 in code, not the fp32 the README states (the code is authoritative). Evaluate the BF16 weights in F32 and take the reference from the author node in fp32.
- Relation to VDN-H3: VDN-H3 is a checkpoint trained with a hybrid architecture that adds a frame-wise linear-attention branch (+ two LoRAs, 8-step distillation, about 82 GB). Applying the refiner LoRA to it mechanically is structurally possible but the training distributions differ, so it means little. Treat it as a separate development branch.

## §12 Out of scope, separate tasks (other items of the LightX2V review memo)

- A diagnostic mode computing the LoRA correction separately (`y = D(W_q)x + sB(Ax)`): `h3_gpu_lora_delta_bf16` already exists, so a small change. Separate branch.
- Query-chunked attention for M4: not needed in this plan, since MFSWA is bounded by window length. A separate task for long sequences on the H3 generation side.
- Partial residency with a budget: for the H3 generation SSD path. Separate task.

## §13 Risks and open points

1. SwiftVR license (non-commercial). The app does not distribute the weights, only downloads them, but the legal judgement is the user's. The notice wording follows the existing MiniMax text.
2. Quality at 2x (training may assume 4x). Measured in P5.
3. The 20 GB download (handled by `ModelDownloader`'s resume) and conversion scratch space (20 GB original + 5–10 GB cache).
4. Differences between the two MFSWA owner-rule implementations (official / LightX2V). Dump both in P1 and check.
5. Agreement between MPSGraph's bilinear and `F.interpolate`. A custom kernel if they differ.
6. M4 performance (no hardware measured).
7. ANE: the weights cannot be redistributed, so a Core ML model must be generated on the device (criteria in §6.7).
8. **Producing the SeedVR2 reference output needs a ≥64 GB Mac or a cloud GPU** (impossible on a 24 GB machine). If unavailable, use the paper values as a provisional baseline and record the lower confidence of the pass judgement.
9. SwiftVR may be trained for 4x, so texture may be weak at 2x directly (compare 4x→1/2 in §9.5-6-4).
10. Videos whose size is not a multiple of 32 (e.g. 1920x1080 → padded to 1088 and cropped; no problem), portrait, extreme aspect ratios.

## §14 References

- The LightX2V review memo (LightX2V 543cff4 vs H3cApp v0.7.0 489ddc3).
- SwiftVR: https://github.com/H-oliday/SwiftVR , https://huggingface.co/H-oliday/SwiftVR , https://arxiv.org/abs/2606.09516
- LightX2V SwiftVR implementation: `lightx2v/models/networks/swiftvr/{reae,attention,model,streaming}.py`, `lightx2v/models/runners/swiftvr/swiftvr_runner.py`
- FlashVSR: https://github.com/OpenImagingLab/FlashVSR , https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
- TAEHV: https://github.com/madebyollin/taehv (MIT)
- diffusers `transformer_wan.py` (WanTransformer3DModel)
- h3c-app: `SPEEDUP_ROADMAP.md` (NAX/ccv/memory measurements), `TECHNICAL.md`, `VideoUpscale.swift`, `TinyPreviewDecoder.swift`, `h3_gpu.h`
- Open-H3-Regenerate-2K: https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K , https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K , https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/ (checked 2026-10-31)
- Latent upscaler: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (MIT), https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler
- ComfyUI ports: https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate (Apache-2.0), https://github.com/1038lab/ComfyUI-Open-H3-Regenerate-2K (GPL-3.0)
- VDN-H3: https://huggingface.co/OpenVDN/vdn-minimax-h3

## Appendix A: Implementation status (2026-10-28, branch vsr-engine)

| Phase | State | Where | Verification |
|---|---|---|---|
| P0 Document | done (main) | `UPSCALE_PLAN(.ja).md` | — |
| P1 Reference dump | done (2026-10-29; the user accepted the 20 GB download; Appendix A.1) | `tools/dump_swiftvr_fixture.py` -> `misc/fixtures/swiftvr-768x544/` (git-ignored) | `tests/test_real_vsr.c` (runs only with the weights and the fixture, skips otherwise) |
| P2 Weight cache | done | `h3_vsr_weights.c`, `h3_vsr_internal.h`, `h3_build_vsr_cache.c` | `tests/test_vsr_cache.c` (synthetic model, bf16 + int8) |
| P3 ReAE | done (F32 conv3d path) | `h3_vsr_reae.c` | `tests/test_vsr_reae.c`: 2e-7 vs a double transcription; chunk / frame-batch state carry bit-exact |
| P4 DiT | done (BF16 / int8, full-window MFSWA and full attention) | `h3_vsr_wan.c`, the `h3_vsr_*` kernels in `h3_shaders.metal`, `h3_gpu_sdpa_batched/cross_bf16` | `tests/test_vsr_wan.c`: 61 dB (bf16) / 60.8 dB (int8) vs a double reference |
| P5 Streamer | done (one context, sequential; only the decoder's 3-frame delay) | `h3_vsr.c` | `tests/test_vsr_session.c`: 7/29/52/70 frames, counts match, deterministic, cancellation |
| P5.5 Quality | partly (appendix A.3: a 5-clip set, fidelity, no-reference metrics, flicker, face A/B sheets; the SeedVR2 reference and the user's blind scoring are still missing) | `tools/quality_metrics.py`, `tools/quality_report.py`, `tests/vsr_run.c`, `misc/quality/set-v1/` | the tables of appendix A.3 |
| P6 App | done (running on real weights waits for P1) | `H3VSR.swift`, `VideoUpscale.swift` (`ChunkedFrameScaler`, `.swiftVR`, the generation guard `upscaleBlockedReason`), `UpscaleSheet.swift`, `SwiftVRModelSetup.swift` (license agreement, download, conversion; `SwiftVRSetup.shared`), `ModelManagerView.swift` (the 拡大モデル section), API `GET /api/upscalers`, Localizable | the four existing methods (standard / metalFX / Real-ESRGAN General / Anime, 2x, 39 frames 512²) are identical frame for frame, after decoding, to the pre-change app's output (measured 2026-10-28, `compare_frames`). Progress is per DiT block (`h3_vsr_wan_predict`'s `on_block`) and 中止 lands at a block boundary. The download goes through `ModelDownloader` (resumable), the conversion through `h3_vsr_build_cache`. The planned quality / speed mode switch in the UI (full attention, precision, overlap) is not implemented (the engine's `full_attention` reaches the Swift bridge only; `dit_overlap` is not wired). "Delete the originals after conversion" is not supported: the ReAE is read from `reae.safetensors` and the cache fingerprint wants the `transformer/` files present (to support it, the cache would carry the ReAE and a copy of the fingerprint) |
| P7 Optimization | not started (owner-only queries, NAX for any shape, tapconv, ReAE F16, two contexts, ANE) | — | — |
| P8 FlashVSR | not started | — | — |

Corrections the implementation forced on the text: 124 frames make 6 chunks (§6.4); even-layer owner widths are 16/16/16/12 (§6.4);
the patch input order is (c, ph, pw) and the output order (ph, pw, c) (`h3_vsr_patch_in_index` / `h3_vsr_patch_out_index` in `h3_vsr.h`);
the int8 linears need at least 128 rows, so token rows are padded to 128 (`h3_vsr_wan.c`).

### Appendix A.1: Verification on the real weights (2026-10-29, M5 24 GB)

The engine was compared stage by stage with the official implementation (github H-oliday/SwiftVR) run in float32 (`tools/dump_swiftvr_fixture.py`; the DiT is evaluated one block at a time so the 20 GB never has to be resident). The clip: 57 frames of the GUIN generation, cropped to 768x544 and downscaled to 384x272 (2x, a 24x17 token grid, three chunks FIRST 28 / MIDDLE 24 / LAST 5). `tests/test_real_vsr.c` (`H3_VSR_STAGES=host,reae,dit,session`, `H3_VSR_PRECISION=bf16|int8|both`).

| Stage | bf16 | int8 | Notes |
|---|---|---|---|
| Window index tables and owners (both parities, T=6,7) | identical | — | `h3_vsr_window.c` equals the official `_WindowRuntimeMetaCache` |
| RoPE tables | max 6.5e-7 | — | the official code uses float32 frequencies on MPS |
| Preprocessing (bilinear, align_corners=False) | identical (max 0) | — | |
| ReAE encoder (3 chunks, carried state) | 139.6 to 140.9 dB | — | float32 rounding level |
| ReAE decoder (chunk 0) | 140.8 dB | — | |
| Cache constants `mod` / `k2` / `v2` | 153 / 73.8 / 76.5 dB | same | k2/v2 are stored as BF16 |
| DiT alone, v (chunk 0, MFSWA / dense) | 60.0 / 60.4 dB | 51.3 / 51.7 dB | tokens after blocks 0 and 1: 87.8 / 85.3 dB |
| Session z_hq (chunks 0/1/2) | 63.4 / 62.9 / 61.9 dB | 54.7 / 51.6 / 52.3 dB | |
| 8-bit output vs the official run (57 frames) | 51.2 dB (max 18/255, 0.002% over 4) | 49.3 dB (max 31/255, 0.16% over 4) | against the source: engine 27.4 dB, official 27.6 dB (bilinear 39.3 dB; the official output is brighter and higher-contrast than the source, and visibly sharper) |
| Chunk time (28/24/5 frames) | 5.8 / 4.6 / 3.9 s | 4.5 / 3.7 / 2.6 s | GPU DiT 4.6 s bf16 over three chunks; the 7.9 s reported for int8 contradicts the DiT-alone 3.9/2.4 s and needs remeasuring |
| Peak live GPU bytes (`peak_live_bytes`) | 12.3 GiB | 8.3 GiB | the process's peak footprint is about 20 GB (MPSGraph workspaces included); P7 breaks it down |

Two defects only the real weights exposed (both fixed; the synthetic-model tests never hit them):

1. In the FFN's tanh-approximate GELU, Metal's `tanh` returns NaN for large arguments; 16 tokens went NaN and spread to everything. The argument is now clamped to +-20 (`h3_geglu_f32` and the ReAE clamp are protected the same way).
2. The ReAE decoder's MemBlock states were sized from the frame grid instead of the latent grid: about 17 GB allocated for nothing at 768x544, a 29 GB peak footprint, swapping, and the OS killing the process. `h3_vsr_reae_state_new` is fixed; the resident set is now 4.7 GiB on int8 when a session starts.

The app itself was checked too: with the rebuilt bundle running, `GET /api/upscalers` reports SwiftVR available (int8 cache) and `POST /api/upscale-file` enlarged a 512x512, 39-frame generated clip 2x to a 1024x1024 HEVC file (sound as it was) in about 30 s. Next to bilinear, hair, teeth and the dress pattern are clearly sharper, and the brightness matches the input.

Speed is below the plan's estimate (§6.4: 6.5 s per chunk at 1920x1088 on int8). At 768x544 (2,856 tokens) int8 takes 3.7 to 4.5 s per chunk, so 1920x1088 (12,240 tokens) extrapolates linearly to 16 to 19 s per chunk. This is where P7 starts.

### Appendix A.2: P7 speed work (2026-10-29, M5 24 GB, int8)

Measured with `tests/bench_vsr.c` (`make h3_vsr_bench`; `./h3_vsr_bench [in_w in_h out_w out_h] [chunks] [int8|bf16|auto] [full]`; `H3_VSR_TRACE_STAGES=1` prints each stage's time after waiting for the GPU, `H3_VSR_TRACE_DIT=1` the time of every op of block 1) at 960x544 -> 1920x1088 (a MIDDLE chunk is 12,240 tokens). Note that the progress callbacks fire while the CPU encodes, so a "DiT 2.2 s" read off the callbacks was wrong: waited for, the DiT takes 7.4 s.

| Version | MIDDLE chunk | Preprocess + ReAE encode | DiT (GPU complete) | ReAE decode | Peak live GPU |
|---|---|---|---|---|---|
| Start (ReAE on F32 MPSGraph, whole chunk at once) | 17.6 s | 1.7 s | — | 13.6 s (includes the DiT's GPU time) | 23.7 GiB (29 GB footprint, killed by the OS) |
| Every ReAE layer on the TensorOps "grid" | **9.6 s** | 0.47 s | 7.4 s | 1.65 s | 9.9 GiB |

- ReAE: the new kernel `h3_vsr_tapconv_bf16_nax` (`h3_shaders.metal`) keeps activations in a "padded grid" (channels-last BF16 with zero border rows, channels padded to a multiple of 64) and computes the nine 3x3 taps as row-offset `matmul2d` accumulations; the MemBlock's `cat[x, past]` is a second input, and bias, residual and ReLU are the epilogue. TPool is two taps at a frame offset, TGrow a 1x1 tap (TGrow(2) then splits frames), the stride-2 convs are computed everywhere and subsampled, and Upsample, pixel (un)shuffle and Clamp are small grid kernels. The decoder's 128-channel MemBlock (272x480, 2 frames) takes 22 ms (13.8 TFLOP/s) against 52 ms for the same three convs on F32 MPSGraph (205 ms inside the old runner). The ReAE now costs 2.1 s of a middle chunk (encode 0.47, decode 1.65) and no longer dominates.
- Precision: ReAE encoder 56 to 57 dB and decoder 59 dB against the official float32 run (the F32 path gives 140 dB). Checked by `tests/test_vsr_tapconv.c` (65 dB vs a double reference), `tests/test_vsr_reae.c` (55 to 60 dB vs the F32 path) and the reae stage of `tests/test_real_vsr.c`. `H3_VSR_REAE_F32=1` selects the old path. The pass width (4/8/12 frames) does not change the time, so the memory-minimal 4 is the default (`reae_frame_batch` 0 = 4).
- DiT per-op GPU time (block 1, 15 windows, 14,280 tokens): qkv 32 ms, out 12, norm2+q2 15, out2 15, fc1 53, **fc2 100** (linears 227 ms, about 65%), window SDPA 67, cross SDPA 9, gather/scatter 11 (attention 87 ms, about 27%), LN/GELU/residual 25 ms. The int8 linears run near 25 TFLOP/s except fc2 (K=14336) at 13. Static-K kernels for SwiftVR's shapes (`h3_linear_int8_local_scales_nax_r128_k3072_n*`, `..._k14336_n3072`, a full-K variant) gained 5 to 10% and left fc2 unchanged (cause not found; `H3_VSR_INT8_GENERIC=1` / `H3_VSR_FC2_LOOP=1` compare them).
- Owner-only queries (§6.5) are in: only each window's owned rows run as queries; windows with similar owned counts are merged into at most three buckets (a window's rows are padded to the bucket length by repeating its first owned token: the duplicates compute the same value and write it to the same token), one `h3_gpu_sdpa_batched_kv_bf16` per bucket, and `h3_vsr_scatter_rows_bf16` writes the rows back to their tokens. Block 1's attention (gather + SDPA + scatter) went from 84 to 64 ms, the DiT of a middle chunk from 7.3 to 6.9 s (9.4 to 9.0 s per chunk). Exact classes (7 buckets) cost the same with more dispatches; one bucket (every row) is 82 ms. `H3_VSR_QUERY_BUCKETS` sets the count and `H3_VSR_ALL_QUERIES=1` restores the full-window SDPA. Precision is unchanged (real weights, v 51.4 dB).
- Two contexts (§6.6) were implemented and measured: the decoder runs asynchronously on a second `h3_gpu` (its own command queue; `h3_vsr_reae_decode_begin` / `h3_vsr_reae_finish`) and is collected after the next chunk's DiT (output lags one chunk; `is_last` flushes). A 1080p middle chunk went from 9.19 s to 9.08 to 9.33 s, no gain, while the encode and output stages stretched from 0.5 to 1.2 s under contention: the DiT saturates the GPU, so there is no room to overlap the ReAE (as §6.6 warned, the floor is the GPU total). The code stays, off by default (`reae_separate_queue` 1 or `H3_VSR_OVERLAP=1` enables it; `tests/test_vsr_session.c` checks both paths give identical frames).
- fc2's low efficiency at K=14336 was studied with `tests/bench_int8.c` (`make h3_int8_bench`; `H3_BENCH_SKIP_QUANT=1` leaves out the quantization, `H3_BENCH_K_SWEEP=1` sweeps K). Row quantization costs 6 ms and is not it. At N=3072 the throughput falls with K: 25 to 26 TFLOP/s up to K=8192, 22.8 at 12288, 16.8 at 14336, 14.7 at 20480, and the stride matters too (19.3 at K=14464). What did not help: static-K kernels (5 to 10%), one full-K `matmul2d` (same), 128x256 tiles (fc2 got worse, 114 ms), a naive split-K (16 TFLOP/s without Morton-ordered threadgroups; 67 ms with them but still on the 14336 stride). What did: **the FFN as two K-contiguous halves** (fc1's output rows and fc2's input columns repacked at load; each half goes through `h3_linear_int8_split_k_nax_r128` into an F32 partial and `h3_linear_sum_partials_bf16` adds them). Block 1's FFN went from 166 to 139 ms and a middle chunk's DiT from 6.95 to 6.53 s (`H3_VSR_FFN_WHOLE=1` restores the single pass). Precision is unchanged (the partials are F32).
- GELU fusion: `h3_vsr_gelu_quantize_int8_rows` applies fc1's bias and GELU and quantizes the row to int8 in one pass (the values stay in registers between the max and the quantization), and fc2's partials take the quantized input as is. Block 1's FFN went from 139 to 129 ms and a middle chunk's DiT from 6.42 to 6.28 s (`H3_VSR_GELU_SEPARATE=1` restores the separate kernels). Precision is unchanged.
- Four FFN slices were measured too (`H3_VSR_FFN_SPLITS=4`): block 1's FFN 127 ms with two slices, 131 ms with four, and a middle chunk's DiT 6.20 s against 6.27 s; the per-slice overhead (quantization, partial writes) eats the gain from the shorter K. Two slices stay the default.
- Where P7 stands: a 1080p middle chunk 17.6 -> about 8.2 s (encode 0.46, DiT 6.0 to 6.3, decode 1.66). Beyond this the time rests on the int8 linear itself (about 25 TFLOP/s) and MPSGraph's SDPA.
- A 5-second clip (124 frames, about 5 chunks) should take 50 to 55 s, within the §1 target (2 minutes on M5). The LAST chunk runs a 7-latent DiT with the previous chunk's latents prepended, as the reference protocol does, so even one leftover frame costs a chunk.

### Appendix A.3: P5.5 quality evaluation (2026-10-29, M5, int8)

The part of the §9.5 quality programme this Mac can run. **No SeedVR2 reference was made** (it needs a 64 GB Mac or a cloud GPU), so the §1 criterion ("95% of SeedVR2-3B") remains unjudged. What is here: fidelity to the source (PSNR/SSIM/LPIPS), no-reference quality (MUSIQ, MANIQA via pyiqa, 8 frames), flicker (mean absolute difference between consecutive frames), the same on a face crop, and visual notes. The evaluation set is `misc/quality/set-v1/` (git-ignored): four generated clips and one real one, 33 frames each, the source downscaled by 2 (bicubic) as the input and the source as the ground truth. Variants come from `h3_vsr_run` (`make h3_vsr_run`; precision and attention selectable) and the app's API; metrics from `tools/quality_metrics.py`; the tables and the face A/B sheets from `tools/quality_report.py`. The face boxes are in `crops.txt`.

| Clip | Source | Input | Content |
|---|---|---|---|
| face-photo-tall | 1024x1792 | 512x896 | generated (photoreal girl, classroom) |
| face-photo-square | 512x512 | 256x256 (128x128 for the 4x comparison) | generated (photoreal woman, indoors) |
| face-real-webcam | 1600x1088 | 800x544 | real (webcam, a man) |
| leopard-man | 1920x1088 | 960x544 | generated (leopard-headed man, fur and skin) |
| anime-face | 672x384 | 336x192 | generated (anime girl) |

### anime-face

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| ground truth | — | — | — | 70.8 | 0.3943 | 4.698 | — | — | — | — |
| bilinear | 32.56 | 0.9479 | 0.1282 | 46.1 | 0.2609 | 3.550 | 29.10 | 0.1709 | 26.2 | 0.2116 |
| SwiftVR int8 dense | 25.68 | 0.8533 | 0.1020 | 73.4 | 0.4306 | 5.602 | 22.26 | 0.1128 | 50.2 | 0.3563 |
| SwiftVR int8 window | 25.65 | 0.8530 | 0.1028 | 73.4 | 0.4320 | 5.594 | 22.20 | 0.1134 | 50.2 | 0.3550 |
| Real-ESRGAN Anime | 31.45 | 0.9280 | 0.0937 | 70.9 | 0.3715 | 4.586 | 28.50 | 0.0638 | 46.8 | 0.3231 |
| Real-ESRGAN General | 29.15 | 0.9235 | 0.0886 | 74.5 | 0.4552 | 5.306 | 26.13 | 0.0764 | 54.3 | 0.3842 |
| standard | 34.04 | 0.9510 | 0.0802 | 53.4 | 0.2675 | 4.176 | 30.77 | 0.1219 | 32.3 | 0.2405 |

### face-photo-square

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| ground truth | — | — | — | 77.4 | 0.5439 | 1.403 | — | — | — | — |
| bilinear | 34.16 | 0.9496 | 0.1243 | 60.9 | 0.3134 | 1.030 | 32.72 | 0.1110 | 34.7 | 0.3130 |
| SwiftVR int8 dense | 29.29 | 0.8980 | 0.0794 | 78.4 | 0.5477 | 1.333 | 26.63 | 0.0771 | 57.7 | 0.5076 |
| SwiftVR int8 4x→1/2 | 29.47 | 0.8838 | 0.0775 | 77.9 | 0.5143 | 1.162 | 27.38 | 0.0578 | 52.2 | 0.4874 |
| SwiftVR int8 window | 29.29 | 0.8980 | 0.0794 | 78.4 | 0.5477 | 1.333 | 26.63 | 0.0771 | 57.7 | 0.5076 |
| Real-ESRGAN General | 32.53 | 0.9401 | 0.0654 | 78.0 | 0.5140 | 1.073 | 30.33 | 0.0698 | 54.3 | 0.4513 |
| standard | 34.85 | 0.9486 | 0.0998 | 69.0 | 0.3473 | 1.098 | 33.77 | 0.0734 | 39.5 | 0.3419 |

### face-photo-tall

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| ground truth | — | — | — | 45.9 | 0.3051 | 6.021 | — | — | — | — |
| bilinear | 44.09 | 0.9906 | 0.0289 | 35.7 | 0.3031 | 5.759 | 39.85 | 0.0482 | 38.9 | 0.3313 |
| SwiftVR int8 dense | 30.45 | 0.9227 | 0.1042 | 63.5 | 0.3614 | 5.773 | 27.05 | 0.1678 | 63.3 | 0.4354 |
| SwiftVR int8 window | 30.09 | 0.9206 | 0.1054 | 64.3 | 0.3611 | 5.908 | 26.89 | 0.1724 | 63.1 | 0.4288 |
| Real-ESRGAN General | 33.93 | 0.9635 | 0.0889 | 71.5 | 0.4092 | 6.312 | 30.66 | 0.1373 | 66.8 | 0.4843 |
| standard | 38.30 | 0.9859 | 0.0217 | 40.8 | 0.3118 | 5.926 | 38.00 | 0.0215 | 42.1 | 0.3330 |

### face-real-webcam

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| ground truth | — | — | — | 37.0 | 0.2083 | 0.991 | — | — | — | — |
| bilinear | 44.76 | 0.9917 | 0.0308 | 30.1 | 0.2107 | 0.917 | 45.15 | 0.0328 | 36.6 | 0.2041 |
| SwiftVR int8 dense | 30.19 | 0.9280 | 0.1472 | 67.0 | 0.3572 | 1.407 | 30.51 | 0.1363 | 59.9 | 0.3172 |
| SwiftVR int8 window | 29.59 | 0.9233 | 0.1580 | 64.8 | 0.3667 | 1.423 | 30.30 | 0.1296 | 59.3 | 0.3094 |
| Real-ESRGAN General | 33.30 | 0.9575 | 0.1075 | 54.3 | 0.3536 | 1.065 | 35.87 | 0.0765 | 51.8 | 0.2762 |
| standard | 40.12 | 0.9835 | 0.0237 | 32.9 | 0.2090 | 1.031 | 41.52 | 0.0292 | 38.0 | 0.1958 |

### leopard-man

| variant | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|---|---|
| ground truth | — | — | — | 46.2 | 0.1896 | 4.319 | — | — | — | — |
| bilinear | 43.53 | 0.9880 | 0.0455 | 38.0 | 0.1950 | 3.896 | 38.28 | 0.0459 | 48.2 | 0.3571 |
| SwiftVR int8 dense | 28.96 | 0.8471 | 0.2781 | 71.6 | 0.3053 | 5.324 | 25.06 | 0.2727 | 75.7 | 0.5133 |
| SwiftVR int8 window | 28.81 | 0.8498 | 0.2647 | 70.9 | 0.2956 | 5.454 | 25.11 | 0.2632 | 75.6 | 0.5073 |
| Real-ESRGAN General | 35.51 | 0.9534 | 0.1099 | 58.0 | 0.2418 | 4.359 | 31.34 | 0.1008 | 67.3 | 0.4101 |
| standard | 41.21 | 0.9773 | 0.0285 | 42.1 | 0.1938 | 4.176 | 39.62 | 0.0187 | 51.3 | 0.3586 |

Observations (face crops of frame 16 side by side: `misc/quality/set-v1/*/ab-crops.png`):

- **SwiftVR draws detail in, so its fidelity to the source (PSNR/SSIM/LPIPS) is the lowest on 4 of 5 clips**, while its no-reference scores (MUSIQ, MANIQA) are the best or equal on photoreal and real faces. The generated sources are soft themselves (MUSIQ 37 to 46), so the output is "sharper than the original".
- **Skin**: on the real clip (face-real-webcam) SwiftVR draws skin with pores and shading, Real-ESRGAN is smooth and waxy, standard is blurred. On the generated girl (face-photo-tall) SwiftVR's skin is somewhat flat and brighter, Real-ESRGAN painterly. On the woman (face-photo-square) SwiftVR deepens the skin's redness a little. 4x then 1/2 on the same clip is gentler and closer to the source (face LPIPS 0.077 -> 0.058, flicker 1.33 -> 1.16) at slightly lower no-reference scores.
- **A systematic brightening**: SwiftVR's faces are +2.9 to +3.7 (of 255) brighter than the source, uniformly, with contrast (standard deviation) x1.01 to x1.07; the other methods stay within ±0.3 (Real-ESRGAN -1.5 to -2.5). Being content-independent, this should disappear with a mean-luma match to the input (the idea behind FlashVSR's colour correction). Not implemented.
- **Flicker**: SwiftVR is 1.2 to 1.4x the source on the real clip and the leopard (the added texture moves between frames); 4x then 1/2 is below the source.
- **Anime**: SwiftVR adds noise-like texture to flat areas; the face LPIPS of Real-ESRGAN Anime (0.064) is far better. Anime-style video should use Real-ESRGAN Anime.
- **Dense vs windowed (MFSWA) attention**: marginal (+0.1 to 0.6 dB PSNR, equal face metrics); the faster windowed mode is enough. At a 256x256 input (one window) they are identical.
- bf16 vs int8 was not put through the same comparison for lack of disk; by the parity with the official implementation in appendix A.1 (51 dB bf16 / 48 dB int8 on 8-bit frames) they differ by about 48 dB, below what the eye sees.

Left: the SeedVR2 comparison, the user's blind A/B (skin, hair, fur, text, dark areas), more real footage in the set, DOVER, overlap=1, and the preprocessing (bicubic / Lanczos) comparison.

**Blind scoring (2026-10-29, the user; face crops of frame 16, shuffled, the source among the candidates)**

| Clip | Best | Runner-up | Relation to the metrics |
|---|---|---|---|
| face-photo-tall (generated girl) | **standard** (the resize) | — | the method with the best fidelity (PSNR 38.3, face LPIPS 0.022); on a soft generated source, the drawn-in detail (SwiftVR face LPIPS 0.17, Real-ESRGAN 0.14) was not preferred |
| face-photo-square (generated woman) | **SwiftVR 4x then 1/2** | — | best face LPIPS (0.058), least flicker (1.16); gentler than direct 2x (0.077) and chosen over it |
| face-real-webcam (real) | **SwiftVR windowed** | SwiftVR dense | best no-reference scores (face MUSIQ 59); the lowest fidelity, yet the drawn-in skin was preferred on real footage |
| leopard-man (generated fur and skin) | **Real-ESRGAN General** | — | between fidelity and no-reference (PSNR 35.5, face MUSIQ 67); SwiftVR (face MUSIQ 76) presumably lost on the brightening and the flicker |
| anime-face (anime) | **Real-ESRGAN Anime** | Real-ESRGAN General | best face LPIPS (0.064); matches the metrics' order |

Reading: on a real face SwiftVR's drawn-in skin won; on soft generated faces the gentler methods won (standard, or SwiftVR's 4x then 1/2); on fur and anime Real-ESRGAN won. Direct 2x SwiftVR was best on no clip. Implications: (1) a "generate at 4x, downscale by 1/2" path is worth making SwiftVR's default candidate (better LPIPS, flicker and score than direct 2x), (2) the systematic +3/255 brightening should be corrected, (3) the default method should follow the content (anime: Real-ESRGAN Anime; real footage: SwiftVR; soft generated faces: standard or 4x then 1/2). The SeedVR2 comparison is still missing.

**Re-measured after the colour match and the 4x-then-half path (2026-10-29, the same set, int8 windowed; `int8-window` = no correction, `int8-match` = colour match (the default), `int8-4x-half` = restored at 4x from the same input and halved with Lanczos (colour match on), `standard` = the Lanczos resize, the app's standard method without compression)**

| Clip | Variants | PSNR | LPIPS↓ | MUSIQ | flicker | face PSNR | face LPIPS↓ | face MUSIQ |
|---|---|---|---|---|---|---|---|---|
| face-photo-tall | no match / match | 30.09 / 31.43 | 0.105 / 0.105 | 64.3 / 64.7 | 5.91 / 5.90 | 26.89 / 27.40 | 0.172 / 0.172 | 63.1 / 63.5 |
| face-photo-square | no match / match / 4x-half | 29.29 / 30.08 / **33.85** | 0.079 / 0.077 / **0.042** | 78.4 / 78.4 / 78.3 | 1.33 / 1.33 / **1.03** | 26.63 / 26.97 / **30.91** | 0.077 / 0.075 / **0.039** | 57.7 / 57.7 / **64.1** |
| face-real-webcam | no match / match | 29.59 / 30.51 | 0.158 / 0.156 | 64.8 / 65.6 | 1.42 / 1.43 | 30.30 / 31.41 | 0.130 / 0.128 | 59.3 / 59.2 |
| leopard-man | no match / match | 28.81 / 29.43 | 0.265 / 0.262 | 70.9 / 70.7 | 5.45 / 5.39 | 25.11 / 25.27 | 0.263 / 0.260 | 75.6 / 75.6 |
| anime-face | no match / match / 4x-half | 25.65 / 25.90 / **29.59** | 0.103 / 0.102 / **0.079** | 73.4 / 73.1 / 74.8 | 5.59 / 5.59 / **5.19** | 22.20 / 22.32 / **26.65** | 0.113 / 0.114 / **0.063** | 50.2 / 50.7 / **57.2** |

- The colour match (each channel's mean shifted to the input's; the offsets were -0.8 to -4.7/255, blue the largest) raises PSNR by 0.3 to 1.3 dB and leaves the no-reference scores and the flicker unchanged. On by default (`h3_vsr_open_params.color_match`; `H3_VSR_NO_COLOR_MATCH=1` disables it).
- 4x then half beats direct 2x on every metric on the two clips it could run (256x256 and 336x192 inputs), and on anime it brings the face LPIPS level with Real-ESRGAN Anime (0.064). The 4x intermediate costs GPU memory, though: for the three clips enlarged to 1024x1792, 1600x1088 and 1920x1088, `h3_vsr_check_requirements` refused it (45 to 53 GB estimated; the estimate was revised to 1.5 KB per pixel for the ReAE's 4-frame passes, and 3840x2176 still needs about 33 GB). On a 24 GB Mac the path fits a 4x intermediate of about 2.5 million pixels (a 448x256 input -> 1792x1024, say). Forced with `H3_VSR_IGNORE_BUDGET=1`, 672x384 -> 2688x1536 ran with a 20.3 GB footprint, swapped, and took 26 s per middle chunk (its 18.2 GB estimate exceeds 90% of the 19.1 GB budget, so it is refused by default). In the app the "4倍で生成してから縮小する" toggle (shown for 2x only, off by default) selects it, and it falls back to direct 2x silently when it does not fit. About four times the time (anime 2.1 -> 8.8 s, square 2.2 -> 9.2 s).

**Wet-skin face close-ups (2026-10-29, clips generated to match the user's sample image, M5, int8 windowed)**

To match the comparison image the user showed (wet slicked-back blonde hair, glistening skin with water droplets, warm light, a smile and a three-quarter turn), 5-second clips were generated in the app with the same prompt and seed (20261029) at 544x960 (portrait-large, 964 s) and 256x448 (portrait-small, 140 s). The set: `~/Movies/H3cApp/skin-eval-2026-10-29/` (clips, crop sheets, scripts). Measured two ways.

(a) The real workflow: each generated clip enlarged 2x through the app's API by the five methods, no-reference scores on the face crop of frame 46 (544x960: cheek, nose and mouth, 600x700; 256x448: eyes to mouth, 400x460); there is no ground truth. At 544x960 the 4x path's 2176x3840 intermediate exceeds the memory budget and fell back silently to direct 2x (frame-identical output, 46 s); at 256x448 it ran (11 -> 39 s).

| Input | Method | Time | face MUSIQ↑ | face MANIQA↑ | face flicker↓ |
|---|---|---|---|---|---|
| 544x960 | standard / MetalFX | 1 s / 1 s | 56.3 / 57.9 | 0.247 / 0.228 | 5.33 / 5.66 |
| 544x960 | Real-ESRGAN General | 13 s | 62.5 | 0.268 | 5.19 |
| 544x960 | SwiftVR 2x | 47 s | **70.1** | **0.331** | 6.23 |
| 256x448 | standard / MetalFX | 1 s / 1 s | 56.7 / 59.2 | 0.391 / 0.385 | 4.87 / 5.25 |
| 256x448 | Real-ESRGAN General | 3 s | 65.8 | 0.419 | 4.85 |
| 256x448 | SwiftVR 2x / 4x then 1/2 | 11 s / 39 s | **72.4** / 72.2 | 0.526 / **0.608** | 5.81 / 5.54 |

(b) With ground truth: frames 30 to 62 of the generated clip are the truth, the input is its bicubic half, each method restores it (`h3_vsr_run`; Real-ESRGAN through the app from an x264 crf 6 encode of the input). Here 4x then 1/2 fits at 544x960 too (a 1088x1920 intermediate).

| Truth | Variant | PSNR | LPIPS↓ | flicker | face PSNR | face LPIPS↓ | face MUSIQ | face MANIQA |
|---|---|---|---|---|---|---|---|---|
| 544x960 | Lanczos / Real-ESRGAN | 43.09 / 37.01 | **0.052** / 0.080 | 3.31 / 3.46 | 40.86 / 35.21 | **0.070** / 0.090 | 52.9 / 65.5 | 0.408 / 0.445 |
| 544x960 | SwiftVR windowed / dense | 31.55 / 31.70 | 0.119 / 0.115 | 4.13 / 4.07 | 30.25 / 30.39 | 0.143 / 0.141 | 71.0 / 70.4 | 0.513 / 0.518 |
| 544x960 | SwiftVR 4x then 1/2 | 33.98 | 0.107 | 3.98 | 32.51 | 0.110 | **71.7** | **0.604** |
| 256x448 | Lanczos / Real-ESRGAN | 39.25 / 34.13 | 0.084 / 0.094 | 3.21 / 3.49 | 37.60 / 32.97 | 0.100 / 0.100 | 49.5 / 59.8 | 0.446 / 0.469 |
| 256x448 | SwiftVR 2x (windowed = dense) | 30.01 | 0.102 | 4.21 | 28.79 | 0.095 | 69.8 | 0.564 |
| 256x448 | SwiftVR 4x then 1/2 | 33.28 | **0.057** | 3.81 | 31.92 | **0.060** | **72.3** | **0.602** |

Checked on the 1:1 crops (whether the result reaches the "pores visible" look of the SeedVR2 column in the user's sample):
- **Skin texture**: SwiftVR draws the pores and droplets most clearly and leads every no-reference score (face MUSIQ 70 to 72; Real-ESRGAN 60 to 66; standard 50 to 57). Real-ESRGAN tightens the existing shapes a little and adds no pores. Standard and MetalFX stay soft.
- **Redrawing**: at 544x960, direct 2x SwiftVR redraws the eye region: the lashes change shape, tear-trough lines that the source lacks appear, and the skin under the eye gets concentric ripples (seen in the real workflow (a) and in (b) alike, with dense attention too). 4x then 1/2 keeps the eye shape close to the truth and loses the ripples, with faint tear-trough lines remaining. Real-ESRGAN and Lanczos change no shapes.
- **What 4x then 1/2 buys**: at 544x960 face LPIPS 0.143 -> 0.110, PSNR +2.3 dB, flicker -4%; at 256x448 face LPIPS 0.095 -> 0.060; the no-reference scores do not drop. For skin close-ups like the sample, 4x then 1/2 beats direct 2x. But at 544x960 the app's 4x path does not fit in memory (an 8.4 Mpx intermediate); (b) fit only because its input was the half.
- **Fidelity**: against ground truth Lanczos is best and SwiftVR is 8 to 12 dB lower in PSNR. The choice follows the purpose: standard to enlarge the generated frame as it is, SwiftVR to show pores and droplets.

Conclusion: the texture closest to the sample's SeedVR2 column comes from SwiftVR, preferably 4x then 1/2. So that this path works on generated video at 544x960 and larger, the tiling below was implemented. Direct 2x tends to redraw the eyes, so on face close-ups the user should check and choose.

**Spatial tiles (2026-10-29, M5 24 GB, int8).** When a chunk's estimate passes 90% of the GPU's recommended working set (19.07 GB on this Mac), the engine splits each frame into equal tiles on the 32 px grid (overlapping by at least 128 px), restores every tile with its own ReAE state and DiT pass, and blends them with linear ramps (normalised per axis so the weights sum to 1; `h3_vsr_tile_plan_build` / `h3_vsr_tile_weights`, the kernel `h3_vsr_tile_accumulate_f32`; `h3_vsr_check_requirements` reports the grid). The estimate gained each tile's ReAE state (about 210 B per pixel) and the F32 composite frame. Along the way a leak was fixed: `decoded` was allocated twice per chunk (about 0.7 GB per chunk at 1920x1088; the resident set grew 10.6 -> 12.1 GiB over a run).

| Case | Grid | Estimate | Peak live | First chunk | Then | Notes |
|---|---|---|---|---|---|---|
| 544x960 -> 2176x3840 (4x) | automatic = 2 x 2 of 1152x1984 | 16.4 GB | 16.0 GiB | 44 s | about 40 s | 1 x 3 (2176x1376) estimates 18.6 GB and is skipped; forced, it peaks at 17.4 GiB at the same speed |
| 272x480 -> 1088x1920 | one | 11.5 GB | 11.7 GiB | 9.7 s | 8.2 s | the same peak on all three chunks after the leak fix |

Quality (gt-large, 272x480 -> 1088x1920 -> 1/2 against the ground truth): one tile PSNR 33.98 / face LPIPS 0.110 / face MUSIQ 71.7; forced 2 x 2 33.88 / 0.108 / 71.2. Tiled and one-tile outputs are 40.9 dB (2 x 2) to 44.7 dB (2 x 1) apart, the difference spread evenly at 1-2/255 per band rather than at the overlaps. The 544x960 4x frame (the automatic grid above) was checked at full size: no line or step along the overlap bands (cheek, the corner at the mouth). The app's path (4x in tiles then Lanczos 1/2) against a direct 2x, without ground truth, on the same face crop: face MUSIQ 55.4 -> 57.3, MANIQA 0.389 -> 0.448, face flicker 6.09 -> 5.69; the brow redrawing of the direct 2x (extra dark hairs) stays closer to the source through 4x.

Through the app's API (the method "SwiftVR 4倍で生成してから縮小" = `swiftVRFourX`, which replaced the earlier toggle; `HalvingChunkScaler`: 4x in tiles, then the standard resize) this 544x960, 121-frame clip took 243 s (a direct 2x 42 s: about 5.5 times). A first run sharing the machine with a heavy analysis script took 15 minutes, so the GPU and RAM should be left to it. Synthetic tests: `tests/test_vsr_session.c` (a forced 2 x 2 run returns every frame deterministically, 26 dB from the one-tile run, and a grid that does not fit is refused) and `tests/test_vsr.c` (the layout and the weight sums).

### Appendix A.4: The Open-H3 regeneration gating experiment (2026-10-31, M5 24 GB, branch regen-api)

The gating experiment of §11.3 was run. Implementation (branch `regen-api` off main): (1) `h3_params.initial_latent_path` and `video_sigmas[]` (a starting latent and an arbitrary σ list; the latent file's audio is held clean and carried through), (2) `h3_latent_upscale.c` (the LBH 3D-conv latent upscaler, agreeing with the author node in fp32 to 122–126 dB, rel-L2 2e-6; 960×544 × 37 latent frames → 1344×768 in 19 s, peak 5.8 GB), (3) AdaLN curve-branch support in the LoRA loader (`adaln_proj.linear` `lora_A/B` and NVIDIA's `curve_A/B` added as `B·A` to a curve-form checkpoint's [96768, 8] projection; `H3_LORA_NO_ADALN=1` disables it), (4) the CLI's `H3SPIKE_INITIAL_LATENT` / `H3SPIKE_SIGMAS` / `H3SPIKE_LORAS`, `h3_latent_upscale`, and `tools/dump_latent_upscaler_fixture.py`.

**Verified facts**

- Latent normalization: the author node's 24 `LATENTS_MEAN/STD` values are **identical** to the VAE config's `latents_mean/std` (difference 0). ComfyUI's H3 latent (`samples`) is the DiT-space latent normalized by the VAE wrapper; the node applies the same constants once more before the network and de-normalizes the output. NVIDIA's pipeline hands the node the normalized latent too, so it does the same. This port does the same with the `.h3lt` latent.
- **The Sharp AdaLN curve branches are empty.** In both the LMS Sharp v1.0 LoRA file and NVIDIA's `sharp_curve_branches.safetensors`, `B` is exactly zero in all 50 blocks (`A` is non-zero, max 0.354). The output is therefore the same with or without them (confirmed bit-identical here), and they are an identity in the published pipeline as well. The §11.3-5 condition ("never reject on the variant without the branches") is satisfied by the branches having no effect.
- Kijai's refiner LoRA (rank 384, BF16, ComfyUI layout, QKV fused, FC1 halves swapped), LightX2V Turbo v1.2 and LMS Sharp r64 land on the 312 linears through the existing loader.
- H3 at 1344×768 × 124 frames (37 latent frames, about 37k tokens) in the int8-cache mode: **282 s per DiT step**, 181 s VAE decode. A 3-step regeneration takes 1027–1087 s, peak footprint 14.3–15.4 GB with the three LoRAs, 8.9 GB without.

**Conditions**: the app's active model (MiniMax-H3 GGUF, Q4_K pruned, curve-form AdaLN), int8-cache mode, seed 1. Two drafts: "osaka" (960×544 × 124 frames, abstract digital motion graphics, the latent of an existing generation) and "face" (1024×576 × 124 frames, the skin-eval wet-hair close-up of a woman, Turbo 4 steps, 503 s). Target 1344×768. A = latent upscale → 3 steps at σ 0.909375/0.857143/0.666667/0, LoRAs = refiner r384 + Turbo v1.2 + LMS Sharp r64. B = SwiftVR 2x (int8, windowed attention) → Lanczos to 1344×768 (69 s). C = the draft resized to 1344×768 with Lanczos. Metrics from `tools/quality_metrics.py` (fidelity is "closeness to the resized draft" with C as reference, MUSIQ/MANIQA are no-reference, flicker is the mean absolute difference between neighbouring frames, the face is the crop 380,220,560,320).

| osaka | PSNR vs C | LPIPS | flicker | MUSIQ | MANIQA | time |
|---|---:|---:|---:|---:|---:|---:|
| C plain resize | — | — | 7.04 | 44.6 | 0.282 | — |
| B SwiftVR | 29.9 dB | 0.22 | 5.58 | 51.3 | 0.359 | 163 s (run alongside A2) |
| A0 no LoRA, 3 steps | 22.6 dB | 0.42 | 8.9 | 44.2 | 0.248 | 1036 s |
| A1 refiner+Turbo+Sharp | 20.0 dB | 0.51 | 14.1 | 51.2 | 0.281 | 1032 s |

| face | PSNR vs C | LPIPS | flicker | MUSIQ | MANIQA | face LPIPS | face MUSIQ | face MANIQA | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C plain resize | — | — | 6.26 | 68.5 | 0.262 | — | 65.3 | 0.394 | — |
| B SwiftVR | 30.8 dB | 0.126 | 6.52 | 76.6 | 0.361 | 0.162 | 70.8 | 0.489 | 69 s |
| A1 refiner+Turbo+Sharp (3 steps) | 21.5 dB | 0.275 | 7.05 | 74.0 | 0.328 | 0.291 | 69.1 | 0.501 | 1027 s |
| A1 without the branches | bit-identical to A1 | | | | | | | | 1087 s |
| A2 same LoRAs, 2 steps from σ 0.7 | 24.9 dB | 0.167 | 6.83 | 71.7 | 0.299 | 0.201 | 65.8 | 0.462 | 894 s (run alongside B) |

**By eye**: A redraws detail while keeping composition and motion (wiring diagrams and strings of text in osaka; lashes, pores and skin grain in face). Detail that was there gets replaced in the process (the droplet under the left eye disappears). B sharpens while keeping the original detail, with ample skin grain. In osaka A's flicker doubles: the σ 0.909 redraw shows as shimmer.

**Decision (criterion of §11.3-5)**: at 1344×768, A is not clearly better than B (SwiftVR) on faces and skin. B leads the no-reference metrics (MUSIQ 76.6 vs 74.0, MANIQA 0.361 vs 0.328, face MUSIQ 70.8 vs 69.1), A leads only face MANIQA slightly (0.501 vs 0.489), B is clearly better on fidelity and flicker, and the time differs 15x. A2 (2 steps from σ 0.7) stays closer to the draft (LPIPS 0.167) but its no-reference metrics drop further (MUSIQ 71.7, face 65.8), still below B. On osaka B equals A1 on MUSIQ (51.3 vs 51.2) and leads on MANIQA, fidelity and flicker. The Sharp branches are an identity, so condition 2 holds. **So the 1080p / 2K work (query-chunked and sparse attention) is not pursued.** Regeneration may still have value as a separate "redraw the detail of the same composition" feature (finishing a draft at 768p) rather than as an upscaler; whether to productize that is a separate decision.

**Not verified / open**: start σ 0.8, a 4x-equivalent comparison, other subjects (fur, text), the difference to NVIDIA's merged FP32 LoRA (rank 576), the user's blind viewing, M1–M4 times. A synthetic-weight `make test` case for the latent upscaler is not added (only the real-weight reference comparison).
