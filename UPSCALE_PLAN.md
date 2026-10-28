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
int h3_vsr_requirements(const h3_vsr *vsr, int in_w, int in_h, int out_w, int out_h, h3_vsr_requirements *out);

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
int  h3_vsr_get_stats(const h3_vsr *vsr, h3_gpu_stats *dit, h3_gpu_stats *reae);
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
| `scale_shift_table[6,3072] + time_proj` → `(1+scale_msa)`, `shift_msa`, `gate_msa`, `(1+c_scale)`, `c_shift`, `c_gate` | F32 (BF16 copies for the LN) | t=1000 fixed |
| `attn2.to_k/to_v` not stored; instead **`K2_b = RMSNorm_k(ctx·Wk+bk)`, `V2_b = ctx·Wv+bv` `[512,3072]`** | BF16 | fixed prompt. 6.3 MB/layer, 189 MB total |
| `patch_embedding [3072,48,1,2,2]` → `[3072,192]` + bias | BF16 + F32 | |
| `norm_out` modulation (`scale_shift_table[2,3072]+temb`) | F32 | |
| `proj_out [192,3072]` + bias | BF16 + F32 (bias folded into unpatchify) | |
| Condition embedder (time 256→3072→3072, time_proj 3072→18432, text 4096→3072→3072) | **not stored**; evaluated once on the CPU in double/F32 at build time | its outputs are the constants above |
| ReAE 40.95M | BF16 | shared by the NAX tapconv and MPSGraph paths |

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
| P4 DiT | `h3_vsr_wan.c` + `h3_vsr_window.c`: full-window MFSWA → owner-only, BF16 → int8, constant conditioning, cross-attention, RoPE (verify t_off=0) | block 0 ≥ 40 dB, final latent ≥ 36/33 dB, RGB ≥ 38/35 dB (BF16/int8); record the difference from dense attention |
| P5 Streamer | `h3_vsr_chunk.c`/`h3_vsr.c`: chunking, state, last-chunk front fill, preprocessing and cropping, two contexts in parallel, cancel | 124-frame end-to-end ≥ 35 dB vs the reference's whole output; time and peak-memory table on M5/M4 (`bench_vsr`) |
| P5.5 Quality evaluation | §9.5: evaluation set, SeedVR2 reference output (a ≥64 GB Mac or cloud), metric scripts, fill the table for every variant of this implementation, blind A/B | record the achieved ratio vs SeedVR2 (§1's 95%) and choose the defaults of the "quality" and "speed" modes |
| P6 App | `ChunkedFrameScaler` and the 3-stage pipeline, **quality / speed mode switch (full attention, precision, overlap)**, `.swiftVR`, download/acceptance/conversion UI, API, records, documents, localization, `isGenerating` guard | the existing four methods are byte-identical and unchanged; the new method works from both the project and file entry points and with continuation |
| P7 Optimization | int8 default (M5), fused kernels (§6.5-9), revisit SSD vs resident, ANE experiment (go/no-go by §6.7) | §1 time targets met, or the reasons recorded |
| P8 FlashVSR Tiny | Second back-end on the shared engine (LQ_proj_in, KV cache, block-selection attention, TCDecoder, color correction) | same parity and speed tables |
| PX Experiment | H3 hires-fix (§11) | comparison table and decision |

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

## §11 Experiment phase: re-noise upscaling with H3 itself (hires-fix)

- Idea: finished latent/video → 2x enlargement (Lanczos or Real-ESRGAN) → `h3_video_vae_encode_fast` to a latent → `x = σ₀·ε + (1−σ₀)·z` (the `h3_forced_trajectory_start` formula, σ₀∈{0.3,0.4,0.5}) → Turbo LoRA 2–4 steps on a schedule truncated below σ₀ → decode.
  Audio stays **clean and pinned** through the existing forced-audio (lip-sync) path.
- Constraints [verified]: `h3_generate` has no API for a starting latent / starting σ (needs `h3_params` additions: `init_video_latent_path`, `start_sigma`). The DiT canvas cap of 768×1344 → output at most 1344×768 (1920×1088 impossible).
- Cost [estimate]: 1344x768 / 124 frames is about 37k video tokens. From the measured 505 s for 4 steps at 768²/124 fr: linears ×1.7, attention ×2.9 → 15–20 min for 4 steps, 8–10 min for 2.
- Decision: PSNR and visual comparison against SwiftVR on the same input. If it is 5x slower or more with no quality advantage, record it as not adopted.

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
