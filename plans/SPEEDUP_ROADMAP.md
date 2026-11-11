# h3c-app / h3.c speedup roadmap

Backlog of T2V/Ref2V speedup work, agreed 2026-09-27, with the analysis and
measurements behind each item. This is a working engineering log, not
polished documentation — it records what was tried, what worked, what
didn't, and why, so the reasoning doesn't have to be redone.

Agreed with the user 2026-09-27 after reviewing techno-edge's MiniMax-H3
speedup article (turbo steps, int8 VAE, sparse attention Sol-Attn/SLA/VSA)
and Sol-Attn. Item #3 (int8 video VAE) was implemented first (2026-09-27):
the 36 block linears of the F32 transformer VAE decoder run on the M5 int8
kernels, `H3_VAE_INT8=0` opts out. Measured 512x512/124 frames/4 steps: VAE
decode 108.0s → 37.0s, VAE GPU peak 9.45 → 2.83 GiB, end-to-end 250–270s →
186s; PSNR vs F32 46.4 dB (worst frame 44.9), visually identical.

Items #1 and #2 were done next (2026-09-27, for v0.2.0): SpeedMode
quality/fast/fastest = exact / 45 layers + core reuse min(4, steps/5) / +
token reduction. Measured 512x512/39fr/20 steps: 232.6s / 82.3s / 73.7s.
Stacking 40 layers + core 6 + token reduction + int8 row FC2 (61.0s) smeared
the subject; int8 row FC2 alone gave no speedup, so it isn't used. PSNR vs
exact is ~12 dB for every approximation (trajectory changes composition) —
judge visually. LoRA entries got `recommendedSteps` (auto-detected from
"4step" names) driving steps.

Remaining backlog, in the recommended order:

## 1. Expose already-implemented engine speed knobs in the app/API

A speed preset: `dit_layers` 45/40, `core_reuse` 4/6, `token_reduction`,
`use_int8_row_fc2` (h3.h:77-92 calls these "validated"). `core_reuse>1` and
`denoise_reuse>1` are mutually exclusive (h3.c check). H3Engine.swift didn't
pass `token_reduction`/`use_int8_row_fc2` at all before this. Small effort.

**Status: done** — see item summary above (SpeedMode quality/fast/fastest).

## 2. Turbo LoRA step presets

4/8 steps — the user has FL2V turbo 4step v1.1/v1.2, 8step v1.0 and
`minimax_h3_ref2v_turbo_4step_v0.1` in `~/Documents/work/h3c/lora`. Add a
"recommended steps" field per LoRA entry. H3 is CFG-free already.

**Status: done** — see item summary above.

## 3. int8 video VAE

**Status: done** — see the top of this document.

## 4. Unfused LoRA

**Status: done** 2026-09-28, pushed as `fe0ad33`. GPU weight patching with
stochastic rounding, all compute modes, stacking; no fused caches anymore.
See the multi-LoRA-stacking work (verified QKV/FC1 layout facts, the
stochastic-rounding requirement, costs, design) recorded separately.

## 5. Sol-Attn-style block-sparse attention

arXiv 2607.24027. The DiT joint attention is MPSGraph's built-in SDPA
(`h3_gpu.m` `h3_gpu_sdpa`) — so this first needs a custom Metal
flash-attention kernel (reference: philipturner/metal-flash-attention, MIT,
same lineage as ccv/Draw Things). Algorithm: 64-token Q/KV blocks, proxy
score = mean(Q_blk)·mean(K_blk), per-query-block threshold
tau = mu + beta·sigma, skipped blocks approximated by a zeroth-order term,
dense for the first 20% of steps and the first layer (~85% sparsity, 2-3x
end-to-end on NVIDIA). Large effort (custom Metal flash-attention + sparse
selection kernel from scratch, real correctness risk).

**Status: in progress, extensively prototyped, not yet integrated.** This
item has had by far the most investigation. Summary of each round below.

### Micro-benchmark (2026-09-28)

Gating check the roadmap itself calls for, before writing the kernel: a
standalone tool linking `libh3.a`'s public `h3_gpu.h`, timing
`h3_gpu_sdpa_bf16` against the four block linears (qkv/out/fc1/fc2) in
isolation at real sequence lengths read off actual H3Spike runs
(`H3_DEBUG_ROWS` temporarily added to `h3_dit.c`, then reverted). Critically,
comparing SDPA against int8-quantized linears (`h3_gpu_linear_int8_bf16`,
matching the real M5 default fast path — `dit->int8_qkv`/`int8_mlp`/
`int8_attention_out`) rather than plain BF16 linears, since quantization
speeds up the linears but not SDPA, raising attention's *share* of the
faster path.

Measured attention_share of one DiT block (M5, int8 linears):

| Resolution / frames | sequence length | attention share |
|---|---|---|
| 512² / 39fr (app's short/default clip) | 3204 | 17.8% |
| 384² / 124fr | 5744 | 28.9% |
| 768² / 124fr | 21728 | 65.7% |

Confirms the FLOPs-estimate direction (attention share grows with sequence
length, dominates for long/large clips) but the short-clip number is lower
than the earlier ~25% FLOPs estimate. Ceiling from Sol-Attn's own ~85%
sparsity: ~15% total block speedup at 512²/39fr vs ~56% at 768²/124fr — so
this item has little payoff for the short 4-step Turbo LoRA clips the app
now leans on, and a lot for long/large T2V and Ref2V.

### The real bar turned out to be higher than expected

The user chose to proceed with #5 for long/large clips. Immediately found
something that changes the difficulty estimate a lot: `h3_gpu_sdpa`
(`h3_gpu.m` `h3_gpu_sdpa_graph`) is **not** a naive/manual MPSGraph
composition — it's MPSGraph's own native
`scaledDotProductAttentionWithQueryTensor:...` op, i.e. Apple's
vendor-optimized fused attention kernel. Measured its achieved throughput
against the BF16 dense linears (same dtype, same GPU, from the microbench
above): SDPA runs at 10.1–12.7 TFLOP/s vs 13.6–14.2 TFLOP/s for
`h3_gpu_linear_bf16` — i.e. **~75–90% of the practical BF16 matmul ceiling
on this M5**. There is no "free" dense-kernel win sitting there; matching
vendor SDPA at all, before any sparsity, is the real bar.

### v1: naive scalar kernel — 63x slower

Built and correctness-tested a first custom tiled online-softmax Metal
kernel (`h3_flash_attn_dense_bf16`, scratch only) matching
`h3_gpu_sdpa_bf16`'s row-major `[seq, heads, head_dim]` layout: `TILE_Q=32`
(one thread per query row, one simdgroup/threadgroup), `TILE_KV=16`,
threadgroup memory for K/V tiles + output accumulator (24 KiB). Verified
correct against a CPU (Accelerate `cblas_sgemm`) double-matmul softmax
reference at real scale (seq=3204, heads=56): `max_abs_err` 8e-6,
`max_rel_err` 0.37%, consistent with BF16 rounding — the tiling/indexing
math is right.

But its **measured performance was 1477 ms for the same 56-head/seq=3204
attention MPSGraph does in 23.3 ms — about 63x slower**, because it's pure
per-thread scalar dot products (32 threads/threadgroup, nowhere near this
GPU's SIMD/matrix-unit width), not `simdgroup_matrix` or the Metal 4 tensor
primitives (`metal_tensor`/`MetalPerformancePrimitives`) already used
elsewhere in this codebase for the NAX kernels.

**Lesson:** a kernel worth shipping needs matrix-unit-based tiling
(`simdgroup_matrix` or Metal 4 tensor ops) to get anywhere near vendor
density before sparsity can help at all — even Sol-Attn's ~85% skip ratio
only pays off once the per-block cost of the *computed* blocks is within a
small factor of vendor SDPA, not 63x off.

### v2: matrix-unit dense attention — down to 2.1x

Rebuilt as a 3-pass design instead of v1's fused per-thread scalar loop,
reusing `h3_linear_bf16_nax_r128` verbatim (the same Metal 4 `matmul2d`
TensorOps kernel this codebase already uses for the DiT's own projections
and for `h3_gpu_lora_delta_bf16`'s validated `b[rows,K] @ at[cols,K]^T` GEMM)
for both QK^T and P@V, plus one new small row-softmax kernel
(`h3_softmax_rows_bf16`) and a transpose kernel for V.

Confirmed empirically (`h3_linear_bf16_nax_r128` has no public API
documentation — `MetalPerformancePrimitives.h` is just a 12-line stub) that
its buffer layout is: `input`/`weight` row-major with the reduction
dimension (`input_dim`) contiguous, `output` row-major with `output_dim`
contiguous — i.e. exactly this project's natural `[rows, head_dim]`-
contiguous Q/K layout, no transpose needed for QK^T; P@V needs V
pre-transposed to `[head_dim, kv_seq]`. Constraint confirmed from
`h3_gpu_lora_delta_bf16`'s own guard: the N/`output_dim` tile must be an
exact multiple of 64 (kv_seq padded up, padding masked to
-inf-equivalent in the softmax kernel via a `valid_columns` cutoff so it
contributes exactly zero); the M/`rows` tile needs no padding (hardware
handles the ragged last row-tile, matching the kernel's own comment).

**Correctness:** validated at real scale (seq=3204, heads=56, head 0
spot-checked against the CPU reference) — `mean_abs_err` 2.1e-6,
`max_abs_err` 1.6e-5 (BF16-precision floor, same order as v1); the one 8%
"max relative error" reading is a near-zero-denominator artifact (occurred
at a true value of 0.000001), not a bug.

**Performance**, iterated down in two steps from v1's 63x:

- Naive per-head dispatch (56 heads x 4 kernels, commit+wait every head):
  78.1 ms (of which 13.8 ms is pure CPU repack-copy, a test-harness
  artifact de-interleaving per-head Q/K/V out of the model's
  `[seq,heads,head_dim]` layout into tightly-packed buffers the kernel
  needs).
- Batching all 56 heads into ONE command buffer (single commit+wait)
  instead of 56: **49.3 ms** — confirms most of the per-head cost was
  command-buffer submission/sync overhead, not compute.
- vs MPSGraph's native SDPA: 23.3 ms. **Gap is ~2.1x, down from 63x.**

This changes the outlook a lot: Sol-Attn's own ~85% block-skip ratio would
already net a real win at a 2.1x per-tile efficiency gap (0.15 × 2.1 ≈ 0.32
of vendor's dense time — still a ~3x reduction in attention time), and the
remaining gap has concrete, not-yet-tried levers: (a) a small Metal repack
kernel (or a strided tensor view straight into the interleaved layout,
avoiding repack entirely) instead of CPU memcpy per head, (b) fewer/larger
dispatches (batch heads within one matmul call rather than 4 dispatches x
56 heads), (c) fusing softmax into the QK^T or P@V pass to cut a kernel
launch and a global-memory round trip for `scores`.

### v3: QK^T block skipping — a real, validated win (1.19–1.26x)

Added mean(Q_blk)·mean(K_blk) proxy scoring, per-row-tile threshold
tau = mu + beta·sigma, and a diagonal±1 anchor (always-kept local block) on
top of v2, skipping the masked QK^T tiles' `matmul2d.run()` entirely
(`h3_linear_bf16_nax_r128_masked` checks a per-(row_tile,col_tile) byte mask
before running; `h3_softmax_rows_masked_bf16` treats masked 64-column
groups as excluded, same zero-contribution mechanism as the seq-padding
mask in v2). PV was left fully dense in this prototype.

The first attempt was a real lesson, not just an implementation bug:
computed mean(Q_tile) and mean(K_tile) fresh inside every
(row_tile,col_tile) threadgroup pair — O(row_tiles·col_tiles·tile_rows·
head_dim) instead of the O(seq·head_dim) it should be, since each tile's
mean only depends on one axis. Result: sparse was *slower* than dense
(94.6ms vs 77.7ms, then 53.8ms vs 49.5ms after batching command buffers)
despite skipping ~72% of QK^T tiles — the naive scalar proxy-scoring pass
cost more wall-clock than the tensor-core matmul it was gating.

> **General lesson for this hardware:** because the dense matmul is now
> genuinely fast (matrix-unit tiles, ~2x off vendor), any gating/scoring
> mechanism has to be comparably cheap (linear in seq, reuse partial results
> across tiles) or it eats the entire win and then some.

Fixed by splitting into `h3_block_means_bf16` (reduce each axis's tile-means
once, O(seq·head_dim)) and `h3_block_dot_means_bf16` (a tiny
O(row_tiles·col_tiles·head_dim) dot pass over the small mean buffers, e.g.
26×51×128 ≈ 170k multiply-adds — trivial next to a QK^T tile).

After the fix, measured on structured-synthetic Q/K (8 random "cluster"
directions assigned to 64-row chunks + noise, since iid random data has no
exploitable redundancy for a fair sparsity test), M5, seq=3204/heads=56,
batched command buffers both phases:

| beta | tiles kept | speedup (QK^T-sparse, PV still dense) |
|---|---|---|
| 0.0 | 29.1% | 1.20x |
| 0.3 | 27.6% | 1.20x |
| 0.6 | 27.5% | 1.22x |
| 0.9 | 27.5% | 1.20x |
| 1.5 | 18.1% | 1.26x |

Speedup plateaus well below the skip ratio (e.g. 82% skipped → only 26%
faster) — expected Amdahl's-law ceiling since PV (~half the FLOPs) stays
fully dense and fixed overhead (proxy scoring, softmax, dispatch) doesn't
shrink with the skip ratio. **This is real, validated evidence the
mechanism works and nets a genuine wall-clock win once the scoring pass
itself is properly cheap** — not just a paper estimate anymore.

Quality signal from this run is **not** trustworthy yet and needs real
model data, not synthetic: `mean_abs_err` (0.0014) exceeded `mean|reference|`
(0.00076) at beta=0.6, and the synthetic generator's 64-row "cluster"
granularity doesn't match the 128-row Q-tile granularity used for
scoring/masking (each Q tile spans two clusters, diluting its mean) —
likely explains at least part of the poor number, but this needs checking
against real DiT block activations before drawing any conclusion about
real video quality impact.

### v4/v5: PV sparsity — negative result

Tried to sparsify PV too. Net result is negative as implemented, a real and
useful finding, not yet a win. Since a single `h3_linear_bf16_nax_r128` call
always reduces over its full declared K extent internally (no
documented/known-safe partial-K slicing), avoided that risk by **compacting**
each row-tile's surviving 64-wide KV blocks into a tightly packed buffer
first (two new gather kernels: `h3_gather_column_blocks_bf16` for P/scores —
the KV axis is the contiguous column axis — and `h3_gather_row_blocks_bf16`
for V — the KV axis is the outer row axis), then reusing the existing dense
`h3_linear_bf16_nax_r128` with a genuinely smaller K = kept_count×64.

Correct (same 0.0014/0.009 error figures as v3, since QK^T-side masking
dominates the error signal), but per (head, row_tile) this needs 4 small
encoder calls (gather-P, gather-V, transpose, matmul) instead of dense PV's
2 (transpose, matmul) — 26 row_tiles × 56 heads × 4 = **5824 tiny
dispatches** vs 112.

Measured: QK-only sparse still 1.20x (40.9ms, matches earlier); adding PV
sparsity on top made it **slower than dense** — 74.5ms (0.66x) vs dense's
49.1ms. The delta (33.6ms over ~5700 extra encoder calls ≈ 5.9us/call) lines
up exactly with CPU-side encode+launch overhead, confirming it's a
dispatch-count problem, not a logic bug.

> **Same lesson as the QK^T scoring pass originally hit:** on this hardware,
> with dense tensor-core matmul already this fast, sparsity only pays off
> when the overhead per skipped-or-kept unit is kept very low — many small
> dispatches lose to a few big ones even when they do genuinely less total
> compute.

### v6: batched PV sparsity — improved, still a net loss

Tried the batched-across-row_tiles fix — dispatch count came down as
predicted, but PV sparsity still doesn't beat dense. Added
`h3_gather_column_blocks_batched_bf16` / `h3_gather_row_blocks_batched_bf16`
(grid now spans all of a head's row_tiles via a 3rd/2nd grid dimension,
using the padding sentinel `0xFFFF` → exact zero contribution),
`h3_transpose_batched_bf16` (transposes each row_tile's own sub-block
independently, row_tile kept as an outer stride — a single flat transpose
of the whole per-head buffer would have interleaved row_tiles wrongly), and
`h3_linear_bf16_nax_r128_batched_weight` (each `group.x`/row_tile reads its
own weight sub-matrix via a pointer offset, since normal linear layers
share one weight across all rows but every row_tile's compacted V^T here is
different KV blocks). Every row_tile's kept-block count is padded up to one
shared per-head max (`kept_max`), so this is 4 dispatches per head total,
the same order as dense PV's 2.

Measured (same beta=0.6, same structured data): **55.4ms (0.88x)** — much
better than v5's 74.5ms (0.66x), but still slower than dense's 48.6ms.

Diagnosed the remaining gap: `kept_max` (the per-head max used for padding)
averaged 19.8 blocks (38.8% of 51) vs the true average kept count of 14.0
blocks (27.5%) — a real but modest 1.41x "waste" from padding every
row_tile up to its head's worst-case row_tile. The bigger cost is elsewhere:
`h3_transpose_batched_bf16` is a naive one-thread-per-element kernel (no
vectorization/coalescing), and it plus the two gather passes now
move/rearrange real data volume comparable to what the smaller matmul
saves — so the compaction machinery's own overhead (not dispatch count
anymore) is what's eating the win.

**Net conclusion:** this gather-then-smaller-dense-matmul strategy for PV
sparsity is validated **not** to pay off on this hardware even after fixing
dispatch count. Would need either:

- genuinely partial-K slicing of `h3_linear_bf16_nax_r128` without any
  separate gather/transpose materializing step (needs verifying `.slice()`'s
  behavior for a non-zero, bounded-width K offset — unconfirmed/risky, this
  API has no documentation on this machine), or
- a real fused kernel that skips blocks inline during a single matmul pass
  rather than compacting data beforehand.

Both are materially larger, riskier undertakings than what was attempted
across v4–v6. **QK^T-only sparsity remains the validated, real win from
this work (1.19–1.26x); PV-side sparsity is currently a dead end via the
compaction approach.**

### External review (2026-09-28) — several corrections, verified true

Another AI reviewed this document (via a shi3z article plus this repo at
`0ebbdbf`) and raised points that materially change the plan. Verified each
one directly rather than taking them on faith:

1. **The "QK-only sparse wins 1.2x" framing was comparing against the wrong
   baseline.** That 1.2x is against *this prototype's own* dense v2
   baseline (~48–49ms), not against the app's actual production SDPA
   (23.3ms). QK-only sparse (40.9ms) is still **~1.76x slower** than what
   the app uses today. Wiring it into `h3_dit.c` now would be a
   regression, not a speedup, until the matrix-unit implementation itself
   beats 23.3ms dense, before any sparsity is even added.

2. **`MetalPerformancePrimitives` is documented — the "no documentation"
   claim above was wrong.** Only the 12-line umbrella header
   (`MetalPerformancePrimitives.h`) was checked; the real API docs are in
   sibling headers under the same framework, e.g. `MPPTensorOpsMatMul2d.h`
   (verified present in the Xcode 27 SDK on this machine, ~660 lines). It
   documents, with worked examples:
   - A **K-tile-loop pattern** (`matMulKLoop`) — pick a `tilek`, loop over
     K in chunks calling `matmul2d.run(tA, tB, tC)` repeatedly into the
     *same* destination tile. This is exactly the genuine partial-K
     accumulation v4–v6 avoided by gathering/compacting data into a
     smaller dense buffer instead. `matmul2d_descriptor` has an explicit
     `mode::multiply_accumulate` vs `mode::multiply`, confirming
     accumulation across calls is a real, intended mode.
   - **`cooperative_tensor`**: a GEMM destination that stays in per-thread
     registers (not written to device/threadgroup memory), so
     post-processing — e.g. softmax normalization — happens in-register
     before one final `.store()`. This is the documented mechanism for
     never materializing a large intermediate score matrix, which v2–v6's
     3–4 pass designs all did.

   A real fused kernel (K-tiled QK^T → in-register online softmax via
   `cooperative_tensor` → K-tiled accumulate into the PV output, skipping
   whole K-chunks by simply not calling `.run()` for them) is a documented,
   intended usage pattern here — not undocumented/risky territory.

3. **`liuliu/ccv` has a real, validated Metal Sol Attention implementation**
   (`NAInt8SolAttentionKernel`, `doc/na-int8-sol-attention.md`, confirmed
   by fetching it directly) using exactly this hardware's INT8 Neural
   Accelerator path, block routing (proxy-score based, tau threshold, a
   configurable local-block radius, and an exact "protected" token range —
   useful for never approximating text/audio/reference-conditioning
   tokens), and critically: **no intermediate token-sized FP32
   numerator/softmax-state buffer and no dense token-by-token score
   matrix** — it shares FP32 numerator and softmax state in registers
   across a 5-launch (B64) pipeline. Its own published numbers (Apple M5
   Max, synthetic, N=1/H=56/D=128, B64/Q64/tau=0.5/radius=1): 2.82x/2.99x/
   2.85x sparse speedup at 32768/65536/103982 tokens — **against ccv's own
   dense INT8 attention, not against MPSGraph SDPA** (the doc says this
   explicitly). H3 quality was not evaluated (synthetic-only benchmark).
   **Studying/porting this design is now the recommended starting point
   for any further work here**, not more tuning of v2–v6.

4. **The 49 GiB memory-footprint problem is real — verified by direct
   calculation, matches exactly.** v2–v6's batched designs materialize a
   full `[seq, kv_padded]` BF16 scores buffer per head, and the "batch all
   56 heads in one command buffer" trick that gave the speedups allocates
   all of them at once. At seq=21728 (768²/124fr — exactly where sparsity
   matters most): 0.88 GiB/head × 56 heads = **49.3 GiB**. This is a
   serious flaw independent of the speedup numbers: the current prototype
   architecture doesn't scale to the long/large clips it's meant to help.
   A streaming, per-KV-block design (per points 2–3) sidesteps this
   entirely.

5. **`h3_cache_set_enabled` (`h3.h`/`h3.c`) is real, already implemented,
   and never called from the app — verified by `grep`.** See the new item
   7 below.

6. **`core_reuse`'s existing reduction is bigger than MotionCache's
   headline number implies — verified by calling `h3_dit_reuse_schedule`
   directly.** See the reframed item 6 below.

7. Miowtion/Veda's own README (re-fetched 2026-09-28) now says it supports
   both FL2VA and Ref2VA checkpoints — the "T2VA only" note below was
   accurate when written but is stale. Its sparse kernel is still CUDA-only
   (FlashAttention-4 CuTe, vendored SM8x patch), so it stays lower priority
   for this Mac-only app regardless.

**Given 1–4, the recommended next step if resuming Sol-Attn is *not* "keep
tuning v2–v6" but a fresh attempt studying `ccv`'s
`na-int8-sol-attention.md` design and using the now-confirmed `matmul2d`
K-loop + `cooperative_tensor` APIs for a genuinely fused kernel that never
materializes a scores matrix — and to benchmark any candidate against the
app's actual 23.3ms production SDPA, not a from-scratch dense baseline,
before claiming a win.**

### Reproducible bench vs production SDPA (2026-09-28)

Cloned `liuliu/ccv` (commit `6a611be`, `unstable` branch) and got its own
official benchmark tools building. Two real findings, one negative and one
strongly positive:

**Update 2026-09-28, same day — the Sol/sparse compile blocker is fixed.**
A patch was proposed (by another AI reviewing this document, working from
the exact error text and a matching report at
[ml-explore/mlx#4533](https://github.com/ml-explore/mlx/issues/4533))
diagnosing the cause as a Metal address-space qualifier surviving into a
`decltype(...)` used as a `get_destination_cooperative_tensor` template
argument, fixed by wrapping each with `metal::remove_addrspace_t<...>` (7
call sites in `NAInt8SolAttentionKernel.cpp`'s runtime-generated shader
source, no change to math/precision/layout). Applied it, rebuilt
`libccv.a` and `na_int8_sol_attention_bench`, and it now **compiles and
runs successfully** — the original diagnosis below (kept for the record)
turned out to be too pessimistic; this was fixable from outside Apple's
compiler after all. See "Sol/sparse kernel, now working" further below for
results.

Original (now-resolved) negative finding, kept for the record:
`bin/mfa/na_int8_sol_attention_bench` already does exactly what's needed
(accepts real captured contiguous-FP16 `[1,T,H,128]` Q/K/V via a
`raw-prefix` argument, compares native/all-exact-sol/sparse-sol). Building
it needed `./configure --enable-mps` in `lib/` (an opt-in flag, not
auto-detected) and `wget`. But its runtime-generated Metal shader **failed
to compile** on this machine: `error: no matching member function for call
to 'get_destination_cooperative_tensor'`, for matmul stages that chain a
`cooperative_tensor` as the left operand of a subsequent `matmul2d` — the
exact "no materialized scores buffer" fusion that makes this design
attractive. Assumed at the time to be an SDK/OS version mismatch (this
machine reports SDK 27.0; ccv's own published numbers were measured on
"macOS 26.6.2"); that assumption was wrong, or at least not the whole
story — the real cause was the address-space issue above.

**Strongly positive — ccv's plain dense int8 attention kernel (no sparsity,
no SDK blocker) already beats this app's production SDPA.** A second,
simpler ccv tool, `bin/mfa/na_int8_attention_bench` (just
`NAInt8AttentionKernel`, no Sol/routing stages), built and ran cleanly.
Run at this app's real shape (D=128, Hq=Hk=56) and its three real
sequence lengths:

| sequence length | ccv int8 NAX (quantize+int8, avg) | this app's real MPSGraph SDPA | speedup |
|---|---|---|---|
| 3214 (~512²/39fr) | 18.9 ms | 23.3 ms | 1.23x |
| 5744 (384²/124fr) | 54.0 ms | 75.0 ms | 1.39x |
| 21728 (768²/124fr) | 840.8 ms | 1340.3 ms | 1.59x |

(Synthetic random inputs — this tool has no raw-data-loading option, but
attention throughput at a given dtype/shape doesn't depend on data
content, only quality does. ccv's own internal FP16 "baseline" in the same
run was noisy and much slower than either column and isn't a useful
reference; the comparison that matters is the int8 column against this
app's own independently-measured MPSGraph number, same shapes. ccv's own
sampled validation against an FP32 reference reported `max_abs_o` ≈
3.8e-4 at all three shapes.)

**This is the clearest, most decisive result of the whole investigation:**
a real, already-working, already-validated-by-its-authors int8 kernel
beats this app's actual production attention by 1.2–1.6x, growing with
sequence length — without needing any sparsity at all. This re-opens
Sol-Attn in a more promising form than "block-sparse or nothing":
integrating a plain int8-quantized dense attention kernel (`ccv`'s
`NAInt8AttentionKernel` or an equivalent built from the same
`matmul2d`/`cooperative_tensor` primitives) is likely worth doing on its
own, before Sol's block-skip logic (which is still blocked by the SDK
issue above, and would only add to this dense win once available).

A capture hook now exists to get real (not synthetic) attention data out
of this engine for a follow-up quality check: `h3_dit.c` gained
`debug_dump_attention_qkv()`, called right after QKV projection/RoPE
(the exact input the production SDPA call consumes), behind three
off-by-default env vars (`H3_DUMP_ATTENTION_QKV=<prefix>`, `_BLOCK=<n>`,
`_STEP=<n>`) — converts BF16 to FP16 and writes `<prefix>.{q,k,v}.bin` in
ccv's expected format. Verified against a real 512²/39fr/4-step
generation, and used below for a real-data run of the Sol/sparse kernel
(not yet fed into `na_int8_attention_bench`, which still needs a small
patch to load real files — the dense kernel above was only checked on
synthetic-shape data).

### Sol/sparse kernel, now working — fast, but not yet a proven quality win

With the address-space patch applied, `na_int8_sol_attention_bench` runs
end-to-end and reports its own internal `native_ms` (plain int8, same
kernel family as the dense-only result above), `exact_ms` (Sol's
all-blocks-exact path, no approximation — a correctness check, not meant
to be fast), and `sparse_ms` (real routing/approximation active), plus
relative-L2 error against an FP32 reference.

Synthetic random Q/K/V, this app's real shape (H=56, head_dim=128) and
three real sequence lengths, default routing (`tau=0.5`,
`local_block_radius=1`, `approximation_start=470`):

| sequence length | native (dense int8) | sparse (Sol routing) | vs. this app's SDPA | sparse relative L2 |
|---|---|---|---|---|
| 3214 | 19.4 ms | 13.4 ms | 1.74x | 0.199 |
| 5744 | 54.8 ms | 38.5 ms | 1.95x | 0.217 |
| 21728 | 829.9 ms | 474.6 ms | 2.82x | 0.238 |

Then the same benchmark against the **real captured Q/K/V** from
`debug_dump_attention_qkv()` (block 25, step 2 of an actual 512²/39fr
generation, seed 7 — exactly T=3214, H=56):

```
native_ms=19.4265 exact_ms=27.2384 sparse_ms=12.6208
paired_sparse_speedup=1.5467  (i.e. 1.85x vs. this app's 23.3ms SDPA)
exact_block_fraction=0.512701
exact_relative_l2=5.94e-06        <- int8 quantization alone: negligible
protected_relative_l2=5.00e-06    <- protected/local region: negligible
sparse_relative_l2=0.1274         <- routed-approximate blocks: NOT negligible
```

**Read this carefully — it is a real speed win with a real, unresolved
quality question, not an unqualified win yet.** The int8 quantization
itself is essentially lossless here (~6e-6 relative error, consistent
with the dense-only kernel's ~3.8e-4 `max_abs_o` figure from ccv's own
validation). The cost is entirely in Sol's block-approximation: on real
attention inputs it produces a **12.7% relative L2 error** in the full
output, versus a fraction of a percent for the quantization-only path.
That is a per-attention-call, per-block number after just one DiT block's
worth of attention — whether it is acceptable depends entirely on how it
compounds or washes out across 50 DiT blocks and a VAE decode, which has
not been tested. A 13% relative-L2 difference could be invisible after
denoising smooths it out, or it could show up as visible drift/artifacts;
this document is not going to guess which without actually rendering a
frame. The synthetic-data error (0.199–0.238) is higher still, confirming
error is data-dependent as expected — real (correlated, RoPE-structured)
attention inputs approximate somewhat better than i.i.d. random noise, but
not enormously so.

Net: the reproducible bench asked for at the top of this section now
exists and gives a clear, honest answer — **dense int8 alone is a safe
1.2–1.6x win with negligible quality cost; adding Sol's sparsity on top
pushes that to 1.7–2.8x but at a quality cost that is not yet validated
end-to-end and must not be treated as free.**

### Correction: the "12.7% error" above was measured against the wrong reference — real ground-truth comparison now done, numbers revised

**The `sparse_relative_l2`/`exact_relative_l2` figures quoted just above are
ccv's own internal self-consistency check (its Sol-exact-mode output vs.
its own plain dense-kernel output, both computed by ccv from the same
input) — not a comparison against this app's actual real SDPA output.**
That distinction turned out to matter a lot once a direct comparison was
attempted, because it surfaced a real bug in how the input was captured.

Building the direct comparison (a new small tool,
`h3_real_attention_compare.cpp`, calling `ccv_nnc_mfa_encode_attention`
and `ccv_nnc_mfa_encode_sol_attention` directly on this app's captured
real Q/K/V, then diffing the result against this app's own captured real
SDPA output) surfaced two real bugs, both now fixed:

1. **The QKV capture itself was wrong.** This app's default (M5, int8
   QKV projection) path leaves `query`/`key`/`value` in `[heads, rows,
   head_dim]` layout for the immediately-following SDPA call to consume
   as a producer/consumer optimization (`h3_gpu.m`'s
   `headMajorSDPAInputs` flag) — but `debug_dump_attention_qkv()` assumed
   plain `[rows, heads, head_dim]` and wrote the head-major buffer out
   unpermuted. Every earlier "real capture" result in this document (the
   1.85x/0.1274 figures above) was computed on this silently-transposed,
   effectively-scrambled data. Fixed by adding a small accessor,
   `h3_gpu_head_major_sdpa_inputs()`, and having the dump function
   transpose to row-major before writing whenever it reports true. The
   first symptom, run through the (also newly fixed) real-vs-real
   comparison tool below, was unmistakable: `relative_l2=1.37` against
   this app's real output — worse than uncorrelated noise, an immediate
   sign of a layout bug rather than a genuine quality problem.
2. **The attention-cache used for capture had a stale weight fingerprint**
   (`h3: warning: attention cache ... model fingerprint does not match
   the loaded checkpoint`) - H3Spike auto-points at
   `~/models/cache/dit_int8_v2.cache` by default, and it was quietly out
   of sync with the currently-loaded checkpoint. The warning doesn't
   block generation (it's non-fatal), so this could have kept silently
   giving wrong int8-quantized weights without a hard failure. Fixed by
   rebuilding the cache (`./build_attention_cache
   ~/models/MiniMax-H3/FL2VA/transformer ~/models/cache/dit_int8_v2.cache`)
   before recapturing.

With both fixed, and also dumping this app's own real SDPA output
alongside the Q/K/V (a new sibling hook, `debug_dump_attention_output()`,
gated behind the same env vars, only meaningful with
`H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1` since the attention-output
tensor has its own, independent head-major toggle), three real
block/step captures from an actual 512²/39fr/4-step generation (seed 7)
were compared directly - ccv's kernel output vs. this app's own real
production SDPA output, not against any reference ccv computes
internally:

| block, step | ccv dense int8 vs. real SDPA | ccv Sol sparse vs. real SDPA |
|---|---|---|
| 0, step 0 | relative L2 1.56%, PSNR 62.0 dB | relative L2 9.97%, PSNR 45.8 dB |
| 25, step 2 | relative L2 1.77%, PSNR 57.7 dB | relative L2 10.18%, PSNR 41.5 dB |
| 49, step 3 | relative L2 4.57%, PSNR 52.9 dB | relative L2 6.14%, PSNR 50.3 dB |

**This is a much stronger and more trustworthy result than the earlier
one, in both directions.** Dense int8 is confirmed, against real
ground truth (not a self-consistency check), to closely reproduce this
app's actual attention output across early/mid/late blocks - 52-62 dB
PSNR is generally in the range other quality-affecting approximations in
this codebase are judged acceptable at (compare the VAE int8 work's 44.9
dB worst-frame PSNR, judged visually identical). Sol sparse is real but
consistently worse - 42-50 dB PSNR, a genuine, non-trivial quality cost
that is smaller than the earlier (buggy-data) 12.7%/41.5 dB estimate
suggested at one point and larger at another, but now backed by a real
reference instead of an internal check. Whether 42-50 dB per-block PSNR
survives 50 blocks + VAE decode without becoming visible is still the
open question - this correction sharpens the number that question is
about, it doesn't answer it.

**Lesson for any future real-data validation in this codebase**: this
engine has several independent, opt-in "head-major" layout optimizations
(SDPA inputs via QKV projection, SDPA's own output via
`h3_gpu_sdpa_bf16_head_major_output`) that silently change which axis is
contiguous, purely as GPU producer/consumer optimizations invisible to
the math - a raw tensor dump is only meaningful if it explicitly checks
and corrects for these, and a wildly-uncorrelated (`relative_l2 > 1`)
comparison against a real reference is a strong, checkable signal that a
layout assumption is wrong before concluding anything about quality.

### End-to-end integration and A/B result (2026-09-28): net negative at this clip length

Per the plan above, built a real, opt-in `H3_ATTENTION_BACKEND=ccv_dense`
path (not Sol — dense only, per instruction, with Sol deferred to its own
evaluation) that substitutes ccv's dense int8 kernel for every block's
production SDPA call, end to end, and ran the actual staged validation
this section called for. Net result: **at this clip length, it is a clear
net negative — no measurable speedup, and a real, visible quality
regression that the per-block numbers above did not predict.**

**What was built** (all opt-in; the default build and default runtime
behavior are unaffected):

- `h3_gpu_ccv_attention.mm` (new file, only built/linked when `CCV_DIR` is
  set — see `Makefile`): bridges h3's own `id<MTLDevice>`/
  `id<MTLCommandBuffer>`/`id<MTLBuffer>` objects to metal-cpp's
  `MTL::Device*`/`MTL::CommandBuffer*`/`MTL::Buffer*` (metal-cpp's
  documented interop — same pointer value, no separate allocation), calls
  `ccv_nnc_mfa_encode_attention` inside h3's *existing* command buffer via
  `ccv_nnc_start_command_batch_from_command_buffer(..., commit_on_finish=0)`
  (no extra command-buffer submit/wait per block), and self-registers via
  a static constructor so no other file needs an `#ifdef` to enable or
  disable it.
- Two new small Metal kernels (`h3_cast_bf16_to_f16_qkv`,
  `h3_cast_f16_to_bf16_flat`, in `h3_shaders.metal`) to convert BF16↔FP16,
  with the QKV cast also correcting for this engine's head-major/row-major
  layout split (see the correction above) so the backend is correct
  regardless of which internal layout was active — not just in the
  forced-row-major layout the validation captures used.
- `h3_gpu_head_major_sdpa_inputs()`/`h3_gpu_set_head_major_sdpa_inputs()`
  and a handful of other small raw accessors (`h3_gpu.h`/`h3_gpu.m`) so the
  bridging file never needs its own copy of h3's internal layout state.
- A process-wide dispatch counter
  (`h3_gpu_ccv_attention_dispatch_count()`) to positively confirm the
  backend ran for every block, rather than trusting that it did.
- `tools/ccv_eval/h3_generate_cli.c` (new): a plain-C equivalent of
  `native/H3Spike`'s generation driver — see "A real, still-unexplained
  Swift-runtime/ccv incompatibility" below for why this exists.

**Build design note**: `h3_gpu_ccv_attention.o` is deliberately kept out of
`libh3.a` itself. A static archive only links in a member that resolves
some other object's undefined symbol; since nothing calls into this file
directly (it self-registers via a constructor), archived it would simply
be dropped by any consumer linking `libh3.a` as `-lh3` — confirmed by
hitting exactly that (zero `ccv_nnc_mfa_*` symbols in the final binary)
before switching to linking it as a loose object file everywhere it's
wanted (`Makefile`'s `h3_generate_cli` target; `Package.swift` reads the
same `CCV_DIR` variable at `swift build` time and adds the same `.o` path
directly to the linker flags). A loose object file on the link line is
always included, sidestepping archive-member selection entirely. An
earlier attempt at a `weak_import` "link anchor" trick to force the
archive to include it broke the **default** (non-`CCV_DIR`) build — a real
regression, caught by testing an actual executable link (`build_
attention_cache`) after the change, not just `ar`-based `libh3.a`
creation, which doesn't check for unresolved symbols at all.

**Resolved (Stage ⑨, 2026-09-30).** In this environment, the SDK version
recorded in the executable decided whether compiling without explicit
options succeeded; stating the required Metal Shading Language 4.0 in the
compile options fixed it for the normal Swift build. Details: ccv compiles its generated shaders with `nil` options, so the
runtime picks the default language version from the SDK version recorded
in the executable's `LC_BUILD_VERSION`. SwiftPM links H3Spike with
`minos 13.0 / sdk 13.0`; the `clang`-built tools record `sdk 27.0`. The
shader needs Metal 4.0 (`MetalPerformancePrimitives`), hence `use of
undeclared identifier 'mpp'` only from Swift. Confirmed both ways: a plain
C binary linked with `-Wl,-platform_version,macos,13.0,13.0` fails the
same way, and H3Spike linked with SDK 27.0 recorded succeeds. Fixed in
ccv by `tools/ccv_eval/ccv-na-attention-msl4-options.patch` (explicit
Metal 4.0 in `NAInt8AttentionKernel`/`NAAttentionKernel`, the same thing
`h3_gpu.m` already does for h3's shaders); with it, H3Spike as normally
built (sdk 13.0) runs the direct path. The earlier notes below are kept as
history — the "not Swift/XPC as such" caution turned out to be right.

**(History) An unexplained shader-compile failure that reproduces only from
H3Spike — root cause not identified, do not over-attribute it.** The
exact same `ccv_nnc_mfa_encode_attention` call, with the exact same shape
and params validated above, fails deterministically with a garbled-
looking Metal shader compile error (`use of undeclared identifier 'mpp'`,
or `redefinition of 'dynamic_extent'` depending on the run) whenever it is
called from `native/H3Spike` (a Swift/SwiftPM executable) — including in
the most minimal possible reproduction (a fresh `h3_gpu`, no model
loaded, called immediately after `h3_gpu_create()`, via a diagnostic
entry point, `h3_debug_ccv_warmup()` in `h3.c`/`h3.h`, exercised from
`main.swift` behind `H3_CCV_WARMUP_DIAG=1`). The exact same call, with the
exact same shapes/params/scratch-buffer setup, succeeds every time from a
plain C or Objective-C++ binary built directly with `clang`.

What this rules out, each via a direct isolated test: GPU memory pressure
(tested up to ~62GB, both as a few large buffers and as ~800 small ones);
h3's own ~100-kernel shader library being compiled on the same device
first; a buffer-array-size bug (a real, separate bug that *was* found and
fixed this way — `ccv_nnc_mfa_encode_attention` reads a fixed 10-slot
tensor/offset array internally, and a 4-slot array reads past its end;
fixed, and confirmed the Swift-only failure persists afterward with the
bug gone, so it is not what's causing this); matching
`-mmacosx-version-min` between the two binaries; and merely linking
`libswiftXPC`/`libswiftCore` into a plain C binary without an actual
Swift runtime present.

**What this does *not* establish: a "Swift runtime / XPC incompatibility"
as a confirmed cause.** An earlier draft of this section claimed that;
it was too strong a conclusion from the tests actually run, for reasons
worth recording so this isn't re-claimed later without better evidence:
a call succeeding from a plain C driver and failing from Swift is equally
consistent with an ordinary memory bug (uninitialized read, use-after-
free, or similar) that happens to behave differently across the two
binaries because of caller-side differences (memory layout, allocation
order, object lifetimes) having nothing to do with Swift or XPC per se —
exactly the kind of thing the array-size bug above turned out to be
before it was found. ccv's author also ships `s4nnc`, a Swift interface to
the same underlying library, which doesn't prove this exact kernel/SDK
combination is fine under Swift, but does mean "Swift and ccv are
inherently incompatible" is not a safe default assumption either. The
"garbled-looking" description of the error text is itself just a
description, not a diagnosis — it hasn't been established whether the
*generated Metal source itself* differs between the two runs (which would
point at generation-time memory corruption) or whether the source is
identical and something about compiling or reporting on it differs (which
would point elsewhere). That specific check — dumping the exact source
string, byte length, and compile options right before `newLibraryWithSource`
in both the C and Swift paths, and diffing them — is the next concrete
step if this is picked back up, before any further theorizing about
XPC or the Swift runtime specifically.

Given the significant time already spent isolating this, it was set aside
as a real, reproducible, but **causally unexplained** finding, recorded
here as exactly that. **Workaround**: `tools/ccv_eval/h3_generate_cli.c`,
a plain-C drop-in replacement for `H3Spike`'s generation driver using the
same public `h3.h` API, sidesteps it completely and was used for
everything below. Its results are not weakened by this open question:
they come from the plain-C path where the encode call is known to run
without error, and the finding below that the integrated path reproduces
the independently-validated standalone kernel almost exactly means the
plain-C driver is measuring the real kernel's real behavior, not some
CLI-specific artifact.

**Stage ① (connectivity check): passed.** Same real generation
(512²/39fr/4-step, seed 7), `H3_ATTENTION_BACKEND=ccv_dense` active for
every block: `h3_gpu_ccv_attention_dispatch_count()` read back **200** —
exactly 50 DiT blocks × 4 denoise steps, confirming the backend ran for
every single attention call with zero silent fallback to the default
path.

**Additional check the external review specifically asked for, and
correctly identified as missing: does the *integrated* path reproduce the
independently-validated *standalone* kernel on the same input, or could
the wrapper itself (layout handling, cast kernels, command-buffer
sharing) be the source of the later quality loss?** Captured real Q/K/V
at block 0, step 0 of the `ccv_dense` run (the one point in the whole run
guaranteed not to have diverged yet from the baseline run, since nothing
attention-related has executed before it — confirmed: the two captures
are bit-identical, `relative_l2=0`) and compared the integrated backend's
actual output for that block against `h3_real_attention_compare`'s
earlier, independently-validated output for the identical input:

```
integrated ccv output vs. standalone ccv output (same Q/K/V): relative_l2=0.17%, PSNR=81.3 dB
integrated ccv output vs. real baseline SDPA:                 relative_l2=1.57%, PSNR=61.9 dB
(reference) standalone ccv output vs. real baseline, from earlier validation: relative_l2=1.56%, PSNR=62.0 dB
```

**The integrated wrapper reproduces the standalone, independently-
validated kernel almost exactly (81 dB — consistent with ordinary FP16
rounding-mode differences between two independently-written BF16→FP16
conversions, not a bug) and matches the earlier standalone-vs-baseline
number to within 0.1 dB.** This is real evidence, not just an assumption,
that the wrapper's layout handling, BF16↔FP16 casts, and command-buffer
sharing are correct — the later quality loss is not explained by an
integration bug at this checkpoint. It does not by itself rule out a
different bug that only manifests after several blocks (e.g. a subtle
buffer-reuse issue in the cached FP16 scratch buffers across many calls),
but combined with GPU API/shader validation being unremarkable in these
runs, it shifts the weight of evidence toward genuine cumulative
approximation error rather than a wiring bug. Also checked and ruled out
as a contributing factor: FP16 overflow/underflow on the real captured
data (`max_abs` 12.7–120.5, `min_nonzero_abs` ~6×10⁻⁵, zero
inf/nan — comfortably inside FP16's representable range, nowhere near its
~65504 ceiling).

As corroborating (not conclusive) evidence of compounding: the same Q/K/V
capture at block 25/step 2 of the `ccv_dense` run — by which point 25
blocks' and 2 steps' worth of `ccv_dense` attention have already run —
had already diverged substantially from the baseline run's block-25 input
(relative L2 16–26% across Q/K/V, vs. exactly 0% at block 0). This is the
expected shape for cumulative divergence (small per-block error changing
each block's output, which becomes the next block's input) but a single
midpoint isn't a full trace; a step-by-step or block-by-block divergence
curve was not collected and would be needed to say more precisely where
or how sharply it grows.

**Stage ② (generation A/B, same clip): dense int8 is a net negative
here.** Config double-checked against the log to rule out confusion with
this document's own earlier, unrelated 20-step "fastest"-preset benchmark
(which happens to report a similar wall-clock number for a different
config): this run's log shows exactly 4 denoise steps (`enqueue 0/4`
through `4/4`), `core_reuse`/`denoise_reuse` at their no-reduction default
of 1, `dit_layers=50`, `token_reduction=0` — a full, unreduced 4-step run,
not an accidental mix-up. One run each (not a repeated/averaged
measurement — see caveat below):

| | baseline (production SDPA) | `H3_ATTENTION_BACKEND=ccv_dense` |
|---|---|---|
| wall clock | 73.7s | 74.2s (no measurable speedup — see caveat) |
| decoded frames vs. baseline | — | mean PSNR 18.19 dB, worst 17.59 dB (frame 4) |

**Timing caveat**: this is one run of each, not a distribution — 73.7s vs.
74.2s does not by itself establish "no speedup" to any precision tighter
than run-to-run noise. What can be said: no speedup large enough to show
above that noise was observed in this single comparison. This is,
however, exactly the outcome the external review's own math predicted
ahead of running it: the earlier measured attention-share (~17.8% of one
DiT block at this short sequence length) combined with dense int8's own
measured 1.23x-per-op speedup predicts a whole-block improvement of only
≈1.03x — small enough that it would need several repeated runs to
distinguish from noise even if real, which was not done here. **The
practical conclusion is unchanged either way: the speed case for this
backend was always about long/large sequences (the ~65.7% attention-share,
768²/124fr case, not tested), not this short clip**, so further chasing
precision on this particular number is not a priority.

**Correction (same day, after showing both videos to the user): the
original description of this as a "banding/striping artifact" was wrong.**
On careful re-inspection — cropped/zoomed regions and a proper side-by-
side, not just a quick look at a thumbnail — there is no periodic
banding or striping in either video. What's actually visible, at seed 7,
is a same-composition/same-pose result with different *fine detail*: the
cat's tabby stripe patterning is more pronounced in the `ccv_dense`
frame, the overall color grading shifts slightly (more golden vs. more
orange-and-white), and the blurred background shows a sharper/more
defined edge (a door or shelf frame) than the baseline's smoother blur.
This is a genuine, visible divergence, but it reads as "a similar but
distinct generation" rather than an image-quality defect — closer to what
this document elsewhere calls a trajectory/composition change (cf. the
core_reuse/token_reduction "~12 dB PSNR, composition changes" note near
the top of this file) than to noise, corruption, or a rendering bug.

**Also checked a second seed (123) for reproducibility, and confirmed
important nuance: the effect is real but seed-dependent in magnitude, not
a fixed artifact.** Same config, `H3_ATTENTION_BACKEND=ccv_dense` vs.
baseline, decoded through `ffmpeg`'s PSNR filter on the final encoded
video (a different, cross-checking measurement from the raw-RGB PSNR
used for seed 7 above — both are reported here since they were measured
differently): seed 7 gives Y=19.7 dB/avg=21.4 dB, seed 123 gives Y=21.1
dB/avg=22.8 dB. Both are real, both are well outside the "visually
identical" range this codebase treats as safe elsewhere (VAE int8: 44.9
dB worst-frame) — but side-by-side, seed 123's two videos look much closer
to each other than seed 7's, consistent with the numeric gap. **The
practical implication: whatever is driving this is not a fixed-magnitude
bug that always produces the same visible defect — it varies with the
specific input/trajectory, which is more consistent with compounding
approximation error (sensitive to the specific values being compounded)
than with a deterministic wiring bug (which would be expected to produce
a more consistent symptom regardless of seed).**

**This still directly confirms the concern raised before this run — per-
block PSNR against real ground truth (52-62 dB) did not predict final
quality — but the corrected visual description matters for what to do
next.** Attention error compounds across 50 blocks × 4 steps in a way a
per-block, non-substituting check cannot show. The integration-vs-
standalone match above (81 dB at block 0) makes it unlikely that the
wrapper itself (layout, casts, command-buffer sharing) is the primary
cause, which shifts weight toward genuine cumulative approximation error
in ccv's int8 quantization as the more likely explanation — but "unlikely
to be the wrapper" is not the same as "proven to be the kernel": a bug
that only manifests after many calls (e.g. in the cached FP16 scratch-
buffer reuse across blocks) is not excluded by a block-0 check, and was
not separately tested. Given the corrected description (subtle,
seed-dependent detail/composition drift, not a fixed visual defect), a
bug that always produces the same wrong symptom regardless of input seems
somewhat less likely than before this correction, but this is still not
confirmed either way.

**Stage ③ (long sequence, 768²/124fr/4-step, seed 7): the first real
measured speedup, smaller than predicted, plus real memory and quality
numbers at this scale.** This is the case the whole speed argument was
actually about (~65.7% attention-share vs. ~17.8% at the short clip
above), measured via `/usr/bin/time -l` for wall clock and peak memory:

| | baseline (production SDPA) | `H3_ATTENTION_BACKEND=ccv_dense` |
|---|---|---|
| wall clock | 505.3s | 462.2s (**1.09x — a real, measured speedup**) |
| peak memory footprint | 15.6 GB | 17.4 GB (+1.8 GB) |
| peak RSS | 7.63 GB | 7.93 GB (+0.3 GB) |
| dispatch count | — | 200/200 (again exactly 50 blocks × 4 steps) |
| decoded video vs. baseline (ffmpeg PSNR) | — | Y 21.3 dB, avg 23.0 dB |

**Speed**: real, positive, but smaller than the naive prediction. Combining
the standalone attention-only benchmark (840.8 ms ccv vs. 1340.3 ms real
SDPA at this sequence length, from earlier in this section) with the
65.7% attention-share measurement predicts a whole-block improvement of
≈1.32x (`1 / (0.657/1.59 + 0.343)`); the actual measured whole-generation
improvement was only 1.09x. The gap is plausibly the BF16↔FP16 cast
kernels, scratch-buffer setup, and command-batch overhead per block (not
present in the isolated attention-only microbenchmark) eating into the
theoretical gain — not measured separately here, so this is a plausible
explanation, not a confirmed one.

**Memory**: `ccv_dense` uses **more** memory, not less — about +1.8 GB
peak footprint, consistent with the four extra FP16 scratch buffers this
backend allocates (`q_f16`/`k_f16`/`v_f16`/`out_f16`, each
`rows×heads×head_dim×2` bytes ≈ 311 MB at this sequence length, ×4 ≈
1.25 GB, roughly matching the observed increase). This machine has 24 GB
total unified memory; both runs stayed well under that (15.6–17.4 GB
peak), but the margin is not huge, and a smaller-memory machine or a
larger clip could matter here in a way it wouldn't have shown up at the
short-clip scale.

**Quality: this is where stage ③ actually diverges from the short clip,
and it's worse, not the same.** PSNR at this scale (Y 21.3 dB, avg 23.0
dB) is numerically close to both short-clip seeds (seed 7: Y 19.7/avg
21.4; seed 123: Y 21.1/avg 22.8) — an early draft of this section inferred
from that alone that the *character* of the divergence was probably the
same "similar-but-distinct generation" seen at the short clip. **That
inference was wrong, and checking it by actually looking (the same
zoomed/side-by-side treatment given to the short clip) matters more than
the PSNR number suggested.** At this scale the two frame-0 decodes show a
**different cat** — different coat pattern (baseline: brown/black
calico-tabby markings; `ccv_dense`: solid orange tabby), different pose
(baseline: cat lying low against the yarn; `ccv_dense`: cat with a paw
raised near the yarn), not the "same cat, slightly different fur/color/
background sharpness" seen at the short clip. This is a composition-level
divergence, closer to what this document elsewhere calls a trajectory
change (cf. the core_reuse/token_reduction "~12 dB PSNR, composition
changes" note) than the more subtle detail-level drift at the short clip —
despite both landing in a similar PSNR range. **PSNR alone did not
distinguish these two qualitatively different outcomes; only looking at
the actual frames did**, which is itself a useful, generalizable lesson
for judging any future approximation here by number alone.

**Net for the long-sequence case**: a real, positive but modest speed win
(1.09x) at the cost of more (not less) peak memory and a quality
divergence that is, on direct visual inspection, *more* severe in kind
than the short clip's — a full composition/content change, not a subtle
detail shift. This makes the long-sequence case, where the speed benefit
actually exists, the one with the clearer-looking quality problem too.
Whether this is genuine cumulative approximation error (more DiT blocks ×
more steps × more spatial-temporal content for error to compound through)
or a bug that only manifests at scale is exactly as open as before — but
the visual evidence now argues for more caution at long sequences, not
less, which is the opposite of what "same PSNR as short clip" would have
suggested on its own.

**Conclusion and recommendation**: do not adopt `H3_ATTENTION_BACKEND=
ccv_dense` as implemented — not because the speed case is false (stage ③
shows it is real, if smaller than predicted, at the long-sequence scale
this was always meant for) but because the quality question underneath it
is still open at every scale tested, short or long, and now with a
confirmed real speed incentive to actually answer it rather than shelve
it. If this is picked back up, the highest-value next steps, in order:
(1) a per-block or per-step divergence trace (Q/K/V relative-L2 at every
block, not just block 0 and block 25) to see whether error grows
gradually or jumps sharply at a specific point; (2) a layout/cast-only
round-trip experiment (cast Q/K/V BF16→FP16→BF16 through the same kernels
this backend uses, then run this engine's own production SDPA on the
round-tripped tensors, comparing against the untouched baseline) to
isolate the cast path from ccv's int8 compute more directly than the
block-0 check above does; (3) a proper visual review of the long-sequence
output at the same level of care as the short-clip side-by-side above
(this section's stage-③ PSNR numbers were checked, but not re-inspected
frame-by-frame with zoomed crops the way the short clip was, so "same
character as the short clip" is an inference from matching PSNR, not a
confirmed visual match). Sol sparse attention was explicitly deferred to
its own evaluation per instruction, and this result is a reason for extra
caution there too, not less — Sol adds approximation on top of a
dense-int8 path
whose end-to-end quality is not yet fully understood.

**Stage ④ (production-length, 768²/362fr ≈ 15s @ 24fps/4-step, seed 7):
a ~19% time reduction at 15s, with loss of fine detail and a strengthened
foreground net pattern — adoption on hold, cause not identified.**
`align_frames`-style
rounding (5 + 17k, matching the GUI's `align_frames`) puts a 15s clip at
362 frames — 2.92x stage ③'s 124 frames. Measured the same way
(`/usr/bin/time -l`, `H3SPIKE_FRAMES=362` otherwise identical params to
stage ③, `h3_generate_cli`):

| | baseline (production SDPA) | `H3_ATTENTION_BACKEND=ccv_dense` |
|---|---|---|
| wall clock | 2976.8s (49.6 min) | 2412.6s (40.2 min) (**1.23x — the best measured speedup yet, and it grows with length as expected**) |
| peak memory footprint | 20.78 GiB | 25.40 GiB (+4.6 GiB) |
| peak RSS | 10.77 GiB | 11.09 GiB (+0.3 GiB) |
| dispatch count | — | 200/200 (again exactly 50 blocks × 4 steps) |
| decoded video vs. baseline (ffmpeg PSNR) | — | Y 23.3 dB, avg 25.0 dB |

**Speed**: real and, as expected from the attention-share argument, larger
than stage ③'s 1.09x — a longer sequence spends a bigger fraction of its
time in attention, so `ccv_dense`'s per-op speedup there compounds into a
bigger whole-generation win. This is the first result that actually
supports the "longer videos benefit more" case the whole effort was aimed
at.

**Memory**: peak footprint grew by +4.6 GiB here vs. stage ③'s +1.8 GiB —
consistent with the four FP16 scratch buffers scaling with `rows` (which
is ~2.92x larger at 362 frames), not a fixed overhead. Both runs still fit
in this machine's 24 GiB unified memory, but the ccv_dense side's peak
(25.40 GiB) is now *above* that nominal figure by the accounting `time -l`
uses. Peak RSS (~11 GiB) alone doesn't settle GPU-side headroom either, so
for a 24 GB target, memory compression/swap activity during the run should
be part of any adoption criterion. In plain terms: 2976.8s → 2412.6s is a
~9.4 minute, ~19% time reduction — practically meaningful — bought with a
+4.62 GiB (~22%) footprint increase, which is a real cost, and the speed
ratio needs re-measuring after any quality fix.

**Quality — corrected.** An earlier version of this paragraph claimed a
"repeating diagonal lattice pattern ... absent from the baseline entirely"
and argued from its regularity that it was a structured kernel artifact
rather than quantization error. **Both claims were wrong.** The user's
own viewing, then a full-size re-check of the baseline frames, shows the
baseline *also* has the diagonal net across the whole frame: it is a
scene element (an out-of-focus foreground net/fence the model put between
the camera and the cat), present in both outputs. What actually differs:
in `ccv_dense` the net is rendered more strongly/in-focus, while the cat
loses fine detail (eyes, whiskers, fur texture) and the grading is
darker/duller. So the observation is "fine-detail loss plus a
strengthened existing net pattern," not "a new lattice." Two hypotheses
remain undistinguished: the model emphasizing an existing scene element
differently (a content/focus change), or a periodic computational error
adding to it — separating them needs full-size, pre-encode RGB frames
(`H3SPIKE_DUMP`) and more than four stills. The "regular, therefore not
int8 quantization" reasoning was also unsound on its own: rounding is
deterministic, and per-tile shared scales can produce errors that follow
input/block structure, so regularity alone can't distinguish quantization
from a layout bug or a RoPE interaction.

PSNR: the 15s clip's Y 23.3/avg 25.0 dB is not comparable to stage ③'s
23.0 dB as "better" — they cover different frame sets, ffmpeg's overall
PSNR aggregates per-frame MSE over whatever frames are compared, and these
YUV numbers are also not comparable to the earlier raw-RGB PSNR figures
in this section.

**What the "dense" int8 path actually quantizes (verified in the local ccv
checkout, commit `6a611be`)**: not just Q/K. `NAInt8AttentionKernel.cpp`
has `quantize_q`/`quantize_k`/`quantize_v` (per-tile max-abs scale,
clamped to ±127), and in the forward pass the attention weights P
(`exp(s − rowmax)`, in (0, 1]) are also multiplied by 127, rounded, and
used as int8 in the PV product (`P_QUANTIZATION_SCALE = 127.0f` when
Hq == Hk, which is h3's case; line ~1550). "Dense" means no blocks are
skipped, not that the approximation is mild. This bridge sets
`use_quantized_attention = 1` (`h3_gpu_ccv_attention.mm`). A plausible —
**unconfirmed** — length-dependent mechanism: with more keys per row,
more of the attention mass sits in small weights that 7-bit P rounding
zeroes out, which would weaken diffuse/global contributions more as
sequence length grows. This is a hypothesis to test, not a finding.

**Net for the production-length case**: long-sequence speedup observed
(~19% time reduction at 15s); fine-detail loss and a strengthened net
pattern observed; adoption on hold; cause not identified.

**Conclusion and recommendation (updated after stage ④)**: do not adopt
`H3_ATTENTION_BACKEND=ccv_dense` yet. Next step, before repeating 15s
generations, is a three-way split on the short clip (where divergence
already shows):

| path | purpose |
|---|---|
| A: production MPSGraph SDPA | reference |
| B: same bridge + layout/cast path, **non-quantized** ccv attention (`use_quantized_attention = 0`) | checks casts, integration, and ccv's float path |
| C: current ccv dense int8 attention | B→C difference isolates the int8 path |

- B also degrades badly → look at the bridge, layout, casts, or ccv's
  float kernel.
- B fine, only C degrades → narrow to the int8 path (still separating
  "inherent limit of this quantization scheme" from "implementation bug").
- Integrated C disagrees with standalone C on the same Q/K/V → fix the
  integration first.

Length-dependent behavior can then be checked cheaply by replaying single
blocks with Q/K/V captured from the 15s run instead of full 50-minute
generations. Note `upcast=1` does not un-quantize P/V, so it is not a
substitute for B. The RoPE-interaction idea stays a hypothesis only.

**Caveat on stages ②–④**: every run above used **4 steps on the base
model without the Turbo LoRA**. The default is 20 steps; 4 steps is the
Turbo LoRA's operating point. Those outputs were under-denoised (soft
overall, baseline included), so stages ②–④ compare blurry outputs with
each other and don't represent production quality. The foreground "net"
in the 15s clip may be an intended scene element or a half-resolved
structure from under-denoising; not determined.

**Stage ⑤ (A/B/C split, 512²/39fr/20 steps, seed 7, `h3_generate_cli`).**
B = `H3_ATTENTION_BACKEND=ccv_fp16` (same bridge, casts and layout
handling as C, but `use_quantized_attention = 0`, i.e. ccv's
non-quantized FP16 NAX kernel; `H3_CCV_UPCAST=1` additionally switches
its intermediates to FP32, unused here). C = `ccv_dense` as before.

Block-level check first (block 0, step 0, where Q/K/V are bit-identical
across all three paths — confirmed, relative L2 = 0 for q/k/v):

| vs. A (production SDPA) | relative L2 | PSNR |
|---|---|---|
| B (ccv FP16) | 0.15% | 82.2 dB |
| C (ccv int8) | 1.57% | 61.9 dB |

So the bridge, casts and layout handling plus ccv's float kernel
reproduce production closely, and int8 quantization adds roughly 10x
more per-call error on top — this is the int8 path's cost, isolated.

Whole generation (1000 dispatches = 50 blocks × 20 steps for B and C):

| | A | B (ccv fp16) | C (ccv int8) |
|---|---|---|---|
| wall clock | 220.3s | 306.1s | 219.8s |
| peak footprint | 12.03 GiB | 12.20 GiB | 12.32 GiB |
| RGB PSNR vs. A (pre-encode, 39 frames) | — | 15.3 dB | 15.1 dB |
| RGB PSNR vs. B | — | — | 20.4 dB |

Findings:
- **At 20 steps all three outputs are sharp and detailed** (whiskers,
  fur, yarn texture). Zoomed crops of the face at frame 19 show
  comparable fine detail in all three; C may be marginally softer than B,
  but that is too subtle to assert without the user's own viewing. The
  heavy detail loss seen at 4 steps does not reproduce here.
- **Generation-level PSNR against A measures trajectory divergence, not
  quality.** B, whose per-call error is only 0.15%, ends up as far from A
  (15.3 dB) as C does (15.1 dB). Over 50 blocks × 20 steps even a tiny
  per-call difference moves the sample to a different — equally valid —
  video. B and C are much closer to each other (20.4 dB; same
  composition: cat centered, yarn left of center) than either is to A
  (cat further left, larger). That means the split away from A comes
  mostly from the non-int8 part (ccv's float kernel vs. MPSGraph), and
  int8 adds a smaller further drift.
- **Speed at this length**: C matches A (no gain at 39 frames, as
  before); B is ~39% slower, so ccv's FP16 kernel is not a speed option
  here.

Interim reading: at 20 steps and short length, no quality defect
attributable to the int8 path is visible; the per-call int8 cost is
real and measured (~1.6% relative L2 vs. ~0.15% for the float path). The
earlier stage ②–④ quality concerns are confounded by the 4-step
under-denoised setting. Not yet tested: other seeds, and long sequences
at 20 steps (where the speed gain is and where the P-rounding hypothesis
predicts more loss — a single-block replay with Q/K/V captured at 15s
length would test that cheaply).

**Stage ⑥ (production-length at 20 steps: 768²/362fr ≈ 15s, seed 7, A
vs. C).** Same `h3_generate_cli` params as stage ④ except
`H3SPIKE_STEPS=20`; pre-encode RGB dumped via `H3SPIKE_DUMP`.

| | A (production SDPA) | C (`ccv_dense`) |
|---|---|---|
| total (internal monotonic timer) | 13727s (3.81 h)* | 10703s (2.97 h) |
| median time per denoise step | 671.2s | 520.2s |
| decode + encode tail | 273.9s | 274.0s |
| peak memory footprint | 21.09 GiB | 25.70 GiB (+4.6 GiB) |
| peak RSS | 11.09 GiB | 10.40 GiB |
| dispatch count | — | 1000/1000 (50 blocks × 20 steps) |
| RGB PSNR vs. A (pre-encode, 362 frames) | — | 14.0 dB |
| mean pixel value | 79.8 | 68.8 |

\* A was paused (SIGSTOP) for ~56.6 min at the user's request between
steps 2 and 3. The raw total was 17120.9s; the step 2→3 interval was
4065.0s against a 671.2s median, so 3393.8s was subtracted. `/usr/bin/time
-l` reported an implausible 1524s wall clock for this stopped-and-resumed
process and is ignored; the engine's own monotonic timestamps are used.

**Speed: 1.28x overall, 50.4 min saved (22.0%)**; per denoise step
1.29x. The decode/encode tail is identical, so essentially all of the
gain is in the DiT, as expected. At 20 steps the DiT is a bigger share of
the total than at 4 steps, so the whole-run gain is larger than stage
④'s ~19%.

**Memory**: same +4.6 GiB footprint cost as stage ④ (it scales with
sequence length, not step count).

**Quality (full-size pre-encode frames at 0/120/240/361 plus zoomed face
crops)**: both outputs are sharp and detailed — fur, whiskers, and in C
clear blue eyes. The two are **different videos**: A is a tabby cat on a
floor with a light-blue yarn ball against a bright window; C is a
colorpoint (Siamese/Ragdoll-like) cat lying on a patterned bedspread with
a dark-blue yarn ball, under lower-key lighting. The lower mean pixel
value in C (68.8 vs. 79.8) matches that dimmer scene, so it is not by
itself evidence of degradation, and the 14 dB PSNR reflects different
content, not a quality score. **Neither shows the foreground net or the
detail loss seen in the 4-step stage-④ runs**, which supports the reading
that those came from the under-denoised 4-step setting. One thing to
watch: in C around frame 120 the paws near the yarn (a gray foreleg with
a white paw, a white paw with pink pads, and a black-and-white paw at the
edge) are hard to read and may be anatomically odd; that can't be judged
from stills and needs viewing in motion. Only one seed so far.

**Net (20 steps, 15s, seed 7)**: 1.28x faster (22% time reduction) at
+4.6 GiB peak footprint, with a different but comparably sharp video and
no visible systematic degradation in this sample. The block-level int8
error (1.6% relative L2) is real, but at this setting it shows up as a
different sample rather than a visibly worse one. Before any adoption:
more seeds and prompts (people/faces, fast motion), the paw region
checked in motion, and memory headroom on 24 GB machines
(compression/swap), since the footprint exceeds 24 GiB.

### Stage ⑦ — 24 GiB machine: compression/swap during a 15s-length run

Setup: 768x768, 362 frames (the 15s configuration), **2 steps** (the DiT
memory peak, not the full 20-step run), seed 7, 50 layers, on the 24 GiB
M5. `vm_stat`, `vm.swapusage` and `memory_pressure` sampled every 5 s
(321 / 268 samples). Other desktop apps were running; the machine was not
quiesced, and swap already held 1.70 GiB (baseline run) / 2.70 GiB
(ccv_dense run) at start, so the swap figures below are **increases from
each run's own start**. All values are system-wide, not per-process. Page
size 16 KiB; every "GiB" below is a page-counter delta × 16 KiB.

| | baseline (A) | ccv_dense |
|---|---|---|
| wall time (2 steps) | 1613 s | 1345 s (1.20x) |
| DiT time (2 steps) | 1314 s | 1047 s (1.25x) |
| per step (step 1 / step 2) | 653 s / 660 s | 522 s / 525 s |
| peak memory footprint (`time -l`) | 20.74 GiB | 25.35 GiB (+4.62 GiB) |
| peak RSS | 10.69 GiB | 10.35 GiB |
| swap used: max increase over run start (DiT) | +6.43 GiB | +9.42 GiB |
| swap used at end | 2.71 GiB | 2.76 GiB |
| compressor, pages occupied (max) | 10.80 GiB | 11.36 GiB |
| compressor, pages stored / uncompressed-equivalent (max) | 12.18 GiB | 11.87 GiB |
| free-memory % (min) | 17% | 2% |
| swap-out, whole run | 30.7 GiB | 32.5 GiB (+6%) |
| swap-out, DiT phase only | 29.7 GiB | 29.7 GiB |
| swap-in, whole run | 25.2 GiB | 25.2 GiB |
| compression volume, DiT phase | 41.6 GiB | 60.8 GiB (+46%) |
| decompression volume, DiT phase | 39.6 GiB | 59.8 GiB (+51%) |
| `Pageouts` counter | 0.010 GiB | 0.015 GiB |

Reading the counters: compression/decompression/swap-out/swap-in are
cumulative counters of *pages processed*, converted to GiB, not event
counts and not SSD bytes written; recompressing the same page counts again.
`Pageouts` is a separate (file-backed) counter and says nothing about
whether swap traffic reached the SSD. "Occupied" is the compressor's real
footprint; "stored" is the uncompressed-equivalent of what it holds.

What this shows:

- The 20-step run finished earlier (stage ⑥) and this 2-step run finished
  under the same conditions on a machine with other apps running, with both
  backends already swapping heavily: the **baseline itself** moves ~8 GiB
  per step through swap and has 17–19% free memory during the DiT.
- ccv_dense's speed-up holds under that pressure: 1.25x on the DiT phase
 , overall 1.20x on this short run
  because load and VAE (~270 s, unchanged) weigh more at 2 steps. Neither
  backend's step time worsened from step 1 to step 2.
- Swap use at peak is higher with ccv_dense (+9.4 vs +6.4 GiB over start,
  a 3.0 GiB difference — not 4.6), the free-memory low is much lower (2%,
  hit within the first ~3 minutes, i.e. load through the start of the DiT, then ~11% for the rest), and
  compression work in the DiT phase is ~46% higher. Total swap-out volume
  is similar (DiT phase identical at 29.7 GiB), so the extra footprint did
  not (measurably) multiply swap traffic; it cost extra compression work.
- **Not shown**: *which* data got pushed out to make room. The system-wide
  counters cannot tell whether ccv's buffers displaced this generation's
  weights or another app's memory. An earlier reading of this as "the extra
  ~4 GiB sits in swap as cold data" was a guess and is withdrawn.
- **Not shown**: how close to unusable the machine was. Free % alone is not
  a pressure measure (Apple weighs swap rate, wired/committed memory and
  file cache); no UI-responsiveness check was made and `memory_pressure`
  was only sampled for its free percentage.
- **Not shown**: behaviour with other apps closed (not measured), or over
  20 steps (only 2 steps were run; the step cadence was flat), or on 16 GB
  machines.

Where the +4.62 GiB comes from (from the code, not yet instrumented — **superseded by Stage ⑧**: rows is 62,847, not ~86.4k, and ccv's internal int8 scratch is part of the total):
`h3_gpu_ccv_attention.mm` `ensure_scratch` allocates **four** private FP16
buffers (q, k, v, out) of `rows × 56 heads × 128 dim × 2 B` each, once, at
the first attention call, and never releases them until the device changes
or the process exits — so they are also resident during the VAE phase.
Four buffers of ~1.15 GiB each ⇒ ~4.6 GiB matches the measured footprint
delta if `rows` ≈ 86.4k; the `rows` value itself was not printed, so this
match is by back-calculation, not confirmed. The bf16→fp16 cast copies
have the same element count as `dit->query/key/value`, and by grep those
three are only read by the attention call sites (and the QKV dump), which
makes an **in-place cast** (or a cast that reuses buffers the attention
call has already consumed) a candidate to cut ~3.5 of the 4.6 GiB. That
needs verifying against the activation-alias mode (`dit->qkv` aliasing)
and the dump hooks before any code change.

Status: **experimental opt-in only** (`H3_ATTENTION_BACKEND=ccv_dense`).
Not a default: image-quality validation (more seeds/prompts, paws in
motion) and memory reduction should both land first. Recorded conclusion:
"on the 24 GiB machine this configuration ran to completion under
swap/compression with other apps running and kept a 1.20x (2-step) /
1.28x (20-step) speed-up; where the extra memory is evicted from and
longer-run memory behaviour are not identified."

Next, in order: (1) instrument `rows` and the scratch sizes/lifetimes
(print at allocation); (2) test the in-place / shared-buffer variant for
the +4.6 GiB and free the scratch after the DiT; (3) re-measure with a
quiesced machine and with the per-phase (load/DiT/VAE) sampling above.

*Superseded (2026-10-04):* (1) and the post-DiT release were done in
Stage ⑧; the +4.6 GiB of FP16 cast copies went away with the direct path
in Stage ⑨, which is what the app's fast mode uses. What remains is ccv's
own int8 scratch (~1.26 GiB) during the DiT, and (3), a quiesced re-measure.

### Stage ⑧ — ccv memory breakdown (measured) and post-DiT release

`H3_CCV_MEMLOG=1` (diagnostic, in `h3_gpu_ccv_attention.mm`) prints the
bridge's and ccv's scratch sizes and `MTLDevice.currentAllocatedSize`
whenever either grows. 15s configuration (768x768, 362 frames), first
attention call:

| item | size |
|---|---|
| rows / heads / head_dim | 62,847 / 56 / 128 |
| input layout | BF16, head-major `[heads, rows, dim]` (int8 QKV path always emits head-major) |
| bridge FP16 Q/K/V/out (4 × 900,974,592 B) | 3.356 GiB |
| ccv context scratch, int8 path (`request_scratch`) | 1.266 GiB (int8 Q/K/V alone 1.259 GiB; rest scales, V means, L) |
| ccv context scratch, fp16 path | 0.016 GiB |
| sum, int8 path | **4.622 GiB** |

The sum matches the measured footprint delta (+4.62 GiB), consistent but
not proof that nothing else differs. The earlier Stage ⑦ back-calculation
(4 outer buffers, rows ≈ 86.4k) was wrong on both counts.

Lifetimes (from ccv `6a611be`, the commit in use): the bridge buffers are
allocated at the first attention call and were kept until process exit;
ccv's `request_scratch` only ever grows (sizes ≥ 512 MiB are allocated
exactly, smaller ones rounded up to a power of two) and is held by the
context until it is destroyed. Both therefore stayed resident through the
VAE.

Also from reading that commit: the int8 path's `data_type` switch accepts
`MTL::DataTypeBFloat`, and `batched` + `batch_dims_q` can describe the
head-major layout as batch = 56, Hq = Hk = 1, which would make the ccv
output head-major too, the layout the default int8 head-major output
projection consumes. That would remove all four bridge buffers (and the
two casts). **Not tried yet**: correctness/speed of the int8 kernel with
BF16 I/O and that shape are unverified.

In-place casting is not a drop-in: the bridge cast is a head-major →
row-major reorder as well as BF16 → FP16, so input and output cannot share
a buffer with the current kernel.

**Post-DiT release** (implemented): `h3_dit_release_backend_scratch()`
right after denoising returns in `h3.c` (cached or not) →
`h3_gpu_ccv_release_scratch()`, which acts only when the DiT's `h3_gpu`
has no open or in-flight command buffer (`h3_gpu_submit` waits for and
clears them), i.e. on GPU completion rather than on the CPU call
returning. It drops the four bridge buffers and shrinks ccv's scratch
back to its initial 64 KiB (`request_scratch` dereferences the current
buffer, so it cannot be null); the context and compiled pipelines stay.

Checks:

- Short clip (512x512, 39 frames, 20 steps, seed 7): RGB dump
  byte-identical to the pre-change ccv_dense run; release freed 0.297 GiB.
- 15s configuration, 2 steps: decoded video frames and audio samples
  identical (md5) to Stage ⑦'s ccv_dense run; device allocation
  27.590 → 22.967 GiB at release (−4.62 GiB). Wall 1340 s (was 1345 s).
- VAE phase (270 s, system-wide, other apps running): compressor
  occupied max 2.51 → 0.52 GiB, stored max 3.05 → 1.08 GiB, free-memory %
  median 68 → 76. Swap-used barely moved in either run during the VAE.
- As expected, the DiT is unchanged: peak footprint 25.35 GiB, swap max
  over start +9.46 GiB (was +9.42). This change does nothing for the DiT
  peak or the early low free-memory reading.

Next: the BF16 / batch = 56 head-major path above (removes up to 3.36 GiB
from the DiT itself), validated at block level against the current int8
path before any generation run; then the quiesced-machine 2-step
re-measurement.

### Stage ⑨ — BF16 head-major direct path (`H3_CCV_DIRECT=1`)

`H3_ATTENTION_BACKEND=ccv_dense H3_CCV_DIRECT=1`: when Q/K/V arrive
head-major (the default int8 QKV path) and the following projection can
take head-major output, the bridge passes the BF16 buffers straight to
ccv's int8 kernel as batch = 56, Hq = Hk = 1, writing BF16 head-major
output into `attention_heads`; the default int8 head-major output
projection then runs as in production. No bridge FP16 buffers and no
cast/reorder kernels. Requires the ccv patch
`tools/ccv_eval/ccv-na-int8-bf16-lse-store.patch` (BF16 `L` store failed
to compile). The attention output dump (`H3_DUMP_ATTENTION_QKV`) now
reorders head-major output on the CPU instead of skipping it, so path A
no longer needs `H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1` to be compared.

Block-level (512x512, 39 frames, block 0 step 0; Q/K/V byte-identical
across the three runs, so this is per-call error):

| vs | relative L2 | PSNR |
|---|---|---|
| ccv_dense (FP16 bridge) vs A | 1.5730% | 61.94 dB |
| direct vs A | 1.5725% | 61.94 dB |
| direct vs ccv_dense | 0.144% | 82.71 dB |

Direct keeps the same error against production as the existing int8 path
at this block; the 0.144% between the two is measured at block 0 step 0
only (not yet a per-call figure for every block) and is presumably BF16 vs
FP16→BF16 output rounding (not verified). Block 30 step 1 was also dumped, but its inputs already
differ between runs (Q differs 12%), so it only shows accumulated
divergence (18.0% direct, 18.5% ccv_dense vs A) and cannot be used to
compare kernel accuracy; a per-call check at a deep block needs a replay
on identical inputs.

15s configuration, 2 steps, same sampling as Stage ⑦ (other apps running,
system-wide counters, swap as increase over each run's start):

| | A | ccv_dense (Stage ⑧) | direct |
|---|---|---|---|
| DiT time (2 steps) | 1314 s | 1042 s | **954 s** |
| per step | 653 / 660 s | 522 / 525 s | 479 / 474 s |
| DiT speed-up vs A | 1.00x | 1.26x | **1.38x** |
| wall time | 1613 s | 1340 s | 1253 s (1.29x) |
| peak footprint (`time -l`) | 20.74 GiB | 25.35 GiB | **22.00 GiB** (+1.26) |
| device allocated, DiT | — | 27.59 GiB | 24.23 GiB |
| swap used, max increase (DiT) | +6.43 GiB | +9.46 GiB | +6.42 GiB |
| compressor occupied, max (DiT) | 10.80 GiB | 10.41 GiB | 6.66 GiB |
| free-memory %, min / median (DiT) | 17 / 20 | 8 / 13 | 17 / 19 |
| swap-out / compression volume, DiT | 29.3 / 41.4 GiB | 30.8 / 55.9 GiB | 26.6 / 39.2 GiB |

The remaining +1.26 GiB is ccv's int8 scratch (int8 Q/K/V 1.259 GiB),
released after the DiT (Stage ⑧). The change as a whole made the DiT ~9%
faster than the FP16 bridge; the cause is not isolated (dropped cast
passes, head-major instead of row-major output projection, the batched
dispatch shape, and less compression work could each contribute). The
baseline itself still swaps heavily on 24 GiB. Single run each, machine
not quiesced.

**Replay on identical inputs** (`tools/ccv_eval/h3_direct_replay.cpp`:
runs one captured call through the FP16 bridge setup and the direct setup,
diffs both against the captured MPS production output; the BF16 inputs
rebuilt from the FP16 captures were exact in every case):

| capture | bridge vs prod | direct vs prod | worst head (both) | last 128 rows (bridge / direct) | direct vs bridge |
|---|---|---|---|---|---|
| short, block 0 step 0 | 1.565% | 1.573% | #49: 3.67% / 3.68% | 1.35% / 1.35% | 0.169% |
| short, block 30 step 1 | 2.767% | 2.772% | #42: 9.62% / 9.63% | 2.55% / 2.56% | 0.169% |
| short, block 49 step 1 | 3.531% | 3.534% | #3: 12.80% / 12.80% | 3.25% / 3.25% | 0.168% |
| 15s (rows 62,847), block 0 step 0 | 1.673% | 1.680% | #49: 4.83% / 4.83% | 1.29% / 1.30% | 0.169% |

Direct tracks the bridge everywhere (≈0.17%, same worst head, no tail
effect), so the direct layout adds little of its own. Separately, for
**both** ccv int8 paths, the error against MPS was larger in the later
captures taken here (1.6% at block 0 step 0, 2.8% at block 30 step 1,
3.5% at block 49 step 1) — the block and the diffusion step both differ
between these captures, so this is not a clean depth trend. The 12.8% is
the worst single head's relative error (block total 3.53%); a head with a
small reference norm inflates relative error, so it does not by itself
show that head matters for image or motion. The shared difference from
MPS is not shown to be int8 quantization alone: ccv's other arithmetic is
common to both paths too, and isolating quantization would need the same
Q/K/V through ccv's non-int8 path.

**Short clip, 20 steps, seed 7** (512x512, 39 frames): direct's video is
very close to ccv_dense's (24.4 dB RGB PSNR between them, same
composition, very similar pose changes); both differ in composition from
A (15 dB), as before. No foreground net, lattice, or shape breakage seen
in any of the three. All three lift the head; A ends with a large turn of
the face to the right that the int8 runs lack, while they show the yarn
pulled taut to the mouth. Mean frame-to-frame difference 10.1 (A) vs 8.3
(both int8 paths), but that also depends on composition, texture and
brightness. In a ~1.6 s clip this cannot separate "a different action was
chosen" from "less ability to produce motion"; more seeds needed. DiT time here: A 182 s,
ccv_dense 182 s, direct 172 s (short sequences gain little).

**Three more seeds, short clip, 20 steps** (seeds 11, 23, 42 fixed in
advance; A vs direct; same prompt/model/sampler). Judged blind: each
seed's pair was shown as X/Y in random order (frames 0/8/16/24/31/38),
the assessment was written to a file before the key was opened. It turned
out Y = direct for all three seeds (a 1-in-4 outcome of the per-seed
shuffle; noted since a fixed position could have been learned, though the
key was not known while judging).

| seed | blind judgement (Y = direct) | mean \|ΔF\| A / direct | RGB PSNR A vs direct |
|---|---|---|---|
| 11 | both nuzzle the yarn, lift the head, turn to the camera; Y's turn slightly fuller | 8.41 / 8.83 | 17.4 dB |
| 23 | both nose-on-yarn, lift the head, thread pulled; Y ends with the thread at the mouth, slightly larger lift | 7.55 / 7.69 | 15.4 dB |
| 42 | both low-motion (head down on the yarn, sniffing); Y a little more mouth action | 6.95 / 6.40 | 17.5 dB |

No consistently weaker side over these three seeds (plus seed 7, where A
had the larger final head turn); no net/lattice or shape breakage visible
in either at half size. Frame-difference means point both ways. So: no
sign that direct systematically weakens the prompted action; proceeding
to the 15s, 20-step direct run.

**15s, 20 steps, seed 7, direct** (same settings as Stage ⑥; A and
ccv_dense from Stage ⑥ as references; engine monotonic timestamps, A
corrected for its 56.6 min pause as in Stage ⑥):

| | A | ccv_dense | direct |
|---|---|---|---|
| total | 13727 s (3.81 h) | 10703 s (2.97 h) | **9506 s (2.64 h)** |
| median per denoise step | 671.2 s | 520.2 s | **459.9 s** |
| decode + encode tail | 273.9 s | 274.0 s | 273.0 s |
| speed-up vs A (total / per step) | — | 1.28x / 1.29x | **1.44x / 1.46x** |
| time saved vs A | — | 50.4 min (22.0%) | **70.4 min (30.7%)** |
| peak footprint (`time -l`) | 21.09 GiB | 25.70 GiB | 22.35 GiB |
| RGB PSNR vs A / vs ccv_dense | — | 14.0 dB / — | 14.1 dB / 19.5 dB |
| mean \|ΔF\| | 5.21 | 3.48 | 4.01 |

The whole run was on this 24 GiB machine with other apps running; the
free disk space was low (16–19 GiB before the run, 8.2 GiB after), which
matters because swap lives on the same volume.

Quality, from full-size frames 0/120/240/361, crops of the paws at 120
and 240, and the video: direct is the same scene as ccv_dense (colorpoint
cat lying on a patterned bedspread with a dark-blue yarn ball, biting the
yarn, lifting its head at the end with clear blue eyes), with very similar
pose changes; A is a different scene (tabby on a floor, walks away at the
end). Direct is sharp (fur, whiskers, eyes); no foreground net or lattice
seen. Paws, after viewing the video (around 4–6 s and 9–11 s): a paw
showing its pads in the foreground overlaps a white paw reaching past the
yarn; no clearly extra limb, but the white toes split like thick fingers
and look somewhat artificial in places, especially in direct around 10 s
— ccv_dense has similar depictions. Recorded as: **the paw shapes are
suspected unnatural, but not confirmed to be made worse by the change to
direct**; A has a different pose and scene, so this comparison cannot
attribute them to int8 either. Direct also shows head movement, paw
movement, mouth to the yarn and the final head lift — motion is not lost.
This is not a reason to hold direct back. The lower mean frame difference of the int8 runs goes with
a lying cat versus A's walking cat; it is not a like-for-like motion
measure.

**Assessment so far**: no consistent degradation that would block
adoption was found in these comparisons (not a proof of equal quality).
No need at this point to route later blocks back to MPS or to change the
quantization. Direct vs A, 15s/20 steps (single run, pause-corrected):
1.44x, 70.4 min (30.7%) saved, +1.26 GiB footprint; vs ccv_dense: 11.2%
(~20 min) faster, 3.35 GiB less footprint. Seed-count checks on the cat
prompt stop here.

**Next** (toward an opt-in "fast mode (experimental)" on M5, telling the
user that the same seed gives a different video than standard mode):
(1) minimal direct run from H3Spike / the Swift app — kernel build and
dispatch first, then a short clip; re-reproduce the earlier Swift-only
problem on the current direct path if it is still there; (2) two short
generations back to back in the same app process — post-DiT release,
scratch re-allocation, cache reuse; (3) one person-motion prompt and one
fast-object prompt, short, 20 steps, same seed, A vs direct — prompt
adherence, shapes, temporal stability. Default-backend change only after
real use.

**Swift app checks** (2026-09-30):

- *Swift-only shader failure — root cause found* (see the Resolved note
  in item 5 above): Metal's default language version follows the SDK
  recorded in the executable (`sdk 13.0` for SwiftPM builds), which
  predates `MetalPerformancePrimitives`. Fixed by
  `tools/ccv_eval/ccv-na-attention-msl4-options.patch` (explicit Metal
  4.0 in ccv's two attention kernels). Not Swift- or XPC-specific.
- *Minimal run*: `H3_CCV_WARMUP_DIAG=1` passes from H3Spike as normally
  built. *Short clip from Swift* (direct, seed 7, 20 steps): decoded video
  md5-identical to the CLI's direct output.
- *Back to back in one process* (`H3SPIKE_REPEAT=2`, new): uncached (the
  app's default — `cache_enabled` is off unless set) and with
  `H3SPIKE_CACHE=1` (prepared DiT reused via `h3_dit_reset_run`): all four
  outputs md5-identical to the single CLI run; the post-DiT release and
  the next run's scratch re-allocation both happen (0.125 GiB each time).
  Peak footprint 13.4 GiB uncached, 18.7 GiB cached (conditioning and VAE
  decoder retained).
- *Unrelated to ccv, found on the way*: without a per-run
  `autoreleasepool`, H3Spike's top-level loop kept run 1's Metal objects
  alive — device allocation 14.4 → 26.2 GiB at run 2 (footprint unchanged,
  consistent with the mmap-backed no-copy weight buffers). With
  `autoreleasepool` per run it stays at 14.4 GiB. H3Engine calls
  `h3_generate` from `DispatchQueue.global().async`; system global queues
  default to `AutoreleaseFrequency.never` and set up no per-item pool, so
  nothing guarantees the app releases these per generation either (an
  earlier note here assumed it would — withdrawn). Fix: an
  `autoreleasepool` for one generation inside the background closure,
  covering `h3_generate` and the result handling/cleanup, exited on
  success, error and cancel alike; the pool is not a substitute for the
  existing GPU-completion waits. Whether the app actually grew by the same
  amount is to be checked after applying it.

**Person and fast-object prompts** (short, 20 steps, seed 7, A vs
direct; frame strips 0/8/16/24/31/38, native-size upper-body crops, and
the video):

- *Person* ("A woman dancing in a sunlit studio, spinning around and
  raising both arms above her head."): both give the same studio and
  composition; the dancer turns back → front → back with both arms raised
  and hair flying in both. Differences are wardrobe colour (grey vs beige
  trousers) and pose details. Faces are motion-blurred mid-turn in both;
  hands are small with little finger detail in both. Nothing seen that
  is worse in direct. Mean |ΔF| 6.12 (A) / 5.11 (direct).
- *Fast object* ("A red rubber ball bouncing quickly across a wooden floor
  and hitting a wall."): both show the ball travelling with a following
  camera to a wall corner; neither shows a clear bounce, so "bouncing"
  adherence is weak in both alike. Mean |ΔF| 6.87 / 8.23.

One seed per prompt; no difference that would block an opt-in fast mode.

**Opt-in fast mode in the app** (2026-09-30, uncommitted at the time of
writing):

- Engine: `h3_params.fast_attention` (appended last; default 0) and
  `h3_fast_attention_available()` (ccv backend linked in *and* the GPU has
  neural matrix accelerators). `h3_generate` resolves the attention path
  once per generation and sets it on the DiT (`h3_dit_set_attention_mode`,
  also on a reused prepared DiT); `run_block` no longer reads the
  environment. `H3_ATTENTION_BACKEND` / `H3_CCV_DIRECT` still work, only as
  a diagnostic override when `fast_attention` is 0. Requesting fast mode
  where it isn't available fails with an explicit error, never a silent
  fallback. `h3_result` reports `ccv_attention_calls` /
  `ccv_attention_direct_calls` — the path that actually ran.
- H3Engine: one `autoreleasepool` per generation inside the background
  closure (call + result handling, left on every exit path);
  `H3GenerationParams.fastAttention`; `H3Engine.fastAttentionAvailable`;
  an `NSLog` line per generation with the requested/actual path and the
  device allocation after the pool drains.
- H3cApp: "高速モード（試験的）" checkbox under the speed picker, shown only
  when available, off by default, separate from the speed presets (none of
  them turns it on), with a note that the same seed gives a different video
  and that combinations with the speed presets are unverified. The value is
  copied into the params at start, so toggling during a run has no effect.
  History shows requested/used; timing calibration keeps a separate
  history for fast mode; HTTP API takes `"fast_attention": true|false`
  (default false; 400 if unavailable).
- Checks: CLI with the parameter — `fast_attention=1` → 1000/1000 direct
  calls, video md5-identical to the earlier env-selected direct run;
  `fast_attention=0` → 0 ccv calls, md5-identical to the earlier standard
  run. Build without `CCV_DIR`: available = false, and a fast request
  errors. **Real app, one process, via the HTTP API, standard → fast →
  standard** (512x512, 56 frames, 20 steps, seed 7): logged paths standard
  (0 ccv) / fast (1000/1000 direct) / standard (0 ccv); runs 1 and 3
  md5-identical (nothing carried over from the fast run); device
  allocation after each generation 0.002 GiB. Times 238 / 244 / 242 s —
  at this short length fast mode gives no gain (consistent with the
  earlier short-clip measurements), so the UI note says the gain grows
  with clip length.
- Follow-ups: UI note now says short clips gain little and can be slower
  under some conditions (one short measurement, not "always slower");
  while generating it also says changes apply from the next generation
  (the other settings behave the same way). History and timing
  calibration are keyed on the path that actually ran
  (`ccv_attention_calls`), since the diagnostic environment override can
  route through ccv with the checkbox off (documented in
  `tools/ccv_eval/README.md`).
- UI checked on screen (window-only captures; a capture-only build moved
  the speed section to the top of the form, since the real one sits below
  the fold under 詳細設定 and scroll events could not be sent): checkbox
  visible with its full note wrapped, off initially; during a run (fast
  mode on via the API, speed preset left at 標準) the "next generation"
  note and the draft summary's 高速モード（試験的） show, and the log
  reports 150/150 direct calls, 0.002 GiB after. Speed-preset independence
  is by construction (nothing in `speedMode` touches `fastAttention`), not
  by clicking. **Scope of this UI check**: the controls' rendering only —
  reaching them in the real layout (open 詳細設定, scroll) and switching
  presets by hand are not yet checked; do that on the next manual use of a
  normal build.
- Call count vs steps: the API asked for 2 steps, but the app clamps
  steps to `stepsRange = 3 ... 20` (`GenerationViewModel.swift`) before
  building the params, so the engine ran 3 steps — 3 × 50 blocks = 150,
  matching the counter (the 20-step runs: 20 × 50 = 1000). Pre-existing
  and unrelated to fast mode: the API accepts steps below the range
  without an error and the draft summary shows the unclamped "Steps 2".

**Fast mode combined with the other speed-ups** (short, 20 steps, seed
7, CLI): the engine and app do not forbid any combination, and both tried
ran through the direct path end to end — reuse 2 (the app's new default)
550/550 direct calls (11 model evaluations × 50 blocks), and the 最速
preset (45 layers, core reuse 4, token reduction) 270/270 (6 core
evaluations × 45 blocks; token reduction's shorter row count still took
the direct path). All four clips coherent, no breakage seen; composition
of each fast clip close to its standard counterpart. DiT time at this
short length: 103.5 / 105.0 s (reuse 2) and 35.4 / 36.9 s (最速) —
no gain, as expected when the other settings already cut the attention
work; gains for long clips in combination are not measured.

**App defaults changed**: denoise reuse now defaults to 2 (also the HTTP
API's default); compute mode was already int8 attention cache on
tensor-capable GPUs and SSD streaming otherwise. Selecting the 高速/最速
preset puts reuse back to 1, so the presets keep their core reuse (which
the engine won't combine with reuse > 1) and their measured 2.8x/3.2x
figures; reuse can still be raised by hand afterwards. Restoring a past
result sets the preset before its saved reuse, and the HTTP API applies
an explicit "reuse" after "speed_mode" (a preset without "reuse" gets 1).
Not runtime-tested beyond the build. Shape and length moved into a
toolbar along the bottom edge of the prompt box (menus); their
standalone sections are gone. Rendering checked on screen; opening the
menus by click not checked.

**Audio — the 45-layer preset breaks it** (2026-10-01; none of the
comparisons above had looked at audio). A user's 15 s generation with a
speed preset + fast mode had loud broadband noise with tonal bands
across the spectrum. Isolated at 5 s (512x512, 124 frames, 20 steps,
seed 7, same prompt), judged from spectrograms and RMS / spectral
flatness:

| setting | audio |
|---|---|
| standard | quiet noise, -49 dB |
| 45 layers only | broadband noise + tonal bands, -24 dB (the broken pattern) |
| core reuse 4 only | like standard, -49 dB |
| core reuse 4 + token reduction | like standard, -53 dB |
| denoiser reuse 2 only | like standard, -44 dB |
| core reuse 4 + fast attention | like standard, -52 dB |
| 高速 / 最速 presets (45 layers) | broken pattern, -33 / -27 dB; with fast attention the same pattern 6–8 dB louder |

So, under these conditions (5 s, seed 7, this prompt), the reduction to
45 layers alone reproduced the problem — the gate-ranked removal of 5
DiT blocks, earlier judged on video only. It occurs without fast
attention; adding fast attention changed the broken audio by +6–8 dB, so
"fast attention has no effect on audio" is not shown either. The ranking
(`h3_dit_schedule_gate_score`) averages the absolute AdaLN gate values
over all modalities (video and audio), so it is not video-only; but a
small average gate does not prove a block is unneeded for audio. What can
be said: gate-based ranking may not capture the effect on audio of
dropping blocks.
Fix in the app: the presets keep all 50 blocks; 高速 = core reuse 4
(DiT-phase time 204 s vs 557 s standard, ~2.7x — DiT only, not the whole
generation), 最速 = + token reduction (189 s, ~2.9x DiT). Also measured
there: core reuse 4 + fast attention took 244 s DiT, ~20% slower than
core reuse alone — so preset and fast-mode gains are not to be multiplied
and stay independent choices. "Same as standard" audio here only means no
added loud noise; whether wanted sounds (calls, speech) survive needs a
prompt with explicit sound, listened to. The engine's own
h3.h note "45 is the validated fast setting" was video-only. Audio of the
standard path itself varies by seed (near-silent noise or clipped loud
audio) — a separate, pre-existing matter.

**Check in the packaged .app after the fix** (2026-10-01; prompt with
explicit sound: "A golden retriever sits on a wooden porch and barks
loudly three times, while a gentle acoustic guitar melody plays in the
background.", 5 s, seed 7, via the HTTP API). H3Engine now logs the
parameters actually handed to the engine:
standard → `dit_layers 50, denoise_reuse 1, core_reuse 1,
token_reduction 0`; 最速 (no "reuse" in the request) →
`dit_layers 50, denoise_reuse 1, core_reuse 4, token_reduction 1` — the
preset reset reuse to 1 and kept 50 layers. Whole-generation time 599 s
vs 167 s. Spectrograms: both have broadband bark-like transients with
tonal (harmonic) lines between them and no broadband-noise/tonal-band
pattern; 4 barks in standard, 5 in 最速 (neither exactly the 3 asked);
RMS -10.2 / -10.7 dB, ~2.8–2.9% clipped samples in both (the loud
barks; clipping also occurs on the standard path). The user confirmed by
ear that the fixed audio is fine, and that prompt input works in the
packaged .app (the earlier "can't type" came from launching the bare
executable). The 599 s vs 167 s is against the reuse 1 reference run, not
a speed-up over the app's default standard setting (reuse 2).
Denoiser reuse 2 stays the app default as the user's explicit choice.

**Layer count as its own setting; presets don't overwrite reuse**
(2026-10-01): 詳細設定 gains 使用する層数 (35–50, default 50, red warning
below 50 that 45 layers broke the audio); it applies independently of the
speed presets, which no longer touch it. Selecting 高速/最速 no longer
rewrites the stored reuse: the engine gets `effectiveDenoiseReuse` (1
under a preset, the stored value under 標準), the reuse picker is
disabled with a note while a preset is active, and going back to 標準
uses the stored value (default 2) again. History records the form value,
the effective reuse and the layer count; timing calibration also keys on
a non-50 layer count; the HTTP API takes `"dit_layers"` (400 outside
35–50) and stores `"reuse"` as given. Checked in the packaged .app via
the API (logged engine params): 最速 → layers 50, reuse 1, core reuse 2
(10 steps), token reduction 1; 標準 without reuse → layers 50, reuse 2;
`dit_layers 45` → 45; `dit_layers 30` → 400. Not click-tested.

| app setting | layers | effective reuse | core reuse | token reduction |
|---|---:|---:|---:|---|
| 標準 (default) | 50 | 2 | 1 | no |
| 高速 | 50 | 1 | 4 (steps/5, max 4) | no |
| 最速 | 50 | 1 | 4 (steps/5, max 4) | yes |

**API rejects out-of-range values instead of clamping** (2026-10-02,
aeb3502; steps in 28043d5): POST /api/generate used to clamp or rewrite
several fields silently — `"steps": 2` ran 3 steps (150 ccv calls),
`"reuse": true` ran reuse 1, and `"seed": "7a"` / `-1` ran seed 7 / 1
because non-digits were stripped. It now returns 400 for `steps` outside
3–40, `reuse` outside 1–3 (the engine's limit), `seconds` outside 1–15
and `dit_layers` outside 35–50, and for JSON booleans in any of them.
`seed` accepts 0–UInt64.max as a number or a decimal string (a string
keeps large seeds intact for clients whose JSON numbers are doubles).
Checked against the running .app: every rejected value returned 400
without starting a job; valid values were checked by pairing them with a
later field that fails, so nothing generated. Then four real runs
(256×256, 1 s, 3 steps, same prompt): seed max as a number and as a
string gave identical decoded video and audio (MD5), also after an app
restart, where the engine log showed `seed 18446744073709551615`; seed
42 gave different video and audio.

**Grid: size × length × speed preset × fast mode** (2026-10-02/03, the
packaged app driven through its HTTP API, one run each, other apps
running): square, compute mode int8 attention cache, 20 steps, prompt "A
cat playing with a ball of yarn.", seed 7 (the engine log shows `seed 7`
and `ssd_streaming 0` in all 36 runs; a later run with the same request had
`~/models/cache/dit_int8_v2.cache` open during generation). Standard ran at
the default reuse 2, fast/fastest at reuse 1 with core reuse 4. Wall time
from request to finished file, fast mode off / on:

| size | length | standard | fast | fastest |
|---|---|---|---|---|
| 256×256 | 5 s | 94 / 99 s | 68 / 69 s | 58 / 58 s |
| 256×256 | 10 s | 174 / 174 s | 108 / 108 s | 88 / 88 s |
| 256×256 | 15 s | 264 / 259 s | 164 / 164 s | 128 / 128 s |
| 512×512 | 5 s | 359 / 339 s | 224 / 209 s | 164 / 158 s |
| 512×512 | 10 s | 956 / 815 s | 565 / 489 s | 374 / 339 s |
| 512×512 | 15 s | 1808 / 1467 s | 1036 / 851 s | 665 / 565 s |

- Fast mode took the direct path in every "on" run (550/550 ccv calls at
  reuse 2 = 11 model evaluations × 50 blocks; 300/300 with core reuse 4 =
  6 × 50) and none in the "off" runs. It gives nothing at 256×256 (-5% at
  5 s standard, otherwise 0–2%); at 512×512 it saves 3–7% at 5 s, 9–15% at
  10 s, 15–19% at 15 s.
- **Correction**: the earlier single run "core reuse 4 + fast attention
  244 s vs 204 s DiT" (5 s, 512×512, audio isolation above) is not borne
  out: here the fast preset with fast mode was faster in every 512×512
  case (209 vs 224 s at 5 s). The two measurements differ in build and
  run conditions and in what was timed (DiT phase vs whole run); the cause
  of the earlier slowdown is not established. The README no longer says
  the gains don't combine.
- Presets vs standard: fast 1.37–1.75x, fastest 1.60–2.72x, growing with
  size and length. Best for 512×512 15 s: fastest + fast mode, 565 s vs
  1808 s (3.2x).
- Length scaling (standard, off): 256×256 1 : 1.85 : 2.81, 512×512
  1 : 2.66 : 5.04 for 5/10/15 s.
- Swap used stayed 559–715 MiB over all 36 runs (no paging at 512×512
  15 s on 24 GB, unlike the 768×768 15 s CLI runs above).
- Output: all 36 files have the requested frame counts (124/243/362);
  middle frames all show a coherent cat and yarn (a red blob in the 512×512
  15 s fastest run was, at full size, a tangled yarn ball); compositions
  differ between settings as expected. Audio mean level -36 to -64 dB, none
  near the broken-audio levels (-24 to -33 dB); not listened to.

Videos, results.csv/json and a contact sheet:
`~/Movies/H3cApp-grid/2026-10-02/` (not in the repo).

**Same request and seed gave different videos - Fixed / verified under
supported application conditions**
(2026-10-04, `feature/projects`; 256×256, 1 s, 3 steps, int8 cache,
standard, fast mode off). While testing projects, about 1 run in 7 of a
fixed request and seed came out as a different video (~30 dB PSNR, a
different sample). Engine params as logged were identical.

Steps that did *not* explain it: H3Engine ended its event stream inside
the generation's autorelease pool, so a batch could start the next run
during the previous run's Metal teardown - fixed (the stream now ends
after the pool drains) and every generation and cache build now runs on
one process-wide serial queue, but a 50-run check afterwards still gave 6
differing seeds (FL2VA singles: 10 runs, 10 different videos). The int8
cache prefetch is correctly ordered (`h3_gpu_submit` waits, then the
reader thread is joined before the slot is reused; partial reads loop).

Located with `H3_DEBUG_HASHES=1` (new, off by default): FNV hashes of the
text embedding, condition video/audio, vision outputs, initial noise,
each Qwen layer's output, the denoised latents and every RGB chunk handed
to the writer. Vision outputs, conditions and noise always matched; the
Qwen text embedding did not, and per-op hashes in a layer (with a GPU
wait after each op) showed `h3_gqa_causal_bf16` returning different output
for byte-identical Q/K/V. Cause: a missing `threadgroup_barrier` - all
threads read the max from `reductions[0]`, then reuse `reductions` for
the sum with no barrier in between, so a fast thread 0 could overwrite
the max with its partial sum before slower threads read it. Longer
sequences (FL2VA/Ref2VA add vision tokens) and GPU load made it more
likely. `h3_vae_encoder_group_norm_silu_f32` had the same mean/variance
hazard; both now have the barrier, and a scan of every threadgroup array
for "broadcast read, then write without a barrier" found no others.

After the fix: FL2VA singles 10/10 identical (were 10/10 different); two
hashed T2V batches of 20 seeds: denoised latents 20/20 and pre-encode RGB
frames 20/20 identical. The MP4s still differed for 2 of 20 seeds at
57-58 dB PSNR - the hardware H.264 encoder (AVFoundation/VideoToolbox)
isn't bit-reproducible; the frames it was given were identical. So a
project video can be regenerated exactly up to that invisible encoder
difference. The earlier small cross-process difference with fast mode
(61 dB, 0.3.0 release check) fits the same encoder explanation.

Production-condition check after the fix (2026-10-04/05, the app via its
API with `H3_DEBUG_HASHES=1`, int8 attention cache, 20 steps, reuse 2, seed
7, five runs each in one process): C1 Ref2VA with one 512×512 reference
image, 512×512, 5 s (421-456 s per run) and C2 T2V 512×512, 15 s
(1959-2094 s per run). In both, every stage hash was identical across all
five runs - text embedding, condition and vision outputs (C1), noise,
denoised video/audio latents and the RGB frames handed to the encoder.
(The RGB hash now also covers the monolithic decode path, which 512×512
takes when no decoder is cached; it had only been in the streamed path.)
Status: fixed. Under the app's supported conditions (512×512, Ref2VA 5 s
and T2V 15 s, 20 steps, reuse 2, int8 cache), five generations of the same
seed matched bit for bit up to the RGB frames handed to the encoder.
768×768 is not an app resolution and is out of scope for this check.

Rules going forward: reproducibility is judged on the RGB frames handed to
the encoder (`h3: hash rgb video` under `H3_DEBUG_HASHES=1`, one hash over
all frames in order, so streamed and monolithic decoding compare alike; it
replaced the per-chunk `rgb chunk` lines on 2026-10-04; noise, text
conditioning and denoised-latent hashes locate a divergence), never on the
MP4's own hash, which is a reference value only. In Metal shaders, every
`threadgroup` array must follow write → barrier → read → barrier →
reuse/write; a "broadcast read" of one slot (`x = shared[0]`) followed by
any thread writing that array again needs a barrier in between. GPU race
detectors rarely catch this, so it is a review item for any kernel that
uses threadgroup memory.

Regression checks (added 2026-10-04, `65c8d90`):

- `h3_determinism_tests`, part of `make test` (~4 s, no weights): the causal
  GQA and the VAE-encoder GroupNorm kernels, 50 runs each on fixed random
  input; every output must equal the first bit for bit. With the two
  barriers removed it failed in 5 of 5 invocations (GQA 14-49 of 50 runs
  differed, GroupNorm 10-49 of 50 at the test's shape; a 3x64x64 GroupNorm
  shape only caught it ~1 in 150, hence 512 planes of 16x16).
- `h3_repro_check` (needs the weights; ~4 min; `make h3_repro_check`):
  Ref2VA with a generated 512x512 reference, 512x512, 25 frames (39
  delivered), 8 steps, reuse 2, int8 cache as the app sets it, generation
  cache off, 3 runs in one process. Each frame delivered through
  `on_frame` (the same buffer the encoder gets) is hashed on its own; a
  failed run, a missing frame or a short frame count also fails. Fixed
  shaders: 3/3 identical. Barriers removed: 5/5 runs differed, from the
  text-encoder hash on. Runs can override the request (`--run
  'seed=8;cache=1'`), which item 7 uses for cache on/off comparisons.

Still to do: the threadgroup review rule above for every new or changed
kernel (the kernel test only covers the two kernels that were fixed).

Ref2VA int8 cache fingerprint warning (2026-10-05). Every Ref2VA run had
logged that `~/models/cache/dit_int8_v2_ref2va.cache`'s model fingerprint
did not match - including the production check above, so it had to be
ruled out as a cause or a confound. The fingerprint
(`h3_weight_store_fingerprint`) hashes each shard's *path*, size and
mtime, not its contents. The cache's kind was Ref2VA and its stored id
`9893f6f715fe5803` equals the fingerprint of today's Ref2VA shards (same
names, sizes, mtimes to the second) under their earlier location
`~/Library/Application Support/h3c-analysis/MiniMax-H3/Ref2VA/transformer`;
the model folder was moved to `~/models` on 2026-09-26 (the FL2VA cache,
rebuilt 2026-09-28, already matched). Contents check: `build_attention_
cache` from the model the app loads, streamed through a FIFO into `cmp`
against the old file (no free space for a second 19.3 GB copy): only the 8
id bytes (offsets 28-35) differed, so every quantized weight was identical
and earlier Ref2VA results used the right weights. The cache was then
rebuilt in place (payload SHA-256 `8b3d8945...29a5b1` before and after,
id now `20c5298b23bef972` = the current fingerprint); the id was not
patched by hand.

Re-check after that, clean build of `de4d6a5` (shader SHA-256
`3329182726ff...e234`, the app bundle's copy identical; H3cApp
`26999aa176a9...8c86`; h3_repro_check `0e1bffd40a8c...998a`; ccv-mfa
`6a611be` with the three `tools/ccv_eval` patches), with
`H3_ATTENTION_CACHE_STRICT=1` and `H3_PROFILE=1`: no fingerprint warning;
every run read 90.1 GiB from the int8 cache ("attention int8 stream");
`h3_determinism_tests` 0 of 50 differing for both kernels;
`h3_repro_check` 3/3 identical; generation cache off / conditioning miss /
conditioning hit 3/3 identical. RGB video hash `fe287b88aebec03b`, the same
as before the rebuild.

Closed: a false positive of the path-based fingerprint, not a weight
mismatch; the earlier reproducibility checks stand. Operating rule: normal
use keeps the warning; regression and release validation run with
`H3_ATTENTION_CACHE_STRICT=1`, so a stale cache stops the run.

Follow-up (small, not scheduled): moving or copying the model folder
invalidates every int8 cache's fingerprint although the weights are
unchanged, and rebuilding means 19.3 GB per cache. Start by dropping the
absolute path: identify by model kind + shard file name (relative) + size
+ mtime. Hashing the ~65 GB of shards on every load is not practical.
*Done (2026-10-05):* `h3_weight_store_fingerprint()` now hashes each
shard's file name (no directory), size and mtime, and tags bytes 8-15 of
`model_id` with `"h3fp-v2"` so a new id can never equal an old one (old
ids are zero past byte 8); the model kind stays the header's separate
`model_kind` field, checked as before. New caches from
`build_attention_cache` get the new id. `attention_cache_validate()`
accepts either the new id or the legacy path-based one
(`h3_weight_store_fingerprint_legacy()`, computed for the directory being
loaded), so both existing caches validate without a rebuild; header format
and version unchanged. STRICT still refuses anything else.
Checks: `tests/test_h3.c` `test_weight_fingerprint` (two fake shards in a
temp dir: the id survives renaming the parent folder while the legacy id
changes; a changed mtime, size or shard name changes it); both real caches
under `H3_ATTENTION_CACHE_STRICT=1 H3_PROFILE=1 h3_repro_check --runs 2`
(Ref2VA and `--ref none`): no warning, 90.1 GiB int8 stream, runs
identical; with sparse copies of the T2V cache header under a temporary
`HOME` (no disk used): a flipped id byte is refused under STRICT (warning
without it), and the new id computed independently in Python validates
under STRICT. The Python re-implementation also reproduced the on-disk
legacy id `28118324f70bb4ae` of `dit_int8_v2.cache`.
Both caches were then rebuilt in place with `build_attention_cache` from
`7900c86` (owner's go-ahead; not patched by hand). Payload SHA-256 (bytes
after the 64-byte header) unchanged: FL2VA `a4f4697ef75b...dee2d84`,
Ref2VA `8b3d8945b30d...3129a5b1`. id bytes on disk, before -> after: FL2VA
`28118324f70bb4ae` -> `ee60019d04e52c70` + tag, Ref2VA `72f9be238b29c520`
(= the `20c5298b23bef972` above, printed as a little-endian uint64) ->
`f96a035117b504d3` + tag. Re-check with `H3_ATTENTION_CACHE_STRICT=1
H3_PROFILE=1 h3_repro_check --runs 2`: no warning, 90.1 GiB int8 stream,
runs identical, video hashes the same as before the rebuild (Ref2VA
`d55429d37cc8455b`, T2V `05e986d2cf5a8673`). The legacy-id acceptance
stays for caches elsewhere. A copy that does not preserve mtime (plain
`cp` without `-p`) changes the id by design.

Not done (from the review of this issue): pass the cache path and MLP
streaming as generation parameters instead of process-wide `setenv`, and
keep ccv's attention state per GPU context instead of one global
`g_state` - both safe today only because generations are serialized.

Earlier plan (kept for the record): (1) done — Makefile relinks `h3_generate_cli` when `libccv.a`
changes; (2) done — replay above; (3) done for seed 7 — direct at 20
steps on the short clip — detail, the foreground net, temporal flicker,
pattern stability, limb shapes, prompt adherence, judged on video, not
PSNR; (4) if clean, 15s at 20 steps (existing A / ccv_dense 20-step videos
are valid references with the same model, sampler and settings). Before
app adoption, direct must also be run from H3Spike/the Swift app: the
earlier Swift-runtime/ccv problem is a separate, unexplained issue.
*All done (2026-10-04):* the direct path shipped as the app's opt-in fast
mode and runs from the Swift app (see "Opt-in fast mode in the app" and the
measured grid above).

Scratch sources, **not committed** (existed only under a session scratchpad
directory — rewrite from this description if resuming):

- `flash_attn.metal` / `flash_test.m` — v1, naive scalar kernel.
- `flash_v2.metal` / `flash_test2.m` — v2, nax-based dense 3-pass.
  `flash_v2.metal` also grew `h3_block_means_bf16` /
  `h3_block_dot_means_bf16` / `h3_linear_bf16_nax_r128_masked` /
  `h3_softmax_rows_masked_bf16` for QK^T sparsity, and the per-row_tile and
  batched-across-row_tiles gather/transpose/matmul kernel pairs for the PV
  compaction attempts.
- `flash_test3.m` — v3, QK^T-only sparsity, properly batched (the
  1.19–1.26x numbers).
- `flash_test4.m` — PV compaction per-row_tile, invalid per-head-sync
  comparison, superseded.
- `flash_test5.m` — the same per-row_tile PV compaction properly batched —
  the valid "PV sparsity loses badly" measurement, 0.66x.
- `flash_test6.m` — PV compaction batched across row_tiles too — the
  current, valid "PV sparsity still loses, less badly" measurement, 0.88x
  (dense 48.6ms / QK-only 40.9ms (1.19x) / QK+PV 55.4ms (0.88x), `kept_max`
  padding waste 1.41x).

### Alternative selector: Veda

Could plug into the same block-sparse kernel once it exists. arXiv
2605.30325, code `veda-sparse/Miowtion`, HF
`Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview` — tokens permuted
into 128-token 3D tiles, a 275M fp8 predictor picks the top-10% tile pairs
per layer/head; end-to-end 1.57x/2.21x/2.76x at 5/10/14s on RTX 4090. 12
fixed geometries, CUDA FA4 (CuTe) kernels only, no Metal path. Design the
kernel to take a block list so Sol-Attn (training-free, any mode) comes
first and Veda can be added later; the 3D tile permutation is worth
borrowing for Sol-Attn.

*Update 2026-09-28:* Miowtion's own README now says both FL2VA and Ref2VA
checkpoints are supported — the "T2VA only" framing above was accurate when
written but is stale. The CUDA-only sparse kernel is still the blocker for
this Mac app, so priority is unchanged (below Sol-Attn).

## 6. Adaptive DiT-reuse schedule

*Reframed 2026-09-28 from "port MotionCache/TeaCache" after checking both
against this codebase — see the external-review notes under item 5 above.*

`core_reuse`'s existing *fixed* schedule (`h3_dit.c` `h3_dit_reuse_schedule`)
already evaluates the DiT body only 6 times in 20 steps at `core_reuse=4`
(steps 0,4,8,12,16,19 — confirmed by calling the function directly), a
bigger reduction than MotionCache's own reported "20→14 calls, ~1.43x"
estimate — so porting MotionCache's specific cache values doesn't make
sense on top of what's already done.

What could still help: making *when* to re-evaluate adaptive (small
frame-to-frame change → reuse longer, large change → recompute sooner)
instead of the fixed interval, built on `core_reuse`'s own
residual-reuse/velocity-extrapolation machinery rather than importing
MotionCache's cache. `H3_REUSE_STEPS` env already exists for custom fixed
schedules. Not started. Medium effort.

## 7. Enable the app's own existing generation cache

*New item, found 2026-09-28 via external review, verified by `grep`.*

`h3_cache_set_enabled` (`h3.h`/`h3.c`) reuses the DiT/video-decoder/
conditioning state across `h3_generate` calls that share the same prepared
key (model, resolution, etc.) — `ctx->cache_enabled` defaults to 0 and is
never set to 1 anywhere in
`native/H3Spike/Sources/H3Engine/H3Engine.swift` or the rest of the app, so
this already-built reuse path is completely unused today.

Would speed up the interactive workflow (tweak prompt/seed, generate again
on the same model+resolution) for free, independent of Sol-Attn or any
other item here. Needs a size-aware eligibility/eviction policy, not
"always on" — a 24GB Mac can't just keep everything resident (same caveat
as the existing "resident" compute mode). Not started. Small-to-medium
effort — the reuse mechanism itself already exists and works; the work is
deciding the eviction policy.

**Status: partly adopted 2026-10-04 - conditioning only.** `h3_cache_set_
targets(ctx, H3_CACHE_*)` now enables the three retained parts separately
(`h3_cache_set_enabled(ctx, 1)` = all three, as before); H3cApp's engine
turns on `H3_CACHE_CONDITIONING` only. Each part keeps one entry and is
dropped when the next request's key differs.

Measured with `h3_repro_check` (M5, 24 GB, int8 cache, reuse 2, seed 7,
one process; "held" is the process footprint between generations, "peak"
the sampled footprint during one):

| request | cache | first run | seed-only rerun | held | peak |
|---|---|---|---|---|---|
| Ref2VA 512, 25 fr, 8 st | off | 78-80 s | 78-80 s | ~0.5 GiB | 6.0-7.4 GiB |
| | conditioning | 78 s | 63.5 s | +2.9 MB | 6.6 GiB |
| | + decoder | 78 s | 60.8 s | ~3.2 GiB | 4.6 GiB |
| | + DiT (all) | 78-81 s | 52-54 s | ~4.7 GiB | 5.0 GiB; 8.0-9.1 GiB on a miss with the others held |
| T2V 512, 25 fr, 8 st | conditioning | 66 s | 58 s | +92 KB | 6.6 GiB (off: 6.1) |
| Ref2VA 512, 5 s, 20 st | off | 400 s | 400-402 s | - | 6.0-6.5 GiB |
| | conditioning | 400 s | 388 s | +2.9 MB | 6.6 GiB |
| | all | 400 s (peak 8.9) | 376 s | ~5.8 GiB | 6.3 GiB; 8.6 GiB on a prompt change |
| T2V 512, 5 s, 20 st | all | 372 s (peak 9.1, off 6.2) | 353 s | ~5.7 GiB | 6.2 GiB |

What each part saves on a hit: conditioning skips tokenizer, reference
VAE encoder, Qwen vision and the text encoder (Ref2VA ~14.5 s, T2V ~8 s);
the DiT skips the AdaLN precompute and core load (~8.5 s); the decoder
skips its load (~3 s). A prompt change misses conditioning and the DiT
(the DiT key contains the prompt), so only the decoder hits there (~3-4 s).
Footprint peaks vary by about ±0.7 GiB between identical runs; the
held/peak increases above for DiT and decoder are larger than that.

Correctness (every comparison bit-identical on the RGB handed to the
encoder): off vs on (on switches decoding from monolithic to streamed);
seed 7 -> 8 -> 7 with every part hit; prompt A -> B -> A; and after each of
fast mode on, denoise reuse 1, steps 6, 45 layers, T2V, 256x256 the cached
result equalled a fresh cache-off run, and returning to the first request
reproduced it. Fast mode and denoise reuse are not in the DiT key, but they
are applied per run to a reused DiT, so that is correct.

Not adopted: the DiT and decoder. They add ~13 s on a 400 s generation
but hold ~5.8 GiB between generations and raise the first/prompt-change
peak by ~2.5-2.9 GiB, which on this 24 GB Mac is the difference between
running clean and swapping on longer clips. Before enabling the DiT in
any form its key must also cover the int8 attention-cache file and
`H3_INT8_STREAM_MLP` (read from the environment at preparation, so the
app's attentionCache <-> resident switch would reuse a stale DiT today),
and LoRA file contents (only path and strength are in the key now). Not
measured: 15 s clips with the cache on, and reference video/audio inputs.

In the packaged app (T2V, 512x512, 1 s, 8 steps, a 2-item batch with seeds
7 and 8 into a project): item 1 logged a conditioning miss and took 66.5 s,
item 2 a hit and 57.8 s - the same as the CLI measurement above. Batches
(seeds count up from a fixed seed) and seed-only reruns get this for free.

**Token refiner output and AdaLN schedule, for batches (2026-10-05).** Two
more targets that keep only seed-independent preparation, not the DiT:
`H3_CACHE_REFINED_TEXT` (the token refiner's BF16 output plus a copy of
its input embedding: 5.5 MiB for the Ref2VA test prompt, 0.2 MiB T2V; key:
model, exact input embedding, row count, each LoRA's path/strength/file
size+mtime - LoRA rounding is seeded per block, not by the generation
seed) and `H3_CACHE_ADALN` (the unpruned AdaLN schedule as host BF16: 111 /
148 / 370 MiB at 6 / 8 / 20 steps; key: model, exact sigma schedule, which
conditions are present - not the prompt, seed or LoRA, since `adaln_proj`
is not LoRA-patched). Both are uploaded into each new DiT's own GPU
context; layer pruning runs on the uploaded copy. Noise, condition
augmentation and all step state are still made per run. H3cApp keeps the
refiner output always (with the conditioning) and the AdaLN schedule only
from the first to the last item of a batch.

Correctness (`h3_repro_check`, now also comparing the decoded audio PCM
via `h3: hash audio waveform`, `H3_ATTENTION_CACHE_STRICT=1`): cache off vs
on, seed 7 -> 8 -> 7, and after a prompt change (refiner recomputed, AdaLN
reused), a step change (AdaLN recomputed, refiner reused), 45 layers
(pruning the reused schedule), T2V and back - every cached result equal to
a fresh cache-off run in RGB and audio.

Time: the two phases cost 0.7 s (refine text) + 7.8-8.4 s (precompute
AdaLN, reading ~25 GB of `adaln_proj` weights) per item; on a hit both drop
to ~0.1 s. Short Ref2VA (512, 25 frames, 8 steps): batch item 63.5 s with
conditioning only -> 55.7-56.6 s. Ref2VA 512, 5 s, 20 steps, one process:
conditioning-only item 391.8 s vs all hit 390.7 s - the 8.5 s saved in
preparation was hidden by the denoise loop drifting upward over the session
(341 -> 343 -> 346 -> 351 -> 348 s, the last with the cache off), so at
this length the gain is ~2% and within run-to-run noise. In the app (same
request, `count` 3): 404.2 / 384.2 / 385.5 s per item.

---

## 8. Custom sizes - done in 0.4.1 (X2 VAE not adopted)

Shipped in v0.4.1: presets per shape at 小/中/大, each optionally upscaled
2x on the host, and a custom finished size in the advanced settings with a
time estimate. The X2 VAE was evaluated and dropped. Measurements and the
decisions behind both follow.

### What shipped

| shape | 小 | 中 | 大 |
|---|---|---|---|
| square | 256 | 512 | 768 |
| landscape | 448x256 | 672x384 | 960x544 |
| portrait | 256x448 | 384x672 | 544x960 |

- **2x upscale** of any preset (up to 1536x1536 and 1920x1088/1088x1920)
  uses `h3_resize_rgb24_high_quality` (Accelerate `vImageScale_ARGB8888`,
  high-quality resampling, Lanczos-class) per decoded chunk. It costs
  ~0.1 s for a 5 s clip; 1 s at 20 steps took 204.8 s at 1920x1088 vs
  206.6 s at 960x544 without it. No license or attribution is needed
  (system framework).
- **Engine limit**: the 768*1344 pixel cap now applies to the canvas the
  DiT runs at (the render canvas when set); an upscaled output may reach
  4x that. Outputs over H.264 level 4.1's 8192 macroblocks (1536x1536) let
  the encoder pick the level (5.0 there).
- **Custom size**: width/height are the *finished* size (multiples of 32,
  or 64 with the 2x upscale, which generates at half). The generated canvas
  must stay within 600,000 pixels and 256-1344 per side; over it, the
  advanced settings show the error and generation is blocked.
- **Estimate**: the advanced settings show the whole generation predicted
  by the progress estimator from this Mac's past runs (TimingCalibration),
  flagging over an hour.
- **Small sizes keep the released RoPE**: the half-scale spatial RoPE that
  `h3.c` applies only at 256x256 was tried at 448x256 and 256x448
  (temporary `H3_EXP_ROPE_SCALE`, not committed; T2V, 39 frames, 20 steps,
  seed 7; cat, night market, snowy mountains; 2026-10-07,
  `~/Movies/H3cApp-grid/2026-10-07-small/`). Neither setting broke a
  frame, but 0.5 framed every scene wider - smaller subjects, more distant
  detail, the woman in the market seen from behind at the edge instead of
  walking toward the camera - which suits a tiny canvas worse. Two frames
  per video compared; a composition difference, not a measured quality
  win either way, so the default stays.
- **Frame grid**: the video VAE packs 17 frames into 5 latent frames, so
  lengths snap up to 5 + 17k frames (`h3_align_frame_count`). The app's
  "1 s" (24 frames) runs 39 frames = 1.63 s and "5 s" 124 frames = 5.17 s;
  the "1 s" rows below are those 39 frames.

### X2 VAE - evaluated, not adopted

Community MiniMax-H3-X2-Detail-VAE (MiniMax H3 Community License): a
fine-tuned decoder whose `proj_out` emits 12 packed channels per pixel
(PixelShuffle x2) instead of the released 3072x2048 `proj_out`; only
`decoder.norm_out` and `proj_out` differ, the 36 transformer blocks are
identical. A working prototype (packed channel = c*4 + 2i + j, stats by
colour channel, scaled tiling/stitching) decoded 960x544 to 1920x1088 for
about 5% extra decode time (~4.6 s on a 5 s clip vs ~0.1 s for vImage).
It measured ~5x the high-frequency energy of the Lanczos upscale but also
a stronger 32-px grid (which its authors document; they also report the
real-detail recovery failed), and side by side on three scenes at 20
steps the user saw no visible difference from Lanczos. Dropped: no gain
for the license terms, weight download (5.2 GB) and decoder changes.

### Measurements

960x544 (multiples of 32, 522k pixels, within the 768x1344 limit; 510 DiT
tokens per latent frame vs 256 at 512x512), M5 24 GB, `h3_repro_check`
(T2V, int8 cache, seed 7):

| run | per evaluated step | video VAE | peak footprint | total |
|---|---|---|---|---|
| 512x512, 1 s, 2 steps, reuse 1 | 6.7 s | 13.3 s | 6.2 GiB | 44.6 s |
| 960x544, 1 s, 2 steps, reuse 1 | 14.9 s | 26.4 s | 7.2 GiB | 73.9 s |
| 512x512, 5 s, 2 steps, reuse 1 | 27.2 s | 38.4 s | 6.5 GiB | 110.7 s |
| 960x544, 5 s, 2 steps, reuse 1 | 79.9 s | 84.1 s | 7.1 GiB | 262.1 s |
| 960x544, 5 s, 20 steps, reuse 2 | (denoise 822 s) | 83 s | 6.3 GiB | 15:23 |
| 960x544, 10 s, 20 steps, reuse 2 | (denoise 2664 s) | 163 s | 7.0 GiB | 47:26 |
| 960x544, 15 s, 20 steps, reuse 2 | (denoise 5822 s) | 241 s | 9.7 GiB | 1:41:23 |

Swap stayed at 1100 MiB through all of them (minimum free memory 32%), so
960x544 fits up to 15 s; the denoise grows 3.2x / 7.1x for 2x / 3x the
length, the decode about linearly. The earlier "768x768 15 s swaps
heavily" figure (Stage 7) came from an older build in the ccv evaluation
and overstated the risk here. Mid frames of all three videos looked
normal (one frame each checked; `~/Movies/H3cApp-grid/2026-10-05-960x544/`).
`h3_repro_check` gained `width`/`height` keys and `H3_REPRO_KEEP_DIR`
(keep each run's video) for these runs.

768x768 vs 960x544 at the app's standard settings (T2V, 1 s = 25 frames
requested / 39 output, 20 steps, reuse 2, 50 layers, int8 cache,
`H3_ATTENTION_CACHE_STRICT=1`, seed 7, one process, 2026-10-06):

| size | tokens per latent frame | denoise | video VAE | total | peak footprint |
|---|---|---|---|---|---|
| 512x512 | 256 | 76.6 s | 13.5 s | 107.4 s | 6.78 GiB |
| 960x544 | 510 | 162.9 s | 26.4 s | 206.6 s | 7.16 GiB |
| 768x768 | 576 | 188.2 s | 28.7 s | 234.3 s | 6.80 GiB |

768x768 has 13% more tokens than 960x544 and took 1.16x the denoise
(1.13x total) - about the token ratio, well short of attention's
quadratic 1.27x. 512x512 to 768x768 is 2.25x the tokens for 2.46x the
denoise. At these sizes the time is close to linear in tokens, so a
custom-size estimate can scale from tokens; peak memory barely moves.

## Lower priority, noted but not planned

MATLOWAI's fused model approximates Ref2VA as FL2VA + rank-1024 SVD delta so
one ~21 GB model/cache serves both modes (disk win, not speed).
FastVideo-FastH3-8-Step-V2 is a full diffusers-layout checkpoint (flat, not
FL2VA/Ref2VA subdirs) — would need a loader check before the Model Manager
could register it.

## Methodology note

Every approximate technique needs a fixed-seed quality comparison first:
the H3Spike CLI's `H3SPIKE_DUMP=<path>` writes raw RGB24 frames for PSNR
checks.
