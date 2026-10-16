/* Public API for the h3-metal MiniMax-H3 inference engine. */
#ifndef H3_H
#define H3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_VERSION "0.1.0-dev"
#define H3_DEFAULT_WIDTH 864
#define H3_DEFAULT_HEIGHT 480
#define H3_DEFAULT_FRAMES 56
#define H3_DEFAULT_STEPS 20
#define H3_DEFAULT_DIT_LAYERS 50
#define H3_MIN_DIT_LAYERS 35

typedef struct h3_ctx h3_ctx;
typedef struct h3_result h3_result;

typedef struct {
    size_t embedding_entries;
    size_t embedding_bytes;
    int prepared_dit;
    int video_decoder;
    size_t refined_text_bytes;
    size_t adaln_bytes;
} h3_cache_info;

typedef enum {
    H3_REFERENCE_IMAGE = 1,
    H3_REFERENCE_VIDEO = 2,
    H3_REFERENCE_AUDIO = 3,
    H3_REFERENCE_VIDEO_AUDIO = 4
} h3_reference_kind;

typedef struct {
    h3_reference_kind kind;
    const char *path;
    const char *audio_path;
    int include_embedded_audio;
} h3_reference;

typedef enum {
    H3_REFERENCE_IMAGE_MATCH = 0,
    H3_REFERENCE_IMAGE_MAX = 1
} h3_reference_image_size;

/* One LoRA adapter to apply to the DiT. Several stack additively. The file
 * may use any of the published H3 layouts: diffusers/PEFT (to_q/to_k/to_v,
 * ff.net.*), ComfyUI (diffusion_model.blocks.N.attn.qkv_proj...),
 * unprefixed native names, or kohya (lora_unet_blocks_N_...), with
 * lora_A/lora_B or lora_down/lora_up factors in BF16, F16 or F32. */
typedef struct {
    const char *path;
    /* Multiplies the adapter's own alpha/rank scale (1 = as trained). */
    float strength;
} h3_lora;

/* What h3_lora_inspect found in an adapter file. Tensors outside the DiT
 * block and token-refiner projections (e.g. adaln_proj, final_layer,
 * proj_out) are counted in `unsupported` and ignored. */
typedef struct {
    int blocks;          /* DiT blocks (of 50) with at least one projection */
    int refiner_blocks;  /* token-refiner blocks (of 2) touched */
    int projections;     /* block projections matched in total */
    int unsupported;     /* tensors that will not be applied */
    int rank_min;
    int rank_max;
    char format[16];     /* "diffusers", "comfyui", "native" or "kohya" */
    char base_model[96]; /* from metadata when the trainer recorded it */
    /* The adapter's recommended sampling, when its metadata records one in
     * the DMAD style ("sampling": "4 steps, ..., audio shift 2, re-noise
     * step rule, ..."): sampler is H3_SAMPLER_RENOISE when it names the
     * re-noise rule, audio_sigma_shift is 0 when not given. */
    int sampler;
    float audio_sigma_shift;
} h3_lora_info;

typedef struct {
    int width;
    int height;
    int stride;
    const uint8_t *rgb;
    int frame_index;
    int frame_count;
    /* Non-negative only for an intermediate denoising preview. */
    int denoise_step;
    int denoise_steps;
} h3_frame;

typedef int (*h3_frame_callback)(const h3_frame *frame, void *opaque);
typedef int (*h3_progress_callback)(const char *phase, int completed, int total,
                                    void *opaque);

typedef struct {
    int width;
    int height;
    int frames;
    int steps;
    uint64_t seed;
    const char *output_path;
    const char *first_frame;
    const char *last_frame;
    const h3_reference *references;
    size_t reference_count;
    h3_reference_image_size reference_image_size;
    /* Evaluate one of every N denoiser steps. 1 is the close-reference path,
     * 2 is the validated fast path, and 3 is the aggressive fast path. */
    int denoise_reuse;
    /* Number of gate-ranked DiT residual blocks to retain. 50 is exact,
     * 45 is the validated fast setting, and 40 is more aggressive. */
    int dit_layers;
    /* Recompute the transformer core every N denoiser steps while refreshing
     * the timestep head each step. 1 is exact, 4 fast, and 6 aggressive. */
    int core_reuse;
    /* Pair adjacent horizontal video tokens through middle DiT blocks while
     * preserving their full-resolution residual. Early noisy evaluations use
     * a deeper reduced interval. This is a validated aggressive speed mode. */
    int token_reduction;
    /* Use one int8 activation scale per FC2 row and the M5 full-K kernel.
     * Faster, but more numerically aggressive than grouped int8. */
    int use_int8_row_fc2;
    /* Restore the released spatial RoPE grid at 256x256. The default applies
     * a visually validated half-scale grid only at that native canvas. */
    int use_reference_rope;
    /* Keep only two original BF16 DiT blocks in memory and overlap reading the
     * next block from the checkpoint with execution of the current block. */
    int ssd_streaming;
    /* Optional lower internal model canvas. Both must be zero (exact output
     * canvas) or valid same-aspect dimensions no larger than width/height.
     * The 768*1344 pixel limit applies to this canvas when set; the output
     * may then reach four times that limit. */
    int render_width;
    int render_height;
    /* Force the portable close-reference BF16/MPS MLP implementation instead
     * of the fastest validated native MLP supported by the current GPU. */
    int use_slower_bf16_mlp;
    /* Force the portable close-reference BF16 QKV projection. */
    int use_slower_bf16_qkv;
    /* Force the portable BF16 attention-output projection. */
    int use_slower_bf16_attention_output;
    /* Materialize row-major BF16 after SDPA before int8 quantization. */
    int use_slower_row_major_attention_output;
    /* Keep int8 projection-input quantization as standalone kernels. */
    int use_slower_unfused_int8_inputs;
    /* Keep Q/K norm and RoPE as a separate kernel after int8 QKV. */
    int use_slower_unfused_qkv_rope;
    /* Force scalar BF16 loads in the fused Q/K RMS reducer. */
    int use_slower_scalar_qkv_rms;
    /* Reread int8 dequantization scales from device memory per output. */
    int use_slower_uncached_int8_scales;
    /* Use the generic runtime-bound FC1 TensorOps K loop. */
    int use_slower_dynamic_fc1_k;
    /* Force the original 256-thread FC2 grouped activation quantizer. */
    int use_slower_grouped_quantizer;
    /* Decode and deliver one representative frame after every Euler step. */
    int preview_denoise;
    h3_frame_callback on_frame;
    h3_progress_callback on_progress;
    void *callback_opaque;
    /* LoRA adapters, applied in every compute mode (resident, int8
     * attention cache, SSD streaming) by patching each block's weights on
     * the GPU as they are loaded or streamed - no fused cache files. */
    const h3_lora *loras;
    size_t lora_count;
    /* Opt-in fast mode (experimental): run the DiT's full attention with
     * ccv's int8 kernel directly on the BF16 head-major Q/K/V. Faster on
     * M5-class GPUs; the same seed gives a different video than the
     * default. Requires h3_fast_attention_available(); fixed for the whole
     * generation. Independent of the other speed settings above. */
    int fast_attention;
    /* Denoising step rule: H3_SAMPLER_EULER (0, the official pipeline's
     * Euler ODE step) or H3_SAMPLER_RENOISE, which predicts the clean latent
     * at every step and adds fresh seeded noise at the next sigma - the
     * DMD-style rule few-step students such as DMAD are trained with.
     * Re-noise evaluates every step, so it requires denoise_reuse 1. */
    int sampler;
    /* Audio sigma-schedule shift. 0 keeps the default (3.0); some few-step
     * adapters were trained with another value (DMAD uses 2.0). */
    float audio_sigma_shift;
    /* Forced audio (lip sync): when set, the audio of this file from
     * forced_audio_offset seconds on, as long as the video, is encoded and
     * given to the DiT as clean audio (audio sigma 0) at every denoising
     * step, so only the picture is generated - to match what it hears. The
     * output carries that audio itself. Any format AVFoundation reads;
     * NULL is off. */
    const char *forced_audio_path;
    double forced_audio_offset;
    /* Clip continuation. save_latent_path: also write this generation's
     * final latents (before decoding) to that file. continue_latent_path: a
     * file written that way by an earlier generation at the same size; the
     * last continue_frames frames of it (5, 22, 39 or 56) are pinned at the
     * start of this clip, and the last continue_audio_frames frames' worth
     * of its sound (0 = continue_frames) end where they end, so this clip
     * carries on from it. Its first continue_frames frames repeat the
     * earlier clip's last ones: join the two by cross-fading those. Works
     * in the default (FL2VA) mode without a first frame; with forced audio
     * only the picture is pinned. */
    const char *save_latent_path;
    const char *continue_latent_path;
    int continue_frames;
    int continue_audio_frames;
} h3_params;

enum {
    H3_SAMPLER_EULER = 0,
    H3_SAMPLER_RENOISE = 1
};

#define H3_PARAMS_DEFAULT { \
    H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT, H3_DEFAULT_FRAMES, H3_DEFAULT_STEPS, \
    UINT64_C(42), NULL, NULL, NULL, NULL, 0, H3_REFERENCE_IMAGE_MATCH, \
    1, H3_DEFAULT_DIT_LAYERS, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, \
    NULL, 0, 0, H3_SAMPLER_EULER, 0.0f, NULL, 0.0, NULL, NULL, 0, 0 \
}

typedef struct {
    char name[128];
    char architecture[128];
    uint64_t physical_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    int apple_gpu_family;
    int metal4;
    int unified_memory;
} h3_device_info;

typedef struct {
    uint64_t bytes;
    uint64_t tensor_bytes;
    size_t files;
    size_t tensors;
} h3_component_info;

typedef struct {
    h3_component_info text_encoder;
    h3_component_info fl2va_transformer;
    h3_component_info ref2va_transformer;
    h3_component_info video_vae;
    h3_component_info audio_vae;
} h3_model_info;

struct h3_result {
    int width;
    int height;
    int frames;
    int fps;
    int sample_rate;
    uint64_t seed;
    /* Full-attention calls this generation served by ccv, and how many of
     * those took the direct path (both 0 on the default MPS path) - the
     * attention path that actually ran, not just the one requested. */
    uint64_t ccv_attention_calls;
    uint64_t ccv_attention_direct_calls;
};

/* Load model metadata and initialize the Metal device. Weights remain unmapped. */
h3_ctx *h3_load_dir(const char *model_dir);
void h3_free(h3_ctx *ctx);

const char *h3_last_error(const h3_ctx *ctx);
const h3_device_info *h3_device(const h3_ctx *ctx);
const h3_model_info *h3_model(const h3_ctx *ctx);

/* Interactive-session reuse. Disabled by default so one-shot callers retain
 * the original phase-by-phase memory lifetime. Each target is kept for the
 * next h3_generate call whose request matches its key, one entry each:
 *   CONDITIONING  text/reference conditioning; host memory, a few MB.
 *                 Hit when only the seed (or DiT/decoder settings) changed.
 *   DIT           the prepared DiT; ~1.5 GiB more held between calls, and
 *                 its key does not cover the int8 attention-cache file or
 *                 LoRA file contents (only their environment/path).
 *   DECODER       the video VAE decoder; ~2.7 GiB held between calls.
 * h3_cache_set_enabled(ctx, 1) is all three; turning a target off frees it. */
enum {
    H3_CACHE_CONDITIONING = 1u << 0,
    H3_CACHE_DIT = 1u << 1,
    H3_CACHE_DECODER = 1u << 2,
    H3_CACHE_ALL = H3_CACHE_CONDITIONING | H3_CACHE_DIT | H3_CACHE_DECODER,
    /* Parts of preparing a DiT that don't depend on the seed, for when the
     * DiT itself is not kept (e.g. the items of a batch):
     *   REFINED_TEXT  the token refiner's output; a few MB of host memory.
     *                 Same model, embedding and LoRA files/strengths.
     *   ADALN         the AdaLN schedule, before layer pruning; time rows x
     *                 50 blocks of BF16, a few hundred MB. Same model, steps
     *                 and condition kinds - any prompt. */
    H3_CACHE_REFINED_TEXT = 1u << 3,
    H3_CACHE_ADALN = 1u << 4
};
void h3_cache_set_targets(h3_ctx *ctx, unsigned targets);
void h3_cache_set_enabled(h3_ctx *ctx, int enabled);
void h3_cache_clear(h3_ctx *ctx);
void h3_cache_get_info(const h3_ctx *ctx, h3_cache_info *info);

/* 1 when h3_params.fast_attention can be used: the engine was built with
 * the ccv backend (CCV_DIR) and this Mac's GPU has the neural matrix
 * accelerators it needs (M5 class). */
int h3_fast_attention_available(void);

/* Generate media, delivering decoded frames incrementally through on_frame. */
h3_result *h3_generate(h3_ctx *ctx, const char *prompt,
                       const h3_params *params);
void h3_result_free(h3_result *result);

/* Reads only an adapter file's header and reports which layout it uses and
 * how much of the DiT it covers. Returns 0 (with error set) for files that
 * are not readable safetensors or match no known H3 projection. */
int h3_lora_inspect(const char *path, h3_lora_info *info,
                    char *error, size_t error_size);

/* Builds a pre-quantized int8 attention cache for one DiT transformer
 * directory (a path ending in "FL2VA/transformer" or "Ref2VA/transformer"),
 * the same cache format h3_generate reads via the H3_ATTENTION_CACHE
 * environment variable. Not tied to an h3_ctx - it opens its own GPU
 * device and weight store - so it can run before or independently of
 * h3_load_dir. Needs a GPU with the int8 tensor-op path
 * (h3_device(ctx)->apple_gpu_family >= 10, i.e. M5 or newer); fails with a
 * descriptive error otherwise rather than crashing.
 *
 * on_progress (optional) is called after each DiT block with phase
 * "attention cache", completed/total = blocks done/50 (mirrors h3_params'
 * on_progress convention: return non-zero to cancel, which deletes the
 * partial output file and fails the call). Takes roughly a minute on an
 * M5. */
int h3_build_attention_cache(const char *transformer_dir,
                             const char *output_path,
                             const char *shader_source_path,
                             h3_progress_callback on_progress,
                             void *callback_opaque,
                             char *error, size_t error_size);

/* Builds a curve-form AdaLN file (see h3_adaln_curve.h) for one DiT
 * transformer directory: a ~0.6 GiB stand-in for its ~24 GiB of dense
 * AdaLN projections that h3_generate reads via the H3_ADALN_CURVE
 * environment variable. grid/rank of 0 select the defaults (4097 / 32).
 * Like h3_build_attention_cache it opens its own GPU device (only for the
 * engine's own time embedding) and weight store, but works on any Apple
 * GPU, and writes through a ".partial" file renamed on success.
 * on_progress gets phase "AdaLN curve", completed/total = projections
 * done/51; non-zero cancels. */
int h3_build_adaln_curve(const char *transformer_dir,
                         const char *output_path,
                         const char *shader_source_path,
                         unsigned grid, unsigned rank,
                         h3_progress_callback on_progress,
                         void *callback_opaque,
                         char *error, size_t error_size);

/* 1 when curve_path is a readable curve-form AdaLN file built from the
 * transformer in transformer_dir (its weight fingerprint matches), so
 * H3_ADALN_CURVE can point at it; 0 otherwise (missing, unreadable, or
 * built from other weights - rebuild it then). Reads only headers. */
int h3_adaln_curve_matches(const char *curve_path, const char *transformer_dir);

/* TEMPORARY diagnostic, not for production use: creates a standalone
 * h3_gpu (no model needed) and makes one throwaway call through the
 * experimental H3_ATTENTION_BACKEND=ccv_dense path, to isolate whether a
 * failure seen from a real generation reproduces purely from being called
 * inside a Swift process (vs a plain C/ObjC++ test binary). Returns 1 on
 * success, 0 on failure (message left in `error`). */
int h3_debug_ccv_warmup(const char *shader_source_path, char *error,
                        size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
