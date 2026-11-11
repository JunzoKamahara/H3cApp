/* What the files of the DiT share among themselves (h3_dit.c and h3_dit_*.c): the model's state, its dimensions and file-scope helpers. Not an API; h3_dit.h is. */
#ifndef H3_DIT_PRIVATE_H
#define H3_DIT_PRIVATE_H

#include "h3_options.h"
#include "h3_dit.h"
#include "h3_adaln_curve.h"
#include "h3_dit_schedule.h"
#include "h3_hyperflow.h"
#include "h3_model_family.h"
#include "h3_lora.h"
#include "h3_quant.h"
#include "h3_weights.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    VIDEO_CHANNELS = 24,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    AUDIO_STREAMS = 2,
    ROPE_FREQS = 16,
    ROPE_HALF = 48,
    SLOTS = 6,
    FINAL_SLOTS = 2
};

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *qkv_int8;
    h3_gpu_tensor *qkv_scales;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *out_int8;
    h3_gpu_tensor *out_scales;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
    h3_gpu_tensor *fc1_int8;
    h3_gpu_tensor *fc1_scales;
    h3_gpu_tensor *fc2_int8;
    h3_gpu_tensor *fc2_scales;
    /* HyperFlow factored linears (h3_hyperflow.h), y = up(down(x)); NULL
     * where a linear is dense. FC1's up-projection is kept as its SwiGLU
     * gate and up halves, and a dense FC1 next to a factored FC2 is split
     * the same way (fc1_gate/fc1_value), so the unfused MLP never needs a
     * [sequence, 2 * FFN] buffer. */
    h3_gpu_tensor *q_down;
    h3_gpu_tensor *q_up;
    h3_gpu_tensor *k_down;
    h3_gpu_tensor *k_up;
    h3_gpu_tensor *v_down;
    h3_gpu_tensor *v_up;
    h3_gpu_tensor *out_down;
    h3_gpu_tensor *out_up;
    h3_gpu_tensor *fc1_down;
    h3_gpu_tensor *fc1_gate_up;
    h3_gpu_tensor *fc1_value_up;
    h3_gpu_tensor *fc1_gate;
    h3_gpu_tensor *fc1_value;
    h3_gpu_tensor *fc2_down;
    h3_gpu_tensor *fc2_up;
    uint32_t rank[H3_HF_LINEARS];
} h3_dit_block;

enum {
    STREAM_QKV,
    STREAM_OUT,
    STREAM_FC1,
    STREAM_FC2,
    STREAM_MATRICES,
    /* HyperFlow factored linears and the halves of a dense FC1 next to a
     * factored FC2 (h3_dit_block). */
    STREAM_Q_DOWN = STREAM_MATRICES,
    STREAM_Q_UP,
    STREAM_K_DOWN,
    STREAM_K_UP,
    STREAM_V_DOWN,
    STREAM_V_UP,
    STREAM_OUT_DOWN,
    STREAM_OUT_UP,
    STREAM_FC1_DOWN,
    STREAM_FC1_GATE_UP,
    STREAM_FC1_VALUE_UP,
    STREAM_FC1_GATE,
    STREAM_FC1_VALUE,
    STREAM_FC2_DOWN,
    STREAM_FC2_UP,
    STREAM_FIELDS
};

/* Most matrices one streamed block can have: a fully factored HyperFlow
 * block (6 attention factors, 2 output factors, FC1 down and its two up
 * halves, 2 FC2 factors). */
#define STREAM_MAX_SOURCES 13u

typedef struct {
    const char *path;
    uint64_t file_offset;
    size_t elements;
    unsigned field;
    /* h3_quant_type of a block-quantized source (a GGUF checkpoint): its
     * raw bytes stream into the slot's staging buffer and are expanded to
     * the BF16 slot on the GPU just before the block runs. 0: BF16. */
    uint32_t quant;
    uint64_t bytes;
    /* Set for a QKV weight stored as [Q; K; V] (h3_weight_qkv_reorder): the
     * expansion also restores the per-head row order. */
    uint32_t qkv_inner;
    uint32_t qkv_head_dim;
    /* quant == STREAM_COMFY_INT8 (a ComfyUI int8 weight, h3_weights.h):
     * the staging holds its int8 payload; its row scales stay resident
     * here (owned) and comfy_group is its ConvRot group (0: none). */
    h3_gpu_tensor *comfy_scales;
    uint32_t comfy_group;
    uint32_t comfy_rows;
} h3_dit_stream_source;

/* A source->quant beyond h3_quant_type's values. */
#define STREAM_COMFY_INT8 0x100u

typedef struct {
    h3_dit_stream_source sources[STREAM_MAX_SOURCES];
    unsigned count;
    /* Per slot: each source's BF16 matrix and (quantized sources) its raw
     * staging, as page-aligned windows of the slot's two arenas. */
    h3_gpu_tensor *target[2][STREAM_MAX_SOURCES];
    h3_gpu_tensor *raw[2][STREAM_MAX_SOURCES];
    /* The tensors above are borrowed from an earlier block with the same
     * matrices (every plain H3 block, every dense HyperFlow block). */
    int borrowed;
} h3_dit_stream_layer;

/* H3_ATTENTION_CACHE: an alternative to --ssd-streaming. QKV/attention-
 * output are always streamed from a pre-quantized int8 cache file (see
 * h3_build_attention_cache.c) instead of being resident. Per layer this
 * moves ~147 MiB (int8 QKV+OUT) instead of ~735 MiB (BF16 all four
 * matrices).
 *
 * By default the MLP (FC1/FC2) still stays int8-resident for every active
 * block, same as the plain resident-int8 path (good for short clips, where
 * per-sequence activations are small and the ~10.8 GiB of resident MLP is
 * cheap next to that). H3_INT8_STREAM_MLP=1 streams FC1/FC2 from the same
 * cache too (an extra ~221 MiB/layer), dropping DiT weight residency to
 * just the two double-buffer slots (~0.72 GiB) - the tradeoff a long
 * (~15s/362-frame) run needs, since activations grow with sequence length
 * and resident MLP for every block stops being the cheap part. */
typedef struct {
    h3_gpu_tensor *qkv_int8;
    h3_gpu_tensor *qkv_scales;
    h3_gpu_tensor *out_int8;
    h3_gpu_tensor *out_scales;
    h3_gpu_tensor *fc1_int8;
    h3_gpu_tensor *fc1_scales;
    h3_gpu_tensor *fc2_int8;
    h3_gpu_tensor *fc2_scales;
} h3_dit_attention_slot;

#define H3_ATTENTION_CACHE_MAGIC "H3AC"

#define H3_ATTENTION_CACHE_VERSION 3u

/* Which transformer directory (see h3.c's dit_path selection: "FL2VA/
 * transformer" vs "Ref2VA/transformer") a cache's weights were quantized
 * from. H3_CACHE_MODEL_UNKNOWN means the cache (or the checkpoint it is
 * about to serve) did not come from that standard layout, so the
 * model_kind check is skipped rather than guessed at. */
typedef enum {
    H3_CACHE_MODEL_UNKNOWN = 0,
    H3_CACHE_MODEL_FL2VA = 1,
    H3_CACHE_MODEL_REF2VA = 2,
} h3_cache_model_kind;

/* v3 adds model_kind + model_id where v2 had unused reserved[10] (40
 * bytes: 4 + 32 + 4 = 40, so the header stays byte-identical in size) -
 * a v2 file simply fails the version check below rather than being
 * misread, since those bytes used to be zero/undefined. model_id is
 * h3_weight_store_fingerprint()'s output, or all-zero if the tool that
 * wrote this cache did not have a checkpoint directory to fingerprint. */
typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t block_count;
    uint32_t hidden;
    uint32_t inner;
    uint32_t ffn;
    uint32_t model_kind;
    uint8_t model_id[32];
    uint32_t reserved[1];
} h3_attention_cache_header;

struct h3_dit {
    h3_gpu *gpu;
    h3_weight_store *weights;
    h3_dit_schedule *schedule;
    int fused_mlp;
    int nax_mlp;
    int int8_mlp;
    int int8_qkv;
    int int8_attention_out;
    int keep_bf16_qkv;
    int keep_bf16_attention_out;
    int use_slower_row_major_attention_output;
    int use_slower_unfused_int8_inputs;
    int use_slower_unfused_qkv_rope;
    int use_slower_scalar_qkv_rms;
    int use_slower_uncached_int8_scales;
    int use_slower_dynamic_fc1_k;
    int use_slower_grouped_quantizer;
    int use_int8_row_fc2;
    int ssd_streaming;
    int keep_bf16_mlp;
    int activation_aliases;
    h3_dit_attention_mode attention_mode;
    int sampler;              /* H3_SAMPLER_EULER or H3_SAMPLER_RENOISE */
    uint64_t sampler_seed;    /* drives the re-noise rule's fresh noise */
    float *forced_audio;      /* clean [32,2,T] audio latent, or NULL */
    int fused_patch_projection;
    int fused_patch_pack;
    int token_reduction;
    int token_reduction_active;
    unsigned token_reduction_begin;
    unsigned token_reduction_end;
    unsigned token_reduction_early_steps;
    unsigned token_reduction_early_end;
    float token_reduction_scale;
    float spatial_rope_scale;
    int bf16_final;
    unsigned core_reuse_interval;
    unsigned core_forward_count;
    int core_residual_ready;
    unsigned active_block_count;
    uint8_t block_active[H3_DIT_BLOCKS];
    h3_layout layout;
    h3_sigma_schedule sigmas;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    uint32_t text_rows;
    uint32_t video_condition_rows;
    uint32_t audio_condition_rows;
    uint32_t audio_rows;
    uint32_t video_rows;
    uint32_t video_total_rows;
    uint32_t audio_total_rows;
    uint32_t audio_target_start;
    uint32_t video_target_start;
    uint32_t sequence;
    uint32_t reduced_sequence;
    uint32_t reduced_video_rows;
    uint32_t token_baseline_rows;
    h3_gpu_tensor *refined_text;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *reduced_rope_cos;
    h3_gpu_tensor *reduced_rope_sin;
    h3_gpu_tensor **row_maps;
    h3_gpu_tensor **reduced_row_maps;
    h3_gpu_tensor **final_audio_maps;
    h3_gpu_tensor **final_video_maps;
    h3_gpu_tensor *video_patch_w;
    h3_gpu_tensor *video_patch_b;
    h3_gpu_tensor *audio_patch_w;
    h3_gpu_tensor *audio_patch_b;
    h3_dit_block blocks[H3_DIT_BLOCKS];
    /* Raw quantized bytes per slot and matrix (field order), allocated only
     * when some streamed source is block-quantized (stream_quantized). */
    /* SSD streaming: per slot, one BF16 arena and one raw-staging arena,
     * each as large as the largest block needs; every block's matrices are
     * aliases into them (h3_dit_stream_layer). */
    h3_gpu_tensor *stream_arena[2];
    h3_gpu_tensor *stream_raw_arena[2];
    int stream_quantized;
    h3_dit_stream_layer stream_layers[H3_DIT_BLOCKS];
    unsigned stream_ready_layer;
    unsigned stream_ready_slot;
    uint64_t stream_bytes;
    double stream_read_seconds;
    double stream_wait_seconds;
    char *attention_cache_path;
    int attention_stream;
    int mlp_stream;
    /* LoRA adapters (h3_params.loras). Resident projections are patched
     * once as they load; streamed ones (SSD BF16 slots, int8 attention
     * cache slots) every time a block is streamed in. NULL without LoRA. */
    h3_lora_set *loras;
    /* h3_params.cancel_requested, via h3_dit_set_cancel. */
    const volatile int *cancel_requested;
    /* h3_dit_set_latent_preview; evaluations counted since it was set */
    int latent_preview_every;
    int latent_preview_evaluations;
    h3_dit_latent_preview latent_preview;
    void *latent_preview_opaque;
    h3_dit_attention_slot attention_slots[2];
    unsigned attn_ready_layer;
    unsigned attn_ready_slot;
    uint64_t attn_stream_bytes;
    double attn_stream_read_seconds;
    double attn_stream_wait_seconds;
    h3_gpu_tensor *final_norm;
    h3_gpu_tensor *final_video_w;
    h3_gpu_tensor *final_video_b;
    h3_gpu_tensor *final_audio_w;
    h3_gpu_tensor *final_audio_b;
    h3_gpu_tensor *video_input;
    h3_gpu_tensor *audio_input;
    h3_gpu_tensor *video_projected_f32;
    h3_gpu_tensor *audio_projected_f32;
    h3_gpu_tensor *video_projected;
    h3_gpu_tensor *audio_projected;
    h3_gpu_tensor *video_projection_map;
    h3_gpu_tensor *audio_projection_map;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *core_input;
    h3_gpu_tensor *core_residual;
    h3_gpu_tensor *mod_attention;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_heads;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *token_pool_pairs;
    h3_gpu_tensor *token_baseline_indices;
    h3_gpu_tensor *token_expand_parents;
    h3_gpu_tensor *token_original;
    int token_original_in_qkv;
    size_t token_original_offset;
    size_t token_baseline_offset;
    h3_gpu_tensor *mod_mlp;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *int8_activation;
    h3_gpu_tensor *int8_activation_scales;
    h3_gpu_tensor *final_audio_input;
    h3_gpu_tensor *final_video_input;
    h3_gpu_tensor *final_audio_inverse;
    h3_gpu_tensor *final_video_inverse;
    h3_gpu_tensor *final_audio_norm;
    h3_gpu_tensor *final_video_norm;
    h3_gpu_tensor *final_audio_f32;
    h3_gpu_tensor *final_video_f32;
    h3_gpu_tensor *audio_output;
    h3_gpu_tensor *video_output;
    h3_gpu_tensor *audio_output_bf16;
    h3_gpu_tensor *video_output_bf16;
    h3_gpu_tensor *previous_audio_velocity;
    h3_gpu_tensor *previous_video_velocity;
    h3_gpu_tensor *forced_audio_velocity; /* F32 rows, GPU sampler only */
    /* HyperFlow H3 Flash (h3_hyperflow.h): set when the checkpoint has the
     * two-time embedder. Blocks at and after hyperflow_info.block_count do
     * not exist and stay inactive. lowrank_tmp holds a factored linear's
     * [rows, rank] middle (and a copy of the MLP input for a dense FC1 in a
     * factored MLP): [sequence, max(max rank, HIDDEN)]. */
    int hyperflow;
    /* Its row in the family table: what applies to it (never NULL once the
     * weights are open). */
    const h3_model_family_desc *family;
    int factored_tensor_ops;
    h3_hyperflow_info hyperflow_info;
    h3_gpu_tensor *lowrank_tmp;
};

typedef struct {
    h3_dit *dit;
    unsigned layer;
    unsigned slot;
    int ok;
    uint64_t bytes;
    double seconds;
    char error[512];
} h3_dit_stream_job;

typedef struct {
    h3_dit *dit;
    unsigned layer;
    unsigned slot;
    int ok;
    uint64_t bytes;
    double seconds;
    char error[512];
} h3_dit_attention_job;

/* File-scope functions and state used by more than one of the files. */
void h3_dit_fail(char *error, size_t error_size, const char *format, ...);
unsigned h3_dit_command_block_interval(const h3_dit *dit);
int h3_dit_gpu_op(h3_dit *dit, int ok, char *error, size_t error_size,
                  const char *operation);
void h3_dit_report(h3_dit_progress progress, void *opaque, const char *phase,
                   int completed, int total);
int h3_dit_stop_if_cancelled(h3_dit *dit, char *error, size_t error_size);
void h3_dit_free_tensor(h3_gpu_tensor **tensor);
h3_gpu_tensor *h3_dit_bf1(h3_dit *dit, const char *name, uint64_t width,
                          char *error, size_t error_size);
h3_gpu_tensor *h3_dit_bf2(h3_dit *dit, const char *name, uint64_t rows,
                          uint64_t columns, char *error, size_t error_size);
h3_gpu_tensor *h3_dit_bf2_patched(h3_dit *dit, const char *name,
                                  uint64_t rows, uint64_t columns,
                                  int refiner, unsigned block, int projection,
                                  char *error, size_t error_size);
h3_gpu_tensor *h3_dit_f1(h3_dit *dit, const char *name, uint64_t width,
                         char *error, size_t error_size);
h3_gpu_tensor *h3_dit_f2(h3_dit *dit, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size);
int h3_dit_copy_layout(h3_dit *dit, const h3_layout *layout,
                       char *error, size_t error_size);
int h3_dit_validate_layout(h3_dit *dit, const h3_text_embedding *text,
                           char *error, size_t error_size);
int h3_dit_configure_token_reduction(h3_dit *dit, int requested,
                                     char *error, size_t error_size);
void h3_dit_token_pool_sources(const h3_dit *dit, uint32_t reduced_row,
                               uint32_t *first, uint32_t *second);
uint32_t h3_dit_token_reduced_parent(const h3_dit *dit, uint32_t full_row);
int h3_dit_load_block(h3_dit *dit, h3_dit_block *block, const char *prefix,
                      int refiner, unsigned index, char *error,
                      size_t error_size);
int h3_dit_load_block_norms(h3_dit *dit, h3_dit_block *block,
                            const char *prefix,
                            char *error, size_t error_size);
int h3_dit_load_block_norms_and_mlp(h3_dit *dit, h3_dit_block *block,
                                    const char *prefix, unsigned index,
                                    char *error, size_t error_size);
void h3_dit_free_block(h3_dit_block *block);
int h3_dit_load_hyperflow_block(h3_dit *dit, unsigned index,
                                char *error, size_t error_size);
double h3_dit_stream_now(void);
int h3_dit_prepare_stream_layer(h3_dit *dit, unsigned layer,
                                char *error, size_t error_size);
h3_gpu_tensor **h3_dit_stream_slot_field(h3_dit_block *slot, unsigned field);
int h3_dit_allocate_stream_slots(h3_dit *dit, char *error,
                                 size_t error_size);
void h3_dit_free_stream_slots(h3_dit *dit);
int h3_dit_read_stream_layer(h3_dit_stream_job *job);
void *h3_dit_read_stream_layer_thread(void *opaque);
h3_cache_model_kind h3_dit_detect_model_kind(const char *weight_directory);
int h3_dit_attention_cache_validate(const char *path, int need_mlp,
                                    h3_cache_model_kind expected_kind,
                                    const uint8_t *expected_model_id,
                                    const uint8_t *legacy_model_id,
                                    char *error, size_t error_size);
int h3_dit_allocate_attention_slot(h3_dit *dit, h3_dit_attention_slot *slot,
                                   char *error, size_t error_size);
void h3_dit_free_attention_slot(h3_dit_attention_slot *slot);
int h3_dit_read_attention_layer(h3_dit_attention_job *job);
void *h3_dit_read_attention_layer_thread(void *opaque);
int h3_dit_quantize_block_mlp(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size);
int h3_dit_quantize_block_qkv(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size);
int h3_dit_quantize_block_attention_out(h3_dit *dit, h3_dit_block *block,
                                        char *error, size_t error_size);
unsigned h3_dit_first_active_block(const h3_dit *dit);
unsigned h3_dit_next_active_block(const h3_dit *dit, unsigned current);
int h3_dit_encode_forward(h3_dit *dit, int step, int begin, int submit,
                          int disable_command_split, char *error,
                          size_t error_size);

#endif
