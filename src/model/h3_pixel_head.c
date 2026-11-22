/* pi-H3's pixel head (h3_pixel_head.h). Follows LakonLab v0.3.0's
 * SolH3DWDHead (lakonlab/models/architectures/sol_h3), whose layer is the
 * Sol-H3-VAE reference decoder's (Apache-2.0); tools/pi_h3/pixel_head_ref.py
 * is the same computation in numpy. */
#include "h3_pixel_head.h"
#include "h3_options.h"

#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FEATURE = H3_PIXEL_HEAD_FEATURE,
    DIM = 2048,
    HEADS = 32,
    HEAD_DIM = 64,
    FFN = 8192,
    LAYERS = 8,
    FIRST_LAYER = 16,   /* of Sol's 24: odd ones shift their windows */
    WINDOW = 16,
    BLOCK = 5,          /* latent steps per native block (17 frames) */
    T_SHIFT = 2,
    REGISTERS = 4,
    GLOBALS = REGISTERS + 1,
    ROPE_HALF = 24,
    ROPE_AXIS = 8,
    PATCH = 16,
    EXPAND = 2,         /* a DiT token is EXPAND x EXPAND of the head's */
    CHANNELS = 3,
    PATCH_FRAMES = 4,
    /* Rows per pass of the linears and key rows gathered per attention
     * call. Both only size working buffers: with 4096 and 24576 the frames
     * were the same and as quick, and the head took 0.7 GiB (384 x 384, 56
     * frames) to 1.2 GiB (1344 x 768, 124 frames) more. */
    CHUNK = 512,
    ATTENTION_ROWS = 4096
};

static const float rgb_mean[CHANNELS] = {-0.03f, -0.088f, -0.188f};
static const float rgb_std[CHANNELS] = {0.458f, 0.448f, 0.45f};

typedef struct {
    uint32_t batch, q_len, kv_len;
    h3_gpu_tensor *q_index;    /* U32 [batch * q_len]: the rows that ask */
    h3_gpu_tensor *kv_index;   /* U32 [batch * kv_len]: the rows they see */
} attention_call;

typedef struct {
    attention_call *calls;
    size_t count;
} attention_layout;

typedef struct {
    h3_gpu_tensor *norm1, *qkv_w, *qkv_b, *out_w, *out_b, *scale1;
    h3_gpu_tensor *norm2, *w1, *w1_b, *w2, *w2_b, *scale2;
    /* With int8 linears each weight above is replaced by its rows in int8
     * and their F32 scales. */
    h3_gpu_tensor *qkv_q, *qkv_s, *out_q, *out_s, *w1_q, *w1_s, *w2_q, *w2_s;
} head_block;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *store;
    int latent_t, token_h, token_w;
    uint32_t T, H, W;      /* the head's token grid */
    uint32_t tokens;       /* T * H * W */
    uint32_t rows;         /* tokens, then GLOBALS per native block */
    uint32_t blocks;
    /* resident activations */
    h3_gpu_tensor *hidden;                 /* BF16 [rows, DIM] */
    h3_gpu_tensor *query, *key, *value;    /* BF16 [rows, DIM] */
    h3_gpu_tensor *rope_cos[2], *rope_sin[2];   /* F32 [rows, ROPE_HALF] */
    /* a chunk of rows */
    h3_gpu_tensor *x, *norm, *qkv, *qf, *kf, *vf, *branch_in, *branch;
    h3_gpu_tensor *ff1, *activated, *half, *cos, *sin;
    /* The layers' four linears in int8, as the video VAE decoder runs its
     * own (h3_video_vae.c): the M5's tensor units, H3_VAE_INT8=0 for F32.
     * Measured on the M5 (the head alone): 1.8 s for 4.2 s at 384 x 384 and
     * 56 frames, 17.3 s for 59.5 s at 1344 x 768 and 124; 46 dB from the F32
     * frames (11 of 255 at most); about 0.9 GiB more while the first layer
     * is set up, which is the head's peak only for small clips. */
    int int8_linears;
    h3_gpu_tensor *lin_in, *lin_out, *lin_quantized, *lin_scales;
    /* attention */
    attention_layout global, window[2];
    h3_gpu_tensor *q_rows, *k_rows, *v_rows, *a_rows;
} head_context;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(head_context *head, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "pixel head: %s: %s", operation,
         h3_gpu_error(head->gpu));
    return 0;
}

#define OP(call, label) do {                                                   \
    if (!gpu_op(head, (call), error, error_size, label)) return 0;             \
} while (0)

/* Round to nearest, ties to even, as the GPU's cast does. */
static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* A chunk of the hidden rows as F32 in head->x, and back. */
static int take_rows(head_context *head, size_t at, size_t count) {
    return h3_gpu_copy_bf16(head->gpu, head->half, 0, head->hidden, at, count) &&
           h3_gpu_cast_bf16_to_f32(head->gpu, head->x, head->half,
                                   (uint32_t)count);
}

static int keep_rows(head_context *head, size_t at, size_t count) {
    return h3_gpu_cast_f32_to_bf16(head->gpu, head->half, head->x,
                                   (uint32_t)count) &&
           h3_gpu_copy_bf16(head->gpu, head->hidden, at, head->half, 0, count);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

int h3_pixel_head_frames(int latent_t) {
    int frames = 0;
    for (int step = 0; step < latent_t; step++)
        frames += step % BLOCK == 0 ? 1 : PATCH_FRAMES;
    return frames;
}

/* --- weights ----------------------------------------------------------- */

static h3_gpu_tensor *weight1(head_context *head, const char *name,
                              uint64_t width, char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(head->store, head->gpu, name, 1, shape, error,
                              error_size);
}

static h3_gpu_tensor *weight2(head_context *head, const char *name,
                              uint64_t rows, uint64_t columns, char *error,
                              size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(head->store, head->gpu, name, 2, shape, error,
                              error_size);
}

static void free_block(head_block *block) {
    free_tensor(&block->norm1); free_tensor(&block->qkv_w);
    free_tensor(&block->qkv_b); free_tensor(&block->out_w);
    free_tensor(&block->out_b); free_tensor(&block->scale1);
    free_tensor(&block->norm2); free_tensor(&block->w1);
    free_tensor(&block->w1_b); free_tensor(&block->w2);
    free_tensor(&block->w2_b); free_tensor(&block->scale2);
    free_tensor(&block->qkv_q); free_tensor(&block->qkv_s);
    free_tensor(&block->out_q); free_tensor(&block->out_s);
    free_tensor(&block->w1_q); free_tensor(&block->w1_s);
    free_tensor(&block->w2_q); free_tensor(&block->w2_s);
}

/* An F32 [rows, columns] weight as per-row int8 and F32 scales, staged
 * through BF16 as the quantizer reads it (h3_video_vae.c's quantize_linear). */
static int quantize_linear(head_context *head, h3_gpu_tensor **weight,
                           h3_gpu_tensor **quantized, h3_gpu_tensor **scales,
                           uint32_t rows, uint32_t columns, char *error,
                           size_t error_size) {
    size_t count = (size_t)rows * columns;
    h3_gpu_tensor *staged = h3_gpu_tensor_new_bf16(head->gpu, count);
    *quantized = h3_gpu_tensor_new_i8(head->gpu, count);
    *scales = h3_gpu_tensor_new_f32(head->gpu, rows);
    int ok = staged && *quantized && *scales &&
             h3_gpu_begin(head->gpu) &&
             h3_gpu_cast_f32_to_bf16(head->gpu, staged, *weight,
                                     (uint32_t)count) &&
             h3_gpu_quantize_weight_int8(head->gpu, *quantized, *scales,
                                         staged, rows, columns) &&
             h3_gpu_submit(head->gpu);
    h3_gpu_tensor_free(staged);
    if (!ok) {
        fail(error, error_size, "pixel head: cannot quantize a weight: %s",
             h3_gpu_error(head->gpu));
        return 0;
    }
    free_tensor(weight);
    return 1;
}

static int load_block(head_context *head, int index, head_block *block,
                      char *error, size_t error_size) {
    char name[128];
#define NAME(suffix) (snprintf(name, sizeof(name),                             \
    "dwd_pixel_refiner.blocks.%d." suffix, index), name)
    memset(block, 0, sizeof(*block));
    block->norm1 = weight1(head, NAME("norm1.weight"), DIM, error, error_size);
    if (block->norm1) block->qkv_w = weight2(head, NAME("attn.to_qkv.weight"),
                                             3 * DIM, DIM, error, error_size);
    if (block->qkv_w) block->qkv_b = weight1(head, NAME("attn.to_qkv.bias"),
                                             3 * DIM, error, error_size);
    if (block->qkv_b) block->out_w = weight2(head, NAME("attn.to_out.weight"),
                                             DIM, DIM, error, error_size);
    if (block->out_w) block->out_b = weight1(head, NAME("attn.to_out.bias"),
                                             DIM, error, error_size);
    if (block->out_b) block->scale1 = weight1(head, NAME("scale1"), DIM,
                                              error, error_size);
    if (block->scale1) block->norm2 = weight1(head, NAME("norm2.weight"), DIM,
                                              error, error_size);
    if (block->norm2) block->w1 = weight2(head, NAME("ff.w1.weight"), 2 * FFN,
                                          DIM, error, error_size);
    if (block->w1) block->w1_b = weight1(head, NAME("ff.w1.bias"), 2 * FFN,
                                         error, error_size);
    if (block->w1_b) block->w2 = weight2(head, NAME("ff.w2.weight"), DIM, FFN,
                                         error, error_size);
    if (block->w2) block->w2_b = weight1(head, NAME("ff.w2.bias"), DIM, error,
                                         error_size);
    if (block->w2_b) block->scale2 = weight1(head, NAME("scale2"), DIM, error,
                                             error_size);
#undef NAME
    if (block->scale2 &&
        (!head->int8_linears ||
         (quantize_linear(head, &block->qkv_w, &block->qkv_q, &block->qkv_s,
                          3 * DIM, DIM, error, error_size) &&
          quantize_linear(head, &block->out_w, &block->out_q, &block->out_s,
                          DIM, DIM, error, error_size) &&
          quantize_linear(head, &block->w1, &block->w1_q, &block->w1_s,
                          2 * FFN, DIM, error, error_size) &&
          quantize_linear(head, &block->w2, &block->w2_q, &block->w2_s,
                          DIM, FFN, error, error_size)))) return 1;
    free_block(block);
    return 0;
}

/* --- which rows attend to which ------------------------------------------- */

typedef struct {
    uint32_t count;       /* tokens in it */
    uint32_t globals;     /* native blocks whose global tokens it sees: 1 or 2 */
    uint32_t block[2];
    uint32_t *tokens;
} window;

static int compare_windows(const void *left, const void *right) {
    const window *a = left, *b = right;
    if (a->count != b->count) return a->count < b->count ? -1 : 1;
    if (a->globals != b->globals) return a->globals < b->globals ? -1 : 1;
    return a->tokens[0] < b->tokens[0] ? -1 : a->tokens[0] > b->tokens[0];
}

static void free_layout(attention_layout *layout) {
    for (size_t index = 0; index < layout->count; index++) {
        h3_gpu_tensor_free(layout->calls[index].q_index);
        h3_gpu_tensor_free(layout->calls[index].kv_index);
    }
    free(layout->calls);
    memset(layout, 0, sizeof(*layout));
}

/* Appends one call of `batch` equal sequences; q and kv hold their rows. */
static int add_call(head_context *head, attention_layout *layout,
                    uint32_t batch, uint32_t q_len, uint32_t kv_len,
                    const uint32_t *q, const uint32_t *kv) {
    attention_call *grown = realloc(layout->calls,
                                    (layout->count + 1) * sizeof(*grown));
    if (!grown) return 0;
    layout->calls = grown;
    attention_call *call = &grown[layout->count];
    call->batch = batch;
    call->q_len = q_len;
    call->kv_len = kv_len;
    call->q_index = h3_gpu_tensor_from_u32(head->gpu, q, (size_t)batch * q_len);
    call->kv_index = h3_gpu_tensor_from_u32(head->gpu, kv,
                                            (size_t)batch * kv_len);
    layout->count++;
    return call->q_index && call->kv_index;
}

/* The windows of one layer parity: tokens in (t, y, x) order sit at
 * [st + t, s + y, s + x] of a grid padded to whole windows, each window
 * BLOCK x WINDOW x WINDOW of it. A window also sees the global tokens of the
 * native blocks its latent steps belong to. Windows of one size go through
 * the attention kernel together. */
static int build_windows(head_context *head, int shifted,
                         attention_layout *layout) {
    uint32_t s = shifted ? WINDOW / 2 : 0, st = shifted ? T_SHIFT : 0;
    uint32_t T = head->T, H = head->H, W = head->W;
    uint32_t tp = (T + st + BLOCK - 1) / BLOCK;
    uint32_t hp = (H + s + WINDOW - 1) / WINDOW;
    uint32_t wp = (W + s + WINDOW - 1) / WINDOW;
    size_t capacity = (size_t)tp * hp * wp, count = 0;
    window *windows = calloc(capacity, sizeof(*windows));
    uint32_t *tokens = malloc((size_t)head->tokens * sizeof(*tokens));
    uint32_t *q = NULL, *kv = NULL;
    int ok = windows && tokens;
    size_t used = 0;
    for (uint32_t j = 0; ok && j < tp; j++) {
        uint32_t t_begin = j * BLOCK > st ? j * BLOCK - st : 0;
        uint32_t t_end = (j + 1) * BLOCK - st < T ? (j + 1) * BLOCK - st : T;
        if (t_begin >= t_end) continue;
        for (uint32_t a = 0; a < hp; a++) {
            uint32_t y_begin = a * WINDOW > s ? a * WINDOW - s : 0;
            uint32_t y_end = (a + 1) * WINDOW - s < H ? (a + 1) * WINDOW - s : H;
            if (y_begin >= y_end) continue;
            for (uint32_t b = 0; b < wp; b++) {
                uint32_t x_begin = b * WINDOW > s ? b * WINDOW - s : 0;
                uint32_t x_end = (b + 1) * WINDOW - s < W ?
                    (b + 1) * WINDOW - s : W;
                if (x_begin >= x_end) continue;
                window *entry = &windows[count++];
                entry->tokens = tokens + used;
                for (uint32_t t = t_begin; t < t_end; t++)
                    for (uint32_t y = y_begin; y < y_end; y++)
                        for (uint32_t x = x_begin; x < x_end; x++)
                            tokens[used++] = (t * H + y) * W + x;
                entry->count = (uint32_t)(tokens + used - entry->tokens);
                entry->block[0] = t_begin / BLOCK;
                entry->block[1] = (t_end - 1) / BLOCK;
                entry->globals = entry->block[1] != entry->block[0] ? 2 : 1;
            }
        }
    }
    if (ok) qsort(windows, count, sizeof(*windows), compare_windows);
    size_t row_capacity = (size_t)ATTENTION_ROWS + BLOCK * WINDOW * WINDOW +
                          2 * GLOBALS;
    q = malloc(row_capacity * sizeof(*q));
    kv = malloc(row_capacity * sizeof(*kv));
    ok = ok && q && kv;
    for (size_t first = 0; ok && first < count;) {
        uint32_t q_len = windows[first].count;
        uint32_t kv_len = q_len + windows[first].globals * GLOBALS;
        uint32_t limit = ATTENTION_ROWS / kv_len ? ATTENTION_ROWS / kv_len : 1;
        uint32_t batch = 0;
        while (first + batch < count && batch < limit &&
               windows[first + batch].count == q_len &&
               windows[first + batch].globals == windows[first].globals) {
            const window *entry = &windows[first + batch];
            memcpy(q + (size_t)batch * q_len, entry->tokens,
                   q_len * sizeof(*q));
            uint32_t *keys = kv + (size_t)batch * kv_len;
            memcpy(keys, entry->tokens, q_len * sizeof(*kv));
            for (uint32_t g = 0; g < entry->globals; g++)
                for (uint32_t i = 0; i < GLOBALS; i++)
                    keys[q_len + g * GLOBALS + i] =
                        head->tokens + entry->block[g] * GLOBALS + i;
            batch++;
        }
        ok = add_call(head, layout, batch, q_len, kv_len, q, kv);
        first += batch;
    }
    free(windows);
    free(tokens);
    free(q);
    free(kv);
    return ok;
}

/* A native block's global tokens see every token of the block and each
 * other. */
static int build_globals(head_context *head, attention_layout *layout) {
    uint32_t per_block = BLOCK * head->H * head->W;
    uint32_t *q = malloc(GLOBALS * sizeof(*q));
    uint32_t *kv = malloc(((size_t)per_block + GLOBALS) * sizeof(*kv));
    int ok = q && kv;
    for (uint32_t block = 0; ok && block < head->blocks; block++) {
        uint32_t begin = block * per_block;
        uint32_t end = begin + per_block < head->tokens ? begin + per_block
                                                         : head->tokens;
        uint32_t count = end - begin;
        for (uint32_t i = 0; i < count; i++) kv[i] = begin + i;
        for (uint32_t i = 0; i < GLOBALS; i++)
            q[i] = kv[count + i] = head->tokens + block * GLOBALS + i;
        ok = add_call(head, layout, 1, GLOBALS, count + GLOBALS, q, kv);
    }
    free(q);
    free(kv);
    return ok;
}

/* cos and sin of a row's rotary angles, [rows, ROPE_HALF]: the position of
 * a token inside its window (time, then y, then x; 8 frequencies each).
 * Global tokens are not rotated. */
static int build_rope(head_context *head, int shifted) {
    uint32_t s = shifted ? WINDOW / 2 : 0, st = shifted ? T_SHIFT : 0;
    size_t count = (size_t)head->rows * ROPE_HALF;
    float *cos_values = malloc(count * sizeof(float));
    float *sin_values = malloc(count * sizeof(float));
    if (!cos_values || !sin_values) {
        free(cos_values); free(sin_values);
        return 0;
    }
    float frequency[ROPE_AXIS];
    for (int i = 0; i < ROPE_AXIS; i++)
        frequency[i] = 1.0f / powf(100.0f, (float)i / (float)ROPE_AXIS);
    size_t at = 0;
    for (uint32_t t = 0; t < head->T; t++)
        for (uint32_t y = 0; y < head->H; y++)
            for (uint32_t x = 0; x < head->W; x++) {
                float position[3] = {
                    ((float)((t + st) % BLOCK) + 0.5f) / (float)BLOCK * 2.0f - 1.0f,
                    ((float)((y + s) % WINDOW) + 0.5f) / (float)WINDOW * 2.0f - 1.0f,
                    ((float)((x + s) % WINDOW) + 0.5f) / (float)WINDOW * 2.0f - 1.0f,
                };
                for (int axis = 0; axis < 3; axis++)
                    for (int i = 0; i < ROPE_AXIS; i++, at++) {
                        float angle = 2.0f * (float)M_PI * position[axis] *
                                      frequency[i];
                        cos_values[at] = cosf(angle);
                        sin_values[at] = sinf(angle);
                    }
            }
    for (; at < count; at++) {
        cos_values[at] = 1.0f;
        sin_values[at] = 0.0f;
    }
    head->rope_cos[shifted] = h3_gpu_tensor_from_f32(head->gpu, cos_values, count);
    head->rope_sin[shifted] = h3_gpu_tensor_from_f32(head->gpu, sin_values, count);
    free(cos_values);
    free(sin_values);
    return head->rope_cos[shifted] && head->rope_sin[shifted];
}

/* --- the computation ------------------------------------------------------ */

/* What the layers work in, beside the hidden rows: made once the feature
 * stage has given its own buffers back, so the two are never held at once. */
static int allocate(head_context *head, char *error, size_t error_size) {
    size_t rows = head->rows;
    head->query = h3_gpu_tensor_new_bf16(head->gpu, rows * DIM);
    head->key = h3_gpu_tensor_new_bf16(head->gpu, rows * DIM);
    head->value = h3_gpu_tensor_new_bf16(head->gpu, rows * DIM);
    head->x = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * DIM);
    head->norm = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * DIM);
    head->qkv = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * 3 * DIM);
    head->qf = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * DIM);
    head->kf = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * DIM);
    head->vf = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * DIM);
    head->branch_in = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * DIM);
    head->branch = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * DIM);
    head->ff1 = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * 2 * FFN);
    head->activated = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * FFN);
    head->half = h3_gpu_tensor_new_bf16(head->gpu, (size_t)CHUNK * DIM);
    head->cos = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * ROPE_HALF);
    head->sin = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * ROPE_HALF);
    size_t q_rows = 1, kv_rows = 1;
    const attention_layout *layouts[] = {&head->global, &head->window[0],
                                         &head->window[1]};
    for (int which = 0; which < 3; which++)
        for (size_t index = 0; index < layouts[which]->count; index++) {
            const attention_call *call = &layouts[which]->calls[index];
            if ((size_t)call->batch * call->q_len > q_rows)
                q_rows = (size_t)call->batch * call->q_len;
            if ((size_t)call->batch * call->kv_len > kv_rows)
                kv_rows = (size_t)call->batch * call->kv_len;
        }
    if (head->int8_linears) {
        /* For the widest linear: FFN in, 2 * FFN out. The int8 kernels read
         * whole 128-row tiles (CHUNK is a multiple of 128). */
        head->lin_in = h3_gpu_tensor_new_bf16(head->gpu, (size_t)CHUNK * FFN);
        head->lin_out = h3_gpu_tensor_new_bf16(head->gpu,
                                               (size_t)CHUNK * 2 * FFN);
        head->lin_quantized = h3_gpu_tensor_new_i8(head->gpu,
                                                   (size_t)CHUNK * FFN);
        head->lin_scales = h3_gpu_tensor_new_f32(head->gpu, CHUNK);
        if (!head->lin_in || !head->lin_out || !head->lin_quantized ||
            !head->lin_scales) {
            fail(error, error_size, "pixel head: cannot allocate its int8 "
                 "scratch: %s", h3_gpu_error(head->gpu));
            return 0;
        }
    }
    head->q_rows = h3_gpu_tensor_new_bf16(head->gpu, q_rows * DIM);
    head->a_rows = h3_gpu_tensor_new_bf16(head->gpu, q_rows * DIM);
    head->k_rows = h3_gpu_tensor_new_bf16(head->gpu, kv_rows * DIM);
    head->v_rows = h3_gpu_tensor_new_bf16(head->gpu, kv_rows * DIM);
    if (head->query && head->key && head->value &&
        head->x && head->norm && head->qkv && head->qf &&
        head->kf && head->vf && head->branch_in && head->branch && head->ff1 &&
        head->activated && head->half && head->cos && head->sin &&
        head->q_rows && head->a_rows && head->k_rows && head->v_rows)
        return 1;
    fail(error, error_size, "pixel head: cannot allocate its activations: %s",
         h3_gpu_error(head->gpu));
    return 0;
}

/* The DiT's rows through the head's AdaLN and feature_expand, each into
 * its EXPAND x EXPAND tokens of the head's grid; then the global tokens. */
static int expand_features(head_context *head, const uint16_t *rows,
                           const float *modulation, char *error,
                           size_t error_size) {
    uint32_t count = (uint32_t)head->latent_t * head->token_h * head->token_w;
    uint32_t expanded = EXPAND * EXPAND * DIM;
    float *scale = malloc(FEATURE * sizeof(float));
    float *host = malloc((size_t)CHUNK * expanded * sizeof(float));
    uint16_t row16[GLOBALS * DIM];
    h3_gpu_tensor *weight = NULL, *bias = NULL, *mod_scale = NULL;
    h3_gpu_tensor *mod_shift = NULL, *in16 = NULL, *in32 = NULL, *out = NULL;
    h3_gpu_tensor *features = NULL, *registers = NULL;
    int ok = scale && host;
    if (ok) {
        for (int i = 0; i < FEATURE; i++) scale[i] = 1.0f + modulation[i];
        mod_scale = h3_gpu_tensor_from_f32(head->gpu, scale, FEATURE);
        mod_shift = h3_gpu_tensor_from_f32(head->gpu, modulation + FEATURE,
                                           FEATURE);
        in16 = h3_gpu_tensor_new_bf16(head->gpu, (size_t)CHUNK * FEATURE);
        in32 = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * FEATURE);
        features = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * FEATURE);
        out = h3_gpu_tensor_new_f32(head->gpu, (size_t)CHUNK * expanded);
        ok = mod_scale && mod_shift && in16 && in32 && features && out;
        if (!ok) fail(error, error_size, "pixel head: cannot allocate the "
                      "feature stage: %s", h3_gpu_error(head->gpu));
    } else {
        fail(error, error_size, "pixel head: out of memory");
    }
    if (ok) {
        weight = weight2(head, "dwd_pixel_refiner.feature_expand.weight",
                         expanded, FEATURE, error, error_size);
        bias = weight ? weight1(head, "dwd_pixel_refiner.feature_expand.bias",
                                expanded, error, error_size) : NULL;
        uint64_t register_shape[] = {1, REGISTERS, DIM};
        registers = bias ? h3_weight_load_f32(
            head->store, head->gpu, "dwd_pixel_refiner.register_tokens", 3,
            register_shape, error, error_size) : NULL;
        ok = registers != NULL;
    }
    uint32_t per_step = (uint32_t)head->token_h * head->token_w;
    for (uint32_t first = 0; ok && first < count; first += CHUNK) {
        uint32_t chunk = count - first < CHUNK ? count - first : CHUNK;
        ok = h3_gpu_tensor_write_bf16_range(in16, 0,
                 rows + (size_t)first * FEATURE, (size_t)chunk * FEATURE) &&
             h3_gpu_begin(head->gpu) &&
             h3_gpu_cast_bf16_to_f32(head->gpu, in32, in16, chunk * FEATURE) &&
             h3_gpu_layer_norm_f32(head->gpu, features, in32, mod_scale,
                                   mod_shift, chunk, FEATURE, 1e-5f) &&
             h3_gpu_linear_f32(head->gpu, out, features, weight, bias, chunk,
                               FEATURE, expanded) &&
             h3_gpu_submit(head->gpu) &&
             h3_gpu_tensor_read_f32_range(out, 0, host, (size_t)chunk * expanded);
        if (!ok) {
            fail(error, error_size, "pixel head: feature stage: %s",
                 h3_gpu_error(head->gpu));
            break;
        }
        for (uint32_t i = 0; ok && i < chunk; i++) {
            uint32_t row = first + i;
            uint32_t t = row / per_step, within = row % per_step;
            uint32_t y = within / (uint32_t)head->token_w;
            uint32_t x = within % (uint32_t)head->token_w;
            for (uint32_t ey = 0; ok && ey < EXPAND; ey++)
                for (uint32_t ex = 0; ok && ex < EXPAND; ex++) {
                    size_t token = ((size_t)t * head->H + y * EXPAND + ey) *
                                   head->W + x * EXPAND + ex;
                    const float *source = host +
                        ((size_t)i * EXPAND * EXPAND + ey * EXPAND + ex) * DIM;
                    for (int d = 0; d < DIM; d++) row16[d] = to_bf16(source[d]);
                    ok = h3_gpu_tensor_write_bf16_range(
                        head->hidden, token * DIM, row16, DIM);
                }
        }
        if (!ok) fail(error, error_size, "pixel head: cannot place the "
                      "expanded tokens");
    }
    if (ok) {
        /* 4 registers and a zero for every native block */
        float *globals = calloc((size_t)GLOBALS * DIM, sizeof(float));
        ok = globals && h3_gpu_tensor_read_f32_range(registers, 0, globals,
                                                     (size_t)REGISTERS * DIM);
        for (int d = 0; ok && d < GLOBALS * DIM; d++)
            row16[d] = to_bf16(globals[d]);
        for (uint32_t block = 0; ok && block < head->blocks; block++)
            ok = h3_gpu_tensor_write_bf16_range(
                head->hidden,
                ((size_t)head->tokens + (size_t)block * GLOBALS) * DIM,
                row16, (size_t)GLOBALS * DIM);
        free(globals);
        if (!ok) fail(error, error_size, "pixel head: cannot set the global "
                      "tokens");
    }
    free(scale);
    free(host);
    h3_gpu_tensor_free(weight); h3_gpu_tensor_free(bias);
    h3_gpu_tensor_free(mod_scale); h3_gpu_tensor_free(mod_shift);
    h3_gpu_tensor_free(in16); h3_gpu_tensor_free(in32);
    h3_gpu_tensor_free(features); h3_gpu_tensor_free(out);
    h3_gpu_tensor_free(registers);
    return ok;
}

/* One of a layer's linears on a chunk of rows: F32, or int8 with the bias
 * added as the result comes back to F32. The int8 kernels take whole
 * 128-row tiles, so the int8 linear always runs CHUNK rows - a last, shorter
 * chunk leaves stale rows after its own, which give rows nobody reads: each
 * row's result depends on that row alone. */
static int block_linear(head_context *head, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight,
                        const h3_gpu_tensor *quantized,
                        const h3_gpu_tensor *scales,
                        const h3_gpu_tensor *bias, uint32_t rows,
                        uint32_t input_dim, uint32_t output_dim) {
    if (!quantized)
        return h3_gpu_linear_f32(head->gpu, output, input, weight, bias, rows,
                                 input_dim, output_dim);
    return h3_gpu_cast_f32_to_bf16(head->gpu, head->lin_in, input,
                                   rows * input_dim) &&
           h3_gpu_linear_int8_bf16(head->gpu, head->lin_out,
                                   head->lin_quantized, head->lin_scales,
                                   head->lin_in, quantized, scales, CHUNK,
                                   input_dim, output_dim, 0) &&
           h3_gpu_cast_bf16_to_f32_bias(head->gpu, output, head->lin_out, bias,
                                        rows, output_dim);
}

/* The rows of the next pass, from `first`. */
static uint32_t next_chunk(const head_context *head, uint32_t first) {
    uint32_t left = head->rows - first;
    return left < CHUNK ? left : CHUNK;
}

static int attend(head_context *head, const attention_layout *layout,
                  char *error, size_t error_size) {
    for (size_t index = 0; index < layout->count; index++) {
        const attention_call *call = &layout->calls[index];
        uint32_t q_rows = call->batch * call->q_len;
        uint32_t kv_rows = call->batch * call->kv_len;
        OP(h3_gpu_gather_rows_bf16(head->gpu, head->q_rows, head->query,
            call->q_index, q_rows, DIM), "gather queries");
        OP(h3_gpu_gather_rows_bf16(head->gpu, head->k_rows, head->key,
            call->kv_index, kv_rows, DIM), "gather keys");
        OP(h3_gpu_gather_rows_bf16(head->gpu, head->v_rows, head->value,
            call->kv_index, kv_rows, DIM), "gather values");
        OP(h3_gpu_sdpa_batched_kv_bf16(head->gpu, head->a_rows, head->q_rows,
            head->k_rows, head->v_rows, call->batch, call->q_len,
            call->kv_len, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "attention");
        /* Every row is a query once a layer, so its attention takes its
         * query's place. */
        OP(h3_gpu_scatter_rows_bf16(head->gpu, head->query,
            head->a_rows, call->q_index, q_rows, DIM), "place attention");
    }
    return 1;
}

static int run_layer(head_context *head, int index, const head_block *block,
                     char *error, size_t error_size) {
    int shifted = (FIRST_LAYER + index) % 2;
    OP(h3_gpu_begin(head->gpu), "begin a layer");
    uint32_t chunk = 0;
    for (uint32_t first = 0; first < head->rows; first += chunk) {
        chunk = next_chunk(head, first);
        size_t at = (size_t)first * DIM, count = (size_t)chunk * DIM;
        OP(take_rows(head, at, count), "take rows");
        OP(h3_gpu_rms_norm_f32(head->gpu, head->norm, head->x, block->norm1,
            chunk, DIM, 1e-5f), "attention norm");
        OP(block_linear(head, head->qkv, head->norm, block->qkv_w,
            block->qkv_q, block->qkv_s, block->qkv_b, chunk, DIM, 3 * DIM),
           "QKV");
        OP(h3_gpu_copy_f32(head->gpu, head->cos, 0, head->rope_cos[shifted],
            (size_t)first * ROPE_HALF, (size_t)chunk * ROPE_HALF), "RoPE cos");
        OP(h3_gpu_copy_f32(head->gpu, head->sin, 0, head->rope_sin[shifted],
            (size_t)first * ROPE_HALF, (size_t)chunk * ROPE_HALF), "RoPE sin");
        OP(h3_gpu_video_qkv_rope_f32(head->gpu, head->qf, head->kf, head->vf,
            head->qkv, head->cos, head->sin, chunk, HEADS, HEAD_DIM,
            ROPE_HALF, 1e-5f), "QK norm/RoPE");
        h3_gpu_tensor *sources[] = {head->qf, head->kf, head->vf};
        h3_gpu_tensor *targets[] = {head->query, head->key, head->value};
        for (int part = 0; part < 3; part++) {
            OP(h3_gpu_cast_f32_to_bf16(head->gpu, head->half, sources[part],
                (uint32_t)count), "Q/K/V to BF16");
            OP(h3_gpu_copy_bf16(head->gpu, targets[part], at, head->half, 0,
                count), "keep Q/K/V");
        }
    }
    if (!attend(head, &head->global, error, error_size) ||
        !attend(head, &head->window[shifted], error, error_size)) return 0;
    for (uint32_t first = 0; first < head->rows; first += chunk) {
        chunk = next_chunk(head, first);
        size_t at = (size_t)first * DIM, count = (size_t)chunk * DIM;
        OP(h3_gpu_copy_bf16(head->gpu, head->half, 0, head->query, at,
            count), "take attention");
        OP(h3_gpu_cast_bf16_to_f32(head->gpu, head->branch_in, head->half,
            (uint32_t)count), "attention to F32");
        OP(block_linear(head, head->branch, head->branch_in, block->out_w,
            block->out_q, block->out_s, block->out_b, chunk, DIM, DIM),
           "attention output");
        OP(take_rows(head, at, count), "take rows");
        OP(h3_gpu_scale_add_f32(head->gpu, head->x, head->x, head->branch,
            block->scale1, chunk, DIM), "attention residual");
        OP(h3_gpu_rms_norm_f32(head->gpu, head->norm, head->x, block->norm2,
            chunk, DIM, 1e-5f), "MLP norm");
        OP(block_linear(head, head->ff1, head->norm, block->w1, block->w1_q,
            block->w1_s, block->w1_b, chunk, DIM, 2 * FFN), "MLP input");
        OP(h3_gpu_swiglu_f32(head->gpu, head->activated, head->ff1, chunk,
            FFN), "SwiGLU");
        OP(block_linear(head, head->branch, head->activated, block->w2,
            block->w2_q, block->w2_s, block->w2_b, chunk, FFN, DIM),
           "MLP output");
        OP(h3_gpu_scale_add_f32(head->gpu, head->x, head->x, head->branch,
            block->scale2, chunk, DIM), "MLP residual");
        OP(keep_rows(head, at, count), "keep rows");
    }
    OP(h3_gpu_submit(head->gpu), "finish a layer");
    return 1;
}

/* The final norm and projection, a latent step at a time: 1 frame for the
 * first step of every native block, 4 for the others. A token's values are
 * (channel, frame, y, x). */
static int project(head_context *head, h3_video_chunk_callback callback,
                   void *callback_opaque, char *error, size_t error_size) {
    uint32_t per_step = head->H * head->W;
    int height = (int)head->H * PATCH, width = (int)head->W * PATCH;
    uint32_t widest = CHANNELS * PATCH_FRAMES * PATCH * PATCH;
    h3_gpu_tensor *norm_w = weight1(head, "dwd_pixel_refiner.norm_out.weight",
                                    DIM, error, error_size);
    h3_gpu_tensor *norm_b = norm_w ? weight1(
        head, "dwd_pixel_refiner.norm_out.bias", DIM, error, error_size) : NULL;
    h3_gpu_tensor *weights[2] = {NULL, NULL}, *biases[2] = {NULL, NULL};
    static const char *const names[2] = {
        "dwd_pixel_refiner.proj_out", "dwd_pixel_refiner.first_frame_proj_out"};
    const uint32_t outputs[2] = {widest, CHANNELS * PATCH * PATCH};
    int ok = norm_b != NULL;
    for (int which = 0; ok && which < 2; which++) {
        char name[96];
        snprintf(name, sizeof(name), "%s.weight", names[which]);
        weights[which] = weight2(head, name, outputs[which], DIM, error,
                                 error_size);
        snprintf(name, sizeof(name), "%s.bias", names[which]);
        biases[which] = weights[which] ? weight1(head, name, outputs[which],
                                                 error, error_size) : NULL;
        ok = biases[which] != NULL;
    }
    h3_gpu_tensor *projected = ok ? h3_gpu_tensor_new_f32(
        head->gpu, (size_t)CHUNK * widest) : NULL;
    float *host = malloc((size_t)CHUNK * widest * sizeof(float));
    float *frames = malloc((size_t)PATCH_FRAMES * height * width * CHANNELS *
                           sizeof(float));
    if (ok && (!projected || !host || !frames)) {
        fail(error, error_size, "pixel head: cannot allocate the projection");
        ok = 0;
    }
    for (uint32_t step = 0; ok && step < head->T; step++) {
        int which = step % BLOCK == 0;
        uint32_t count = which ? 1 : PATCH_FRAMES;
        uint32_t values = outputs[which];
        for (uint32_t first = 0; ok && first < per_step; first += CHUNK) {
            uint32_t chunk = per_step - first < CHUNK ? per_step - first : CHUNK;
            size_t at = ((size_t)step * per_step + first) * DIM;
            ok = h3_gpu_begin(head->gpu) &&
                 take_rows(head, at, (size_t)chunk * DIM) &&
                 h3_gpu_layer_norm_f32(head->gpu, head->norm, head->x, norm_w,
                                       norm_b, chunk, DIM, 1e-5f) &&
                 h3_gpu_linear_f32(head->gpu, projected, head->norm,
                                   weights[which], biases[which], chunk, DIM,
                                   values) &&
                 h3_gpu_submit(head->gpu) &&
                 h3_gpu_tensor_read_f32_range(projected, 0, host,
                                              (size_t)chunk * values);
            if (!ok) {
                fail(error, error_size, "pixel head: projection: %s",
                     h3_gpu_error(head->gpu));
                break;
            }
            for (uint32_t i = 0; i < chunk; i++) {
                uint32_t token = first + i;
                uint32_t ty = token / head->W, tx = token % head->W;
                const float *source = host + (size_t)i * values;
                for (uint32_t channel = 0; channel < CHANNELS; channel++)
                    for (uint32_t frame = 0; frame < count; frame++)
                        for (uint32_t py = 0; py < PATCH; py++) {
                            const float *line = source +
                                ((channel * count + frame) * PATCH + py) * PATCH;
                            float *target = frames +
                                (((size_t)frame * height + ty * PATCH + py) *
                                 width + tx * PATCH) * CHANNELS + channel;
                            for (uint32_t px = 0; px < PATCH; px++) {
                                float value = line[px] * rgb_std[channel] +
                                              rgb_mean[channel];
                                value = value < -1.0f ? -1.0f :
                                        value > 1.0f ? 1.0f : value;
                                target[(size_t)px * CHANNELS] =
                                    (value + 1.0f) * 0.5f;
                            }
                        }
            }
        }
        if (ok && !callback(callback_opaque, frames, (int)count, height, width,
                            error, error_size)) {
            if (error && error_size && !error[0])
                fail(error, error_size, "pixel head: stopped");
            ok = 0;
        }
    }
    free(host);
    free(frames);
    h3_gpu_tensor_free(projected);
    h3_gpu_tensor_free(norm_w); h3_gpu_tensor_free(norm_b);
    for (int which = 0; which < 2; which++) {
        h3_gpu_tensor_free(weights[which]);
        h3_gpu_tensor_free(biases[which]);
    }
    return ok;
}

static void free_context(head_context *head) {
    free_tensor(&head->hidden); free_tensor(&head->query);
    free_tensor(&head->key); free_tensor(&head->value);
    for (int shifted = 0; shifted < 2; shifted++) {
        free_tensor(&head->rope_cos[shifted]);
        free_tensor(&head->rope_sin[shifted]);
        free_layout(&head->window[shifted]);
    }
    free_layout(&head->global);
    free_tensor(&head->x); free_tensor(&head->norm); free_tensor(&head->qkv);
    free_tensor(&head->qf); free_tensor(&head->kf); free_tensor(&head->vf);
    free_tensor(&head->branch_in); free_tensor(&head->branch);
    free_tensor(&head->ff1); free_tensor(&head->activated);
    free_tensor(&head->half); free_tensor(&head->cos); free_tensor(&head->sin);
    free_tensor(&head->lin_in); free_tensor(&head->lin_out);
    free_tensor(&head->lin_quantized); free_tensor(&head->lin_scales);
    free_tensor(&head->q_rows); free_tensor(&head->k_rows);
    free_tensor(&head->v_rows); free_tensor(&head->a_rows);
    h3_weight_store_free(head->store);
    h3_gpu_free(head->gpu);
}

int h3_pixel_head_decode(const char *adapter_path,
                         const char *shader_source_path,
                         const uint16_t *rows, const float *modulation,
                         int latent_t, int token_h, int token_w,
                         h3_video_chunk_callback callback,
                         void *callback_opaque,
                         h3_video_vae_progress progress,
                         void *progress_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!adapter_path || !shader_source_path || !rows || !modulation ||
        !callback || latent_t < 1 || token_h < 1 || token_w < 1 ||
        (uint64_t)latent_t * (uint64_t)token_h * (uint64_t)token_w *
            EXPAND * EXPAND > UINT32_MAX / DIM) {
        fail(error, error_size, "invalid pixel head arguments");
        return 0;
    }
    head_context context;
    head_context *head = &context;
    memset(head, 0, sizeof(*head));
    head->latent_t = latent_t;
    head->token_h = token_h;
    head->token_w = token_w;
    head->T = (uint32_t)latent_t;
    head->H = (uint32_t)token_h * EXPAND;
    head->W = (uint32_t)token_w * EXPAND;
    head->tokens = head->T * head->H * head->W;
    head->blocks = (head->T + BLOCK - 1) / BLOCK;
    head->rows = head->tokens + head->blocks * GLOBALS;
    int ok = 0;
    head->store = h3_weight_store_open_file(adapter_path, error, error_size);
    if (!head->store) goto done;
    head->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!head->gpu) goto done;
    h3_gpu_profile_set_label(head->gpu, "pi-H3 pixel head");
    {
        /* As the video VAE decoder decides for its own block linears. */
        const char *value = h3_option("H3_VAE_INT8");
        head->int8_linears = !(value && !strcmp(value, "0")) &&
                             h3_gpu_has_int8_mlp(head->gpu);
        if (h3_option("H3_PROFILE"))
            fprintf(stderr, "h3: pixel head linears: %s\n",
                    head->int8_linears ? "int8" : "F32");
    }
    if (!build_globals(head, &head->global) ||
        !build_windows(head, 0, &head->window[0]) ||
        !build_windows(head, 1, &head->window[1]) ||
        !build_rope(head, 0) || !build_rope(head, 1)) {
        fail(error, error_size, "pixel head: cannot lay out its windows: %s",
             h3_gpu_error(head->gpu));
        goto done;
    }
    head->hidden = h3_gpu_tensor_new_bf16(head->gpu, (size_t)head->rows * DIM);
    if (!head->hidden) {
        fail(error, error_size, "pixel head: cannot allocate its rows: %s",
             h3_gpu_error(head->gpu));
        goto done;
    }
    if (!expand_features(head, rows, modulation, error, error_size) ||
        !allocate(head, error, error_size)) goto done;
    if (progress) progress(0, LAYERS, progress_opaque);
    for (int index = 0; index < LAYERS; index++) {
        head_block block;
        if (!load_block(head, index, &block, error, error_size)) goto done;
        int ran = run_layer(head, index, &block, error, error_size);
        free_block(&block);
        if (!ran) goto done;
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    ok = project(head, callback, callback_opaque, error, error_size);
done:
    free_context(head);
    return ok;
}
