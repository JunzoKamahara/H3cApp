#include "h3_image_vae.h"

#include "h3_weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = 24,
    HEAD_DIM = 64,
    REGISTERS = 4,
    SUFFIX = 5,              /* the registers and one zero token */
    ROPE_FREQUENCIES = 8,
    ROPE_HALF = 2 * ROPE_FREQUENCIES,   /* (h, w): half of each head rotates */
    PATCH = 16,
    OUTPUT_PATCH = 3 * PATCH * PATCH,
    TILE_LATENT = 16,        /* 256 pixels */
    TILE_OVERLAP_MIN = 4,    /* 64 pixels */
    MAX_LAYERS = 64,
    MAX_HIDDEN = 4096
};

typedef struct {
    h3_gpu_tensor *norm1, *qkv_w, *qkv_b, *out_w, *out_b, *scale1;
    h3_gpu_tensor *norm2, *w1, *w1_b, *w2, *w2_b, *scale2;
} vae_block;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    /* From the weights. */
    uint32_t dim, heads, ffn;
    int layers;
    vae_block blocks[MAX_LAYERS];
    h3_gpu_tensor *post_w, *post_b, *embed_w, *embed_b, *registers;
    h3_gpu_tensor *norm_out_w, *norm_out_b, *proj_w, *proj_b;
    /* One tile's activations; every tile of an image has the same shape. */
    h3_gpu_tensor *latent, *post, *patch_hidden, *hidden, *norm, *qkv;
    h3_gpu_tensor *query, *key, *value, *attended, *branch, *ff1, *activated;
    h3_gpu_tensor *projected, *rope_cos, *rope_sin, *zero;
    int tile_h, tile_w;
    uint32_t patches, sequence;
} vae_context;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void cleanup(vae_context *vae) {
    for (int index = 0; index < MAX_LAYERS; index++) {
        vae_block *block = &vae->blocks[index];
        h3_gpu_tensor **all[] = {
            &block->norm1, &block->qkv_w, &block->qkv_b, &block->out_w,
            &block->out_b, &block->scale1, &block->norm2, &block->w1,
            &block->w1_b, &block->w2, &block->w2_b, &block->scale2
        };
        for (size_t each = 0; each < sizeof(all) / sizeof(*all); each++)
            free_tensor(all[each]);
    }
    h3_gpu_tensor **all[] = {
        &vae->post_w, &vae->post_b, &vae->embed_w, &vae->embed_b,
        &vae->registers, &vae->norm_out_w, &vae->norm_out_b, &vae->proj_w,
        &vae->proj_b, &vae->latent, &vae->post, &vae->patch_hidden,
        &vae->hidden, &vae->norm, &vae->qkv, &vae->query, &vae->key,
        &vae->value, &vae->attended, &vae->branch, &vae->ff1,
        &vae->activated, &vae->projected, &vae->rope_cos, &vae->rope_sin,
        &vae->zero
    };
    for (size_t each = 0; each < sizeof(all) / sizeof(*all); each++)
        free_tensor(all[each]);
    h3_gpu_free(vae->gpu);
    h3_weight_store_free(vae->weights);
    memset(vae, 0, sizeof(*vae));
}

/* ---- Weights ---- */

static h3_gpu_tensor *load(vae_context *vae, const char *name, int ndim,
                           const uint64_t *shape, char *error,
                           size_t error_size) {
    return h3_weight_load_f32(vae->weights, vae->gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *f1(vae_context *vae, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load(vae, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f2(vae_context *vae, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return load(vae, name, 2, shape, error, error_size);
}

/* The decoder's sizes, from the shapes of its weights. */
static int read_shape(vae_context *vae, char *error, size_t error_size) {
    const h3_st_tensor *embed =
        h3_weight_find(vae->weights, "decoder.x_embedder.weight", NULL);
    const h3_st_tensor *proj =
        h3_weight_find(vae->weights, "decoder.proj_out.weight", NULL);
    const h3_st_tensor *w2 = h3_weight_find(
        vae->weights, "decoder.transformer_blocks.0.ff.w2.weight", NULL);
    if (!embed || embed->ndim != 2 || embed->shape[1] != LATENT_CHANNELS ||
        !proj || proj->ndim != 2 || !w2 || w2->ndim != 2) {
        fail(error, error_size, "not an H3 image VAE: its decoder weights "
             "are missing or have another shape");
        return 0;
    }
    if (proj->shape[0] != OUTPUT_PATCH) {
        fail(error, error_size, "this VAE decodes %llu values per patch, not "
             "the %d of a still image (a video VAE?)",
             (unsigned long long)proj->shape[0], OUTPUT_PATCH);
        return 0;
    }
    uint64_t hidden = embed->shape[0], ffn = w2->shape[1];
    if (hidden < HEAD_DIM || hidden > MAX_HIDDEN || hidden % HEAD_DIM ||
        ffn < 1 || ffn > 16 * MAX_HIDDEN || proj->shape[1] != hidden ||
        w2->shape[0] != hidden) {
        fail(error, error_size, "unsupported H3 image VAE decoder size");
        return 0;
    }
    vae->dim = (uint32_t)hidden;
    vae->heads = (uint32_t)(hidden / HEAD_DIM);
    vae->ffn = (uint32_t)ffn;
    char name[96];
    int layers = 0;
    for (; layers <= MAX_LAYERS; layers++) {
        snprintf(name, sizeof(name),
                 "decoder.transformer_blocks.%d.attn.to_qkv.weight", layers);
        if (!h3_weight_find(vae->weights, name, NULL)) break;
    }
    if (layers < 1 || layers > MAX_LAYERS) {
        fail(error, error_size, "unsupported H3 image VAE decoder depth");
        return 0;
    }
    vae->layers = layers;
    return 1;
}

static int load_block(vae_context *vae, int index, char *error,
                      size_t error_size) {
    vae_block *block = &vae->blocks[index];
    uint32_t hidden = vae->dim, ffn = vae->ffn;
    char prefix[64], name[128];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
#define F1(field, suffix, width) do {                                           \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                      \
    block->field = f1(vae, name, width, error, error_size);                     \
    if (!block->field) return 0;                                                \
} while (0)
#define F2(field, suffix, rows, columns) do {                                   \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                      \
    block->field = f2(vae, name, rows, columns, error, error_size);             \
    if (!block->field) return 0;                                                \
} while (0)
    F1(norm1, "norm1.weight", hidden);
    F2(qkv_w, "attn.to_qkv.weight", hidden * 3, hidden);
    F1(qkv_b, "attn.to_qkv.bias", hidden * 3);
    F2(out_w, "attn.to_out.weight", hidden, hidden);
    F1(out_b, "attn.to_out.bias", hidden);
    F1(scale1, "scale1", hidden);
    F1(norm2, "norm2.weight", hidden);
    F2(w1, "ff.w1.weight", ffn * 2, hidden);
    F1(w1_b, "ff.w1.bias", ffn * 2);
    F2(w2, "ff.w2.weight", hidden, ffn);
    F1(w2_b, "ff.w2.bias", hidden);
    F1(scale2, "scale2", hidden);
#undef F1
#undef F2
    return 1;
}

static int load_weights(vae_context *vae, char *error, size_t error_size) {
    uint32_t hidden = vae->dim;
    uint64_t post_shape[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1};
    uint64_t register_shape[] = {1, REGISTERS, hidden};
    if (!(vae->post_w = load(vae, "post_quant_conv.weight", 4, post_shape,
                             error, error_size)) ||
        !(vae->post_b = f1(vae, "post_quant_conv.bias", LATENT_CHANNELS,
                           error, error_size)) ||
        !(vae->embed_w = f2(vae, "decoder.x_embedder.weight", hidden,
                            LATENT_CHANNELS, error, error_size)) ||
        !(vae->embed_b = f1(vae, "decoder.x_embedder.bias", hidden,
                            error, error_size)) ||
        !(vae->registers = load(vae, "decoder.register_tokens", 3,
                                register_shape, error, error_size)))
        return 0;
    for (int index = 0; index < vae->layers; index++)
        if (!load_block(vae, index, error, error_size)) return 0;
    return (vae->norm_out_w = f1(vae, "decoder.norm_out.weight", hidden,
                                 error, error_size)) &&
           (vae->norm_out_b = f1(vae, "decoder.norm_out.bias", hidden,
                                 error, error_size)) &&
           (vae->proj_w = f2(vae, "decoder.proj_out.weight", OUTPUT_PATCH,
                             hidden, error, error_size)) &&
           (vae->proj_b = f1(vae, "decoder.proj_out.bias", OUTPUT_PATCH,
                             error, error_size));
}

/* ---- Latent normalization (config.json) ---- */

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "image VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);     /* skips leading space */
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "image VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end + strspn(end, " \t\r\n");
        if (*cursor != (index + 1 < count ? ',' : ']')) {
            fail(error, error_size, "image VAE config has %s %s",
                 index + 1 < count ? "a short" : "a long", key);
            return 0;
        }
        cursor++;
    }
    return 1;
}

static char *read_small_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    char *text = NULL;
    long end = fseek(file, 0, SEEK_END) ? -1 : ftell(file);
    if (end > 0 && end <= 1024 * 1024 && !fseek(file, 0, SEEK_SET) &&
        (text = malloc((size_t)end + 1))) {
        if (fread(text, 1, (size_t)end, file) == (size_t)end) {
            text[end] = '\0';
        } else {
            free(text);
            text = NULL;
        }
    }
    fclose(file);
    return text;
}

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
                                     char *error, size_t error_size) {
    static const char *places[] = {"%s/config.json", "%s/../config.json"};
    char *json = NULL;
    for (size_t place = 0; place < 2 && !json; place++) {
        char path[4096];
        if (snprintf(path, sizeof(path), places[place], weight_directory) <
            (int)sizeof(path)) json = read_small_file(path);
    }
    if (!json) {
        fail(error, error_size, "cannot read the image VAE's config.json in "
             "%s or the folder above it", weight_directory);
        return 0;
    }
    int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    for (int channel = 0; ok && channel < LATENT_CHANNELS; channel++)
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size,
                 "image VAE latent standard deviation is invalid");
            ok = 0;
        }
    return ok;
}

/* ---- One tile ---- */

static int prepare_tile_buffers(vae_context *vae, char *error,
                                size_t error_size) {
    size_t patches = vae->patches, sequence = vae->sequence;
    size_t hidden = vae->dim, ffn = vae->ffn;
    /* Positions are the tile's own (-1..1 across it), the same for every
     * tile; the suffix tokens sit at the origin and do not rotate. */
    size_t count = sequence * ROPE_HALF;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    float *zeros = calloc(hidden, sizeof(*zeros));
    if (!cosines || !sines || !zeros) {
        free(cosines); free(sines); free(zeros);
        fail(error, error_size, "out of memory preparing the image VAE");
        return 0;
    }
    for (size_t row = 0; row < sequence; row++) {
        float axes[2] = {0.0f, 0.0f};
        if (row < patches) {
            size_t y = row / (size_t)vae->tile_w, x = row % (size_t)vae->tile_w;
            axes[0] = ((float)y + 0.5f) * 2.0f / (float)vae->tile_h - 1.0f;
            axes[1] = ((float)x + 0.5f) * 2.0f / (float)vae->tile_w - 1.0f;
        }
        for (int axis = 0; axis < 2; axis++)
            for (int frequency = 0; frequency < ROPE_FREQUENCIES; frequency++) {
                double angle = 2.0 * 3.14159265358979323846 * (double)axes[axis] *
                    pow(100.0, -(double)frequency / (double)ROPE_FREQUENCIES);
                size_t offset = row * ROPE_HALF +
                                (size_t)(axis * ROPE_FREQUENCIES + frequency);
                cosines[offset] = (float)cos(angle);
                sines[offset] = (float)sin(angle);
            }
    }
    vae->rope_cos = h3_gpu_tensor_from_f32(vae->gpu, cosines, count);
    vae->rope_sin = h3_gpu_tensor_from_f32(vae->gpu, sines, count);
    vae->zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, hidden);
    free(cosines); free(sines); free(zeros);
#define F32(field, elements) (vae->field = h3_gpu_tensor_new_f32(vae->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        vae->rope_cos, vae->rope_sin, vae->zero,
        F32(latent, patches * LATENT_CHANNELS),
        F32(post, patches * LATENT_CHANNELS),
        F32(patch_hidden, patches * hidden),
        F32(hidden, sequence * hidden), F32(norm, sequence * hidden),
        F32(qkv, sequence * hidden * 3),
        F32(query, sequence * hidden), F32(key, sequence * hidden),
        F32(value, sequence * hidden), F32(attended, sequence * hidden),
        F32(branch, sequence * hidden), F32(ff1, sequence * ffn * 2),
        F32(activated, sequence * ffn),
        F32(projected, sequence * OUTPUT_PATCH)
    };
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        if (!all[index]) {
            fail(error, error_size, "cannot allocate image VAE buffers: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    return 1;
}

/* Decodes the tile whose de-normalized latent rows are in vae->latent into
 * vae->projected: one row of 3x16x16 values per latent position. */
static int run_tile(vae_context *vae, char *error, size_t error_size) {
    h3_gpu *gpu = vae->gpu;
    uint32_t patches = vae->patches, rows = vae->sequence;
    uint32_t hidden = vae->dim, heads = vae->heads, ffn = vae->ffn;
#define OP(call, label) do {                                                    \
    if (!(call)) {                                                              \
        fail(error, error_size, "%s: %s", label, h3_gpu_error(gpu));            \
        return 0;                                                               \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(gpu), "begin image VAE tile");
    OP(h3_gpu_linear_f32(gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "image VAE post-quant projection");
    OP(h3_gpu_linear_f32(gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, patches, LATENT_CHANNELS, hidden),
       "image VAE latent embedding");
    OP(h3_gpu_copy_f32(gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)patches * hidden), "pack image VAE patches");
    OP(h3_gpu_copy_f32(gpu, vae->hidden, (size_t)patches * hidden,
        vae->registers, 0, (size_t)REGISTERS * hidden),
       "pack image VAE registers");
    OP(h3_gpu_copy_f32(gpu, vae->hidden,
        (size_t)(patches + REGISTERS) * hidden, vae->zero, 0, hidden),
       "pack image VAE suffix");
    for (int index = 0; index < vae->layers; index++) {
        const vae_block *block = &vae->blocks[index];
        OP(h3_gpu_rms_norm_f32(gpu, vae->norm, vae->hidden, block->norm1,
            rows, hidden, 1e-5f), "image VAE attention norm");
        OP(h3_gpu_linear_f32(gpu, vae->qkv, vae->norm, block->qkv_w,
            block->qkv_b, rows, hidden, hidden * 3), "image VAE QKV");
        OP(h3_gpu_video_qkv_rope_f32(gpu, vae->query, vae->key, vae->value,
            vae->qkv, vae->rope_cos, vae->rope_sin, rows, heads, HEAD_DIM,
            ROPE_HALF, 1e-5f), "image VAE QK norm/RoPE");
        OP(h3_gpu_sdpa_f32(gpu, vae->attended, vae->query, vae->key,
            vae->value, rows, heads, HEAD_DIM,
            1.0f / sqrtf((float)HEAD_DIM)), "image VAE attention");
        OP(h3_gpu_linear_f32(gpu, vae->branch, vae->attended, block->out_w,
            block->out_b, rows, hidden, hidden), "image VAE attention output");
        OP(h3_gpu_scale_add_f32(gpu, vae->hidden, vae->hidden, vae->branch,
            block->scale1, rows, hidden), "image VAE attention residual");
        OP(h3_gpu_rms_norm_f32(gpu, vae->norm, vae->hidden, block->norm2,
            rows, hidden, 1e-5f), "image VAE MLP norm");
        OP(h3_gpu_linear_f32(gpu, vae->ff1, vae->norm, block->w1, block->w1_b,
            rows, hidden, ffn * 2), "image VAE MLP input");
        OP(h3_gpu_swiglu_f32(gpu, vae->activated, vae->ff1, rows, ffn),
           "image VAE SwiGLU");
        OP(h3_gpu_linear_f32(gpu, vae->branch, vae->activated, block->w2,
            block->w2_b, rows, ffn, hidden), "image VAE MLP output");
        OP(h3_gpu_scale_add_f32(gpu, vae->hidden, vae->hidden, vae->branch,
            block->scale2, rows, hidden), "image VAE MLP residual");
    }
    OP(h3_gpu_layer_norm_f32(gpu, vae->norm, vae->hidden, vae->norm_out_w,
        vae->norm_out_b, rows, hidden, 1e-5f), "image VAE output LayerNorm");
    OP(h3_gpu_linear_f32(gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, patches, hidden, OUTPUT_PATCH),
       "image VAE output projection");
    OP(h3_gpu_submit(gpu), "submit image VAE tile");
#undef OP
    return 1;
}

/* ---- Tiles ---- */

/* Where the tiles of one axis start (latent units): the fewest 16-wide
 * tiles overlapping by at least 4, the slack spread over the first
 * overlaps - the released decoder's rule. Returns the tile length. */
static int tile_starts(int size, int **starts, int *count) {
    if (size <= TILE_LATENT) {
        *count = 1;
        *starts = calloc(1, sizeof(**starts));
        return size;
    }
    int step = TILE_LATENT - TILE_OVERLAP_MIN;
    int tiles = (size - TILE_OVERLAP_MIN + step - 1) / step;
    int extra = TILE_LATENT * tiles - TILE_OVERLAP_MIN * (tiles - 1) - size;
    *count = tiles;
    *starts = calloc((size_t)tiles, sizeof(**starts));
    if (!*starts) return TILE_LATENT;
    for (int index = 1; index < tiles; index++) {
        int gap = step - extra / (tiles - 1) -
                  (index - 1 < extra % (tiles - 1));
        (*starts)[index] = (*starts)[index - 1] + gap;
    }
    return TILE_LATENT;
}

void h3_image_pixels_free(h3_image_pixels *pixels) {
    if (!pixels) return;
    free(pixels->rgb);
    memset(pixels, 0, sizeof(*pixels));
}

int h3_image_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent,
                        int latent_height, int latent_width,
                        const volatile int *cancel_requested,
                        h3_image_vae_progress progress, void *progress_opaque,
                        h3_image_pixels *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !normalized_latent ||
        !output || latent_height < 1 || latent_width < 1 ||
        latent_height > 4096 || latent_width > 4096) {
        fail(error, error_size, "invalid image VAE arguments");
        return 0;
    }
    float mean[LATENT_CHANNELS], deviation[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, mean, deviation, error,
                                   error_size)) return 0;

    int *ys = NULL, *xs = NULL, y_count = 0, x_count = 0;
    int tile_h = tile_starts(latent_height, &ys, &y_count);
    int tile_w = tile_starts(latent_width, &xs, &x_count);
    size_t pixel_h = (size_t)latent_height * PATCH;
    size_t pixel_w = (size_t)latent_width * PATCH;
    size_t tile_ph = (size_t)tile_h * PATCH, tile_pw = (size_t)tile_w * PATCH;
    size_t tile_patches = (size_t)tile_h * (size_t)tile_w;
    /* The decoder's own output, before it becomes RGB in [0,1]: tiles are
     * cross-faded in this space. */
    float *canvas = calloc(pixel_h * pixel_w * 3, sizeof(*canvas));
    float *strip = malloc(tile_ph * pixel_w * 3 * sizeof(*strip));
    float *tile = malloc(tile_ph * tile_pw * 3 * sizeof(*tile));
    float *rows = malloc(tile_patches * OUTPUT_PATCH * sizeof(*rows));
    float *latent = malloc(tile_patches * LATENT_CHANNELS * sizeof(*latent));
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    int ok = 0;
    if (!ys || !xs || !canvas || !strip || !tile || !rows || !latent) {
        fail(error, error_size, "out of memory decoding the image");
        goto done;
    }
    vae.tile_h = tile_h;
    vae.tile_w = tile_w;
    vae.patches = (uint32_t)tile_patches;
    vae.sequence = vae.patches + SUFFIX;
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!vae.weights || !read_shape(&vae, error, error_size)) goto done;
    vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!vae.gpu) goto done;
    h3_gpu_profile_set_label(vae.gpu, "image VAE decoder");
    if (!load_weights(&vae, error, error_size) ||
        !prepare_tile_buffers(&vae, error, error_size)) goto done;

    int total = y_count * x_count, completed = 0;
    for (int i = 0; i < y_count; i++) {
        size_t y0 = (size_t)ys[i];
        for (int j = 0; j < x_count; j++) {
            size_t x0 = (size_t)xs[j];
            if (cancel_requested && *cancel_requested) {
                fail(error, error_size, "generation cancelled");
                goto done;
            }
            /* The tile's latent rows, position-major, de-normalized. */
            for (size_t y = 0; y < (size_t)tile_h; y++)
                for (size_t x = 0; x < (size_t)tile_w; x++)
                    for (size_t channel = 0; channel < LATENT_CHANNELS; channel++)
                        latent[(y * (size_t)tile_w + x) * LATENT_CHANNELS + channel] =
                            normalized_latent[(channel * (size_t)latent_height +
                                               y0 + y) * (size_t)latent_width + x0 + x] *
                            deviation[channel] + mean[channel];
            if (!h3_gpu_tensor_write_f32(vae.latent, latent,
                                         tile_patches * LATENT_CHANNELS)) {
                fail(error, error_size, "cannot write the image VAE latent");
                goto done;
            }
            if (!run_tile(&vae, error, error_size)) goto done;
            if (!h3_gpu_tensor_read_f32(vae.projected, rows,
                                        tile_patches * OUTPUT_PATCH)) {
                fail(error, error_size, "cannot read image VAE output patches");
                goto done;
            }
            /* Patch rows [3, 16, 16] into the tile's pixels. */
            for (size_t y = 0; y < tile_ph; y++)
                for (size_t x = 0; x < tile_pw; x++) {
                    size_t patch = (y / PATCH) * (size_t)tile_w + x / PATCH;
                    const float *values = rows + patch * OUTPUT_PATCH;
                    for (size_t channel = 0; channel < 3; channel++) {
                        float value = values[(channel * PATCH + y % PATCH) * PATCH + x % PATCH];
                        if (!isfinite(value)) {
                            fail(error, error_size,
                                 "the image VAE produced a non-finite value");
                            goto done;
                        }
                        tile[(y * tile_pw + x) * 3 + channel] = value;
                    }
                }
            /* Fade in over what the row above and the tile to the left
             * already cover: from theirs at the tile's edge to its own. */
            if (i) {
                size_t overlap = ((size_t)(ys[i - 1] + tile_h) - y0) * PATCH;
                for (size_t y = 0; y < overlap; y++) {
                    float alpha = (float)y / (float)overlap;
                    const float *old = canvas + ((y0 * PATCH + y) * pixel_w + x0 * PATCH) * 3;
                    float *mine = tile + y * tile_pw * 3;
                    for (size_t value = 0; value < tile_pw * 3; value++)
                        mine[value] = old[value] + (mine[value] - old[value]) * alpha;
                }
            }
            if (j) {
                size_t overlap = ((size_t)(xs[j - 1] + tile_w) - x0) * PATCH;
                for (size_t y = 0; y < tile_ph; y++) {
                    const float *old = strip + (y * pixel_w + x0 * PATCH) * 3;
                    float *mine = tile + y * tile_pw * 3;
                    for (size_t x = 0; x < overlap; x++) {
                        float alpha = (float)x / (float)overlap;
                        for (size_t channel = 0; channel < 3; channel++)
                            mine[x * 3 + channel] = old[x * 3 + channel] +
                                (mine[x * 3 + channel] - old[x * 3 + channel]) * alpha;
                    }
                }
            }
            for (size_t y = 0; y < tile_ph; y++)
                memcpy(strip + (y * pixel_w + x0 * PATCH) * 3,
                       tile + y * tile_pw * 3, tile_pw * 3 * sizeof(*tile));
            if (progress) progress(++completed, total, progress_opaque);
        }
        memcpy(canvas + y0 * PATCH * pixel_w * 3, strip,
               tile_ph * pixel_w * 3 * sizeof(*strip));
    }
    /* ImageNet statistics back to RGB. */
    static const float rgb_mean[] = {0.485f, 0.456f, 0.406f};
    static const float rgb_std[] = {0.229f, 0.224f, 0.225f};
    for (size_t pixel = 0; pixel < pixel_h * pixel_w; pixel++)
        for (size_t channel = 0; channel < 3; channel++) {
            float value = canvas[pixel * 3 + channel] * rgb_std[channel] +
                          rgb_mean[channel];
            canvas[pixel * 3 + channel] = value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;
        }
    output->width = (int)pixel_w;
    output->height = (int)pixel_h;
    output->rgb = canvas;
    canvas = NULL;
    ok = h3_gpu_get_stats(vae.gpu, &output->gpu_stats);
    if (!ok) fail(error, error_size, "cannot read image VAE GPU statistics");

done:
    if (!ok) h3_image_pixels_free(output);
    free(ys); free(xs); free(canvas); free(strip); free(tile); free(rows);
    free(latent);
    cleanup(&vae);
    return ok;
}
