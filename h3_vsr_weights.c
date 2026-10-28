/* One-step VSR weight pipeline: reading a model directory, the .h3vc cache
 * (built once on the device from the downloaded F32 safetensors; the raw
 * weights are never run directly) and the constant conditioning that a
 * fixed timestep and a fixed prompt let us compute ahead of time.
 *
 * Cache layout (h3_vsr_internal.h): a 128-byte header, 64-byte aligned
 * payloads, then a 128-byte-per-entry name table. Entries per block b
 * (dim D, ffn F, text tokens S; W = int8 + f32 ".scales", or bf16):
 *   blocks.b.qkv.weight [3D, D] W   (+ .scales [3D])   blocks.b.qkv.bias [3D] f32
 *   blocks.b.attn1.norm_q [D] f32   blocks.b.attn1.norm_k [D] f32
 *   blocks.b.attn1.out.weight [D, D] W (+ .scales)      blocks.b.attn1.out.bias [D] f32
 *   blocks.b.attn2.q.weight [D, D] W (+ .scales)        blocks.b.attn2.q.bias [D] f32
 *   blocks.b.attn2.norm_q [D] f32
 *   blocks.b.attn2.k2 [S, D] bf16   blocks.b.attn2.v2 [S, D] bf16   (precomputed)
 *   blocks.b.attn2.out.weight [D, D] W (+ .scales)      blocks.b.attn2.out.bias [D] f32
 *   blocks.b.norm2.weight [D] f32   blocks.b.norm2.bias [D] f32
 *   blocks.b.ffn.fc1.weight [F, D] W (+ .scales)        blocks.b.ffn.fc1.bias [F] f32
 *   blocks.b.ffn.fc2.weight [D, F] W (+ .scales)        blocks.b.ffn.fc2.bias [D] f32
 *   blocks.b.mod [6, D] f32  = scale_shift_table + time_proj(silu(temb))
 *                              (shift, scale, gate, c_shift, c_scale, c_gate)
 * and once:
 *   patch_embedding.weight [D, 192] bf16   patch_embedding.bias [D] f32
 *   proj_out.weight [192, D] bf16          proj_out.bias [192] f32
 *   norm_out.mod [2, D] f32 = scale_shift_table + temb (shift, scale)
 *   temb [D] f32                           context [S, D] bf16
 * The ReAE (164 MB F32) is read from reae.safetensors at load time and is
 * only fingerprinted here. */
#include "h3_vsr_internal.h"
#include "h3_metal.h"
#include "h3_weights.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int join(char *out, size_t size, const char *a, const char *b) {
    return snprintf(out, size, "%s/%s", a, b) < (int)size;
}

/* ---------------------------------------------------------------- */
/* config.json */

static const char *json_find_key(const char *json, const char *key) {
    char quoted[128];
    snprintf(quoted, sizeof(quoted), "\"%s\"", key);
    const char *at = strstr(json, quoted);
    if (!at) return NULL;
    at += strlen(quoted);
    while (*at == ' ' || *at == '\t' || *at == '\n' || *at == '\r') at++;
    if (*at != ':') return NULL;
    at++;
    while (*at == ' ' || *at == '\t' || *at == '\n' || *at == '\r') at++;
    return at;
}

static int json_int(const char *json, const char *key, int *value) {
    const char *at = json_find_key(json, key);
    if (!at) return 0;
    char *end;
    long parsed = strtol(at, &end, 10);
    if (end == at) return 0;
    *value = (int)parsed;
    return 1;
}

static int json_bool(const char *json, const char *key, int *value) {
    const char *at = json_find_key(json, key);
    if (!at) return 0;
    if (!strncmp(at, "true", 4)) { *value = 1; return 1; }
    if (!strncmp(at, "false", 5)) { *value = 0; return 1; }
    return 0;
}

static int json_int_array3(const char *json, const char *key, int out[3]) {
    const char *at = json_find_key(json, key);
    if (!at || *at != '[') return 0;
    at++;
    for (int i = 0; i < 3; i++) {
        char *end;
        long parsed = strtol(at, &end, 10);
        if (end == at) return 0;
        out[i] = (int)parsed;
        at = end;
        while (*at == ' ' || *at == ',' || *at == '\n' || *at == '\r' || *at == '\t') at++;
    }
    return 1;
}

static char *read_text_file(const char *path, char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fail(error, error_size, "cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return NULL; }
    long end = ftell(file);
    if (end < 0 || end > (1 << 20) || fseek(file, 0, SEEK_SET) != 0) {
        fail(error, error_size, "%s is not a small text file", path);
        fclose(file);
        return NULL;
    }
    char *text = malloc((size_t)end + 1);
    if (!text || fread(text, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read %s", path);
        free(text);
        fclose(file);
        return NULL;
    }
    text[end] = '\0';
    fclose(file);
    return text;
}

int h3_vsr_read_wan_config(const char *model_dir, h3_vsr_wan_config *config,
                           char *error, size_t error_size) {
    if (!model_dir || !config) return 0;
    memset(config, 0, sizeof(*config));
    char path[4096];
    if (!join(path, sizeof(path), model_dir, "transformer/config.json")) return 0;
    char *json = read_text_file(path, error, error_size);
    if (!json) return 0;
    int patch[3] = {1, 2, 2};
    int ok = json_int(json, "num_layers", &config->layers) &&
             json_int(json, "num_attention_heads", &config->heads) &&
             json_int(json, "attention_head_dim", &config->head_dim) &&
             json_int(json, "ffn_dim", &config->ffn) &&
             json_int(json, "in_channels", &config->in_channels) &&
             json_int(json, "out_channels", &config->out_channels) &&
             json_int(json, "text_dim", &config->text_dim) &&
             json_int(json, "freq_dim", &config->freq_dim);
    if (ok && !json_int_array3(json, "patch_size", patch)) ok = 0;
    if (!json_bool(json, "cross_attn_norm", &config->cross_attn_norm))
        config->cross_attn_norm = 1;
    free(json);
    if (!ok) {
        fail(error, error_size, "%s lacks the WanTransformer3DModel fields", path);
        return 0;
    }
    config->dim = config->heads * config->head_dim;
    config->patch_t = patch[0];
    config->patch_h = patch[1];
    config->patch_w = patch[2];
    if (config->layers < 1 || config->dim < 1 || config->ffn < 1 ||
        config->in_channels < 1 || config->text_dim < 1 || config->freq_dim < 2 ||
        config->patch_t != 1) {
        fail(error, error_size, "unsupported Wan configuration in %s", path);
        return 0;
    }

    /* The prompt embedding gives the text token count. */
    if (!join(path, sizeof(path), model_dir, "prompt_embedding.safetensors")) return 0;
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return 0;
    const h3_st_tensor *prompt = h3_st_find(&header, "prompt_emb");
    if (!prompt || prompt->ndim != 3 || (int)prompt->shape[2] != config->text_dim) {
        fail(error, error_size, "%s has no prompt_emb [1, S, %d]", path,
             config->text_dim);
        h3_st_free_header(&header);
        return 0;
    }
    config->text_tokens = (int)prompt->shape[1];
    h3_st_free_header(&header);
    return 1;
}

/* ---------------------------------------------------------------- */
/* Inspection */

int h3_vsr_int8_supported(void) {
    h3_device_info device;
    char probe_error[256];
    return h3_metal_probe(&device, probe_error, sizeof(probe_error)) && device.apple_gpu_family >= 10;
}

int h3_vsr_inspect_dir(const char *model_dir, h3_vsr_info *info,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model_dir || !info) return 0;
    memset(info, 0, sizeof(*info));
    h3_vsr_wan_config config;
    if (!h3_vsr_read_wan_config(model_dir, &config, error, error_size)) return 0;
    char path[4096];
    if (!join(path, sizeof(path), model_dir, "reae.safetensors") || !file_exists(path)) {
        fail(error, error_size, "%s has no reae.safetensors (SwiftVR is the only model yet)",
             model_dir);
        return 0;
    }
    if (!join(path, sizeof(path), model_dir,
              "transformer/diffusion_pytorch_model.safetensors") || !file_exists(path)) {
        fail(error, error_size, "%s has no transformer/diffusion_pytorch_model.safetensors",
             model_dir);
        return 0;
    }
    info->kind = H3_VSR_SWIFTVR;
    snprintf(info->name, sizeof(info->name), "SwiftVR");
    info->fixed_factor_x100 = 0;
    info->output_multiple = H3_VSR_TOKEN_PIXELS;
    info->min_output_side = 4 * H3_VSR_TOKEN_PIXELS;
    info->max_output_side = 4096;
    info->first_chunk_frames = H3_VSR_CLIP_FRAMES + 4;
    info->chunk_frames = H3_VSR_CLIP_FRAMES;
    uint64_t d = (uint64_t)config.dim, f = (uint64_t)config.ffn;
    uint64_t block_params = 6 * d * d + 2 * d * f;
    uint64_t kv2 = 2 * (uint64_t)config.text_tokens * d;
    uint64_t once = 2 * d * (uint64_t)(config.in_channels * 4) +
                    (uint64_t)config.text_tokens * d;
    info->weight_bytes_bf16 = (block_params + kv2) * 2 * (uint64_t)config.layers + once * 2;
    info->weight_bytes_int8 = (block_params + kv2 * 2) * (uint64_t)config.layers + once * 2;
    /* Tensor hardware (Metal 4 TensorOps, MTLGPUFamilyApple10) without
     * creating a GPU context. */
    info->int8_supported = h3_vsr_int8_supported();
    snprintf(info->license_id, sizeof(info->license_id), "SwiftVR-Research-License");
    info->commercial_use_allowed = 0;
    return 1;
}

/* ---------------------------------------------------------------- */
/* Fingerprints and paths */

int h3_vsr_reae_fingerprint(const char *model_dir, uint8_t out[32]) {
    /* The model directory itself holds reae.safetensors and
     * prompt_embedding.safetensors (the transformer sits in transformer/),
     * so a store opened on it fingerprints exactly those two. */
    char error[256];
    h3_weight_store *store = h3_weight_store_open(model_dir, error, sizeof(error));
    if (!store) return 0;
    h3_weight_store_fingerprint(store, out);
    h3_weight_store_free(store);
    return 1;
}

int h3_vsr_cache_path(const char *model_dir, h3_vsr_precision precision,
                      char *path, size_t size) {
    const char *suffix = precision == H3_VSR_PRECISION_INT8 ? "int8" : "bf16";
    return snprintf(path, size, "%s/swiftvr-dit-%s.h3vc", model_dir, suffix) < (int)size;
}

/* ---------------------------------------------------------------- */
/* Constant conditioning (CPU) */

void h3_vsr_sinusoidal_embedding(double t, int freq_dim, float *out) {
    int half = freq_dim / 2;
    for (int i = 0; i < half; i++) {
        double frequency = exp(-log(10000.0) * (double)i / (double)half);
        double angle = t * frequency;
        out[i] = (float)cos(angle);          /* flip_sin_to_cos: cos first */
        out[half + i] = (float)sin(angle);
    }
}

void h3_vsr_silu_f32(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

void h3_vsr_gelu_tanh_f32(float *x, size_t n) {
    const float k = 0.7978845608028654f; /* sqrt(2/pi) */
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
    }
}

void h3_vsr_linear_f32(const float *in, int rows, int in_dim,
                       const float *weight, const float *bias, int out_dim,
                       float *out) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out_dim, in_dim,
                1.0f, in, in_dim, weight, in_dim, 0.0f, out, out_dim);
    if (bias)
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < out_dim; c++) out[(size_t)r * out_dim + c] += bias[c];
}

void h3_vsr_rms_norm_rows_f32(float *x, int rows, int dim,
                              const float *weight, float eps) {
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * dim;
        double sum = 0.0;
        for (int c = 0; c < dim; c++) sum += (double)row[c] * row[c];
        float scale = (float)(1.0 / sqrt(sum / dim + (double)eps));
        for (int c = 0; c < dim; c++) row[c] = row[c] * scale * (weight ? weight[c] : 1.0f);
    }
}

/* ---------------------------------------------------------------- */
/* Reading F32 host copies of store tensors (through the GPU converter,
 * which also accepts BF16 and F16 files). */

static float *load_host_f32(const h3_weight_store *store, h3_gpu *gpu,
                            const char *name, int ndim, const uint64_t *shape,
                            char *error, size_t error_size) {
    h3_gpu_tensor *tensor = h3_weight_load_f32(store, gpu, name, ndim, shape,
                                               error, error_size);
    if (!tensor) return NULL;
    size_t elements = h3_gpu_tensor_elements(tensor);
    float *host = malloc(elements * sizeof(float));
    if (!host || !h3_gpu_tensor_read_f32(tensor, host, elements)) {
        fail(error, error_size, "cannot read %s", name);
        free(host);
        host = NULL;
    }
    h3_gpu_tensor_free(tensor);
    return host;
}

static float *load_host_f32_2d(const h3_weight_store *store, h3_gpu *gpu,
                               const char *name, uint64_t rows, uint64_t cols,
                               char *error, size_t error_size) {
    uint64_t shape[2] = {rows, cols};
    return load_host_f32(store, gpu, name, 2, shape, error, error_size);
}

static float *load_host_f32_1d(const h3_weight_store *store, h3_gpu *gpu,
                               const char *name, uint64_t n,
                               char *error, size_t error_size) {
    uint64_t shape[1] = {n};
    return load_host_f32(store, gpu, name, 1, shape, error, error_size);
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;   /* round to nearest even */
    return (uint16_t)(bits >> 16);
}

/* ---------------------------------------------------------------- */
/* Cache writer */

typedef struct {
    FILE *file;
    h3_vsr_cache_entry *entries;
    size_t count, capacity;
    uint64_t offset;
    char *error;
    size_t error_size;
} cache_writer;

static int writer_pad(cache_writer *w) {
    static const uint8_t zeros[H3_VSR_CACHE_ALIGN] = {0};
    uint64_t pad = (H3_VSR_CACHE_ALIGN - w->offset % H3_VSR_CACHE_ALIGN) % H3_VSR_CACHE_ALIGN;
    if (pad && fwrite(zeros, 1, pad, w->file) != pad) return 0;
    w->offset += pad;
    return 1;
}

static int writer_entry(cache_writer *w, const char *name, h3_vsr_cache_dtype dtype,
                        uint32_t rows, uint32_t cols, const void *data, uint64_t bytes) {
    if (!writer_pad(w)) goto io_error;
    if (w->count == w->capacity) {
        size_t capacity = w->capacity ? w->capacity * 2 : 256;
        h3_vsr_cache_entry *grown = realloc(w->entries, capacity * sizeof(*grown));
        if (!grown) {
            fail(w->error, w->error_size, "out of memory");
            return 0;
        }
        w->entries = grown;
        w->capacity = capacity;
    }
    h3_vsr_cache_entry *entry = &w->entries[w->count];
    memset(entry, 0, sizeof(*entry));
    if (strlen(name) >= H3_VSR_CACHE_NAME) {
        fail(w->error, w->error_size, "cache entry name too long: %s", name);
        return 0;
    }
    strcpy(entry->name, name);
    entry->dtype = (uint32_t)dtype;
    entry->rows = rows;
    entry->cols = cols;
    entry->offset = w->offset;
    entry->bytes = bytes;
    if (bytes && fwrite(data, 1, bytes, w->file) != bytes) goto io_error;
    w->offset += bytes;
    w->count++;
    return 1;
io_error:
    fail(w->error, w->error_size, "cannot write cache payload for %s: %s", name,
         strerror(errno));
    return 0;
}

static int writer_f32(cache_writer *w, const char *name, uint32_t rows, uint32_t cols,
                      const float *data) {
    return writer_entry(w, name, H3_VSR_CACHE_F32, rows, cols, data,
                        (uint64_t)rows * cols * sizeof(float));
}

static int writer_bf16_from_f32(cache_writer *w, const char *name, uint32_t rows,
                                uint32_t cols, const float *data) {
    size_t n = (size_t)rows * cols;
    uint16_t *bf16 = malloc(n * sizeof(uint16_t));
    if (!bf16) {
        fail(w->error, w->error_size, "out of memory");
        return 0;
    }
    for (size_t i = 0; i < n; i++) bf16[i] = f32_to_bf16(data[i]);
    int ok = writer_entry(w, name, H3_VSR_CACHE_BF16, rows, cols, bf16, n * 2);
    free(bf16);
    return ok;
}

/* ---------------------------------------------------------------- */
/* Builder */

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *store;   /* transformer/ */
    h3_weight_store *top;     /* the model directory: prompt_emb, reae */
    const h3_vsr_wan_config *config;
    int int8;
    cache_writer *writer;
    char *error;
    size_t error_size;
} builder;

/* Reads one or more [rows_i, cols] weights into a fused BF16 GPU tensor
 * (rows stacked), quantizes it to int8 if asked, and writes the cache
 * entries `name.weight` (+ `.scales`). */
static int write_weight(builder *b, const char *entry_name, const char **sources,
                        int source_count, uint32_t rows_each, uint32_t cols) {
    uint32_t rows = rows_each * (uint32_t)source_count;
    size_t elements = (size_t)rows * cols;
    char name[H3_VSR_CACHE_NAME];
    int ok = 0;
    h3_gpu_tensor *fused = h3_gpu_tensor_new_bf16(b->gpu, elements);
    uint16_t *host = malloc((size_t)rows_each * cols * sizeof(uint16_t));
    if (!fused || !host) {
        fail(b->error, b->error_size, "out of memory for %s", entry_name);
        goto done;
    }
    for (int s = 0; s < source_count; s++) {
        uint64_t shape[2] = {rows_each, cols};
        h3_gpu_tensor *part = h3_weight_load_bf16(b->store, b->gpu, sources[s], 2, shape,
                                                  b->error, b->error_size);
        if (!part) goto done;
        int read_ok = h3_gpu_tensor_read_bf16(part, host, (size_t)rows_each * cols) &&
                      h3_gpu_tensor_write_bf16_range(fused, (size_t)s * rows_each * cols,
                                                     host, (size_t)rows_each * cols);
        h3_gpu_tensor_free(part);
        if (!read_ok) {
            fail(b->error, b->error_size, "cannot stage %s", sources[s]);
            goto done;
        }
    }
    snprintf(name, sizeof(name), "%s.weight", entry_name);
    if (!b->int8) {
        uint16_t *all = malloc(elements * sizeof(uint16_t));
        ok = all && h3_gpu_tensor_read_bf16(fused, all, elements) &&
             writer_entry(b->writer, name, H3_VSR_CACHE_BF16, rows, cols, all, elements * 2);
        if (!ok && b->error && !b->error[0])
            fail(b->error, b->error_size, "cannot write %s", name);
        free(all);
        goto done;
    }
    h3_gpu_tensor *i8 = h3_gpu_tensor_new_i8(b->gpu, elements);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(b->gpu, rows);
    int8_t *i8_host = malloc(elements);
    float *scale_host = malloc((size_t)rows * sizeof(float));
    ok = i8 && scales && i8_host && scale_host &&
         h3_gpu_begin(b->gpu) &&
         h3_gpu_quantize_weight_int8(b->gpu, i8, scales, fused, rows, cols) &&
         h3_gpu_submit(b->gpu) &&
         h3_gpu_tensor_read_i8(i8, i8_host, elements) &&
         h3_gpu_tensor_read_f32(scales, scale_host, rows);
    if (!ok) {
        fail(b->error, b->error_size, "cannot quantize %s: %s", entry_name,
             h3_gpu_error(b->gpu));
    } else {
        ok = writer_entry(b->writer, name, H3_VSR_CACHE_I8, rows, cols, i8_host, elements);
        snprintf(name, sizeof(name), "%s.scales", entry_name);
        ok = ok && writer_f32(b->writer, name, rows, 1, scale_host);
    }
    free(i8_host);
    free(scale_host);
    h3_gpu_tensor_free(i8);
    h3_gpu_tensor_free(scales);
done:
    free(host);
    h3_gpu_tensor_free(fused);
    return ok;
}

static int write_vector_f32(builder *b, const char *source, const char *entry_name,
                            uint32_t n) {
    float *host = load_host_f32_1d(b->store, b->gpu, source, n, b->error, b->error_size);
    if (!host) return 0;
    int ok = writer_f32(b->writer, entry_name, n, 1, host);
    free(host);
    return ok;
}

/* temb [D], time_proj(silu(temb)) [6D], context [S, D] (after the text
 * embedder), all F32 host arrays owned by the caller. */
typedef struct {
    float *temb;
    float *time_proj;
    float *context;
} constants;

static void constants_free(constants *c) {
    free(c->temb);
    free(c->time_proj);
    free(c->context);
    memset(c, 0, sizeof(*c));
}

static int compute_constants(builder *b, constants *out) {
    const h3_vsr_wan_config *cfg = b->config;
    int D = cfg->dim, S = cfg->text_tokens, T = cfg->text_dim, F = cfg->freq_dim;
    memset(out, 0, sizeof(*out));
    float *emb = malloc((size_t)F * sizeof(float));
    float *hidden = malloc((size_t)D * sizeof(float));
    out->temb = malloc((size_t)D * sizeof(float));
    out->time_proj = malloc((size_t)6 * D * sizeof(float));
    float *text_hidden = malloc((size_t)S * D * sizeof(float));
    out->context = malloc((size_t)S * D * sizeof(float));
    if (!emb || !hidden || !out->temb || !out->time_proj || !text_hidden || !out->context) {
        fail(b->error, b->error_size, "out of memory for the constants");
        goto fail_free;
    }

    /* Time: sinusoidal(1000) -> linear_1 -> silu -> linear_2 = temb;
     * time_proj(silu(temb)) = the 6 modulation rows. */
    h3_vsr_sinusoidal_embedding(1000.0, F, emb);
    float *w1 = load_host_f32_2d(b->store, b->gpu, "condition_embedder.time_embedder.linear_1.weight", D, F, b->error, b->error_size);
    float *b1 = load_host_f32_1d(b->store, b->gpu, "condition_embedder.time_embedder.linear_1.bias", D, b->error, b->error_size);
    float *w2 = load_host_f32_2d(b->store, b->gpu, "condition_embedder.time_embedder.linear_2.weight", D, D, b->error, b->error_size);
    float *b2 = load_host_f32_1d(b->store, b->gpu, "condition_embedder.time_embedder.linear_2.bias", D, b->error, b->error_size);
    float *wp = load_host_f32_2d(b->store, b->gpu, "condition_embedder.time_proj.weight", 6 * D, D, b->error, b->error_size);
    float *bp = load_host_f32_1d(b->store, b->gpu, "condition_embedder.time_proj.bias", 6 * D, b->error, b->error_size);
    int ok = w1 && b1 && w2 && b2 && wp && bp;
    if (ok) {
        h3_vsr_linear_f32(emb, 1, F, w1, b1, D, hidden);
        h3_vsr_silu_f32(hidden, (size_t)D);
        h3_vsr_linear_f32(hidden, 1, D, w2, b2, D, out->temb);
        memcpy(hidden, out->temb, (size_t)D * sizeof(float));
        h3_vsr_silu_f32(hidden, (size_t)D);
        h3_vsr_linear_f32(hidden, 1, D, wp, bp, 6 * D, out->time_proj);
    }
    free(w1); free(b1); free(w2); free(b2); free(wp); free(bp);
    if (!ok) goto fail_free;

    /* Text: prompt_emb [S, T] -> linear_1 -> gelu(tanh) -> linear_2. */
    float *prompt = NULL;
    {
        uint64_t shape[3] = {1, (uint64_t)S, (uint64_t)T};
        h3_gpu_tensor *tensor = h3_weight_load_f32(b->top, b->gpu, "prompt_emb", 3, shape,
                                                   b->error, b->error_size);
        if (!tensor) goto fail_free;
        prompt = malloc((size_t)S * T * sizeof(float));
        ok = prompt && h3_gpu_tensor_read_f32(tensor, prompt, (size_t)S * T);
        h3_gpu_tensor_free(tensor);
        if (!ok) {
            fail(b->error, b->error_size, "cannot read prompt_emb");
            free(prompt);
            goto fail_free;
        }
    }
    float *tw1 = load_host_f32_2d(b->store, b->gpu, "condition_embedder.text_embedder.linear_1.weight", D, T, b->error, b->error_size);
    float *tb1 = load_host_f32_1d(b->store, b->gpu, "condition_embedder.text_embedder.linear_1.bias", D, b->error, b->error_size);
    float *tw2 = load_host_f32_2d(b->store, b->gpu, "condition_embedder.text_embedder.linear_2.weight", D, D, b->error, b->error_size);
    float *tb2 = load_host_f32_1d(b->store, b->gpu, "condition_embedder.text_embedder.linear_2.bias", D, b->error, b->error_size);
    ok = tw1 && tb1 && tw2 && tb2;
    if (ok) {
        h3_vsr_linear_f32(prompt, S, T, tw1, tb1, D, text_hidden);
        h3_vsr_gelu_tanh_f32(text_hidden, (size_t)S * D);
        h3_vsr_linear_f32(text_hidden, S, D, tw2, tb2, D, out->context);
    }
    free(prompt); free(tw1); free(tb1); free(tw2); free(tb2);
    if (!ok) goto fail_free;
    free(emb); free(hidden); free(text_hidden);
    return 1;
fail_free:
    free(emb); free(hidden); free(text_hidden);
    constants_free(out);
    return 0;
}

static int write_block(builder *b, int block, const constants *c) {
    const h3_vsr_wan_config *cfg = b->config;
    uint32_t D = (uint32_t)cfg->dim, Fd = (uint32_t)cfg->ffn, S = (uint32_t)cfg->text_tokens;
    char q[160], k[160], v[160], e[H3_VSR_CACHE_NAME], e2[H3_VSR_CACHE_NAME];
    const char *sources[3];
#define SRC(buffer, suffix) (snprintf(buffer, sizeof(buffer), "blocks.%d.%s", block, suffix), buffer)
#define ENT(buffer, suffix) (snprintf(buffer, sizeof(buffer), "blocks.%d.%s", block, suffix), buffer)

    /* Self-attention: fused QKV weight and bias. */
    sources[0] = SRC(q, "attn1.to_q.weight");
    sources[1] = SRC(k, "attn1.to_k.weight");
    sources[2] = SRC(v, "attn1.to_v.weight");
    if (!write_weight(b, ENT(e, "qkv"), sources, 3, D, D)) return 0;
    {
        float *bias = malloc((size_t)3 * D * sizeof(float));
        if (!bias) return 0;
        const char *names[3] = {"attn1.to_q.bias", "attn1.to_k.bias", "attn1.to_v.bias"};
        for (int i = 0; i < 3; i++) {
            float *part = load_host_f32_1d(b->store, b->gpu, SRC(q, names[i]), D, b->error, b->error_size);
            if (!part) { free(bias); return 0; }
            memcpy(bias + (size_t)i * D, part, (size_t)D * sizeof(float));
            free(part);
        }
        int ok = writer_f32(b->writer, ENT(e, "qkv.bias"), 3 * D, 1, bias);
        free(bias);
        if (!ok) return 0;
    }
    if (!write_vector_f32(b, SRC(q, "attn1.norm_q.weight"), ENT(e, "attn1.norm_q"), D)) return 0;
    if (!write_vector_f32(b, SRC(q, "attn1.norm_k.weight"), ENT(e, "attn1.norm_k"), D)) return 0;
    sources[0] = SRC(q, "attn1.to_out.0.weight");
    if (!write_weight(b, ENT(e, "attn1.out"), sources, 1, D, D)) return 0;
    if (!write_vector_f32(b, SRC(q, "attn1.to_out.0.bias"), ENT(e, "attn1.out.bias"), D)) return 0;

    /* Cross-attention: query side as weights, key/value precomputed from
     * the constant context (RMSNorm_k applied to the keys). */
    sources[0] = SRC(q, "attn2.to_q.weight");
    if (!write_weight(b, ENT(e, "attn2.q"), sources, 1, D, D)) return 0;
    if (!write_vector_f32(b, SRC(q, "attn2.to_q.bias"), ENT(e, "attn2.q.bias"), D)) return 0;
    if (!write_vector_f32(b, SRC(q, "attn2.norm_q.weight"), ENT(e, "attn2.norm_q"), D)) return 0;
    {
        float *wk = load_host_f32_2d(b->store, b->gpu, SRC(q, "attn2.to_k.weight"), D, D, b->error, b->error_size);
        float *bk = load_host_f32_1d(b->store, b->gpu, SRC(k, "attn2.to_k.bias"), D, b->error, b->error_size);
        float *nk = load_host_f32_1d(b->store, b->gpu, SRC(v, "attn2.norm_k.weight"), D, b->error, b->error_size);
        float *k2 = malloc((size_t)S * D * sizeof(float));
        int ok = wk && bk && nk && k2;
        if (ok) {
            h3_vsr_linear_f32(c->context, (int)S, (int)D, wk, bk, (int)D, k2);
            h3_vsr_rms_norm_rows_f32(k2, (int)S, (int)D, nk, 1e-6f);
            ok = writer_bf16_from_f32(b->writer, ENT(e, "attn2.k2"), S, D, k2);
        }
        free(wk); free(bk); free(nk); free(k2);
        if (!ok) return 0;
        float *wv = load_host_f32_2d(b->store, b->gpu, SRC(q, "attn2.to_v.weight"), D, D, b->error, b->error_size);
        float *bv = load_host_f32_1d(b->store, b->gpu, SRC(k, "attn2.to_v.bias"), D, b->error, b->error_size);
        float *v2 = malloc((size_t)S * D * sizeof(float));
        ok = wv && bv && v2;
        if (ok) {
            h3_vsr_linear_f32(c->context, (int)S, (int)D, wv, bv, (int)D, v2);
            ok = writer_bf16_from_f32(b->writer, ENT(e, "attn2.v2"), S, D, v2);
        }
        free(wv); free(bv); free(v2);
        if (!ok) return 0;
    }
    sources[0] = SRC(q, "attn2.to_out.0.weight");
    if (!write_weight(b, ENT(e, "attn2.out"), sources, 1, D, D)) return 0;
    if (!write_vector_f32(b, SRC(q, "attn2.to_out.0.bias"), ENT(e, "attn2.out.bias"), D)) return 0;
    if (cfg->cross_attn_norm) {
        if (!write_vector_f32(b, SRC(q, "norm2.weight"), ENT(e, "norm2.weight"), D)) return 0;
        if (!write_vector_f32(b, SRC(q, "norm2.bias"), ENT(e, "norm2.bias"), D)) return 0;
    }

    /* FFN. */
    sources[0] = SRC(q, "ffn.net.0.proj.weight");
    if (!write_weight(b, ENT(e, "ffn.fc1"), sources, 1, Fd, D)) return 0;
    if (!write_vector_f32(b, SRC(q, "ffn.net.0.proj.bias"), ENT(e, "ffn.fc1.bias"), Fd)) return 0;
    sources[0] = SRC(q, "ffn.net.2.weight");
    if (!write_weight(b, ENT(e, "ffn.fc2"), sources, 1, D, Fd)) return 0;
    if (!write_vector_f32(b, SRC(q, "ffn.net.2.bias"), ENT(e, "ffn.fc2.bias"), D)) return 0;

    /* Modulation: scale_shift_table [1, 6, D] + time_proj [6, D]. */
    {
        uint64_t shape[3] = {1, 6, D};
        float *table = load_host_f32(b->store, b->gpu, SRC(q, "scale_shift_table"), 3, shape,
                                     b->error, b->error_size);
        if (!table) return 0;
        for (size_t i = 0; i < (size_t)6 * D; i++) table[i] += c->time_proj[i];
        int ok = writer_f32(b->writer, ENT(e2, "mod"), 6, D, table);
        free(table);
        if (!ok) return 0;
    }
#undef SRC
#undef ENT
    return 1;
}

static int write_once(builder *b, const constants *c) {
    const h3_vsr_wan_config *cfg = b->config;
    uint32_t D = (uint32_t)cfg->dim, S = (uint32_t)cfg->text_tokens;
    uint32_t patch_in = (uint32_t)(cfg->in_channels * cfg->patch_h * cfg->patch_w);
    uint32_t patch_out = (uint32_t)(cfg->out_channels * cfg->patch_t * cfg->patch_h * cfg->patch_w);
    {
        /* Conv3d [D, C, 1, ph, pw] is the linear [D, C*ph*pw] in (c, ph, pw) order. */
        uint64_t shape[5] = {D, (uint64_t)cfg->in_channels, 1, (uint64_t)cfg->patch_h, (uint64_t)cfg->patch_w};
        float *w = load_host_f32(b->store, b->gpu, "patch_embedding.weight", 5, shape, b->error, b->error_size);
        if (!w) return 0;
        int ok = writer_bf16_from_f32(b->writer, "patch_embedding.weight", D, patch_in, w);
        free(w);
        if (!ok) return 0;
        if (!write_vector_f32(b, "patch_embedding.bias", "patch_embedding.bias", D)) return 0;
    }
    {
        float *w = load_host_f32_2d(b->store, b->gpu, "proj_out.weight", patch_out, D, b->error, b->error_size);
        if (!w) return 0;
        int ok = writer_bf16_from_f32(b->writer, "proj_out.weight", patch_out, D, w);
        free(w);
        if (!ok) return 0;
        if (!write_vector_f32(b, "proj_out.bias", "proj_out.bias", patch_out)) return 0;
    }
    {
        uint64_t shape[3] = {1, 2, D};
        float *table = load_host_f32(b->store, b->gpu, "scale_shift_table", 3, shape, b->error, b->error_size);
        if (!table) return 0;
        for (size_t i = 0; i < (size_t)2 * D; i++) table[i] += c->temb[i % D];
        int ok = writer_f32(b->writer, "norm_out.mod", 2, D, table);
        free(table);
        if (!ok) return 0;
    }
    if (!writer_f32(b->writer, "temb", D, 1, c->temb)) return 0;
    if (!writer_bf16_from_f32(b->writer, "context", S, D, c->context)) return 0;
    return 1;
}

int h3_vsr_build_cache(const char *model_dir, const char *cache_path,
                       h3_vsr_precision precision,
                       const char *shader_source_path,
                       h3_progress_callback on_progress, void *opaque,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model_dir || !cache_path) {
        fail(error, error_size, "invalid VSR cache build arguments");
        return 0;
    }
    h3_vsr_wan_config config;
    if (!h3_vsr_read_wan_config(model_dir, &config, error, error_size)) return 0;

    h3_gpu *gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!gpu) return 0;
    int int8 = precision == H3_VSR_PRECISION_INT8 ||
               (precision == H3_VSR_PRECISION_AUTO && h3_gpu_has_int8_mlp(gpu));
    if (int8 && !h3_gpu_has_int8_mlp(gpu)) {
        fail(error, error_size, "this GPU lacks the int8 tensor-op path (needs an M5-class GPU)");
        h3_gpu_free(gpu);
        return 0;
    }
    if (int8 && (config.dim % 128 || config.ffn % 128)) {
        fail(error, error_size, "int8 needs dim and ffn to be multiples of 128");
        h3_gpu_free(gpu);
        return 0;
    }

    char transformer_dir[4096];
    if (!join(transformer_dir, sizeof(transformer_dir), model_dir, "transformer")) {
        h3_gpu_free(gpu);
        return 0;
    }
    /* One store over transformer/ (the DiT) and one over the model
     * directory (prompt_emb, reae). */
    h3_weight_store *store = h3_weight_store_open(transformer_dir, error, error_size);
    h3_weight_store *top = store ? h3_weight_store_open(model_dir, error, error_size) : NULL;
    if (!store || !top) {
        h3_weight_store_free(store);
        h3_weight_store_free(top);
        h3_gpu_free(gpu);
        return 0;
    }

    char partial[4096];
    snprintf(partial, sizeof(partial), "%s.partial", cache_path);
    FILE *file = fopen(partial, "wb");
    if (!file) {
        fail(error, error_size, "cannot open %s: %s", partial, strerror(errno));
        h3_weight_store_free(store);
        h3_weight_store_free(top);
        h3_gpu_free(gpu);
        return 0;
    }

    cache_writer writer = {file, NULL, 0, 0, 0, error, error_size};
    h3_vsr_cache_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, H3_VSR_CACHE_MAGIC, 4);
    header.version = H3_VSR_CACHE_VERSION;
    header.model_kind = H3_VSR_SWIFTVR;
    header.precision = int8 ? H3_VSR_PRECISION_INT8 : H3_VSR_PRECISION_BF16;
    header.blocks = (uint32_t)config.layers;
    header.dim = (uint32_t)config.dim;
    header.ffn = (uint32_t)config.ffn;
    header.heads = (uint32_t)config.heads;
    header.head_dim = (uint32_t)config.head_dim;
    header.text_tokens = (uint32_t)config.text_tokens;
    header.patch_in = (uint32_t)(config.in_channels * config.patch_h * config.patch_w);
    header.latent_channels = (uint32_t)config.in_channels;
    h3_weight_store_fingerprint(store, header.fingerprint);
    h3_weight_store_fingerprint(top, header.reae_fingerprint);
    int ok = fwrite(&header, sizeof(header), 1, file) == 1;
    writer.offset = sizeof(header);
    if (!ok) fail(error, error_size, "cannot write cache header: %s", strerror(errno));

    builder b = {gpu, store, top, &config, int8, &writer, error, error_size};
    constants c = {0};
    if (ok) ok = compute_constants(&b, &c);
    if (ok && on_progress && on_progress("vsr cache", 0, config.layers + 1, opaque)) {
        fail(error, error_size, "vsr cache build cancelled");
        ok = 0;
    }
    for (int block = 0; ok && block < config.layers; block++) {
        if (error) error[0] = '\0';
        ok = write_block(&b, block, &c);
        if (ok && on_progress &&
            on_progress("vsr cache", block + 1, config.layers + 1, opaque)) {
            fail(error, error_size, "vsr cache build cancelled");
            ok = 0;
        }
    }
    if (ok) ok = write_once(&b, &c);
    if (ok) {
        ok = writer_pad(&writer);
        header.table_offset = writer.offset;
        header.table_count = (uint32_t)writer.count;
        ok = ok && fwrite(writer.entries, sizeof(h3_vsr_cache_entry), writer.count, file) == writer.count &&
             fseek(file, 0, SEEK_SET) == 0 &&
             fwrite(&header, sizeof(header), 1, file) == 1;
        if (!ok) fail(error, error_size, "cannot finish %s: %s", partial, strerror(errno));
    }
    if (ok && on_progress &&
        on_progress("vsr cache", config.layers + 1, config.layers + 1, opaque)) {
        fail(error, error_size, "vsr cache build cancelled");
        ok = 0;
    }
    constants_free(&c);
    free(writer.entries);
    fclose(file);
    h3_weight_store_free(store);
    h3_weight_store_free(top);
    h3_gpu_free(gpu);
    if (!ok) {
        unlink(partial);
        return 0;
    }
    if (rename(partial, cache_path) != 0) {
        fail(error, error_size, "cannot rename %s: %s", partial, strerror(errno));
        unlink(partial);
        return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------- */
/* Reader */

int h3_vsr_cache_open(const char *path, h3_vsr_cache *cache,
                      char *error, size_t error_size) {
    if (!path || !cache) return 0;
    memset(cache, 0, sizeof(*cache));
    FILE *file = fopen(path, "rb");
    if (!file) {
        fail(error, error_size, "cannot open %s: %s", path, strerror(errno));
        return 0;
    }
    int ok = fread(&cache->header, sizeof(cache->header), 1, file) == 1 &&
             memcmp(cache->header.magic, H3_VSR_CACHE_MAGIC, 4) == 0 &&
             cache->header.version == H3_VSR_CACHE_VERSION &&
             cache->header.table_count > 0 && cache->header.table_count < 100000;
    if (!ok) {
        fail(error, error_size, "%s is not a VSR cache this engine reads", path);
        fclose(file);
        return 0;
    }
    cache->count = cache->header.table_count;
    cache->entries = malloc(cache->count * sizeof(h3_vsr_cache_entry));
    cache->path = strdup(path);
    ok = cache->entries && cache->path &&
         fseeko(file, (off_t)cache->header.table_offset, SEEK_SET) == 0 &&
         fread(cache->entries, sizeof(h3_vsr_cache_entry), cache->count, file) == cache->count;
    fclose(file);
    if (!ok) {
        fail(error, error_size, "cannot read the name table of %s", path);
        h3_vsr_cache_close(cache);
        return 0;
    }
    return 1;
}

void h3_vsr_cache_close(h3_vsr_cache *cache) {
    if (!cache) return;
    free(cache->entries);
    free(cache->path);
    memset(cache, 0, sizeof(*cache));
}

const h3_vsr_cache_entry *h3_vsr_cache_find(const h3_vsr_cache *cache,
                                            const char *name) {
    if (!cache || !name) return NULL;
    for (size_t i = 0; i < cache->count; i++)
        if (strncmp(cache->entries[i].name, name, H3_VSR_CACHE_NAME) == 0)
            return &cache->entries[i];
    return NULL;
}

h3_gpu_tensor *h3_vsr_cache_load(const h3_vsr_cache *cache, h3_gpu *gpu,
                                 const char *name, char *error,
                                 size_t error_size) {
    const h3_vsr_cache_entry *entry = h3_vsr_cache_find(cache, name);
    if (!entry) {
        fail(error, error_size, "%s has no entry %s", cache->path, name);
        return NULL;
    }
    size_t elements = (size_t)entry->rows * entry->cols;
    h3_gpu_tensor *tensor = NULL;
    int ok = 0;
    switch (entry->dtype) {
    case H3_VSR_CACHE_F32:
        tensor = h3_gpu_tensor_new_f32(gpu, elements);
        ok = tensor && h3_gpu_tensor_stream_file_f32(tensor, cache->path, entry->offset,
                                                     elements, error, error_size);
        break;
    case H3_VSR_CACHE_BF16:
        tensor = h3_gpu_tensor_new_bf16(gpu, elements);
        ok = tensor && h3_gpu_tensor_stream_file_bf16(tensor, cache->path, entry->offset,
                                                      elements, error, error_size);
        break;
    case H3_VSR_CACHE_I8:
        tensor = h3_gpu_tensor_new_i8(gpu, elements);
        ok = tensor && h3_gpu_tensor_stream_file_i8(tensor, cache->path, entry->offset,
                                                    elements, error, error_size);
        break;
    default:
        fail(error, error_size, "%s: unknown dtype for %s", cache->path, name);
    }
    if (!ok) {
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

int h3_vsr_cache_read_f32(const h3_vsr_cache *cache, const char *name,
                          float *values, size_t elements,
                          char *error, size_t error_size) {
    const h3_vsr_cache_entry *entry = h3_vsr_cache_find(cache, name);
    if (!entry || entry->dtype != H3_VSR_CACHE_F32 ||
        (size_t)entry->rows * entry->cols != elements) {
        fail(error, error_size, "%s: no F32 entry %s of %zu values", cache->path, name, elements);
        return 0;
    }
    int fd = open(cache->path, O_RDONLY);
    if (fd < 0) {
        fail(error, error_size, "cannot open %s: %s", cache->path, strerror(errno));
        return 0;
    }
    ssize_t got = pread(fd, values, elements * sizeof(float), (off_t)entry->offset);
    close(fd);
    if (got != (ssize_t)(elements * sizeof(float))) {
        fail(error, error_size, "cannot read %s from %s", name, cache->path);
        return 0;
    }
    return 1;
}

int h3_vsr_cache_matches(const char *cache_path, const char *model_dir) {
    h3_vsr_cache cache;
    char error[256];
    if (!h3_vsr_cache_open(cache_path, &cache, error, sizeof(error))) return 0;
    char transformer_dir[4096];
    int ok = join(transformer_dir, sizeof(transformer_dir), model_dir, "transformer");
    uint8_t fingerprint[32] = {0}, reae[32] = {0};
    if (ok) {
        h3_weight_store *store = h3_weight_store_open(transformer_dir, error, sizeof(error));
        ok = store != NULL;
        if (store) h3_weight_store_fingerprint(store, fingerprint);
        h3_weight_store_free(store);
    }
    ok = ok && h3_vsr_reae_fingerprint(model_dir, reae) &&
         memcmp(fingerprint, cache.header.fingerprint, 32) == 0 &&
         memcmp(reae, cache.header.reae_fingerprint, 32) == 0;
    h3_vsr_cache_close(&cache);
    return ok;
}
