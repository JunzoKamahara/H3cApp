/* SeedVR2-3B on the engine's GPU primitives: see h3_seedvr2.h. Reference:
 * mflux src/mflux/models/seedvr2 (MIT); weights: mlx-community/
 * SeedVR2-3B-mlx-int8 (MLX affine 8-bit export of ByteDance's Apache-2.0
 * checkpoint). Numbers and names below are the 3B configuration. */
#include "h3_seedvr2.h"
#include "h3_gpu.h"
#include "h3_metal.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------------------------------------------------------- */
/* Configuration (mflux SeedVR2Transformer / SeedVR2VAE defaults) */

#define SV2_DIM 2560
#define SV2_HEADS 20
#define SV2_HEAD_DIM 128
#define SV2_LAYERS 32
#define SV2_MM_LAYERS 10
#define SV2_HIDDEN 6912
#define SV2_TXT_TOKENS 58
#define SV2_TXT_IN 5120
#define SV2_VID_IN 33
#define SV2_VID_OUT 16
#define SV2_PATCH_IN (SV2_VID_IN * 4)      /* 132 */
#define SV2_PATCH_OUT (SV2_VID_OUT * 4)    /* 64 */
#define SV2_ROPE_FREQS 21                   /* rope_dim 128 / 3 axes / 2 */
#define SV2_ROPE_PAIRS (3 * SV2_ROPE_FREQS) /* 63 pairs = 126 rotated dims */
#define SV2_NORM_EPS 1e-5f
#define SV2_VAE_EPS 1e-6f
#define SV2_VAE_GROUPS 32
#define SV2_LATENT 16
#define SV2_SCALING 0.9152f
#define SV2_TIMESTEP 1000.0f
#define SV2_PIXEL_MULTIPLE 16
#define SV2_QUANT_GROUP 64

/* ---------------------------------------------------------------- */
/* Small host helpers */

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int trace_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("H3_SV2_TRACE") != NULL;
    return cached;
}

static void trace(double *mark, const char *label) {
    if (!trace_enabled()) return;
    double t = now_seconds();
    fprintf(stderr, "sv2 %-24s %8.1f ms\n", label, (t - *mark) * 1000.0);
    *mark = t;
}

/* H3_SV2_SYNC: the trace waits for the GPU first, so the stage times are
 * real (slower overall: the command buffer is split at every mark). */
static void trace_sync(h3_gpu *gpu, double *mark, const char *label) {
    if (!trace_enabled()) return;
    if (getenv("H3_SV2_SYNC") && h3_gpu_command_open(gpu)) { h3_gpu_submit(gpu); h3_gpu_begin(gpu); }
    trace(mark, label);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static int round_up(int value, int multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

/* ---------------------------------------------------------------- */
/* Weights */

typedef struct {
    h3_gpu_tensor *w;      /* BF16 [out, in]; NULL once quantized */
    h3_gpu_tensor *b;      /* F32 [out] or NULL */
    h3_gpu_tensor *q;      /* int8 [out, in] with F32 `scales` [out] (tensor hardware) */
    h3_gpu_tensor *scales;
    uint32_t in, out;
    int owned;             /* 0: shares another linear's tensors */
} sv2_linear;

typedef struct {
    int shared, last;
    sv2_linear qkv_vid, out_vid, qkv_txt, out_txt;
    sv2_linear fc1_vid, fc2_vid, fc1_txt, fc2_txt;   /* fc1 = [gate; up] */
    h3_gpu_tensor *nq_vid, *nk_vid, *nq_txt, *nk_txt;   /* F32 [128]; txt = vid when shared */
    /* attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate:
     * emb + the block's parameters, F32 [dim]; txt NULL in the last block */
    h3_gpu_tensor *mod_vid[6], *mod_txt[6];
} sv2_block;

typedef struct {
    h3_gpu_tensor *w;      /* F32 [out, in, kh, kw] (OIHW); NULL on the TensorOps path */
    h3_gpu_tensor *b;      /* F32 [out] */
    int in, out, kernel;
    /* TensorOps path: tap-major BF16 [taps * out_p, in_p] on channels padded
     * to 64, and the bias padded to out_p (h3_gpu_vsr_tapconv_bf16) */
    h3_gpu_tensor *nw, *nb;
    int in_p, out_p;
} sv2_conv;

typedef struct {
    h3_gpu_tensor *w, *b;  /* F32 [C] */
} sv2_norm;

typedef struct {
    int in, out;
    sv2_norm norm1, norm2;
    sv2_conv conv1, conv2, shortcut;   /* shortcut.w NULL when in == out */
} sv2_resnet;

typedef struct {
    sv2_norm norm;
    h3_gpu_tensor *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;   /* F32 [C, C], [C] */
} sv2_attention;

typedef struct {
    sv2_conv conv_in;
    sv2_resnet down[4][2];
    sv2_conv downsample[3];            /* 3x3 stride 2 (temporal taps folded) */
    sv2_resnet mid_a, mid_b;
    sv2_attention mid_attn;
    sv2_norm norm_out;
    sv2_conv conv_out;                 /* 512 -> 32 */
} sv2_encoder;

typedef struct {
    sv2_conv conv_in;                  /* 16 -> 512 */
    sv2_resnet mid_a, mid_b;
    sv2_attention mid_attn;
    sv2_resnet up[4][3];
    h3_gpu_tensor *up_w[3], *up_b[3];  /* upscale_conv 1x1: F32 [C * 4 * tf, C] */
    sv2_conv up_nax[3];                /* the same as a 1-tap TensorOps conv */
    int up_temporal[3];
    sv2_conv up_conv[3];               /* the 3x3 after the shuffle */
    sv2_norm norm_out;
    sv2_conv conv_out;                 /* 128 -> 3 */
} sv2_decoder;

struct h3_seedvr2 {
    h3_gpu *gpu;
    h3_seedvr2_info info;
    h3_seedvr2_open_params params;
    char model_dir[1024];
    int nax_vae;                       /* the VAE on the TensorOps grid (M5) */
    int int8_dit;                      /* the transformer's linears per-row int8 (M5) */
    /* transformer */
    sv2_linear vid_in, vid_out;
    h3_gpu_tensor *txt0;               /* BF16 [58, 2560]: txt_in(pos_emb) */
    sv2_block blocks[SV2_LAYERS];
    h3_gpu_tensor *out_scale, *out_shift;   /* F32 [dim]: norm weight * (scale_a + out_scale), shift_a + out_shift */
    /* vae */
    sv2_encoder enc;
    sv2_decoder dec;
    /* activation pool (F32 and BF16 buffers reused across frames) */
    struct { h3_gpu_tensor *t; size_t bytes; int bf16; } pool[64];
    int pool_count;
    /* load-time temporaries freed after the next submit */
    h3_gpu_tensor *deferred[256];
    int deferred_count;
    char error[512];
};

static void defer_free(h3_seedvr2 *m, h3_gpu_tensor *t) {
    if (!t) return;
    if (m->deferred_count < (int)(sizeof(m->deferred) / sizeof(*m->deferred))) {
        m->deferred[m->deferred_count++] = t;
    } else {
        /* full: flush now (waits for the GPU), then keep this one */
        h3_gpu_submit(m->gpu);
        for (int i = 0; i < m->deferred_count; i++) h3_gpu_tensor_free(m->deferred[i]);
        m->deferred_count = 0;
        m->deferred[m->deferred_count++] = t;
        h3_gpu_begin(m->gpu);
    }
}

/* Submits the open command buffer and frees the load temporaries. */
static int flush_loads(h3_seedvr2 *m, char *error, size_t error_size) {
    int ok = h3_gpu_submit(m->gpu);
    for (int i = 0; i < m->deferred_count; i++) h3_gpu_tensor_free(m->deferred[i]);
    m->deferred_count = 0;
    if (!ok) fail(error, error_size, "SeedVR2: GPU error while loading: %s", h3_gpu_error(m->gpu));
    return ok;
}

/* --- safetensors access --- */

typedef struct {
    h3_st_header transformer, vae, pos;
    int have_transformer, have_vae, have_pos;
} sv2_files;

static void files_close(sv2_files *f) {
    if (f->have_transformer) h3_st_free_header(&f->transformer);
    if (f->have_vae) h3_st_free_header(&f->vae);
    if (f->have_pos) h3_st_free_header(&f->pos);
    memset(f, 0, sizeof(*f));
}

static int files_open(sv2_files *f, const char *dir, char *error, size_t error_size) {
    memset(f, 0, sizeof(*f));
    char path[1200];
    snprintf(path, sizeof(path), "%s/transformer.safetensors", dir);
    if (!h3_st_read_header(path, &f->transformer, error, error_size)) return 0;
    f->have_transformer = 1;
    snprintf(path, sizeof(path), "%s/vae.safetensors", dir);
    if (!h3_st_read_header(path, &f->vae, error, error_size)) { files_close(f); return 0; }
    f->have_vae = 1;
    snprintf(path, sizeof(path), "%s/pos_emb.safetensors", dir);
    if (!h3_st_read_header(path, &f->pos, error, error_size)) { files_close(f); return 0; }
    f->have_pos = 1;
    return 1;
}

/* Reads a whole tensor into a fresh host buffer (bytes as in the file). */
static void *read_tensor(const h3_st_header *h, const char *name, h3_dtype dtype,
                         uint64_t *elements, char *error, size_t error_size) {
    const h3_st_tensor *t = h3_st_find(h, name);
    if (!t) { fail(error, error_size, "SeedVR2: %s is missing from %s", name, h->path); return NULL; }
    if (dtype != H3_DTYPE_UNKNOWN && t->dtype != dtype) {
        fail(error, error_size, "SeedVR2: %s is %s, expected %s", name,
             h3_dtype_name(t->dtype), h3_dtype_name(dtype));
        return NULL;
    }
    uint64_t n = h3_st_tensor_elements(t);
    size_t bytes = (size_t)n * h3_dtype_size(t->dtype);
    void *data = malloc(bytes ? bytes : 1);
    if (!data) { fail(error, error_size, "SeedVR2: out of memory reading %s", name); return NULL; }
    if (!h3_st_read_data(h, t, data, bytes, error, error_size)) { free(data); return NULL; }
    if (elements) *elements = n;
    return data;
}

/* F16 tensor -> host F32. */
static float *read_f16_as_f32(const h3_st_header *h, const char *name, uint64_t expected,
                              char *error, size_t error_size) {
    uint64_t n = 0;
    uint16_t *raw = read_tensor(h, name, H3_DTYPE_F16, &n, error, error_size);
    if (!raw) return NULL;
    if (expected && n != expected) {
        fail(error, error_size, "SeedVR2: %s has %llu values, expected %llu", name,
             (unsigned long long)n, (unsigned long long)expected);
        free(raw);
        return NULL;
    }
    float *out = malloc(n * sizeof(float));
    if (!out) { free(raw); fail(error, error_size, "SeedVR2: out of memory"); return NULL; }
    for (uint64_t i = 0; i < n; i++) out[i] = f16_to_f32(raw[i]);
    free(raw);
    return out;
}

static h3_gpu_tensor *upload_f32(h3_gpu *gpu, const float *values, size_t n) {
    return h3_gpu_tensor_from_f32(gpu, values, n);
}

static h3_gpu_tensor *upload_f32_as_bf16(h3_gpu *gpu, const float *values, size_t n) {
    uint16_t *bf = malloc(n * sizeof(uint16_t));
    if (!bf) return NULL;
    for (size_t i = 0; i < n; i++) bf[i] = f32_to_bf16(values[i]);
    h3_gpu_tensor *t = h3_gpu_tensor_from_bf16(gpu, bf, n);
    free(bf);
    return t;
}

/* An F16 tensor straight to a GPU F32 tensor. */
static h3_gpu_tensor *load_f16_f32(h3_gpu *gpu, const h3_st_header *h, const char *name,
                                   uint64_t expected, char *error, size_t error_size) {
    float *v = read_f16_as_f32(h, name, expected, error, error_size);
    if (!v) return NULL;
    h3_gpu_tensor *t = upload_f32(gpu, v, expected ? expected : 0);
    if (!expected) { /* unknown size: recount */
        const h3_st_tensor *st = h3_st_find(h, name);
        t = upload_f32(gpu, v, (size_t)h3_st_tensor_elements(st));
    }
    free(v);
    if (!t) fail(error, error_size, "SeedVR2: out of GPU memory for %s", name);
    return t;
}

/* Whether `prefix`.weight is MLX 8-bit (has .scales) in this export. */
static int is_quantized(const h3_st_header *h, const char *prefix) {
    char name[256];
    snprintf(name, sizeof(name), "%s.scales", prefix);
    return h3_st_find(h, name) != NULL;
}

/* Loads `prefix`.{weight,scales,biases} (MLX 8-bit) or an F16 weight into a
 * BF16 [out, in] GPU tensor, plus the F32 bias when present. The dequantise
 * is encoded into the open command buffer; its inputs are freed at the next
 * flush_loads. */
static int load_linear(h3_seedvr2 *m, const h3_st_header *h, const char *prefix,
                       uint32_t out, uint32_t in, int want_bias, sv2_linear *l,
                       char *error, size_t error_size) {
    memset(l, 0, sizeof(*l));
    l->in = in; l->out = out; l->owned = 1;
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const h3_st_tensor *wt = h3_st_find(h, name);
    if (!wt) { fail(error, error_size, "SeedVR2: %s is missing", name); return 0; }
    l->w = h3_gpu_tensor_new_bf16(m->gpu, (size_t)out * in);
    if (!l->w) { fail(error, error_size, "SeedVR2: out of GPU memory for %s", name); return 0; }
    if (is_quantized(h, prefix)) {
        if (wt->dtype != H3_DTYPE_U32 || wt->ndim != 2 || wt->shape[0] != out ||
            wt->shape[1] * 4 != in) {
            fail(error, error_size, "SeedVR2: %s is not a U32 [%u, %u] 8-bit weight", name, out, in / 4);
            return 0;
        }
        uint64_t n = 0;
        uint32_t *packed = read_tensor(h, name, H3_DTYPE_U32, &n, error, error_size);
        if (!packed) return 0;
        h3_gpu_tensor *tp = h3_gpu_tensor_from_u32(m->gpu, packed, (size_t)n);
        free(packed);
        defer_free(m, tp);
        snprintf(name, sizeof(name), "%s.scales", prefix);
        void *scales = read_tensor(h, name, H3_DTYPE_F16, &n, error, error_size);
        if (!scales) return 0;
        if (n != (uint64_t)out * (in / SV2_QUANT_GROUP)) {
            free(scales);
            fail(error, error_size, "SeedVR2: %s has an unexpected group size", name);
            return 0;
        }
        h3_gpu_tensor *ts = h3_gpu_tensor_from_i8(m->gpu, scales, (size_t)n * 2);
        free(scales);
        defer_free(m, ts);
        snprintf(name, sizeof(name), "%s.biases", prefix);
        void *biases = read_tensor(h, name, H3_DTYPE_F16, &n, error, error_size);
        if (!biases) return 0;
        h3_gpu_tensor *tb = h3_gpu_tensor_from_i8(m->gpu, biases, (size_t)n * 2);
        free(biases);
        defer_free(m, tb);
        if (!tp || !ts || !tb) {
            fail(error, error_size, "SeedVR2: out of GPU memory loading %s", prefix);
            return 0;
        }
        if (!h3_gpu_sv2_dequant_q8_bf16(m->gpu, l->w, tp, ts, tb, out, in, SV2_QUANT_GROUP)) {
            fail(error, error_size, "SeedVR2: dequantise %s: %s", prefix, h3_gpu_error(m->gpu));
            return 0;
        }
    } else {
        float *w = read_f16_as_f32(h, name, (uint64_t)out * in, error, error_size);
        if (!w) return 0;
        uint16_t *bf = malloc((size_t)out * in * sizeof(uint16_t));
        if (!bf) { free(w); fail(error, error_size, "SeedVR2: out of memory"); return 0; }
        for (size_t i = 0; i < (size_t)out * in; i++) bf[i] = f32_to_bf16(w[i]);
        free(w);
        int ok = h3_gpu_tensor_write_bf16(l->w, bf, (size_t)out * in);
        free(bf);
        if (!ok) { fail(error, error_size, "SeedVR2: cannot write %s", name); return 0; }
    }
    if (want_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        l->b = load_f16_f32(m->gpu, h, name, out, error, error_size);
        if (!l->b) return 0;
    }
    return 1;
}

/* Two linears stacked into one [2 * out, in] (the SwiGLU gate then up). */
static int load_linear_pair(h3_seedvr2 *m, const h3_st_header *h, const char *gate_prefix,
                            const char *up_prefix, uint32_t out, uint32_t in, sv2_linear *l,
                            char *error, size_t error_size) {
    sv2_linear gate, up;
    if (!load_linear(m, h, gate_prefix, out, in, 0, &gate, error, error_size)) return 0;
    if (!load_linear(m, h, up_prefix, out, in, 0, &up, error, error_size)) {
        h3_gpu_tensor_free(gate.w);
        return 0;
    }
    memset(l, 0, sizeof(*l));
    l->in = in; l->out = 2 * out; l->owned = 1;
    l->w = h3_gpu_tensor_new_bf16(m->gpu, (size_t)2 * out * in);
    int ok = l->w &&
        h3_gpu_copy_bf16(m->gpu, l->w, 0, gate.w, 0, (size_t)out * in) &&
        h3_gpu_copy_bf16(m->gpu, l->w, (size_t)out * in, up.w, 0, (size_t)out * in);
    defer_free(m, gate.w);
    defer_free(m, up.w);
    if (!ok) fail(error, error_size, "SeedVR2: cannot stack %s and %s", gate_prefix, up_prefix);
    return ok;
}

static void free_linear(sv2_linear *l) {
    if (!l->owned) return;
    h3_gpu_tensor_free(l->w);
    h3_gpu_tensor_free(l->b);
    h3_gpu_tensor_free(l->q);
    h3_gpu_tensor_free(l->scales);
    memset(l, 0, sizeof(*l));
}

/* BF16 [out, in] -> per-row int8 + F32 scales on the GPU (inside the open
 * command buffer); the BF16 copy is freed at the next flush. Shapes the
 * int8 kernel cannot take (in or out not a multiple of 128) stay BF16. */
static int quantize_linear(h3_seedvr2 *m, sv2_linear *l, char *error, size_t error_size) {
    if (!l->owned || !l->w || l->in % 128 || l->out % 128) return 1;
    l->q = h3_gpu_tensor_new_i8(m->gpu, (size_t)l->out * l->in);
    l->scales = h3_gpu_tensor_new_f32(m->gpu, l->out);
    if (!l->q || !l->scales || !h3_gpu_quantize_weight_int8(m->gpu, l->q, l->scales, l->w, l->out, l->in)) {
        fail(error, error_size, "SeedVR2: int8 quantisation: %s", h3_gpu_error(m->gpu));
        return 0;
    }
    defer_free(m, l->w);
    l->w = NULL;
    return 1;
}

/* CPU dequantise of an MLX 8-bit [out, in] linear to F32 (the constant
 * time-embedding MLP runs on the host once). */
static float *dequant_host(const h3_st_header *h, const char *prefix, uint32_t out, uint32_t in,
                           char *error, size_t error_size) {
    char name[256];
    uint64_t n = 0;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    uint32_t *packed = read_tensor(h, name, H3_DTYPE_U32, &n, error, error_size);
    if (!packed) return NULL;
    snprintf(name, sizeof(name), "%s.scales", prefix);
    uint16_t *scales = read_tensor(h, name, H3_DTYPE_F16, NULL, error, error_size);
    snprintf(name, sizeof(name), "%s.biases", prefix);
    uint16_t *biases = scales ? read_tensor(h, name, H3_DTYPE_F16, NULL, error, error_size) : NULL;
    float *w = (scales && biases) ? malloc((size_t)out * in * sizeof(float)) : NULL;
    if (w) {
        uint32_t groups = in / SV2_QUANT_GROUP, words = in / 4;
        for (uint32_t r = 0; r < out; r++) {
            for (uint32_t c = 0; c < in; c++) {
                uint32_t g = r * groups + c / SV2_QUANT_GROUP;
                uint32_t q = (packed[(size_t)r * words + c / 4] >> (8 * (c % 4))) & 0xffu;
                w[(size_t)r * in + c] = f16_to_f32(scales[g]) * (float)q + f16_to_f32(biases[g]);
            }
        }
    } else if (scales && biases) {
        fail(error, error_size, "SeedVR2: out of memory");
    }
    free(packed); free(scales); free(biases);
    return w;
}

/* y = W x + b on the host. */
static void host_linear(const float *w, const float *b, const float *x, float *y,
                        uint32_t out, uint32_t in) {
    for (uint32_t o = 0; o < out; o++) {
        const float *row = w + (size_t)o * in;
        double acc = b ? b[o] : 0.0;
        for (uint32_t i = 0; i < in; i++) acc += (double)row[i] * x[i];
        y[o] = (float)acc;
    }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* The constant modulation: emb_in(t = 1000) as F32 [6 * dim], laid out
 * (channel, layer, kind) like the reference's reshape(dim, 2, 3). */
static float *compute_emb(const h3_st_header *h, char *error, size_t error_size) {
    float sinusoid[256];
    for (int i = 0; i < 128; i++) {
        float freq = expf((float)i * (-logf(10000.0f) / 128.0f));
        float a = SV2_TIMESTEP * freq;
        sinusoid[i] = sinf(a);
        sinusoid[128 + i] = cosf(a);
    }
    float *w1 = dequant_host(h, "emb_in.proj_in", SV2_DIM, 256, error, error_size);
    float *b1 = w1 ? read_f16_as_f32(h, "emb_in.proj_in.bias", SV2_DIM, error, error_size) : NULL;
    float *w2 = b1 ? dequant_host(h, "emb_in.proj_hid", SV2_DIM, SV2_DIM, error, error_size) : NULL;
    float *b2 = w2 ? read_f16_as_f32(h, "emb_in.proj_hid.bias", SV2_DIM, error, error_size) : NULL;
    float *w3 = b2 ? dequant_host(h, "emb_in.proj_out", 6 * SV2_DIM, SV2_DIM, error, error_size) : NULL;
    float *b3 = w3 ? read_f16_as_f32(h, "emb_in.proj_out.bias", 6 * SV2_DIM, error, error_size) : NULL;
    float *emb = NULL;
    if (b3) {
        float *h1 = malloc(SV2_DIM * sizeof(float)), *h2 = malloc(SV2_DIM * sizeof(float));
        emb = malloc(6 * SV2_DIM * sizeof(float));
        if (h1 && h2 && emb) {
            host_linear(w1, b1, sinusoid, h1, SV2_DIM, 256);
            for (int i = 0; i < SV2_DIM; i++) h1[i] = silu(h1[i]);
            host_linear(w2, b2, h1, h2, SV2_DIM, SV2_DIM);
            for (int i = 0; i < SV2_DIM; i++) h2[i] = silu(h2[i]);
            host_linear(w3, b3, h2, emb, 6 * SV2_DIM, SV2_DIM);
        } else {
            free(emb); emb = NULL;
            fail(error, error_size, "SeedVR2: out of memory");
        }
        free(h1); free(h2);
    }
    free(w1); free(b1); free(w2); free(b2); free(w3); free(b3);
    return emb;
}

/* emb[c, layer, kind] + params -> an F32 [dim] GPU vector. */
static h3_gpu_tensor *modulation_vector(h3_gpu *gpu, const float *emb, int layer, int kind,
                                        const float *params) {
    float *v = malloc(SV2_DIM * sizeof(float));
    if (!v) return NULL;
    for (int c = 0; c < SV2_DIM; c++) v[c] = emb[c * 6 + layer * 3 + kind] + (params ? params[c] : 0.0f);
    h3_gpu_tensor *t = upload_f32(gpu, v, SV2_DIM);
    free(v);
    return t;
}

/* --- transformer blocks --- */

static int load_block(h3_seedvr2 *m, const h3_st_header *h, int b, const float *emb,
                      char *error, size_t error_size) {
    sv2_block *k = &m->blocks[b];
    memset(k, 0, sizeof(*k));
    k->shared = b >= SV2_MM_LAYERS;
    k->last = b == SV2_LAYERS - 1;
    char p[128];
    const char *stream = k->shared ? "all" : "vid";
    /* attention */
    snprintf(p, sizeof(p), "blocks.%d.attn.proj_qkv_vid", b);
    if (!load_linear(m, h, p, 3 * SV2_DIM, SV2_DIM, 0, &k->qkv_vid, error, error_size)) return 0;
    snprintf(p, sizeof(p), "blocks.%d.attn.proj_out_vid", b);
    if (!load_linear(m, h, p, SV2_DIM, SV2_DIM, 1, &k->out_vid, error, error_size)) return 0;
    snprintf(p, sizeof(p), "blocks.%d.attn.norm_q_vid.weight", b);
    if (!(k->nq_vid = load_f16_f32(m->gpu, h, p, SV2_HEAD_DIM, error, error_size))) return 0;
    snprintf(p, sizeof(p), "blocks.%d.attn.norm_k_vid.weight", b);
    if (!(k->nk_vid = load_f16_f32(m->gpu, h, p, SV2_HEAD_DIM, error, error_size))) return 0;
    if (k->shared) {
        k->qkv_txt = k->qkv_vid; k->qkv_txt.owned = 0;
        k->out_txt = k->out_vid; k->out_txt.owned = 0;
        k->nq_txt = k->nq_vid; k->nk_txt = k->nk_vid;
    } else {
        snprintf(p, sizeof(p), "blocks.%d.attn.proj_qkv_txt", b);
        if (!load_linear(m, h, p, 3 * SV2_DIM, SV2_DIM, 0, &k->qkv_txt, error, error_size)) return 0;
        snprintf(p, sizeof(p), "blocks.%d.attn.proj_out_txt", b);
        if (!load_linear(m, h, p, SV2_DIM, SV2_DIM, 1, &k->out_txt, error, error_size)) return 0;
        snprintf(p, sizeof(p), "blocks.%d.attn.norm_q_txt.weight", b);
        if (!(k->nq_txt = load_f16_f32(m->gpu, h, p, SV2_HEAD_DIM, error, error_size))) return 0;
        snprintf(p, sizeof(p), "blocks.%d.attn.norm_k_txt.weight", b);
        if (!(k->nk_txt = load_f16_f32(m->gpu, h, p, SV2_HEAD_DIM, error, error_size))) return 0;
    }
    /* feed-forward */
    char gate[128], up[128];
    snprintf(gate, sizeof(gate), "blocks.%d.mlp.%s.proj_in_gate", b, stream);
    snprintf(up, sizeof(up), "blocks.%d.mlp.%s.proj_in", b, stream);
    if (!load_linear_pair(m, h, gate, up, SV2_HIDDEN, SV2_DIM, &k->fc1_vid, error, error_size)) return 0;
    snprintf(p, sizeof(p), "blocks.%d.mlp.%s.proj_out", b, stream);
    if (!load_linear(m, h, p, SV2_DIM, SV2_HIDDEN, 0, &k->fc2_vid, error, error_size)) return 0;
    if (k->shared) {
        k->fc1_txt = k->fc1_vid; k->fc1_txt.owned = 0;
        k->fc2_txt = k->fc2_vid; k->fc2_txt.owned = 0;
    } else {
        snprintf(gate, sizeof(gate), "blocks.%d.mlp.txt.proj_in_gate", b);
        snprintf(up, sizeof(up), "blocks.%d.mlp.txt.proj_in", b);
        if (!load_linear_pair(m, h, gate, up, SV2_HIDDEN, SV2_DIM, &k->fc1_txt, error, error_size)) return 0;
        snprintf(p, sizeof(p), "blocks.%d.mlp.txt.proj_out", b);
        if (!load_linear(m, h, p, SV2_DIM, SV2_HIDDEN, 0, &k->fc2_txt, error, error_size)) return 0;
    }
    if (m->int8_dit) {
        /* H3_SV2_INT8_SKIP=qkv,out,fc1,fc2 keeps those BF16 (measuring) */
        const char *skip = getenv("H3_SV2_INT8_SKIP");
        int sq = skip && strstr(skip, "qkv"), so = skip && strstr(skip, "out");
        int s1 = skip && strstr(skip, "fc1"), s2 = skip && strstr(skip, "fc2");
        sv2_linear *lins[8] = {&k->qkv_vid, &k->out_vid, &k->qkv_txt, &k->out_txt,
                               &k->fc1_vid, &k->fc2_vid, &k->fc1_txt, &k->fc2_txt};
        int skips[8] = {sq, so, sq, so, s1, s2, s1, s2};
        for (int i = 0; i < 8; i++) if (!skips[i] && !quantize_linear(m, lins[i], error, error_size)) return 0;
        if (k->shared) {
            k->qkv_txt = k->qkv_vid; k->qkv_txt.owned = 0;
            k->out_txt = k->out_vid; k->out_txt.owned = 0;
            k->fc1_txt = k->fc1_vid; k->fc1_txt.owned = 0;
            k->fc2_txt = k->fc2_vid; k->fc2_txt.owned = 0;
        }
    }
    /* modulation constants: emb + ada.params_{vid,txt,all}.{attn,mlp}_{shift,scale,gate} */
    static const char *kinds[3] = {"shift", "scale", "gate"};
    static const char *layers[2] = {"attn", "mlp"};
    for (int pass = 0; pass < 2; pass++) {
        int txt = pass == 1;
        if (txt && k->last) break;   /* the last block leaves the text stream alone */
        const char *which = k->shared ? "all" : (txt ? "txt" : "vid");
        for (int layer = 0; layer < 2; layer++) {
            for (int kind = 0; kind < 3; kind++) {
                snprintf(p, sizeof(p), "blocks.%d.ada.params_%s.%s_%s", b, which, layers[layer], kinds[kind]);
                float *params = read_f16_as_f32(h, p, SV2_DIM, error, error_size);
                if (!params) return 0;
                h3_gpu_tensor *t = modulation_vector(m->gpu, emb, layer, kind, params);
                free(params);
                if (!t) { fail(error, error_size, "SeedVR2: out of GPU memory"); return 0; }
                (txt ? k->mod_txt : k->mod_vid)[layer * 3 + kind] = t;
            }
        }
    }
    return 1;
}

static void free_block(sv2_block *k) {
    free_linear(&k->qkv_vid); free_linear(&k->out_vid);
    free_linear(&k->qkv_txt); free_linear(&k->out_txt);
    free_linear(&k->fc1_vid); free_linear(&k->fc2_vid);
    free_linear(&k->fc1_txt); free_linear(&k->fc2_txt);
    h3_gpu_tensor_free(k->nq_vid); h3_gpu_tensor_free(k->nk_vid);
    if (!k->shared) { h3_gpu_tensor_free(k->nq_txt); h3_gpu_tensor_free(k->nk_txt); }
    for (int i = 0; i < 6; i++) { h3_gpu_tensor_free(k->mod_vid[i]); h3_gpu_tensor_free(k->mod_txt[i]); }
    memset(k, 0, sizeof(*k));
}

/* --- VAE weights: every causal 3D convolution on one frame is the sum of
 * its temporal taps (the frame is replicated into the causal padding), so
 * each is loaded as a 2D OIHW convolution. Weights are stored (O, kt, kh, kw, I). --- */

static int round64(int value) { return (value + 63) / 64 * 64; }

/* OIHW F32 -> the tap-major BF16 weights of h3_gpu_vsr_tapconv_bf16 on
 * channels padded to 64, tap t = ky * k + kx; the bias padded to out_p. */
static int pack_nax_conv(h3_seedvr2 *m, sv2_conv *c, const float *w, char *error, size_t error_size) {
    int k = c->kernel, taps = k * k;
    c->in_p = round64(c->in); c->out_p = round64(c->out);
    size_t n = (size_t)taps * c->out_p * c->in_p;
    uint16_t *packed = calloc(n, sizeof(uint16_t));
    float *bias = calloc((size_t)c->out_p, sizeof(float));
    if (!packed || !bias) { free(packed); free(bias); fail(error, error_size, "SeedVR2: out of memory"); return 0; }
    for (int ky = 0; ky < k; ky++)
        for (int kx = 0; kx < k; kx++) {
            int t = ky * k + kx;
            for (int o = 0; o < c->out; o++)
                for (int i = 0; i < c->in; i++)
                    packed[((size_t)t * c->out_p + o) * c->in_p + i] =
                        f32_to_bf16(w[(((size_t)o * c->in + i) * k + ky) * k + kx]);
        }
    if (c->b) h3_gpu_tensor_read_f32(c->b, bias, (size_t)c->out);
    c->nw = h3_gpu_tensor_from_bf16(m->gpu, packed, n);
    c->nb = h3_gpu_tensor_from_f32(m->gpu, bias, (size_t)c->out_p);
    free(packed); free(bias);
    if (!c->nw || !c->nb) { fail(error, error_size, "SeedVR2: out of GPU memory for the TensorOps weights"); return 0; }
    return 1;
}

static int load_conv(h3_seedvr2 *m, const h3_st_header *h, const char *prefix, int out, int in,
                     int kernel, sv2_conv *c, char *error, size_t error_size) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const h3_st_tensor *t = h3_st_find(h, name);
    if (!t || t->ndim != 5) { fail(error, error_size, "SeedVR2: %s is missing or not 5-D", name); return 0; }
    int kt = (int)t->shape[1], kh = (int)t->shape[2], kw = (int)t->shape[3];
    if ((int)t->shape[0] != out || (int)t->shape[4] != in || kh != kernel || kw != kernel) {
        fail(error, error_size, "SeedVR2: %s has shape [%d,%d,%d,%d,%d], expected [%d,*,%d,%d,%d]",
             name, (int)t->shape[0], kt, kh, kw, (int)t->shape[4], out, kernel, kernel, in);
        return 0;
    }
    float *w5 = read_f16_as_f32(h, name, (uint64_t)out * kt * kh * kw * in, error, error_size);
    if (!w5) return 0;
    size_t n = (size_t)out * in * kh * kw;
    float *w = calloc(n, sizeof(float));
    if (!w) { free(w5); fail(error, error_size, "SeedVR2: out of memory"); return 0; }
    for (int o = 0; o < out; o++)
        for (int tt = 0; tt < kt; tt++)
            for (int y = 0; y < kh; y++)
                for (int x = 0; x < kw; x++) {
                    const float *src = w5 + ((((size_t)o * kt + tt) * kh + y) * kw + x) * in;
                    float *dst = w + (((size_t)o * in) * kh + y) * kw + x;
                    for (int i = 0; i < in; i++) dst[(size_t)i * kh * kw] += src[i];
                }
    free(w5);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    c->b = load_f16_f32(m->gpu, h, name, out, error, error_size);
    c->in = in; c->out = out; c->kernel = kernel;
    if (m->nax_vae) {
        if (!pack_nax_conv(m, c, w, error, error_size)) { free(w); return 0; }
    } else {
        c->w = upload_f32(m->gpu, w, n);
    }
    free(w);
    if ((!c->w && !c->nw) || !c->b) { fail(error, error_size, "SeedVR2: out of GPU memory for %s", prefix); return 0; }
    return 1;
}

static int load_norm(h3_seedvr2 *m, const h3_st_header *h, const char *prefix, int channels,
                     sv2_norm *n, char *error, size_t error_size) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    n->w = load_f16_f32(m->gpu, h, name, channels, error, error_size);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    n->b = n->w ? load_f16_f32(m->gpu, h, name, channels, error, error_size) : NULL;
    return n->w && n->b;
}

static int load_resnet(h3_seedvr2 *m, const h3_st_header *h, const char *prefix, int in, int out,
                       sv2_resnet *r, char *error, size_t error_size) {
    char p[256];
    r->in = in; r->out = out;
    snprintf(p, sizeof(p), "%s.norm1", prefix);
    if (!load_norm(m, h, p, in, &r->norm1, error, error_size)) return 0;
    snprintf(p, sizeof(p), "%s.norm2", prefix);
    if (!load_norm(m, h, p, out, &r->norm2, error, error_size)) return 0;
    snprintf(p, sizeof(p), "%s.conv1", prefix);
    if (!load_conv(m, h, p, out, in, 3, &r->conv1, error, error_size)) return 0;
    snprintf(p, sizeof(p), "%s.conv2", prefix);
    if (!load_conv(m, h, p, out, out, 3, &r->conv2, error, error_size)) return 0;
    if (in != out) {
        snprintf(p, sizeof(p), "%s.conv_shortcut", prefix);
        if (!load_conv(m, h, p, out, in, 1, &r->shortcut, error, error_size)) return 0;
    }
    return 1;
}

static int load_attention(h3_seedvr2 *m, const h3_st_header *h, const char *prefix, int channels,
                          sv2_attention *a, char *error, size_t error_size) {
    char p[256];
    snprintf(p, sizeof(p), "%s.group_norm", prefix);
    if (!load_norm(m, h, p, channels, &a->norm, error, error_size)) return 0;
    struct { const char *name; h3_gpu_tensor **w, **b; } parts[4] = {
        {"to_q", &a->wq, &a->bq}, {"to_k", &a->wk, &a->bk}, {"to_v", &a->wv, &a->bv}, {"to_out.0", &a->wo, &a->bo}};
    for (int i = 0; i < 4; i++) {
        snprintf(p, sizeof(p), "%s.%s.weight", prefix, parts[i].name);
        *parts[i].w = load_f16_f32(m->gpu, h, p, (uint64_t)channels * channels, error, error_size);
        snprintf(p, sizeof(p), "%s.%s.bias", prefix, parts[i].name);
        *parts[i].b = *parts[i].w ? load_f16_f32(m->gpu, h, p, channels, error, error_size) : NULL;
        if (!*parts[i].b) return 0;
    }
    return 1;
}

static int load_vae(h3_seedvr2 *m, const h3_st_header *h, char *error, size_t error_size) {
    static const int channels[4] = {128, 256, 512, 512};
    sv2_encoder *e = &m->enc;
    sv2_decoder *d = &m->dec;
    char p[256];
    if (!load_conv(m, h, "encoder.conv_in", 128, 3, 3, &e->conv_in, error, error_size)) return 0;
    int c_in = 128;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            snprintf(p, sizeof(p), "encoder.down_blocks.%d.resnets.%d", i, j);
            if (!load_resnet(m, h, p, j == 0 ? c_in : channels[i], channels[i], &e->down[i][j], error, error_size)) return 0;
        }
        c_in = channels[i];
        if (i < 3) {
            snprintf(p, sizeof(p), "encoder.down_blocks.%d.downsamplers.0.conv", i);
            if (!load_conv(m, h, p, channels[i], channels[i], 3, &e->downsample[i], error, error_size)) return 0;
        }
    }
    if (!load_resnet(m, h, "encoder.mid_block.resnets.0", 512, 512, &e->mid_a, error, error_size) ||
        !load_attention(m, h, "encoder.mid_block.attentions.0", 512, &e->mid_attn, error, error_size) ||
        !load_resnet(m, h, "encoder.mid_block.resnets.1", 512, 512, &e->mid_b, error, error_size) ||
        !load_norm(m, h, "encoder.conv_norm_out", 512, &e->norm_out, error, error_size) ||
        !load_conv(m, h, "encoder.conv_out", 2 * SV2_LATENT, 512, 3, &e->conv_out, error, error_size))
        return 0;
    /* decoder: reversed channels (512, 512, 256, 128), 3 resnets each */
    static const int rev[4] = {512, 512, 256, 128};
    if (!load_conv(m, h, "decoder.conv_in", 512, SV2_LATENT, 3, &d->conv_in, error, error_size) ||
        !load_resnet(m, h, "decoder.mid_block.resnets.0", 512, 512, &d->mid_a, error, error_size) ||
        !load_attention(m, h, "decoder.mid_block.attentions.0", 512, &d->mid_attn, error, error_size) ||
        !load_resnet(m, h, "decoder.mid_block.resnets.1", 512, 512, &d->mid_b, error, error_size))
        return 0;
    c_in = 512;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            snprintf(p, sizeof(p), "decoder.up_blocks.%d.resnets.%d", i, j);
            if (!load_resnet(m, h, p, j == 0 ? c_in : rev[i], rev[i], &d->up[i][j], error, error_size)) return 0;
        }
        c_in = rev[i];
        if (i < 3) {
            int tf = i < 2 ? 2 : 1;
            d->up_temporal[i] = tf;
            snprintf(p, sizeof(p), "decoder.up_blocks.%d.upsamplers.0.upscale_conv", i);
            /* 1x1x1 conv (O, 1, 1, 1, I) == linear [O, I] */
            char name[300];
            snprintf(name, sizeof(name), "%s.weight", p);
            d->up_w[i] = load_f16_f32(m->gpu, h, name, (uint64_t)rev[i] * 4 * tf * rev[i], error, error_size);
            snprintf(name, sizeof(name), "%s.bias", p);
            d->up_b[i] = d->up_w[i] ? load_f16_f32(m->gpu, h, name, (uint64_t)rev[i] * 4 * tf, error, error_size) : NULL;
            if (!d->up_b[i]) return 0;
            if (m->nax_vae) {
                sv2_conv *u = &d->up_nax[i];
                u->in = rev[i]; u->out = rev[i] * 4 * tf; u->kernel = 1; u->b = d->up_b[i];
                float *wu = malloc((size_t)u->out * u->in * sizeof(float));
                if (!wu || !h3_gpu_tensor_read_f32(d->up_w[i], wu, (size_t)u->out * u->in) ||
                    !pack_nax_conv(m, u, wu, error, error_size)) { free(wu); return 0; }
                free(wu);
                u->b = NULL;   /* owned by up_b */
            }
            snprintf(p, sizeof(p), "decoder.up_blocks.%d.upsamplers.0.conv", i);
            if (!load_conv(m, h, p, rev[i], rev[i], 3, &d->up_conv[i], error, error_size)) return 0;
        }
    }
    if (!load_norm(m, h, "decoder.conv_norm_out", 128, &d->norm_out, error, error_size) ||
        !load_conv(m, h, "decoder.conv_out", 3, 128, 3, &d->conv_out, error, error_size))
        return 0;
    return 1;
}

static void free_conv(sv2_conv *c) {
    h3_gpu_tensor_free(c->w); h3_gpu_tensor_free(c->b);
    h3_gpu_tensor_free(c->nw); h3_gpu_tensor_free(c->nb);
    memset(c, 0, sizeof(*c));
}
static void free_norm(sv2_norm *n) { h3_gpu_tensor_free(n->w); h3_gpu_tensor_free(n->b); memset(n, 0, sizeof(*n)); }
static void free_resnet(sv2_resnet *r) {
    free_norm(&r->norm1); free_norm(&r->norm2);
    free_conv(&r->conv1); free_conv(&r->conv2); free_conv(&r->shortcut);
}
static void free_attention(sv2_attention *a) {
    free_norm(&a->norm);
    h3_gpu_tensor *t[8] = {a->wq, a->bq, a->wk, a->bk, a->wv, a->bv, a->wo, a->bo};
    for (int i = 0; i < 8; i++) h3_gpu_tensor_free(t[i]);
    memset(a, 0, sizeof(*a));
}

static void free_vae(h3_seedvr2 *m) {
    sv2_encoder *e = &m->enc;
    sv2_decoder *d = &m->dec;
    free_conv(&e->conv_in);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) free_resnet(&e->down[i][j]);
    for (int i = 0; i < 3; i++) free_conv(&e->downsample[i]);
    free_resnet(&e->mid_a); free_resnet(&e->mid_b); free_attention(&e->mid_attn);
    free_norm(&e->norm_out); free_conv(&e->conv_out);
    free_conv(&d->conv_in);
    free_resnet(&d->mid_a); free_resnet(&d->mid_b); free_attention(&d->mid_attn);
    for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) free_resnet(&d->up[i][j]);
    for (int i = 0; i < 3; i++) {
        h3_gpu_tensor_free(d->up_w[i]); h3_gpu_tensor_free(d->up_b[i]);
        free_conv(&d->up_conv[i]); free_conv(&d->up_nax[i]);
    }
    free_norm(&d->norm_out); free_conv(&d->conv_out);
}

/* --- activation pool --- */

/* Best fit: the smallest pooled buffer of the dtype that holds `elements`
 * (a larger one is fine: every kernel takes explicit sizes). */
static h3_gpu_tensor *pool_take(h3_seedvr2 *m, size_t elements, int bf16) {
    size_t bytes = elements * (bf16 ? 2 : 4);
    int best = -1;
    for (int i = 0; i < m->pool_count; i++) {
        if (m->pool[i].bf16 != bf16 || m->pool[i].bytes < bytes) continue;
        if (best < 0 || m->pool[i].bytes < m->pool[best].bytes) best = i;
    }
    if (best >= 0 && m->pool[best].bytes <= bytes * 2) {
        h3_gpu_tensor *t = m->pool[best].t;
        m->pool[best] = m->pool[--m->pool_count];
        return t;
    }
    return bf16 ? h3_gpu_tensor_new_bf16(m->gpu, elements) : h3_gpu_tensor_new_f32(m->gpu, elements);
}

static void pool_give(h3_seedvr2 *m, h3_gpu_tensor *t) {
    if (!t) return;
    int bf16 = h3_gpu_tensor_dtype(t) == H3_GPU_BF16;
    size_t bytes = h3_gpu_tensor_elements(t) * (bf16 ? 2 : 4);
    if (m->pool_count < (int)(sizeof(m->pool) / sizeof(*m->pool))) {
        m->pool[m->pool_count].t = t;
        m->pool[m->pool_count].bytes = bytes;
        m->pool[m->pool_count].bf16 = bf16;
        m->pool_count++;
    } else {
        h3_gpu_tensor_free(t);
    }
}

static void pool_drain(h3_seedvr2 *m) {
    for (int i = 0; i < m->pool_count; i++) h3_gpu_tensor_free(m->pool[i].t);
    m->pool_count = 0;
}

/* ---------------------------------------------------------------- */
/* Model open / close / inspect */

static int read_config(const char *dir, h3_seedvr2_info *info, char *error, size_t error_size) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fail(error, error_size, "SeedVR2: cannot open %s", path); return 0; }
    char text[8192];
    size_t n = fread(text, 1, sizeof(text) - 1, f);
    fclose(f);
    text[n] = '\0';
    if (!strstr(text, "\"model_type\"") || !strstr(text, "seedvr2")) {
        fail(error, error_size, "SeedVR2: %s is not a SeedVR2 config", path);
        return 0;
    }
    const char *v = strstr(text, "\"variant\"");
    if (v) {
        v = strchr(v + 9, '"');
        if (v) {
            const char *e = strchr(v + 1, '"');
            if (e && e - v - 1 < (long)sizeof(info->variant)) {
                memcpy(info->variant, v + 1, (size_t)(e - v - 1));
                info->variant[e - v - 1] = '\0';
            }
        }
    }
    const char *d = strstr(text, "\"dtype\"");
    if (d) {
        d = strchr(d + 7, '"');
        if (d) {
            const char *e = strchr(d + 1, '"');
            if (e && e - d - 1 < (long)sizeof(info->dtype)) {
                memcpy(info->dtype, d + 1, (size_t)(e - d - 1));
                info->dtype[e - d - 1] = '\0';
            }
        }
    }
    const char *bits = strstr(text, "\"bits\"");
    const char *group = strstr(text, "\"group_size\"");
    if (bits) info->quant_bits = atoi(bits + 6 + strspn(bits + 6, " :"));
    if (group) info->quant_group = atoi(group + 12 + strspn(group + 12, " :"));
    if (strcmp(info->variant, "seedvr2-3b") != 0) {
        fail(error, error_size, "SeedVR2: variant \"%s\" is not supported (seedvr2-3b only)", info->variant);
        return 0;
    }
    if (info->quant_bits && (info->quant_bits != 8 || info->quant_group != SV2_QUANT_GROUP)) {
        fail(error, error_size, "SeedVR2: %d-bit group %d weights are not supported (8-bit, group 64, or fp16)",
             info->quant_bits, info->quant_group);
        return 0;
    }
    return 1;
}

int h3_seedvr2_inspect_dir(const char *model_dir, h3_seedvr2_info *info,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model_dir || !info) return 0;
    memset(info, 0, sizeof(*info));
    snprintf(info->name, sizeof(info->name), "SeedVR2-3B");
    if (!read_config(model_dir, info, error, error_size)) return 0;
    sv2_files f;
    if (!files_open(&f, model_dir, error, error_size)) return 0;
    info->file_bytes = f.transformer.file_size + f.vae.file_size + f.pos.file_size;
    /* parameters: the BF16 transformer (the shared blocks' text copies are
     * not loaded) and the F32 VAE (temporal taps folded: a third) */
    uint64_t mm = (uint64_t)SV2_MM_LAYERS * (2 * (3 + 1) * SV2_DIM * SV2_DIM + 2 * 3 * (uint64_t)SV2_HIDDEN * SV2_DIM);
    uint64_t shared = (uint64_t)(SV2_LAYERS - SV2_MM_LAYERS) * ((3 + 1) * SV2_DIM * SV2_DIM + 3 * (uint64_t)SV2_HIDDEN * SV2_DIM);
    uint64_t other = (uint64_t)SV2_DIM * SV2_PATCH_IN + (uint64_t)SV2_PATCH_OUT * SV2_DIM;
    uint64_t vae_params = 0;
    for (size_t i = 0; i < f.vae.tensor_count; i++) {
        const h3_st_tensor *t = &f.vae.tensors[i];
        uint64_t n = h3_st_tensor_elements(t);
        if (t->ndim == 5 && t->shape[1] == 3) n /= 3;
        vae_params += n;
    }
    info->resident_bytes = (mm + shared + other) * 2 + vae_params * 4;
    files_close(&f);
    h3_device_info device;
    char probe[256];
    info->tensor_ops = h3_metal_probe(&device, probe, sizeof(probe)) && device.apple_gpu_family >= 10;
    return 1;
}

void h3_seedvr2_close(h3_seedvr2 *m) {
    if (!m) return;
    if (m->gpu) {
        free_linear(&m->vid_in); free_linear(&m->vid_out);
        h3_gpu_tensor_free(m->txt0);
        for (int b = 0; b < SV2_LAYERS; b++) free_block(&m->blocks[b]);
        h3_gpu_tensor_free(m->out_scale); h3_gpu_tensor_free(m->out_shift);
        free_vae(m);
        pool_drain(m);
        for (int i = 0; i < m->deferred_count; i++) h3_gpu_tensor_free(m->deferred[i]);
        h3_gpu_free(m->gpu);
    }
    free(m);
}

h3_seedvr2 *h3_seedvr2_open(const char *model_dir, const h3_seedvr2_open_params *params,
                            h3_progress_callback on_progress, void *opaque,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!model_dir) { fail(error, error_size, "SeedVR2: a model directory is required"); return NULL; }
    h3_seedvr2 *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    if (params) m->params = *params;
    else { m->params.color_transfer = 1; m->params.nax_linear = -1; m->params.seed = 42; }
    snprintf(m->model_dir, sizeof(m->model_dir), "%s", model_dir);
    if (!h3_seedvr2_inspect_dir(model_dir, &m->info, error, error_size)) { free(m); return NULL; }
    m->gpu = h3_gpu_create(m->params.shader_source_path ? m->params.shader_source_path : "h3_shaders.metal",
                           error, error_size);
    if (!m->gpu) { free(m); return NULL; }
    int nax = m->params.nax_linear < 0 ? h3_gpu_has_tensor_ops(m->gpu) : m->params.nax_linear;
    if (getenv("H3_SV2_NO_NAX")) nax = 0;
    h3_gpu_set_nax_linear_generic(m->gpu, nax);
    m->nax_vae = h3_gpu_has_tensor_ops(m->gpu) && !m->params.vae_f32 && !getenv("H3_SV2_VAE_F32");
    /* int8 (per-row, on top of the export's group-wise 8-bit) costs 10 dB
     * against the reference (41.8 vs 52 dB on the 512x288 frame) for 0.8 s
     * of the frame: off unless asked (H3_SV2_INT8=1 or params.int8 = 1). */
    m->int8_dit = m->params.int8 > 0 || (m->params.int8 < 0 && getenv("H3_SV2_INT8") != NULL);
    if (m->int8_dit && !h3_gpu_has_tensor_ops(m->gpu)) m->int8_dit = 0;
    sv2_files f;
    if (!files_open(&f, model_dir, error, error_size)) { h3_seedvr2_close(m); return NULL; }
    double mark = now_seconds(), start = mark;
    int ok = 0;
    float *emb = compute_emb(&f.transformer, error, error_size);
    if (!emb) goto done;
    trace(&mark, "emb (host)");
    /* text tokens: txt_in(pos_emb), constant */
    {
        h3_gpu_begin(m->gpu);
        sv2_linear txt_in;
        float *pos = read_f16_as_f32(&f.pos, "embedding", (uint64_t)SV2_TXT_TOKENS * SV2_TXT_IN, error, error_size);
        if (!pos) goto done;
        h3_gpu_tensor *pos_t = upload_f32_as_bf16(m->gpu, pos, (size_t)SV2_TXT_TOKENS * SV2_TXT_IN);
        free(pos);
        defer_free(m, pos_t);
        if (!pos_t || !load_linear(m, &f.transformer, "txt_in", SV2_DIM, SV2_TXT_IN, 1, &txt_in, error, error_size))
            goto done;
        m->txt0 = h3_gpu_tensor_new_bf16(m->gpu, (size_t)SV2_TXT_TOKENS * SV2_DIM);
        if (!m->txt0 || !h3_gpu_linear_bf16(m->gpu, m->txt0, pos_t, txt_in.w, NULL, SV2_TXT_TOKENS,
                                           SV2_TXT_IN, SV2_DIM) ||
            !h3_gpu_vsr_bias_gate_add_bf16(m->gpu, m->txt0, m->txt0, txt_in.b, NULL, NULL, SV2_TXT_TOKENS, SV2_DIM)) {
            fail(error, error_size, "SeedVR2: txt_in: %s", h3_gpu_error(m->gpu));
            goto done;
        }
        defer_free(m, txt_in.w); defer_free(m, txt_in.b);
        if (!load_linear(m, &f.transformer, "vid_in.proj", SV2_DIM, SV2_PATCH_IN, 1, &m->vid_in, error, error_size) ||
            !load_linear(m, &f.transformer, "vid_out.proj", SV2_PATCH_OUT, SV2_DIM, 1, &m->vid_out, error, error_size))
            goto done;
        /* output modulation: norm weight * (scale_a + out_scale), shift_a + out_shift */
        float *nw = read_f16_as_f32(&f.transformer, "vid_out_norm.weight", SV2_DIM, error, error_size);
        float *osc = nw ? read_f16_as_f32(&f.transformer, "out_scale", SV2_DIM, error, error_size) : NULL;
        float *osh = osc ? read_f16_as_f32(&f.transformer, "out_shift", SV2_DIM, error, error_size) : NULL;
        if (osh) {
            for (int c = 0; c < SV2_DIM; c++) {
                osc[c] = nw[c] * (emb[c * 6 + 0 * 3 + 1] + osc[c]);
                osh[c] = emb[c * 6 + 0 * 3 + 0] + osh[c];
            }
            m->out_scale = upload_f32(m->gpu, osc, SV2_DIM);
            m->out_shift = upload_f32(m->gpu, osh, SV2_DIM);
        }
        free(nw); free(osc); free(osh);
        if (!m->out_scale || !m->out_shift) { fail(error, error_size, "SeedVR2: output modulation"); goto done; }
        if (!flush_loads(m, error, error_size)) goto done;
    }
    trace(&mark, "txt_in / vid_in / out");
    for (int b = 0; b < SV2_LAYERS; b++) {
        if (on_progress && on_progress("seedvr2 load", b, SV2_LAYERS + 4, opaque)) {
            fail(error, error_size, "vsr cancelled");
            goto done;
        }
        h3_gpu_begin(m->gpu);
        if (!load_block(m, &f.transformer, b, emb, error, error_size)) goto done;
        if (!flush_loads(m, error, error_size)) goto done;
    }
    trace(&mark, "transformer blocks");
    if (on_progress && on_progress("seedvr2 load", SV2_LAYERS, SV2_LAYERS + 4, opaque)) {
        fail(error, error_size, "vsr cancelled");
        goto done;
    }
    if (!load_vae(m, &f.vae, error, error_size)) goto done;
    trace(&mark, "vae");
    if (on_progress) on_progress("seedvr2 load", SV2_LAYERS + 4, SV2_LAYERS + 4, opaque);
    if (trace_enabled()) {
        h3_gpu_stats st;
        if (h3_gpu_get_stats(m->gpu, &st))
            fprintf(stderr, "sv2 loaded in %.1f s, %.2f GiB live on the GPU, nax linear %d, dit %s, vae %s\n",
                    now_seconds() - start, (double)st.live_bytes / 1073741824.0, nax,
                    m->int8_dit ? "int8" : "bf16", m->nax_vae ? "tensorops bf16" : "mpsgraph f32");
    }
    ok = 1;
done:
    free(emb);
    files_close(&f);
    if (!ok) { h3_seedvr2_close(m); return NULL; }
    return m;
}

int h3_seedvr2_get_stats(const h3_seedvr2 *m, double *gpu_seconds, uint64_t *peak_live_bytes) {
    h3_gpu_stats st;
    if (!m || !h3_gpu_get_stats(m->gpu, &st)) return 0;
    if (gpu_seconds) *gpu_seconds = st.gpu_seconds;
    if (peak_live_bytes) *peak_live_bytes = st.peak_live_bytes;
    return 1;
}

/* ---------------------------------------------------------------- */
/* VAE on one frame: F32 channels-last [H, W, C] planes */

typedef struct { h3_gpu_tensor *t; int h, w, c; } act;

#define GPU_OP(call) do { if (!(call)) { fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; } } while (0)

static int act_new(h3_seedvr2 *m, act *a, int h, int w, int c) {
    a->h = h; a->w = w; a->c = c;
    a->t = pool_take(m, (size_t)h * w * c, 0);
    return a->t != NULL;
}

static void act_free(h3_seedvr2 *m, act *a) {
    pool_give(m, a->t);
    a->t = NULL;
}

/* 3x3 convolution with symmetric zero padding 1 (stride 1), or stride 2
 * with padding (0, 1) right/bottom (the downsampler); the padding is
 * applied inside the convolution graph. */
static int conv3x3(h3_seedvr2 *m, const act *x, const sv2_conv *c, int stride, act *y,
                   char *error, size_t error_size) {
    uint32_t top = stride == 2 ? 0 : 1, left = stride == 2 ? 0 : 1;
    int ph = x->h + (int)top + 1, pw = x->w + (int)left + 1;
    int oh = (ph - 3) / stride + 1, ow = (pw - 3) / stride + 1;
    if (!act_new(m, y, oh, ow, c->out)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    if (!h3_gpu_conv2d_padded_f32(m->gpu, y->t, x->t, c->w, c->b, (uint32_t)x->h, (uint32_t)x->w,
                                  (uint32_t)c->in, (uint32_t)c->out, 3, 3, (uint32_t)stride, (uint32_t)stride,
                                  top, 1, left, 1)) {
        fail(error, error_size, "SeedVR2 VAE conv: %s", h3_gpu_error(m->gpu));
        return 0;
    }
    return 1;
}

/* 1x1 convolution = linear over the pixels. */
static int conv1x1(h3_seedvr2 *m, const act *x, const h3_gpu_tensor *w, const h3_gpu_tensor *b,
                   int out, act *y, char *error, size_t error_size) {
    if (!act_new(m, y, x->h, x->w, out)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_linear_f32(m->gpu, y->t, x->t, w, b, (uint32_t)(x->h * x->w), (uint32_t)x->c, (uint32_t)out));
    return 1;
}

/* in_place: y reuses x's buffer (the kernel reads every element before it
 * writes it; x must not be needed afterwards). */
static int group_norm_silu(h3_seedvr2 *m, const act *x, const sv2_norm *n, act *y, int in_place,
                           char *error, size_t error_size) {
    if (in_place) *y = *x;
    else if (!act_new(m, y, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_vae_encoder_group_norm_silu_chunked_f32(m->gpu, y->t, x->t, n->w, n->b, 1, 1,
                                                          (uint32_t)x->h, (uint32_t)x->w, (uint32_t)x->c,
                                                          SV2_VAE_GROUPS, SV2_VAE_EPS));
    return 1;
}

static int resnet(h3_seedvr2 *m, const act *x, const sv2_resnet *r, act *y, char *error, size_t error_size) {
    act h1, c1, h2, c2, sc;
    if (!group_norm_silu(m, x, &r->norm1, &h1, 0, error, error_size)) return 0;
    int ok = conv3x3(m, &h1, &r->conv1, 1, &c1, error, error_size);
    act_free(m, &h1);
    if (!ok) return 0;
    ok = group_norm_silu(m, &c1, &r->norm2, &h2, 1, error, error_size);   /* in place */
    if (!ok) { act_free(m, &c1); return 0; }
    ok = conv3x3(m, &h2, &r->conv2, 1, &c2, error, error_size);
    act_free(m, &h2);
    if (!ok) return 0;
    const act *residual = x;
    if (r->shortcut.w) {
        if (!conv1x1(m, x, r->shortcut.w, r->shortcut.b, r->out, &sc, error, error_size)) { act_free(m, &c2); return 0; }
        residual = &sc;
    }
    *y = c2;   /* in place: y = c2 + residual */
    ok = h3_gpu_add_scaled_f32(m->gpu, y->t, c2.t, residual->t, 1.0f, 1.0f, (uint32_t)((size_t)y->h * y->w * y->c));
    if (r->shortcut.w) act_free(m, &sc);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
    return 1;
}

static int attention(h3_seedvr2 *m, const act *x, const sv2_attention *a, act *y,
                     char *error, size_t error_size) {
    int C = x->c;
    uint32_t n = (uint32_t)(x->h * x->w);
    act g, q, k, v, o, p;
    if (!act_new(m, &g, x->h, x->w, C)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    GPU_OP(h3_gpu_sv2_group_norm_f32(m->gpu, g.t, x->t, a->norm.w, a->norm.b, 1, (uint32_t)x->h, (uint32_t)x->w,
                                     (uint32_t)C, SV2_VAE_GROUPS, SV2_VAE_EPS));
    int ok = conv1x1(m, &g, a->wq, a->bq, C, &q, error, error_size) &&
             conv1x1(m, &g, a->wk, a->bk, C, &k, error, error_size) &&
             conv1x1(m, &g, a->wv, a->bv, C, &v, error, error_size);
    act_free(m, &g);
    if (!ok) return 0;
    if (!act_new(m, &o, x->h, x->w, C)) { fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
    ok = h3_gpu_sdpa_f32(m->gpu, o.t, q.t, k.t, v.t, n, 1, (uint32_t)C, 1.0f / sqrtf((float)C));
    act_free(m, &q); act_free(m, &k); act_free(m, &v);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE attention: %s", h3_gpu_error(m->gpu)); act_free(m, &o); return 0; }
    ok = conv1x1(m, &o, a->wo, a->bo, C, &p, error, error_size);
    act_free(m, &o);
    if (!ok) return 0;
    *y = p;
    GPU_OP(h3_gpu_add_scaled_f32(m->gpu, y->t, p.t, x->t, 1.0f, 1.0f, (uint32_t)((size_t)n * C)));
    return 1;
}

/* Encoder: RGB [H, W, 3] in [-1, 1] (multiples of 16) -> latent [h, w, 32]
 * (mean in the first 16 channels, unscaled). */
/* ---- The same VAE on the TensorOps grid (M5): BF16 rows [guard | (H+2) x
 * (W+2) | tail] with channels padded to 64; every convolution is one
 * h3_gpu_vsr_tapconv_bf16 (3x3: 9 taps over the padded frame, 1x1: 1 tap),
 * the residual added inside it; GroupNorm + SiLU in three chunked passes
 * on the grid; the attention block goes through the F32 path. ---- */

typedef struct { h3_gpu_tensor *t; int h, w, c, cp; uint32_t rows, guard, x_rows; } gact;

static int gact_new(h3_seedvr2 *m, gact *a, int h, int w, int c) {
    memset(a, 0, sizeof(*a));
    a->h = h; a->w = w; a->c = c; a->cp = round64(c);
    uint32_t wp = (uint32_t)w + 2;
    a->rows = (uint32_t)(h + 2) * wp;
    a->guard = (wp + 3 + 127) / 128 * 128;
    a->x_rows = a->guard + a->rows + a->guard + 128;
    a->t = pool_take(m, (size_t)a->x_rows * a->cp, 1);
    if (!a->t) return 0;
    return h3_gpu_vsr_zero_rows_bf16(m->gpu, a->t, 0, a->guard, (uint32_t)a->cp) &&
           h3_gpu_vsr_zero_rows_bf16(m->gpu, a->t, a->guard + a->rows, a->x_rows - a->guard - a->rows, (uint32_t)a->cp);
}

static void gact_free(h3_seedvr2 *m, gact *a) { pool_give(m, a->t); a->t = NULL; }

static void gact_offsets(const gact *a, int k, int32_t *offsets) {
    for (int dy = 0; dy < k; dy++)
        for (int dx = 0; dx < k; dx++)
            offsets[dy * k + dx] = (dy - k / 2) * (a->w + 2) + (dx - k / 2);
}

#define GRID_OP(call) do { if (!(call)) { fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; } } while (0)

/* out = conv(x) + bias (+ residual), same geometry. */
static int gconv(h3_seedvr2 *m, const gact *x, const sv2_conv *c, const gact *residual, gact *y,
                 char *error, size_t error_size) {
    if (x->cp != c->in_p) { fail(error, error_size, "SeedVR2 VAE (grid): %d padded channels, the conv wants %d", x->cp, c->in_p); return 0; }
    if (!gact_new(m, y, x->h, x->w, c->out)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    int32_t offsets[9];
    gact_offsets(x, c->kernel, offsets);
    GRID_OP(h3_gpu_vsr_tapconv_bf16(m->gpu, y->t, x->t, NULL, c->nw, NULL, c->nb, residual ? residual->t : NULL,
                                    x->rows, x->x_rows, x->guard, (uint32_t)x->cp, (uint32_t)y->cp,
                                    (uint32_t)(c->kernel * c->kernel), offsets, 0));
    GRID_OP(h3_gpu_vsr_zero_border_bf16(m->gpu, y->t, 1, (uint32_t)y->h, (uint32_t)y->w, (uint32_t)y->cp, y->guard, y->x_rows));
    return 1;
}

/* The stride-2 3x3 convolution with padding (0, 1): the stride-1 result,
 * then its odd interior positions. */
static int gconv_s2(h3_seedvr2 *m, const gact *x, const sv2_conv *c, gact *y, char *error, size_t error_size) {
    gact full;
    if (!gconv(m, x, c, NULL, &full, error, error_size)) return 0;
    if (!gact_new(m, y, x->h / 2, x->w / 2, c->out)) { gact_free(m, &full); fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    int ok = h3_gpu_sv2_grid_select_bf16(m->gpu, y->t, (uint32_t)y->h, (uint32_t)y->w, y->guard, full.t,
                                         (uint32_t)full.h, (uint32_t)full.w, full.guard, (uint32_t)y->cp, 1, 1);
    gact_free(m, &full);
    if (!ok) { fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
    return 1;
}

static int ggn(h3_seedvr2 *m, const gact *x, const sv2_norm *n, gact *y, int in_place, int silu,
               char *error, size_t error_size) {
    if (in_place) *y = *x;
    else if (!gact_new(m, y, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_sv2_grid_group_norm_bf16(m->gpu, y->t, x->t, n->w, n->b, (uint32_t)x->h, (uint32_t)x->w,
                                            (uint32_t)x->c, (uint32_t)x->cp, x->guard, x->x_rows,
                                            SV2_VAE_GROUPS, SV2_VAE_EPS, silu));
    /* the norm writes the interior only: a pooled buffer's border rows must
     * be zero before the next convolution's taps read them */
    if (!in_place)
        GRID_OP(h3_gpu_vsr_zero_border_bf16(m->gpu, y->t, 1, (uint32_t)y->h, (uint32_t)y->w, (uint32_t)y->cp, y->guard, y->x_rows));
    return 1;
}

static int gresnet(h3_seedvr2 *m, const gact *x, const sv2_resnet *r, gact *y, char *error, size_t error_size) {
    gact h1, c1, c2, sc;
    if (!ggn(m, x, &r->norm1, &h1, 0, 1, error, error_size)) return 0;
    int ok = gconv(m, &h1, &r->conv1, NULL, &c1, error, error_size);
    gact_free(m, &h1);
    if (!ok) return 0;
    gact h2;
    if (!ggn(m, &c1, &r->norm2, &h2, 1, 1, error, error_size)) { gact_free(m, &c1); return 0; }
    const gact *residual = x;
    if (r->shortcut.nw) {
        if (!gconv(m, x, &r->shortcut, NULL, &sc, error, error_size)) { gact_free(m, &c1); return 0; }
        residual = &sc;
    }
    ok = gconv(m, &h2, &r->conv2, residual, &c2, error, error_size);
    gact_free(m, &c1);
    if (r->shortcut.nw) gact_free(m, &sc);
    if (!ok) return 0;
    *y = c2;
    return 1;
}

/* F32 [h, w, c] <-> grid. */
static int gact_pack(h3_seedvr2 *m, const act *x, gact *g, char *error, size_t error_size) {
    if (!gact_new(m, g, x->h, x->w, x->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_vsr_pack_grid_bf16(m->gpu, g->t, x->t, 1, (uint32_t)x->h, (uint32_t)x->w, (uint32_t)x->c,
                                      (uint32_t)g->cp, g->guard, g->x_rows, 0));
    return 1;
}

static int gact_unpack(h3_seedvr2 *m, const gact *g, act *x, char *error, size_t error_size) {
    if (!act_new(m, x, g->h, g->w, g->c)) { fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
    GRID_OP(h3_gpu_vsr_unpack_grid_f32(m->gpu, x->t, g->t, 1, (uint32_t)g->h, (uint32_t)g->w, (uint32_t)g->c,
                                       (uint32_t)g->cp, g->guard, g->x_rows));
    return 1;
}

/* The mid block's attention through the F32 path (small: latent size). */
static int gattention(h3_seedvr2 *m, const gact *x, const sv2_attention *a, gact *y, char *error, size_t error_size) {
    act f, o;
    if (!gact_unpack(m, x, &f, error, error_size)) return 0;
    int ok = attention(m, &f, a, &o, error, error_size);
    act_free(m, &f);
    if (!ok) return 0;
    ok = gact_pack(m, &o, y, error, error_size);
    act_free(m, &o);
    return ok;
}

static int vae_encode_grid(h3_seedvr2 *m, const act *rgb, act *out, int stop, char *error, size_t error_size) {
    sv2_encoder *e = &m->enc;
    gact in, x, y;
    double mark = now_seconds();
    if (!gact_pack(m, rgb, &in, error, error_size)) return 0;
    int ok = gconv(m, &in, &e->conv_in, NULL, &x, error, error_size);
    gact_free(m, &in);
    if (!ok) return 0;
    if (stop == 0) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            if (!gresnet(m, &x, &e->down[i][j], &y, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
        }
        if (i < 3) {
            if (!gconv_s2(m, &x, &e->downsample[i], &y, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
        }
        if (stop == i + 1) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    }
    trace_sync(m->gpu, &mark, "enc down (grid)");
    if (!gresnet(m, &x, &e->mid_a, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gattention(m, &x, &e->mid_attn, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gresnet(m, &x, &e->mid_b, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (stop == 5) { ok = gact_unpack(m, &x, out, error, error_size); gact_free(m, &x); return ok; }
    if (!ggn(m, &x, &e->norm_out, &y, 1, 1, error, error_size)) { gact_free(m, &x); return 0; }
    x = y;
    if (!gconv(m, &x, &e->conv_out, NULL, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x);
    ok = gact_unpack(m, &y, out, error, error_size);
    gact_free(m, &y);
    trace_sync(m->gpu, &mark, "enc mid/out (grid)");
    return ok;
}

static int vae_decode_grid(h3_seedvr2 *m, const act *z, act *out, char *error, size_t error_size) {
    sv2_decoder *d = &m->dec;
    gact in, x, y;
    double mark = now_seconds();
    if (!gact_pack(m, z, &in, error, error_size)) return 0;
    int ok = gconv(m, &in, &d->conv_in, NULL, &x, error, error_size);
    gact_free(m, &in);
    if (!ok) return 0;
    if (!gresnet(m, &x, &d->mid_a, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gattention(m, &x, &d->mid_attn, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    if (!gresnet(m, &x, &d->mid_b, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x); x = y;
    trace_sync(m->gpu, &mark, "dec mid (grid)");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (!gresnet(m, &x, &d->up[i][j], &y, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x); x = y;
        }
        if (i < 3) {
            int C = x.c, tf = d->up_temporal[i];
            gact u, s;
            if (!gconv(m, &x, &d->up_nax[i], NULL, &u, error, error_size)) { gact_free(m, &x); return 0; }
            gact_free(m, &x);
            if (!gact_new(m, &s, 2 * u.h, 2 * u.w, C)) { gact_free(m, &u); fail(error, error_size, "SeedVR2 VAE (grid): out of memory"); return 0; }
            ok = h3_gpu_sv2_grid_shuffle_bf16(m->gpu, s.t, (uint32_t)s.cp, s.guard, u.t, (uint32_t)u.h, (uint32_t)u.w,
                                              (uint32_t)u.cp, u.guard, (uint32_t)C, (uint32_t)tf);
            gact_free(m, &u);
            if (!ok) { gact_free(m, &s); fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
            if (!gconv(m, &s, &d->up_conv[i], NULL, &x, error, error_size)) { gact_free(m, &s); return 0; }
            gact_free(m, &s);
            if (!h3_gpu_continue(m->gpu)) { gact_free(m, &x); fail(error, error_size, "SeedVR2 VAE (grid): %s", h3_gpu_error(m->gpu)); return 0; }
        }
        char label[32];
        snprintf(label, sizeof(label), "dec up%d (grid)", i);
        trace_sync(m->gpu, &mark, label);
    }
    if (!ggn(m, &x, &d->norm_out, &y, 1, 1, error, error_size)) { gact_free(m, &x); return 0; }
    x = y;
    if (!gconv(m, &x, &d->conv_out, NULL, &y, error, error_size)) { gact_free(m, &x); return 0; }
    gact_free(m, &x);
    ok = gact_unpack(m, &y, out, error, error_size);
    gact_free(m, &y);
    trace_sync(m->gpu, &mark, "dec out (grid)");
    return ok;
}

/* stop: 0 after conv_in, 1..4 after down block i-1, 5 after the mid block,
 * 6 (or more) the latent. */
static int vae_encode_stage(h3_seedvr2 *m, const act *rgb, act *out, int stop, char *error, size_t error_size) {
    if (m->nax_vae) return vae_encode_grid(m, rgb, out, stop, error, error_size);
    sv2_encoder *e = &m->enc;
    act x, y;
    double mark = now_seconds();
    if (!conv3x3(m, rgb, &e->conv_in, 1, &x, error, error_size)) return 0;
    if (stop == 0) { *out = x; return 1; }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            if (!resnet(m, &x, &e->down[i][j], &y, error, error_size)) { act_free(m, &x); return 0; }
            act_free(m, &x); x = y;
        }
        if (i < 3) {
            if (!conv3x3(m, &x, &e->downsample[i], 2, &y, error, error_size)) { act_free(m, &x); return 0; }
            act_free(m, &x); x = y;
        }
        if (stop == i + 1) { *out = x; return 1; }
    }
    trace_sync(m->gpu, &mark, "enc down");
    if (!resnet(m, &x, &e->mid_a, &y, error, error_size)) { act_free(m, &x); return 0; }
    act_free(m, &x); x = y;
    if (!attention(m, &x, &e->mid_attn, &y, error, error_size)) { act_free(m, &x); return 0; }
    act_free(m, &x); x = y;
    if (!resnet(m, &x, &e->mid_b, &y, error, error_size)) { act_free(m, &x); return 0; }
    act_free(m, &x); x = y;
    if (stop == 5) { *out = x; return 1; }
    if (!group_norm_silu(m, &x, &e->norm_out, &y, 1, error, error_size)) { act_free(m, &x); return 0; }
    x = y;   /* in place: the same buffer */
    if (!conv3x3(m, &x, &e->conv_out, 1, out, error, error_size)) { act_free(m, &x); return 0; }
    act_free(m, &x);
    trace_sync(m->gpu, &mark, "enc mid/out");
    return 1;
}

static int vae_encode(h3_seedvr2 *m, const act *rgb, act *out, char *error, size_t error_size) {
    return vae_encode_stage(m, rgb, out, 99, error, error_size);
}

/* Decoder: latent [h, w, 16] (already divided by the scaling factor) ->
 * RGB [8h, 8w, 3] in about [-1, 1]. */
static int vae_decode(h3_seedvr2 *m, const act *z, act *out, char *error, size_t error_size) {
    if (m->nax_vae) return vae_decode_grid(m, z, out, error, error_size);
    sv2_decoder *d = &m->dec;
    act x, y;
    double mark = now_seconds();
    if (!conv3x3(m, z, &d->conv_in, 1, &x, error, error_size)) return 0;
    if (!resnet(m, &x, &d->mid_a, &y, error, error_size)) { act_free(m, &x); return 0; }
    act_free(m, &x); x = y;
    if (!attention(m, &x, &d->mid_attn, &y, error, error_size)) { act_free(m, &x); return 0; }
    act_free(m, &x); x = y;
    if (!resnet(m, &x, &d->mid_b, &y, error, error_size)) { act_free(m, &x); return 0; }
    act_free(m, &x); x = y;
    trace_sync(m->gpu, &mark, "dec mid");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (!resnet(m, &x, &d->up[i][j], &y, error, error_size)) { act_free(m, &x); return 0; }
            act_free(m, &x); x = y;
        }
        if (i < 3) {
            int C = x.c, tf = d->up_temporal[i];
            act u, s;
            if (!conv1x1(m, &x, d->up_w[i], d->up_b[i], C * 4 * tf, &u, error, error_size)) { act_free(m, &x); return 0; }
            act_free(m, &x);
            if (!act_new(m, &s, 2 * u.h, 2 * u.w, C)) { act_free(m, &u); fail(error, error_size, "SeedVR2 VAE: out of memory"); return 0; }
            int ok = h3_gpu_sv2_shuffle_f32(m->gpu, s.t, u.t, (uint32_t)u.h, (uint32_t)u.w, (uint32_t)C, (uint32_t)tf);
            act_free(m, &u);
            if (!ok) { act_free(m, &s); fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
            if (!conv3x3(m, &s, &d->up_conv[i], 1, &x, error, error_size)) { act_free(m, &s); return 0; }
            act_free(m, &s);
            if (!h3_gpu_continue(m->gpu)) { act_free(m, &x); fail(error, error_size, "SeedVR2 VAE: %s", h3_gpu_error(m->gpu)); return 0; }
        }
        char label[32];
        snprintf(label, sizeof(label), "dec up%d", i);
        trace_sync(m->gpu, &mark, label);
    }
    if (!group_norm_silu(m, &x, &d->norm_out, &y, 1, error, error_size)) { act_free(m, &x); return 0; }
    x = y;   /* in place: the same buffer */
    if (!conv3x3(m, &x, &d->conv_out, 1, out, error, error_size)) { act_free(m, &x); return 0; }
    act_free(m, &x);
    trace_sync(m->gpu, &mark, "dec out");
    return 1;
}

/* ---------------------------------------------------------------- */
/* Transformer: window layout and the 32-block forward on one frame */

#define SV2_MAX_CLASSES 64

typedef struct { int hs, he, ws, we; } sv2_window;

typedef struct {
    int tokens_w, tokens_h, N;
    int windows;                   /* W */
    int rows;                      /* every window's video rows + text rows */
    int classes;
    struct { int first, count, length, row_offset; } cls[SV2_MAX_CLASSES];
    h3_gpu_tensor *index[SV2_MAX_CLASSES], *cosine[SV2_MAX_CLASSES], *sine[SV2_MAX_CLASSES],
                  *dest[SV2_MAX_CLASSES];
    h3_gpu_tensor *q[SV2_MAX_CLASSES], *k[SV2_MAX_CLASSES], *v[SV2_MAX_CLASSES], *a[SV2_MAX_CLASSES];
    h3_gpu_tensor *base;           /* U32 [W]: N + w * 58, the text rows' home in `att` */
} sv2_layout;

/* mflux WindowPartitioner._make_windows for one frame: window counts
 * (3, 3) over the grid resized to 45 x 80 tokens, shifted by half a window
 * in the odd blocks. Returns the count. */
static int make_windows(int h, int w, int shifted, sv2_window *out, int capacity) {
    double scale = sqrt((45.0 * 80.0) / ((double)h * w));
    int rh = (int)lround(h * scale), rw = (int)lround(w * scale);
    /* Python's round() is half-to-even; lround is half-away-from-zero. */
    double fh = h * scale, fw = w * scale;
    if (fabs(fh - floor(fh) - 0.5) < 1e-9) rh = ((int)floor(fh) % 2 == 0) ? (int)floor(fh) : (int)floor(fh) + 1;
    if (fabs(fw - floor(fw) - 0.5) < 1e-9) rw = ((int)floor(fw) % 2 == 0) ? (int)floor(fw) : (int)floor(fw) + 1;
    int wh = (rh + 2) / 3, ww = (rw + 2) / 3;
    double sh = 0, sw = 0;
    int nh, nw;
    if (shifted) {
        sh = wh < h ? 0.5 : 0.0;
        sw = ww < w ? 0.5 : 0.0;
        nh = sh > 0 ? (int)ceil((h - sh) / wh) + 1 : 1;
        nw = sw > 0 ? (int)ceil((w - sw) / ww) + 1 : 1;
    } else {
        nh = (h + wh - 1) / wh;
        nw = (w + ww - 1) / ww;
    }
    int count = 0;
    for (int iw = 0; iw < nw; iw++) {
        int ws = (int)((iw - sw) * ww), we = (int)((iw - sw + 1) * ww);
        if (ws < 0) ws = 0;
        if (we > w) we = w;
        if (we <= ws) continue;
        for (int ih = 0; ih < nh; ih++) {
            int hs = (int)((ih - sh) * wh), he = (int)((ih - sh + 1) * wh);
            if (hs < 0) hs = 0;
            if (he > h) he = h;
            if (he <= hs) continue;
            if (count < capacity) out[count] = (sv2_window){hs, he, ws, we};
            count++;
        }
    }
    return count;
}

/* The RoPE angle tables of one gathered row. */
static void rope_row(float *cosine, float *sine, int pos_t, int pos_h, int pos_w) {
    int pos[3] = {pos_t, pos_h, pos_w};
    for (int p = 0; p < SV2_ROPE_PAIRS; p++) {
        int axis = p / SV2_ROPE_FREQS, f = p % SV2_ROPE_FREQS;
        float freq = 1.0f / powf(10000.0f, (float)(2 * f) / (float)(2 * SV2_ROPE_FREQS));
        float angle = (float)pos[axis] * freq;
        cosine[p] = cosf(angle);
        sine[p] = sinf(angle);
    }
}

static void layout_free(h3_seedvr2 *m, sv2_layout *L) {
    (void)m;
    for (int c = 0; c < L->classes; c++) {
        h3_gpu_tensor_free(L->index[c]); h3_gpu_tensor_free(L->cosine[c]);
        h3_gpu_tensor_free(L->sine[c]); h3_gpu_tensor_free(L->dest[c]);
        h3_gpu_tensor_free(L->q[c]); h3_gpu_tensor_free(L->k[c]);
        h3_gpu_tensor_free(L->v[c]); h3_gpu_tensor_free(L->a[c]);
    }
    h3_gpu_tensor_free(L->base);
    memset(L, 0, sizeof(*L));
}

static int layout_build(h3_seedvr2 *m, sv2_layout *L, int tokens_w, int tokens_h, int shifted,
                        char *error, size_t error_size) {
    memset(L, 0, sizeof(*L));
    L->tokens_w = tokens_w; L->tokens_h = tokens_h; L->N = tokens_w * tokens_h;
    sv2_window windows[1024];
    int W = make_windows(tokens_h, tokens_w, shifted, windows, 1024);
    if (W <= 0 || W > 1024) { fail(error, error_size, "SeedVR2: %d windows for a %dx%d grid", W, tokens_w, tokens_h); return 0; }
    /* order windows by length (descending, stable) so equal lengths are consecutive */
    int order[1024];
    for (int i = 0; i < W; i++) order[i] = i;
    for (int i = 1; i < W; i++) {
        int j = i, key = order[i];
        int len_key = (windows[key].he - windows[key].hs) * (windows[key].we - windows[key].ws);
        while (j > 0) {
            const sv2_window *o = &windows[order[j - 1]];
            if ((o->he - o->hs) * (o->we - o->ws) >= len_key) break;
            order[j] = order[j - 1];
            j--;
        }
        order[j] = key;
    }
    L->windows = W;
    int total_rows = 0;
    for (int i = 0; i < W; i++) {
        const sv2_window *o = &windows[order[i]];
        total_rows += (o->he - o->hs) * (o->we - o->ws) + SV2_TXT_TOKENS;
    }
    L->rows = total_rows;
    uint32_t *base = malloc((size_t)W * sizeof(uint32_t));
    if (!base) { fail(error, error_size, "SeedVR2: out of memory"); return 0; }
    int i = 0, row_offset = 0;
    while (i < W) {
        const sv2_window *o = &windows[order[i]];
        int vid_len = (o->he - o->hs) * (o->we - o->ws);
        int length = vid_len + SV2_TXT_TOKENS;
        int count = 1;
        while (i + count < W) {
            const sv2_window *p = &windows[order[i + count]];
            if ((p->he - p->hs) * (p->we - p->ws) != vid_len) break;
            count++;
        }
        if (L->classes >= SV2_MAX_CLASSES) { free(base); fail(error, error_size, "SeedVR2: too many window classes"); return 0; }
        int c = L->classes++;
        L->cls[c].first = i; L->cls[c].count = count; L->cls[c].length = length; L->cls[c].row_offset = row_offset;
        size_t rows = (size_t)count * length;
        int32_t *index = malloc(rows * sizeof(int32_t));
        uint32_t *dest = malloc(rows * sizeof(uint32_t));
        float *cosine = malloc(rows * SV2_ROPE_PAIRS * sizeof(float));
        float *sine = malloc(rows * SV2_ROPE_PAIRS * sizeof(float));
        if (!index || !dest || !cosine || !sine) {
            free(index); free(dest); free(cosine); free(sine); free(base);
            fail(error, error_size, "SeedVR2: out of memory");
            return 0;
        }
        size_t r = 0;
        for (int wi = 0; wi < count; wi++) {
            int wg = i + wi;   /* global window number (in sorted order) */
            const sv2_window *win = &windows[order[wg]];
            for (int hh = win->hs; hh < win->he; hh++)
                for (int xx = win->ws; xx < win->we; xx++) {
                    index[r] = hh * tokens_w + xx;
                    dest[r] = (uint32_t)(hh * tokens_w + xx);
                    rope_row(cosine + r * SV2_ROPE_PAIRS, sine + r * SV2_ROPE_PAIRS,
                             SV2_TXT_TOKENS, hh - win->hs, xx - win->ws);
                    r++;
                }
            for (int j = 0; j < SV2_TXT_TOKENS; j++) {
                index[r] = -1 - j;
                dest[r] = (uint32_t)(L->N + wg * SV2_TXT_TOKENS + j);
                rope_row(cosine + r * SV2_ROPE_PAIRS, sine + r * SV2_ROPE_PAIRS, j, j, j);
                r++;
            }
            base[wg] = (uint32_t)(L->N + wg * SV2_TXT_TOKENS);
        }
        L->index[c] = h3_gpu_tensor_from_u32(m->gpu, (const uint32_t *)index, rows);
        L->dest[c] = h3_gpu_tensor_from_u32(m->gpu, dest, rows);
        L->cosine[c] = h3_gpu_tensor_from_f32(m->gpu, cosine, rows * SV2_ROPE_PAIRS);
        L->sine[c] = h3_gpu_tensor_from_f32(m->gpu, sine, rows * SV2_ROPE_PAIRS);
        free(index); free(dest); free(cosine); free(sine);
        L->q[c] = h3_gpu_tensor_new_bf16(m->gpu, rows * SV2_DIM);
        L->k[c] = h3_gpu_tensor_new_bf16(m->gpu, rows * SV2_DIM);
        L->v[c] = h3_gpu_tensor_new_bf16(m->gpu, rows * SV2_DIM);
        L->a[c] = h3_gpu_tensor_new_bf16(m->gpu, rows * SV2_DIM);
        if (!L->index[c] || !L->dest[c] || !L->cosine[c] || !L->sine[c] || !L->q[c] || !L->k[c] || !L->v[c] || !L->a[c]) {
            free(base);
            fail(error, error_size, "SeedVR2: out of GPU memory for the attention windows");
            return 0;
        }
        row_offset += (int)rows;
        i += count;
    }
    L->base = h3_gpu_tensor_from_u32(m->gpu, base, (size_t)W);
    free(base);
    if (!L->base) { fail(error, error_size, "SeedVR2: out of GPU memory"); return 0; }
    return 1;
}

/* A linear of the transformer: the int8 kernel when the weights were
 * quantised (rows a multiple of 128, the activations quantised per row
 * into q_in / q_scales), else BF16. */
static int run_lin(h3_seedvr2 *m, h3_gpu_tensor *out, const h3_gpu_tensor *in, const sv2_linear *l,
                   uint32_t rows, h3_gpu_tensor *q_in, h3_gpu_tensor *q_scales) {
    if (l->q)
        return h3_gpu_linear_int8_bf16(m->gpu, out, q_in, q_scales, in, l->q, l->scales, rows, l->in, l->out, 0);
    return h3_gpu_linear_bf16(m->gpu, out, in, l->w, NULL, rows, l->in, l->out);
}

#define DIT_OP(call) do { if (!(call)) { fail(error, error_size, "SeedVR2 DiT: %s", h3_gpu_error(m->gpu)); goto failed; } } while (0)

/* tokens BF16 [N, 132] -> pred BF16 [N, 64]; layouts[0] even blocks,
 * layouts[1] odd blocks. dump_block >= 0 writes that block's video rows
 * (F32, [N, 2560]) to the file named by H3_SV2_DUMP. */
static int dit_forward(h3_seedvr2 *m, sv2_layout *layouts, const h3_gpu_tensor *tokens,
                       h3_gpu_tensor *pred, int dump_block,
                       h3_progress_callback on_progress, void *opaque,
                       const volatile int *cancel_requested, char *error, size_t error_size) {
    const uint32_t N = (uint32_t)layouts[0].N, D = SV2_DIM, T = SV2_TXT_TOKENS;
    /* The text stream's linears run on 128 rows (the last 70 zero) so the
     * TensorOps kernel takes them instead of an MPSGraph launch; the MLP is
     * two such linears around the SwiGLU kernel for the same reason. */
    const int direct = h3_gpu_has_tensor_ops(m->gpu) && !getenv("H3_SV2_NO_NAX");
    const uint32_t TR = direct ? 128 : T;
    /* int8 linears quantise per row and want 128-row multiples: the video
     * rows are allocated padded and the linears run over the padding */
    const uint32_t NP = m->int8_dit ? (N + 127u) & ~127u : N;
    const uint32_t NL = NP;   /* rows the linears see */
    const float scale = 1.0f / sqrtf((float)SV2_HEAD_DIM);
    uint32_t att_rows = N + (uint32_t)(layouts[0].windows > layouts[1].windows ? layouts[0].windows : layouts[1].windows) * T;
    if (att_rows < NP) att_rows = NP;
    h3_gpu_tensor *x_v = pool_take(m, (size_t)NP * D, 1), *h_v = pool_take(m, (size_t)NP * D, 1);
    h3_gpu_tensor *o_v = pool_take(m, (size_t)NP * D, 1), *qkv_v = pool_take(m, (size_t)NP * 3 * D, 1);
    h3_gpu_tensor *x_t = pool_take(m, (size_t)TR * D, 1), *h_t = pool_take(m, (size_t)TR * D, 1);
    h3_gpu_tensor *o_t = pool_take(m, (size_t)TR * D, 1), *qkv_t = pool_take(m, (size_t)TR * 3 * D, 1);
    h3_gpu_tensor *tm = pool_take(m, (size_t)TR * D, 1);
    h3_gpu_tensor *att = pool_take(m, (size_t)att_rows * D, 1);
    h3_gpu_tensor *f_v = direct ? pool_take(m, (size_t)NP * 2 * SV2_HIDDEN, 1) : NULL;
    h3_gpu_tensor *g_v = direct ? pool_take(m, (size_t)NP * SV2_HIDDEN, 1) : NULL;
    h3_gpu_tensor *f_t = direct ? pool_take(m, (size_t)TR * 2 * SV2_HIDDEN, 1) : NULL;
    h3_gpu_tensor *g_t = direct ? pool_take(m, (size_t)TR * SV2_HIDDEN, 1) : NULL;
    /* the int8 kernels' per-row quantised activations and scales */
    h3_gpu_tensor *q_in = m->int8_dit ? h3_gpu_tensor_new_i8(m->gpu, (size_t)NP * SV2_HIDDEN) : NULL;
    h3_gpu_tensor *q_scales = m->int8_dit ? h3_gpu_tensor_new_f32(m->gpu, NP) : NULL;
    if (!x_v || !h_v || !o_v || !qkv_v || !x_t || !h_t || !o_t || !qkv_t || !tm || !att ||
        (direct && (!f_v || !g_v || !f_t || !g_t)) || (m->int8_dit && (!q_in || !q_scales))) {
        fail(error, error_size, "SeedVR2 DiT: out of GPU memory for %u tokens", N);
        goto failed;
    }
    double mark = now_seconds();
    DIT_OP(h3_gpu_begin(m->gpu));
    DIT_OP(h3_gpu_linear_bf16(m->gpu, x_v, tokens, m->vid_in.w, NULL, N, SV2_PATCH_IN, D));
    DIT_OP(h3_gpu_vsr_bias_gate_add_bf16(m->gpu, x_v, x_v, m->vid_in.b, NULL, NULL, N, D));
    DIT_OP(h3_gpu_copy_bf16(m->gpu, x_t, 0, m->txt0, 0, (size_t)T * D));
    if (TR > T) {
        /* the padding rows stay zero through every op that reads them */
        DIT_OP(h3_gpu_vsr_zero_rows_bf16(m->gpu, x_t, T, TR - T, D));
        DIT_OP(h3_gpu_vsr_zero_rows_bf16(m->gpu, tm, T, TR - T, D));
    }
    for (int b = 0; b < SV2_LAYERS; b++) {
        const sv2_block *k = &m->blocks[b];
        sv2_layout *L = &layouts[b % 2];
        if ((cancel_requested && *cancel_requested) ||
            (on_progress && on_progress("vsr dit", b, SV2_LAYERS, opaque))) {
            fail(error, error_size, "vsr cancelled");
            goto failed;
        }
        /* attention */
        DIT_OP(h3_gpu_sv2_rms_mod_bf16(m->gpu, h_v, x_v, k->mod_vid[1], k->mod_vid[0], N, D, SV2_NORM_EPS));
        DIT_OP(h3_gpu_sv2_rms_mod_bf16(m->gpu, h_t, x_t, k->last ? NULL : k->mod_txt[1],
                                       k->last ? NULL : k->mod_txt[0], TR, D, SV2_NORM_EPS));
        DIT_OP(run_lin(m, qkv_v, h_v, &k->qkv_vid, NL, q_in, q_scales));
        DIT_OP(run_lin(m, qkv_t, h_t, &k->qkv_txt, TR, q_in, q_scales));
        for (int c = 0; c < L->classes; c++) {
            uint32_t rows = (uint32_t)(L->cls[c].count * L->cls[c].length);
            DIT_OP(h3_gpu_sv2_qkv_window_bf16(m->gpu, L->q[c], L->k[c], L->v[c], qkv_v, qkv_t, L->index[c],
                                              L->cosine[c], L->sine[c], k->nq_vid, k->nk_vid, k->nq_txt, k->nk_txt,
                                              rows, SV2_HEADS, SV2_HEAD_DIM, SV2_ROPE_PAIRS, SV2_NORM_EPS));
            DIT_OP(h3_gpu_sdpa_batched_bf16(m->gpu, L->a[c], L->q[c], L->k[c], L->v[c],
                                            (uint32_t)L->cls[c].count, (uint32_t)L->cls[c].length,
                                            SV2_HEADS, SV2_HEAD_DIM, scale));
            DIT_OP(h3_gpu_vsr_scatter_rows_bf16(m->gpu, att, L->a[c], L->dest[c], rows, D));
        }
        DIT_OP(run_lin(m, o_v, att, &k->out_vid, NL, q_in, q_scales));
        DIT_OP(h3_gpu_vsr_bias_gate_add_bf16(m->gpu, x_v, o_v, k->out_vid.b, k->mod_vid[2], x_v, N, D));
        if (!k->last) {
            DIT_OP(h3_gpu_sv2_txt_mean_bf16(m->gpu, tm, att, L->base, (uint32_t)L->windows, T, D));
            DIT_OP(run_lin(m, o_t, tm, &k->out_txt, TR, q_in, q_scales));
            DIT_OP(h3_gpu_vsr_bias_gate_add_bf16(m->gpu, x_t, o_t, k->out_txt.b, k->mod_txt[2], x_t, T, D));
        }
        /* feed-forward */
        DIT_OP(h3_gpu_sv2_rms_mod_bf16(m->gpu, h_v, x_v, k->mod_vid[4], k->mod_vid[3], N, D, SV2_NORM_EPS));
        if (direct) {
            DIT_OP(run_lin(m, f_v, h_v, &k->fc1_vid, NL, q_in, q_scales));
            DIT_OP(h3_gpu_swiglu_bf16(m->gpu, g_v, f_v, NL, SV2_HIDDEN));
            DIT_OP(run_lin(m, o_v, g_v, &k->fc2_vid, NL, q_in, q_scales));
        } else {
            DIT_OP(h3_gpu_mlp_bf16(m->gpu, o_v, h_v, k->fc1_vid.w, k->fc2_vid.w, N, D, SV2_HIDDEN, D));
        }
        DIT_OP(h3_gpu_vsr_bias_gate_add_bf16(m->gpu, x_v, o_v, NULL, k->mod_vid[5], x_v, N, D));
        if (!k->last) {
            DIT_OP(h3_gpu_sv2_rms_mod_bf16(m->gpu, h_t, x_t, k->mod_txt[4], k->mod_txt[3], TR, D, SV2_NORM_EPS));
            if (direct) {
                DIT_OP(run_lin(m, f_t, h_t, &k->fc1_txt, TR, q_in, q_scales));
                DIT_OP(h3_gpu_swiglu_bf16(m->gpu, g_t, f_t, TR, SV2_HIDDEN));
                DIT_OP(run_lin(m, o_t, g_t, &k->fc2_txt, TR, q_in, q_scales));
            } else {
                DIT_OP(h3_gpu_mlp_bf16(m->gpu, o_t, h_t, k->fc1_txt.w, k->fc2_txt.w, T, D, SV2_HIDDEN, D));
            }
            DIT_OP(h3_gpu_vsr_bias_gate_add_bf16(m->gpu, x_t, o_t, NULL, k->mod_txt[5], x_t, T, D));
        }
        DIT_OP(h3_gpu_continue(m->gpu));
        if (b == dump_block) {
            const char *path = getenv("H3_SV2_DUMP");
            if (path && h3_gpu_submit(m->gpu)) {
                size_t n = (size_t)N * D;
                uint16_t *bf = malloc(n * 2);
                float *f = malloc(n * 4);
                if (bf && f && h3_gpu_tensor_read_bf16(x_v, bf, n)) {
                    for (size_t i = 0; i < n; i++) { uint32_t bits = (uint32_t)bf[i] << 16; memcpy(&f[i], &bits, 4); }
                    FILE *fp = fopen(path, "wb");
                    if (fp) { fwrite(f, 4, n, fp); fclose(fp); }
                }
                free(bf); free(f);
                h3_gpu_begin(m->gpu);
            }
        }
        if (trace_enabled() && (b == 0 || b == SV2_LAYERS - 1)) {
            char label[32];
            snprintf(label, sizeof(label), "dit block %d encoded", b);
            trace_sync(m->gpu, &mark, label);
        }
    }
    if (on_progress && on_progress("vsr dit", SV2_LAYERS, SV2_LAYERS, opaque)) {
        fail(error, error_size, "vsr cancelled");
        goto failed;
    }
    DIT_OP(h3_gpu_sv2_rms_mod_bf16(m->gpu, h_v, x_v, m->out_scale, m->out_shift, N, D, SV2_NORM_EPS));
    DIT_OP(h3_gpu_linear_bf16(m->gpu, pred, h_v, m->vid_out.w, NULL, N, D, SV2_PATCH_OUT));
    DIT_OP(h3_gpu_vsr_bias_gate_add_bf16(m->gpu, pred, pred, m->vid_out.b, NULL, NULL, N, SV2_PATCH_OUT));
    DIT_OP(h3_gpu_submit(m->gpu));
    trace(&mark, "dit submitted + done");
    pool_give(m, x_v); pool_give(m, h_v); pool_give(m, o_v); pool_give(m, qkv_v);
    pool_give(m, x_t); pool_give(m, h_t); pool_give(m, o_t); pool_give(m, qkv_t);
    pool_give(m, tm); pool_give(m, att);
    pool_give(m, f_v); pool_give(m, g_v); pool_give(m, f_t); pool_give(m, g_t);
    h3_gpu_tensor_free(q_in); h3_gpu_tensor_free(q_scales);
    return 1;
failed:
    pool_give(m, x_v); pool_give(m, h_v); pool_give(m, o_v); pool_give(m, qkv_v);
    pool_give(m, x_t); pool_give(m, h_t); pool_give(m, o_t); pool_give(m, qkv_t);
    pool_give(m, tm); pool_give(m, att);
    pool_give(m, f_v); pool_give(m, g_v); pool_give(m, f_t); pool_give(m, g_t);
    h3_gpu_tensor_free(q_in); h3_gpu_tensor_free(q_scales);
    return 0;
}

/* The DiT's input tokens from the host latents: noise [16, h, w], cond
 * [16, h, w] (scaled encoder latent), mask 1 -> BF16 [N, 132] in the patch
 * order (ph, pw, c). */
static void pack_tokens(const float *noise, const float *cond, int lw, int lh, uint16_t *tokens) {
    int tw = lw / 2, th = lh / 2;
    for (int hp = 0; hp < th; hp++)
        for (int wp = 0; wp < tw; wp++) {
            uint16_t *row = tokens + ((size_t)hp * tw + wp) * SV2_PATCH_IN;
            for (int ph = 0; ph < 2; ph++)
                for (int pw = 0; pw < 2; pw++) {
                    size_t pixel = (size_t)(2 * hp + ph) * lw + (2 * wp + pw);
                    for (int c = 0; c < SV2_VID_IN; c++) {
                        float v = c < 16 ? noise[(size_t)c * lw * lh + pixel]
                                : c < 32 ? cond[(size_t)(c - 16) * lw * lh + pixel] : 1.0f;
                        row[(ph * 2 + pw) * SV2_VID_IN + c] = f32_to_bf16(v);
                    }
                }
        }
}

/* pred BF16 [N, 64] (order (ph, pw, c)) -> F32 [16, h, w]. */
static void unpack_pred(const uint16_t *pred, int lw, int lh, float *out) {
    int tw = lw / 2, th = lh / 2;
    for (int hp = 0; hp < th; hp++)
        for (int wp = 0; wp < tw; wp++) {
            const uint16_t *row = pred + ((size_t)hp * tw + wp) * SV2_PATCH_OUT;
            for (int ph = 0; ph < 2; ph++)
                for (int pw = 0; pw < 2; pw++)
                    for (int c = 0; c < SV2_VID_OUT; c++) {
                        uint32_t bits = (uint32_t)row[(ph * 2 + pw) * SV2_VID_OUT + c] << 16;
                        float v;
                        memcpy(&v, &bits, 4);
                        out[(size_t)c * lw * lh + (size_t)(2 * hp + ph) * lw + (2 * wp + pw)] = v;
                    }
        }
}

/* ---------------------------------------------------------------- */
/* Colour transfer (mflux SeedVR2Util.apply_color_correction): the
 * content's wavelet high frequencies over the style's low ones, then the
 * a/b channels (and 20% of L) histogram-matched to the style in L*a*b*. */

#define SV2_HIST_BINS 4096

/* Histogram matching of `src` to `ref` through binned CDFs: the LUT maps
 * each source bin to the reference value at the same quantile. */
static void hist_lut(const float *src, const float *ref, size_t n, int stride, float minimum, float step,
                     float *lut) {
    uint32_t *hs = calloc(SV2_HIST_BINS, sizeof(uint32_t)), *hr = calloc(SV2_HIST_BINS, sizeof(uint32_t));
    if (!hs || !hr) { free(hs); free(hr); for (int b = 0; b < SV2_HIST_BINS; b++) lut[b] = minimum + ((float)b + 0.5f) * step; return; }
    for (size_t i = 0; i < n; i++) {
        int bs = (int)((src[i * stride] - minimum) / step), br = (int)((ref[i * stride] - minimum) / step);
        if (bs < 0) bs = 0; if (bs >= SV2_HIST_BINS) bs = SV2_HIST_BINS - 1;
        if (br < 0) br = 0; if (br >= SV2_HIST_BINS) br = SV2_HIST_BINS - 1;
        hs[bs]++; hr[br]++;
    }
    /* cumulative counts; for each source bin, the reference bin whose CDF
     * first reaches the source bin's mid-CDF, linearly interpolated */
    uint64_t cs = 0, cr = 0;
    int rb = 0;
    uint64_t cr_prev = 0;
    for (int b = 0; b < SV2_HIST_BINS; b++) {
        uint64_t target = cs + hs[b] / 2 + (hs[b] & 1);   /* mid-rank of this bin's pixels */
        cs += hs[b];
        while (rb < SV2_HIST_BINS - 1 && cr + hr[rb] < target) { cr += hr[rb]; rb++; }
        /* position inside reference bin rb */
        float frac = hr[rb] ? (float)((double)(target - cr) / (double)hr[rb]) : 0.5f;
        if (frac < 0.0f) frac = 0.0f;
        if (frac > 1.0f) frac = 1.0f;
        lut[b] = minimum + ((float)rb + frac) * step;
        (void)cr_prev;
    }
    free(hs); free(hr);
}

/* content and style: F32 [H, W, 3] in [-1, 1] on the GPU; out may be content. */
static int color_transfer(h3_seedvr2 *m, h3_gpu_tensor *out, const h3_gpu_tensor *content,
                          const h3_gpu_tensor *style, int width, int height,
                          char *error, size_t error_size) {
    size_t n = (size_t)width * height * 3, pixels = (size_t)width * height;
    h3_gpu_tensor *high = pool_take(m, n, 0), *cur = pool_take(m, n, 0), *low = pool_take(m, n, 0);
    h3_gpu_tensor *slow = pool_take(m, n, 0), *lab_c = pool_take(m, n, 0), *lab_s = pool_take(m, n, 0);
    h3_gpu_tensor *luts = NULL;
    float *hc = NULL, *hsy = NULL, *lut = NULL;
    int ok = 0;
    if (!high || !cur || !low || !slow || !lab_c || !lab_s) { fail(error, error_size, "SeedVR2 colour: out of memory"); goto done; }
    double mark = now_seconds();
    GPU_OP(h3_gpu_begin(m->gpu));
    /* content: high = sum over 5 levels of (cur - blur(cur)) */
    GPU_OP(h3_gpu_zero_f32(m->gpu, high, n));
    GPU_OP(h3_gpu_copy_f32(m->gpu, cur, 0, content, 0, n));
    for (int level = 0; level < 5; level++) {
        int radius = 1 << level;
        int max_radius = (width < height ? width : height) / 8;
        if (max_radius < 1) max_radius = 1;
        if (radius > max_radius) radius = max_radius;
        GPU_OP(h3_gpu_sv2_wavelet_blur_f32(m->gpu, low, cur, (uint32_t)height, (uint32_t)width, radius));
        GPU_OP(h3_gpu_add_scaled_f32(m->gpu, slow, cur, low, 1.0f, -1.0f, (uint32_t)n));   /* slow as scratch */
        GPU_OP(h3_gpu_add_scaled_f32(m->gpu, high, high, slow, 1.0f, 1.0f, (uint32_t)n));
        h3_gpu_tensor *t = cur; cur = low; low = t;
    }
    /* style: low after 5 levels */
    GPU_OP(h3_gpu_copy_f32(m->gpu, slow, 0, style, 0, n));
    for (int level = 0; level < 5; level++) {
        int radius = 1 << level;
        int max_radius = (width < height ? width : height) / 8;
        if (max_radius < 1) max_radius = 1;
        if (radius > max_radius) radius = max_radius;
        GPU_OP(h3_gpu_sv2_wavelet_blur_f32(m->gpu, low, slow, (uint32_t)height, (uint32_t)width, radius));
        h3_gpu_tensor *t = slow; slow = low; low = t;
    }
    GPU_OP(h3_gpu_add_scaled_f32(m->gpu, cur, high, slow, 1.0f, 1.0f, (uint32_t)n));
    GPU_OP(h3_gpu_clip_f32(m->gpu, cur, cur, (uint32_t)n, -1.0f, 1.0f));
    GPU_OP(h3_gpu_sv2_rgb_to_lab_f32(m->gpu, lab_c, cur, (uint32_t)pixels));
    GPU_OP(h3_gpu_sv2_rgb_to_lab_f32(m->gpu, lab_s, style, (uint32_t)pixels));
    GPU_OP(h3_gpu_submit(m->gpu));
    trace(&mark, "colour wavelet + lab");
    hc = malloc(n * sizeof(float)); hsy = malloc(n * sizeof(float));
    lut = malloc(3 * SV2_HIST_BINS * sizeof(float));
    if (!hc || !hsy || !lut || !h3_gpu_tensor_read_f32(lab_c, hc, n) || !h3_gpu_tensor_read_f32(lab_s, hsy, n)) {
        fail(error, error_size, "SeedVR2 colour: cannot read back");
        goto done;
    }
    float minimum[3], step[3];
    for (int c = 0; c < 3; c++) {
        float lo = 1e30f, hi = -1e30f;
        for (size_t i = 0; i < pixels; i++) {
            float a = hc[i * 3 + c], b = hsy[i * 3 + c];
            if (a < lo) lo = a; if (a > hi) hi = a;
            if (b < lo) lo = b; if (b > hi) hi = b;
        }
        if (hi <= lo) hi = lo + 1e-3f;
        minimum[c] = lo;
        step[c] = (hi - lo) / (float)(SV2_HIST_BINS - 1);
        hist_lut(hc + c, hsy + c, pixels, 3, minimum[c], step[c], lut + (size_t)c * SV2_HIST_BINS);
    }
    trace(&mark, "colour histograms (host)");
    luts = h3_gpu_tensor_from_f32(m->gpu, lut, 3 * SV2_HIST_BINS);
    if (!luts) { fail(error, error_size, "SeedVR2 colour: out of memory"); goto done; }
    GPU_OP(h3_gpu_begin(m->gpu));
    GPU_OP(h3_gpu_sv2_lab_match_to_rgb_f32(m->gpu, out, lab_c, luts, (uint32_t)pixels, SV2_HIST_BINS, 0.8f, minimum, step));
    GPU_OP(h3_gpu_submit(m->gpu));
    trace(&mark, "colour apply");
    ok = 1;
done:
    free(hc); free(hsy); free(lut);
    h3_gpu_tensor_free(luts);
    pool_give(m, high); pool_give(m, cur); pool_give(m, low);
    pool_give(m, slow); pool_give(m, lab_c); pool_give(m, lab_s);
    return ok;
}

/* ---------------------------------------------------------------- */
/* Session: one output size, two window layouts, the frame's noise */

struct h3_seedvr2_session {
    h3_seedvr2 *m;
    int in_w, in_h, out_w, out_h;
    int full_w, full_h;            /* multiples of 16 */
    int lw, lh;                    /* latent grid */
    sv2_layout layouts[2];
    float *noise;                  /* [16, lh, lw] */
    h3_gpu_tensor *pixels_in;      /* I8 staging for the input frame */
    size_t pixels_in_bytes;
    h3_gpu_tensor *pixels_out;
    size_t pixels_out_bytes;
    uint16_t *tokens_host;
    float *cond_host, *pred_host, *latent_host;
    uint16_t *pred_bf16;
};

/* Box-Muller normals from a 64-bit xorshift generator. */
static void fill_noise(float *out, size_t n, uint64_t seed) {
    uint64_t s = seed ? seed * 0x9E3779B97F4A7C15ull : 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; i += 2) {
        double u[2];
        for (int j = 0; j < 2; j++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            u[j] = ((double)(s >> 11) + 0.5) / 9007199254740992.0;
        }
        double r = sqrt(-2.0 * log(u[0])), t = 2.0 * M_PI * u[1];
        out[i] = (float)(r * cos(t));
        if (i + 1 < n) out[i + 1] = (float)(r * sin(t));
    }
}

void h3_seedvr2_session_close(h3_seedvr2_session *s) {
    if (!s) return;
    layout_free(s->m, &s->layouts[0]);
    layout_free(s->m, &s->layouts[1]);
    free(s->noise);
    h3_gpu_tensor_free(s->pixels_in);
    h3_gpu_tensor_free(s->pixels_out);
    free(s->tokens_host); free(s->cond_host); free(s->pred_host); free(s->latent_host); free(s->pred_bf16);
    free(s);
}

h3_seedvr2_session *h3_seedvr2_session_open(h3_seedvr2 *m, int in_w, int in_h, int out_w, int out_h,
                                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!m || in_w < 1 || in_h < 1 || out_w < 16 || out_h < 16) { fail(error, error_size, "SeedVR2: bad sizes"); return NULL; }
    h3_seedvr2_session *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->m = m;
    s->in_w = in_w; s->in_h = in_h; s->out_w = out_w; s->out_h = out_h;
    s->full_w = round_up(out_w, SV2_PIXEL_MULTIPLE);
    s->full_h = round_up(out_h, SV2_PIXEL_MULTIPLE);
    s->lw = s->full_w / 8; s->lh = s->full_h / 8;
    int N = (s->lw / 2) * (s->lh / 2);
    if (!layout_build(m, &s->layouts[0], s->lw / 2, s->lh / 2, 0, error, error_size) ||
        !layout_build(m, &s->layouts[1], s->lw / 2, s->lh / 2, 1, error, error_size)) {
        h3_seedvr2_session_close(s);
        return NULL;
    }
    size_t latent = (size_t)SV2_LATENT * s->lw * s->lh;
    s->noise = malloc(latent * sizeof(float));
    s->cond_host = malloc(latent * sizeof(float));
    s->pred_host = malloc(latent * sizeof(float));
    s->latent_host = malloc(latent * sizeof(float));
    s->tokens_host = malloc((size_t)N * SV2_PATCH_IN * sizeof(uint16_t));
    s->pred_bf16 = malloc((size_t)N * SV2_PATCH_OUT * sizeof(uint16_t));
    if (!s->noise || !s->cond_host || !s->pred_host || !s->latent_host || !s->tokens_host || !s->pred_bf16) {
        fail(error, error_size, "SeedVR2: out of memory");
        h3_seedvr2_session_close(s);
        return NULL;
    }
    fill_noise(s->noise, latent, m->params.seed);
    return s;
}

int h3_seedvr2_check_requirements(const h3_seedvr2 *m, int in_w, int in_h, int out_w, int out_h,
                                  h3_seedvr2_requirements *out) {
    if (!m || !out) return 0;
    memset(out, 0, sizeof(*out));
    if (in_w < 16 || in_h < 16) { snprintf(out->reason, sizeof(out->reason), "the input is too small"); return 1; }
    if (out_w < 64 || out_h < 64) { snprintf(out->reason, sizeof(out->reason), "the output must be at least 64 pixels a side"); return 1; }
    if (out_w > 4096 || out_h > 4096) { snprintf(out->reason, sizeof(out->reason), "the output must be at most 4096 pixels a side"); return 1; }
    int full_w = round_up(out_w, SV2_PIXEL_MULTIPLE), full_h = round_up(out_h, SV2_PIXEL_MULTIPLE);
    int tw = full_w / 16, th = full_h / 16;
    out->tokens = tw * th;
    sv2_window windows[1024];
    out->windows_even = make_windows(th, tw, 0, windows, 1024);
    out->windows_odd = make_windows(th, tw, 1, windows, 1024);
    /* peak live GPU memory, from the measurements of appendix A.7 (M5,
     * 1344x768: 9.5 GiB on the TensorOps BF16 grid, 10.9 GiB on the F32
     * path; 1920x1088: 12.7 GiB): the resident weights plus about 3 GB
     * (BF16) or 4.5 GB (F32) per megapixel of output */
    double megapixels = (double)full_w * full_h / 1e6;
    out->gpu_bytes = m->info.resident_bytes + (uint64_t)((m->nax_vae ? 3.0e9 : 4.5e9) * megapixels);
    h3_device_info device;
    char probe[256];
    uint64_t budget = 0;
    if (h3_metal_probe(&device, probe, sizeof(probe)))
        budget = device.recommended_working_set ? device.recommended_working_set : device.physical_memory;
    budget = budget * 9 / 10;
    if (budget && out->gpu_bytes > budget) {
        snprintf(out->reason, sizeof(out->reason),
                 "about %.1f GB of GPU memory is needed for %dx%d, this Mac has %.1f GB",
                 (double)out->gpu_bytes / 1e9, out_w, out_h, (double)budget / 0.9 / 1e9);
        return 1;
    }
    /* measured on an M5 (appendix A.7): 4.1 s at 1344x768, 10.3 s at
     * 1920x1088 with the TensorOps paths; the F32/MPSGraph path took 6.1 s
     * at 1344x768 on the same GPU, and other Apple GPUs are slower still */
    out->estimated_seconds_per_frame = megapixels * (m->nax_vae ? 4.8 : 15.0);
    out->supported = 1;
    return 1;
}

/* The frame pipeline on an F32 [full_h, full_w, 3] input already in
 * [-1, 1]; writes the restored F32 frame (same layout) into `out`. */
static int restore_frame(h3_seedvr2_session *s, const h3_gpu_tensor *rgb_in, h3_gpu_tensor *rgb_out,
                         const float *noise_override, const float *cond_override,
                         int dump_block, h3_progress_callback on_progress, void *opaque,
                         const volatile int *cancel_requested, char *error, size_t error_size) {
    h3_seedvr2 *m = s->m;
    int lw = s->lw, lh = s->lh, N = (lw / 2) * (lh / 2);
    size_t latent = (size_t)SV2_LATENT * lw * lh;
    double mark = now_seconds();
    /* encode */
    if (!cond_override) {
        if (on_progress && on_progress("vsr encode", 0, 1, opaque)) { fail(error, error_size, "vsr cancelled"); return 0; }
        act in = {(h3_gpu_tensor *)rgb_in, s->full_h, s->full_w, 3}, enc;
        if (!h3_gpu_begin(m->gpu)) { fail(error, error_size, "SeedVR2: %s", h3_gpu_error(m->gpu)); return 0; }
        if (!vae_encode(m, &in, &enc, error, error_size)) { h3_gpu_submit(m->gpu); return 0; }
        if (!h3_gpu_submit(m->gpu)) { act_free(m, &enc); fail(error, error_size, "SeedVR2 encode: %s", h3_gpu_error(m->gpu)); return 0; }
        /* [lh, lw, 32] -> cond [16, lh, lw] * scaling */
        float *enc_host = malloc((size_t)lw * lh * 2 * SV2_LATENT * sizeof(float));
        int ok = enc_host && h3_gpu_tensor_read_f32(enc.t, enc_host, (size_t)lw * lh * 2 * SV2_LATENT);
        act_free(m, &enc);
        if (!ok) { free(enc_host); fail(error, error_size, "SeedVR2: cannot read the latent"); return 0; }
        for (int y = 0; y < lh; y++)
            for (int x = 0; x < lw; x++)
                for (int c = 0; c < SV2_LATENT; c++)
                    s->cond_host[(size_t)c * lw * lh + (size_t)y * lw + x] =
                        enc_host[((size_t)y * lw + x) * 2 * SV2_LATENT + c] * SV2_SCALING;
        free(enc_host);
        trace(&mark, "encode");
        if (on_progress && on_progress("vsr encode", 1, 1, opaque)) { fail(error, error_size, "vsr cancelled"); return 0; }
    } else {
        memcpy(s->cond_host, cond_override, latent * sizeof(float));
    }
    const float *noise = noise_override ? noise_override : s->noise;
    /* transformer */
    pack_tokens(noise, s->cond_host, lw, lh, s->tokens_host);
    h3_gpu_tensor *tokens = h3_gpu_tensor_from_bf16(m->gpu, s->tokens_host, (size_t)N * SV2_PATCH_IN);
    h3_gpu_tensor *pred = pool_take(m, (size_t)N * SV2_PATCH_OUT, 1);
    if (!tokens || !pred) { h3_gpu_tensor_free(tokens); pool_give(m, pred); fail(error, error_size, "SeedVR2: out of GPU memory"); return 0; }
    int ok = dit_forward(m, s->layouts, tokens, pred, dump_block, on_progress, opaque, cancel_requested, error, error_size);
    h3_gpu_tensor_free(tokens);
    if (ok) ok = h3_gpu_tensor_read_bf16(pred, s->pred_bf16, (size_t)N * SV2_PATCH_OUT);
    pool_give(m, pred);
    if (!ok) { if (error && !error[0]) fail(error, error_size, "SeedVR2: cannot read the prediction"); return 0; }
    unpack_pred(s->pred_bf16, lw, lh, s->pred_host);
    trace(&mark, "dit");
    /* one Euler step from t = 1: z = noise - v; the decoder divides by the scaling */
    float *z = malloc((size_t)lw * lh * SV2_LATENT * sizeof(float));
    if (!z) { fail(error, error_size, "SeedVR2: out of memory"); return 0; }
    for (int c = 0; c < SV2_LATENT; c++)
        for (int y = 0; y < lh; y++)
            for (int x = 0; x < lw; x++) {
                size_t i = (size_t)c * lw * lh + (size_t)y * lw + x;
                s->latent_host[i] = noise[i] - s->pred_host[i];
                z[((size_t)y * lw + x) * SV2_LATENT + c] = s->latent_host[i] / SV2_SCALING;
            }
    /* decode */
    if (on_progress && on_progress("vsr decode", 0, 1, opaque)) { free(z); fail(error, error_size, "vsr cancelled"); return 0; }
    act zt = {h3_gpu_tensor_from_f32(m->gpu, z, (size_t)lw * lh * SV2_LATENT), lh, lw, SV2_LATENT}, dec;
    free(z);
    if (!zt.t) { fail(error, error_size, "SeedVR2: out of GPU memory"); return 0; }
    if (!h3_gpu_begin(m->gpu)) { h3_gpu_tensor_free(zt.t); fail(error, error_size, "SeedVR2: %s", h3_gpu_error(m->gpu)); return 0; }
    ok = vae_decode(m, &zt, &dec, error, error_size);
    if (!ok) { h3_gpu_submit(m->gpu); h3_gpu_tensor_free(zt.t); return 0; }
    ok = h3_gpu_copy_f32(m->gpu, rgb_out, 0, dec.t, 0, (size_t)s->full_w * s->full_h * 3) && h3_gpu_submit(m->gpu);
    h3_gpu_tensor_free(zt.t);
    act_free(m, &dec);
    if (!ok) { fail(error, error_size, "SeedVR2 decode: %s", h3_gpu_error(m->gpu)); return 0; }
    trace(&mark, "decode");
    if (on_progress && on_progress("vsr decode", 1, 1, opaque)) { fail(error, error_size, "vsr cancelled"); return 0; }
    return 1;
}

int h3_seedvr2_session_process(h3_seedvr2_session *s, const h3_vsr_image *in, h3_vsr_image_out *out,
                               h3_progress_callback on_progress, void *opaque,
                               const volatile int *cancel_requested, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!s || !in || !out || !in->pixels || !out->pixels) { fail(error, error_size, "SeedVR2: bad arguments"); return 0; }
    if (in->width != s->in_w || in->height != s->in_h || out->width != s->out_w || out->height != s->out_h) {
        fail(error, error_size, "SeedVR2: the frame is %dx%d -> %dx%d, the session %dx%d -> %dx%d",
             in->width, in->height, out->width, out->height, s->in_w, s->in_h, s->out_w, s->out_h);
        return 0;
    }
    h3_seedvr2 *m = s->m;
    int bgra_in = in->format == H3_VSR_BGRA8, bgra_out = out->format == H3_VSR_BGRA8;
    size_t in_bytes = (size_t)in->stride_bytes * in->height;
    size_t out_bytes = (size_t)out->stride_bytes * out->height;
    /* the input bytes go up as a fresh I8 tensor (there is no I8 write) */
    h3_gpu_tensor_free(s->pixels_in);
    s->pixels_in = h3_gpu_tensor_from_i8(m->gpu, (const int8_t *)in->pixels, in_bytes);
    s->pixels_in_bytes = in_bytes;
    if (!s->pixels_out || s->pixels_out_bytes != out_bytes) {
        h3_gpu_tensor_free(s->pixels_out);
        s->pixels_out = h3_gpu_tensor_new_i8(m->gpu, out_bytes);
        s->pixels_out_bytes = out_bytes;
    }
    size_t n = (size_t)s->full_w * s->full_h * 3;
    h3_gpu_tensor *rgb_in = pool_take(m, n, 0), *rgb_out = pool_take(m, n, 0);
    int ok = 0;
    if (!s->pixels_in || !s->pixels_out || !rgb_in || !rgb_out) { fail(error, error_size, "SeedVR2: out of GPU memory"); goto done; }
    double mark = now_seconds();
    if (!h3_gpu_begin(m->gpu) ||
        !h3_gpu_sv2_bicubic_pixels_f32(m->gpu, rgb_in, s->pixels_in, (uint32_t)in->width, (uint32_t)in->height,
                                       (uint32_t)in->stride_bytes, bgra_in, (uint32_t)s->out_w, (uint32_t)s->out_h,
                                       (uint32_t)s->full_w, (uint32_t)s->full_h, -1.0f) ||
        !h3_gpu_submit(m->gpu)) {
        fail(error, error_size, "SeedVR2 resize: %s", h3_gpu_error(m->gpu));
        goto done;
    }
    trace(&mark, "resize");
    if (!restore_frame(s, rgb_in, rgb_out, NULL, NULL, -1, on_progress, opaque, cancel_requested, error, error_size))
        goto done;
    if (m->params.color_transfer) {
        /* the reference matches the cropped frames; the padding is black in
         * both, so matching the padded frames differs only by those pixels'
         * share of the histograms */
        if (!color_transfer(m, rgb_out, rgb_out, rgb_in, s->full_w, s->full_h, error, error_size)) goto done;
    }
    if (!h3_gpu_begin(m->gpu) ||
        !h3_gpu_sv2_f32_to_pixels(m->gpu, s->pixels_out, rgb_out, (uint32_t)s->full_w, (uint32_t)s->full_h,
                                  (uint32_t)s->out_w, (uint32_t)s->out_h, (uint32_t)out->stride_bytes, bgra_out) ||
        !h3_gpu_submit(m->gpu)) {
        fail(error, error_size, "SeedVR2 output: %s", h3_gpu_error(m->gpu));
        goto done;
    }
    if (!h3_gpu_tensor_read_i8(s->pixels_out, (int8_t *)out->pixels, out_bytes)) {
        fail(error, error_size, "SeedVR2: cannot read the output frame");
        goto done;
    }
    trace(&mark, "output");
    ok = 1;
done:
    pool_give(m, rgb_in); pool_give(m, rgb_out);
    return ok;
}

/* ---------------------------------------------------------------- */
/* Stage entry points for the parity test */

/* CHW [-1, 1] host -> HWC GPU tensor and back. */
static h3_gpu_tensor *upload_chw(h3_gpu *gpu, const float *chw, int channels, int width, int height) {
    size_t n = (size_t)width * height;
    float *hwc = malloc(n * channels * sizeof(float));
    if (!hwc) return NULL;
    for (int c = 0; c < channels; c++)
        for (size_t i = 0; i < n; i++) hwc[i * channels + c] = chw[(size_t)c * n + i];
    h3_gpu_tensor *t = h3_gpu_tensor_from_f32(gpu, hwc, n * channels);
    free(hwc);
    return t;
}

static int download_chw(const h3_gpu_tensor *t, float *chw, int channels, int width, int height) {
    size_t n = (size_t)width * height;
    float *hwc = malloc(n * channels * sizeof(float));
    if (!hwc || !h3_gpu_tensor_read_f32(t, hwc, n * channels)) { free(hwc); return 0; }
    for (int c = 0; c < channels; c++)
        for (size_t i = 0; i < n; i++) chw[(size_t)c * n + i] = hwc[i * channels + c];
    free(hwc);
    return 1;
}

int h3_seedvr2_debug_encode(h3_seedvr2 *m, const float *rgb, int width, int height, float *latent,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (width % 16 || height % 16) { fail(error, error_size, "SeedVR2: sides must be multiples of 16"); return 0; }
    act in = {upload_chw(m->gpu, rgb, 3, width, height), height, width, 3}, enc;
    if (!in.t) { fail(error, error_size, "SeedVR2: out of GPU memory"); return 0; }
    int ok = h3_gpu_begin(m->gpu) && vae_encode(m, &in, &enc, error, error_size);
    if (!ok) { h3_gpu_submit(m->gpu); h3_gpu_tensor_free(in.t); return 0; }
    ok = h3_gpu_submit(m->gpu);
    h3_gpu_tensor_free(in.t);
    int lw = width / 8, lh = height / 8;
    float *full = malloc((size_t)lw * lh * 2 * SV2_LATENT * sizeof(float));
    if (ok && full) ok = download_chw(enc.t, full, 2 * SV2_LATENT, lw, lh);
    act_free(m, &enc);
    if (!ok || !full) { free(full); fail(error, error_size, "SeedVR2 encode: %s", h3_gpu_error(m->gpu)); return 0; }
    for (size_t i = 0; i < (size_t)SV2_LATENT * lw * lh; i++) latent[i] = full[i] * SV2_SCALING;
    free(full);
    return 1;
}

int h3_seedvr2_debug_dit(h3_seedvr2 *m, const float *dit_in, int lw, int lh, float *pred, int dump_block,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (lw % 2 || lh % 2) { fail(error, error_size, "SeedVR2: the latent grid must be even"); return 0; }
    h3_seedvr2_session *s = h3_seedvr2_session_open(m, lw * 8, lh * 8, lw * 8, lh * 8, error, error_size);
    if (!s) return 0;
    size_t plane = (size_t)lw * lh;
    int N = (lw / 2) * (lh / 2);
    pack_tokens(dit_in, dit_in + 16 * plane, lw, lh, s->tokens_host);
    h3_gpu_tensor *tokens = h3_gpu_tensor_from_bf16(m->gpu, s->tokens_host, (size_t)N * SV2_PATCH_IN);
    h3_gpu_tensor *out = pool_take(m, (size_t)N * SV2_PATCH_OUT, 1);
    int ok = tokens && out &&
             dit_forward(m, s->layouts, tokens, out, dump_block, NULL, NULL, NULL, error, error_size) &&
             h3_gpu_tensor_read_bf16(out, s->pred_bf16, (size_t)N * SV2_PATCH_OUT);
    if (ok) unpack_pred(s->pred_bf16, lw, lh, pred);
    h3_gpu_tensor_free(tokens);
    pool_give(m, out);
    h3_seedvr2_session_close(s);
    return ok;
}

int h3_seedvr2_debug_decode(h3_seedvr2 *m, const float *latent, int lw, int lh, float *rgb,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    size_t plane = (size_t)lw * lh;
    float *scaled = malloc(plane * SV2_LATENT * sizeof(float));
    if (!scaled) return 0;
    for (size_t i = 0; i < plane * SV2_LATENT; i++) scaled[i] = latent[i] / SV2_SCALING;
    act z = {upload_chw(m->gpu, scaled, SV2_LATENT, lw, lh), lh, lw, SV2_LATENT}, dec;
    free(scaled);
    if (!z.t) { fail(error, error_size, "SeedVR2: out of GPU memory"); return 0; }
    int ok = h3_gpu_begin(m->gpu) && vae_decode(m, &z, &dec, error, error_size);
    if (!ok) { h3_gpu_submit(m->gpu); h3_gpu_tensor_free(z.t); return 0; }
    ok = h3_gpu_submit(m->gpu);
    h3_gpu_tensor_free(z.t);
    if (ok) ok = download_chw(dec.t, rgb, 3, lw * 8, lh * 8);
    act_free(m, &dec);
    if (!ok) fail(error, error_size, "SeedVR2 decode: %s", h3_gpu_error(m->gpu));
    return ok;
}

int h3_seedvr2_debug_color(h3_seedvr2 *m, const float *content, const float *style, int width, int height,
                           float *out, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    h3_gpu_tensor *c = upload_chw(m->gpu, content, 3, width, height);
    h3_gpu_tensor *st = upload_chw(m->gpu, style, 3, width, height);
    h3_gpu_tensor *o = h3_gpu_tensor_new_f32(m->gpu, (size_t)width * height * 3);
    int ok = c && st && o && color_transfer(m, o, c, st, width, height, error, error_size) &&
             download_chw(o, out, 3, width, height);
    h3_gpu_tensor_free(c); h3_gpu_tensor_free(st); h3_gpu_tensor_free(o);
    return ok;
}

int h3_seedvr2_debug_windows(int tokens_w, int tokens_h, int shifted, int *windows, int capacity) {
    sv2_window w[1024];
    int n = make_windows(tokens_h, tokens_w, shifted, w, 1024);
    for (int i = 0; i < n && i < capacity; i++) {
        windows[i * 4] = w[i].hs; windows[i * 4 + 1] = w[i].he;
        windows[i * 4 + 2] = w[i].ws; windows[i * 4 + 3] = w[i].we;
    }
    return n;
}

int h3_seedvr2_debug_encode_stage(h3_seedvr2 *m, const float *rgb, int width, int height, int stage,
                                  float *out, int *out_w, int *out_h, int *out_c, size_t capacity,
                                  char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    act in = {upload_chw(m->gpu, rgb, 3, width, height), height, width, 3}, y;
    if (!in.t) { fail(error, error_size, "SeedVR2: out of GPU memory"); return 0; }
    int ok = h3_gpu_begin(m->gpu) && vae_encode_stage(m, &in, &y, stage, error, error_size);
    if (!ok) { h3_gpu_submit(m->gpu); h3_gpu_tensor_free(in.t); return 0; }
    ok = h3_gpu_submit(m->gpu);   /* the input must outlive the command buffer */
    h3_gpu_tensor_free(in.t);
    *out_w = y.w; *out_h = y.h; *out_c = y.c;
    if (ok && (size_t)y.w * y.h * y.c <= capacity) ok = download_chw(y.t, out, y.c, y.w, y.h);
    else if (ok) { fail(error, error_size, "SeedVR2: stage output needs %d values", y.w * y.h * y.c); ok = 0; }
    act_free(m, &y);
    return ok;
}
