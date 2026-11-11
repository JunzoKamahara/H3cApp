/* See h3_lora_extract.h. Streams one tensor at a time (the largest of the
 * Open-H3 merge is 28672 x 576 F32, 66 MB), so a 12.5 GB input needs no
 * memory to speak of. */
#include "h3_lora_extract.h"
#include "h3_safetensors.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exponent = (h >> 10) & 0x1fu, mantissa = h & 0x3ffu, bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else { exponent = 127 - 15 + 1; while (!(mantissa & 0x400u)) { mantissa <<= 1; exponent--; } mantissa &= 0x3ffu; bits = sign | (exponent << 23) | (mantissa << 13); }
    } else if (exponent == 31) bits = sign | 0x7f800000u | (mantissa << 13);
    else bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float load_f32(const void *data, h3_dtype dtype, size_t index) {
    switch (dtype) {
    case H3_DTYPE_F32: return ((const float *)data)[index];
    case H3_DTYPE_F16: return f16_to_f32(((const uint16_t *)data)[index]);
    case H3_DTYPE_BF16: { uint32_t bits = (uint32_t)((const uint16_t *)data)[index] << 16; float f; memcpy(&f, &bits, 4); return f; }
    default: return 0.0f;
    }
}

/* A JSON string: the names only hold [A-Za-z0-9._] in practice, but escape
 * anyway. */
static void json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') { fputc('\\', f); fputc(*s, f); }
        else if ((unsigned char)*s < 0x20) fprintf(f, "\\u%04x", (unsigned char)*s);
        else fputc(*s, f);
    }
    fputc('"', f);
}

typedef struct { const h3_st_tensor *t; int is_a, is_b; uint64_t rows, cols; h3_dtype out_dtype; uint64_t bytes; } plan;

int h3_lora_extract_ranks(const char *in_path, const char *out_path,
                          int rank_begin, int rank_end, int bf16, const char *note,
                          h3_progress_callback on_progress, void *opaque,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!in_path || !out_path || rank_begin < 0 || rank_end <= rank_begin) {
        fail(error, error_size, "lora extract: bad arguments");
        return 0;
    }
    h3_st_header h;
    if (!h3_st_read_header(in_path, &h, error, error_size)) return 0;
    size_t n = h.tensor_count;
    plan *plans = calloc(n, sizeof(*plans));
    if (!plans) { h3_st_free_header(&h); fail(error, error_size, "lora extract: out of memory"); return 0; }
    int keep = rank_end - rank_begin;
    size_t max_bytes = 0;
    for (size_t i = 0; i < n; i++) {
        const h3_st_tensor *t = &h.tensors[i];
        plan *p = &plans[i];
        p->t = t;
        if (t->dtype != H3_DTYPE_F32 && t->dtype != H3_DTYPE_F16 && t->dtype != H3_DTYPE_BF16) {
            fail(error, error_size, "lora extract: %s is %s, not a float tensor", t->name, h3_dtype_name(t->dtype));
            goto failed;
        }
        size_t len = strlen(t->name);
        static const char suffix_a[] = ".lora_A.weight", suffix_b[] = ".lora_B.weight";
        const size_t suffix_len = sizeof(suffix_a) - 1;
        p->is_a = len > suffix_len && !strcmp(t->name + len - suffix_len, suffix_a);
        p->is_b = len > suffix_len && !strcmp(t->name + len - suffix_len, suffix_b);
        if ((p->is_a || p->is_b) && t->ndim != 2) {
            fail(error, error_size, "lora extract: %s is not 2-D", t->name);
            goto failed;
        }
        p->rows = t->ndim >= 1 ? t->shape[0] : 1;
        p->cols = 1;
        for (int d = 1; d < t->ndim; d++) p->cols *= t->shape[d];
        if (p->is_a && (uint64_t)rank_end > p->rows) { fail(error, error_size, "lora extract: %s has rank %llu < %d", t->name, (unsigned long long)p->rows, rank_end); goto failed; }
        if (p->is_b && (uint64_t)rank_end > p->cols) { fail(error, error_size, "lora extract: %s has rank %llu < %d", t->name, (unsigned long long)p->cols, rank_end); goto failed; }
        uint64_t out_rows = p->is_a ? (uint64_t)keep : p->rows;
        uint64_t out_cols = p->is_b ? (uint64_t)keep : p->cols;
        p->out_dtype = bf16 ? H3_DTYPE_BF16 : H3_DTYPE_F32;
        p->bytes = out_rows * out_cols * h3_dtype_size(p->out_dtype);
        size_t in_bytes = (size_t)h3_st_tensor_elements(t) * h3_dtype_size(t->dtype);
        if (in_bytes > max_bytes) max_bytes = in_bytes;
    }
    /* header */
    char tmp_path[2048];
    snprintf(tmp_path, sizeof(tmp_path), "%s.partial", out_path);
    FILE *out = fopen(tmp_path, "wb");
    if (!out) { fail(error, error_size, "lora extract: cannot write %s", tmp_path); goto failed; }
    /* the JSON in memory first, to know its length */
    char *json = NULL;
    size_t json_len = 0;
    FILE *js = open_memstream(&json, &json_len);
    if (!js) { fclose(out); fail(error, error_size, "lora extract: out of memory"); goto failed; }
    fputs("{\"__metadata__\":{\"format\":\"lora\",\"h3c_extract\":", js);
    char meta[512];
    snprintf(meta, sizeof(meta), "ranks %d:%d of %s", rank_begin, rank_end, in_path);
    json_string(js, meta);
    if (note && *note) { fputs(",\"note\":", js); json_string(js, note); }
    fputs("}", js);
    uint64_t at = 0;
    for (size_t i = 0; i < n; i++) {
        plan *p = &plans[i];
        uint64_t out_rows = p->is_a ? (uint64_t)keep : p->rows;
        uint64_t out_cols = p->is_b ? (uint64_t)keep : p->cols;
        fputs(",", js);
        json_string(js, p->t->name);
        fprintf(js, ":{\"dtype\":\"%s\",\"shape\":[", p->out_dtype == H3_DTYPE_BF16 ? "BF16" : "F32");
        if (p->t->ndim == 2) fprintf(js, "%llu,%llu", (unsigned long long)out_rows, (unsigned long long)out_cols);
        else for (int d = 0; d < p->t->ndim; d++) fprintf(js, "%s%llu", d ? "," : "", (unsigned long long)p->t->shape[d]);
        fprintf(js, "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)at, (unsigned long long)(at + p->bytes));
        at += p->bytes;
    }
    fputs("}", js);
    fclose(js);
    /* pad the header to a multiple of 8 with spaces, as the format allows */
    size_t padded = (json_len + 7) / 8 * 8;
    uint64_t header_size = padded;
    int ok = fwrite(&header_size, 8, 1, out) == 1 && fwrite(json, 1, json_len, out) == json_len;
    for (size_t i = json_len; ok && i < padded; i++) ok = fputc(' ', out) != EOF;
    free(json);
    if (!ok) { fclose(out); unlink(tmp_path); fail(error, error_size, "lora extract: write failed"); goto failed; }
    /* tensors */
    void *in_buffer = malloc(max_bytes ? max_bytes : 1);
    void *out_buffer = malloc((size_t)max_bytes ? (size_t)max_bytes : 1);
    if (!in_buffer || !out_buffer) { free(in_buffer); free(out_buffer); fclose(out); unlink(tmp_path); fail(error, error_size, "lora extract: out of memory"); goto failed; }
    for (size_t i = 0; i < n && ok; i++) {
        plan *p = &plans[i];
        if (on_progress && on_progress("lora extract", (int)i, (int)n, opaque)) {
            fail(error, error_size, "lora extract cancelled");
            ok = 0;
            break;
        }
        size_t in_bytes = (size_t)h3_st_tensor_elements(p->t) * h3_dtype_size(p->t->dtype);
        if (!h3_st_read_data(&h, p->t, in_buffer, in_bytes, error, error_size)) { ok = 0; break; }
        uint64_t out_rows = p->is_a ? (uint64_t)keep : p->rows;
        uint64_t out_cols = p->is_b ? (uint64_t)keep : p->cols;
        uint64_t row0 = p->is_a ? (uint64_t)rank_begin : 0, col0 = p->is_b ? (uint64_t)rank_begin : 0;
        for (uint64_t r = 0; r < out_rows; r++) {
            for (uint64_t c = 0; c < out_cols; c++) {
                float v = load_f32(in_buffer, p->t->dtype, (size_t)((row0 + r) * p->cols + col0 + c));
                size_t o = (size_t)(r * out_cols + c);
                if (p->out_dtype == H3_DTYPE_BF16) ((uint16_t *)out_buffer)[o] = f32_to_bf16(v);
                else ((float *)out_buffer)[o] = v;
            }
        }
        if (fwrite(out_buffer, 1, (size_t)p->bytes, out) != (size_t)p->bytes) {
            fail(error, error_size, "lora extract: write failed (disk full?)");
            ok = 0;
        }
    }
    free(in_buffer); free(out_buffer);
    if (fclose(out) != 0) { fail(error, error_size, "lora extract: close failed (disk full?)"); ok = 0; }
    if (ok && on_progress) on_progress("lora extract", (int)n, (int)n, opaque);
    if (ok && rename(tmp_path, out_path) != 0) { fail(error, error_size, "lora extract: cannot rename to %s", out_path); ok = 0; }
    if (!ok) unlink(tmp_path);
    free(plans);
    h3_st_free_header(&h);
    return ok;
failed:
    free(plans);
    h3_st_free_header(&h);
    return 0;
}
