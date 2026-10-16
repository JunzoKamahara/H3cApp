/* Curve-form AdaLN files: reading them for h3_dit_schedule_precompute_curve
 * and building them from a transformer's dense AdaLN projections. See
 * h3_adaln_curve.h for the idea and the file layout.
 *
 * Building: the curve is sampled at grid evenly spaced t in [0, 1] with the
 * engine's own time embedding (h3_dit_time_silu, so the samples are exactly
 * the BF16 values the dense projections consume), its rank-k basis V is the
 * top eigenvectors of S^T S (S = samples, [grid, 2688]), the table is S*V,
 * and each projection W becomes W*V. Measured on the released FL2VA
 * checkpoint (quant-import/curve_test_all.py, k=32, grid 4097): worst
 * relative L2 error 1.2e-4 over all 51 projections, against 1.3e-3..1.7e-3
 * for rounding those outputs to BF16 at all. The error comes from the
 * linear interpolation (it shrinks with the grid, barely with k). */
#define ACCELERATE_NEW_LAPACK
#include "h3_adaln_curve.h"

#include "h3_dit_schedule.h"

#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CURVE_FORMAT "h3-adaln-curve"
#define CURVE_VERSION "1"
#define CURVE_TABLE "adaln_t_table"
#define BLOCK_OUTPUT (H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN)
#define FINAL_OUTPUT (2u * H3_DIT_HIDDEN)
#define PROJECTIONS (H3_DIT_BLOCKS + 1u)

struct h3_adaln_curve {
    h3_st_header header;
    uint32_t grid;
    uint32_t rank;
    float *table;
    uint8_t fingerprint[32];
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint64_t fnv1a64(uint64_t hash, const void *data, size_t size) {
    const uint8_t *bytes = data;
    for (size_t index = 0; index < size; index++) {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

static void hex_encode(const uint8_t bytes[32], char out[65]) {
    static const char digits[] = "0123456789abcdef";
    for (int index = 0; index < 32; index++) {
        out[index * 2] = digits[bytes[index] >> 4];
        out[index * 2 + 1] = digits[bytes[index] & 15];
    }
    out[64] = '\0';
}

static int parse_u32(const char *text, uint32_t *value) {
    if (!text || !*text) return 0;
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno || !end || *end || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static const char *base_name(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

h3_adaln_curve *h3_adaln_curve_open(const char *path, const uint8_t model_id[32],
                                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path) {
        fail(error, error_size, "AdaLN curve path is required");
        return NULL;
    }
    h3_adaln_curve *curve = calloc(1, sizeof(*curve));
    if (!curve) {
        fail(error, error_size, "out of memory opening the AdaLN curve");
        return NULL;
    }
    char detail[384];
    if (!h3_st_read_header(path, &curve->header, detail, sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        free(curve);
        return NULL;
    }
    const char *format = h3_st_metadata(&curve->header, "format");
    const char *version = h3_st_metadata(&curve->header, "version");
    if (!format || strcmp(format, CURVE_FORMAT) ||
        !version || strcmp(version, CURVE_VERSION) ||
        !parse_u32(h3_st_metadata(&curve->header, "grid"), &curve->grid) ||
        !parse_u32(h3_st_metadata(&curve->header, "rank"), &curve->rank) ||
        curve->grid < 2 || !curve->rank || curve->rank > H3_DIT_TIME_DIM) {
        fail(error, error_size, "%s is not an H3 AdaLN curve file (version %s)",
             path, CURVE_VERSION);
        goto failed;
    }
    if (model_id) {
        const char *stored = h3_st_metadata(&curve->header, "model_id");
        char expected[65];
        hex_encode(model_id, expected);
        if (!stored || strcmp(stored, expected)) {
            fail(error, error_size,
                 "%s was built from a different transformer checkpoint - "
                 "rebuild it with build_adaln_curve", path);
            goto failed;
        }
    }
    const h3_st_tensor *table = h3_st_find(&curve->header, CURVE_TABLE);
    if (!table || table->dtype != H3_DTYPE_F32 || table->ndim != 2 ||
        table->shape[0] != curve->grid || table->shape[1] != curve->rank) {
        fail(error, error_size, "%s has no valid %s", path, CURVE_TABLE);
        goto failed;
    }
    size_t table_count = (size_t)curve->grid * curve->rank;
    curve->table = malloc(table_count * sizeof(*curve->table));
    if (!curve->table ||
        !h3_st_read_data(&curve->header, table, curve->table,
                         table_count * sizeof(*curve->table), detail,
                         sizeof(detail))) {
        fail(error, error_size, "cannot read %s: %s", CURVE_TABLE,
             curve->table ? detail : "out of memory");
        goto failed;
    }
    for (size_t index = 0; index < table_count; index++) {
        if (!isfinite(curve->table[index])) {
            fail(error, error_size, "%s has a non-finite %s value", path,
                 CURVE_TABLE);
            goto failed;
        }
    }

    struct stat status;
    int64_t mtime = stat(path, &status) == 0 ? (int64_t)status.st_mtime : 0;
    uint64_t size = curve->header.file_size;
    const char *name = base_name(path);
    uint64_t hash = 1469598103934665603ull;
    hash = fnv1a64(hash, name, strlen(name) + 1);
    hash = fnv1a64(hash, &size, sizeof(size));
    hash = fnv1a64(hash, &mtime, sizeof(mtime));
    memcpy(curve->fingerprint, &hash, sizeof(hash));
    memcpy(curve->fingerprint + 8, "h3curve1", 8);
    return curve;

failed:
    h3_adaln_curve_free(curve);
    return NULL;
}

void h3_adaln_curve_free(h3_adaln_curve *curve) {
    if (!curve) return;
    h3_st_free_header(&curve->header);
    free(curve->table);
    free(curve);
}

uint32_t h3_adaln_curve_rank(const h3_adaln_curve *curve) {
    return curve ? curve->rank : 0;
}

uint32_t h3_adaln_curve_grid(const h3_adaln_curve *curve) {
    return curve ? curve->grid : 0;
}

void h3_adaln_curve_fingerprint(const h3_adaln_curve *curve, uint8_t out[32]) {
    memset(out, 0, 32);
    if (curve) memcpy(out, curve->fingerprint, 32);
}

void h3_adaln_curve_lerp(const float *table, uint32_t grid, uint32_t rank,
                         float t, float *out) {
    if (!(t > 0.0f)) t = 0.0f; /* also maps NaN to the curve's start */
    if (t > 1.0f) t = 1.0f;
    float position = t * (float)(grid - 1);
    uint32_t lower = (uint32_t)floorf(position);
    if (lower > grid - 2) lower = grid - 2;
    float fraction = position - (float)lower;
    const float *a = table + (size_t)lower * rank;
    const float *b = a + rank;
    for (uint32_t index = 0; index < rank; index++)
        out[index] = a[index] + fraction * (b[index] - a[index]);
}

void h3_adaln_curve_coordinates(const h3_adaln_curve *curve,
                                const float *times, uint32_t count,
                                float *out) {
    for (uint32_t row = 0; row < count; row++)
        h3_adaln_curve_lerp(curve->table, curve->grid, curve->rank,
                            times[row], out + (size_t)row * curve->rank);
}

static h3_gpu_tensor *load_f32(const h3_adaln_curve *curve, h3_gpu *gpu,
                               const char *name, uint32_t rows,
                               uint32_t columns, char *error,
                               size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(&curve->header, name);
    int ndim = columns ? 2 : 1;
    if (!tensor || tensor->dtype != H3_DTYPE_F32 || tensor->ndim != ndim ||
        tensor->shape[0] != rows || (columns && tensor->shape[1] != columns)) {
        fail(error, error_size, "AdaLN curve tensor %s is missing or has the "
             "wrong shape", name);
        return NULL;
    }
    size_t elements = (size_t)rows * (columns ? columns : 1);
    h3_gpu_tensor *result = h3_gpu_tensor_load_f32(
        gpu, curve->header.path, tensor->file_offset, elements);
    if (!result)
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    return result;
}

h3_gpu_tensor *h3_adaln_curve_load_weight(const h3_adaln_curve *curve,
                                          h3_gpu *gpu, const char *prefix,
                                          uint32_t rows, char *error,
                                          size_t error_size) {
    char name[128];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    return load_f32(curve, gpu, name, rows, curve->rank, error, error_size);
}

h3_gpu_tensor *h3_adaln_curve_load_bias(const h3_adaln_curve *curve,
                                        h3_gpu *gpu, const char *prefix,
                                        uint32_t rows, char *error,
                                        size_t error_size) {
    char name[128];
    snprintf(name, sizeof(name), "%s.bias", prefix);
    return load_f32(curve, gpu, name, rows, 0, error, error_size);
}

/* --- building --------------------------------------------------------- */

typedef struct {
    char name[96];
    uint32_t rows;     /* outputs */
    uint32_t columns;  /* 0 for a bias */
} curve_entry;

static void projection_prefix(unsigned projection, char *out, size_t size) {
    if (projection < H3_DIT_BLOCKS)
        snprintf(out, size, "blocks.%u.adaln_proj.linear", projection);
    else
        snprintf(out, size, "final_layer.adaln_proj.linear");
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t widened = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &widened, sizeof(value));
    return value;
}

/* Top-rank right singular vectors of samples [grid, width] as a row-major
 * [width, rank] basis, largest first, via the eigenvectors of S^T S. */
static double *curve_basis(const double *samples, uint32_t grid,
                           uint32_t width, uint32_t rank, char *error,
                           size_t error_size) {
    double *gram = calloc((size_t)width * width, sizeof(*gram));
    double *values = malloc((size_t)width * sizeof(*values));
    double *vectors = malloc((size_t)width * rank * sizeof(*vectors));
    __LAPACK_int *support = malloc(2 * (size_t)rank * sizeof(*support));
    double *basis = malloc((size_t)width * rank * sizeof(*basis));
    double *work = NULL;
    __LAPACK_int *iwork = NULL;
    if (!gram || !values || !vectors || !support || !basis) {
        fail(error, error_size, "out of memory for the AdaLN curve basis");
        goto failed;
    }
    /* Row-major upper triangle of S^T S = column-major lower for LAPACK. */
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, (__LAPACK_int)width,
                (__LAPACK_int)grid, 1.0, samples, (__LAPACK_int)width, 0.0,
                gram, (__LAPACK_int)width);
    char jobz = 'V', range = 'I', uplo = 'L';
    __LAPACK_int n = (__LAPACK_int)width, lda = n, ldz = n;
    __LAPACK_int il = n - (__LAPACK_int)rank + 1, iu = n, found = 0, info = 0;
    double vl = 0.0, vu = 0.0, abstol = 0.0, work_size = 0.0;
    __LAPACK_int lwork = -1, liwork = -1, iwork_size = 0;
    dsyevr_(&jobz, &range, &uplo, &n, gram, &lda, &vl, &vu, &il, &iu, &abstol,
            &found, values, vectors, &ldz, support, &work_size, &lwork,
            &iwork_size, &liwork, &info);
    if (info) {
        fail(error, error_size, "AdaLN curve basis workspace query failed (%d)",
             (int)info);
        goto failed;
    }
    lwork = (__LAPACK_int)work_size;
    liwork = iwork_size;
    work = malloc((size_t)lwork * sizeof(*work));
    iwork = malloc((size_t)liwork * sizeof(*iwork));
    if (!work || !iwork) {
        fail(error, error_size, "out of memory for the AdaLN curve basis");
        goto failed;
    }
    dsyevr_(&jobz, &range, &uplo, &n, gram, &lda, &vl, &vu, &il, &iu, &abstol,
            &found, values, vectors, &ldz, support, work, &lwork, iwork,
            &liwork, &info);
    if (info || found != (__LAPACK_int)rank) {
        fail(error, error_size, "AdaLN curve basis failed (info %d, %d of %u "
             "vectors)", (int)info, (int)found, rank);
        goto failed;
    }
    /* dsyevr orders eigenvalues ascending; put the largest first. */
    for (uint32_t row = 0; row < width; row++)
        for (uint32_t column = 0; column < rank; column++)
            basis[(size_t)row * rank + column] =
                vectors[row + (size_t)width * (rank - 1 - column)];
    free(gram);
    free(values);
    free(vectors);
    free(support);
    free(work);
    free(iwork);
    return basis;
failed:
    free(gram);
    free(values);
    free(vectors);
    free(support);
    free(work);
    free(iwork);
    free(basis);
    return NULL;
}

static int write_all(FILE *out, const void *data, size_t bytes) {
    return fwrite(data, 1, bytes, out) == bytes;
}

/* W*V for one projection, plus its bias, appended to out as F32. */
static int write_projection(const h3_weight_store *store, unsigned projection,
                            const float *basis, uint32_t rank, FILE *out,
                            char *error, size_t error_size) {
    char prefix[96], name[128], detail[384];
    projection_prefix(projection, prefix, sizeof(prefix));
    uint32_t outputs = projection < H3_DIT_BLOCKS ? BLOCK_OUTPUT : FINAL_OUTPUT;
    const uint32_t width = H3_DIT_TIME_DIM;

    snprintf(name, sizeof(name), "%s.weight", prefix);
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != outputs || tensor->shape[1] != width) {
        fail(error, error_size, "dense AdaLN weight %s is missing or has the "
             "wrong shape", name);
        return 0;
    }
    size_t elements = (size_t)outputs * width;
    uint16_t *dense = malloc(elements * sizeof(*dense));
    enum { CHUNK = 4096 };
    float *rows = malloc((size_t)CHUNK * width * sizeof(*rows));
    float *projected = malloc((size_t)CHUNK * rank * sizeof(*projected));
    int ok = dense && rows && projected;
    if (!ok) fail(error, error_size, "out of memory projecting %s", name);
    if (ok && !h3_st_read_data(header, tensor, dense, elements * sizeof(*dense),
                               detail, sizeof(detail))) {
        fail(error, error_size, "cannot read %s: %s", name, detail);
        ok = 0;
    }
    for (uint32_t first = 0; ok && first < outputs; first += CHUNK) {
        uint32_t count = outputs - first < CHUNK ? outputs - first : CHUNK;
        const uint16_t *source = dense + (size_t)first * width;
        for (size_t index = 0; index < (size_t)count * width; index++)
            rows[index] = bf16_to_f32(source[index]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (__LAPACK_int)count, (__LAPACK_int)rank,
                    (__LAPACK_int)width, 1.0f, rows, (__LAPACK_int)width,
                    basis, (__LAPACK_int)rank, 0.0f, projected,
                    (__LAPACK_int)rank);
        if (!write_all(out, projected, (size_t)count * rank * sizeof(float))) {
            fail(error, error_size, "cannot write %s: %s", name,
                 strerror(errno));
            ok = 0;
        }
    }
    free(rows);
    free(projected);

    if (ok) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        tensor = h3_weight_find(store, name, &header);
        if (!tensor || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 1 ||
            tensor->shape[0] != outputs) {
            fail(error, error_size, "dense AdaLN bias %s is missing or has "
                 "the wrong shape", name);
            ok = 0;
        }
    }
    if (ok && !h3_st_read_data(header, tensor, dense,
                               (size_t)outputs * sizeof(*dense), detail,
                               sizeof(detail))) {
        fail(error, error_size, "cannot read %s: %s", name, detail);
        ok = 0;
    }
    if (ok) {
        float *bias = malloc((size_t)outputs * sizeof(*bias));
        ok = bias != NULL;
        for (uint32_t index = 0; ok && index < outputs; index++)
            bias[index] = bf16_to_f32(dense[index]);
        if (ok && !write_all(out, bias, (size_t)outputs * sizeof(*bias))) {
            fail(error, error_size, "cannot write %s: %s", name,
                 strerror(errno));
            ok = 0;
        }
        if (!bias) fail(error, error_size, "out of memory for %s", name);
        free(bias);
    }
    free(dense);
    return ok;
}

/* The safetensors header for the whole file, padded to 8 bytes. */
static char *curve_header_json(uint32_t grid, uint32_t rank,
                               const uint8_t model_id[32], size_t *length) {
    size_t capacity = 32768, used = 0;
    char *json = malloc(capacity);
    if (!json) return NULL;
    char hex[65];
    hex_encode(model_id, hex);
    used += (size_t)snprintf(json + used, capacity - used,
        "{\"__metadata__\":{\"format\":\"%s\",\"version\":\"%s\","
        "\"grid\":\"%u\",\"rank\":\"%u\",\"model_id\":\"%s\"},"
        "\"%s\":{\"dtype\":\"F32\",\"shape\":[%u,%u],\"data_offsets\":[0,%llu]}",
        CURVE_FORMAT, CURVE_VERSION, grid, rank, hex, CURVE_TABLE, grid, rank,
        (unsigned long long)((uint64_t)grid * rank * 4));
    uint64_t offset = (uint64_t)grid * rank * 4;
    for (unsigned projection = 0; projection < PROJECTIONS; projection++) {
        char prefix[96];
        projection_prefix(projection, prefix, sizeof(prefix));
        uint32_t outputs = projection < H3_DIT_BLOCKS ? BLOCK_OUTPUT : FINAL_OUTPUT;
        uint64_t weight_bytes = (uint64_t)outputs * rank * 4;
        uint64_t bias_bytes = (uint64_t)outputs * 4;
        used += (size_t)snprintf(json + used, capacity - used,
            ",\"%s.weight\":{\"dtype\":\"F32\",\"shape\":[%u,%u],"
            "\"data_offsets\":[%llu,%llu]}"
            ",\"%s.bias\":{\"dtype\":\"F32\",\"shape\":[%u],"
            "\"data_offsets\":[%llu,%llu]}",
            prefix, outputs, rank, (unsigned long long)offset,
            (unsigned long long)(offset + weight_bytes), prefix, outputs,
            (unsigned long long)(offset + weight_bytes),
            (unsigned long long)(offset + weight_bytes + bias_bytes));
        offset += weight_bytes + bias_bytes;
        if (used + 512 > capacity) {
            free(json);
            return NULL;
        }
    }
    json[used++] = '}';
    while (used % 8) json[used++] = ' ';
    *length = used;
    return json;
}

int h3_build_adaln_curve(const char *transformer_dir,
                         const char *output_path,
                         const char *shader_source_path,
                         unsigned grid, unsigned rank,
                         h3_progress_callback on_progress,
                         void *callback_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!grid) grid = H3_ADALN_CURVE_DEFAULT_GRID;
    if (!rank) rank = H3_ADALN_CURVE_DEFAULT_RANK;
    if (!transformer_dir || !*transformer_dir || !output_path ||
        !*output_path || grid < 2 || grid > 65537 || rank > grid ||
        rank > H3_DIT_TIME_DIM) {
        fail(error, error_size, "invalid AdaLN curve build arguments");
        return 0;
    }
    const uint32_t width = H3_DIT_TIME_DIM;
    h3_weight_store *store = h3_weight_store_open(transformer_dir, error,
                                                   error_size);
    if (!store) return 0;
    uint8_t model_id[32];
    h3_weight_store_fingerprint(store, model_id);

    float *times = malloc((size_t)grid * sizeof(*times));
    float *samples = malloc((size_t)grid * width * sizeof(*samples));
    double *wide = malloc((size_t)grid * width * sizeof(*wide));
    double *basis = NULL, *table = NULL;
    float *basis_f32 = NULL, *table_f32 = NULL;
    char *json = NULL;
    FILE *out = NULL;
    char *partial = NULL;
    int ok = 0;
    if (!times || !samples || !wide) {
        fail(error, error_size, "out of memory sampling the AdaLN curve");
        goto done;
    }
    for (uint32_t index = 0; index < grid; index++)
        times[index] = (float)index / (float)(grid - 1);
    {
        h3_gpu *gpu = h3_gpu_create(shader_source_path, error, error_size);
        if (!gpu) goto done;
        int sampled = h3_dit_time_silu(store, gpu, times, grid, samples,
                                       error, error_size);
        h3_gpu_free(gpu);
        if (!sampled) goto done;
    }
    for (size_t index = 0; index < (size_t)grid * width; index++)
        wide[index] = samples[index];
    basis = curve_basis(wide, grid, width, rank, error, error_size);
    if (!basis) goto done;
    table = malloc((size_t)grid * rank * sizeof(*table));
    basis_f32 = malloc((size_t)width * rank * sizeof(*basis_f32));
    table_f32 = malloc((size_t)grid * rank * sizeof(*table_f32));
    if (!table || !basis_f32 || !table_f32) {
        fail(error, error_size, "out of memory for the AdaLN curve table");
        goto done;
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (__LAPACK_int)grid,
                (__LAPACK_int)rank, (__LAPACK_int)width, 1.0, wide,
                (__LAPACK_int)width, basis, (__LAPACK_int)rank, 0.0, table,
                (__LAPACK_int)rank);
    for (size_t index = 0; index < (size_t)grid * rank; index++)
        table_f32[index] = (float)table[index];
    for (size_t index = 0; index < (size_t)width * rank; index++)
        basis_f32[index] = (float)basis[index];

    size_t json_length = 0;
    json = curve_header_json(grid, rank, model_id, &json_length);
    size_t partial_length = strlen(output_path) + sizeof(".partial");
    partial = malloc(partial_length);
    if (!json || !partial) {
        fail(error, error_size, "out of memory writing the AdaLN curve");
        goto done;
    }
    snprintf(partial, partial_length, "%s.partial", output_path);
    out = fopen(partial, "wb");
    if (!out) {
        fail(error, error_size, "cannot open %s: %s", partial, strerror(errno));
        goto done;
    }
    uint64_t header_length = json_length;
    if (!write_all(out, &header_length, sizeof(header_length)) ||
        !write_all(out, json, json_length) ||
        !write_all(out, table_f32, (size_t)grid * rank * sizeof(float))) {
        fail(error, error_size, "cannot write %s: %s", partial, strerror(errno));
        goto done;
    }
    for (unsigned projection = 0; projection < PROJECTIONS; projection++) {
        if (!write_projection(store, projection, basis_f32, rank, out, error,
                              error_size))
            goto done;
        if (on_progress && on_progress("AdaLN curve", (int)projection + 1,
                                       (int)PROJECTIONS, callback_opaque)) {
            fail(error, error_size, "AdaLN curve build cancelled");
            goto done;
        }
    }
    if (fclose(out) != 0) {
        out = NULL;
        fail(error, error_size, "cannot finish %s: %s", partial, strerror(errno));
        goto done;
    }
    out = NULL;
    if (rename(partial, output_path) != 0) {
        fail(error, error_size, "cannot rename %s: %s", partial, strerror(errno));
        goto done;
    }
    ok = 1;
done:
    if (out) fclose(out);
    if (!ok && partial) unlink(partial);
    free(partial);
    free(json);
    free(times);
    free(samples);
    free(wide);
    free(basis);
    free(table);
    free(basis_f32);
    free(table_f32);
    h3_weight_store_free(store);
    return ok;
}
