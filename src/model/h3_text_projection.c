#define ACCELERATE_NEW_LAPACK
#include "h3_text_projection.h"

#include "h3_quant.h"
#include "h3_safetensors.h"
#include "h3_text_encoder.h"

#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PROJECTION_MAX_INPUT = 8192, PROJECTION_MAX_TAP = 64 };

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* A tensor of the file as F32 on the host: F32 as stored, F16 widened. */
static float *read_values(const h3_st_header *header, const char *name,
                          uint64_t rows, uint64_t columns, char *error,
                          size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        fail(error, error_size, "the text projection has no %s", name);
        return NULL;
    }
    int shaped = columns ?
        tensor->ndim == 2 && tensor->shape[0] == rows &&
        tensor->shape[1] == columns :
        tensor->ndim == 1 && tensor->shape[0] == rows;
    if (!shaped || (tensor->dtype != H3_DTYPE_F32 &&
                    tensor->dtype != H3_DTYPE_F16)) {
        fail(error, error_size, "the text projection's %s is not F32 or F16 "
             "of %llu%s values", name, (unsigned long long)rows,
             columns ? " rows of its output's" : "");
        return NULL;
    }
    size_t count = (size_t)(rows * (columns ? columns : 1));
    float *values = malloc(count * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory reading the text projection");
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_st_read_data(header, tensor, values, count * sizeof(*values),
                             error, error_size)) {
            free(values);
            return NULL;
        }
    } else {
        uint16_t *halves = malloc(count * sizeof(*halves));
        if (!halves || !h3_st_read_data(header, tensor, halves,
                                        count * sizeof(*halves), error,
                                        error_size)) {
            if (!halves)
                fail(error, error_size,
                     "out of memory reading the text projection");
            free(halves);
            free(values);
            return NULL;
        }
        for (size_t index = 0; index < count; index++)
            values[index] = h3_half_to_float(halves[index]);
        free(halves);
    }
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            fail(error, error_size, "the text projection's %s has a "
                 "non-finite value", name);
            free(values);
            return NULL;
        }
    }
    return values;
}

void h3_text_projection_free(h3_text_projection *projection) {
    if (!projection) return;
    free(projection->weight);
    free(projection->input_mean);
    free(projection->input_std);
    free(projection->output_mean);
    free(projection->output_std);
    free(projection->sink);
    memset(projection, 0, sizeof(*projection));
}

int h3_text_projection_load(const char *path, h3_text_projection *projection,
                            char *error, size_t error_size) {
    if (projection) memset(projection, 0, sizeof(*projection));
    if (!path || !projection) {
        fail(error, error_size, "invalid text projection arguments");
        return 0;
    }
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return 0;
    int ok = 0;
    for (size_t index = 0; index < header.tensor_count; index++) {
        if (!strncmp(header.tensors[index].name, "mlp.", 4)) {
            fail(error, error_size, "the text projection carries a residual "
                 "network (%s: a -mlp file), which is not supported; use the "
                 "matrix-only file", header.tensors[index].name);
            goto done;
        }
    }
    const h3_st_tensor *input = h3_st_find(&header, "mean_in");
    const h3_st_tensor *output = h3_st_find(&header, "mean_out");
    const h3_st_tensor *weight = h3_st_find(&header, "W");
    if (!input || !output || !weight || input->ndim != 1 ||
        output->ndim != 1 || weight->ndim != 2) {
        fail(error, error_size, "the text projection lacks W, mean_in or "
             "mean_out");
        goto done;
    }
    uint64_t input_width = input->shape[0], output_width = output->shape[0];
    if (!input_width || input_width > PROJECTION_MAX_INPUT ||
        output_width != H3_TEXT_HIDDEN_SIZE) {
        fail(error, error_size, "the text projection goes from %llu to %llu "
             "values, not to the %u of H3's text rows",
             (unsigned long long)input_width,
             (unsigned long long)output_width, H3_TEXT_HIDDEN_SIZE);
        goto done;
    }
    if (weight->shape[0] != input_width || weight->shape[1] != output_width) {
        /* [output, input] would be a Linear's weight: another convention
         * than ClipProj's `hidden @ W`. */
        fail(error, error_size, "the text projection's W is [%llu, %llu], "
             "not [%llu, %llu]", (unsigned long long)weight->shape[0],
             (unsigned long long)weight->shape[1],
             (unsigned long long)input_width,
             (unsigned long long)output_width);
        goto done;
    }
    const char *tap = h3_st_metadata(&header, "tap");
    char *end = NULL;
    errno = 0;
    long tap_value = tap ? strtol(tap, &end, 10) : 0;
    if (!tap || end == tap || *end || errno || tap_value < 1 ||
        tap_value > PROJECTION_MAX_TAP) {
        fail(error, error_size, "the text projection's metadata has %s%s%s, "
             "not the hidden state it reads", tap ? "tap \"" : "no tap",
             tap ? tap : "", tap ? "\"" : "");
        goto done;
    }
    projection->tap = (int)tap_value;
    projection->input_width = (size_t)input_width;
    projection->output_width = (size_t)output_width;
    projection->weight = read_values(&header, "W", input_width, output_width,
                                     error, error_size);
    if (projection->weight)
        projection->input_mean = read_values(&header, "mean_in", input_width,
                                             0, error, error_size);
    if (projection->input_mean)
        projection->input_std = read_values(&header, "std_in", input_width, 0,
                                            error, error_size);
    if (projection->input_std)
        projection->output_mean = read_values(&header, "mean_out",
                                              output_width, 0, error,
                                              error_size);
    if (projection->output_mean)
        projection->output_std = read_values(&header, "std_out", output_width,
                                             0, error, error_size);
    if (projection->output_std)
        projection->sink = read_values(&header, "sink_out", output_width, 0,
                                       error, error_size);
    if (!projection->sink) goto done;
    for (size_t index = 0; index < projection->input_width; index++) {
        if (!(projection->input_std[index] > 0.0f)) {
            fail(error, error_size, "the text projection's std_in has a "
                 "value that is not positive");
            goto done;
        }
    }
    ok = 1;
done:
    h3_st_free_header(&header);
    if (!ok) h3_text_projection_free(projection);
    return ok;
}

static float bf16_value(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static uint16_t bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

int h3_text_projection_apply(const h3_text_projection *projection,
                             const uint16_t *hidden, size_t tokens,
                             uint16_t *output, char *error,
                             size_t error_size) {
    if (!projection || !projection->weight || !hidden || !output || !tokens ||
        tokens > INT32_MAX) {
        fail(error, error_size, "invalid text projection arguments");
        return 0;
    }
    size_t input_width = projection->input_width;
    size_t output_width = projection->output_width;
    float *standardized = malloc(tokens * input_width * sizeof(*standardized));
    float *projected = malloc(tokens * output_width * sizeof(*projected));
    int ok = standardized && projected;
    if (!ok) fail(error, error_size, "out of memory for the text projection");
    if (ok) {
        for (size_t token = 0; token < tokens; token++)
            for (size_t column = 0; column < input_width; column++)
                standardized[token * input_width + column] =
                    (bf16_value(hidden[token * input_width + column]) -
                     projection->input_mean[column]) /
                    projection->input_std[column];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)tokens,
                    (int)output_width, (int)input_width, 1.0f, standardized,
                    (int)input_width, projection->weight, (int)output_width,
                    0.0f, projected, (int)output_width);
        for (size_t token = 0; ok && token < tokens; token++)
            for (size_t column = 0; column < output_width; column++) {
                float value = token == 0 ? projection->sink[column] :
                    projected[token * output_width + column] *
                    projection->output_std[column] +
                    projection->output_mean[column];
                if (!isfinite(value)) {
                    fail(error, error_size, "the text projection produced a "
                         "non-finite value");
                    ok = 0;
                    break;
                }
                output[token * output_width + column] = bf16_bits(value);
            }
    }
    free(standardized);
    free(projected);
    return ok;
}
