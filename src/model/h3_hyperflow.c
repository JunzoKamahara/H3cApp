#include "h3_hyperflow.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

enum {
    HF_HIDDEN = 5376,
    HF_INNER = 7168,
    HF_FFN = 14336,
    HF_TIME_INPUT = 256,
    HF_ADALN_OUTPUT = 3 * 6 * 5376,
    HF_FINAL_OUTPUT = 2 * 5376
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static const char *const linear_names[H3_HF_LINEARS] = {
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj",
    "mlp.fc1", "mlp.fc2"
};

/* in, out of each linear (the dense weight is [out, in]). */
static const uint32_t linear_in[H3_HF_LINEARS] = {
    HF_HIDDEN, HF_HIDDEN, HF_HIDDEN, HF_INNER, HF_HIDDEN, HF_FFN
};
static const uint32_t linear_out[H3_HF_LINEARS] = {
    HF_INNER, HF_INNER, HF_INNER, HF_HIDDEN, 2 * HF_FFN, HF_HIDDEN
};

const char *h3_hyperflow_linear_name(h3_hyperflow_linear linear) {
    return linear >= 0 && linear < H3_HF_LINEARS ? linear_names[linear] : "?";
}

int h3_hyperflow_block_factored(const h3_hyperflow_info *info, unsigned block,
                                h3_hyperflow_linear first,
                                h3_hyperflow_linear last) {
    if (!info || !info->two_time || block >= info->block_count) return 0;
    for (int linear = (int)first;
         linear <= (int)last && linear < H3_HF_LINEARS; linear++)
        if (info->rank[block][linear]) return 1;
    return 0;
}

static const h3_st_tensor *find(const h3_weight_store *store,
                                const char *name,
                                const h3_st_header **header) {
    return h3_weight_find(store, name, header);
}

static int has_shape(const h3_st_tensor *tensor, uint64_t rows,
                     uint64_t columns) {
    return tensor && tensor->ndim == 2 && tensor->shape[0] == rows &&
           tensor->shape[1] == columns;
}

static int has_vector(const h3_st_tensor *tensor, uint64_t width) {
    return tensor && tensor->ndim == 1 && tensor->shape[0] == width;
}

static int read_f32(const h3_weight_store *store, const char *name,
                    float *values, uint32_t capacity, uint32_t *count,
                    char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = find(store, name, &header);
    if (!tensor) return 0;
    uint64_t elements = h3_st_tensor_elements(tensor);
    if (tensor->dtype != H3_DTYPE_F32 || tensor->ndim != 1 || !elements ||
        elements > capacity) {
        fail(error, error_size, "%s: expected a short F32 vector", name);
        return -1;
    }
    char detail[256] = {0};
    if (!h3_st_read_data(header, tensor, values, (size_t)elements * 4u,
                         detail, sizeof(detail))) {
        fail(error, error_size, "%s: %s", name, detail);
        return -1;
    }
    *count = (uint32_t)elements;
    return 1;
}

static int check_embedder(const h3_weight_store *store, const char *prefix,
                          uint32_t time_dim, char *error, size_t error_size) {
    char name[128];
    const h3_st_tensor *in_w, *in_b, *out_w, *out_b;
    snprintf(name, sizeof(name), "%s.proj_in.weight", prefix);
    in_w = find(store, name, NULL);
    snprintf(name, sizeof(name), "%s.proj_in.bias", prefix);
    in_b = find(store, name, NULL);
    snprintf(name, sizeof(name), "%s.proj_out.weight", prefix);
    out_w = find(store, name, NULL);
    snprintf(name, sizeof(name), "%s.proj_out.bias", prefix);
    out_b = find(store, name, NULL);
    if (!has_shape(in_w, HF_HIDDEN, HF_TIME_INPUT) ||
        !has_vector(in_b, HF_HIDDEN) ||
        !has_shape(out_w, time_dim, HF_HIDDEN) ||
        !has_vector(out_b, time_dim)) {
        fail(error, error_size,
             "HyperFlow checkpoint: %s is not a %u -> %u -> %u embedder",
             prefix, HF_TIME_INPUT, HF_HIDDEN, time_dim);
        return 0;
    }
    return 1;
}

/* 1: dense, 2: factored (rank set), 0: missing or malformed (error set). */
static int inspect_linear(const h3_weight_store *store, unsigned block,
                          h3_hyperflow_linear linear, uint32_t *rank,
                          char *error, size_t error_size) {
    char name[160];
    snprintf(name, sizeof(name), "blocks.%u.%s.down.weight", block,
             linear_names[linear]);
    const h3_st_tensor *down = find(store, name, NULL);
    snprintf(name, sizeof(name), "blocks.%u.%s.up.weight", block,
             linear_names[linear]);
    const h3_st_tensor *up = find(store, name, NULL);
    if (!down && !up) {
        *rank = 0;
        return 1;
    }
    uint64_t r = down && down->ndim == 2 ? down->shape[0] : 0;
    if (!r || r >= linear_in[linear] || r >= linear_out[linear] ||
        !has_shape(down, r, linear_in[linear]) ||
        !has_shape(up, linear_out[linear], r)) {
        fail(error, error_size,
             "HyperFlow checkpoint: blocks.%u.%s down/up factors do not "
             "make a [%u, %u] linear", block, linear_names[linear],
             linear_out[linear], linear_in[linear]);
        return 0;
    }
    *rank = (uint32_t)r;
    return 2;
}

int h3_hyperflow_inspect(const h3_weight_store *store, h3_hyperflow_info *info,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!store || !info) {
        fail(error, error_size, "invalid HyperFlow inspect arguments");
        return 0;
    }
    memset(info, 0, sizeof(*info));
    info->block_count = H3_HYPERFLOW_MAX_BLOCKS;
    if (!find(store, "endpoint_time_embedder.proj_in.weight", NULL))
        return 1;
    info->two_time = 1;

    const h3_st_tensor *time_out = find(store, "time_embedder.proj_out.weight",
                                        NULL);
    if (!time_out || time_out->ndim != 2 || !time_out->shape[0] ||
        time_out->shape[0] > HF_HIDDEN) {
        fail(error, error_size,
             "HyperFlow checkpoint: time_embedder.proj_out.weight is missing "
             "or has an unexpected shape");
        return 0;
    }
    info->time_dim = (uint32_t)time_out->shape[0];
    if (!check_embedder(store, "time_embedder", info->time_dim, error,
                        error_size) ||
        !check_embedder(store, "endpoint_time_embedder", info->time_dim,
                        error, error_size))
        return 0;

    uint32_t count = 0;
    if (read_f32(store, "hyperflow_gate", &info->gate, 1, &count, error,
                 error_size) != 1 || count != 1 || !isfinite(info->gate)) {
        if (!error || !*error)
            fail(error, error_size,
                 "HyperFlow checkpoint: hyperflow_gate (F32[1]) is missing");
        return 0;
    }
    int sigmas = read_f32(store, "hyperflow_sigmas", info->sigmas,
                          H3_HYPERFLOW_MAX_SIGMAS, &info->sigma_count, error,
                          error_size);
    if (sigmas < 0) return 0;
    if (sigmas == 0) info->sigma_count = 0;

    uint32_t blocks = 0;
    while (blocks < H3_HYPERFLOW_MAX_BLOCKS) {
        char name[96];
        snprintf(name, sizeof(name), "blocks.%u.norm1.weight", blocks);
        if (!find(store, name, NULL)) break;
        blocks++;
    }
    for (uint32_t block = blocks; block < H3_HYPERFLOW_MAX_BLOCKS + 8u;
         block++) {
        char name[96];
        snprintf(name, sizeof(name), "blocks.%u.norm1.weight", block);
        if (find(store, name, NULL)) {
            fail(error, error_size,
                 "HyperFlow checkpoint: block %u exists after the missing "
                 "block %u (blocks must be numbered without gaps)",
                 block, blocks);
            return 0;
        }
    }
    if (!blocks) {
        fail(error, error_size, "HyperFlow checkpoint has no blocks");
        return 0;
    }
    info->block_count = blocks;

    char name[160];
    snprintf(name, sizeof(name), "final_layer.adaln_proj.linear.weight");
    if (!has_shape(find(store, name, NULL), HF_FINAL_OUTPUT, info->time_dim)) {
        fail(error, error_size,
             "HyperFlow checkpoint: %s is not [%u, %u]", name,
             HF_FINAL_OUTPUT, info->time_dim);
        return 0;
    }
    for (uint32_t block = 0; block < blocks; block++) {
        snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear.weight",
                 block);
        if (!has_shape(find(store, name, NULL), HF_ADALN_OUTPUT,
                       info->time_dim)) {
            fail(error, error_size,
                 "HyperFlow checkpoint: %s is not [%u, %u]", name,
                 HF_ADALN_OUTPUT, info->time_dim);
            return 0;
        }
        int kind[H3_HF_LINEARS];
        for (int linear = 0; linear < H3_HF_LINEARS; linear++) {
            kind[linear] = inspect_linear(store, block,
                                          (h3_hyperflow_linear)linear,
                                          &info->rank[block][linear],
                                          error, error_size);
            if (!kind[linear]) return 0;
            if (info->rank[block][linear] > info->max_rank)
                info->max_rank = info->rank[block][linear];
        }
        if (kind[H3_HF_Q] != kind[H3_HF_K] || kind[H3_HF_Q] != kind[H3_HF_V]) {
            fail(error, error_size,
                 "HyperFlow checkpoint: block %u factors only some of Q, K "
                 "and V", block);
            return 0;
        }
        /* The dense counterpart must exist exactly when a linear is not
         * factored. */
        static const char *const dense[4] = {
            "attn.qkv_proj.weight", "attn.out_proj.weight",
            "mlp.fc1.weight", "mlp.fc2.weight"
        };
        static const int dense_kind[4] = {H3_HF_Q, H3_HF_OUT, H3_HF_FC1,
                                          H3_HF_FC2};
        for (int index = 0; index < 4; index++) {
            snprintf(name, sizeof(name), "blocks.%u.%s", block, dense[index]);
            int present = find(store, name, NULL) != NULL;
            int factored = kind[dense_kind[index]] == 2;
            if (present == factored) {
                fail(error, error_size,
                     "HyperFlow checkpoint: block %u has %s %s", block,
                     factored ? "both a factored and a dense" : "neither a "
                     "factored nor a dense", dense[index]);
                return 0;
            }
        }
        for (int linear = 0; linear < H3_HF_LINEARS; linear++)
            if (kind[linear] == 2) {
                info->factored_blocks++;
                break;
            }
    }
    return 1;
}
