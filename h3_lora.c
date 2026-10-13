#include "h3_lora.h"

#include "h3_safetensors.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = H3_LORA_HIDDEN,
    HEAD_DIM = H3_LORA_HEAD_DIM,
    INNER = H3_LORA_INNER,
    FFN = H3_LORA_FFN,
    PARTS = 5,
    MAX_TOKENS = 24,
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* --- key parsing ------------------------------------------------------- */

static int is_number(const char *token) {
    if (!*token) return 0;
    for (const char *c = token; *c; c++)
        if (!isdigit((unsigned char)*c)) return 0;
    return 1;
}

static int tokens_match(char **tokens, int count, int at, const char *const *want,
                        int want_count) {
    if (at + want_count > count) return 0;
    for (int index = 0; index < want_count; index++)
        if (strcmp(tokens[at + index], want[index])) return 0;
    return 1;
}

/* '_' and '.' are both separators so kohya's flattened names
 * ("lora_unet_blocks_0_attn_qkv_proj") parse like dotted ones. */
int h3_lora_parse_key(const char *name, h3_lora_key *key) {
    if (!name || !key) return 0;
    char buffer[256];
    size_t length = strlen(name);
    if (length >= sizeof(buffer)) return 0;
    for (size_t index = 0; index <= length; index++) {
        char c = (char)tolower((unsigned char)name[index]);
        buffer[index] = c == '_' ? '.' : c;
    }
    char *tokens[MAX_TOKENS];
    int count = 0;
    for (char *cursor = strtok(buffer, "."); cursor && count < MAX_TOKENS;
         cursor = strtok(NULL, "."))
        tokens[count++] = cursor;

    int blocks_at = -1;
    for (int index = 0; index + 1 < count; index++)
        if (!strcmp(tokens[index], "blocks") && is_number(tokens[index + 1])) {
            blocks_at = index;
            break;
        }
    if (blocks_at < 0) return 0;

    h3_lora_key result;
    memset(&result, 0, sizeof(result));
    result.style = H3_LORA_STYLE_NATIVE;
    for (int index = 0; index < blocks_at; index++) {
        if (!strcmp(tokens[index], "refiner")) result.refiner = 1;
        if (!strcmp(tokens[index], "diffusion")) result.style = H3_LORA_STYLE_COMFYUI;
        if (!strcmp(tokens[index], "unet")) result.style = H3_LORA_STYLE_KOHYA;
        if (!strcmp(tokens[index], "transformer") &&
            result.style == H3_LORA_STYLE_NATIVE)
            result.style = H3_LORA_STYLE_DIFFUSERS;
    }
    result.block = atoi(tokens[blocks_at + 1]);
    if (result.block < 0 || result.block >= (result.refiner ?
            H3_LORA_REFINER_BLOCKS : H3_LORA_DIT_BLOCKS)) return 0;

    static const char *const qkv[] = {"attn", "qkv", "proj"};
    static const char *const to_q[] = {"attn", "to", "q"};
    static const char *const to_k[] = {"attn", "to", "k"};
    static const char *const to_v[] = {"attn", "to", "v"};
    static const char *const out_proj[] = {"attn", "out", "proj"};
    static const char *const to_out[] = {"attn", "to", "out", "0"};
    static const char *const fc1[] = {"mlp", "fc1"};
    static const char *const fc2[] = {"mlp", "fc2"};
    static const char *const ff0[] = {"ff", "net", "0", "proj"};
    static const char *const ff2[] = {"ff", "net", "2"};
    static const struct {
        const char *const *tokens;
        int count;
        int projection;
        h3_lora_part part;
        int diffusers;
    } modules[] = {
        {qkv, 3, H3_LORA_QKV, H3_LORA_PART_ALL, 0},
        {to_q, 3, H3_LORA_QKV, H3_LORA_PART_Q, 1},
        {to_k, 3, H3_LORA_QKV, H3_LORA_PART_K, 1},
        {to_v, 3, H3_LORA_QKV, H3_LORA_PART_V, 1},
        {out_proj, 3, H3_LORA_OUT, H3_LORA_PART_ALL, 0},
        {to_out, 4, H3_LORA_OUT, H3_LORA_PART_ALL, 1},
        {fc1, 2, H3_LORA_FC1, H3_LORA_PART_ALL, 0},
        {fc2, 2, H3_LORA_FC2, H3_LORA_PART_ALL, 0},
        {ff0, 4, H3_LORA_FC1, H3_LORA_PART_FC1_VALUE_GATE, 1},
        {ff2, 3, H3_LORA_FC2, H3_LORA_PART_ALL, 1},
    };
    int module_at = blocks_at + 2;
    int consumed = 0;
    for (size_t index = 0; index < sizeof(modules) / sizeof(*modules); index++) {
        if (!tokens_match(tokens, count, module_at, modules[index].tokens,
                          modules[index].count)) continue;
        result.projection = modules[index].projection;
        result.part = modules[index].part;
        if (modules[index].diffusers) result.style = H3_LORA_STYLE_DIFFUSERS;
        consumed = modules[index].count;
        break;
    }
    if (!consumed) return 0;

    int suffix_at = module_at + consumed;
    if (suffix_at < count && !strcmp(tokens[count - 1], "alpha") &&
        suffix_at == count - 1) {
        result.role = H3_LORA_ROLE_ALPHA;
    } else {
        for (int index = suffix_at; index + 1 < count; index++) {
            if (strcmp(tokens[index], "lora")) continue;
            const char *which = tokens[index + 1];
            if (!strcmp(which, "a") || !strcmp(which, "down"))
                result.role = H3_LORA_ROLE_A;
            else if (!strcmp(which, "b") || !strcmp(which, "up"))
                result.role = H3_LORA_ROLE_B;
            break;
        }
    }
    if (!result.role) return 0;
    *key = result;
    return 1;
}

/* --- engine layout ----------------------------------------------------- */

uint32_t h3_lora_group_count(int projection) {
    return projection == H3_LORA_QKV ? 3u : 1u;
}

uint32_t h3_lora_group_rows(int projection) {
    switch (projection) {
    case H3_LORA_QKV: return INNER;
    case H3_LORA_FC1: return 2u * FFN;
    default: return HIDDEN;
    }
}

uint32_t h3_lora_rows(int projection) {
    return h3_lora_group_rows(projection) * h3_lora_group_count(projection);
}

uint32_t h3_lora_columns(int projection) {
    switch (projection) {
    case H3_LORA_OUT: return INNER;
    case H3_LORA_FC2: return FFN;
    default: return HIDDEN;
    }
}

uint32_t h3_lora_engine_row(int projection, int group, uint32_t row) {
    if (projection != H3_LORA_QKV) return row;
    return (row / HEAD_DIM) * 3u * HEAD_DIM + (uint32_t)group * HEAD_DIM +
           row % HEAD_DIM;
}

/* --- adapter index ----------------------------------------------------- */

typedef struct {
    int a, b, alpha;   /* tensor indices, -1 when absent */
} module_slot;

typedef struct {
    h3_st_header header;
    int has_header;
    float strength;
    float metadata_alpha;   /* <= 0 when the file records none */
    module_slot main[H3_LORA_DIT_BLOCKS][H3_LORA_PROJECTIONS][PARTS];
    module_slot refiner[H3_LORA_REFINER_BLOCKS][H3_LORA_PROJECTIONS][PARTS];
    int unsupported;
    int style_counts[4];
} adapter_index;

static module_slot *slot_for(adapter_index *adapter, const h3_lora_key *key) {
    return key->refiner ?
        &adapter->refiner[key->block][key->projection][key->part] :
        &adapter->main[key->block][key->projection][key->part];
}

static float metadata_alpha(const h3_st_header *header) {
    static const char *const keys[] = {"alpha", "lora_alpha", "ss_network_alpha"};
    for (size_t index = 0; index < sizeof(keys) / sizeof(*keys); index++) {
        const char *text = h3_st_metadata(header, keys[index]);
        if (!text) continue;
        char *end = NULL;
        float value = strtof(text, &end);
        if (end != text && isfinite(value) && value > 0.0f) return value;
    }
    return 0.0f;
}

static int index_adapter(const char *path, adapter_index *adapter,
                         char *error, size_t error_size) {
    memset(adapter, 0, sizeof(*adapter));
    memset(adapter->main, 0xff, sizeof(adapter->main));
    memset(adapter->refiner, 0xff, sizeof(adapter->refiner));
    if (!h3_st_read_header(path, &adapter->header, error, error_size))
        return 0;
    adapter->has_header = 1;
    adapter->metadata_alpha = metadata_alpha(&adapter->header);
    for (size_t index = 0; index < adapter->header.tensor_count; index++) {
        h3_lora_key key;
        if (!h3_lora_parse_key(adapter->header.tensors[index].name, &key)) {
            adapter->unsupported++;
            continue;
        }
        adapter->style_counts[key.style]++;
        module_slot *slot = slot_for(adapter, &key);
        if (key.role == H3_LORA_ROLE_A) slot->a = (int)index;
        else if (key.role == H3_LORA_ROLE_B) slot->b = (int)index;
        else slot->alpha = (int)index;
    }
    return 1;
}

static void free_adapter(adapter_index *adapter) {
    if (adapter->has_header) h3_st_free_header(&adapter->header);
    adapter->has_header = 0;
}

static float half_to_float(uint16_t bits) {
    __fp16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static float bf16_to_float(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* Reads any 16/32-bit float tensor as F32. */
static float *read_f32(const adapter_index *adapter, int tensor_index,
                       char *error, size_t error_size) {
    const h3_st_tensor *tensor = &adapter->header.tensors[tensor_index];
    uint64_t elements = h3_st_tensor_elements(tensor);
    size_t item = h3_dtype_size(tensor->dtype);
    if (tensor->dtype != H3_DTYPE_F32 && tensor->dtype != H3_DTYPE_F16 &&
        tensor->dtype != H3_DTYPE_BF16) {
        fail(error, error_size, "LoRA tensor %s has unsupported dtype %s",
             tensor->name, h3_dtype_name(tensor->dtype));
        return NULL;
    }
    void *raw = malloc((size_t)elements * item);
    float *values = malloc((size_t)(elements ? elements : 1) * sizeof(float));
    if (!raw || !values) {
        free(raw); free(values);
        fail(error, error_size, "out of memory reading %s", tensor->name);
        return NULL;
    }
    if (!h3_st_read_data(&adapter->header, tensor, raw,
                         (size_t)elements * item, error, error_size)) {
        free(raw); free(values);
        return NULL;
    }
    for (uint64_t index = 0; index < elements; index++) {
        if (tensor->dtype == H3_DTYPE_F32)
            values[index] = ((const float *)raw)[index];
        else if (tensor->dtype == H3_DTYPE_F16)
            values[index] = half_to_float(((const uint16_t *)raw)[index]);
        else
            values[index] = bf16_to_float(((const uint16_t *)raw)[index]);
    }
    free(raw);
    return values;
}

/* --- inspection -------------------------------------------------------- */

static const char *style_name(int style) {
    switch (style) {
    case H3_LORA_STYLE_COMFYUI: return "comfyui";
    case H3_LORA_STYLE_DIFFUSERS: return "diffusers";
    case H3_LORA_STYLE_KOHYA: return "kohya";
    default: return "native";
    }
}

/* DMAD records its sampling as free text, e.g. "4 steps, video shift 12,
 * audio shift 2, re-noise step rule, 1344x768, ...". */
static void parse_sampling(const char *text, h3_lora_info *info) {
    if (!text) return;
    char lower[256];
    size_t length = 0;
    for (; text[length] && length + 1 < sizeof(lower); length++)
        lower[length] = (char)tolower((unsigned char)text[length]);
    lower[length] = '\0';
    if (strstr(lower, "re-noise") || strstr(lower, "renoise"))
        info->sampler = H3_SAMPLER_RENOISE;
    const char *shift = strstr(lower, "audio shift");
    if (shift) {
        float value = strtof(shift + strlen("audio shift"), NULL);
        if (value > 0.0f && value <= 100.0f) info->audio_sigma_shift = value;
    }
}

int h3_lora_inspect(const char *path, h3_lora_info *info, char *error,
                    size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !info) {
        fail(error, error_size, "invalid LoRA inspection arguments");
        return 0;
    }
    memset(info, 0, sizeof(*info));
    adapter_index *adapter = malloc(sizeof(*adapter));
    if (!adapter) {
        fail(error, error_size, "out of memory inspecting %s", path);
        return 0;
    }
    if (!index_adapter(path, adapter, error, error_size)) {
        free(adapter);
        return 0;
    }
    info->unsupported = adapter->unsupported;
    int best = 0;
    for (int style = 1; style < 4; style++)
        if (adapter->style_counts[style] > adapter->style_counts[best]) best = style;
    snprintf(info->format, sizeof(info->format), "%s", style_name(best));
    static const char *const base_keys[] = {
        "base_model", "ss_base_model_version", "modelspec.architecture",
        "ss_sd_model_name"};
    for (size_t index = 0; index < sizeof(base_keys) / sizeof(*base_keys); index++) {
        const char *value = h3_st_metadata(&adapter->header, base_keys[index]);
        if (value) {
            snprintf(info->base_model, sizeof(info->base_model), "%s", value);
            break;
        }
    }
    parse_sampling(h3_st_metadata(&adapter->header, "sampling"), info);
    for (int refiner = 0; refiner < 2; refiner++) {
        int blocks = refiner ? H3_LORA_REFINER_BLOCKS : H3_LORA_DIT_BLOCKS;
        for (int block = 0; block < blocks; block++) {
            int touched = 0;
            for (int projection = 0; projection < H3_LORA_PROJECTIONS; projection++) {
                int matched = 0;
                for (int part = 0; part < PARTS; part++) {
                    module_slot slot = refiner ?
                        adapter->refiner[block][projection][part] :
                        adapter->main[block][projection][part];
                    if (slot.a < 0 || slot.b < 0) {
                        if (slot.a >= 0 || slot.b >= 0) info->unsupported++;
                        continue;
                    }
                    const h3_st_tensor *a = &adapter->header.tensors[slot.a];
                    int rank = a->ndim == 2 ? (int)a->shape[0] : 0;
                    if (!info->rank_min || rank < info->rank_min) info->rank_min = rank;
                    if (rank > info->rank_max) info->rank_max = rank;
                    matched = 1;
                }
                if (matched) {
                    info->projections++;
                    touched = 1;
                }
            }
            if (touched) {
                if (refiner) info->refiner_blocks++;
                else info->blocks++;
            }
        }
    }
    free_adapter(adapter);
    free(adapter);
    if (!info->projections) {
        fail(error, error_size,
             "%s has no LoRA factors for any MiniMax-H3 block projection",
             path);
        return 0;
    }
    return 1;
}

/* --- set --------------------------------------------------------------- */

typedef struct {
    uint32_t rank;
    h3_gpu_tensor *b;    /* BF16 [group_rows, rank], scale folded in */
    h3_gpu_tensor *at;   /* BF16 [columns, rank] = A transposed */
} group_factors;

typedef struct {
    group_factors groups[3];
} target_factors;

struct h3_lora_set {
    target_factors main[H3_LORA_DIT_BLOCKS][H3_LORA_PROJECTIONS];
    target_factors refiner[H3_LORA_REFINER_BLOCKS][H3_LORA_PROJECTIONS];
    h3_gpu_tensor *delta;
    size_t delta_elements;
};

static target_factors *target_for(h3_lora_set *set, int refiner,
                                  unsigned block, int projection) {
    if (projection < 0 || projection >= H3_LORA_PROJECTIONS) return NULL;
    if (refiner) return block < H3_LORA_REFINER_BLOCKS ?
        &set->refiner[block][projection] : NULL;
    return block < H3_LORA_DIT_BLOCKS ? &set->main[block][projection] : NULL;
}

/* Which group a part feeds and where its B rows for that group start. */
static int part_covers_group(int projection, h3_lora_part part, int group,
                             uint32_t *source_row_offset) {
    *source_row_offset = 0;
    if (projection != H3_LORA_QKV) return group == 0;
    if (part == H3_LORA_PART_ALL) {
        *source_row_offset = (uint32_t)group * INNER;
        return 1;
    }
    return (int)part - (int)H3_LORA_PART_Q == group;
}

static uint32_t part_rows(int projection, h3_lora_part part) {
    if (projection == H3_LORA_QKV && part != H3_LORA_PART_ALL) return INNER;
    return h3_lora_rows(projection);
}

typedef struct {
    float *a;           /* [rank, columns] */
    float *b;           /* [part_rows, rank] */
    uint32_t rank;
    float scale;
    h3_lora_part part;
} contribution;

static int load_contribution(const adapter_index *adapter, module_slot slot,
                             int projection, h3_lora_part part,
                             const char *path, contribution *out,
                             char *error, size_t error_size) {
    memset(out, 0, sizeof(*out));
    const h3_st_tensor *a = &adapter->header.tensors[slot.a];
    const h3_st_tensor *b = &adapter->header.tensors[slot.b];
    uint32_t columns = h3_lora_columns(projection);
    uint32_t rows = part_rows(projection, part);
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != columns ||
        b->shape[0] != rows || b->shape[1] != a->shape[0] ||
        a->shape[0] == 0 || a->shape[0] > 4096) {
        fail(error, error_size,
             "%s: %s / %s have shapes incompatible with MiniMax-H3 "
             "([%u, rank] / [rank, %u] expected)", path, b->name, a->name,
             rows, columns);
        return 0;
    }
    out->rank = (uint32_t)a->shape[0];
    float alpha = adapter->metadata_alpha;
    if (slot.alpha >= 0) {
        float *value = read_f32(adapter, slot.alpha, error, error_size);
        if (!value) return 0;
        alpha = value[0];
        free(value);
    }
    out->scale = adapter->strength *
        (alpha > 0.0f ? alpha / (float)out->rank : 1.0f);
    out->part = part;
    out->a = read_f32(adapter, slot.a, error, error_size);
    out->b = out->a ? read_f32(adapter, slot.b, error, error_size) : NULL;
    if (!out->a || !out->b) {
        free(out->a); free(out->b);
        out->a = out->b = NULL;
        return 0;
    }
    return 1;
}

/* diffusers' [value; gate] FC1 lands on the engine's [gate; value] by
 * swapping halves; everything else is the group's slice of B. */
static uint32_t source_row(const contribution *part, uint32_t row,
                           uint32_t offset) {
    return part->part == H3_LORA_PART_FC1_VALUE_GATE ?
        (row + FFN) % (2u * FFN) : row + offset;
}

static int upload_group(h3_gpu *gpu, int projection, int group,
                        const contribution *parts, int part_count,
                        group_factors *out, char *error, size_t error_size) {
    uint32_t rows = h3_lora_group_rows(projection);
    uint32_t columns = h3_lora_columns(projection);
    /* Rank columns whose B slice is all zero for this group contribute
     * nothing - e.g. every ComfyUI conversion of a diffusers adapter stores
     * qkv_proj as a block-diagonal B, so two thirds of each q/k/v group's
     * rank would otherwise be multiplied through as zeros. */
    uint8_t *keep[H3_MAX_LORAS * 5] = {0};
    uint32_t rank = 0;
    int ok = 1;
    for (int index = 0; index < part_count && ok; index++) {
        const contribution *part = &parts[index];
        uint32_t offset;
        if (!part_covers_group(projection, part->part, group, &offset)) continue;
        keep[index] = calloc(part->rank, 1);
        if (!keep[index]) { ok = 0; break; }
        for (uint32_t row = 0; row < rows; row++) {
            const float *values = part->b +
                (size_t)source_row(part, row, offset) * part->rank;
            for (uint32_t k = 0; k < part->rank; k++)
                if (values[k] != 0.0f) keep[index][k] = 1;
        }
        for (uint32_t k = 0; k < part->rank; k++) rank += keep[index][k];
    }
    uint16_t *b = NULL, *at = NULL;
    if (ok && rank) {
        /* Zero columns up to a multiple of 32 let the delta matmul use the
         * M5 TensorOps tile kernel; they add nothing to the product. */
        rank = (rank + 31u) & ~31u;
        b = calloc((size_t)rows * rank, sizeof(*b));
        at = calloc((size_t)columns * rank, sizeof(*at));
        ok = b && at;
    }
    uint32_t base = 0;
    for (int index = 0; index < part_count && ok && rank; index++) {
        const contribution *part = &parts[index];
        uint32_t offset;
        if (!keep[index] ||
            !part_covers_group(projection, part->part, group, &offset)) continue;
        for (uint32_t k = 0; k < part->rank; k++) {
            if (!keep[index][k]) continue;
            for (uint32_t row = 0; row < rows; row++)
                b[(size_t)row * rank + base] = float_to_bf16(
                    part->b[(size_t)source_row(part, row, offset) * part->rank + k] *
                    part->scale);
            const float *a_row = part->a + (size_t)k * columns;
            for (uint32_t column = 0; column < columns; column++)
                at[(size_t)column * rank + base] = float_to_bf16(a_row[column]);
            base++;
        }
    }
    for (int index = 0; index < part_count; index++) free(keep[index]);
    if (!ok) {
        free(b); free(at);
        fail(error, error_size, "out of memory staging LoRA factors");
        return 0;
    }
    if (!rank) return 1;
    out->rank = rank;
    out->b = h3_gpu_tensor_from_bf16(gpu, b, (size_t)rows * rank);
    out->at = h3_gpu_tensor_from_bf16(gpu, at, (size_t)columns * rank);
    free(b); free(at);
    if (!out->b || !out->at) {
        fail(error, error_size, "cannot upload LoRA factors: %s",
             h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

static void free_target(target_factors *target) {
    for (int group = 0; group < 3; group++) {
        h3_gpu_tensor_free(target->groups[group].b);
        h3_gpu_tensor_free(target->groups[group].at);
        memset(&target->groups[group], 0, sizeof(target->groups[group]));
    }
}

void h3_lora_set_free(h3_lora_set *set) {
    if (!set) return;
    for (int block = 0; block < H3_LORA_DIT_BLOCKS; block++)
        for (int projection = 0; projection < H3_LORA_PROJECTIONS; projection++)
            free_target(&set->main[block][projection]);
    for (int block = 0; block < H3_LORA_REFINER_BLOCKS; block++)
        for (int projection = 0; projection < H3_LORA_PROJECTIONS; projection++)
            free_target(&set->refiner[block][projection]);
    h3_gpu_tensor_free(set->delta);
    free(set);
}

h3_lora_set *h3_lora_set_load(h3_gpu *gpu, const h3_lora *loras,
                              size_t count, char *error, size_t error_size) {
    if (!gpu || (count && !loras) || count > H3_MAX_LORAS) {
        fail(error, error_size, "at most %d LoRA adapters are supported",
             H3_MAX_LORAS);
        return NULL;
    }
    h3_lora_set *set = calloc(1, sizeof(*set));
    adapter_index *adapters = calloc(count ? count : 1, sizeof(*adapters));
    if (!set || !adapters) {
        free(set); free(adapters);
        fail(error, error_size, "out of memory loading LoRA adapters");
        return NULL;
    }
    int ok = 1;
    for (size_t index = 0; index < count && ok; index++) {
        if (!loras[index].path || !*loras[index].path ||
            !isfinite(loras[index].strength)) {
            fail(error, error_size, "LoRA %zu has no path or a non-finite strength",
                 index + 1);
            ok = 0;
            break;
        }
        ok = index_adapter(loras[index].path, &adapters[index], error,
                           error_size);
        adapters[index].strength = loras[index].strength;
        if (ok && adapters[index].unsupported)
            fprintf(stderr, "h3: LoRA %s: %d tensors outside the supported "
                    "block projections are ignored\n", loras[index].path,
                    adapters[index].unsupported);
    }

    contribution parts[H3_MAX_LORAS * PARTS];
    for (int refiner = 0; refiner < 2 && ok; refiner++) {
        int blocks = refiner ? H3_LORA_REFINER_BLOCKS : H3_LORA_DIT_BLOCKS;
        for (int block = 0; block < blocks && ok; block++)
            for (int projection = 0; projection < H3_LORA_PROJECTIONS && ok;
                 projection++) {
                int part_count = 0;
                for (size_t index = 0; index < count && ok; index++)
                    for (int part = 0; part < PARTS && ok; part++) {
                        module_slot slot = refiner ?
                            adapters[index].refiner[block][projection][part] :
                            adapters[index].main[block][projection][part];
                        if (slot.a < 0 || slot.b < 0) continue;
                        ok = load_contribution(
                            &adapters[index], slot, projection,
                            (h3_lora_part)part, loras[index].path,
                            &parts[part_count], error, error_size);
                        if (ok) part_count++;
                    }
                target_factors *target =
                    target_for(set, refiner, (unsigned)block, projection);
                for (uint32_t group = 0;
                     ok && group < h3_lora_group_count(projection); group++)
                    ok = upload_group(gpu, projection, (int)group, parts,
                                      part_count, &target->groups[group],
                                      error, error_size);
                for (int index = 0; index < part_count; index++) {
                    free(parts[index].a);
                    free(parts[index].b);
                }
                if (ok && part_count) {
                    size_t elements = (size_t)h3_lora_group_rows(projection) *
                                      h3_lora_columns(projection);
                    if (elements > set->delta_elements)
                        set->delta_elements = elements;
                }
            }
    }
    for (size_t index = 0; index < count; index++) free_adapter(&adapters[index]);
    free(adapters);
    if (ok && set->delta_elements) {
        set->delta = h3_gpu_tensor_new_bf16(gpu, set->delta_elements);
        if (!set->delta) {
            fail(error, error_size, "cannot allocate LoRA delta scratch: %s",
                 h3_gpu_error(gpu));
            ok = 0;
        }
    }
    if (!ok) {
        h3_lora_set_free(set);
        return NULL;
    }
    return set;
}

void h3_lora_set_release(h3_lora_set *set, int refiner, int projection) {
    if (!set) return;
    int blocks = refiner ? H3_LORA_REFINER_BLOCKS : H3_LORA_DIT_BLOCKS;
    for (int block = 0; block < blocks; block++) {
        target_factors *target =
            target_for(set, refiner, (unsigned)block, projection);
        if (target) free_target(target);
    }
}

int h3_lora_set_covers(const h3_lora_set *set, int refiner, unsigned block,
                       int projection) {
    if (!set) return 0;
    const target_factors *target =
        target_for((h3_lora_set *)set, refiner, block, projection);
    if (!target) return 0;
    for (int group = 0; group < 3; group++)
        if (target->groups[group].rank) return 1;
    return 0;
}

static int apply(h3_lora_set *set, h3_gpu *gpu, int refiner, unsigned block,
                 int projection, h3_gpu_tensor *weight, h3_gpu_tensor *scales) {
    target_factors *target = set ?
        target_for(set, refiner, block, projection) : NULL;
    if (!target) return 1;
    uint32_t rows = h3_lora_group_rows(projection);
    uint32_t columns = h3_lora_columns(projection);
    uint32_t interleaved = projection == H3_LORA_QKV;
    for (uint32_t group = 0; group < h3_lora_group_count(projection); group++) {
        group_factors *factors = &target->groups[group];
        if (!factors->rank) continue;
        if (!h3_gpu_lora_delta_bf16(gpu, set->delta, factors->b, factors->at,
                                    rows, factors->rank, columns)) return 0;
        /* Fixed per projection and block, so every step's patch of a
         * streamed block rounds identically. */
        uint32_t seed = ((uint32_t)refiner << 24) ^ (block << 8) ^
                        ((uint32_t)projection << 4) ^ group;
        int ok = scales ?
            h3_gpu_lora_add_int8(gpu, weight, scales, set->delta, rows,
                                 columns, interleaved, group, HEAD_DIM,
                                 seed) :
            h3_gpu_lora_add_rows_bf16(gpu, weight, set->delta, rows, columns,
                                      interleaved, group, HEAD_DIM, seed);
        if (!ok) return 0;
    }
    return 1;
}

int h3_lora_set_apply_bf16(h3_lora_set *set, h3_gpu *gpu, int refiner,
                           unsigned block, int projection,
                           h3_gpu_tensor *weight) {
    return apply(set, gpu, refiner, block, projection, weight, NULL);
}

int h3_lora_set_apply_int8(h3_lora_set *set, h3_gpu *gpu, int refiner,
                           unsigned block, int projection,
                           h3_gpu_tensor *weight, h3_gpu_tensor *scales) {
    return apply(set, gpu, refiner, block, projection, weight, scales);
}
