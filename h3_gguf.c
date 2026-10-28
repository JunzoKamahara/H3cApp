/* GGUF tensor tables (h3_gguf_read_header in h3_safetensors.h), so a GGUF
 * file can sit in a weight store next to safetensors shards. Layout (GGUF
 * v2/v3): "GGUF", u32 version, u64 tensor count, u64 key/value count; the
 * key/values (u64-length strings, a u32 type, then the value - arrays as a
 * u32 element type and u64 count); the tensor infos (name, u32 dims, u64
 * extents fastest-first, u32 ggml type, u64 offset into the data section);
 * then the data section, aligned to general.alignment (default 32). */
#include "h3_quant.h"
#include "h3_safetensors.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum {
    GGUF_UINT8 = 0, GGUF_INT8, GGUF_UINT16, GGUF_INT16, GGUF_UINT32,
    GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY,
    GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64,
};

typedef struct {
    FILE *file;
    uint64_t position;
    uint64_t size;
    const char *path;
    char *error;
    size_t error_size;
} reader;

static int fail(reader *r, const char *format, ...) {
    if (r->error && r->error_size && !r->error[0]) {
        char detail[256];
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(detail, sizeof(detail), format, arguments);
        va_end(arguments);
        snprintf(r->error, r->error_size, "%s: %s", r->path, detail);
    }
    return 0;
}

static int take(reader *r, void *out, uint64_t bytes) {
    if (bytes > r->size - r->position) return fail(r, "truncated GGUF header");
    if (bytes && fread(out, 1, (size_t)bytes, r->file) != bytes)
        return fail(r, "cannot read GGUF header");
    r->position += bytes;
    return 1;
}

static int skip(reader *r, uint64_t bytes) {
    if (bytes > r->size - r->position) return fail(r, "truncated GGUF header");
    if (fseeko(r->file, (off_t)bytes, SEEK_CUR) != 0)
        return fail(r, "cannot seek in GGUF header");
    r->position += bytes;
    return 1;
}

static int u32(reader *r, uint32_t *value) {
    uint8_t b[4];
    if (!take(r, b, 4)) return 0;
    *value = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
             ((uint32_t)b[3] << 24);
    return 1;
}

static int u64(reader *r, uint64_t *value) {
    uint8_t b[8];
    if (!take(r, b, 8)) return 0;
    *value = 0;
    for (int index = 7; index >= 0; index--) *value = (*value << 8) | b[index];
    return 1;
}

/* A u64-length string; longer than limit is an error (names, keys). */
static char *string(reader *r, uint64_t limit) {
    uint64_t length;
    if (!u64(r, &length)) return NULL;
    if (length > limit) {
        fail(r, "GGUF string of %llu bytes", (unsigned long long)length);
        return NULL;
    }
    char *text = malloc((size_t)length + 1);
    if (!text) {
        fail(r, "out of memory");
        return NULL;
    }
    if (!take(r, text, length)) {
        free(text);
        return NULL;
    }
    text[length] = '\0';
    return text;
}

static uint64_t scalar_size(uint32_t type) {
    switch (type) {
    case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL: return 1;
    case GGUF_UINT16: case GGUF_INT16: return 2;
    case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: return 4;
    case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: return 8;
    default: return 0;
    }
}

/* Arrays of arrays recurse in value(); real metadata nests at most one
 * level, so deeper nesting is refused rather than run off the stack. */
#define GGUF_MAX_ARRAY_DEPTH 8

/* Skips one value; strings and u32/u64 scalars are handed back when the
 * caller wants them (text/number may be NULL). depth is the number of
 * arrays the value sits inside (0 for a key's own value). */
static int value(reader *r, uint32_t type, char **text, uint64_t *number,
                 int depth) {
    if (type == GGUF_STRING) {
        char *s = string(r, UINT32_MAX);
        if (!s) return 0;
        if (text) *text = s;
        else free(s);
        return 1;
    }
    if (type == GGUF_ARRAY) {
        uint32_t element;
        uint64_t count;
        if (depth >= GGUF_MAX_ARRAY_DEPTH) return fail(r, "GGUF arrays nested too deeply");
        if (!u32(r, &element) || !u64(r, &count)) return 0;
        uint64_t size = scalar_size(element);
        if (size) {
            if (count > (r->size - r->position) / size)
                return fail(r, "truncated GGUF array");
            return skip(r, count * size);
        }
        if (element != GGUF_STRING && element != GGUF_ARRAY)
            return fail(r, "unknown GGUF array type %u", element);
        for (uint64_t index = 0; index < count; index++)
            if (!value(r, element, NULL, NULL, depth + 1)) return 0;
        return 1;
    }
    uint64_t size = scalar_size(type);
    if (!size) return fail(r, "unknown GGUF value type %u", type);
    if (number && (type == GGUF_UINT32 || type == GGUF_UINT64)) {
        if (type == GGUF_UINT32) {
            uint32_t v;
            if (!u32(r, &v)) return 0;
            *number = v;
            return 1;
        }
        return u64(r, number);
    }
    return skip(r, size);
}

/* ComfyUI-GGUF stores a tensor whose rows don't fill whole quant blocks
 * (H3's AdaLN projections, [96768, 2688]) flattened to rows of 256 and
 * records its PyTorch shape as comfy.gguf.orig_shape.<name>: the same
 * values in the same order, so the tensor takes that shape back. */
#define ORIG_SHAPE_PREFIX "comfy.gguf.orig_shape."

typedef struct {
    char *name;
    int ndim;
    uint64_t shape[8];
} orig_shape;

/* Reads an array value as a shape; anything but up to 8 integers leaves
 * ndim 0. */
static int shape_value(reader *r, orig_shape *out) {
    uint32_t element;
    uint64_t count;
    out->ndim = 0;
    if (!u32(r, &element) || !u64(r, &count)) return 0;
    int wide = element == GGUF_UINT64 || element == GGUF_INT64;
    if ((wide || element == GGUF_UINT32 || element == GGUF_INT32) &&
        count >= 1 && count <= 8) {
        for (uint64_t index = 0; index < count; index++) {
            uint64_t v = 0;
            uint32_t narrow;
            if (wide ? !u64(r, &v) : !u32(r, &narrow)) return 0;
            if (!wide) v = element == GGUF_INT32 ? (uint64_t)(int64_t)(int32_t)narrow : narrow;
            out->shape[index] = v;
        }
        out->ndim = (int)count;
        return 1;
    }
    uint64_t size = scalar_size(element);
    if (size) {
        if (count > (r->size - r->position) / size)
            return fail(r, "truncated GGUF array");
        return skip(r, count * size);
    }
    for (uint64_t index = 0; index < count; index++)
        if (!value(r, element, NULL, NULL, 1)) return 0;
    return 1;
}

static void apply_orig_shapes(h3_st_header *header, const orig_shape *shapes,
                              size_t count) {
    for (size_t s = 0; s < count; s++) {
        uint64_t want = 1;
        for (int d = 0; d < shapes[s].ndim; d++) {
            if (!shapes[s].shape[d] || shapes[s].shape[d] > (1ull << 40)) want = 0;
            else want *= shapes[s].shape[d];
        }
        for (size_t t = 0; want && t < header->tensor_count; t++) {
            h3_st_tensor *tensor = &header->tensors[t];
            if (strcmp(tensor->name, shapes[s].name)) continue;
            uint64_t have = 1;
            for (int d = 0; d < tensor->ndim; d++) have *= tensor->shape[d];
            if (have == want) {
                tensor->ndim = shapes[s].ndim;
                memcpy(tensor->shape, shapes[s].shape, sizeof(shapes[s].shape));
            }
            break;
        }
    }
}

int h3_gguf_read_header(const char *path, h3_st_header *header,
                        char *error, size_t error_size) {
    if (!path || !header) return 0;
    memset(header, 0, sizeof(*header));
    if (error && error_size) error[0] = '\0';
    reader r = {NULL, 0, 0, path, error, error_size};
    struct stat status;
    if (stat(path, &status) != 0) {
        if (error && error_size)
            snprintf(error, error_size, "%s: %s", path, strerror(errno));
        return 0;
    }
    r.size = (uint64_t)status.st_size;
    r.file = fopen(path, "rb");
    if (!r.file) {
        if (error && error_size)
            snprintf(error, error_size, "%s: %s", path, strerror(errno));
        return 0;
    }
    char magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0, alignment = 32;
    size_t metadata_capacity = 0;
    orig_shape *shapes = NULL;
    size_t shape_count = 0, shape_capacity = 0;
    if (!take(&r, magic, 4) || memcmp(magic, "GGUF", 4)) {
        fail(&r, "not a GGUF file");
        goto failed;
    }
    if (!u32(&r, &version) || (version != 2 && version != 3)) {
        fail(&r, "unsupported GGUF version %u", version);
        goto failed;
    }
    if (!u64(&r, &tensor_count) || !u64(&r, &kv_count)) goto failed;
    if (tensor_count > 1u << 20 || kv_count > 1u << 20) {
        fail(&r, "implausible GGUF counts");
        goto failed;
    }
    header->path = strdup(path);
    header->file_size = r.size;
    if (!header->path) {
        fail(&r, "out of memory");
        goto failed;
    }
    for (uint64_t index = 0; index < kv_count; index++) {
        char *key = string(&r, 1u << 16);
        uint32_t type;
        if (!key) goto failed;
        if (!u32(&r, &type)) {
            free(key);
            goto failed;
        }
        if (type == GGUF_ARRAY && !strncmp(key, ORIG_SHAPE_PREFIX, strlen(ORIG_SHAPE_PREFIX))) {
            orig_shape shape;
            if (!shape_value(&r, &shape)) {
                free(key);
                goto failed;
            }
            if (shape.ndim && shape_count == shape_capacity) {
                size_t next = shape_capacity ? shape_capacity * 2 : 16;
                orig_shape *grown = realloc(shapes, next * sizeof(*grown));
                if (!grown) {
                    free(key);
                    fail(&r, "out of memory");
                    goto failed;
                }
                shapes = grown;
                shape_capacity = next;
            }
            if (shape.ndim) {
                shape.name = strdup(key + strlen(ORIG_SHAPE_PREFIX));
                if (shape.name) shapes[shape_count++] = shape;
            }
            free(key);
            continue;
        }
        char *text = NULL;
        uint64_t number = 0;
        if (!value(&r, type, &text, &number, 0)) {
            free(key);
            goto failed;
        }
        if (!strcmp(key, "general.alignment")) {
            if (!number || number > 1u << 16 || (number & (number - 1))) {
                free(key);
                free(text);
                fail(&r, "invalid general.alignment");
                goto failed;
            }
            alignment = number;
        }
        if (text && strlen(text) < 4096) {
            if (header->metadata_count == metadata_capacity) {
                size_t next = metadata_capacity ? metadata_capacity * 2 : 16;
                h3_st_metadata_entry *grown = realloc(
                    header->metadata, next * sizeof(*grown));
                if (!grown) {
                    free(key);
                    free(text);
                    fail(&r, "out of memory");
                    goto failed;
                }
                header->metadata = grown;
                metadata_capacity = next;
            }
            header->metadata[header->metadata_count].key = key;
            header->metadata[header->metadata_count].value = text;
            header->metadata_count++;
        } else {
            free(key);
            free(text);
        }
    }
    header->tensors = calloc(tensor_count ? (size_t)tensor_count : 1,
                             sizeof(*header->tensors));
    if (!header->tensors) {
        fail(&r, "out of memory");
        goto failed;
    }
    for (uint64_t index = 0; index < tensor_count; index++) {
        h3_st_tensor *tensor = &header->tensors[index];
        uint32_t dims, ggml_type;
        uint64_t extents[8], offset;
        tensor->name = string(&r, 1024);
        header->tensor_count = (size_t)index + 1;
        if (!tensor->name || !u32(&r, &dims)) goto failed;
        if (dims > 8) {
            fail(&r, "tensor %s has %u dimensions", tensor->name, dims);
            goto failed;
        }
        uint64_t elements = 1;
        for (uint32_t d = 0; d < dims; d++) {
            if (!u64(&r, &extents[d])) goto failed;
            if (extents[d] && elements > UINT64_MAX / extents[d]) {
                fail(&r, "tensor %s is too large", tensor->name);
                goto failed;
            }
            elements *= extents[d];
        }
        if (!u32(&r, &ggml_type) || !u64(&r, &offset)) goto failed;
        tensor->ndim = (int)dims;
        for (uint32_t d = 0; d < dims; d++) tensor->shape[d] = extents[dims - 1 - d];
        h3_quant_type quant = h3_quant_from_ggml(ggml_type);
        tensor->quant = (uint32_t)quant;
        tensor->dtype = quant == H3_QUANT_F32 ? H3_DTYPE_F32 :
                        quant == H3_QUANT_F16 ? H3_DTYPE_F16 :
                        quant == H3_QUANT_BF16 ? H3_DTYPE_BF16 : H3_DTYPE_UNKNOWN;
        /* An unsupported type keeps its entry (so the store can name it
         * when something asks for it) but no payload range. */
        uint64_t bytes = quant == H3_QUANT_NONE ? 0 : h3_quant_bytes(quant, elements);
        if (quant != H3_QUANT_NONE && !bytes) {
            fail(&r, "tensor %s does not fill whole %s blocks", tensor->name,
                 h3_quant_name(quant));
            goto failed;
        }
        tensor->data_begin = offset;
        tensor->data_end = offset + bytes;
    }
    uint64_t data_start = (r.position + alignment - 1) / alignment * alignment;
    header->header_size = data_start;
    for (size_t index = 0; index < header->tensor_count; index++) {
        h3_st_tensor *tensor = &header->tensors[index];
        if (tensor->data_end < tensor->data_begin ||
            data_start > r.size || tensor->data_end > r.size - data_start ||
            tensor->data_begin % alignment) {
            fail(&r, "tensor %s lies outside the file", tensor->name);
            goto failed;
        }
        tensor->file_offset = data_start + tensor->data_begin;
    }
    apply_orig_shapes(header, shapes, shape_count);
    for (size_t index = 0; index < shape_count; index++) free(shapes[index].name);
    free(shapes);
    fclose(r.file);
    return 1;

failed:
    for (size_t index = 0; index < shape_count; index++) free(shapes[index].name);
    free(shapes);
    if (r.file) fclose(r.file);
    h3_st_free_header(header);
    return 0;
}

/* --- inspecting and trimming (h3.h) ---------------------------------- */

#include "h3.h"

#include <unistd.h>

static int has_prefix(const char *text, const char *prefix) {
    return !strncmp(text, prefix, strlen(prefix));
}

/* The language-layer index of a text-encoder tensor name in any of the
 * spellings (llama.cpp, transformers, the released checkpoint), or -1. */
static int text_layer_index(const char *name) {
    static const char *prefixes[] = {
        "blk.", "model.layers.", "model.language_model.layers.",
        "language_model.model.layers.",
    };
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(*prefixes); i++) {
        if (has_prefix(name, prefixes[i])) {
            const char *at = name + strlen(prefixes[i]);
            char *end = NULL;
            long layer = strtol(at, &end, 10);
            if (end != at && *end == '.') return (int)layer;
        }
    }
    return -1;
}

static int is_output_head(const char *name) {
    return !strcmp(name, "output.weight") || !strcmp(name, "output_norm.weight") ||
           !strcmp(name, "lm_head.weight") || !strcmp(name, "model.norm.weight") ||
           !strcmp(name, "model.language_model.norm.weight");
}

int h3_gguf_inspect(const char *path, h3_gguf_info *info,
                    char *error, size_t error_size) {
    if (!info) return 0;
    memset(info, 0, sizeof(*info));
    h3_st_header header;
    if (!h3_gguf_read_header(path, &header, error, error_size)) return 0;
    info->file_bytes = header.file_size;
    info->tensors = header.tensor_count;
    const char *arch = h3_st_metadata(&header, "general.architecture");
    const char *name = h3_st_metadata(&header, "general.name");
    if (arch) snprintf(info->architecture, sizeof(info->architecture), "%s", arch);
    if (name) snprintf(info->name, sizeof(info->name), "%s", name);
    uint64_t by_type[16] = {0};
    int dit = 0, text = 0, vision = 0, max_layer = -1;
    for (size_t index = 0; index < header.tensor_count; index++) {
        const h3_st_tensor *tensor = &header.tensors[index];
        uint64_t bytes = tensor->data_end - tensor->data_begin;
        info->tensor_bytes += bytes;
        if (tensor->quant < 16 && bytes >= (1u << 20)) by_type[tensor->quant] += bytes;
        const char *n = tensor->name;
        if (!strcmp(n, "video_patch_proj.weight")) dit = 1;
        if (!strcmp(n, "adaln_t_table")) info->curve_adaln = 1;
        if (!strcmp(n, "blocks.0.adaln_proj.linear.weight") &&
            tensor->ndim == 2 && tensor->shape[1] == 2688)
            info->dense_adaln = 1;
        if (has_prefix(n, "v.blk.") || has_prefix(n, "visual.") ||
            has_prefix(n, "model.visual.") || has_prefix(n, "vision_tower."))
            vision = 1;
        int layer = text_layer_index(n);
        if (layer >= 0) {
            text = 1;
            if (layer > max_layer) max_layer = layer;
        }
    }
    if (dit) info->role = H3_GGUF_DIT;
    else if (text) info->role = H3_GGUF_TEXT_ENCODER;
    else if (vision) info->role = H3_GGUF_VISION;
    info->text_layers = text ? max_layer + 1 : 0;
    info->has_vision = text && vision;
    uint64_t best = 0;
    for (int type = 0; type < 16; type++) {
        if (by_type[type] > best) {
            best = by_type[type];
            snprintf(info->main_type, sizeof(info->main_type), "%s",
                     h3_quant_name((h3_quant_type)type));
        }
    }
    h3_st_free_header(&header);
    return 1;
}

static int write_bytes(FILE *out, const void *data, size_t size) {
    return fwrite(data, 1, size, out) == size;
}

static int write_u32_le(FILE *out, uint32_t value) {
    uint8_t b[4] = {(uint8_t)value, (uint8_t)(value >> 8), (uint8_t)(value >> 16),
                    (uint8_t)(value >> 24)};
    return write_bytes(out, b, 4);
}

static int write_u64_le(FILE *out, uint64_t value) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(value >> (8 * i));
    return write_bytes(out, b, 8);
}

static int write_string(FILE *out, const char *text) {
    size_t length = strlen(text);
    return write_u64_le(out, length) && write_bytes(out, text, length);
}

/* ggml type id for an h3_quant_type (the inverse of h3_quant_from_ggml). */
static uint32_t ggml_type_of(h3_quant_type type) {
    for (uint32_t id = 0; id < 40; id++)
        if (h3_quant_from_ggml(id) == type) return id;
    return UINT32_MAX;
}

int h3_gguf_trim_text_encoder(const char *source_path, const char *output_path,
                              int layers, h3_progress_callback on_progress,
                              void *callback_opaque,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!source_path || !output_path || layers < 1) {
        if (error && error_size) snprintf(error, error_size, "invalid trim arguments");
        return 0;
    }
    h3_st_header header;
    if (!h3_gguf_read_header(source_path, &header, error, error_size)) return 0;
    const uint64_t alignment = 32;
    size_t kept = 0;
    uint64_t total = 0;
    int *keep = calloc(header.tensor_count ? header.tensor_count : 1, sizeof(*keep));
    for (size_t i = 0; keep && i < header.tensor_count; i++) {
        const h3_st_tensor *t = &header.tensors[i];
        int layer = text_layer_index(t->name);
        keep[i] = !is_output_head(t->name) && layer < layers &&
                  t->quant != 0 && ggml_type_of((h3_quant_type)t->quant) != UINT32_MAX;
        if (keep[i]) {
            kept++;
            total += t->data_end - t->data_begin;
        }
    }
    size_t partial_length = strlen(output_path) + sizeof(".partial");
    char *partial = malloc(partial_length);
    enum { BUFFER = 64 << 20 };
    uint8_t *buffer = malloc(BUFFER);
    FILE *in = fopen(source_path, "rb");
    FILE *out = NULL;
    int ok = keep && partial && buffer && in;
    if (!ok && error && error_size)
        snprintf(error, error_size, "cannot open %s", source_path);
    if (ok) {
        snprintf(partial, partial_length, "%s.partial", output_path);
        out = fopen(partial, "wb");
        ok = out != NULL;
        if (!ok && error && error_size)
            snprintf(error, error_size, "cannot create %s: %s", partial, strerror(errno));
    }
    /* Header: string metadata, then tensor infos (dims fastest-first). */
    if (ok) {
        uint64_t kv = header.metadata_count + 1;
        ok = write_bytes(out, "GGUF", 4) && write_u32_le(out, 3) &&
             write_u64_le(out, kept) && write_u64_le(out, kv);
        for (size_t i = 0; ok && i < header.metadata_count; i++)
            ok = write_string(out, header.metadata[i].key) && write_u32_le(out, 8) &&
                 write_string(out, header.metadata[i].value);
        char note[64];
        snprintf(note, sizeof(note), "%d", layers);
        ok = ok && write_string(out, "h3c.text_layers") && write_u32_le(out, 8) &&
             write_string(out, note);
        uint64_t offset = 0;
        for (size_t i = 0; ok && i < header.tensor_count; i++) {
            if (!keep[i]) continue;
            const h3_st_tensor *t = &header.tensors[i];
            ok = write_string(out, t->name) && write_u32_le(out, (uint32_t)t->ndim);
            for (int d = t->ndim - 1; ok && d >= 0; d--) ok = write_u64_le(out, t->shape[d]);
            ok = ok && write_u32_le(out, ggml_type_of((h3_quant_type)t->quant)) &&
                 write_u64_le(out, offset);
            offset += (t->data_end - t->data_begin + alignment - 1) / alignment * alignment;
        }
        long position = ok ? ftell(out) : 0;
        for (long pad = position; ok && pad % (long)alignment; pad++) ok = fputc(0, out) != EOF;
        if (!ok && error && error_size && !error[0])
            snprintf(error, error_size, "cannot write %s: %s", partial, strerror(errno));
    }
    /* Payloads, each padded to the alignment. */
    uint64_t copied = 0;
    for (size_t i = 0; ok && i < header.tensor_count; i++) {
        if (!keep[i]) continue;
        const h3_st_tensor *t = &header.tensors[i];
        uint64_t bytes = t->data_end - t->data_begin;
        ok = fseeko(in, (off_t)t->file_offset, SEEK_SET) == 0;
        for (uint64_t done = 0; ok && done < bytes;) {
            size_t chunk = bytes - done < BUFFER ? (size_t)(bytes - done) : BUFFER;
            ok = fread(buffer, 1, chunk, in) == chunk && write_bytes(out, buffer, chunk);
            done += chunk;
        }
        for (uint64_t pad = bytes; ok && pad % alignment; pad++) ok = fputc(0, out) != EOF;
        copied += bytes;
        if (!ok && error && error_size)
            snprintf(error, error_size, "cannot copy %s: %s", t->name, strerror(errno));
        if (ok && on_progress &&
            on_progress("trim text encoder", (int)(copied >> 20), (int)(total >> 20),
                        callback_opaque)) {
            if (error && error_size) snprintf(error, error_size, "trimming cancelled");
            ok = 0;
        }
    }
    if (out && fclose(out) != 0 && ok) {
        ok = 0;
        if (error && error_size) snprintf(error, error_size, "cannot finish %s", partial);
    }
    if (in) fclose(in);
    if (ok && rename(partial, output_path) != 0) {
        ok = 0;
        if (error && error_size)
            snprintf(error, error_size, "cannot rename %s: %s", partial, strerror(errno));
    }
    if (!ok && partial) unlink(partial);
    free(partial);
    free(buffer);
    free(keep);
    h3_st_free_header(&header);
    return ok;
}
