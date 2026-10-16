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

/* Skips one value; strings and u32/u64 scalars are handed back when the
 * caller wants them (text/number may be NULL). */
static int value(reader *r, uint32_t type, char **text, uint64_t *number) {
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
            if (!value(r, element, NULL, NULL)) return 0;
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
        char *text = NULL;
        uint64_t number = 0;
        if (!value(&r, type, &text, &number)) {
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
    fclose(r.file);
    return 1;

failed:
    if (r.file) fclose(r.file);
    h3_st_free_header(header);
    return 0;
}
