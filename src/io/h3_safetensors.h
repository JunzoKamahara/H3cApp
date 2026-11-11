#ifndef H3_SAFETENSORS_H
#define H3_SAFETENSORS_H

#include "h3.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_DTYPE_UNKNOWN = 0,
    H3_DTYPE_BOOL,
    H3_DTYPE_I8,
    H3_DTYPE_U8,
    H3_DTYPE_I16,
    H3_DTYPE_U16,
    H3_DTYPE_F16,
    H3_DTYPE_BF16,
    H3_DTYPE_I32,
    H3_DTYPE_U32,
    H3_DTYPE_F32,
    H3_DTYPE_I64,
    H3_DTYPE_U64,
    H3_DTYPE_F64
} h3_dtype;

typedef struct {
    char *name;
    h3_dtype dtype;
    int ndim;
    uint64_t shape[8];
    uint64_t data_begin;
    uint64_t data_end;
    uint64_t file_offset;
    /* h3_quant_type (h3_quant.h) for tensors read from a GGUF file: the
     * block format of the payload. 0 (H3_QUANT_NONE) for safetensors, whose
     * dtype says it all; GGUF F32/F16/BF16 tensors also set dtype, so code
     * that only reads those types needs no change. */
    uint32_t quant;
} h3_st_tensor;

typedef struct {
    char *key;
    char *value;
} h3_st_metadata_entry;

typedef struct {
    char *path;
    uint64_t file_size;
    uint64_t header_size;
    h3_st_tensor *tensors;
    size_t tensor_count;
    /* String-valued entries of the optional top-level "__metadata__"
     * object (e.g. a LoRA adapter's "alpha"). Non-string metadata values
     * are skipped, not stored. */
    h3_st_metadata_entry *metadata;
    size_t metadata_count;
} h3_st_header;

int h3_st_read_header(const char *path, h3_st_header *header,
                      char *error, size_t error_size);
/* A GGUF file's tensor table in the same form (h3_gguf.c): shapes in
 * row-major order (GGUF lists them fastest-first), file_offset absolute,
 * data_begin/data_end relative to the aligned data section, quant set,
 * and the file's string key/values as metadata. */
int h3_gguf_read_header(const char *path, h3_st_header *header,
                        char *error, size_t error_size);
void h3_st_free_header(h3_st_header *header);
const h3_st_tensor *h3_st_find(const h3_st_header *header, const char *name);
/* Looks up a string-valued "__metadata__" key (e.g. "alpha"). Returns NULL
 * if the header has no metadata object or the key is absent/non-string. */
const char *h3_st_metadata(const h3_st_header *header, const char *key);
uint64_t h3_st_tensor_elements(const h3_st_tensor *tensor);
int h3_st_read_data(const h3_st_header *header, const h3_st_tensor *tensor,
                    void *data, size_t bytes, char *error, size_t error_size);
size_t h3_dtype_size(h3_dtype dtype);
const char *h3_dtype_name(h3_dtype dtype);

int h3_st_inventory_dir(const char *directory, h3_component_info *info,
                        char *error, size_t error_size);

#endif
