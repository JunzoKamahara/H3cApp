/* Helper for the video text projection (H3_TEXT_PROJECTION): the engine's
 * text rows of one prompt from a small Qwen3-VL and a ClipProj projection,
 * written for comparison with tools/perf/ref_text_projection.py. Not in
 * `make test`.
 *
 *   h3_text_projection_embed <tokenizer.json> <Qwen3-VL folder>
 *                            <projection.safetensors> <prompt>
 *                            <out.safetensors> [shaders.metal]
 *
 * Writes "embedding" BF16 [tokens, 5120] and "ids" I32 [tokens]. */
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed % 10 == 0)
        fprintf(stderr, "text encoder %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <tokenizer.json> <Qwen3-VL folder> "
                        "<projection.safetensors> <prompt> <out.safetensors> "
                        "[shaders.metal]\n", argv[0]);
        return 2;
    }
    const char *shaders = argc > 6 ? argv[6] : "h3_shaders.metal";
    char error[512] = "";
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[1], error, sizeof(error));
    uint32_t *ids = NULL;
    size_t count = 0;
    if (!tokenizer || !h3_tokenizer_encode(tokenizer, argv[4], 1, &ids, &count,
                                           error, sizeof(error)) || !count) {
        fprintf(stderr, "tokenizer: %s\n", error[0] ? error : "empty prompt");
        return 1;
    }
    printf("%zu tokens:", count);
    for (size_t index = 0; index < count; index++) printf(" %u", ids[index]);
    printf("\n");

    h3_text_embedding text = {0};
    if (!h3_text_encode_projected_bf16(argv[2], argv[3], shaders, ids, count,
                                       progress, NULL, &text, error,
                                       sizeof(error))) {
        fprintf(stderr, "text encoder: %s\n", error);
        return 1;
    }
    if (text.tokens != count || text.width != H3_TEXT_HIDDEN_SIZE) {
        fprintf(stderr, "unexpected embedding shape [%zu,%zu]\n", text.tokens,
                text.width);
        return 1;
    }
    size_t embedding_bytes = count * H3_TEXT_HIDDEN_SIZE * sizeof(uint16_t);
    size_t id_bytes = count * sizeof(int32_t);
    char header[512];
    int length = snprintf(header, sizeof(header),
        "{\"embedding\":{\"dtype\":\"BF16\",\"shape\":[%zu,%u],"
        "\"data_offsets\":[0,%zu]},"
        "\"ids\":{\"dtype\":\"I32\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]}}",
        count, H3_TEXT_HIDDEN_SIZE, embedding_bytes,
        count, embedding_bytes, embedding_bytes + id_bytes);
    while (length % 8) header[length++] = ' ';
    int32_t *id_values = malloc(id_bytes);
    FILE *out = fopen(argv[5], "wb");
    if (!id_values || !out) {
        fprintf(stderr, "cannot create %s\n", argv[5]);
        return 1;
    }
    for (size_t index = 0; index < count; index++)
        id_values[index] = (int32_t)ids[index];
    uint64_t header_length = (uint64_t)length; /* little-endian host */
    int ok = fwrite(&header_length, 8, 1, out) == 1 &&
             fwrite(header, 1, (size_t)length, out) == (size_t)length &&
             fwrite(text.values, 1, embedding_bytes, out) == embedding_bytes &&
             fwrite(id_values, 1, id_bytes, out) == id_bytes;
    ok = fclose(out) == 0 && ok;
    free(id_values);
    h3_text_embedding_free(&text);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    if (!ok) {
        fprintf(stderr, "cannot write %s\n", argv[5]);
        return 1;
    }
    printf("wrote %s\n", argv[5]);
    return 0;
}
