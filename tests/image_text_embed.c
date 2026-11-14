/* Phase-0 helper for plans/IMAGE_PLAN.ja.md: the engine's text-only path
 * (tokenizer, then the first 50 Qwen3-VL layers) on one prompt, written in
 * the layout h3-image-mlx's DiT stage reads - so the reference
 * implementation can be driven by this engine's own text embedding, and
 * the two text paths compared. Not in `make test`.
 *
 *   h3_image_text_embed <model dir with tokenizer/ and text_encoder/>
 *                       <prompt> <out.safetensors> [shaders.metal]
 *
 * Writes "embedding" BF16 [tokens, 5120], "tags" I32 [tokens] (all 1:
 * language rows) and "ids" I32 [tokens]. */
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_TOKENS = 4096, TEXT_DIM = 5120 };

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed % 10 == 0)
        fprintf(stderr, "text encoder %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model dir> <prompt> <out.safetensors> "
                        "[shaders.metal]\n", argv[0]);
        return 2;
    }
    const char *shaders = argc > 4 ? argv[4] : "h3_shaders.metal";
    char tokenizer_path[2048], text_path[2048], error[512] = "";
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer/tokenizer.json",
             argv[1]);
    snprintf(text_path, sizeof(text_path), "%s/text_encoder", argv[1]);
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    uint32_t *ids = NULL;
    size_t count = 0;
    if (!tokenizer || !h3_tokenizer_encode(tokenizer, argv[2], 0, &ids, &count,
                                           error, sizeof(error)) || !count) {
        fprintf(stderr, "tokenizer: %s\n", error[0] ? error : "empty prompt");
        return 1;
    }
    if (count > MAX_TOKENS) count = MAX_TOKENS;
    printf("%zu tokens:", count);
    for (size_t i = 0; i < count; i++) printf(" %u", ids[i]);
    printf("\n");

    h3_text_embedding text = {0};
    if (!h3_text_encode_bf16(text_path, shaders, ids, count, progress, NULL,
                             &text, error, sizeof(error))) {
        fprintf(stderr, "text encoder: %s\n", error);
        return 1;
    }
    if (text.tokens != count || text.width != TEXT_DIM) {
        fprintf(stderr, "unexpected embedding shape [%zu,%zu]\n", text.tokens,
                text.width);
        return 1;
    }
    size_t embedding_bytes = count * TEXT_DIM * sizeof(uint16_t);
    size_t int_bytes = count * sizeof(int32_t);
    char header[512];
    int length = snprintf(header, sizeof(header),
        "{\"embedding\":{\"dtype\":\"BF16\",\"shape\":[%zu,%d],\"data_offsets\":[0,%zu]},"
        "\"tags\":{\"dtype\":\"I32\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]},"
        "\"ids\":{\"dtype\":\"I32\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]}}",
        count, TEXT_DIM, embedding_bytes,
        count, embedding_bytes, embedding_bytes + int_bytes,
        count, embedding_bytes + int_bytes, embedding_bytes + 2 * int_bytes);
    while (length % 8) header[length++] = ' ';
    int32_t *tags = malloc(int_bytes), *id_values = malloc(int_bytes);
    FILE *out = fopen(argv[3], "wb");
    if (!tags || !id_values || !out) {
        fprintf(stderr, "cannot create %s\n", argv[3]);
        return 1;
    }
    for (size_t i = 0; i < count; i++) {
        tags[i] = 1;
        id_values[i] = (int32_t)ids[i];
    }
    uint64_t header_length = (uint64_t)length; /* little-endian host */
    int ok = fwrite(&header_length, 8, 1, out) == 1 &&
             fwrite(header, 1, (size_t)length, out) == (size_t)length &&
             fwrite(text.values, 1, embedding_bytes, out) == embedding_bytes &&
             fwrite(tags, 1, int_bytes, out) == int_bytes &&
             fwrite(id_values, 1, int_bytes, out) == int_bytes;
    ok = fclose(out) == 0 && ok;
    free(tags);
    free(id_values);
    h3_text_embedding_free(&text);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    if (!ok) {
        fprintf(stderr, "cannot write %s\n", argv[3]);
        return 1;
    }
    printf("wrote %s\n", argv[3]);
    return 0;
}
