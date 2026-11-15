/* Phase-0 helper for plans/IMAGE_PLAN.ja.md: the engine's text-only path
 * (tokenizer, then the first 50 Qwen3-VL layers) on one prompt, written in
 * the layout h3-image-mlx's DiT stage reads - so the reference
 * implementation can be driven by this engine's own text embedding, and
 * the two text paths compared. Not in `make test`.
 *
 *   h3_image_text_embed <model dir with tokenizer/ and text_encoder/>
 *                       <prompt> <out.safetensors> [shaders.metal]
 *                       [reference image ...]
 *
 * Writes "embedding" BF16 [tokens, 5120], "tags" I32 [tokens] (all 1:
 * language rows) and "ids" I32 [tokens].
 *
 * With reference images (phase 5, editing) the sequence is the edit
 * presentation - `Picture n: ` and the n-th picture's vision tokens for
 * each, then the prompt - and "tags" has 0 on the vision spans; "ids" is
 * left out. */
#include "h3_av_reader.h"
#include "h3_image.h"
#include "h3_multimodal.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_TOKENS = 4096, TEXT_DIM = 5120 };

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed % 10 == 0)
        fprintf(stderr, "text encoder %d/%d\n", completed, total);
}

static int write_file(const char *path, const char *header, int length,
                      const void *first, size_t first_bytes,
                      const void *second, size_t second_bytes) {
    char padded[1024];
    memcpy(padded, header, (size_t)length);
    while (length % 8) padded[length++] = ' ';
    uint64_t header_length = (uint64_t)length; /* little-endian host */
    FILE *out = fopen(path, "wb");
    int ok = out && fwrite(&header_length, 8, 1, out) == 1 &&
             fwrite(padded, 1, (size_t)length, out) == (size_t)length &&
             fwrite(first, 1, first_bytes, out) == first_bytes &&
             fwrite(second, 1, second_bytes, out) == second_bytes;
    if (out) ok = fclose(out) == 0 && ok;
    return ok;
}

/* The edit presentation of the prompt and the reference images. */
static int edit(const char *tokenizer_path, const char *text_path,
                const char *shaders, const char *prompt, char **references,
                int count, const char *out) {
    char error[512] = "";
    h3_vision_output vision[H3_IMAGE_MAX_REFERENCES];
    memset(vision, 0, sizeof(vision));
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error, sizeof(error));
    if (!tokenizer || count > H3_IMAGE_MAX_REFERENCES) {
        fprintf(stderr, "tokenizer: %s\n", error[0] ? error : "too many references");
        return 1;
    }
    for (int index = 0; index < count; index++) {
        int width, height, encode_w, encode_h, vision_w, vision_h;
        float *pixels = NULL;
        if (!h3_av_visual_size(references[index], &width, &height, error, sizeof(error)) ||
            !h3_image_reference_canvas(width, height, 0, &encode_w, &encode_h,
                                       &vision_w, &vision_h) ||
            !h3_av_read_image_f32(references[index], vision_w, vision_h,
                                  H3_IMAGE_FIT_STRETCH, &pixels, error, sizeof(error)) ||
            !h3_vision_encode_bf16(text_path, shaders, pixels, 1, vision_h, vision_w,
                                   NULL, NULL, &vision[index], error, sizeof(error))) {
            fprintf(stderr, "reference %d: %s\n", index + 1,
                    error[0] ? error : "unusable size");
            return 1;
        }
        free(pixels);
        /* The vision tower's own output, to tell its error from the
         * language layers': <out>.vision<n>.safetensors. */
        char vision_path[2048], vision_header[512];
        size_t vision_bytes = vision[index].tokens * TEXT_DIM * sizeof(uint16_t);
        snprintf(vision_path, sizeof(vision_path), "%s.vision%d.safetensors",
                 out, index + 1);
        int vision_length = snprintf(vision_header, sizeof(vision_header),
            "{\"merged\":{\"dtype\":\"BF16\",\"shape\":[%zu,%d],\"data_offsets\":[0,%zu]},"
            "\"deepstack0\":{\"dtype\":\"BF16\",\"shape\":[%zu,%d],\"data_offsets\":[%zu,%zu]}}",
            vision[index].tokens, TEXT_DIM, vision_bytes,
            vision[index].tokens, TEXT_DIM, vision_bytes, 2 * vision_bytes);
        if (!write_file(vision_path, vision_header, vision_length,
                        vision[index].merged, vision_bytes,
                        vision[index].deepstack[0], vision_bytes)) {
            fprintf(stderr, "cannot write %s\n", vision_path);
            return 1;
        }
        printf("reference %d: %dx%d, seen at %dx%d (%zu vision tokens), "
               "encoded at %dx%d\n", index + 1, width, height, vision_w,
               vision_h, vision[index].tokens, encode_w, encode_h);
    }
    h3_text_embedding text = {0};
    size_t prompt_tokens = 0;
    if (!h3_multimodal_encode_image_edit_bf16(tokenizer, text_path, shaders, prompt,
            vision, (size_t)count, MAX_TOKENS, &prompt_tokens, progress, NULL,
            &text, error, sizeof(error))) {
        fprintf(stderr, "text encoder: %s\n", error);
        return 1;
    }
    printf("%zu rows, %zu of them the prompt\n", text.tokens, prompt_tokens);
    size_t embedding_bytes = text.tokens * TEXT_DIM * sizeof(uint16_t);
    size_t int_bytes = text.tokens * sizeof(int32_t);
    int32_t *tags = malloc(int_bytes);
    if (!tags || !text.tags || text.width != TEXT_DIM) {
        fprintf(stderr, "unexpected embedding\n");
        return 1;
    }
    for (size_t row = 0; row < text.tokens; row++) tags[row] = text.tags[row];
    char header[512];
    int length = snprintf(header, sizeof(header),
        "{\"embedding\":{\"dtype\":\"BF16\",\"shape\":[%zu,%d],\"data_offsets\":[0,%zu]},"
        "\"tags\":{\"dtype\":\"I32\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]}}",
        text.tokens, TEXT_DIM, embedding_bytes,
        text.tokens, embedding_bytes, embedding_bytes + int_bytes);
    if (!write_file(out, header, length, text.values, embedding_bytes, tags, int_bytes)) {
        fprintf(stderr, "cannot write %s\n", out);
        return 1;
    }
    printf("wrote %s\n", out);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model dir> <prompt> <out.safetensors> "
                        "[shaders.metal] [reference image ...]\n", argv[0]);
        return 2;
    }
    const char *shaders = argc > 4 ? argv[4] : "h3_shaders.metal";
    char tokenizer_path[2048], text_path[2048], error[512] = "";
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer/tokenizer.json",
             argv[1]);
    snprintf(text_path, sizeof(text_path), "%s/text_encoder", argv[1]);
    if (argc > 5)
        return edit(tokenizer_path, text_path, shaders, argv[2], argv + 5,
                    argc - 5, argv[3]);
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
