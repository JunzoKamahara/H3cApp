/* h3_lora_extract on a small synthetic merge: ranks are cut from lora_A
 * rows and lora_B columns, other tensors are copied, the values match. */
#include "h3_lora_extract.h"
#include "h3_safetensors.h"
#include "h3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int tests_run;
#define CHECK(c) do { tests_run++; if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

static void write_f32_tensor(FILE *f, int rows, int cols, float seed) {
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) { float v = seed + (float)r * 0.5f + (float)c * 0.001f; fwrite(&v, 4, 1, f); }
}

int main(void) {
    const char *in = "/tmp/h3_lora_extract_in.safetensors", *out = "/tmp/h3_lora_extract_out.safetensors";
    /* three tensors: A [12, 7], B [5, 12], a scalar table [3] */
    const char *header = "{\"__metadata__\":{\"format\":\"lora\"},"
        "\"transformer.transformer_blocks.0.attn.to_q.lora_A.weight\":{\"dtype\":\"F32\",\"shape\":[12,7],\"data_offsets\":[0,336]},"
        "\"transformer.transformer_blocks.0.attn.to_q.lora_B.weight\":{\"dtype\":\"F32\",\"shape\":[5,12],\"data_offsets\":[336,576]},"
        "\"table\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[576,588]}}";
    FILE *f = fopen(in, "wb");
    CHECK(f);
    uint64_t hs = strlen(header);
    fwrite(&hs, 8, 1, f); fwrite(header, 1, hs, f);
    write_f32_tensor(f, 12, 7, 1.0f);
    write_f32_tensor(f, 5, 12, 100.0f);
    write_f32_tensor(f, 1, 3, 1000.0f);
    fclose(f);
    char error[256];
    CHECK(h3_lora_extract_ranks(in, out, 3, 8, 1, "test", NULL, NULL, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    h3_st_header h;
    CHECK(h3_st_read_header(out, &h, error, sizeof(error)));
    CHECK(h.tensor_count == 3);
    const h3_st_tensor *a = h3_st_find(&h, "transformer.transformer_blocks.0.attn.to_q.lora_A.weight");
    const h3_st_tensor *b = h3_st_find(&h, "transformer.transformer_blocks.0.attn.to_q.lora_B.weight");
    const h3_st_tensor *t = h3_st_find(&h, "table");
    CHECK(a && a->dtype == H3_DTYPE_BF16 && a->ndim == 2 && a->shape[0] == 5 && a->shape[1] == 7);
    CHECK(b && b->dtype == H3_DTYPE_BF16 && b->shape[0] == 5 && b->shape[1] == 5);
    CHECK(t && t->dtype == H3_DTYPE_BF16 && t->ndim == 1 && t->shape[0] == 3);
    CHECK(h3_st_metadata(&h, "h3c_extract") && strstr(h3_st_metadata(&h, "h3c_extract"), "ranks 3:8"));
    uint16_t av[35], bv[25], tv[3];
    CHECK(h3_st_read_data(&h, a, av, sizeof(av), error, sizeof(error)));
    CHECK(h3_st_read_data(&h, b, bv, sizeof(bv), error, sizeof(error)));
    CHECK(h3_st_read_data(&h, t, tv, sizeof(tv), error, sizeof(error)));
    for (int r = 0; r < 5; r++) for (int c = 0; c < 7; c++) {
        uint32_t bits = (uint32_t)av[r * 7 + c] << 16; float v; memcpy(&v, &bits, 4);
        float want = 1.0f + (float)(r + 3) * 0.5f + (float)c * 0.001f;
        CHECK(fabsf(v - want) <= fabsf(want) * 0.004f + 1e-3f);
    }
    for (int r = 0; r < 5; r++) for (int c = 0; c < 5; c++) {
        uint32_t bits = (uint32_t)bv[r * 5 + c] << 16; float v; memcpy(&v, &bits, 4);
        float want = 100.0f + (float)r * 0.5f + (float)(c + 3) * 0.001f;
        CHECK(fabsf(v - want) <= fabsf(want) * 0.004f);
    }
    uint32_t bits = (uint32_t)tv[2] << 16; float v; memcpy(&v, &bits, 4);
    CHECK(fabsf(v - 1000.002f) <= 4.0f);
    h3_st_free_header(&h);
    /* The loader sees the cut file as a diffusers-layout adapter of rank 5. */
    h3_lora_info info;
    CHECK(h3_lora_inspect(out, &info, error, sizeof(error)) || (fprintf(stderr, "%s\n", error), 0));
    CHECK(!strcmp(info.format, "diffusers") && info.rank_min == 5 && info.rank_max == 5 && info.blocks == 1);
    unlink(in); unlink(out);
    printf("ok: %d checks (lora rank extraction)\n", tests_run);
    return 0;
}
