/* Compares a quantized re-publication of a checkpoint against the released
 * weights, tensor by tensor, through the engine's own weight store (so the
 * name aliases and the GPU expansion are what gets tested).
 *
 * Usage (from the repository root):
 *   h3_weight_compare <released dir> <candidate dir> [name substring]
 * Prints, per tensor present in both, the relative L2 and max errors of the
 * candidate (expanded to F32) against the released values, and a summary
 * per tensor group. Exits 1 when a shape differs or an error is large
 * (relative L2 > 0.1 - a layout mistake, not quantization). */
#include "../h3_gpu.h"
#include "../h3_quant.h"
#include "../h3_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *load(const h3_weight_store *store, h3_gpu *gpu, const char *name,
                   const h3_st_tensor *tensor, size_t *count) {
    uint64_t shape[8];
    size_t elements = 1;
    for (int d = 0; d < tensor->ndim; d++) {
        shape[d] = tensor->shape[d];
        elements *= (size_t)shape[d];
    }
    char error[512];
    h3_gpu_tensor *f32 = h3_weight_load_f32(store, gpu, name, tensor->ndim,
                                            shape, error, sizeof(error));
    float *values = malloc(elements * sizeof(float));
    if (f32) {
        int ok = values && h3_gpu_tensor_read_f32(f32, values, elements);
        h3_gpu_tensor_free(f32);
        if (!ok) { free(values); return NULL; }
    } else {
        h3_gpu_tensor *bf16 = h3_weight_load_bf16(store, gpu, name, tensor->ndim,
                                                  shape, error, sizeof(error));
        uint16_t *bits = malloc(elements * sizeof(uint16_t));
        int ok = bf16 && bits && values &&
                 h3_gpu_tensor_read_bf16(bf16, bits, elements);
        for (size_t i = 0; ok && i < elements; i++) {
            uint32_t w = (uint32_t)bits[i] << 16;
            memcpy(&values[i], &w, sizeof(w));
        }
        h3_gpu_tensor_free(bf16);
        free(bits);
        if (!ok) {
            fprintf(stderr, "cannot load %s: %s\n", name, error);
            free(values);
            return NULL;
        }
    }
    *count = elements;
    return values;
}

static void group_of(const char *name, char *out, size_t size) {
    size_t j = 0;
    for (const char *p = name; *p && j + 1 < size; p++) {
        if (*p >= '0' && *p <= '9' && p > name && p[-1] == '.') {
            while (p[1] >= '0' && p[1] <= '9') p++;
            out[j++] = 'N';
        } else {
            out[j++] = *p;
        }
    }
    out[j] = '\0';
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <released dir> <candidate dir> [name substring]\n", argv[0]);
        return 2;
    }
    char error[512];
    h3_weight_store *reference = h3_weight_store_open(argv[1], error, sizeof(error));
    h3_weight_store *candidate = reference ?
        h3_weight_store_open(argv[2], error, sizeof(error)) : NULL;
    h3_gpu *gpu = candidate ? h3_gpu_create("h3_shaders.metal", error, sizeof(error)) : NULL;
    if (!gpu) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    enum { GROUPS = 256 };
    struct { char name[160]; int count; double worst_l2, worst_max; char type[16]; } groups[GROUPS];
    int group_count = 0, failures = 0, compared = 0, missing = 0;
    size_t total = h3_weight_store_tensor_count(reference);
    for (size_t index = 0; index < total; index++) {
        const h3_st_tensor *ref = h3_weight_store_tensor(reference, index, NULL);
        if (argc > 3 && !strstr(ref->name, argv[3])) continue;
        const h3_st_tensor *cand = h3_weight_find(candidate, ref->name, NULL);
        /* A tensor the store assembles from parts (the split patch
         * embedding) has no single entry: compare it if it loads. */
        int assembled = !cand && !strcmp(ref->name, "model.visual.patch_embed.proj.weight");
        if (!cand && !assembled) {
            missing++;
            continue;
        }
        int same_shape = assembled || cand->ndim == ref->ndim;
        for (int d = 0; !assembled && same_shape && d < ref->ndim; d++)
            same_shape = cand->shape[d] == ref->shape[d];
        if (!same_shape) {
            printf("SHAPE %s\n", ref->name);
            failures++;
            continue;
        }
        size_t n1 = 0, n2 = 0;
        float *a = load(reference, gpu, ref->name, ref, &n1);
        float *b = a ? load(candidate, gpu, ref->name, ref, &n2) : NULL;
        if (!a || !b || n1 != n2) {
            free(a);
            free(b);
            failures++;
            continue;
        }
        double err = 0, norm = 0, peak = 0, worst = 0;
        for (size_t i = 0; i < n1; i++) {
            double d = (double)b[i] - a[i];
            err += d * d;
            norm += (double)a[i] * a[i];
            if (fabs(a[i]) > peak) peak = fabs(a[i]);
            if (fabs(d) > worst) worst = fabs(d);
        }
        double l2 = norm > 0 ? sqrt(err / norm) : sqrt(err);
        double mx = peak > 0 ? worst / peak : worst;
        free(a);
        free(b);
        compared++;
        if (l2 > 0.1) {
            printf("LARGE %s rel-L2 %.3e\n", ref->name, l2);
            failures++;
        }
        char group[160];
        group_of(ref->name, group, sizeof(group));
        int g = 0;
        while (g < group_count && strcmp(groups[g].name, group)) g++;
        if (g == group_count && group_count < GROUPS) {
            snprintf(groups[g].name, sizeof(groups[g].name), "%s", group);
            groups[g].count = 0;
            groups[g].worst_l2 = groups[g].worst_max = 0;
            snprintf(groups[g].type, sizeof(groups[g].type), "%s",
                     !cand ? "parts" : cand->quant ?
                     h3_quant_name((h3_quant_type)cand->quant) : "safetensors");
            group_count++;
        }
        if (g < group_count) {
            groups[g].count++;
            if (l2 > groups[g].worst_l2) groups[g].worst_l2 = l2;
            if (mx > groups[g].worst_max) groups[g].worst_max = mx;
        }
    }
    for (int g = 0; g < group_count; g++)
        printf("%-60s %-6s x%-3d worst rel-L2 %.3e  worst max %.3e\n", groups[g].name,
               groups[g].type, groups[g].count, groups[g].worst_l2, groups[g].worst_max);
    printf("compared %d tensors, %d not in the candidate, %d failures\n", compared,
           missing, failures);
    h3_gpu_free(gpu);
    h3_weight_store_free(candidate);
    h3_weight_store_free(reference);
    return failures ? 1 : 0;
}
