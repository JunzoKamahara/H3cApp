/* Re-noise step rule: the GPU kernel must match the CPU reference, and the
 * hashed noise must look like independent standard normals per step. */
#include "h3_gpu.h"
#include "h3_host.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int condition, const char *what) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

/* Values that are exact in BF16, so the CPU and the GPU see the same velocity. */
static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void noise_statistics(void) {
    enum { COUNT = 1 << 20 };
    uint32_t a[2], b[2];
    h3_renoise_keys(42, 0, 0, a);
    h3_renoise_keys(42, 1, 0, b);
    double sum = 0, square = 0, cross = 0, fourth = 0;
    for (uint32_t index = 0; index < COUNT; index++) {
        double x = h3_renoise_normal(index, a);
        double y = h3_renoise_normal(index, b);
        sum += x;
        square += x * x;
        fourth += x * x * x * x;
        cross += x * y;
    }
    const double count = COUNT;
    double mean = sum / count, variance = square / count - mean * mean;
    double kurtosis = fourth / count / (variance * variance);
    double correlation = cross / count;
    printf("noise: mean %.5f variance %.5f kurtosis %.4f step-correlation %.5f\n",
           mean, variance, kurtosis, correlation);
    check(fabs(mean) < 0.005, "noise mean near 0");
    check(fabs(variance - 1.0) < 0.01, "noise variance near 1");
    check(fabs(kurtosis - 3.0) < 0.05, "noise kurtosis near 3 (Gaussian)");
    check(fabs(correlation) < 0.005, "consecutive steps uncorrelated");

    uint32_t video[2], audio[2], other_seed[2];
    h3_renoise_keys(42, 0, 1, audio);
    h3_renoise_keys(43, 0, 0, other_seed);
    memcpy(video, a, sizeof(video));
    check(memcmp(video, audio, sizeof(video)) != 0, "video and audio keys differ");
    check(memcmp(video, other_seed, sizeof(video)) != 0, "seeds give different keys");
}

static void gpu_matches_cpu(h3_gpu *gpu) {
    enum { COUNT = 300007, OFFSET = 1000 };
    float *sample = malloc((OFFSET + COUNT) * sizeof(*sample));
    float *expected = malloc(COUNT * sizeof(*expected));
    float *velocity = malloc(COUNT * sizeof(*velocity));
    float *actual = malloc(COUNT * sizeof(*actual));
    uint16_t *velocity_bf16 = malloc(COUNT * sizeof(*velocity_bf16));
    h3_rng rng;
    h3_rng_seed(&rng, 7);
    for (size_t index = 0; index < OFFSET + COUNT; index++)
        sample[index] = h3_rng_normal(&rng);
    for (size_t index = 0; index < COUNT; index++) {
        velocity_bf16[index] = to_bf16(h3_rng_normal(&rng));
        velocity[index] = from_bf16(velocity_bf16[index]);
    }
    const float sigmas[][2] = {{1.0f, 0.97297f}, {0.5f, 0.25f}, {0.3f, 0.0f}};
    for (size_t pass = 0; pass < sizeof(sigmas) / sizeof(*sigmas); pass++) {
        uint32_t keys[2];
        h3_renoise_keys(12345, (int)pass, 0, keys);
        memcpy(expected, sample + OFFSET, COUNT * sizeof(*expected));
        check(h3_renoise_velocity_step(expected, velocity, COUNT,
                                       sigmas[pass][0], sigmas[pass][1], keys),
              "CPU re-noise step accepts valid sigmas");
        h3_gpu_tensor *gpu_sample =
            h3_gpu_tensor_from_f32(gpu, sample, OFFSET + COUNT);
        h3_gpu_tensor *gpu_velocity =
            h3_gpu_tensor_from_bf16(gpu, velocity_bf16, COUNT);
        int ok = gpu_sample && gpu_velocity && h3_gpu_begin(gpu) &&
                 h3_gpu_renoise_bf16(gpu, gpu_sample, OFFSET, gpu_velocity,
                                     COUNT, sigmas[pass][0], sigmas[pass][1],
                                     keys) &&
                 h3_gpu_submit(gpu) &&
                 h3_gpu_tensor_read_f32_range(gpu_sample, OFFSET, actual, COUNT);
        check(ok, "GPU re-noise step runs");
        float untouched = 0;
        if (ok && h3_gpu_tensor_read_f32_range(gpu_sample, OFFSET - 1,
                                               &untouched, 1))
            check(untouched == sample[OFFSET - 1],
                  "GPU re-noise leaves elements before the offset alone");
        double worst = 0;
        for (size_t index = 0; ok && index < COUNT; index++) {
            double error = fabs((double)actual[index] - expected[index]);
            if (error > worst) worst = error;
        }
        printf("pass %zu (sigma %.5f -> %.5f): max |GPU - CPU| = %.3g\n", pass,
               sigmas[pass][0], sigmas[pass][1], worst);
        check(worst < 2e-5, "GPU matches CPU within float rounding");
        if (sigmas[pass][1] == 0.0f) {
            double clean_error = 0;
            for (size_t index = 0; index < COUNT; index++) {
                double clean = sample[OFFSET + index] +
                               sigmas[pass][0] * velocity[index];
                double error = fabs(clean - expected[index]);
                if (error > clean_error) clean_error = error;
            }
            check(clean_error < 1e-6, "last step returns the clean prediction");
        }
        h3_gpu_tensor_free(gpu_sample);
        h3_gpu_tensor_free(gpu_velocity);
    }
    free(sample); free(expected); free(velocity); free(actual);
    free(velocity_bf16);
}

int main(void) {
    noise_statistics();
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create GPU: %s\n", error);
        return 1;
    }
    gpu_matches_cpu(gpu);
    h3_gpu_free(gpu);
    if (failures) {
        fprintf(stderr, "%d re-noise check(s) failed\n", failures);
        return 1;
    }
    printf("re-noise tests passed\n");
    return 0;
}
