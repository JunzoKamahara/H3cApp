/* RefMod extraction on the released Ref2VA VAEs: a Full RefMod made from
 * an image carries the same rows the ordinary reference path encodes (up
 * to F16 rounding), a compressed one pools and refines, and an audio one
 * matches the audio reference path. Needs the model folder:
 *
 *   ./h3_real_refmod_test [MiniMax-H3]
 */
#include "h3_refmod.h"
#include "h3_audio_vae.h"
#include "h3_av_reader.h"
#include "h3_dit.h"
#include "h3_host.h"
#include "h3_video_encoder.h"

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int checks;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

/* The same neutral 512x512 image tests/repro_check.c draws. */
static int write_reference_png(const char *path, int size) {
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef bitmap = CGBitmapContextCreate(
        NULL, (size_t)size, (size_t)size, 8, (size_t)size * 4, space,
        (CGBitmapInfo)kCGImageAlphaNoneSkipLast);
    if (!bitmap) { CGColorSpaceRelease(space); return 0; }
    uint8_t *pixels = CGBitmapContextGetData(bitmap);
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            uint8_t *p = pixels + ((size_t)y * size + x) * 4;
            int dx = x - size / 2, dy = y - size * 2 / 5;
            int disc = dx * dx + dy * dy < (size / 5) * (size / 5);
            p[0] = (uint8_t)(disc ? 230 : 60 + x * 120 / size);
            p[1] = (uint8_t)(disc ? 150 : 90 + y * 100 / size);
            p[2] = (uint8_t)(disc ? 60 : 160 - y * 80 / size);
            p[3] = 255;
        }
    CGImageRef image = CGBitmapContextCreateImage(bitmap);
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        NULL, (const UInt8 *)path, (CFIndex)strlen(path), false);
    CGImageDestinationRef destination =
        url ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, NULL)
            : NULL;
    int ok = 0;
    if (destination && image) {
        CGImageDestinationAddImage(destination, image, NULL);
        ok = CGImageDestinationFinalize(destination);
    }
    if (destination) CFRelease(destination);
    if (url) CFRelease(url);
    if (image) CGImageRelease(image);
    CGContextRelease(bitmap);
    CGColorSpaceRelease(space);
    return ok;
}

/* A 3-second stereo WAV (two tones) at 32 kHz. */
static int write_tone_wav(const char *path) {
    const int rate = 32000, seconds = 3, channels = 2;
    const int frames = rate * seconds;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    uint32_t data_bytes = (uint32_t)frames * channels * 2;
    uint32_t riff_size = 36 + data_bytes;
    uint32_t fmt_size = 16, byte_rate = (uint32_t)rate * channels * 2;
    uint16_t format = 1, block_align = channels * 2, bits = 16, ch = channels;
    fwrite("RIFF", 1, 4, file); fwrite(&riff_size, 4, 1, file);
    fwrite("WAVE", 1, 4, file); fwrite("fmt ", 1, 4, file);
    fwrite(&fmt_size, 4, 1, file); fwrite(&format, 2, 1, file);
    fwrite(&ch, 2, 1, file); fwrite(&rate, 4, 1, file);
    fwrite(&byte_rate, 4, 1, file); fwrite(&block_align, 2, 1, file);
    fwrite(&bits, 2, 1, file); fwrite("data", 1, 4, file);
    fwrite(&data_bytes, 4, 1, file);
    for (int frame = 0; frame < frames; frame++) {
        double t = (double)frame / rate;
        int16_t left = (int16_t)(12000.0 * sin(2 * M_PI * 220.0 * t) *
                                 (0.5 + 0.5 * sin(2 * M_PI * 0.5 * t)));
        int16_t right = (int16_t)(9000.0 * sin(2 * M_PI * 330.0 * t));
        fwrite(&left, 2, 1, file); fwrite(&right, 2, 1, file);
    }
    return fclose(file) == 0;
}

static float max_abs_difference(const float *a, const float *b, size_t count,
                                float *max_magnitude) {
    float worst = 0.0f, magnitude = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(a[index] - b[index]);
        if (difference > worst) worst = difference;
        if (fabsf(a[index]) > magnitude) magnitude = fabsf(a[index]);
    }
    if (max_magnitude) *max_magnitude = magnitude;
    return worst;
}

static int progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed % 100 == 0)
        fprintf(stderr, "  %s %d/%d\n", phase, completed, total);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char vae_path[1024], audio_vae_path[1024];
    snprintf(vae_path, sizeof(vae_path), "%s/Ref2VA/video_vae/source", model_root);
    snprintf(audio_vae_path, sizeof(audio_vae_path), "%s/Ref2VA/audio_vae", model_root);
    char probe[1200];
    snprintf(probe, sizeof(probe), "%s/model.safetensors", vae_path);
    if (access(probe, R_OK) != 0) {
        printf("skip: released Ref2VA video VAE is not installed at %s\n", vae_path);
        return 0;
    }
    char directory[1024];
    const char *base = getenv("TMPDIR");
    snprintf(directory, sizeof(directory), "%s/h3-real-refmod-XXXXXX",
             base && *base ? base : "/tmp");
    CHECK(mkdtemp(directory) != NULL);
    char image_path[1200], full_path[1200], compressed_path[1200];
    snprintf(image_path, sizeof(image_path), "%s/reference.png", directory);
    snprintf(full_path, sizeof(full_path), "%s/full.safetensors", directory);
    snprintf(compressed_path, sizeof(compressed_path), "%s/compressed.safetensors", directory);
    CHECK(write_reference_png(image_path, 512));
    char error[512];

    /* (a) The ordinary reference path: 512x512 stays 512x512 for a 512x512
     * target (h3_reference_image_canvas), which is also what the RefMod
     * canvas rule gives, so the two encodes see the same pixels. */
    int width, height;
    CHECK(h3_reference_image_canvas(512, 512, 512, 512, 0, &width, &height));
    CHECK(width == 512 && height == 512);
    int w2, h2;
    CHECK(h3_refmod_extract_canvas(512, 512, 1024, &w2, &h2));
    CHECK(w2 == 512 && h2 == 512);
    float *pixels = NULL;
    CHECK(h3_av_read_image_f32(image_path, 512, 512, H3_IMAGE_FIT_STRETCH, &pixels,
                               error, sizeof(error)));
    h3_video_latent reference;
    memset(&reference, 0, sizeof(reference));
    fprintf(stderr, "encoding the reference image the ordinary way\n");
    CHECK(h3_video_vae_encode(vae_path, "h3_shaders.metal", pixels, 1, 512, 512,
                              NULL, NULL, &reference, error, sizeof(error)));
    free(pixels);
    CHECK(reference.time == 1 && reference.height == 32 && reference.width == 32);
    size_t row_elements = (size_t)16 * 16 * 96;
    float *reference_rows = malloc(row_elements * sizeof(*reference_rows));
    CHECK(reference_rows && h3_dit_patchify_video(reference.values, 24, 1, 32, 32,
                                                  reference_rows, row_elements));

    /* (b) A Full RefMod from the same image. */
    h3_refmod_extract_params params = {model_root, image_path, full_path, "probe",
                                       H3_REFMOD_IMAGE, H3_REFMOD_FULL, 1024, 16,
                                       16, 0, 0, NULL, NULL};
    fprintf(stderr, "extracting a Full RefMod\n");
    CHECK(h3_refmod_extract(&params, progress, NULL, error, sizeof(error)));
    h3_refmod_info info;
    CHECK(h3_refmod_inspect(full_path, &info, error, sizeof(error)));
    CHECK(info.member_count == 1 && info.members[0].kind == H3_REFMOD_IMAGE &&
          info.members[0].mode == H3_REFMOD_FULL && info.members[0].tokens == 256 &&
          info.members[0].latent_h == 32 && info.members[0].latent_w == 32);
    CHECK(strstr(info.members[0].pool, "512x512px") != NULL);
    h3_refmod_info_free(&info);
    h3_refmod_latent loaded;
    CHECK(h3_refmod_load(full_path, -1, 1.0f, &loaded, NULL, error, sizeof(error)));
    float *refmod_rows = malloc(row_elements * sizeof(*refmod_rows));
    CHECK(refmod_rows && h3_dit_patchify_video(loaded.values, 24, 1, 32, 32,
                                               refmod_rows, row_elements));
    float magnitude;
    float worst = max_abs_difference(reference_rows, refmod_rows, row_elements, &magnitude);
    /* F16 keeps 11 significant bits: the rounding error is at most half an
     * ulp of the largest value. */
    float tolerance = magnitude / 1024.0f + 1e-4f;
    printf("full RefMod vs reference rows: max |diff| %.5f (tolerance %.5f, max |value| %.3f)\n",
           worst, tolerance, magnitude);
    CHECK(worst <= tolerance);
    h3_refmod_latent_free(&loaded);

    /* (c) A compressed RefMod: 8-cell grid, 50 steps, and the refinement
     * brings the upsampled small latent closer to the full one. */
    params.output_path = compressed_path;
    params.mode = H3_REFMOD_COMPRESSED;
    params.pool_long_edge = 8;
    params.optimize_steps = 50;
    fprintf(stderr, "extracting a compressed RefMod\n");
    CHECK(h3_refmod_extract(&params, progress, NULL, error, sizeof(error)));
    CHECK(h3_refmod_inspect(compressed_path, &info, error, sizeof(error)));
    CHECK(info.members[0].mode == H3_REFMOD_COMPRESSED && info.members[0].tokens == 16 &&
          info.members[0].optimize_steps == 50 && !strcmp(info.members[0].pool, "1x8x8"));
    h3_refmod_info_free(&info);
    CHECK(h3_refmod_load(compressed_path, -1, 1.0f, &loaded, NULL, error, sizeof(error)));
    CHECK(loaded.time == 1 && loaded.height == 8 && loaded.width == 8);
    size_t full_count = (size_t)24 * 32 * 32;
    float *up = malloc(full_count * sizeof(*up));
    float *pooled = malloc((size_t)24 * 8 * 8 * sizeof(*pooled));
    CHECK(up && pooled);
    CHECK(h3_refmod_pool3d(reference.values, 24, 1, 32, 32, 1, 8, 8, pooled));
    CHECK(h3_refmod_trilinear(pooled, 24, 1, 8, 8, 1, 32, 32, up));
    double pooled_mse = 0.0, refined_mse = 0.0;
    for (size_t index = 0; index < full_count; index++)
        pooled_mse += (up[index] - reference.values[index]) * (up[index] - reference.values[index]);
    CHECK(h3_refmod_trilinear(loaded.values, 24, 1, 8, 8, 1, 32, 32, up));
    for (size_t index = 0; index < full_count; index++)
        refined_mse += (up[index] - reference.values[index]) * (up[index] - reference.values[index]);
    printf("compressed RefMod: mse of the pooled start %.5f, after 50 steps %.5f\n",
           pooled_mse / (double)full_count, refined_mse / (double)full_count);
    CHECK(refined_mse < pooled_mse);
    h3_refmod_latent_free(&loaded);
    free(up); free(pooled);
    free(reference_rows); free(refmod_rows);
    h3_video_latent_free(&reference);

    /* (d) Audio, when the audio VAE is installed. */
    snprintf(probe, sizeof(probe), "%s/model.safetensors", audio_vae_path);
    if (access(probe, R_OK) == 0) {
        char wav_path[1200], audio_path[1200];
        snprintf(wav_path, sizeof(wav_path), "%s/tone.wav", directory);
        snprintf(audio_path, sizeof(audio_path), "%s/tone.safetensors", directory);
        CHECK(write_tone_wav(wav_path));
        float *pcm = NULL;
        int samples = 0;
        CHECK(h3_av_read_audio_f32(wav_path, 32000 * 15, 1, &pcm, &samples,
                                   error, sizeof(error)));
        h3_audio_latent audio;
        memset(&audio, 0, sizeof(audio));
        fprintf(stderr, "encoding the tone the ordinary way\n");
        CHECK(h3_audio_vae_encode(audio_vae_path, "h3_shaders.metal", pcm, samples,
                                  NULL, NULL, &audio, error, sizeof(error)));
        free(pcm);
        CHECK(audio.channels == 32 && audio.stereo == 2 && audio.length == 120);
        h3_refmod_extract_params audio_params = {model_root, wav_path, audio_path,
                                                 "tone", H3_REFMOD_AUDIO, H3_REFMOD_FULL,
                                                 0, 0, 0, 0, 15, NULL, NULL};
        fprintf(stderr, "extracting an audio RefMod\n");
        CHECK(h3_refmod_extract(&audio_params, progress, NULL, error, sizeof(error)));
        CHECK(h3_refmod_load(audio_path, -1, 1.0f, &loaded, NULL, error, sizeof(error)));
        CHECK(loaded.kind == H3_REFMOD_AUDIO && loaded.audio_t == 120);
        worst = max_abs_difference(audio.values, loaded.values, (size_t)32 * 2 * 120,
                                   &magnitude);
        tolerance = magnitude / 1024.0f + 1e-4f;
        printf("audio RefMod vs reference latent: max |diff| %.5f (tolerance %.5f)\n",
               worst, tolerance);
        CHECK(worst <= tolerance);
        h3_refmod_latent_free(&loaded);
        h3_audio_latent_free(&audio);
    } else {
        printf("skip: released Ref2VA audio VAE is not installed at %s\n", audio_vae_path);
    }
    char command[1200];
    snprintf(command, sizeof(command), "rm -rf '%s'", directory);
    (void)system(command);
    printf("ok: %d real RefMod checks\n", checks);
    return 0;
}
