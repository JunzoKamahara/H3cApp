/* SeedVR2 on PNG frames through the engine's session (development and
 * evaluation; the app uses the same C API from Swift):
 *   h3_seedvr2_cli --model DIR --in IN.png --out OUT.png [--width W --height H]
 *                  [--no-color] [--seed N] [--frames IN_DIR OUT_DIR]
 * Without --width/--height the output is 2x. --frames restores every PNG
 * of a folder (sorted) into another, with one session. */
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include "h3_seedvr2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint8_t *read_png_bgra(const char *path, int *width, int *height) {
    NSURL *url = [NSURL fileURLWithPath:@(path)];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (!source) return NULL;
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CFRelease(source);
    if (!image) return NULL;
    int w = (int)CGImageGetWidth(image), h = (int)CGImageGetHeight(image);
    uint8_t *pixels = calloc((size_t)w * h * 4, 1);
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pixels, w, h, 8, (size_t)w * 4, space,
        (CGBitmapInfo)kCGBitmapByteOrder32Little | (CGBitmapInfo)kCGImageAlphaNoneSkipFirst);   /* BGRA in memory */
    CGContextDrawImage(context, CGRectMake(0, 0, w, h), image);
    CGContextRelease(context);
    CGColorSpaceRelease(space);
    CGImageRelease(image);
    *width = w; *height = h;
    return pixels;
}

static int write_png_bgra(const char *path, const uint8_t *pixels, int width, int height) {
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate((void *)pixels, width, height, 8, (size_t)width * 4, space,
        (CGBitmapInfo)kCGBitmapByteOrder32Little | (CGBitmapInfo)kCGImageAlphaNoneSkipFirst);
    CGImageRef image = CGBitmapContextCreateImage(context);
    NSURL *url = [NSURL fileURLWithPath:@(path)];
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, NULL);
    int ok = 0;
    if (dest && image) {
        CGImageDestinationAddImage(dest, image, NULL);
        ok = CGImageDestinationFinalize(dest);
    }
    if (dest) CFRelease(dest);
    if (image) CGImageRelease(image);
    CGContextRelease(context);
    CGColorSpaceRelease(space);
    return ok;
}

static int progress(const char *phase, int done, int total, void *opaque) {
    (void)opaque;
    if (getenv("H3_SV2_PROGRESS")) fprintf(stderr, "  %s %d/%d\n", phase, done, total);
    return 0;
}

static int restore_one(h3_seedvr2 *model, h3_seedvr2_session **session, const char *in_path,
                       const char *out_path, int out_w, int out_h) {
    int w, h;
    uint8_t *in = read_png_bgra(in_path, &w, &h);
    if (!in) { fprintf(stderr, "cannot read %s\n", in_path); return 0; }
    int ow = out_w ? out_w : 2 * w, oh = out_h ? out_h : 2 * h;
    char error[512];
    if (!*session) {
        h3_seedvr2_requirements req;
        h3_seedvr2_check_requirements(model, w, h, ow, oh, &req);
        fprintf(stderr, "%dx%d -> %dx%d: %d tokens, windows %d/%d, est. %.1f s/frame, %.1f GB; %s\n",
                w, h, ow, oh, req.tokens, req.windows_even, req.windows_odd, req.estimated_seconds_per_frame,
                (double)req.gpu_bytes / 1e9, req.supported ? "supported" : req.reason);
        *session = h3_seedvr2_session_open(model, w, h, ow, oh, error, sizeof(error));
        if (!*session) { fprintf(stderr, "session: %s\n", error); free(in); return 0; }
    }
    uint8_t *out = malloc((size_t)ow * oh * 4);
    h3_vsr_image image = {in, w, h, w * 4, H3_VSR_BGRA8};
    h3_vsr_image_out result = {out, ow, oh, ow * 4, H3_VSR_BGRA8};
    double t0 = now_seconds();
    int ok = h3_seedvr2_session_process(*session, &image, &result, progress, NULL, NULL, error, sizeof(error));
    double dt = now_seconds() - t0;
    if (!ok) fprintf(stderr, "%s: %s\n", in_path, error);
    else if (!write_png_bgra(out_path, out, ow, oh)) { fprintf(stderr, "cannot write %s\n", out_path); ok = 0; }
    else fprintf(stderr, "%s -> %s  %.2f s\n", in_path, out_path, dt);
    free(in); free(out);
    return ok;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *in_path = NULL, *out_path = NULL, *frames_in = NULL, *frames_out = NULL;
    int out_w = 0, out_h = 0, color = 1;
    uint64_t seed = 42;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--width") && i + 1 < argc) out_w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i + 1 < argc) out_h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--no-color")) color = 0;
        else if (!strcmp(argv[i], "--frames") && i + 2 < argc) { frames_in = argv[++i]; frames_out = argv[++i]; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model_dir || (!(in_path && out_path) && !(frames_in && frames_out))) {
        fprintf(stderr, "usage: %s --model DIR (--in IN.png --out OUT.png | --frames IN_DIR OUT_DIR) [--width W --height H] [--no-color] [--seed N]\n", argv[0]);
        return 2;
    }
    char error[512];
    h3_seedvr2_open_params params = {"h3_shaders.metal", color, seed, -1};
    double t0 = now_seconds();
    h3_seedvr2 *model = h3_seedvr2_open(model_dir, &params, NULL, NULL, error, sizeof(error));
    if (!model) { fprintf(stderr, "open: %s\n", error); return 1; }
    fprintf(stderr, "model loaded in %.1f s\n", now_seconds() - t0);
    h3_seedvr2_session *session = NULL;
    int failures = 0;
    @autoreleasepool {
        if (in_path) {
            if (!restore_one(model, &session, in_path, out_path, out_w, out_h)) failures++;
        } else {
            NSArray<NSString *> *names = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:@(frames_in) error:NULL]
                sortedArrayUsingSelector:@selector(compare:)];
            [[NSFileManager defaultManager] createDirectoryAtPath:@(frames_out) withIntermediateDirectories:YES attributes:nil error:NULL];
            double start = now_seconds();
            int count = 0;
            for (NSString *name in names) {
                if (![name.pathExtension.lowercaseString isEqualToString:@"png"]) continue;
                NSString *src = [@(frames_in) stringByAppendingPathComponent:name];
                NSString *dst = [@(frames_out) stringByAppendingPathComponent:name];
                if (!restore_one(model, &session, src.UTF8String, dst.UTF8String, out_w, out_h)) failures++;
                count++;
            }
            double total = now_seconds() - start;
            fprintf(stderr, "%d frames in %.1f s (%.2f s/frame)\n", count, total, count ? total / count : 0);
        }
    }
    double gpu; uint64_t peak;
    if (h3_seedvr2_get_stats(model, &gpu, &peak))
        fprintf(stderr, "gpu %.1f s, peak live %.2f GiB\n", gpu, (double)peak / 1073741824.0);
    h3_seedvr2_session_close(session);
    h3_seedvr2_close(model);
    return failures ? 1 : 0;
}
