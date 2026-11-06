/* SeedVR2 on PNG frames through the engine's session (development and
 * evaluation; the app uses the same C API from Swift):
 *   h3_seedvr2_cli --model DIR --in IN.png --out OUT.png [--width W --height H]
 *                  [--no-color] [--seed N] [--frames IN_DIR OUT_DIR]
 *                  [--video-frames IN_DIR OUT_DIR [--window N] [--overlap M] [--blend 0|1] [--vae-split K]]
 * Without --width/--height the output is 2x. --frames restores every PNG
 * of a folder (sorted) into another, one frame at a time, with one session.
 * --video-frames restores every PNG through the window session
 * (h3_seedvr2_video_session_*): windows of --window frames (4n+1, default
 * 17) overlapping by --overlap (0 or 4n+1, default 1), the overlap from the
 * earlier window (--blend 0) or cross-faded (1). */
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

/* --video-frames: every PNG of a folder through the window session, fed as
 * next_frames asks, outputs written as they come. */
static int restore_video(h3_seedvr2 *model, const char *frames_in, const char *frames_out, int window, int overlap,
                         int blend, int vae_split, uint64_t seed, int out_w, int out_h) {
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    for (NSString *name in [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:@(frames_in) error:NULL]
                            sortedArrayUsingSelector:@selector(compare:)])
        if ([name.pathExtension.lowercaseString isEqualToString:@"png"]) [names addObject:name];
    int total = (int)names.count;
    if (total < 1) { fprintf(stderr, "%s: no PNG frames\n", frames_in); return 0; }
    [[NSFileManager defaultManager] createDirectoryAtPath:@(frames_out) withIntermediateDirectories:YES attributes:nil error:NULL];
    int w = 0, h = 0;
    uint8_t *first = read_png_bgra([@(frames_in) stringByAppendingPathComponent:names[0]].UTF8String, &w, &h);
    if (!first) { fprintf(stderr, "cannot read %s\n", names[0].UTF8String); return 0; }
    free(first);
    if (!out_w && !out_h) { out_w = 2 * w; out_h = 2 * h; }
    else if (!out_w) out_w = (int)lround((double)out_h * w / h);
    else if (!out_h) out_h = (int)lround((double)out_w * h / w);
    h3_seedvr2_video_params params = {window, overlap, vae_split, blend, seed, -1};
    char error[512];
    h3_seedvr2_video_requirements req;
    if (h3_seedvr2_video_check_requirements(model, w, h, out_w, out_h, window, &req))
        fprintf(stderr, "window %d: %s, about %.1f GB, vae %s, %.1f s/frame estimated\n", window,
                req.supported ? "fits" : req.reason, (double)req.gpu_bytes / 1e9, req.vae_split ? "sliced" : "whole",
                req.estimated_seconds_per_frame);
    h3_seedvr2_video_session *s = h3_seedvr2_video_session_open(model, w, h, out_w, out_h, total, &params, error, sizeof(error));
    if (!s) { fprintf(stderr, "session: %s\n", error); return 0; }
    fprintf(stderr, "%d frames %dx%d -> %dx%d, window %d, overlap %d, blend %d\n", total, w, h, out_w, out_h, window, overlap, blend);
    int ok = 1, sent = 0, written = 0;
    double start = now_seconds();
    while (sent < total && ok) {
        int want = h3_seedvr2_video_session_next_frames(s);
        if (want < 1) { fprintf(stderr, "session asks for no frames at %d of %d\n", sent, total); ok = 0; break; }
        int give = want < total - sent ? want : total - sent;
        int is_last = sent + give >= total;
        h3_vsr_image *in = calloc((size_t)give, sizeof(*in));
        uint8_t **in_px = calloc((size_t)give, sizeof(*in_px));
        for (int t = 0; t < give && ok; t++) {
            int fw, fh;
            in_px[t] = read_png_bgra([@(frames_in) stringByAppendingPathComponent:names[sent + t]].UTF8String, &fw, &fh);
            if (!in_px[t] || fw != w || fh != h) { fprintf(stderr, "cannot read %s\n", names[sent + t].UTF8String); ok = 0; }
            else in[t] = (h3_vsr_image){in_px[t], w, h, w * 4, H3_VSR_BGRA8};
        }
        int capacity = h3_seedvr2_video_session_pending_frames(s) + give, produced = 0;
        h3_vsr_image_out *out = calloc((size_t)capacity, sizeof(*out));
        for (int t = 0; t < capacity; t++) out[t] = (h3_vsr_image_out){calloc((size_t)out_w * out_h * 4, 1), out_w, out_h, out_w * 4, H3_VSR_BGRA8};
        if (ok && !h3_seedvr2_video_session_process(s, in, give, is_last, out, capacity, &produced, NULL, NULL, NULL, error, sizeof(error))) {
            fprintf(stderr, "process: %s\n", error);
            ok = 0;
        }
        for (int t = 0; t < produced && ok; t++) {
            if (!write_png_bgra([@(frames_out) stringByAppendingPathComponent:names[written]].UTF8String, out[t].pixels, out_w, out_h)) {
                fprintf(stderr, "cannot write %s\n", names[written].UTF8String);
                ok = 0;
            }
            written++;
        }
        for (int t = 0; t < capacity; t++) free(out[t].pixels);
        for (int t = 0; t < give; t++) free(in_px[t]);
        free(out); free(in); free(in_px);
        sent += give;
        int frames_run, padded; double seconds;
        int windows = h3_seedvr2_video_session_windows(s, &frames_run, &padded, &seconds);
        fprintf(stderr, "  %d/%d in, %d out, %d windows (%d frames run, %d padding), %.1f s\n", sent, total, written, windows, frames_run, padded, seconds);
    }
    double elapsed = now_seconds() - start;
    if (ok && written != total) { fprintf(stderr, "%d frames written of %d\n", written, total); ok = 0; }
    fprintf(stderr, "%d frames in %.1f s (%.2f s/frame)\n", written, elapsed, written ? elapsed / written : 0);
    h3_seedvr2_video_session_close(s);
    return ok;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *in_path = NULL, *out_path = NULL, *frames_in = NULL, *frames_out = NULL;
    int out_w = 0, out_h = 0, color = 1, video = 0, window = 17, overlap = 1, blend = 0, vae_split = -1;
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
        else if (!strcmp(argv[i], "--video-frames") && i + 2 < argc) { frames_in = argv[++i]; frames_out = argv[++i]; video = 1; }
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vae-split") && i + 1 < argc) vae_split = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--overlap") && i + 1 < argc) overlap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blend") && i + 1 < argc) blend = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model_dir || (!(in_path && out_path) && !(frames_in && frames_out))) {
        fprintf(stderr, "usage: %s --model DIR (--in IN.png --out OUT.png | --frames IN_DIR OUT_DIR | --video-frames IN_DIR OUT_DIR [--window N] [--overlap M] [--blend 0|1] [--vae-split K]) [--width W --height H] [--no-color] [--seed N]\n", argv[0]);
        return 2;
    }
    char error[512];
    h3_seedvr2_open_params params = {"h3_shaders.metal", color, seed, -1, 0, -1};
    double t0 = now_seconds();
    h3_seedvr2 *model = h3_seedvr2_open(model_dir, &params, NULL, NULL, error, sizeof(error));
    if (!model) { fprintf(stderr, "open: %s\n", error); return 1; }
    fprintf(stderr, "model loaded in %.1f s\n", now_seconds() - t0);
    h3_seedvr2_session *session = NULL;
    int failures = 0;
    @autoreleasepool {
        if (in_path) {
            if (!restore_one(model, &session, in_path, out_path, out_w, out_h)) failures++;
        } else if (video) {
            if (!restore_video(model, frames_in, frames_out, window, overlap, blend, vae_split, seed, out_w, out_h)) failures++;
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
