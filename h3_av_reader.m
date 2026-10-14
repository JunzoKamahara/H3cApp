/* Decodes user-supplied reference images/video/audio through
 * ImageIO/AVFoundation.
 *
 * By the time width/height reach h3_av_read_image_f32 (H3_IMAGE_FIT_STRETCH)
 * or h3_av_read_video_f32, h3_reference_image_canvas/h3_reference_video_canvas
 * (h3_host.c) have already picked a target canvas that preserves the source's
 * aspect ratio - so a plain non-uniform stretch to exactly (width, height) is
 * correct there and needs no separate aspect-fit crop. H3_IMAGE_FIT_COVER
 * (used only for first/last frame, where target is a fixed render size that
 * may not match the source aspect) does its own aspect-fill + center crop
 * below. */
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include "h3_av_reader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void h3_av_set_error(char *error, size_t error_size, NSString *message) {
    if (!error || !error_size || !message) return;
    strncpy(error, message.UTF8String, error_size - 1);
    error[error_size - 1] = '\0';
}

/* ---- image metadata / decode (ImageIO) ------------------------------- */

/* CGImageSourceCreateThumbnailAtIndex with WithTransform=true and a max
 * pixel size at least as large as the source bakes the EXIF orientation
 * into the returned CGImage's pixels (no rotation math of our own needed)
 * and, since the request is >= the source size, returns it undownsampled. */
static CGImageRef h3_copy_oriented_image(CGImageSourceRef source, size_t *outWidth,
                                         size_t *outHeight, NSString **errorOut) {
    CFDictionaryRef sourceProps = CGImageSourceCopyPropertiesAtIndex(source, 0, NULL);
    if (!sourceProps) {
        *errorOut = @"cannot read image properties";
        return NULL;
    }
    NSDictionary *props = (__bridge NSDictionary *)sourceProps;
    NSNumber *pixelWidth = props[(id)kCGImagePropertyPixelWidth];
    NSNumber *pixelHeight = props[(id)kCGImagePropertyPixelHeight];
    CFRelease(sourceProps);
    if (!pixelWidth || !pixelHeight) {
        *errorOut = @"image has no pixel dimensions";
        return NULL;
    }
    size_t maxSide = (size_t)MAX(pixelWidth.doubleValue, pixelHeight.doubleValue);
    NSDictionary *thumbOptions = @{
        (id)kCGImageSourceCreateThumbnailWithTransform: @YES,
        (id)kCGImageSourceCreateThumbnailFromImageAlways: @YES,
        (id)kCGImageSourceThumbnailMaxPixelSize: @(maxSide),
    };
    CGImageRef image = CGImageSourceCreateThumbnailAtIndex(
        source, 0, (__bridge CFDictionaryRef)thumbOptions);
    if (!image) {
        *errorOut = @"cannot decode image";
        return NULL;
    }
    if (outWidth) *outWidth = CGImageGetWidth(image);
    if (outHeight) *outHeight = CGImageGetHeight(image);
    return image;
}

static int h3_probe_image_size(const char *path, int *width, int *height) {
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (!source) return 0;
    CFDictionaryRef props = CGImageSourceCopyPropertiesAtIndex(source, 0, NULL);
    if (!props) {
        CFRelease(source);
        return 0;
    }
    NSDictionary *dict = (__bridge NSDictionary *)props;
    NSNumber *pixelWidth = dict[(id)kCGImagePropertyPixelWidth];
    NSNumber *pixelHeight = dict[(id)kCGImagePropertyPixelHeight];
    NSNumber *orientation = dict[(id)kCGImagePropertyOrientation];
    int ok = 0;
    if (pixelWidth && pixelHeight) {
        int w = pixelWidth.intValue, h = pixelHeight.intValue;
        /* EXIF orientations 5-8 are the 90/270-degree rotations, where the
         * pixel dimensions on disk are transposed relative to how the image
         * displays. */
        int value = orientation ? orientation.intValue : 1;
        if (value >= 5 && value <= 8) { int t = w; w = h; h = t; }
        if (width) *width = w;
        if (height) *height = h;
        ok = 1;
    }
    CFRelease(props);
    CFRelease(source);
    return ok;
}

/* ---- video metadata (AVFoundation) ------------------------------------ */

static int h3_probe_video_size(const char *path, int *width, int *height) {
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
        #pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        /* Synchronous by design: this whole C ABI is synchronous, and for a
         * local file tracksWithMediaType: already blocks until track info is
         * available - the async replacement would need the same blocking
         * wrapper here for no behavioral difference. */
        NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        #pragma clang diagnostic pop
        if (tracks.count == 0) return 0;
        AVAssetTrack *track = tracks.firstObject;
        CGSize natural = track.naturalSize;
        CGRect displayRect = CGRectApplyAffineTransform(
            CGRectMake(0, 0, natural.width, natural.height), track.preferredTransform);
        int w = (int)lround(fabs(displayRect.size.width));
        int h = (int)lround(fabs(displayRect.size.height));
        if (w < 1 || h < 1) return 0;
        if (width) *width = w;
        if (height) *height = h;
        return 1;
    }
}

int h3_av_visual_size(const char *path, int *width, int *height,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (width) *width = 0;
    if (height) *height = 0;
    if (!path || !*path || !width || !height) {
        h3_av_set_error(error, error_size, @"invalid visual-size arguments");
        return 0;
    }
    @autoreleasepool {
        if (h3_probe_image_size(path, width, height)) return 1;
        if (h3_probe_video_size(path, width, height)) return 1;
    }
    h3_av_set_error(error, error_size,
        [NSString stringWithFormat:@"cannot inspect visual stream %s", path]);
    return 0;
}

/* ---- image decode ------------------------------------------------------ */

int h3_av_read_image_f32(const char *path, int width, int height,
                         h3_image_fit fit, float **pixels,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (!path || !*path || !pixels || width < 1 || height < 1 ||
        (fit != H3_IMAGE_FIT_STRETCH && fit != H3_IMAGE_FIT_COVER)) {
        h3_av_set_error(error, error_size, @"invalid image input arguments");
        return 0;
    }
    if ((size_t)width > SIZE_MAX / (size_t)height) {
        h3_av_set_error(error, error_size, @"decoded image size overflows");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3 || area * 3 > SIZE_MAX / sizeof(float)) {
        h3_av_set_error(error, error_size, @"decoded image size overflows");
        return 0;
    }
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (!source) {
            h3_av_set_error(error, error_size,
                [NSString stringWithFormat:@"cannot open image %s", path]);
            return 0;
        }
        NSString *decodeError = nil;
        size_t sourceW = 0, sourceH = 0;
        CGImageRef image = h3_copy_oriented_image(source, &sourceW, &sourceH, &decodeError);
        CFRelease(source);
        if (!image) {
            h3_av_set_error(error, error_size, decodeError);
            return 0;
        }

        CGRect destRect;
        if (fit == H3_IMAGE_FIT_STRETCH) {
            destRect = CGRectMake(0, 0, width, height);
        } else {
            /* Aspect-fill + center crop: scale up so the shorter edge covers
             * the target canvas, then draw the image offset so the excess
             * hangs off both sides evenly (drawn into a context clipped to
             * the target canvas, so the overhang is simply cropped away). */
            double scale = fmax((double)width / (double)sourceW,
                                (double)height / (double)sourceH);
            double drawW = (double)sourceW * scale;
            double drawH = (double)sourceH * scale;
            destRect = CGRectMake((width - drawW) / 2.0, (height - drawH) / 2.0,
                                  drawW, drawH);
        }

        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        size_t bytesPerRow = (size_t)width * 4;
        uint8_t *rgba = calloc(1, bytesPerRow * (size_t)height);
        if (!rgba) {
            CGImageRelease(image);
            CGColorSpaceRelease(colorSpace);
            h3_av_set_error(error, error_size, @"out of memory decoding image");
            return 0;
        }
        CGContextRef context = CGBitmapContextCreate(
            rgba, (size_t)width, (size_t)height, 8, bytesPerRow, colorSpace,
            (CGBitmapInfo)kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(colorSpace);
        if (!context) {
            free(rgba);
            CGImageRelease(image);
            h3_av_set_error(error, error_size, @"cannot create image decode context");
            return 0;
        }
        CGContextClipToRect(context, CGRectMake(0, 0, width, height));
        CGContextDrawImage(context, destRect, image);
        CGContextRelease(context);
        CGImageRelease(image);

        float *channelMajor = malloc(area * 3 * sizeof(*channelMajor));
        if (!channelMajor) {
            free(rgba);
            h3_av_set_error(error, error_size, @"out of memory converting image");
            return 0;
        }
        const float scale = 1.0f / 255.0f;
        for (size_t pixel = 0; pixel < area; pixel++) {
            const uint8_t *rgbaPixel = rgba + pixel * 4;
            channelMajor[pixel] = (float)rgbaPixel[0] * scale;
            channelMajor[area + pixel] = (float)rgbaPixel[1] * scale;
            channelMajor[2 * area + pixel] = (float)rgbaPixel[2] * scale;
        }
        free(rgba);
        *pixels = channelMajor;
        return 1;
    }
}

/* ---- video decode ------------------------------------------------------ */

/* Builds the transform an AVMutableVideoCompositionLayerInstruction needs to
 * map the track's raw pixels into an upright, (width, height)-sized render
 * canvas in one pass. Only preferredTransform's rotation/mirror component
 * (a/b/c/d) is trusted here, not its translation (tx/ty): real camera
 * files' preferredTransform pairs the rotation with a translation that
 * shifts the rotated content back into the positive quadrant, but that
 * pairing isn't guaranteed (verified against a test asset carrying a pure
 * rotate-about-origin preferredTransform with tx=ty=0, which - if passed
 * straight through instead of rebuilding the translation here - maps the
 * entire frame outside the render canvas and decodes as solid black). The
 * four cases below are the standard set for 0/90/180/270-degree rotation;
 * anything else (e.g. a mirror flip) falls back to identity - best effort,
 * since H3's reference-video inputs aren't expected to be mirrored.
 * A plain non-uniform scale then reaches the exact target size (see the
 * file header comment for why a plain stretch, not a further aspect-fit
 * crop, is correct here). */
static CGAffineTransform h3_video_render_transform(AVAssetTrack *track,
                                                    int width, int height) {
    CGAffineTransform preferred = track.preferredTransform;
    CGSize natural = track.naturalSize;
    CGAffineTransform rotate;
    CGSize displaySize;
    if (fabs(preferred.a) < 0.01 && fabs(preferred.d) < 0.01 && preferred.b > 0.5) {
        rotate = CGAffineTransformMake(0, 1, -1, 0, natural.height, 0); /* +90 */
        displaySize = CGSizeMake(natural.height, natural.width);
    } else if (fabs(preferred.a) < 0.01 && fabs(preferred.d) < 0.01 && preferred.b < -0.5) {
        rotate = CGAffineTransformMake(0, -1, 1, 0, 0, natural.width); /* -90 */
        displaySize = CGSizeMake(natural.height, natural.width);
    } else if (preferred.a < -0.5 && preferred.d < -0.5) {
        rotate = CGAffineTransformMake(-1, 0, 0, -1, natural.width, natural.height); /* 180 */
        displaySize = natural;
    } else {
        rotate = CGAffineTransformIdentity;
        displaySize = natural;
    }
    if (displaySize.width < 1 || displaySize.height < 1) return rotate;
    CGAffineTransform scale = CGAffineTransformMakeScale(
        (double)width / displaySize.width, (double)height / displaySize.height);
    return CGAffineTransformConcat(rotate, scale);
}

int h3_av_read_video_f32(const char *path, int width, int height,
                         int max_frames, float **pixels, int *frames,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (frames) *frames = 0;
    if (!path || !*path || !pixels || !frames || width < 1 || height < 1 ||
        max_frames < 5 || (size_t)width > SIZE_MAX / (size_t)height) {
        h3_av_set_error(error, error_size, @"invalid video input arguments");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3) {
        h3_av_set_error(error, error_size, @"decoded video frame size overflows");
        return 0;
    }
    size_t frameBytes = area * 3;
    if ((size_t)max_frames > SIZE_MAX / frameBytes) {
        h3_av_set_error(error, error_size, @"decoded video size overflows");
        return 0;
    }
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
        #pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        /* Synchronous by design: this whole C ABI is synchronous, and for a
         * local file tracksWithMediaType: already blocks until track info is
         * available - the async replacement would need the same blocking
         * wrapper here for no behavioral difference. */
        NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        #pragma clang diagnostic pop
        if (tracks.count == 0) {
            h3_av_set_error(error, error_size,
                [NSString stringWithFormat:@"%s has no video track", path]);
            return 0;
        }
        AVAssetTrack *track = tracks.firstObject;

        AVMutableVideoComposition *composition = [AVMutableVideoComposition videoComposition];
        composition.renderSize = CGSizeMake(width, height);
        composition.frameDuration = CMTimeMake(1, 24); /* resample to 24 fps */
        AVMutableVideoCompositionInstruction *instruction =
            [AVMutableVideoCompositionInstruction videoCompositionInstruction];
        instruction.timeRange = CMTimeRangeMake(kCMTimeZero, asset.duration);
        AVMutableVideoCompositionLayerInstruction *layerInstruction =
            [AVMutableVideoCompositionLayerInstruction
                videoCompositionLayerInstructionWithAssetTrack:track];
        [layerInstruction setTransform:h3_video_render_transform(track, width, height)
                                 atTime:kCMTimeZero];
        instruction.layerInstructions = @[layerInstruction];
        composition.instructions = @[instruction];

        NSError *readerError = nil;
        AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:&readerError];
        if (!reader) {
            h3_av_set_error(error, error_size,
                readerError.localizedDescription ? readerError.localizedDescription
                                                  : @"cannot open video for reading");
            return 0;
        }
        AVAssetReaderVideoCompositionOutput *output = [AVAssetReaderVideoCompositionOutput
            assetReaderVideoCompositionOutputWithVideoTracks:@[track]
                                               videoSettings:@{
                (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_24RGB),
            }];
        output.videoComposition = composition;
        if (![reader canAddOutput:output]) {
            h3_av_set_error(error, error_size, @"cannot add video composition output");
            return 0;
        }
        [reader addOutput:output];
        if (![reader startReading]) {
            h3_av_set_error(error, error_size,
                reader.error.localizedDescription ? reader.error.localizedDescription
                                                   : @"cannot start reading video");
            return 0;
        }

        size_t capacity = (size_t)max_frames * frameBytes;
        uint8_t *rgb = malloc(capacity);
        if (!rgb) {
            [reader cancelReading];
            h3_av_set_error(error, error_size, @"out of memory decoding video");
            return 0;
        }
        int frameCount = 0;
        while (frameCount < max_frames) {
            CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
            if (!sampleBuffer) break;
            CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
            if (!imageBuffer) {
                CFRelease(sampleBuffer);
                continue;
            }
            CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
            const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
            size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
            uint8_t *dest = rgb + (size_t)frameCount * frameBytes;
            size_t rowBytes = (size_t)width * 3;
            for (int y = 0; y < height; y++)
                memcpy(dest + (size_t)y * rowBytes, base + (size_t)y * bytesPerRow, rowBytes);
            CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
            CFRelease(sampleBuffer);
            frameCount++;
        }
        if (reader.status == AVAssetReaderStatusFailed) {
            free(rgb);
            h3_av_set_error(error, error_size,
                reader.error.localizedDescription ? reader.error.localizedDescription
                                                   : @"video decode failed");
            return 0;
        }
        [reader cancelReading];
        if (frameCount < 5) {
            free(rgb);
            h3_av_set_error(error, error_size,
                @"reference videos require at least 5 decoded frames");
            return 0;
        }
        while (frameCount % 17 != 5) frameCount--;
        if ((size_t)frameCount > SIZE_MAX / 3 / area ||
            (size_t)frameCount * 3 * area > SIZE_MAX / sizeof(float)) {
            free(rgb);
            h3_av_set_error(error, error_size, @"converted video size overflows");
            return 0;
        }
        size_t values = (size_t)frameCount * 3 * area;
        float *channelMajor = malloc(values * sizeof(*channelMajor));
        if (!channelMajor) {
            free(rgb);
            h3_av_set_error(error, error_size, @"out of memory converting video");
            return 0;
        }
        const float scale = 1.0f / 255.0f;
        for (int time = 0; time < frameCount; time++)
            for (size_t pixel = 0; pixel < area; pixel++)
                for (size_t channel = 0; channel < 3; channel++) {
                    size_t sourceIndex = ((size_t)time * area + pixel) * 3 + channel;
                    size_t destination = (channel * (size_t)frameCount + (size_t)time) * area + pixel;
                    channelMajor[destination] = (float)rgb[sourceIndex] * scale;
                }
        free(rgb);
        *pixels = channelMajor;
        *frames = frameCount;
        return 1;
    }
}

/* ---- audio decode -------------------------------------------------------
 *
 * AVAssetReaderTrackOutput transcodes on the fly when outputSettings ask for
 * a format other than the source's - requesting 32 kHz stereo interleaved
 * float32 directly here does the resample/downmix in one step, no separate
 * AVAudioConverter needed. */

int h3_av_read_audio_f32(const char *path, int max_samples,
                         int truncate_at_limit,
                         float **pcm, int *samples,
                         char *error, size_t error_size) {
    enum { AUDIO_RATE = 32000, AUDIO_CHANNELS = 2, MIN_SAMPLES = 64000 };
    if (error && error_size) error[0] = '\0';
    if (pcm) *pcm = NULL;
    if (samples) *samples = 0;
    if (!path || !*path || !pcm || !samples || max_samples < MIN_SAMPLES ||
        max_samples > AUDIO_RATE * 15 ||
        (truncate_at_limit != 0 && truncate_at_limit != 1)) {
        h3_av_set_error(error, error_size, @"invalid audio input arguments");
        return 0;
    }
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
        #pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        /* Synchronous by design: this whole C ABI is synchronous, and for a
         * local file tracksWithMediaType: already blocks until track info is
         * available - the async replacement would need the same blocking
         * wrapper here for no behavioral difference. */
        NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeAudio];
        #pragma clang diagnostic pop
        if (tracks.count == 0) {
            h3_av_set_error(error, error_size,
                [NSString stringWithFormat:@"%s has no audio track", path]);
            return 0;
        }
        NSError *readerError = nil;
        AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:&readerError];
        if (!reader) {
            h3_av_set_error(error, error_size,
                readerError.localizedDescription ? readerError.localizedDescription
                                                  : @"cannot open audio for reading");
            return 0;
        }
        NSDictionary *outputSettings = @{
            AVFormatIDKey: @(kAudioFormatLinearPCM),
            AVSampleRateKey: @(AUDIO_RATE),
            AVNumberOfChannelsKey: @(AUDIO_CHANNELS),
            AVLinearPCMBitDepthKey: @32,
            AVLinearPCMIsFloatKey: @YES,
            AVLinearPCMIsNonInterleaved: @NO,
            AVLinearPCMIsBigEndianKey: @NO,
        };
        AVAssetReaderTrackOutput *output = [AVAssetReaderTrackOutput
            assetReaderTrackOutputWithTrack:tracks.firstObject outputSettings:outputSettings];
        if (![reader canAddOutput:output]) {
            h3_av_set_error(error, error_size, @"cannot add audio output");
            return 0;
        }
        [reader addOutput:output];
        if (![reader startReading]) {
            h3_av_set_error(error, error_size,
                reader.error.localizedDescription ? reader.error.localizedDescription
                                                   : @"cannot start reading audio");
            return 0;
        }

        size_t frameStride = AUDIO_CHANNELS * sizeof(float);
        size_t capacitySamples = (size_t)max_samples;
        /* Standalone clips (truncate_at_limit == 0) must be allowed to
         * exceed max_samples by at least one sample so the overflow can be
         * detected and reported, rather than silently clipped. */
        size_t capacity = (capacitySamples + 1) * frameStride;
        uint8_t *interleavedBytes = malloc(capacity);
        if (!interleavedBytes) {
            [reader cancelReading];
            h3_av_set_error(error, error_size, @"out of memory decoding audio");
            return 0;
        }
        size_t exactCapacity = capacitySamples * frameStride;
        size_t filledBytes = 0;
        int overflow = 0;
        while (1) {
            CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
            if (!sampleBuffer) break;
            CMBlockBufferRef blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer);
            if (blockBuffer) {
                size_t length = CMBlockBufferGetDataLength(blockBuffer);
                if (truncate_at_limit) {
                    /* Cap at exactly max_samples - never the +1 overflow-
                     * detection headroom below, which is only meaningful
                     * for the non-truncating case. */
                    size_t roomLeft = exactCapacity > filledBytes ? exactCapacity - filledBytes : 0;
                    size_t toCopy = length < roomLeft ? length : roomLeft;
                    if (toCopy) {
                        CMBlockBufferCopyDataBytes(blockBuffer, 0, toCopy,
                                                   interleavedBytes + filledBytes);
                        filledBytes += toCopy;
                    }
                    /* Extra source data past max_samples is simply dropped;
                     * keep draining the reader instead of cancelling early
                     * (some AVFoundation versions warn about that). */
                } else if (filledBytes + length > capacity) {
                    overflow = 1;
                    size_t roomLeft = capacity > filledBytes ? capacity - filledBytes : 0;
                    if (roomLeft) {
                        CMBlockBufferCopyDataBytes(blockBuffer, 0, roomLeft,
                                                   interleavedBytes + filledBytes);
                        filledBytes += roomLeft;
                    }
                    /* Non-truncating callers just need to know overflow
                     * happened; the extra bytes themselves are never used. */
                } else {
                    CMBlockBufferCopyDataBytes(blockBuffer, 0, length,
                                               interleavedBytes + filledBytes);
                    filledBytes += length;
                }
            }
            CFRelease(sampleBuffer);
        }
        if (reader.status == AVAssetReaderStatusFailed) {
            free(interleavedBytes);
            h3_av_set_error(error, error_size,
                reader.error.localizedDescription ? reader.error.localizedDescription
                                                   : @"audio decode failed");
            return 0;
        }
        int sampleCount = (int)(filledBytes / frameStride);
        if (!truncate_at_limit && overflow) {
            free(interleavedBytes);
            h3_av_set_error(error, error_size,
                @"reference audio exceeds the 15 second total limit");
            return 0;
        }
        if (sampleCount < MIN_SAMPLES) {
            free(interleavedBytes);
            h3_av_set_error(error, error_size,
                @"reference audio requires at least 2 seconds at 32 kHz");
            return 0;
        }
        const float *interleaved = (const float *)interleavedBytes;
        size_t outputElements = (size_t)sampleCount * AUDIO_CHANNELS;
        float *channelMajor = malloc(outputElements * sizeof(*channelMajor));
        if (!channelMajor) {
            free(interleavedBytes);
            h3_av_set_error(error, error_size, @"out of memory converting audio");
            return 0;
        }
        for (int sample = 0; sample < sampleCount; sample++)
            for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
                channelMajor[(size_t)channel * (size_t)sampleCount + (size_t)sample] =
                    interleaved[(size_t)sample * AUDIO_CHANNELS + (size_t)channel];
        free(interleavedBytes);
        *pcm = channelMajor;
        *samples = sampleCount;
        return 1;
    }
}

int h3_av_read_audio_range_f32(const char *path, double start_seconds,
                               int samples, float **pcm,
                               char *error, size_t error_size) {
    enum { AUDIO_CHANNELS = 2 };
    const int AUDIO_RATE = 32000;
    if (error && error_size) error[0] = '\0';
    if (pcm) *pcm = NULL;
    if (!path || !*path || !pcm || samples < 1 ||
        samples > AUDIO_RATE * 60 || !isfinite(start_seconds) ||
        start_seconds < 0.0) {
        h3_av_set_error(error, error_size, @"invalid forced audio arguments");
        return 0;
    }
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        /* Synchronous for the same reason as h3_av_read_audio_f32. */
        NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeAudio];
        CMTime duration = asset.duration;
#pragma clang diagnostic pop
        if (tracks.count == 0) {
            h3_av_set_error(error, error_size,
                [NSString stringWithFormat:@"%s has no audio track", path]);
            return 0;
        }
        if (CMTIME_IS_NUMERIC(duration) &&
            start_seconds >= CMTimeGetSeconds(duration)) {
            h3_av_set_error(error, error_size,
                [NSString stringWithFormat:
                    @"forced audio start %.3fs is past the end of %s (%.3fs)",
                    start_seconds, path, CMTimeGetSeconds(duration)]);
            return 0;
        }
        NSError *readerError = nil;
        AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:&readerError];
        if (!reader) {
            h3_av_set_error(error, error_size,
                readerError.localizedDescription ? readerError.localizedDescription
                                                  : @"cannot open audio for reading");
            return 0;
        }
        NSDictionary *outputSettings = @{
            AVFormatIDKey: @(kAudioFormatLinearPCM),
            AVSampleRateKey: @(AUDIO_RATE),
            AVNumberOfChannelsKey: @(AUDIO_CHANNELS),
            AVLinearPCMBitDepthKey: @32,
            AVLinearPCMIsFloatKey: @YES,
            AVLinearPCMIsNonInterleaved: @NO,
            AVLinearPCMIsBigEndianKey: @NO,
        };
        AVAssetReaderTrackOutput *output = [AVAssetReaderTrackOutput
            assetReaderTrackOutputWithTrack:tracks.firstObject outputSettings:outputSettings];
        if (![reader canAddOutput:output]) {
            h3_av_set_error(error, error_size, @"cannot add audio output");
            return 0;
        }
        [reader addOutput:output];
        /* Read from the start with a little slack before the wanted range,
         * then place samples by their presentation time: a compressed
         * track's first decoded buffer need not begin exactly at the
         * requested time. */
        double lead = start_seconds > 0.5 ? 0.5 : start_seconds;
        reader.timeRange = CMTimeRangeMake(
            CMTimeMakeWithSeconds(start_seconds - lead, AUDIO_RATE),
            CMTimeMakeWithSeconds(lead + (double)samples / AUDIO_RATE + 0.5,
                                  AUDIO_RATE));
        if (![reader startReading]) {
            h3_av_set_error(error, error_size,
                reader.error.localizedDescription ? reader.error.localizedDescription
                                                   : @"cannot start reading audio");
            return 0;
        }
        size_t outputElements = (size_t)samples * AUDIO_CHANNELS;
        float *channelMajor = calloc(outputElements, sizeof(*channelMajor));
        if (!channelMajor) {
            [reader cancelReading];
            h3_av_set_error(error, error_size, @"out of memory decoding audio");
            return 0;
        }
        int64_t first = (int64_t)llround(start_seconds * AUDIO_RATE);
        int64_t written = 0;
        float scratch[4096 * AUDIO_CHANNELS];
        while (1) {
            CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
            if (!sampleBuffer) break;
            CMBlockBufferRef blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer);
            CMTime when = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
            if (blockBuffer && CMTIME_IS_NUMERIC(when)) {
                int64_t position = (int64_t)llround(CMTimeGetSeconds(when) *
                                                    AUDIO_RATE);
                size_t length = CMBlockBufferGetDataLength(blockBuffer);
                size_t frameStride = AUDIO_CHANNELS * sizeof(float);
                size_t frames = length / frameStride;
                for (size_t done = 0; done < frames;) {
                    size_t chunk = frames - done < 4096 ? frames - done : 4096;
                    CMBlockBufferCopyDataBytes(blockBuffer, done * frameStride,
                                               chunk * frameStride, scratch);
                    for (size_t frame = 0; frame < chunk; frame++) {
                        int64_t target = position + (int64_t)(done + frame) -
                                         first;
                        if (target < 0 || target >= samples) continue;
                        for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
                            channelMajor[(size_t)channel * (size_t)samples +
                                         (size_t)target] =
                                scratch[frame * AUDIO_CHANNELS + channel];
                        if (target + 1 > written) written = target + 1;
                    }
                    done += chunk;
                }
            }
            CFRelease(sampleBuffer);
        }
        if (reader.status == AVAssetReaderStatusFailed) {
            free(channelMajor);
            h3_av_set_error(error, error_size,
                reader.error.localizedDescription ? reader.error.localizedDescription
                                                   : @"audio decode failed");
            return 0;
        }
        if (written == 0) {
            free(channelMajor);
            h3_av_set_error(error, error_size,
                [NSString stringWithFormat:
                    @"no audio decoded from %s at %.3fs", path, start_seconds]);
            return 0;
        }
        for (size_t index = 0; index < outputElements; index++) {
            float value = channelMajor[index];
            channelMajor[index] = value > 1.0f ? 1.0f :
                                  value < -1.0f ? -1.0f : value;
        }
        *pcm = channelMajor;
        return 1;
    }
}
