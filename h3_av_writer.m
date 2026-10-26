/* Encodes the generated RGB24 frames (and, when present, the generated
 * stereo PCM soundtrack) straight through AVFoundation/VideoToolbox.
 *
 * Settings mirror Draw Things' VideoExporter (H.264 High@4.1, an average
 * bitrate floor, AAC audio at up to 192 kbps) rather than a CRF-style
 * constant-quality target, since AVFoundation's own H.264 encoder wrapper
 * doesn't expose one.
 *
 * Frames still arrive as CPU-side RGB24 (h3_video_vae.c already reads the
 * decoded tensor back from the GPU before this point), so this isn't a
 * Metal-texture zero-copy path - it replaces the external process and pipe
 * with a single CVPixelBuffer copy per frame straight into AVAssetWriter. */
#import <AVFoundation/AVFoundation.h>
#import <CoreAudio/CoreAudioTypes.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include "h3_av_writer.h"
#include <stdlib.h>
#include <string.h>

static void h3_av_set_error(char *error, size_t error_size, NSString *message) {
    if (!error || !error_size || !message) return;
    strncpy(error, message.UTF8String, error_size - 1);
    error[error_size - 1] = '\0';
}

/* Every MP4 says what it is (MiniMax H3 Community License §III.3(b)
 * encourages an AI-generation identifier on generated files): a
 * description and the software that made it, in the file's metadata. Not
 * a visible label - publishing still needs a clear disclosure (its
 * Acceptable Use Policy, item 12). */
static AVMetadataItem *h3_av_metadata_item(AVMetadataKeySpace space,
                                            NSString *key, NSString *value) {
    AVMutableMetadataItem *item = [AVMutableMetadataItem metadataItem];
    item.keySpace = space;
    item.key = key;
    item.value = value;
    return item;
}

/* In the common keys (MP4 user data, read by AVFoundation) and the iTunes
 * ones (description, comment, encoder - what ffmpeg and most players show). */
static NSArray<AVMetadataItem *> *h3_av_ai_metadata(void) {
    NSString *label = @"AI-generated with MiniMax H3";
    NSString *software = @"h3.c (MiniMax H3)";
    return @[
        h3_av_metadata_item(AVMetadataKeySpaceCommon, AVMetadataCommonKeyDescription, label),
        h3_av_metadata_item(AVMetadataKeySpaceCommon, AVMetadataCommonKeySoftware, software),
        h3_av_metadata_item(AVMetadataKeySpaceiTunes, AVMetadataiTunesMetadataKeyDescription, label),
        h3_av_metadata_item(AVMetadataKeySpaceiTunes, AVMetadataiTunesMetadataKeyUserComment, label),
        h3_av_metadata_item(AVMetadataKeySpaceiTunes, AVMetadataiTunesMetadataKeyEncodingTool, software),
    ];
}

static NSDictionary *h3_av_video_settings(int width, int height, int hevc) {
    /* Quality 0.85: against the uncompressed frames, a dark 1920x1088 clip
     * went 47.3 -> 48.2 dB at 2.0 -> 2.4 Mbps and a bright 1600x1088 one
     * 40.5 -> 41.0 dB at 8.0 -> 5.7 Mbps, with far fewer 8-pixel blocks
     * in the dark. */
    if (hevc)
        return @{
            AVVideoCodecKey: AVVideoCodecTypeHEVC,
            AVVideoWidthKey: @(width),
            AVVideoHeightKey: @(height),
            AVVideoCompressionPropertiesKey: @{
                AVVideoQualityKey: @0.85,
                AVVideoMaxKeyFrameIntervalKey: @30,
                AVVideoAllowFrameReorderingKey: @YES,
            },
        };
    int64_t bitrate = (int64_t)width * (int64_t)height * 5;
    if (bitrate < 9500000) bitrate = 9500000;
    /* Level 4.1 caps a frame at 8192 macroblocks (1920x1088 fits); larger
     * upscaled outputs such as 1536x1536 let the encoder pick the level. */
    int64_t macroblocks = (int64_t)((width + 15) / 16) * ((height + 15) / 16);
    NSString *level = macroblocks > 8192 ? AVVideoProfileLevelH264HighAutoLevel
                                         : AVVideoProfileLevelH264High41;
    return @{
        AVVideoCodecKey: AVVideoCodecTypeH264,
        AVVideoWidthKey: @(width),
        AVVideoHeightKey: @(height),
        AVVideoCompressionPropertiesKey: @{
            AVVideoAverageBitRateKey: @(bitrate),
            AVVideoProfileLevelKey: level,
            AVVideoMaxKeyFrameIntervalKey: @30,
            AVVideoAllowFrameReorderingKey: @YES,
        },
    };
}

static NSDictionary *h3_av_pixel_buffer_attributes(int width, int height) {
    return @{
        (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (id)kCVPixelBufferWidthKey: @(width),
        (id)kCVPixelBufferHeightKey: @(height),
    };
}

/* Opaque handle definition (declared in h3_av_writer.h). Every Objective-C
 * object is kept alive by an explicit __bridge_retained reference so this
 * struct can be malloc'd/free'd like any other C value; h3_writer_teardown
 * balances each one with CFBridgingRelease. */
struct h3_av_writer {
    void *writer;   /* AVAssetWriter*        */
    void *video_input;  /* AVAssetWriterInput*   */
    void *adaptor;  /* AVAssetWriterInputPixelBufferAdaptor* */
    int width;
    int height;
    int fps;
    int64_t frame_index;
    int failed;
};

/* Builds one CMSampleBuffer covering the whole (already fully known) PCM
 * track and appends+finishes the audio input immediately, as soon as the
 * writer opens, independent of how video chunks arrive afterwards. */
static int h3_av_append_whole_audio_track(
        AVAssetWriterInput *audioInput, CMAudioFormatDescriptionRef sourceFormat,
        const float *pcm, int samples, int channels, int sample_rate,
        NSString **errorOut) {
    size_t frameBytes = sizeof(float) * (size_t)channels;
    size_t byteCount = (size_t)samples * frameBytes;
    float *interleaved = malloc(byteCount);
    if (!interleaved) {
        *errorOut = @"out of memory interleaving generated PCM";
        return 0;
    }
    for (int sample = 0; sample < samples; sample++)
        for (int channel = 0; channel < channels; channel++)
            interleaved[(size_t)sample * (size_t)channels + (size_t)channel] =
                pcm[(size_t)channel * (size_t)samples + (size_t)sample];

    CMBlockBufferRef blockBuffer = NULL;
    OSStatus blockStatus = CMBlockBufferCreateWithMemoryBlock(
        kCFAllocatorDefault, NULL, byteCount, kCFAllocatorDefault, NULL, 0,
        byteCount, 0, &blockBuffer);
    if (blockStatus != kCMBlockBufferNoErr || !blockBuffer) {
        free(interleaved);
        *errorOut = @"cannot allocate audio block buffer";
        return 0;
    }
    CMBlockBufferReplaceDataBytes(interleaved, blockBuffer, 0, byteCount);
    free(interleaved);

    CMSampleTimingInfo timing = {
        .duration = CMTimeMake(1, sample_rate),
        .presentationTimeStamp = kCMTimeZero,
        .decodeTimeStamp = kCMTimeInvalid,
    };
    const size_t sampleSize = frameBytes;
    CMSampleBufferRef sampleBuffer = NULL;
    OSStatus sbStatus = CMSampleBufferCreate(
        kCFAllocatorDefault, blockBuffer, true, NULL, NULL, sourceFormat,
        (CMItemCount)samples, 1, &timing, 1, &sampleSize, &sampleBuffer);
    CFRelease(blockBuffer);
    if (sbStatus != noErr || !sampleBuffer) {
        *errorOut = @"cannot build audio sample buffer";
        return 0;
    }
    BOOL appended = [audioInput appendSampleBuffer:sampleBuffer];
    CFRelease(sampleBuffer);
    [audioInput markAsFinished];
    if (!appended) *errorOut = @"cannot append audio sample buffer";
    return appended;
}

/* Shared setup for all three public entry points below. pcm == NULL opens a
 * video-only writer (no audio track at all), matching h3_av_write_rgb24's
 * silent-video behavior; the public h3_av_writer_open keeps requiring pcm,
 * since every caller of the streaming writer already has the whole
 * soundtrack decoded up front. */
static h3_av_writer *h3_open_writer(
        const char *path, int width, int height, int fps,
        const float *pcm, int samples, int channels, int sample_rate,
        int hevc, char *error, size_t error_size) {
    @autoreleasepool {
        NSString *nsPath = [NSString stringWithUTF8String:path];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:nsPath error:nil];
        NSString *parent = [nsPath stringByDeletingLastPathComponent];
        if (parent.length) {
            NSError *dirError = nil;
            if (![fm createDirectoryAtPath:parent withIntermediateDirectories:YES
                                 attributes:nil error:&dirError] &&
                ![fm fileExistsAtPath:parent]) {
                h3_av_set_error(error, error_size, dirError.localizedDescription ?
                    dirError.localizedDescription : @"cannot create output directory");
                return NULL;
            }
        }
        NSError *nsError = nil;
        AVAssetWriter *writer = [[AVAssetWriter alloc]
            initWithURL:[NSURL fileURLWithPath:nsPath]
               fileType:AVFileTypeMPEG4 error:&nsError];
        if (!writer) {
            h3_av_set_error(error, error_size, nsError.localizedDescription ?
                nsError.localizedDescription : @"cannot create AVAssetWriter");
            return NULL;
        }
        writer.shouldOptimizeForNetworkUse = YES;
        writer.metadata = h3_av_ai_metadata();

        AVAssetWriterInput *videoInput = [AVAssetWriterInput
            assetWriterInputWithMediaType:AVMediaTypeVideo
                            outputSettings:h3_av_video_settings(width, height, hevc)];
        videoInput.expectsMediaDataInRealTime = NO;
        if (![writer canAddInput:videoInput]) {
            h3_av_set_error(error, error_size, @"cannot add video input to AVAssetWriter");
            return NULL;
        }
        [writer addInput:videoInput];
        AVAssetWriterInputPixelBufferAdaptor *adaptor = [AVAssetWriterInputPixelBufferAdaptor
            assetWriterInputPixelBufferAdaptorWithAssetWriterInput:videoInput
                                        sourcePixelBufferAttributes:h3_av_pixel_buffer_attributes(width, height)];

        AVAssetWriterInput *audioInput = nil;
        CMAudioFormatDescriptionRef sourceFormat = NULL;
        if (pcm) {
            AudioStreamBasicDescription asbd = {0};
            asbd.mSampleRate = sample_rate;
            asbd.mFormatID = kAudioFormatLinearPCM;
            asbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
            asbd.mChannelsPerFrame = (UInt32)channels;
            asbd.mBitsPerChannel = 32;
            asbd.mBytesPerFrame = 4u * (UInt32)channels;
            asbd.mFramesPerPacket = 1;
            asbd.mBytesPerPacket = asbd.mBytesPerFrame;
            CMAudioFormatDescriptionCreate(kCFAllocatorDefault, &asbd, 0, NULL,
                                           0, NULL, NULL, &sourceFormat);
            /* Matches draw-things-cli's MP4 AAC bitrate formula. */
            int64_t audioBitrate = (int64_t)sample_rate * 6;
            if (audioBitrate > 192000) audioBitrate = 192000;
            NSDictionary *audioSettings = @{
                AVFormatIDKey: @(kAudioFormatMPEG4AAC),
                AVSampleRateKey: @(sample_rate),
                AVNumberOfChannelsKey: @(channels),
                AVEncoderBitRateKey: @(audioBitrate),
            };
            audioInput = [AVAssetWriterInput
                assetWriterInputWithMediaType:AVMediaTypeAudio
                                outputSettings:audioSettings
                              sourceFormatHint:sourceFormat];
            audioInput.expectsMediaDataInRealTime = NO;
            if (![writer canAddInput:audioInput]) {
                CFRelease(sourceFormat);
                h3_av_set_error(error, error_size, @"cannot add audio input to AVAssetWriter");
                return NULL;
            }
            [writer addInput:audioInput];
        }

        if (![writer startWriting]) {
            if (sourceFormat) CFRelease(sourceFormat);
            h3_av_set_error(error, error_size, writer.error.localizedDescription ?
                writer.error.localizedDescription : @"cannot start AVAssetWriter");
            return NULL;
        }
        [writer startSessionAtSourceTime:kCMTimeZero];

        if (pcm) {
            NSString *audioError = nil;
            int ok = h3_av_append_whole_audio_track(
                audioInput, sourceFormat, pcm, samples, channels, sample_rate,
                &audioError);
            CFRelease(sourceFormat);
            if (!ok) {
                h3_av_set_error(error, error_size, audioError);
                [writer cancelWriting];
                return NULL;
            }
        }

        h3_av_writer *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            h3_av_set_error(error, error_size, @"out of memory creating writer context");
            [writer cancelWriting];
            return NULL;
        }
        ctx->writer = (__bridge_retained void *)writer;
        ctx->video_input = (__bridge_retained void *)videoInput;
        ctx->adaptor = (__bridge_retained void *)adaptor;
        ctx->width = width;
        ctx->height = height;
        ctx->fps = fps;
        return ctx;
    }
}

/* Appends frame_count consecutive RGB24 frames starting at `frames`,
 * converting each into a BGRA CVPixelBuffer drawn from the adaptor's pool. */
static int h3_append_frames(h3_av_writer *ctx, const uint8_t *frames,
                            int frame_count, char *error, size_t error_size) {
    @autoreleasepool {
        AVAssetWriterInput *videoInput = (__bridge AVAssetWriterInput *)ctx->video_input;
        AVAssetWriterInputPixelBufferAdaptor *adaptor =
            (__bridge AVAssetWriterInputPixelBufferAdaptor *)ctx->adaptor;
        size_t frameBytes = (size_t)ctx->width * (size_t)ctx->height * 3;
        for (int index = 0; index < frame_count; index++) {
            int spins = 0;
            while (!videoInput.readyForMoreMediaData) {
                if (++spins > 100000) {
                    ctx->failed = 1;
                    h3_av_set_error(error, error_size,
                        @"AVAssetWriter video input never became ready");
                    return 0;
                }
                [NSThread sleepForTimeInterval:0.001];
            }
            CVPixelBufferRef pixelBuffer = NULL;
            CVReturn status = CVPixelBufferPoolCreatePixelBuffer(
                NULL, adaptor.pixelBufferPool, &pixelBuffer);
            if (status != kCVReturnSuccess || !pixelBuffer) {
                ctx->failed = 1;
                h3_av_set_error(error, error_size, @"cannot allocate pixel buffer");
                return 0;
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, 0);
            uint8_t *dst = (uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);
            size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
            const uint8_t *src = frames + (size_t)index * frameBytes;
            for (int y = 0; y < ctx->height; y++) {
                const uint8_t *srcRow = src + (size_t)y * (size_t)ctx->width * 3;
                uint8_t *dstRow = dst + (size_t)y * bytesPerRow;
                for (int x = 0; x < ctx->width; x++) {
                    dstRow[x * 4 + 0] = srcRow[x * 3 + 2]; /* B */
                    dstRow[x * 4 + 1] = srcRow[x * 3 + 1]; /* G */
                    dstRow[x * 4 + 2] = srcRow[x * 3 + 0]; /* R */
                    dstRow[x * 4 + 3] = 255;
                }
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
            CMTime pts = CMTimeMake(ctx->frame_index, ctx->fps);
            BOOL appended = [adaptor appendPixelBuffer:pixelBuffer withPresentationTime:pts];
            CVPixelBufferRelease(pixelBuffer);
            if (!appended) {
                ctx->failed = 1;
                AVAssetWriter *writer = (__bridge AVAssetWriter *)ctx->writer;
                h3_av_set_error(error, error_size, writer.error.localizedDescription ?
                    writer.error.localizedDescription : @"cannot append video frame");
                return 0;
            }
            ctx->frame_index++;
        }
        return 1;
    }
}

static int h3_close_writer(h3_av_writer *ctx, char *error, size_t error_size) {
    int ok = !ctx->failed;
    @autoreleasepool {
        AVAssetWriter *writer = (__bridge AVAssetWriter *)ctx->writer;
        AVAssetWriterInput *videoInput = (__bridge AVAssetWriterInput *)ctx->video_input;
        if (ok) {
            [videoInput markAsFinished];
            dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
            [writer finishWritingWithCompletionHandler:^{
                dispatch_semaphore_signal(semaphore);
            }];
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
            if (writer.status != AVAssetWriterStatusCompleted) {
                ok = 0;
                h3_av_set_error(error, error_size, writer.error.localizedDescription ?
                    writer.error.localizedDescription : @"AVAssetWriter failed to finish");
            }
        } else {
            h3_av_set_error(error, error_size, @"AVFoundation writer failed earlier in the stream");
            [writer cancelWriting];
        }
    }
    CFBridgingRelease(ctx->writer);
    CFBridgingRelease(ctx->video_input);
    CFBridgingRelease(ctx->adaptor);
    free(ctx);
    return ok;
}

int h3_av_write_rgb24(const char *path, const uint8_t *frames,
                      int frame_count, int width, int height, int fps,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !frames || frame_count < 1 || width < 2 ||
        height < 2 || fps < 1 || width % 2 || height % 2) {
        h3_av_set_error(error, error_size, @"invalid AVFoundation RGB output arguments");
        return 0;
    }
    h3_av_writer *ctx = h3_open_writer(path, width, height, fps, NULL, 0, 0, 0,
                                       0, error, error_size);
    if (!ctx) return 0;
    int ok = h3_append_frames(ctx, frames, frame_count, error, error_size);
    return h3_close_writer(ctx, ok ? error : NULL, ok ? error_size : 0) && ok;
}

int h3_av_write_rgb24_f32(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height,
                          int fps, const float *pcm, int samples,
                          int channels, int sample_rate, int hevc,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !frames || frame_count < 1 || width < 2 ||
        height < 2 || fps < 1 || width % 2 || height % 2 || !pcm ||
        samples < 1 || channels < 1 || sample_rate < 1) {
        h3_av_set_error(error, error_size, @"invalid AVFoundation A/V output arguments");
        return 0;
    }
    h3_av_writer *ctx = h3_open_writer(path, width, height, fps, pcm, samples,
                                       channels, sample_rate, hevc, error,
                                       error_size);
    if (!ctx) return 0;
    int ok = h3_append_frames(ctx, frames, frame_count, error, error_size);
    return h3_close_writer(ctx, ok ? error : NULL, ok ? error_size : 0) && ok;
}

h3_av_writer *h3_av_writer_open(const char *path, int width,
                                int height, int fps,
                                const float *pcm, int samples,
                                int channels, int sample_rate, int hevc,
                                char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || width < 2 || height < 2 || fps < 1 ||
        width % 2 || height % 2 || !pcm || samples < 1 || channels < 1 ||
        sample_rate < 1) {
        h3_av_set_error(error, error_size, @"invalid AVFoundation A/V writer arguments");
        return NULL;
    }
    return h3_open_writer(path, width, height, fps, pcm, samples, channels,
                          sample_rate, hevc, error, error_size);
}

int h3_av_writer_write_video(h3_av_writer *writer,
                             const uint8_t *frames, int frame_count,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!writer || !frames || frame_count < 1) {
        h3_av_set_error(error, error_size, @"invalid AVFoundation writer video chunk");
        return 0;
    }
    if (writer->failed) {
        h3_av_set_error(error, error_size,
            @"cannot write video chunk: writer already failed");
        return 0;
    }
    return h3_append_frames(writer, frames, frame_count, error, error_size);
}

int h3_av_writer_close(h3_av_writer *writer,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!writer) return 1;
    return h3_close_writer(writer, error, error_size);
}
