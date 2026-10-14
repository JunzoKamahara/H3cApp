#ifndef H3_AV_READER_H
#define H3_AV_READER_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_IMAGE_FIT_STRETCH = 0,
    H3_IMAGE_FIT_COVER = 1
} h3_image_fit;

/* Inspect the first visual stream (image or video) without decoding it. */
int h3_av_visual_size(const char *path, int *width, int *height,
                      char *error, size_t error_size);

/* Decode one visual stream. The caller owns channel-major F32 [3,height,width]
 * RGB in [0,1]. */
int h3_av_read_image_f32(const char *path, int width, int height,
                         h3_image_fit fit, float **pixels,
                         char *error, size_t error_size);

/* Decode a 24 fps visual stream to channel-major F32 [3,T,H,W] in [0,1].
 * The returned frame count is trimmed down to the released 5+17k cadence. */
int h3_av_read_video_f32(const char *path, int width, int height,
                         int max_frames, float **pixels, int *frames,
                         char *error, size_t error_size);

/* Decode the first audio stream as channel-major stereo F32 at 32 kHz.
 * max_samples bounds allocation. truncate_at_limit is used for a video's
 * soundtrack; standalone clips report an error instead of silently trimming. */
int h3_av_read_audio_f32(const char *path, int max_samples,
                         int truncate_at_limit,
                         float **pcm, int *samples,
                         char *error, size_t error_size);

/* Decode exactly `samples` stereo 32 kHz frames of the first audio stream,
 * starting `start_seconds` into it, as channel-major F32 [2,samples]. A
 * stream that ends early is padded with silence; one that ends before
 * start_seconds is an error. Used for forced audio, which may come from a
 * long track. */
int h3_av_read_audio_range_f32(const char *path, double start_seconds,
                               int samples, float **pcm,
                               char *error, size_t error_size);

#endif
