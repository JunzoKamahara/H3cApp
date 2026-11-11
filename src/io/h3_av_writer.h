#ifndef H3_AV_WRITER_H
#define H3_AV_WRITER_H

#include <stddef.h>
#include <stdint.h>

int h3_av_write_rgb24(const char *path, const uint8_t *frames,
                      int frame_count, int width, int height, int fps,
                      char *error, size_t error_size);

/* Encode RGB24 video and channel-major F32 PCM through concurrent pipelines.
 * No intermediate uncompressed media file is created. hevc: HEVC at a
 * constant quality (quality, 0-1; 0 = 0.85) instead of H.264 at an
 * average bitrate (h3_params video_hevc, video_quality). */
int h3_av_write_rgb24_f32(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height,
                          int fps, const float *pcm, int samples,
                          int channels, int sample_rate, int hevc,
                          float quality, char *error, size_t error_size);

typedef struct h3_av_writer h3_av_writer;

/* Incremental counterpart to h3_av_write_rgb24_f32: the whole PCM track is
 * already known up front (audio decodes before video in the generation
 * pipeline) and starts writing immediately, but video frames are handed
 * over in chunks via h3_av_writer_write_video as they're produced instead of
 * requiring the whole video to already be in memory. */
h3_av_writer *h3_av_writer_open(const char *path, int width,
                                int height, int fps,
                                const float *pcm, int samples,
                                int channels, int sample_rate, int hevc,
                                float quality, char *error, size_t error_size);
int h3_av_writer_write_video(h3_av_writer *writer,
                             const uint8_t *frames, int frame_count,
                             char *error, size_t error_size);
/* Ends the video stream, waits for the writer to finish, and frees it either
 * way - always call this exactly once for a writer that was opened, even
 * after a write_video failure. */
int h3_av_writer_close(h3_av_writer *writer,
                       char *error, size_t error_size);

#endif
