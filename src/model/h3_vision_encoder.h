#ifndef H3_VISION_ENCODER_H
#define H3_VISION_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

/* H3's Qwen3-VL-32B; P-H3-Image's 4B gives 2560 (h3_vision_output.width). */
#define H3_VISION_OUTPUT_WIDTH 5120u
#define H3_VISION_DEEPSTACKS 3u

typedef struct {
    int grid_h;
    int grid_w;
    size_t tokens;
    uint16_t *merged;
    uint16_t *deepstack[H3_VISION_DEEPSTACKS];
    h3_gpu_stats gpu_stats;
    /* Values per row: 5120 (H3's Qwen3-VL-32B) or 2560 (P-H3-Image's 4B). */
    size_t width;
} h3_vision_output;

typedef void (*h3_vision_progress)(int completed_layers, int total_layers,
                                   void *opaque);

/* Encode one image or one two-frame video block. Pixels are F32 [T,3,H,W] in
 * [0,1], T is 1 or 2, and H/W are multiples of 32. The output owns BF16 Qwen
 * presentation rows and three same-shaped deepstack additions. */
int h3_vision_encode_bf16(const char *weight_directory,
                          const char *shader_source_path,
                          const float *pixels, int frames,
                          int height, int width,
                          h3_vision_progress progress, void *progress_opaque,
                          h3_vision_output *output,
                          char *error, size_t error_size);
void h3_vision_output_free(h3_vision_output *output);

#endif
