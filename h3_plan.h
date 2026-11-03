/* The arithmetic behind h3_plan_generation (h3.h): memory per stage, the
 * verdict and its message, from token counts and reference geometry that
 * h3.c has already worked out the way h3_generate will. Kept apart from
 * reading media so tests can drive it directly. */
#ifndef H3_PLAN_H
#define H3_PLAN_H

#include "h3.h"

#include <stddef.h>
#include <stdint.h>

/* One image or video the video VAE and Qwen's vision tower will see. */
typedef struct {
    int width;
    int height;
    int frames;        /* 1 for an image */
} h3_plan_visual;

typedef struct {
    uint64_t memory_bytes;     /* RAM */
    int ssd_streaming;         /* h3_params.ssd_streaming */
    int full_sdpa;             /* the GPU runs dense attention in one call */
    int text_prefetch_depth;   /* Qwen layers read ahead (H3_QWEN_PREFETCH_DEPTH) */
    /* DiT weights held for the whole denoise beyond what SSD streaming and
     * the int8 attention cache keep (both stream blocks): the resident
     * mode's whole transformer. */
    uint64_t dit_resident_bytes;
    size_t text_tokens;
    size_t vision_tokens;
    size_t dit_tokens;
    size_t dit_generated_tokens;   /* the target video and audio rows */
    size_t dit_reference_tokens;   /* Ref2VA reference rows */
    const h3_plan_visual *visuals;
    size_t visual_count;
} h3_plan_input;

/* Qwen's causal attention: its fast kernel keeps one F32 score row per
 * threadgroup in 32 KiB (less 1 KiB of its own), and the MPSGraph fallback
 * past that refuses sequences whose mask would pass 512 MiB (h3_gpu.m). */
#define H3_PLAN_TEXT_FAST_LIMIT 7936
#define H3_PLAN_TEXT_TOKEN_LIMIT 16384

void h3_plan_compute(const h3_plan_input *input, h3_plan *plan);

/* DiT compute of a sequence relative to one of `base` tokens. */
double h3_plan_dit_cost_ratio(size_t tokens, size_t base);

#endif
