#ifndef H3_REFMOD_H
#define H3_REFMOD_H

/* RefMods: reference latents saved to .safetensors files, in the format of
 * the ComfyUI node pack ComfyUI-MiniMaxH3Mod (MIT; "refmod_meta" header
 * metadata, format versions 2-5). A visual RefMod holds the Ref2VA video
 * VAE's normalized latent, F16 [1,24,T,H,W]; an audio RefMod holds the
 * audio VAE's normalized latent, F16 [1,32,2,T]. Both are exactly what this
 * engine's own encoders produce for a reference, so a RefMod goes into the
 * DiT's reference rows unchanged - the only difference from an ordinary
 * reference is that Qwen is not shown it (no <Picture n> label).
 *
 * A bundle (format 5) carries several members as tensors ref_0..ref_N-1;
 * a single file (formats 2-4) carries one member as "latent". */

#include "h3.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    H3_REFMOD_IMAGE = 1,
    H3_REFMOD_VIDEO = 2,
    H3_REFMOD_AUDIO = 3
} h3_refmod_kind;

typedef enum {
    H3_REFMOD_FULL = 0,       /* the stored latent is the encoder's output ("encode") */
    H3_REFMOD_COMPRESSED = 1  /* pooled to a small grid and refined ("training") */
} h3_refmod_mode;

#define H3_REFMOD_MAX_MEMBERS 256

typedef struct {
    h3_refmod_kind kind;
    int mode;                  /* h3_refmod_mode */
    int latent_t, latent_h, latent_w;   /* audio: latent_t only, h = w = 0 */
    size_t tokens;             /* DiT rows: visual T*(H/2)*(W/2), audio 2*T */
    int format_version;        /* the member's own metadata version (2-4) */
    int optimize_steps;
    int supported;             /* always 1 today; 0 would name a member this engine cannot use */
    char name[128];
    char source[32];
    /* "TxHxW", or one per source joined by " +" for a stack or bundle
     * built from many sources (upstream writes the whole list). */
    char source_shape[1024];
    char pool[96];
    char concept_type[32];
    char description[256];
    char tensor[16];           /* "latent" or "ref_N" */
} h3_refmod_member;

typedef struct {
    int format_version;        /* file level: 2-5 */
    int bundle;                /* 1 for a format-5 bundle */
    char name[128];
    h3_refmod_member *members;
    size_t member_count;       /* 1 for a single-latent file */
} h3_refmod_info;

/* Reads the header only (no latent data). */
int h3_refmod_inspect(const char *path, h3_refmod_info *info,
                      char *error, size_t error_size);
void h3_refmod_info_free(h3_refmod_info *info);

typedef struct {
    h3_refmod_kind kind;
    int time, height, width;   /* visual: channel-major F32 [24,T,H,W] */
    int audio_t;               /* audio: channel-major F32 [32,2,T] */
    float *values;
} h3_refmod_latent;

/* Loads one member (-1 = the single file's "latent"; otherwise the bundle
 * member index) as F32 with `strength` applied: 1 leaves the stored latent
 * as it is, below 1 mixes it toward a blurred copy of itself, as the
 * upstream Apply node does. `meta` (optional) receives the member's
 * metadata. */
int h3_refmod_load(const char *path, int member, float strength,
                   h3_refmod_latent *latent, h3_refmod_member *meta,
                   char *error, size_t error_size);
void h3_refmod_latent_free(h3_refmod_latent *latent);

/* The arithmetic behind strength and compression, public for the tests.
 * All follow PyTorch's definitions: adaptive_avg_pool3d bins and
 * align_corners=False interpolation. Layouts are channel-major
 * [c][t][h][w]. */
int h3_refmod_pool3d(const float *in, int c, int t, int h, int w,
                     int ot, int oh, int ow, float *out);
int h3_refmod_trilinear(const float *in, int c, int t, int h, int w,
                        int ot, int oh, int ow, float *out);
/* Visual: latent = s*latent + (1-s)*blur(latent), blur = trilinear
 * upsample of the (t, max(1,h/8), max(1,w/8)) average pool. */
int h3_refmod_apply_strength(float *latent, int c, int t, int h, int w,
                             float strength);
/* Audio [32,2,T]: each series is pooled to max(1,t/8) along time and
 * linearly interpolated back; t <= 1 is left unchanged. */
int h3_refmod_apply_strength_audio(float *latent, int t, float strength);

/* ---- making RefMods ---------------------------------------------------
 * The upstream extractor's rules, so a file made here reads the same way
 * there, and a Full one reproduces this engine's own reference rows. */

/* The pooling grid of a compressed RefMod: the latent's longer side gets
 * `pool_long_edge` cells, the shorter one its share by aspect, both even
 * and at least 2, never more than the latent itself. */
void h3_refmod_aspect_grid(int pool_long_edge, int latent_h, int latent_w,
                           int *gh, int *gw);
/* The pixel canvas a source is encoded at: shrunk (never enlarged) so its
 * short side is at most `cap_px` (0 = 1024), snapped to multiples of 32,
 * then enlarged to a short side of at least 320. */
int h3_refmod_extract_canvas(int source_w, int source_h, int cap_px,
                             int *w, int *h);
/* Which of a video's `source_frames` (24 fps) a RefMod keeps: at most
 * `latent_frames` (0 = 16) spread evenly, snapped down to 17k+5 of them;
 * fewer than 5 means the first frame alone. `indices` holds at least
 * `latent_frames` (or 16) entries. */
int h3_refmod_extract_frames(int source_frames, int latent_frames,
                             int *indices, int *count);
/* Adam (lr 0.02, PyTorch defaults otherwise) on the small latent so that
 * its trilinear upsample matches `full` in mean squared error, for
 * `steps` iterations. Progress phase "refmod refine"; a nonzero callback
 * return or *cancel stops it with an error. */
int h3_refmod_refine(float *small, int c, int st, int sh, int sw,
                     const float *full, int t, int h, int w,
                     int steps, float lr,
                     h3_progress_callback on_progress, void *opaque,
                     const volatile int *cancel, char *error, size_t error_size);

typedef struct {
    h3_refmod_kind kind;
    int mode;                  /* h3_refmod_mode */
    int optimize_steps;
    const char *name;
    const char *source;        /* "image", "video", "audio" */
    const char *source_shape;  /* "TxHxW" of the full latent */
    const char *pool;          /* "TxGHxGW" or the full-res note */
    const char *description;   /* optional */
    const char *concept_type;  /* NULL = "generic" */
    const char *const *tags;   /* optional */
    size_t tag_count;
} h3_refmod_meta;
/* Writes a single-latent file (format 4): the latent as F16 "latent"
 * [1,24,T,H,W] or [1,32,2,T] and refmod_meta in the header, through a
 * .partial file renamed into place. */
int h3_refmod_write(const char *path, const h3_refmod_latent *latent,
                    const h3_refmod_meta *meta, char *error, size_t error_size);

typedef struct {
    const char *model_dir;     /* the MiniMax-H3 folder; Ref2VA's VAEs are used */
    const char *source_path;   /* an image, a video or an audio file */
    const char *output_path;   /* the .safetensors to write */
    const char *name;          /* refmod_meta.name (NULL = the file's stem) */
    h3_refmod_kind kind;       /* what source_path is */
    int mode;                  /* h3_refmod_mode; audio is always full */
    int resolution_cap;        /* short-edge cap in pixels, 0 = 1024 */
    int latent_frames;         /* video: frames kept, 0 = 16 */
    int pool_long_edge;        /* compressed: 0 = 16 */
    int optimize_steps;        /* compressed: Adam steps, 0 = none */
    int max_seconds;           /* audio: 0 = 15 */
    const char *description;   /* optional */
    const volatile int *cancel_requested;   /* optional */
    /* Two or more still images stacked into one RefMod, as upstream's
     * "stack" source: each is encoded on its own and the latents are
     * joined along time (one latent frame per image), so the file is
     * kind "video" with latent_t = image_count. Full: the first image sets
     * a shared canvas (short edge <= resolution_cap) and the others are
     * scaled to cover it and cropped at the centre. Compressed: each image
     * is encoded at its own size and pooled to the first image's grid,
     * then refined against its own encoding. kind must be
     * H3_REFMOD_IMAGE; source_path is ignored. 0 or 1 = a single source. */
    const char *const *image_paths;
    size_t image_count;
} h3_refmod_extract_params;
/* The most images one stacked RefMod takes. */
#define H3_REFMOD_MAX_STACK 32
/* Encodes the source with the Ref2VA VAE and writes the RefMod. Progress
 * phases: "refmod read", "video VAE encoder" / "audio VAE encoder",
 * "refmod refine" (compressed), "refmod write"; a stack also reports
 * "refmod stack" (images done / images) before each image. */
int h3_refmod_extract(const h3_refmod_extract_params *params,
                      h3_progress_callback on_progress, void *opaque,
                      char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
