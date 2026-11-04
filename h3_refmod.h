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
    char source_shape[64];
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

#ifdef __cplusplus
}
#endif

#endif
