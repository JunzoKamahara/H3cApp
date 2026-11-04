#include "h3_refmod.h"
#include "h3_audio_vae.h"
#include "h3_av_reader.h"
#include "h3_json.h"
#include "h3_metal.h"
#include "h3_quant.h"
#include "h3_safetensors.h"
#include "h3_video_encoder.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define H3_REFMOD_META_KEY "refmod_meta"
#define H3_REFMOD_AUDIO_META_KEY "audio_refmod_meta"
#define H3_REFMOD_MAX_SIDECAR (4u * 1024u * 1024u)

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list args;
    va_start(args, format);
    vsnprintf(error, error_size, format, args);
    va_end(args);
}

static void copy_text(char *destination, size_t size, const char *source) {
    if (!source) source = "";
    snprintf(destination, size, "%s", source);
}

/* ---- metadata JSON -------------------------------------------------- */

typedef struct {
    h3_refmod_member member;
    int has_latent_t, has_latent_h, has_latent_w;
    int is_bundle;
    char *members_json;   /* the "members" array, kept for a second pass */
    size_t members_json_length;
} meta_scan;

static int parse_mode(const char *text, int *mode) {
    if (!strcmp(text, "encode") || !strcmp(text, "full") ||
        !strcmp(text, "Full Reference")) { *mode = H3_REFMOD_FULL; return 1; }
    if (!strcmp(text, "training") || !strcmp(text, "pooled") ||
        !strcmp(text, "Compressed Reference")) {
        *mode = H3_REFMOD_COMPRESSED; return 1;
    }
    return 0;
}

/* Reads one string value into `destination`, or skips a non-string. */
static int take_text(h3_json_cursor *cursor, char *destination, size_t size) {
    if (h3_json_peek(cursor) != '"') return h3_json_skip(cursor);
    char *text = h3_json_string(cursor);
    if (!text) return 0;
    copy_text(destination, size, text);
    free(text);
    return 1;
}

static int take_int(h3_json_cursor *cursor, int *value, int *present) {
    char next = h3_json_peek(cursor);
    if (next != '-' && (next < '0' || next > '9')) return h3_json_skip(cursor);
    if (!h3_json_int(cursor, value)) return 0;
    if (present) *present = 1;
    return 1;
}

/* Parses one metadata object (a file's top level or a bundle member). The
 * "members" array is captured as raw text for the bundle pass. */
static int scan_meta_object(h3_json_cursor *cursor, meta_scan *scan,
                            int capture_members) {
    if (!h3_json_take(cursor, '{')) return 0;
    if (h3_json_peek(cursor) == '}') { cursor->at++; return 1; }
    for (;;) {
        char *key = h3_json_string(cursor);
        if (!key) return 0;
        int ok = h3_json_take(cursor, ':');
        h3_refmod_member *member = &scan->member;
        if (!ok) {
        } else if (!strcmp(key, "name")) {
            ok = take_text(cursor, member->name, sizeof(member->name));
        } else if (!strcmp(key, "kind")) {
            char kind[32] = "";
            ok = take_text(cursor, kind, sizeof(kind));
            if (ok) {
                if (!strcmp(kind, "image")) member->kind = H3_REFMOD_IMAGE;
                else if (!strcmp(kind, "video")) member->kind = H3_REFMOD_VIDEO;
                else if (!strcmp(kind, "audio")) member->kind = H3_REFMOD_AUDIO;
                else if (!strcmp(kind, "bundle")) scan->is_bundle = 1;
                else ok = h3_json_fail(cursor, "unknown RefMod kind");
            }
        } else if (!strcmp(key, "mode")) {
            char mode[32] = "";
            ok = take_text(cursor, mode, sizeof(mode));
            if (ok && *mode && !parse_mode(mode, &member->mode))
                ok = h3_json_fail(cursor, "unknown RefMod mode");
        } else if (!strcmp(key, "latent_t")) {
            ok = take_int(cursor, &member->latent_t, &scan->has_latent_t);
        } else if (!strcmp(key, "latent_h")) {
            ok = take_int(cursor, &member->latent_h, &scan->has_latent_h);
        } else if (!strcmp(key, "latent_w")) {
            ok = take_int(cursor, &member->latent_w, &scan->has_latent_w);
        } else if (!strcmp(key, "optimize_steps")) {
            ok = take_int(cursor, &member->optimize_steps, NULL);
        } else if (!strcmp(key, "_format_version")) {
            ok = take_int(cursor, &member->format_version, NULL);
        } else if (!strcmp(key, "source")) {
            ok = take_text(cursor, member->source, sizeof(member->source));
        } else if (!strcmp(key, "source_shape")) {
            ok = take_text(cursor, member->source_shape,
                           sizeof(member->source_shape));
        } else if (!strcmp(key, "pool")) {
            ok = take_text(cursor, member->pool, sizeof(member->pool));
        } else if (!strcmp(key, "concept_type")) {
            ok = take_text(cursor, member->concept_type,
                           sizeof(member->concept_type));
        } else if (!strcmp(key, "description")) {
            ok = take_text(cursor, member->description,
                           sizeof(member->description));
        } else if (!strcmp(key, "members") && capture_members) {
            h3_json_ws(cursor);
            const char *start = cursor->at;
            ok = h3_json_skip(cursor);
            if (ok) {
                scan->members_json_length = (size_t)(cursor->at - start);
                scan->members_json = malloc(scan->members_json_length + 1);
                if (!scan->members_json) {
                    ok = h3_json_fail(cursor, "out of memory reading RefMod members");
                } else {
                    memcpy(scan->members_json, start, scan->members_json_length);
                    scan->members_json[scan->members_json_length] = '\0';
                }
            }
        } else {
            ok = h3_json_skip(cursor);
        }
        free(key);
        if (!ok) return 0;
        char next = h3_json_peek(cursor);
        if (next == '}') { cursor->at++; return 1; }
        if (next != ',') return h3_json_fail(cursor, "expected JSON comma");
        cursor->at++;
    }
}

static int scan_meta(const char *json, meta_scan *scan, int capture_members,
                     char *error, size_t error_size) {
    memset(scan, 0, sizeof(*scan));
    scan->member.supported = 1;
    h3_json_cursor cursor = {json, json + strlen(json), error, error_size, 0};
    if (!scan_meta_object(&cursor, scan, capture_members)) {
        free(scan->members_json);
        scan->members_json = NULL;
        return 0;
    }
    return 1;
}

/* The sidecar "<path minus .safetensors>.json" that older extractors wrote
 * next to the file. */
static char *read_sidecar(const char *path) {
    size_t length = strlen(path);
    const char *suffix = ".safetensors";
    size_t suffix_length = strlen(suffix);
    if (length <= suffix_length ||
        strcmp(path + length - suffix_length, suffix)) return NULL;
    char *sidecar = malloc(length + 2);
    if (!sidecar) return NULL;
    memcpy(sidecar, path, length - suffix_length);
    strcpy(sidecar + length - suffix_length, ".json");
    FILE *file = fopen(sidecar, "rb");
    free(sidecar);
    if (!file) return NULL;
    char *text = NULL;
    if (fseek(file, 0, SEEK_END) == 0) {
        long size = ftell(file);
        if (size > 0 && (unsigned long)size <= H3_REFMOD_MAX_SIDECAR &&
            fseek(file, 0, SEEK_SET) == 0) {
            text = malloc((size_t)size + 1);
            if (text) {
                size_t got = fread(text, 1, (size_t)size, file);
                text[got] = '\0';
            }
        }
    }
    fclose(file);
    return text;
}

/* ---- tensor checks --------------------------------------------------- */

static int check_tensor(const h3_st_tensor *tensor, h3_refmod_member *member,
                        int has_t, int has_h, int has_w,
                        char *error, size_t error_size) {
    if (!tensor) {
        fail(error, error_size, "RefMod has no tensor \"%s\"", member->tensor);
        return 0;
    }
    if (tensor->dtype != H3_DTYPE_F16 && tensor->dtype != H3_DTYPE_BF16 &&
        tensor->dtype != H3_DTYPE_F32) {
        fail(error, error_size, "RefMod tensor \"%s\" is %s, not F16, BF16 or F32",
             member->tensor, h3_dtype_name(tensor->dtype));
        return 0;
    }
    if (member->kind == H3_REFMOD_AUDIO) {
        if (tensor->ndim != 4 || tensor->shape[0] != 1 || tensor->shape[1] != 32 ||
            tensor->shape[2] != 2 || tensor->shape[3] < 1 ||
            tensor->shape[3] > 1u << 20) {
            fail(error, error_size,
                 "audio RefMod \"%s\" must be [1,32,2,T]", member->name);
            return 0;
        }
        member->latent_t = (int)tensor->shape[3];
        member->latent_h = member->latent_w = 0;
        member->tokens = (size_t)member->latent_t * 2;
        return 1;
    }
    if (member->kind != H3_REFMOD_IMAGE && member->kind != H3_REFMOD_VIDEO) {
        fail(error, error_size, "RefMod \"%s\" has no kind", member->name);
        return 0;
    }
    if (tensor->ndim != 5 || tensor->shape[0] != 1 || tensor->shape[1] != 24 ||
        tensor->shape[2] < 1 || tensor->shape[3] < 2 || tensor->shape[4] < 2 ||
        tensor->shape[2] > 4096 || tensor->shape[3] > 4096 ||
        tensor->shape[4] > 4096) {
        fail(error, error_size,
             "visual RefMod \"%s\" must be [1,24,T,H,W]", member->name);
        return 0;
    }
    int t = (int)tensor->shape[2], h = (int)tensor->shape[3],
        w = (int)tensor->shape[4];
    if (h % 2 || w % 2) {
        fail(error, error_size,
             "RefMod \"%s\" latent %dx%d is not even-sized", member->name, h, w);
        return 0;
    }
    if (member->kind == H3_REFMOD_IMAGE && t != 1) {
        fail(error, error_size,
             "image RefMod \"%s\" has %d latent frames", member->name, t);
        return 0;
    }
    if ((has_t && member->latent_t != t) || (has_h && member->latent_h != h) ||
        (has_w && member->latent_w != w)) {
        fail(error, error_size,
             "RefMod \"%s\" metadata (%dx%dx%d) disagrees with its tensor (%dx%dx%d)",
             member->name, member->latent_t, member->latent_h,
             member->latent_w, t, h, w);
        return 0;
    }
    member->latent_t = t;
    member->latent_h = h;
    member->latent_w = w;
    member->tokens = (size_t)t * (size_t)(h / 2) * (size_t)(w / 2);
    return 1;
}

/* ---- inspect --------------------------------------------------------- */

static int inspect_header(const h3_st_header *header, const char *path,
                          h3_refmod_info *info, char *error, size_t error_size) {
    memset(info, 0, sizeof(*info));
    const char *json = h3_st_metadata(header, H3_REFMOD_META_KEY);
    if (!json) json = h3_st_metadata(header, H3_REFMOD_AUDIO_META_KEY);
    char *sidecar = NULL;
    if (!json) {
        sidecar = read_sidecar(path);
        json = sidecar;
    }
    if (!json) {
        fail(error, error_size, "%s is not a RefMod file (no refmod_meta)", path);
        return 0;
    }
    meta_scan top;
    char detail[256];
    if (!scan_meta(json, &top, 1, detail, sizeof(detail))) {
        fail(error, error_size, "%s has unreadable RefMod metadata: %s", path,
             detail);
        free(sidecar);
        return 0;
    }
    free(sidecar);
    info->format_version = top.member.format_version;
    copy_text(info->name, sizeof(info->name), top.member.name);
    int ok = 0;
    if (top.is_bundle) {
        info->bundle = 1;
        if (!top.members_json) {
            fail(error, error_size, "%s is a RefMod bundle without members", path);
            goto done;
        }
        h3_json_cursor cursor = {top.members_json,
            top.members_json + top.members_json_length, detail, sizeof(detail), 0};
        if (!h3_json_take(&cursor, '[')) goto malformed;
        size_t capacity = 8;
        info->members = calloc(capacity, sizeof(*info->members));
        if (!info->members) goto memory;
        if (h3_json_peek(&cursor) == ']') {
            fail(error, error_size, "%s is a RefMod bundle without members", path);
            goto done;
        }
        for (;;) {
            if (info->member_count >= H3_REFMOD_MAX_MEMBERS) {
                fail(error, error_size, "%s has more than %d RefMod members",
                     path, H3_REFMOD_MAX_MEMBERS);
                goto done;
            }
            if (info->member_count == capacity) {
                capacity *= 2;
                h3_refmod_member *grown =
                    realloc(info->members, capacity * sizeof(*grown));
                if (!grown) goto memory;
                info->members = grown;
            }
            meta_scan scan;
            memset(&scan, 0, sizeof(scan));
            scan.member.supported = 1;
            if (!scan_meta_object(&cursor, &scan, 0)) goto malformed;
            if (scan.is_bundle) {
                fail(error, error_size, "%s nests a RefMod bundle", path);
                goto done;
            }
            h3_refmod_member *member = &info->members[info->member_count];
            *member = scan.member;
            snprintf(member->tensor, sizeof(member->tensor), "ref_%zu",
                     info->member_count);
            if (!check_tensor(h3_st_find(header, member->tensor), member,
                              scan.has_latent_t, scan.has_latent_h,
                              scan.has_latent_w, error, error_size)) goto done;
            info->member_count++;
            char next = h3_json_peek(&cursor);
            if (next == ']') break;
            if (next != ',') goto malformed;
            cursor.at++;
        }
    } else {
        info->members = calloc(1, sizeof(*info->members));
        if (!info->members) goto memory;
        info->members[0] = top.member;
        copy_text(info->members[0].tensor, sizeof(info->members[0].tensor),
                  "latent");
        if (!check_tensor(h3_st_find(header, "latent"), &info->members[0],
                          top.has_latent_t, top.has_latent_h, top.has_latent_w,
                          error, error_size)) goto done;
        info->member_count = 1;
    }
    ok = 1;
    goto done;
malformed:
    fail(error, error_size, "%s has unreadable RefMod members: %s", path, detail);
    goto done;
memory:
    fail(error, error_size, "out of memory reading RefMod %s", path);
done:
    free(top.members_json);
    if (!ok) h3_refmod_info_free(info);
    return ok;
}

int h3_refmod_inspect(const char *path, h3_refmod_info *info,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!info) return 0;
    memset(info, 0, sizeof(*info));
    if (!path || !*path) {
        fail(error, error_size, "RefMod path is required");
        return 0;
    }
    h3_st_header header;
    char detail[384];
    if (!h3_st_read_header(path, &header, detail, sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        return 0;
    }
    int ok = inspect_header(&header, path, info, error, error_size);
    h3_st_free_header(&header);
    return ok;
}

void h3_refmod_info_free(h3_refmod_info *info) {
    if (!info) return;
    free(info->members);
    memset(info, 0, sizeof(*info));
}

/* ---- arithmetic ------------------------------------------------------ */

/* PyTorch adaptive pooling: output cell i of O covers input
 * [floor(i*I/O), ceil((i+1)*I/O)). */
static void pool_bounds(int index, int out_size, int in_size, int *start, int *end) {
    *start = (int)(((long long)index * in_size) / out_size);
    *end = (int)(((long long)(index + 1) * in_size + out_size - 1) / out_size);
}

int h3_refmod_pool3d(const float *in, int c, int t, int h, int w,
                     int ot, int oh, int ow, float *out) {
    if (!in || !out || c < 1 || t < 1 || h < 1 || w < 1 || ot < 1 || oh < 1 ||
        ow < 1 || ot > t || oh > h || ow > w) return 0;
    for (int channel = 0; channel < c; channel++) {
        const float *source = in + (size_t)channel * t * h * w;
        float *destination = out + (size_t)channel * ot * oh * ow;
        for (int z = 0; z < ot; z++) {
            int t0, t1; pool_bounds(z, ot, t, &t0, &t1);
            for (int y = 0; y < oh; y++) {
                int h0, h1; pool_bounds(y, oh, h, &h0, &h1);
                for (int x = 0; x < ow; x++) {
                    int w0, w1; pool_bounds(x, ow, w, &w0, &w1);
                    double sum = 0.0;
                    for (int tt = t0; tt < t1; tt++)
                        for (int hh = h0; hh < h1; hh++)
                            for (int ww = w0; ww < w1; ww++)
                                sum += source[((size_t)tt * h + hh) * w + ww];
                    destination[((size_t)z * oh + y) * ow + x] =
                        (float)(sum / ((t1 - t0) * (h1 - h0) * (w1 - w0)));
                }
            }
        }
    }
    return 1;
}

/* align_corners=False source coordinate and its two taps. */
static void interp_taps(int index, int out_size, int in_size,
                        int *i0, int *i1, float *weight1) {
    if (out_size == in_size) { *i0 = *i1 = index; *weight1 = 0.0f; return; }
    float scale = (float)in_size / (float)out_size;
    float source = ((float)index + 0.5f) * scale - 0.5f;
    if (source < 0.0f) source = 0.0f;
    int low = (int)source;
    if (low > in_size - 1) low = in_size - 1;
    *i0 = low;
    *i1 = low + 1 < in_size ? low + 1 : low;
    *weight1 = source - (float)low;
}

int h3_refmod_trilinear(const float *in, int c, int t, int h, int w,
                        int ot, int oh, int ow, float *out) {
    if (!in || !out || c < 1 || t < 1 || h < 1 || w < 1 || ot < 1 || oh < 1 ||
        ow < 1) return 0;
    for (int channel = 0; channel < c; channel++) {
        const float *source = in + (size_t)channel * t * h * w;
        float *destination = out + (size_t)channel * ot * oh * ow;
        for (int z = 0; z < ot; z++) {
            int t0, t1; float wt; interp_taps(z, ot, t, &t0, &t1, &wt);
            for (int y = 0; y < oh; y++) {
                int h0, h1; float wh; interp_taps(y, oh, h, &h0, &h1, &wh);
                for (int x = 0; x < ow; x++) {
                    int w0, w1; float ww; interp_taps(x, ow, w, &w0, &w1, &ww);
                    #define AT(tt, hh, xx) source[((size_t)(tt) * h + (hh)) * w + (xx)]
                    float c00 = AT(t0, h0, w0) * (1.0f - ww) + AT(t0, h0, w1) * ww;
                    float c01 = AT(t0, h1, w0) * (1.0f - ww) + AT(t0, h1, w1) * ww;
                    float c10 = AT(t1, h0, w0) * (1.0f - ww) + AT(t1, h0, w1) * ww;
                    float c11 = AT(t1, h1, w0) * (1.0f - ww) + AT(t1, h1, w1) * ww;
                    #undef AT
                    float c0 = c00 * (1.0f - wh) + c01 * wh;
                    float c1 = c10 * (1.0f - wh) + c11 * wh;
                    destination[((size_t)z * oh + y) * ow + x] =
                        c0 * (1.0f - wt) + c1 * wt;
                }
            }
        }
    }
    return 1;
}

int h3_refmod_apply_strength(float *latent, int c, int t, int h, int w,
                             float strength) {
    if (!latent || c < 1 || t < 1 || h < 1 || w < 1 || !isfinite(strength) ||
        strength < 0.0f || strength > 1.0f) return 0;
    if (strength >= 1.0f) return 1;
    int ph = h / 8 > 1 ? h / 8 : 1, pw = w / 8 > 1 ? w / 8 : 1;
    size_t elements = (size_t)c * t * h * w;
    float *pooled = malloc((size_t)c * t * ph * pw * sizeof(*pooled));
    float *blur = malloc(elements * sizeof(*blur));
    int ok = pooled && blur &&
             h3_refmod_pool3d(latent, c, t, h, w, t, ph, pw, pooled) &&
             h3_refmod_trilinear(pooled, c, t, ph, pw, t, h, w, blur);
    if (ok)
        for (size_t index = 0; index < elements; index++)
            latent[index] = strength * latent[index] +
                            (1.0f - strength) * blur[index];
    free(pooled);
    free(blur);
    return ok;
}

int h3_refmod_apply_strength_audio(float *latent, int t, float strength) {
    if (!latent || t < 1 || !isfinite(strength) || strength < 0.0f ||
        strength > 1.0f) return 0;
    if (strength >= 1.0f || t <= 1) return 1;
    int pooled_t = t / 8 > 1 ? t / 8 : 1;
    float *pooled = malloc((size_t)pooled_t * sizeof(*pooled));
    float *blur = malloc((size_t)t * sizeof(*blur));
    if (!pooled || !blur) { free(pooled); free(blur); return 0; }
    for (int series = 0; series < 32 * 2; series++) {
        float *values = latent + (size_t)series * t;
        h3_refmod_pool3d(values, 1, 1, 1, t, 1, 1, pooled_t, pooled);
        h3_refmod_trilinear(pooled, 1, 1, 1, pooled_t, 1, 1, t, blur);
        for (int index = 0; index < t; index++)
            values[index] = strength * values[index] +
                            (1.0f - strength) * blur[index];
    }
    free(pooled);
    free(blur);
    return 1;
}

/* ---- load ------------------------------------------------------------ */

static float bf16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int h3_refmod_load(const char *path, int member, float strength,
                   h3_refmod_latent *latent, h3_refmod_member *meta,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!latent) return 0;
    memset(latent, 0, sizeof(*latent));
    if (meta) memset(meta, 0, sizeof(*meta));
    if (!isfinite(strength) || strength <= 0.0f || strength > 1.0f) {
        fail(error, error_size, "RefMod strength must be above 0 and at most 1");
        return 0;
    }
    h3_st_header header;
    char detail[384];
    if (!h3_st_read_header(path, &header, detail, sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        return 0;
    }
    h3_refmod_info info;
    int ok = 0;
    void *raw = NULL;
    if (!inspect_header(&header, path, &info, error, error_size)) goto done;
    if (info.bundle && member < 0) {
        fail(error, error_size, "%s is a RefMod bundle: choose a member", path);
        goto done;
    }
    if (member >= (int)info.member_count) {
        fail(error, error_size, "%s has %zu RefMod member%s, not %d", path,
             info.member_count, info.member_count == 1 ? "" : "s", member + 1);
        goto done;
    }
    const h3_refmod_member *chosen = &info.members[member < 0 ? 0 : member];
    if (!chosen->supported) {
        fail(error, error_size, "RefMod member \"%s\" is not supported",
             chosen->name);
        goto done;
    }
    const h3_st_tensor *tensor = h3_st_find(&header, chosen->tensor);
    uint64_t elements = h3_st_tensor_elements(tensor);
    size_t element_size = h3_dtype_size(tensor->dtype);
    if (!elements || elements > SIZE_MAX / sizeof(float) ||
        elements > SIZE_MAX / element_size) {
        fail(error, error_size, "RefMod tensor \"%s\" is too large", chosen->tensor);
        goto done;
    }
    raw = malloc((size_t)elements * element_size);
    latent->values = malloc((size_t)elements * sizeof(float));
    if (!raw || !latent->values) {
        fail(error, error_size, "out of memory loading RefMod %s", path);
        goto done;
    }
    if (!h3_st_read_data(&header, tensor, raw, (size_t)elements * element_size,
                         detail, sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        goto done;
    }
    for (uint64_t index = 0; index < elements; index++) {
        float value;
        if (tensor->dtype == H3_DTYPE_F16)
            value = h3_half_to_float(((const uint16_t *)raw)[index]);
        else if (tensor->dtype == H3_DTYPE_BF16)
            value = bf16_to_float(((const uint16_t *)raw)[index]);
        else
            value = ((const float *)raw)[index];
        if (!isfinite(value)) {
            fail(error, error_size, "RefMod \"%s\" has a non-finite value",
                 chosen->name);
            goto done;
        }
        latent->values[index] = value;
    }
    latent->kind = chosen->kind;
    if (chosen->kind == H3_REFMOD_AUDIO) {
        latent->audio_t = chosen->latent_t;
        ok = h3_refmod_apply_strength_audio(latent->values, latent->audio_t,
                                            strength);
    } else {
        latent->time = chosen->latent_t;
        latent->height = chosen->latent_h;
        latent->width = chosen->latent_w;
        ok = h3_refmod_apply_strength(latent->values, 24, latent->time,
                                      latent->height, latent->width, strength);
    }
    if (!ok) fail(error, error_size, "cannot apply RefMod strength");
    if (ok && meta) *meta = *chosen;
done:
    free(raw);
    if (!ok) h3_refmod_latent_free(latent);
    h3_refmod_info_free(&info);
    h3_st_free_header(&header);
    return ok;
}

void h3_refmod_latent_free(h3_refmod_latent *latent) {
    if (!latent) return;
    free(latent->values);
    memset(latent, 0, sizeof(*latent));
}

/* ---- extraction rules ------------------------------------------------ */

static int round_half_even(double value) {
    return (int)nearbyint(value);   /* the default rounding mode, as Python's round */
}

static int round_to_even_cells(double cells) {
    int result = round_half_even(cells / 2.0) * 2;
    return result < 2 ? 2 : result;
}

void h3_refmod_aspect_grid(int pool_long_edge, int latent_h, int latent_w,
                           int *gh, int *gw) {
    if (pool_long_edge < 2) pool_long_edge = 16;
    int long_edge = round_to_even_cells(pool_long_edge);
    int h, w;
    if (latent_w >= latent_h) {
        w = long_edge;
        h = round_to_even_cells((double)long_edge * latent_h / latent_w);
    } else {
        h = long_edge;
        w = round_to_even_cells((double)long_edge * latent_w / latent_h);
    }
    if (h > latent_h) h = latent_h;
    if (w > latent_w) w = latent_w;
    if (h < 1) h = 1;
    if (w < 1) w = 1;
    *gh = h;
    *gw = w;
}

static int snap32(double pixels, int minimum) {
    int snapped = round_half_even(pixels / 32.0) * 32;
    return snapped < minimum ? minimum : snapped;
}

int h3_refmod_extract_canvas(int source_w, int source_h, int cap_px,
                             int *w, int *h) {
    if (source_w < 1 || source_h < 1 || !w || !h) return 0;
    if (cap_px <= 0) cap_px = 1024;
    int shorter = source_w < source_h ? source_w : source_h;
    double scale = shorter > cap_px ? (double)cap_px / shorter : 1.0;
    int width = snap32(source_w * scale, 32);
    int height = snap32(source_h * scale, 32);
    int short_side = width < height ? width : height;
    if (short_side < 320) {
        double grow = 320.0 / short_side;
        width = snap32(width * grow, 320);
        height = snap32(height * grow, 320);
    }
    *w = width;
    *h = height;
    return 1;
}

/* round(linspace(0, last, n)) into out. */
static void linspace_round(int last, int n, int *out) {
    for (int index = 0; index < n; index++)
        out[index] = n == 1 ? 0 :
            round_half_even((double)last * index / (double)(n - 1));
}

int h3_refmod_extract_frames(int source_frames, int latent_frames,
                             int *indices, int *count) {
    if (source_frames < 1 || !indices || !count) return 0;
    if (latent_frames <= 0) latent_frames = 16;
    int first_count = source_frames < latent_frames ? source_frames : latent_frames;
    if (first_count < 5) {
        indices[0] = 0;
        *count = 1;
        return 1;
    }
    int *first = malloc((size_t)first_count * sizeof(*first));
    if (!first) return 0;
    linspace_round(source_frames - 1, first_count, first);
    int kept = first_count;
    while (kept % 17 != 5) kept--;
    int *picks = malloc((size_t)kept * sizeof(*picks));
    if (!picks) { free(first); return 0; }
    linspace_round(first_count - 1, kept, picks);
    for (int index = 0; index < kept; index++) indices[index] = first[picks[index]];
    *count = kept;
    free(first);
    free(picks);
    return 1;
}

/* ---- refinement ------------------------------------------------------ */

/* The adjoint of h3_refmod_trilinear: each output's gradient goes to its
 * eight taps with the interpolation weights. */
static void trilinear_adjoint(const float *grad_out, int c, int st, int sh, int sw,
                              int t, int h, int w, float *grad_small) {
    memset(grad_small, 0, (size_t)c * st * sh * sw * sizeof(*grad_small));
    for (int channel = 0; channel < c; channel++) {
        const float *source = grad_out + (size_t)channel * t * h * w;
        float *destination = grad_small + (size_t)channel * st * sh * sw;
        for (int z = 0; z < t; z++) {
            int t0, t1; float wt; interp_taps(z, t, st, &t0, &t1, &wt);
            for (int y = 0; y < h; y++) {
                int h0, h1; float wh; interp_taps(y, h, sh, &h0, &h1, &wh);
                for (int x = 0; x < w; x++) {
                    int w0, w1; float ww; interp_taps(x, w, sw, &w0, &w1, &ww);
                    float g = source[((size_t)z * h + y) * w + x];
                    #define ADD(tt, hh, xx, weight) \
                        destination[((size_t)(tt) * sh + (hh)) * sw + (xx)] += g * (weight)
                    ADD(t0, h0, w0, (1.0f - wt) * (1.0f - wh) * (1.0f - ww));
                    ADD(t0, h0, w1, (1.0f - wt) * (1.0f - wh) * ww);
                    ADD(t0, h1, w0, (1.0f - wt) * wh * (1.0f - ww));
                    ADD(t0, h1, w1, (1.0f - wt) * wh * ww);
                    ADD(t1, h0, w0, wt * (1.0f - wh) * (1.0f - ww));
                    ADD(t1, h0, w1, wt * (1.0f - wh) * ww);
                    ADD(t1, h1, w0, wt * wh * (1.0f - ww));
                    ADD(t1, h1, w1, wt * wh * ww);
                    #undef ADD
                }
            }
        }
    }
}

int h3_refmod_refine(float *small, int c, int st, int sh, int sw,
                     const float *full, int t, int h, int w,
                     int steps, float lr,
                     h3_progress_callback on_progress, void *opaque,
                     const volatile int *cancel, char *error, size_t error_size) {
    if (!small || !full || c < 1 || st < 1 || sh < 1 || sw < 1 || t < 1 ||
        h < 1 || w < 1 || steps < 0 || !(lr > 0.0f)) {
        fail(error, error_size, "invalid RefMod refinement arguments");
        return 0;
    }
    if (!steps) return 1;
    size_t small_count = (size_t)c * st * sh * sw;
    size_t full_count = (size_t)c * t * h * w;
    float *up = malloc(full_count * sizeof(*up));
    float *grad = malloc(small_count * sizeof(*grad));
    float *m = calloc(small_count, sizeof(*m));
    float *v = calloc(small_count, sizeof(*v));
    if (!up || !grad || !m || !v) {
        free(up); free(grad); free(m); free(v);
        fail(error, error_size, "out of memory refining the RefMod");
        return 0;
    }
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    const float loss_scale = 2.0f / (float)full_count;   /* d mean((up-full)^2) / d up */
    int ok = 1;
    for (int step = 1; step <= steps; step++) {
        if ((cancel && *cancel) ||
            (on_progress && step % 25 == 1 &&
             on_progress("refmod refine", step - 1, steps, opaque))) {
            fail(error, error_size, "RefMod extraction cancelled");
            ok = 0;
            break;
        }
        h3_refmod_trilinear(small, c, st, sh, sw, t, h, w, up);
        for (size_t index = 0; index < full_count; index++)
            up[index] = (up[index] - full[index]) * loss_scale;
        trilinear_adjoint(up, c, st, sh, sw, t, h, w, grad);
        float correction1 = 1.0f - powf(beta1, (float)step);
        float correction2 = 1.0f - powf(beta2, (float)step);
        for (size_t index = 0; index < small_count; index++) {
            float g = grad[index];
            m[index] = beta1 * m[index] + (1.0f - beta1) * g;
            v[index] = beta2 * v[index] + (1.0f - beta2) * g * g;
            float m_hat = m[index] / correction1;
            float v_hat = v[index] / correction2;
            small[index] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
    if (ok && on_progress) on_progress("refmod refine", steps, steps, opaque);
    free(up); free(grad); free(m); free(v);
    return ok;
}

/* ---- writing --------------------------------------------------------- */

static void json_string(FILE *out, const char *text) {
    fputc('"', out);
    for (const unsigned char *at = (const unsigned char *)text; *at; at++) {
        if (*at == '"' || *at == '\\') { fputc('\\', out); fputc((int)*at, out); }
        else if (*at < 0x20) fprintf(out, "\\u%04x", *at);
        else fputc((int)*at, out);
    }
    fputc('"', out);
}

static char *refmod_meta_json(const h3_refmod_latent *latent,
                              const h3_refmod_meta *meta) {
    char *json = NULL;
    size_t length = 0;
    FILE *out = open_memstream(&json, &length);
    if (!out) return NULL;
    const char *kind = meta->kind == H3_REFMOD_IMAGE ? "image" :
                       meta->kind == H3_REFMOD_VIDEO ? "video" : "audio";
    fputs("{\"name\": ", out); json_string(out, meta->name ? meta->name : "");
    fprintf(out, ", \"kind\": \"%s\", \"latent_h\": %d, \"latent_w\": %d, "
            "\"latent_t\": %d, \"mode\": \"%s\", \"source\": ", kind,
            meta->kind == H3_REFMOD_AUDIO ? 0 : latent->height,
            meta->kind == H3_REFMOD_AUDIO ? 0 : latent->width,
            meta->kind == H3_REFMOD_AUDIO ? latent->audio_t : latent->time,
            meta->mode == H3_REFMOD_COMPRESSED ? "training" : "encode");
    json_string(out, meta->source ? meta->source : kind);
    fputs(", \"source_shape\": ", out);
    json_string(out, meta->source_shape ? meta->source_shape : "");
    fputs(", \"pool\": ", out);
    json_string(out, meta->pool ? meta->pool : "");
    fprintf(out, ", \"optimize_steps\": %d, \"tags\": [", meta->optimize_steps);
    for (size_t index = 0; index < meta->tag_count; index++) {
        if (index) fputs(", ", out);
        json_string(out, meta->tags[index]);
    }
    fputs("], \"description\": ", out);
    json_string(out, meta->description ? meta->description : "");
    fputs(", \"concept_type\": ", out);
    json_string(out, meta->concept_type ? meta->concept_type : "generic");
    fputs(", \"_format_version\": 4, \"sample_rate\": 32000, "
          "\"h3c_extract\": \"H3cApp\"}", out);
    if (fclose(out) != 0) { free(json); return NULL; }
    return json;
}

int h3_refmod_write(const char *path, const h3_refmod_latent *latent,
                    const h3_refmod_meta *meta, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !latent || !latent->values || !meta) {
        fail(error, error_size, "invalid RefMod write arguments");
        return 0;
    }
    size_t elements;
    char shape[96];
    if (meta->kind == H3_REFMOD_AUDIO) {
        if (latent->audio_t < 1) {
            fail(error, error_size, "audio RefMod latent is empty");
            return 0;
        }
        elements = (size_t)32 * 2 * latent->audio_t;
        snprintf(shape, sizeof(shape), "[1,32,2,%d]", latent->audio_t);
    } else {
        if (latent->time < 1 || latent->height < 2 || latent->width < 2 ||
            latent->height % 2 || latent->width % 2 ||
            (meta->kind == H3_REFMOD_IMAGE && latent->time != 1)) {
            fail(error, error_size, "RefMod latent geometry %dx%dx%d is invalid",
                 latent->time, latent->height, latent->width);
            return 0;
        }
        elements = (size_t)24 * latent->time * latent->height * latent->width;
        snprintf(shape, sizeof(shape), "[1,24,%d,%d,%d]", latent->time,
                 latent->height, latent->width);
    }
    char *meta_json = refmod_meta_json(latent, meta);
    char *header = NULL;
    size_t header_length = 0;
    FILE *header_stream = meta_json ? open_memstream(&header, &header_length) : NULL;
    if (!header_stream) {
        free(meta_json);
        fail(error, error_size, "out of memory writing the RefMod header");
        return 0;
    }
    fputs("{\"__metadata__\":{\"refmod_meta\":", header_stream);
    json_string(header_stream, meta_json);
    fprintf(header_stream, "},\"latent\":{\"dtype\":\"F16\",\"shape\":%s,"
            "\"data_offsets\":[0,%zu]}}", shape, elements * 2);
    free(meta_json);
    if (fclose(header_stream) != 0) {
        free(header);
        fail(error, error_size, "out of memory writing the RefMod header");
        return 0;
    }
    uint16_t *halves = malloc(elements * sizeof(*halves));
    if (!halves) {
        free(header);
        fail(error, error_size, "out of memory converting the RefMod latent");
        return 0;
    }
    for (size_t index = 0; index < elements; index++)
        halves[index] = h3_float_to_half(latent->values[index]);
    char partial[2048];
    snprintf(partial, sizeof(partial), "%s.partial", path);
    FILE *out = fopen(partial, "wb");
    int ok = out != NULL;
    if (ok) {
        size_t padded = (header_length + 7) / 8 * 8;
        uint64_t header_size = padded;
        ok = fwrite(&header_size, 8, 1, out) == 1 &&
             fwrite(header, 1, header_length, out) == header_length;
        for (size_t index = header_length; ok && index < padded; index++)
            ok = fputc(' ', out) != EOF;
        ok = ok && fwrite(halves, sizeof(*halves), elements, out) == elements;
        if (fclose(out) != 0) ok = 0;
    }
    free(header);
    free(halves);
    if (!ok) {
        unlink(partial);
        fail(error, error_size, "cannot write %s: %s", partial, strerror(errno));
        return 0;
    }
    if (rename(partial, path) != 0) {
        unlink(partial);
        fail(error, error_size, "cannot rename %s into place: %s", partial,
             strerror(errno));
        return 0;
    }
    return 1;
}

/* ---- extraction ------------------------------------------------------ */

typedef struct {
    h3_progress_callback on_progress;
    void *opaque;
    const volatile int *cancel;
    int cancelled;
} extract_progress;

static int extract_emit(extract_progress *progress, const char *phase,
                        int completed, int total) {
    if (progress->cancel && *progress->cancel) progress->cancelled = 1;
    if (progress->on_progress &&
        progress->on_progress(phase, completed, total, progress->opaque))
        progress->cancelled = 1;
    return !progress->cancelled;
}

static void video_encoder_bridge(int completed, int total, void *opaque) {
    extract_emit(opaque, "video VAE encoder", completed, total);
}

static void audio_encoder_bridge(int completed, int total, void *opaque) {
    extract_emit(opaque, "audio VAE encoder", completed, total);
}

static const char *file_stem(const char *path, char *buffer, size_t size) {
    const char *slash = strrchr(path, '/');
    const char *name = slash ? slash + 1 : path;
    snprintf(buffer, size, "%s", name);
    char *dot = strrchr(buffer, '.');
    if (dot && dot != buffer) *dot = '\0';
    return buffer;
}

static int extract_audio(const h3_refmod_extract_params *params,
                         extract_progress *progress, const char *name,
                         char *error, size_t error_size) {
    int max_seconds = params->max_seconds > 0 ? params->max_seconds : 15;
    if (max_seconds > 16) max_seconds = 16;
    if (params->mode == H3_REFMOD_COMPRESSED) {
        fail(error, error_size, "audio RefMods have no compressed mode");
        return 0;
    }
    if (!extract_emit(progress, "refmod read", 0, 1)) goto cancelled;
    float *pcm = NULL;
    int samples = 0;
    char detail[512];
    if (!h3_av_read_audio_f32(params->source_path, 32000 * max_seconds, 1,
                              &pcm, &samples, detail, sizeof(detail))) {
        free(pcm);
        fail(error, error_size, "%s", detail);
        return 0;
    }
    if (!extract_emit(progress, "refmod read", 1, 1)) { free(pcm); goto cancelled; }
    char vae_path[2048];
    snprintf(vae_path, sizeof(vae_path), "%s/Ref2VA/audio_vae", params->model_dir);
    h3_audio_latent latent;
    memset(&latent, 0, sizeof(latent));
    void *pool = h3_autorelease_push();
    int ok = h3_audio_vae_encode(vae_path, "h3_shaders.metal", pcm, samples,
                                 audio_encoder_bridge, progress, &latent,
                                 detail, sizeof(detail));
    h3_autorelease_pop(pool);
    free(pcm);
    if (!ok) {
        h3_audio_latent_free(&latent);
        fail(error, error_size, "%s", detail);
        return 0;
    }
    if (progress->cancelled) { h3_audio_latent_free(&latent); goto cancelled; }
    if (latent.channels != 32 || latent.stereo != 2 || latent.length < 1) {
        h3_audio_latent_free(&latent);
        fail(error, error_size, "audio encoder produced invalid geometry");
        return 0;
    }
    if (!extract_emit(progress, "refmod write", 0, 1)) {
        h3_audio_latent_free(&latent);
        goto cancelled;
    }
    h3_refmod_latent output = {H3_REFMOD_AUDIO, 0, 0, 0, latent.length, latent.values};
    char source_shape[64];
    snprintf(source_shape, sizeof(source_shape), "%dx0x0", latent.length);
    const char *tags[] = {"0 img, 0 vid, 1 aud"};
    h3_refmod_meta meta = {H3_REFMOD_AUDIO, H3_REFMOD_FULL, 0, name, "audio",
                           source_shape, "4x4x1", params->description,
                           "generic", tags, 1};
    ok = h3_refmod_write(params->output_path, &output, &meta, error, error_size);
    h3_audio_latent_free(&latent);
    if (ok) extract_emit(progress, "refmod write", 1, 1);
    return ok;
cancelled:
    fail(error, error_size, "RefMod extraction cancelled");
    return 0;
}

int h3_refmod_extract(const h3_refmod_extract_params *params,
                      h3_progress_callback on_progress, void *opaque,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!params || !params->model_dir || !*params->model_dir ||
        !params->source_path || !*params->source_path ||
        !params->output_path || !*params->output_path ||
        (params->kind != H3_REFMOD_IMAGE && params->kind != H3_REFMOD_VIDEO &&
         params->kind != H3_REFMOD_AUDIO) ||
        (params->mode != H3_REFMOD_FULL && params->mode != H3_REFMOD_COMPRESSED) ||
        params->optimize_steps < 0 || params->optimize_steps > 100000) {
        fail(error, error_size, "invalid RefMod extraction arguments");
        return 0;
    }
    extract_progress progress = {on_progress, opaque, params->cancel_requested, 0};
    char stem[256];
    const char *name = params->name && *params->name ? params->name :
        file_stem(params->source_path, stem, sizeof(stem));
    if (params->kind == H3_REFMOD_AUDIO)
        return extract_audio(params, &progress, name, error, error_size);

    int source_w, source_h, width, height;
    char detail[512];
    if (!extract_emit(&progress, "refmod read", 0, 1)) goto cancelled;
    if (!h3_av_visual_size(params->source_path, &source_w, &source_h,
                           detail, sizeof(detail)) ||
        !h3_refmod_extract_canvas(source_w, source_h, params->resolution_cap,
                                  &width, &height)) {
        fail(error, error_size, "%s", *detail ? detail :
             "cannot size the RefMod source");
        return 0;
    }
    float *pixels = NULL;
    int frames = 1;
    if (params->kind == H3_REFMOD_IMAGE) {
        if (!h3_av_read_image_f32(params->source_path, width, height,
                                  H3_IMAGE_FIT_STRETCH, &pixels,
                                  detail, sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            return 0;
        }
    } else {
        int source_frames = 0;
        if (!h3_av_video_frame_count(params->source_path, &source_frames,
                                     detail, sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            return 0;
        }
        int latent_frames = params->latent_frames > 0 ? params->latent_frames : 16;
        int *indices = malloc((size_t)latent_frames * sizeof(*indices));
        if (!indices || !h3_refmod_extract_frames(source_frames, latent_frames,
                                                  indices, &frames)) {
            free(indices);
            fail(error, error_size, "cannot choose the RefMod's frames");
            return 0;
        }
        int ok = h3_av_read_video_frames_f32(params->source_path, width, height,
                                             indices, frames, &pixels,
                                             detail, sizeof(detail));
        free(indices);
        if (!ok) {
            fail(error, error_size, "%s", detail);
            return 0;
        }
    }
    if (!extract_emit(&progress, "refmod read", 1, 1)) { free(pixels); goto cancelled; }
    char vae_path[2048];
    snprintf(vae_path, sizeof(vae_path), "%s/Ref2VA/video_vae/source",
             params->model_dir);
    h3_video_latent latent;
    memset(&latent, 0, sizeof(latent));
    void *pool = h3_autorelease_push();
    int ok = h3_video_vae_encode(vae_path, "h3_shaders.metal", pixels, frames,
                                 height, width, video_encoder_bridge, &progress,
                                 &latent, detail, sizeof(detail));
    h3_autorelease_pop(pool);
    free(pixels);
    if (!ok) {
        h3_video_latent_free(&latent);
        fail(error, error_size, "%s", detail);
        return 0;
    }
    if (progress.cancelled) { h3_video_latent_free(&latent); goto cancelled; }
    if (latent.time < 1 || latent.height != height / 16 ||
        latent.width != width / 16) {
        h3_video_latent_free(&latent);
        fail(error, error_size, "video encoder produced unexpected geometry");
        return 0;
    }
    const char *source = params->kind == H3_REFMOD_IMAGE ? "image" : "video";
    const char *tags[] = {params->kind == H3_REFMOD_IMAGE ? "1 img, 0 vid" :
                                                             "0 img, 1 vid"};
    char source_shape[64], pool_text[96];
    snprintf(source_shape, sizeof(source_shape), "%dx%dx%d", latent.time,
             latent.height, latent.width);
    h3_refmod_latent output = {params->kind, latent.time, latent.height,
                               latent.width, 0, latent.values};
    float *small = NULL;
    int steps = 0;
    if (params->mode == H3_REFMOD_COMPRESSED) {
        int long_edge = params->pool_long_edge > 0 ? params->pool_long_edge : 16;
        int gh, gw;
        h3_refmod_aspect_grid(long_edge, latent.height, latent.width, &gh, &gw);
        int latent_frames = params->latent_frames > 0 ? params->latent_frames : 16;
        int pool_t = params->kind == H3_REFMOD_IMAGE ? 1 :
                     latent.time < latent_frames ? latent.time : latent_frames;
        size_t small_count = (size_t)24 * pool_t * gh * gw;
        small = malloc(small_count * sizeof(*small));
        if (!small || !h3_refmod_pool3d(latent.values, 24, latent.time,
                                        latent.height, latent.width,
                                        pool_t, gh, gw, small)) {
            free(small);
            h3_video_latent_free(&latent);
            fail(error, error_size, "cannot pool the RefMod latent");
            return 0;
        }
        /* The upstream extractor pools in F32, keeps the result as F16 and
         * refines from there; rounding here reproduces its start point. */
        for (size_t index = 0; index < small_count; index++)
            small[index] = h3_half_to_float(h3_float_to_half(small[index]));
        steps = params->optimize_steps;
        if (!h3_refmod_refine(small, 24, pool_t, gh, gw, latent.values,
                              latent.time, latent.height, latent.width,
                              steps, 0.02f, on_progress, opaque,
                              params->cancel_requested, error, error_size)) {
            free(small);
            h3_video_latent_free(&latent);
            return 0;
        }
        output.time = pool_t;
        output.height = gh;
        output.width = gw;
        output.values = small;
        snprintf(pool_text, sizeof(pool_text), "%dx%dx%d", pool_t, gh, gw);
    } else {
        snprintf(pool_text, sizeof(pool_text),
                 "full-res %dx%dpx (short-edge cap %dpx)", width, height,
                 params->resolution_cap > 0 ? params->resolution_cap : 1024);
    }
    if (!extract_emit(&progress, "refmod write", 0, 1)) {
        free(small);
        h3_video_latent_free(&latent);
        goto cancelled;
    }
    h3_refmod_meta meta = {params->kind,
                           params->mode == H3_REFMOD_COMPRESSED ?
                               H3_REFMOD_COMPRESSED : H3_REFMOD_FULL,
                           steps, name, source, source_shape, pool_text,
                           params->description, "generic", tags, 1};
    ok = h3_refmod_write(params->output_path, &output, &meta, error, error_size);
    free(small);
    h3_video_latent_free(&latent);
    if (ok) extract_emit(&progress, "refmod write", 1, 1);
    return ok;
cancelled:
    fail(error, error_size, "RefMod extraction cancelled");
    return 0;
}
