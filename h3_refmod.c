#include "h3_refmod.h"
#include "h3_json.h"
#include "h3_quant.h"
#include "h3_safetensors.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
