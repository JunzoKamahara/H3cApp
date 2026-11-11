/* CLI wrapper around h3_build_attention_cache() (h3_attention_cache.c),
 * which does the actual work - shared with the native app so it can build
 * a missing cache itself rather than just reporting one's absence. This
 * file only adds the "model root dir, build both FL2VA and Ref2VA in one
 * go" convenience and prints per-block progress to stderr.
 *
 * Usage:
 *   build_attention_cache <FL2VA/transformer dir> <output cache file>
 *     - single-component mode, backward compatible with v2's usage.
 *   build_attention_cache <model root dir> <output cache directory>
 *     - detected when <model root dir>/FL2VA/transformer/config.json
 *       exists: writes <output dir>/fl2va.cache, and, if a Ref2VA
 *       transformer is present too, <output dir>/ref2va.cache - matching
 *       H3_ATTENTION_CACHE_DIR's auto-selection by generation mode.
 */
#include "h3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int path_exists(const char *path) {
    struct stat status;
    return stat(path, &status) == 0;
}

static char *join_path(const char *base, const char *suffix) {
    size_t length = strlen(base) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (result) snprintf(result, length, "%s/%s", base, suffix);
    return result;
}

static int print_progress(const char *phase, int completed, int total,
                          void *opaque) {
    (void)phase;
    (void)opaque;
    fprintf(stderr, "h3: attention cache block %2d/%d\n", completed, total);
    return 0;
}

static int build_one_cache(const char *transformer_dir,
                           const char *output_path) {
    char error[512];
    if (h3_build_attention_cache(transformer_dir, output_path,
                                 "h3_shaders.metal", print_progress, NULL,
                                 error, sizeof(error))) {
        fprintf(stderr, "h3: wrote attention cache to %s\n", output_path);
        return 1;
    }
    fprintf(stderr, "h3: %s\n", error);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr,
                "usage: %s <FL2VA/transformer dir> <output cache file>\n"
                "       %s <model root dir> <output cache directory>\n",
                argv[0], argv[0]);
        return 1;
    }
    char *fl2va_dir = join_path(argv[1], "FL2VA/transformer");
    char *fl2va_marker = fl2va_dir ? join_path(fl2va_dir, "config.json") : NULL;
    int model_root_mode = fl2va_marker && path_exists(fl2va_marker);
    free(fl2va_marker);

    int ok;
    if (model_root_mode) {
        mkdir(argv[2], 0755); /* ignore EEXIST - a pre-existing dir is fine */
        char *fl2va_out = join_path(argv[2], "fl2va.cache");
        char *ref2va_dir = join_path(argv[1], "Ref2VA/transformer");
        char *ref2va_marker = ref2va_dir ? join_path(ref2va_dir, "config.json") : NULL;
        ok = fl2va_out && build_one_cache(fl2va_dir, fl2va_out);
        if (ok) {
            if (ref2va_marker && path_exists(ref2va_marker)) {
                char *ref2va_out = join_path(argv[2], "ref2va.cache");
                ok = ref2va_out && build_one_cache(ref2va_dir, ref2va_out);
                free(ref2va_out);
            } else {
                fprintf(stderr,
                        "h3: no Ref2VA transformer under %s - built "
                        "fl2va.cache only\n", argv[1]);
            }
        }
        free(fl2va_out);
        free(ref2va_dir);
        free(ref2va_marker);
    } else {
        ok = build_one_cache(argv[1], argv[2]);
    }

    free(fl2va_dir);
    return ok ? 0 : 1;
}
