/* Inspect GGUF files the way the importer does, or trim a text encoder.
 *   h3_gguf_tool inspect <file.gguf>...
 *   h3_gguf_tool trim <text encoder.gguf> <output.gguf> [layers=50] */
#include "../h3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *role_name(h3_gguf_role role) {
    switch (role) {
    case H3_GGUF_DIT: return "DiT";
    case H3_GGUF_TEXT_ENCODER: return "text encoder";
    case H3_GGUF_VISION: return "vision (mmproj)";
    default: return "unknown";
    }
}

static int progress(const char *phase, int completed, int total, void *opaque) {
    (void)phase;
    (void)opaque;
    if (completed % 1024 < 64 || completed == total)
        fprintf(stderr, "\rtrim %d / %d MiB", completed, total);
    return 0;
}

int main(int argc, char **argv) {
    char error[512];
    if (argc >= 3 && !strcmp(argv[1], "inspect")) {
        for (int i = 2; i < argc; i++) {
            h3_gguf_info info;
            if (!h3_gguf_inspect(argv[i], &info, error, sizeof(error))) {
                fprintf(stderr, "%s\n", error);
                return 1;
            }
            printf("%s: %s, %s, %zu tensors, %.2f GiB, arch '%s', name '%s'",
                   argv[i], role_name(info.role), info.main_type, info.tensors,
                   (double)info.file_bytes / (1 << 30), info.architecture, info.name);
            if (info.role == H3_GGUF_DIT)
                printf(", AdaLN %s", info.curve_adaln ? "curve" : info.dense_adaln ? "dense" : "none");
            if (info.role == H3_GGUF_TEXT_ENCODER)
                printf(", %d layers%s", info.text_layers, info.has_vision ? ", with vision" : "");
            printf("\n");
        }
        return 0;
    }
    if (argc >= 4 && !strcmp(argv[1], "trim")) {
        int layers = argc > 4 ? atoi(argv[4]) : 50;
        if (!h3_gguf_trim_text_encoder(argv[2], argv[3], layers, progress, NULL,
                                       error, sizeof(error))) {
            fprintf(stderr, "\n%s\n", error);
            return 1;
        }
        fprintf(stderr, "\nwrote %s\n", argv[3]);
        return 0;
    }
    fprintf(stderr, "usage: %s inspect <file.gguf>... | trim <in.gguf> <out.gguf> [layers]\n", argv[0]);
    return 2;
}
