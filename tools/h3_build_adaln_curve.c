/* CLI wrapper around h3_build_adaln_curve() (h3_adaln_curve.c).
 *
 * Usage:
 *   build_adaln_curve <transformer dir> <output file> [grid] [rank]
 *     - e.g. build_adaln_curve ~/models/.../FL2VA/transformer fl2va.adaln
 *       (grid/rank default to 4097/32). Run from the repository root so
 *       h3_shaders.metal is found.
 */
#include "h3.h"

#include <stdio.h>
#include <stdlib.h>

static int print_progress(const char *phase, int completed, int total,
                          void *opaque) {
    (void)phase;
    (void)opaque;
    fprintf(stderr, "h3: AdaLN curve projection %2d/%d\n", completed, total);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr,
                "usage: %s <transformer dir> <output file> [grid] [rank]\n",
                argv[0]);
        return 1;
    }
    unsigned grid = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 10) : 0;
    unsigned rank = argc > 4 ? (unsigned)strtoul(argv[4], NULL, 10) : 0;
    char error[512];
    if (!h3_build_adaln_curve(argv[1], argv[2], "h3_shaders.metal", grid,
                              rank, print_progress, NULL, error,
                              sizeof(error))) {
        fprintf(stderr, "h3: %s\n", error);
        return 1;
    }
    fprintf(stderr, "h3: wrote AdaLN curve to %s\n", argv[2]);
    return 0;
}
