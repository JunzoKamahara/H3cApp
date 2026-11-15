/* h3_image_reference_canvas (include/h3_image.h): the sizes a reference
 * image is used at when an image is edited. No GPU, no weights. */
#include "h3_image.h"

#include <stdio.h>

static int checks, failures;
#define CHECK(condition) do { checks++; if (!(condition)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); } } while (0)

static int canvas(int width, int height, int limit, int *out) {
    return h3_image_reference_canvas(width, height, limit, &out[0], &out[1],
                                     &out[2], &out[3]);
}

int main(void) {
    int size[4];
    /* A generated 1024 x 1024: used as it is, by both. */
    CHECK(canvas(1024, 1024, 0, size));
    CHECK(size[0] == 1024 && size[1] == 1024 && size[2] == 1024 && size[3] == 1024);
    /* Smaller: enlarged to a short side of 1024. */
    CHECK(canvas(768, 512, 0, size));
    CHECK(size[0] == 1536 && size[1] == 1024 && size[2] == 1536 && size[3] == 1024);
    /* Between 1024 and 2048: its own size for the latent, 1024 for vision. */
    CHECK(canvas(1344, 1568, 0, size));
    CHECK(size[0] == 1344 && size[1] == 1568 && size[2] == 1024 && size[3] == 1184);
    /* Larger: reduced to a short side of 2048. */
    CHECK(canvas(4096, 4096, 0, size));
    CHECK(size[0] == 2048 && size[1] == 2048 && size[2] == 1024 && size[3] == 1024);
    /* A lower limit for the latent; vision stays at 1024. */
    CHECK(canvas(4096, 4096, 1024, size));
    CHECK(size[0] == 1024 && size[1] == 1024 && size[2] == 1024);
    CHECK(canvas(1600, 1200, 1024, size));
    CHECK(size[0] == 1376 && size[1] == 1024);
    /* Not a multiple of 32: rounded to the nearest. */
    CHECK(canvas(1000, 1500, 0, size));
    CHECK(size[0] == 1024 && size[1] == 1536 && size[2] == 1024 && size[3] == 1536);
    /* A wide picture at a long short side stays within 2048 x 2048 in area. */
    CHECK(canvas(6000, 3000, 0, size));
    CHECK(size[0] % 32 == 0 && size[1] % 32 == 0);
    CHECK((long)size[0] * size[1] <= 2048L * 2048L + 2048L * 64L);
    CHECK(size[0] > size[1] && size[1] >= 1024);
    CHECK(size[2] == 2048 && size[3] == 1024);
    /* The vision encoder's aspect budget: 12.25 to 1 is the last accepted. */
    CHECK(canvas(1225, 100, 0, size));
    CHECK(!canvas(1226, 100, 0, size));
    CHECK(!canvas(100, 1300, 0, size));
    CHECK(!canvas(0, 100, 0, size));
    CHECK(!h3_image_reference_canvas(100, 100, 0, NULL, NULL, NULL, NULL));
    printf("image reference canvas: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
