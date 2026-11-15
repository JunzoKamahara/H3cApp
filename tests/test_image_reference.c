/* h3_image_reference_canvas (include/h3_image.h): the sizes a reference
 * image is used at when an image is edited; and P-H3-Image's packed layout
 * against diffusers' PH3ImageEditPackedLayoutStep. No GPU, no weights. */
#include "h3_image.h"
#include "h3_host.h"

#include <math.h>
#include <stdio.h>

static int checks, failures;
#define CHECK(condition) do { checks++; if (!(condition)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); } } while (0)

/* diffusers' _frame_position_grid along one axis: n = size / 2 patches over
 * the area-normalized span, times 32. */
static double axis(int size, int index, double sqrt_area) {
    double ratio = size / sqrt_area, n = size / 2;
    return (index * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
}

static int near(double a, double b) { return fabs(a - b) < 1e-9; }

/* [text 7 | one 4 x 4 reference | audio 2 latents x 2 channels | image
 * 6 x 8], as P-H3-Image edits: the reference takes time 7, the audio and
 * the image start at 8, each audio channel sits at an edge of the image's
 * width grid. */
static void ph3_layout(void) {
    h3_layout_ref reference = {.kind = H3_LAYOUT_REF_IMAGE, .latent_t = 1,
                               .latent_h = 4, .latent_w = 4};
    h3_layout_spec spec = {.text_len = 7, .latent_t = 1, .latent_h = 6,
                           .latent_w = 8, .audio_t = 2, .frame_count = 5,
                           .references = &reference, .reference_count = 1};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    if (failures) return;
    CHECK(layout.seq_len == 7 + 4 + 4 + 12);
    CHECK(layout.segment_count == 4);
    if (layout.segment_count != 4) { h3_layout_free(&layout); return; }
    CHECK(layout.segments[0].kind == H3_SEG_TEXT && layout.segments[0].stop == 7);
    CHECK(layout.segments[1].kind == H3_SEG_REF_IMAGE && layout.segments[1].stop == 11);
    CHECK(layout.segments[2].kind == H3_SEG_AUDIO && layout.segments[2].stop == 15);
    CHECK(layout.segments[3].kind == H3_SEG_VIDEO && layout.segments[3].stop == 27);
    const h3_position *p = layout.positions;
    for (int row = 0; row < 7; row++)
        CHECK(near(p[row].t, row) && p[row].h == 0.0 && p[row].w == 0.0);
    double ref_area = sqrt(4.0 * 4.0), area = sqrt(6.0 * 8.0);
    for (int row = 0; row < 4; row++)
        CHECK(near(p[7 + row].t, 7.0) &&
              near(p[7 + row].h, axis(4, row / 2, ref_area)) &&
              near(p[7 + row].w, axis(4, row % 2, ref_area)));
    double low = axis(8, 0, area), high = axis(8, 3, area);
    for (int row = 0; row < 4; row++)
        CHECK(near(p[11 + row].t, 8.0 + row % 2) && p[11 + row].h == 0.0 &&
              near(p[11 + row].w, row < 2 ? low : high));
    for (int row = 0; row < 12; row++)
        CHECK(near(p[15 + row].t, 8.0) &&
              near(p[15 + row].h, axis(6, row / 4, area)) &&
              near(p[15 + row].w, axis(8, row % 4, area)));
    h3_layout_free(&layout);
}

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
    /* P-H3-Image: the VAE takes the picture's own size within 1 to 8 MP,
     * the vision tower about a megapixel (diffusers' sizes, rounded the
     * same way). */
    int ph3[4];
    CHECK(h3_image_reference_canvas_ph3(1024, 1024, &ph3[0], &ph3[1], &ph3[2], &ph3[3]));
    CHECK(ph3[0] == 1024 && ph3[1] == 1024 && ph3[2] == 1024 && ph3[3] == 1024);
    CHECK(h3_image_reference_canvas_ph3(512, 512, &ph3[0], &ph3[1], &ph3[2], &ph3[3]));
    CHECK(ph3[0] == 1024 && ph3[1] == 1024 && ph3[2] == 992 && ph3[3] == 992);
    CHECK(h3_image_reference_canvas_ph3(4096, 4096, &ph3[0], &ph3[1], &ph3[2], &ph3[3]));
    CHECK(ph3[0] == 2912 && ph3[1] == 2912 && ph3[2] == 992 && ph3[3] == 992);
    CHECK(h3_image_reference_canvas_ph3(2496, 3360, &ph3[0], &ph3[1], &ph3[2], &ph3[3]));
    CHECK(ph3[0] == 2496 && ph3[1] == 3360 && ph3[2] == 864 && ph3[3] == 1152);
    CHECK(h3_image_reference_canvas_ph3(3000, 2000, &ph3[0], &ph3[1], &ph3[2], &ph3[3]));
    CHECK(ph3[0] == 3008 && ph3[1] == 1984 && ph3[2] == 1216 && ph3[3] == 832);
    CHECK(h3_image_reference_canvas_ph3(1344, 768, &ph3[0], &ph3[1], &ph3[2], &ph3[3]));
    CHECK(ph3[0] == 1344 && ph3[1] == 768 && ph3[2] == 1344 && ph3[3] == 768);
    CHECK(!h3_image_reference_canvas_ph3(1300, 100, &ph3[0], &ph3[1], &ph3[2], &ph3[3]));
    ph3_layout();
    printf("image reference canvas and P-H3 layout: %d checks, %d failures\n",
           checks, failures);
    return failures != 0;
}
