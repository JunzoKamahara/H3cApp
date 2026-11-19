#ifndef H3_TEXT_PROJECTION_H
#define H3_TEXT_PROJECTION_H

#include <stddef.h>
#include <stdint.h>

/* A ClipProj projection (NicoLab28/ClipProj-MiniMax-H3): a matrix fitted by
 * ridge regression that takes one hidden state of a small Qwen3-VL to the
 * text rows H3's own text encoder, the first 50 layers of Qwen3-VL-32B,
 * gives the DiT. As ComfyUI-ClipProj applies it:
 *
 *     rows = ((hidden - mean_in) / std_in) @ W * std_out + mean_out
 *     rows[0] = sink_out
 *
 * `hidden` is hidden_states[tap] of the small model - the input of its
 * layer `tap`, counting from 0 - so only its first `tap` layers run. The
 * first row of a sequence is the attention sink, which the matrix was never
 * fitted on; its measured value replaces it.
 *
 * The file is a safetensors with W [input, output], mean_in and std_in
 * [input], mean_out, std_out and sink_out [output], F32 or F16, and `tap`
 * in its metadata. A file without `tap` is refused rather than given a
 * default, and so is one with a residual network (`mlp.*`, the -mlp files):
 * applying its matrix alone would be another projection than the file's. */
typedef struct {
    int tap;
    size_t input_width;
    size_t output_width;
    float *weight;      /* [input_width, output_width], row-major */
    float *input_mean;
    float *input_std;
    float *output_mean;
    float *output_std;
    float *sink;
} h3_text_projection;

int h3_text_projection_load(const char *path, h3_text_projection *projection,
                            char *error, size_t error_size);
void h3_text_projection_free(h3_text_projection *projection);

/* hidden: BF16 [tokens, input_width], its first row the first row of the
 * whole sequence. output: BF16 [tokens, output_width]. The arithmetic is
 * F32; only the result is rounded. */
int h3_text_projection_apply(const h3_text_projection *projection,
                             const uint16_t *hidden, size_t tokens,
                             uint16_t *output, char *error, size_t error_size);

#endif
