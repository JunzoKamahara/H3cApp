/* Extracts a rank range of a LoRA safetensors file into a new file: for
 * NVIDIA's Open-H3-Regenerate-2K merge (rank 576 = refiner 0-383 | Turbo
 * 384-511 | LMS Sharp 512-575, F32, alpha == rank with the strengths
 * folded in), the refiner part alone as BF16 (UPSCALE_PLAN appendix A.5,
 * policy 3: the extraction happens on the device, nothing is
 * redistributed). lora_A rows and lora_B columns [rank_begin, rank_end) are
 * kept, names unchanged; every other tensor is copied whole (BF16 when
 * `bf16`). The output's __metadata__ records the source and the range. */
#ifndef H3_LORA_EXTRACT_H
#define H3_LORA_EXTRACT_H

#include "h3.h"

#include <stddef.h>

/* Progress phase "lora extract" (tensors done / total). Returns 1 on
 * success; the output is written to a temporary name and renamed at the
 * end. */
int h3_lora_extract_ranks(const char *in_path, const char *out_path,
                          int rank_begin, int rank_end, int bf16,
                          const char *note,
                          h3_progress_callback on_progress, void *opaque,
                          char *error, size_t error_size);

#endif
