/* The model families, samplers and compute modes as tables (the public side
 * is in h3.h: h3_model_family, h3_sampler, h3_compute_mode). This header adds what only the engine
 * needs: how a family builds its sigma schedule and names itself in logs.
 * A new family is a row in h3_model_family.c; h3.c and h3_dit.c ask the row
 * what applies instead of naming the family. What a family's blocks look
 * like (HyperFlow's factored linears and two-time AdaLN, h3_hyperflow.h)
 * stays with the DiT. */
#ifndef H3_MODEL_FAMILY_H
#define H3_MODEL_FAMILY_H

#include "h3.h"
#include "h3_host.h"

typedef struct {
    h3_model_family_info info;
    /* The short name in log lines ("HyperFlow"); NULL for plain H3, whose
     * lines carry none. */
    const char *log_name;
    /* Why a step count outside step_choices is refused. */
    const char *steps_problem;
    /* The sigma schedule for `steps` evaluations; `audio_shift` is the
     * request's (a family without H3_CAN_AUDIO_SHIFT ignores it). */
    int (*build_schedule)(int steps, float audio_shift, h3_sigma_schedule *schedule);
} h3_model_family_desc;

/* The row of a checkpoint: `two_time` is h3_hyperflow_info.two_time. */
const h3_model_family_desc *h3_model_family_of(int two_time);

#endif
