#include "h3_model_family.h"

#include <stddef.h>

static int h3_family_schedule(int steps, float audio_shift, h3_sigma_schedule *schedule) {
    return h3_serving_schedule_build_shifted(steps, audio_shift, schedule);
}

static int hyperflow_family_schedule(int steps, float audio_shift, h3_sigma_schedule *schedule) {
    (void)audio_shift;
    return h3_hyperflow_schedule_build(steps, schedule);
}

static const h3_model_family_desc families[] = {
    {
        .info = {
            .id = H3_FAMILY_H3, .name = "h3", .label = "MiniMax H3",
            .step_choice_count = 0, .default_steps = H3_DEFAULT_STEPS,
            .capabilities = H3_CAN_LORA | H3_CAN_RENOISE | H3_CAN_STEP_REUSE |
                            H3_CAN_LAYER_SKIP | H3_CAN_CUSTOM_SIGMAS |
                            H3_CAN_AUDIO_SHIFT | H3_CAN_ATTENTION_CACHE |
                            H3_CAN_ADALN_CURVE,
        },
        .build_schedule = h3_family_schedule,
    },
    {
        /* Its own distilled 4/6/8-step sampler and none of the H3
         * speed-ups that change blocks or steps (h3_hyperflow.h). */
        .info = {
            .id = H3_FAMILY_HYPERFLOW, .name = "hyperflow", .label = "HyperFlow H3 Flash",
            .step_choices = {4, 6, 8}, .step_choice_count = 3, .default_steps = 6,
            .capabilities = 0,
        },
        .log_name = "HyperFlow",
        .steps_problem = "it samples in 4, 6 or 8 steps",
        .build_schedule = hyperflow_family_schedule,
    },
};

int h3_model_family_count(void) {
    return (int)(sizeof(families) / sizeof(*families));
}

static const h3_model_family_desc *family_desc(int id) {
    for (size_t i = 0; i < sizeof(families) / sizeof(*families); i++)
        if (families[i].info.id == id) return &families[i];
    return NULL;
}

const h3_model_family_info *h3_model_family(int id) {
    const h3_model_family_desc *desc = family_desc(id);
    return desc ? &desc->info : NULL;
}

const h3_model_family_desc *h3_model_family_of(int two_time) {
    return family_desc(two_time ? H3_FAMILY_HYPERFLOW : H3_FAMILY_H3);
}

const char *h3_model_family_problem(const h3_model_family_info *family,
                                    const h3_params *params) {
    if (!family || !params) return NULL;
    unsigned can = family->capabilities;
    if (family->step_choice_count) {
        int found = 0;
        for (int i = 0; i < family->step_choice_count; i++)
            if (family->step_choices[i] == params->steps) found = 1;
        if (!found) {
            const h3_model_family_desc *desc = family_desc(family->id);
            return desc && desc->steps_problem ? desc->steps_problem
                                               : "it does not sample in that many steps";
        }
    }
    if (!(can & H3_CAN_LORA) && params->lora_count)
        return "LoRA adapters do not apply to it";
    if (!(can & H3_CAN_RENOISE) && params->sampler != H3_SAMPLER_EULER)
        return "it uses its own Euler sampler (no re-noise)";
    if ((!(can & H3_CAN_STEP_REUSE) &&
         (params->denoise_reuse != 1 || params->core_reuse != 1 ||
          params->token_reduction)) ||
        (!(can & H3_CAN_LAYER_SKIP) && params->dit_layers != H3_DEFAULT_DIT_LAYERS))
        return "speed settings that skip blocks or evaluations do not "
               "apply to it";
    if (!(can & H3_CAN_CUSTOM_SIGMAS) && params->video_sigmas)
        return "it has its own sigma schedule";
    if (!(can & H3_CAN_AUDIO_SHIFT) && params->audio_sigma_shift > 0.0f &&
        params->audio_sigma_shift != (float)H3_AUDIO_SIGMA_SHIFT)
        return "its audio shift is fixed at 3";
    return NULL;
}

static const h3_sampler_info samplers[] = {
    {H3_SAMPLER_EULER, "euler", 0},
    /* Predicts the clean latent and re-noises it to the next sigma. */
    {H3_SAMPLER_RENOISE, "renoise", 1},
};

int h3_sampler_count(void) {
    return (int)(sizeof(samplers) / sizeof(*samplers));
}

const h3_sampler_info *h3_sampler(int id) {
    for (size_t i = 0; i < sizeof(samplers) / sizeof(*samplers); i++)
        if (samplers[i].id == id) return &samplers[i];
    return NULL;
}

static const h3_compute_mode_info compute_modes[] = {
    {H3_COMPUTE_ATTENTION_CACHE, "attentionCache", 1, 1, 1},
    {H3_COMPUTE_RESIDENT, "resident", 0, 0, 0},
    {H3_COMPUTE_SSD_STREAMING, "ssdStreaming", 0, 0, 1},
};

int h3_compute_mode_count(void) {
    return (int)(sizeof(compute_modes) / sizeof(*compute_modes));
}

const h3_compute_mode_info *h3_compute_mode(int id) {
    for (size_t i = 0; i < sizeof(compute_modes) / sizeof(*compute_modes); i++)
        if (compute_modes[i].id == id) return &compute_modes[i];
    return NULL;
}
