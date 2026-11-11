/* The model family and sampler tables (h3.h, h3_model_family.h): what each
 * family refuses, in the order h3_generate reports it. No GPU, no weights. */
#include "h3.h"
#include "h3_model_family.h"

#include <stdio.h>
#include <string.h>

static int checks, failures;
#define CHECK(condition) do { checks++; if (!(condition)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); } } while (0)

static int is(const char *problem, const char *expected) {
    return expected ? problem && !strcmp(problem, expected) : problem == NULL;
}

int main(void) {
    const h3_model_family_info *h3 = h3_model_family(H3_FAMILY_H3);
    const h3_model_family_info *hf = h3_model_family(H3_FAMILY_HYPERFLOW);
    CHECK(h3 && hf && h3_model_family_count() == 2 && !h3_model_family(99));
    CHECK(!strcmp(h3->name, "h3") && !strcmp(hf->name, "hyperflow"));
    CHECK(h3->step_choice_count == 0 && h3->default_steps == H3_DEFAULT_STEPS);
    CHECK(hf->step_choice_count == 3 && hf->step_choices[0] == 4 &&
          hf->step_choices[1] == 6 && hf->step_choices[2] == 8 && hf->default_steps == 6);
    CHECK(h3_model_family_of(0)->info.id == H3_FAMILY_H3 && !h3_model_family_of(0)->log_name);
    CHECK(h3_model_family_of(1)->info.id == H3_FAMILY_HYPERFLOW);

    h3_params params = H3_PARAMS_DEFAULT;
    /* Plain H3 takes everything. */
    params.steps = 20; params.lora_count = 2; params.sampler = H3_SAMPLER_RENOISE;
    params.dit_layers = 40; params.audio_sigma_shift = 2.0f;
    CHECK(is(h3_model_family_problem(h3, &params), NULL));

    /* HyperFlow, in h3_generate's order. */
    h3_params base = H3_PARAMS_DEFAULT;
    base.steps = 6;
    CHECK(is(h3_model_family_problem(hf, &base), NULL));
    params = base; params.steps = 5; params.lora_count = 1;
    CHECK(is(h3_model_family_problem(hf, &params), "it samples in 4, 6 or 8 steps"));
    params = base; params.lora_count = 1; params.sampler = H3_SAMPLER_RENOISE;
    CHECK(is(h3_model_family_problem(hf, &params), "LoRA adapters do not apply to it"));
    params = base; params.sampler = H3_SAMPLER_RENOISE; params.denoise_reuse = 2;
    CHECK(is(h3_model_family_problem(hf, &params), "it uses its own Euler sampler (no re-noise)"));
    const char *speed = "speed settings that skip blocks or evaluations do not apply to it";
    params = base; params.denoise_reuse = 2; CHECK(is(h3_model_family_problem(hf, &params), speed));
    params = base; params.core_reuse = 2; CHECK(is(h3_model_family_problem(hf, &params), speed));
    params = base; params.token_reduction = 1; CHECK(is(h3_model_family_problem(hf, &params), speed));
    params = base; params.dit_layers = 40; CHECK(is(h3_model_family_problem(hf, &params), speed));
    float sigmas[7] = {1, 0.9f, 0.7f, 0.5f, 0.3f, 0.1f, 0};
    params = base; params.video_sigmas = sigmas; params.audio_sigma_shift = 2.0f;
    CHECK(is(h3_model_family_problem(hf, &params), "it has its own sigma schedule"));
    params = base; params.audio_sigma_shift = 2.0f;
    CHECK(is(h3_model_family_problem(hf, &params), "its audio shift is fixed at 3"));
    params = base; params.audio_sigma_shift = 3.0f;
    CHECK(is(h3_model_family_problem(hf, &params), NULL));
    for (int steps = 4; steps <= 8; steps += 2) {
        params = base; params.steps = steps;
        CHECK(is(h3_model_family_problem(hf, &params), NULL));
    }

    /* Each family's schedule is the builder it replaced. */
    h3_sigma_schedule a, b;
    CHECK(h3_model_family_of(1)->build_schedule(6, 2.0f, &a) && h3_hyperflow_schedule_build(6, &b) &&
          !memcmp(&a, &b, sizeof(a)));
    CHECK(!h3_model_family_of(1)->build_schedule(5, 3.0f, &a));
    CHECK(h3_model_family_of(0)->build_schedule(20, 2.0f, &a) &&
          h3_serving_schedule_build_shifted(20, 2.0f, &b) && !memcmp(&a, &b, sizeof(a)));

    CHECK(h3_sampler_count() == 2 && !h3_sampler(2) && !h3_sampler(-1));
    CHECK(h3_sampler(H3_SAMPLER_EULER) && !h3_sampler(H3_SAMPLER_EULER)->evaluates_every_step);
    CHECK(h3_sampler(H3_SAMPLER_RENOISE) && h3_sampler(H3_SAMPLER_RENOISE)->evaluates_every_step);

    CHECK(h3_compute_mode_count() == 3 && !h3_compute_mode(3));
    CHECK(h3_compute_mode(H3_COMPUTE_ATTENTION_CACHE)->needs_tensor_ops &&
          h3_compute_mode(H3_COMPUTE_ATTENTION_CACHE)->needs_cache_file);
    CHECK(!h3_compute_mode(H3_COMPUTE_RESIDENT)->streams_blocks &&
          !h3_compute_mode(H3_COMPUTE_RESIDENT)->needs_tensor_ops);
    CHECK(h3_compute_mode(H3_COMPUTE_SSD_STREAMING)->streams_blocks &&
          !strcmp(h3_compute_mode(H3_COMPUTE_SSD_STREAMING)->name, "ssdStreaming"));

    if (failures) { fprintf(stderr, "%d of %d model family checks failed\n", failures, checks); return 1; }
    printf("h3 model family tests passed (%d checks)\n", checks);
    return 0;
}
