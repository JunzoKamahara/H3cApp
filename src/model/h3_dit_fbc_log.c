/* H3_DIT_FBC_LOG=<file>: what a FirstBlockCache-style reuse would have to go
 * on, measured without reusing anything. The hidden state is read back at
 * three points of every step - before the first block, after it, after the
 * last block - and the file gets, for every pair of steps, the indicator that
 * scheme decides by and the error its decision would leave.
 *
 * With R the first block's residual (hidden after it minus hidden before
 * it), T the remaining blocks' residual and C = R + T the whole core's, of
 * step t and an earlier step s:
 *
 *   head  mean|R_t - R_s| / mean|R_s|   the indicator (Diffusers'
 *         FirstBlockCache reuses when it is at most its threshold; s is the
 *         last step it computed in full)
 *   tail  mean|T_t - T_s| / mean|C_t|   the error of step t's core residual
 *         if the remaining blocks were reused from step s
 *   core  mean|C_t - C_s| / mean|C_t|   the same if the whole core were
 *         (the fixed-interval core reuse)
 *
 * each over all rows and over the text, reference/condition, audio and
 * video rows. Pairs are computed on a sample of rows (at most SAMPLE_ROWS a
 * segment, weighted back); the indicator between consecutive steps is also
 * computed on every row ("exact" lines), which shows what the sample is
 * worth. The run itself is not changed: the reads only add waits. Not with
 * token reduction or core reuse, which move or skip what is read here. */
#include "h3_dit_private.h"

enum { FBC_KINDS = 4, SAMPLE_ROWS = 256 };
static const char *const kind_names[FBC_KINDS] = {
    "text", "reference", "audio", "video"};

struct h3_dit_fbc_log {
    char *path;
    size_t sampled;            /* rows in the sample */
    uint32_t *rows;            /* their row numbers */
    double *weights;           /* rows each one stands for */
    uint8_t *sample_kinds;
    uint8_t *row_kinds;        /* every row's kind */
    size_t kind_rows[FBC_KINDS];
    int steps;                 /* steps logged so far */
    int step_index[H3_MAX_STEPS];
    float *head[H3_MAX_STEPS]; /* sampled R of each logged step */
    float *tail[H3_MAX_STEPS]; /* sampled T */
    uint16_t *input;           /* this step's hidden before the first block */
    uint16_t *scratch;         /* a hidden state just read */
    float *first;              /* sampled hidden after the first block */
    float *previous;           /* the previous logged step's R, every row */
    int has_previous;
    /* Consecutive-step indicator on every row: all rows, then each kind. */
    double exact[H3_MAX_STEPS][FBC_KINDS + 1];
};

static float bf16_value(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static int kind_of(h3_segment_kind kind) {
    switch (kind) {
    case H3_SEG_TEXT: return 0;
    case H3_SEG_AUDIO: return 2;
    case H3_SEG_VIDEO: return 3;
    default: return 1;
    }
}

void h3_dit_fbc_log_free(h3_dit_fbc_log *log) {
    if (!log) return;
    for (int step = 0; step < H3_MAX_STEPS; step++) {
        free(log->head[step]);
        free(log->tail[step]);
    }
    free(log->path); free(log->rows); free(log->weights);
    free(log->sample_kinds); free(log->row_kinds); free(log->input);
    free(log->scratch); free(log->first); free(log->previous);
    free(log);
}

static h3_dit_fbc_log *fbc_log_create(const h3_dit *dit, const char *path) {
    h3_dit_fbc_log *log = calloc(1, sizeof(*log));
    if (!log) return NULL;
    size_t sequence = dit->sequence, elements = sequence * HIDDEN;
    size_t capacity = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        size_t rows = dit->layout.segments[index].stop -
                      dit->layout.segments[index].start;
        capacity += rows < SAMPLE_ROWS ? rows : SAMPLE_ROWS;
    }
    log->path = strdup(path);
    log->rows = malloc((capacity ? capacity : 1) * sizeof(*log->rows));
    log->weights = malloc((capacity ? capacity : 1) * sizeof(*log->weights));
    log->sample_kinds = malloc(capacity ? capacity : 1);
    log->row_kinds = calloc(sequence ? sequence : 1, 1);
    log->input = malloc(elements * sizeof(*log->input));
    log->scratch = malloc(elements * sizeof(*log->scratch));
    log->previous = malloc(elements * sizeof(*log->previous));
    if (!log->path || !log->rows || !log->weights || !log->sample_kinds ||
        !log->row_kinds || !log->input || !log->scratch || !log->previous) {
        h3_dit_fbc_log_free(log);
        return NULL;
    }
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        const h3_segment *segment = &dit->layout.segments[index];
        size_t rows = segment->stop - segment->start;
        if (!rows || segment->stop > sequence) continue;
        int kind = kind_of(segment->kind);
        log->kind_rows[kind] += rows;
        memset(log->row_kinds + segment->start, kind, rows);
        /* An odd stride, so that a grid's columns are not sampled alike. */
        size_t stride = (rows + SAMPLE_ROWS - 1) / SAMPLE_ROWS;
        if (stride > 1 && stride % 2 == 0) stride++;
        size_t count = (rows + stride - 1) / stride;
        for (size_t pick = 0; pick < count; pick++) {
            log->rows[log->sampled] = (uint32_t)(segment->start + pick * stride);
            log->weights[log->sampled] = (double)rows / (double)count;
            log->sample_kinds[log->sampled] = (uint8_t)kind;
            log->sampled++;
        }
    }
    log->first = malloc((log->sampled ? log->sampled : 1) * HIDDEN *
                        sizeof(*log->first));
    if (!log->first) {
        h3_dit_fbc_log_free(log);
        return NULL;
    }
    return log;
}

/* The sampled rows of a hidden state just read, as floats. */
static void sample_rows(const h3_dit_fbc_log *log, const uint16_t *hidden,
                        float *sample) {
    for (size_t pick = 0; pick < log->sampled; pick++) {
        const uint16_t *row = hidden + (size_t)log->rows[pick] * HIDDEN;
        float *target = sample + pick * HIDDEN;
        for (size_t column = 0; column < HIDDEN; column++)
            target[column] = bf16_value(row[column]);
    }
}

int h3_dit_fbc_log_capture(h3_dit *dit, int step, int stage, char *error,
                           size_t error_size) {
    const char *path = h3_option("H3_DIT_FBC_LOG");
    if (!path || !*path) return 1;
    if (dit->token_reduction || dit->core_reuse_interval > 1) {
        h3_dit_fail(error, error_size, "H3_DIT_FBC_LOG measures the exact "
             "run: turn token reduction and core reuse off");
        return 0;
    }
    if (!dit->fbc_log) dit->fbc_log = fbc_log_create(dit, path);
    h3_dit_fbc_log *log = dit->fbc_log;
    if (!log) {
        h3_dit_fail(error, error_size, "out of memory for H3_DIT_FBC_LOG");
        return 0;
    }
    if (log->steps >= H3_MAX_STEPS) return 1;
    size_t sequence = dit->sequence, elements = sequence * HIDDEN;
    size_t sample_elements = log->sampled * HIDDEN;
    uint16_t *target = stage == 0 ? log->input : log->scratch;
    if (!h3_dit_gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                       "submit before the reuse log's read")) return 0;
    if (!h3_gpu_tensor_read_bf16(dit->hidden, target, elements)) {
        h3_dit_fail(error, error_size, "cannot read the hidden state for "
             "H3_DIT_FBC_LOG: %s", h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!h3_dit_gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                       "continue after the reuse log's read")) return 0;
    if (stage == 0) return 1;
    int slot = log->steps;
    if (stage == 1) {
        /* R on every row against the previous step's, then kept for the
         * next one. */
        double changed[FBC_KINDS] = {0}, before[FBC_KINDS] = {0};
        for (size_t row = 0; row < sequence; row++) {
            const uint16_t *in = log->input + row * HIDDEN;
            const uint16_t *out = log->scratch + row * HIDDEN;
            float *kept = log->previous + row * HIDDEN;
            double row_changed = 0.0, row_before = 0.0;
            for (size_t column = 0; column < HIDDEN; column++) {
                float residual = bf16_value(out[column]) -
                                 bf16_value(in[column]);
                row_changed += fabs((double)residual - (double)kept[column]);
                row_before += fabs((double)kept[column]);
                kept[column] = residual;
            }
            changed[log->row_kinds[row]] += row_changed;
            before[log->row_kinds[row]] += row_before;
        }
        double all_changed = 0.0, all_before = 0.0;
        for (int kind = 0; kind < FBC_KINDS; kind++) {
            all_changed += changed[kind];
            all_before += before[kind];
            log->exact[slot][kind + 1] = log->has_previous && before[kind] > 0.0
                ? changed[kind] / before[kind] : -1.0;
        }
        log->exact[slot][0] = log->has_previous && all_before > 0.0
            ? all_changed / all_before : -1.0;
        log->has_previous = 1;
        log->head[slot] = malloc((sample_elements ? sample_elements : 1) *
                                 sizeof(float));
        log->tail[slot] = malloc((sample_elements ? sample_elements : 1) *
                                 sizeof(float));
        if (!log->head[slot] || !log->tail[slot]) {
            h3_dit_fail(error, error_size, "out of memory for H3_DIT_FBC_LOG");
            return 0;
        }
        sample_rows(log, log->input, log->head[slot]);
        sample_rows(log, log->scratch, log->first);
        for (size_t index = 0; index < sample_elements; index++)
            log->head[slot][index] = log->first[index] - log->head[slot][index];
        return 1;
    }
    if (!log->head[slot] || !log->tail[slot]) return 1;
    sample_rows(log, log->scratch, log->tail[slot]);
    for (size_t index = 0; index < sample_elements; index++)
        log->tail[slot][index] -= log->first[index];
    log->step_index[slot] = step;
    log->steps++;
    return 1;
}

/* Weighted sums of |a| or |a - b| over the sampled rows, per kind. */
static void sample_sums(const h3_dit_fbc_log *log, const float *a,
                        const float *b, double sums[FBC_KINDS]) {
    for (int kind = 0; kind < FBC_KINDS; kind++) sums[kind] = 0.0;
    for (size_t pick = 0; pick < log->sampled; pick++) {
        const float *left = a + pick * HIDDEN;
        const float *right = b ? b + pick * HIDDEN : NULL;
        double sum = 0.0;
        for (size_t column = 0; column < HIDDEN; column++)
            sum += fabs(right ? (double)left[column] - (double)right[column]
                              : (double)left[column]);
        sums[log->sample_kinds[pick]] += log->weights[pick] * sum;
    }
}

static void print_ratios(FILE *file, const double top[FBC_KINDS],
                         const double bottom[FBC_KINDS]) {
    double all_top = 0.0, all_bottom = 0.0;
    for (int kind = 0; kind < FBC_KINDS; kind++) {
        all_top += top[kind];
        all_bottom += bottom[kind];
    }
    fprintf(file, "\t%.6f", all_bottom > 0.0 ? all_top / all_bottom : -1.0);
    for (int kind = 0; kind < FBC_KINDS; kind++)
        fprintf(file, "\t%.6f",
                bottom[kind] > 0.0 ? top[kind] / bottom[kind] : -1.0);
}

void h3_dit_fbc_log_write(const h3_dit *dit) {
    const h3_dit_fbc_log *log = dit->fbc_log;
    if (!log || !log->steps) return;
    size_t sample_elements = log->sampled * HIDDEN;
    float *core[H3_MAX_STEPS] = {0};
    FILE *file = fopen(log->path, "w");
    int ok = file != NULL;
    for (int step = 0; ok && step < log->steps; step++) {
        core[step] = malloc((sample_elements ? sample_elements : 1) *
                            sizeof(float));
        if (!core[step]) { ok = 0; break; }
        for (size_t index = 0; index < sample_elements; index++)
            core[step][index] = log->head[step][index] + log->tail[step][index];
    }
    if (ok) {
        fprintf(file, "# h3 reuse log 1: %u tokens, %zu sampled rows, %d steps\n",
                (unsigned)dit->sequence, log->sampled, log->steps);
        fprintf(file, "# rows:");
        for (int kind = 0; kind < FBC_KINDS; kind++)
            fprintf(file, " %s %zu", kind_names[kind], log->kind_rows[kind]);
        fprintf(file, "\n# every ratio is given for: all");
        for (int kind = 0; kind < FBC_KINDS; kind++)
            fprintf(file, " %s", kind_names[kind]);
        fprintf(file, " (-1: no such rows)\n");
        fprintf(file, "# step <step> <video sigma> then mean|R| mean|T| "
                      "mean|C| over all rows (sampled)\n");
        fprintf(file, "# exact <step> <previous step> then head on every row\n");
        fprintf(file, "# pair <step> <earlier step> then head x5, tail x5, "
                      "core x5 (sampled)\n");
        double rows = (double)dit->sequence * (double)HIDDEN;
        for (int step = 0; step < log->steps; step++) {
            double head[FBC_KINDS], tail[FBC_KINDS], whole[FBC_KINDS];
            sample_sums(log, log->head[step], NULL, head);
            sample_sums(log, log->tail[step], NULL, tail);
            sample_sums(log, core[step], NULL, whole);
            double sums[3] = {0};
            for (int kind = 0; kind < FBC_KINDS; kind++) {
                sums[0] += head[kind];
                sums[1] += tail[kind];
                sums[2] += whole[kind];
            }
            int index = log->step_index[step];
            fprintf(file, "step\t%d\t%.6f\t%.6g\t%.6g\t%.6g\n", index,
                    index >= 0 && index <= dit->sigmas.steps
                        ? (double)dit->sigmas.video[index] : -1.0,
                    sums[0] / rows, sums[1] / rows, sums[2] / rows);
        }
        for (int step = 1; step < log->steps; step++) {
            fprintf(file, "exact\t%d\t%d", log->step_index[step],
                    log->step_index[step - 1]);
            for (int column = 0; column <= FBC_KINDS; column++)
                fprintf(file, "\t%.6f", log->exact[step][column]);
            fprintf(file, "\n");
        }
        for (int step = 1; step < log->steps; step++) {
            double whole[FBC_KINDS];
            sample_sums(log, core[step], NULL, whole);
            for (int earlier = 0; earlier < step; earlier++) {
                double top[FBC_KINDS], bottom[FBC_KINDS];
                fprintf(file, "pair\t%d\t%d", log->step_index[step],
                        log->step_index[earlier]);
                sample_sums(log, log->head[step], log->head[earlier], top);
                sample_sums(log, log->head[earlier], NULL, bottom);
                print_ratios(file, top, bottom);
                sample_sums(log, log->tail[step], log->tail[earlier], top);
                print_ratios(file, top, whole);
                sample_sums(log, core[step], core[earlier], top);
                print_ratios(file, top, whole);
                fprintf(file, "\n");
            }
        }
    }
    if (file && fclose(file) != 0) ok = 0;
    for (int step = 0; step < H3_MAX_STEPS; step++) free(core[step]);
    fprintf(stderr, ok ? "h3: reuse log: %d steps written to %s\n"
                       : "h3: reuse log: cannot write %d steps to %s\n",
            log->steps, log->path);
}
