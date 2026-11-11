#include "h3_options.h"
#include "h3_memory.h"

#include <libproc.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

static const double gib = 1024.0 * 1024.0 * 1024.0;

static pthread_once_t watch_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t phase_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t phase_peak;
static uint64_t limit_bytes;
static int tracing;
static char phase_name[64];

uint64_t h3_memory_footprint(void) {
    struct rusage_info_v4 info;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&info))
        return 0;
    return info.ri_phys_footprint;
}

static void raise_peak(uint64_t now) {
    uint64_t seen = atomic_load(&phase_peak);
    while (now > seen && !atomic_compare_exchange_weak(&phase_peak, &seen, now)) {
    }
}

static void *watch(void *unused) {
    (void)unused;
    for (;;) {
        uint64_t now = h3_memory_footprint();
        raise_peak(now);
        if (limit_bytes && now > limit_bytes) {
            /* No locks, no allocation: the phase name may be mid-copy,
             * which only garbles this one message. */
            fprintf(stderr,
                    "h3: memory guard: footprint %.2f GiB passed "
                    "H3_MEMORY_LIMIT_GIB (%.2f GiB) during %.63s; "
                    "ending the process\n",
                    (double)now / gib, (double)limit_bytes / gib, phase_name);
            _exit(86);
        }
        usleep(20000);
    }
    return NULL;
}

static void start(void) {
    const char *trace = h3_option("H3_MEMORY_TRACE");
    const char *limit = h3_option("H3_MEMORY_LIMIT_GIB");
    tracing = trace && *trace && strcmp(trace, "0");
    if (limit && *limit) {
        double value = strtod(limit, NULL);
        if (value >= 0.5) limit_bytes = (uint64_t)(value * gib);
        else fprintf(stderr, "h3: warning: H3_MEMORY_LIMIT_GIB=%s is below "
                     "0.5; ignoring\n", limit);
    }
    if (!tracing && !limit_bytes) return;
    atomic_store(&phase_peak, h3_memory_footprint());
    pthread_t thread;
    if (pthread_create(&thread, NULL, watch, NULL)) {
        fprintf(stderr, "h3: warning: cannot start the memory watcher\n");
        return;
    }
    pthread_detach(thread);
}

void h3_memory_watch_start(void) {
    pthread_once(&watch_once, start);
}

void h3_memory_phase(const char *phase) {
    if (!tracing) return;
    pthread_mutex_lock(&phase_lock);
    if (!phase || strncmp(phase, phase_name, sizeof(phase_name) - 1)) {
        uint64_t now = h3_memory_footprint();
        raise_peak(now);
        if (phase_name[0])
            fprintf(stderr, "h3: memory: %s peaked at %.2f GiB "
                    "(now %.2f GiB)\n", phase_name,
                    (double)atomic_load(&phase_peak) / gib, (double)now / gib);
        atomic_store(&phase_peak, now);
        snprintf(phase_name, sizeof(phase_name), "%s", phase ? phase : "");
    }
    pthread_mutex_unlock(&phase_lock);
}
