/* Process memory: the current footprint, and two opt-in developer aids
 * driven by the environment.
 *
 *   H3_MEMORY_TRACE=1      print the footprint each generation phase
 *                          reached (sampled every 20 ms) when it ends.
 *   H3_MEMORY_LIMIT_GIB=n  end the process as soon as its footprint passes
 *                          n GiB, before the Mac runs out of memory.
 *
 * The limit is for measuring how much memory a job takes: a job that would
 * exhaust memory otherwise swaps until the kernel watchdog restarts the
 * Mac, and a watcher in another process (ps, footprint) stalls along with
 * everything else by then. This one is a thread inside the process that
 * reads its own footprint with one system call. The app never sets it. */
#ifndef H3_MEMORY_H
#define H3_MEMORY_H

#include <stdint.h>

/* Physical footprint of this process in bytes (0 if it cannot be read). */
uint64_t h3_memory_footprint(void);

/* Starts the sampling thread once, when either variable above is set. */
void h3_memory_watch_start(void);

/* Marks the start of a phase. With H3_MEMORY_TRACE, a change of phase
 * prints the highest footprint seen during the previous one. NULL ends
 * the last phase. */
void h3_memory_phase(const char *phase);

#endif
