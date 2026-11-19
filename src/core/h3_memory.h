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

#include <stddef.h>
#include <stdint.h>

/* Physical footprint of this process in bytes (0 if it cannot be read). */
uint64_t h3_memory_footprint(void);

/* Starts the sampling thread once, when either variable above is set. */
void h3_memory_watch_start(void);

/* Marks the start of a phase. With H3_MEMORY_TRACE, a change of phase
 * prints the highest footprint seen during the previous one. NULL ends
 * the last phase. */
void h3_memory_phase(const char *phase);

/* Large scratch that is filled once and dropped, many times over: a GGUF
 * text encoder's read and expansion buffers. A released block is kept for
 * the next request of the same size, and h3_memory_scratch_drain() returns
 * them all to the system. Taken from malloc instead, macOS keeps the freed
 * blocks for reuse and they stay in the footprint: a layer's worth (1.1 GiB)
 * for every layer read ahead was still counted at the end of the denoise,
 * and malloc_zone_pressure_relief() did not return them. Returning every
 * block as it is released costs the text encoder 1.8 s in page faults.
 * NULL when the memory cannot be had. Safe from several threads.
 * H3_KEEP_FREED_MEMORY=1 uses malloc and free, for comparison. */
void *h3_memory_scratch(size_t bytes);
void h3_memory_scratch_free(void *scratch, size_t bytes);
void h3_memory_scratch_drain(void);

#endif
