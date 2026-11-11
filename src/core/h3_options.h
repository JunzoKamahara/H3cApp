/* The engine's environment switches in one place. Almost all of them are
 * for development: turning one kernel off to compare, tracing, dumping.
 * The app sets a handful per job (H3Engine.swift) and nothing else needs
 * any of them.
 *
 * Code reads a switch with h3_option("H3_...") - getenv and nothing more -
 * and every name read that way has a row in h3_options.c saying what it
 * does. tools/check_options.py (part of `make test`) fails when a name is
 * read without a row, a row is never read, or getenv is called on an H3_
 * name directly. `build/bin/h3_options` prints the table. */
#ifndef H3_OPTIONS_H
#define H3_OPTIONS_H

#include <stddef.h>
#include <stdlib.h>

/* The switch's value, or NULL when it is not set. */
static inline const char *h3_option(const char *name) {
    return getenv(name);
}

typedef struct {
    const char *name;
    const char *area;    /* what part of the engine it belongs to */
    const char *value;   /* what it takes: "set" (present or not), "0|1", "<n>", ... */
    const char *help;
} h3_option_info;

/* Every switch, sorted by name. */
const h3_option_info *h3_options(size_t *count);

#endif
