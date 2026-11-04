#ifndef H3_JSON_H
#define H3_JSON_H

/* The small JSON cursor h3_safetensors.c uses for safetensors headers,
 * shared with readers of JSON that travels inside a header's string
 * metadata (h3_refmod.c's refmod_meta). Values come from untrusted files:
 * every function fails cleanly on malformed input and h3_json_skip bounds
 * its nesting depth. */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *at;
    const char *end;
    char *error;
    size_t error_size;
    unsigned depth;
} h3_json_cursor;

int h3_json_fail(h3_json_cursor *cursor, const char *message);
void h3_json_ws(h3_json_cursor *cursor);
/* Consumes one expected punctuation character after whitespace. */
int h3_json_take(h3_json_cursor *cursor, char expected);
/* Peeks at the next non-whitespace character, or '\0' at the end. */
char h3_json_peek(h3_json_cursor *cursor);
/* A string with escapes decoded; the caller frees it. */
char *h3_json_string(h3_json_cursor *cursor);
int h3_json_uint(h3_json_cursor *cursor, uint64_t *result);
/* A JSON integer, optionally negative, that fits an int. */
int h3_json_int(h3_json_cursor *cursor, int *result);
/* Skips one value of any kind. */
int h3_json_skip(h3_json_cursor *cursor);

#endif
