#ifndef H3_METAL_H
#define H3_METAL_H

#include "h3.h"

int h3_metal_probe(h3_device_info *info, char *error, size_t error_size);

/* An Objective-C autorelease pool around a stretch of C code. Some Metal
 * objects the GPU layer makes are autoreleased, and without a pool of
 * their own they live until the caller's drains - the whole generation in
 * the app and the CLI. Each Qwen vision call (its own GPU and weights) left
 * about 1 GiB that way: 11 GiB more for the 11 two-frame blocks of a
 * 243-frame reference video. Pops must pair with pushes, innermost first. */
void *h3_autorelease_push(void);
void h3_autorelease_pop(void *pool);

#endif
