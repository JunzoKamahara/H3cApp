/* Prints the engine's environment switches (h3_options.h): name, what it
 * takes and what it does, by area; a switch that is set shows its value.
 *   build/bin/h3_options [area]                                        */
#include "h3_options.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    size_t count = 0;
    const h3_option_info *options = h3_options(&count);
    const char *only = argc > 1 ? argv[1] : NULL;
    const char *areas[64];
    size_t area_count = 0;
    for (size_t i = 0; i < count; i++) {
        size_t a = 0;
        while (a < area_count && strcmp(areas[a], options[i].area)) a++;
        if (a == area_count && area_count < sizeof(areas) / sizeof(*areas)) areas[area_count++] = options[i].area;
    }
    for (size_t a = 0; a < area_count; a++) {
        if (only && strcmp(only, areas[a])) continue;
        printf("[%s]\n", areas[a]);
        for (size_t i = 0; i < count; i++) {
            if (strcmp(options[i].area, areas[a])) continue;
            const char *now = h3_option(options[i].name);
            printf("  %s=%s\n      %s\n", options[i].name, options[i].value, options[i].help);
            if (now) printf("      now: \"%s\"\n", now);
        }
    }
    return 0;
}
