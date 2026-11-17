#!/usr/bin/env python3
"""The steps H3_DIT_CORE_REUSE_CURVE computes the DiT core at.

Same rule as core_reuse_curve_computes in src/model/h3_dit_block.c: as many
computed steps as the fixed interval gives (every Nth step and the last),
the j-th of the M at (steps - 1) * (j / (M - 1))^exponent, rounded, and at
least one after the one before.

    tools/perf/reuse_curve.py <steps> <interval> [exponent ...]

Prints, for the fixed interval and for each exponent (default 1.3), the
computed steps, the reused steps and the longest run of reused steps.
"""
import math
import sys


def fixed_steps(steps, interval):
    return [step for step in range(steps)
            if step % interval == 0 or step == steps - 1]


def curve_steps(steps, interval, exponent):
    count = len(fixed_steps(steps, interval))
    computed = []
    for index in range(count):
        at = (int(math.floor((steps - 1) *
                             (index / (count - 1)) ** exponent + 0.5))
              if count > 1 else 0)
        if computed and at <= computed[-1]:
            at = computed[-1] + 1
        computed.append(at)
    return computed


def longest_run(steps, computed):
    longest = current = 0
    for step in range(steps):
        current = 0 if step in computed else current + 1
        longest = max(longest, current)
    return longest


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    steps, interval = int(sys.argv[1]), int(sys.argv[2])
    exponents = [float(value) for value in sys.argv[3:]] or [1.3]
    rows = [("every %d" % interval, fixed_steps(steps, interval))] + [
        ("curve %g" % exponent, curve_steps(steps, interval, exponent))
        for exponent in exponents]
    for name, computed in rows:
        print("%-10s computes %s" % (
            name, ",".join(str(step) for step in computed)))
        print("%-10s reuses   %s (%d, longest run %d)" % (
            "", ",".join(str(step) for step in range(steps)
                         if step not in computed),
            steps - len(computed), longest_run(steps, computed)))


if __name__ == "__main__":
    main()
