#!/usr/bin/env python3
"""What a FirstBlockCache-style reuse would do on a logged run.

Reads the file H3_DIT_FBC_LOG wrote (src/model/h3_dit_fbc_log.c) and, for a
list of thresholds, walks the steps the way Diffusers' FirstBlockCache does:
a step whose first-block residual differs from the last fully computed
step's by at most the threshold (mean absolute difference over that step's
mean absolute value) reuses the remaining blocks' residual of that step.
Prints how many steps each threshold would reuse and the error that leaves
in the core residual, the same for the fixed-interval core reuse, and the
lowest error any choice of as many steps could reach. "weighted" sums each
reused step's video error times that step's share of the sigma range: the
step an Euler update takes with the wrong velocity.

The log is of an exact run, so this is a first estimate: once a step is
reused the later steps see slightly different inputs.

    tools/perf/fbc_sim.py <log.tsv> [threshold ...]
"""
import sys

KINDS = ["all", "text", "reference", "audio", "video"]


def read(path):
    steps, exact, head, tail, core = [], {}, {}, {}, {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if fields[0] == "step":
            steps.append((int(fields[1]), float(fields[2]),
                          [float(value) for value in fields[3:6]]))
        elif fields[0] == "exact":
            exact[int(fields[1])] = [float(value) for value in fields[3:8]]
        elif fields[0] == "pair":
            key = (int(fields[1]), int(fields[2]))
            values = [float(value) for value in fields[3:18]]
            head[key], tail[key], core[key] = (
                values[0:5], values[5:10], values[10:15])
    return steps, exact, head, tail, core


def summary(errors, shares=()):
    """Mean and largest error of the reused steps, for video and audio."""
    if not errors:
        return "-"
    video = [error[4] for error in errors]
    audio = [error[3] for error in errors]
    text = "video %.3f (max %.3f), audio %.3f (max %.3f)" % (
        sum(video) / len(video), max(video),
        sum(audio) / len(audio), max(audio))
    if shares:
        text += ", weighted %.4f" % sum(
            share * error for share, error in zip(shares, video))
    return text


def rank_correlation(left, right):
    """Spearman's: 1 when one orders the pairs exactly as the other does."""
    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0] * len(values)
        for rank, index in enumerate(order):
            result[index] = rank
        return result
    a, b, count = ranks(left), ranks(right), len(left)
    if count < 3:
        return float("nan")
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(a, b)) / (
        count * (count * count - 1))


def best_choice(order, tail, share, count, weighted):
    """The lowest total video error of reusing `count` steps, the first and
    last steps computed: each reused step takes the last computed one's
    remaining blocks."""
    states = {(order[0], 0): (0.0, [])}
    for step in order[1:]:
        following = {}
        for (reference, used), (cost, reused) in states.items():
            key = (step, used)
            if key not in following or cost < following[key][0]:
                following[key] = (cost, reused)
            if step != order[-1] and used < count:
                error = tail[(step, reference)][4]
                added = cost + (share[step] * error if weighted else error)
                key = (reference, used + 1)
                if key not in following or added < following[key][0]:
                    following[key] = (added, reused + [step])
        states = following
    found = [value for (_, used), value in states.items() if used == count]
    return min(found, key=lambda value: value[0]) if found else None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    steps, exact, head, tail, core = read(sys.argv[1])
    thresholds = [float(value) for value in sys.argv[2:]] or [
        0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
    order = [step for step, _, _ in steps]
    sigmas = [sigma for _, sigma, _ in steps] + [0.0]
    span = (sigmas[0] - sigmas[-1]) or 1.0
    share = {step: (sigmas[position] - sigmas[position + 1]) / span
             for position, step in enumerate(order)}
    print("%d steps logged" % len(order))
    print("\nindicator against the step before (every row | sampled rows),")
    print("and the error of reusing that step's remaining blocks:")
    print("step  sigma   all     video   audio  | all     | tail error video / audio")
    for position, (step, sigma, _) in enumerate(steps):
        if position == 0:
            print("%4d  %.4f  (first step)" % (step, sigma))
            continue
        before = order[position - 1]
        every = exact[step]
        print("%4d  %.4f  %.4f  %.4f  %.4f | %.4f  | %.4f / %.4f" % (
            step, sigma, every[0], every[4], every[3],
            head[(step, before)][0], tail[(step, before)][4],
            tail[(step, before)][3]))

    print("\nFirstBlockCache, by threshold (the last step is always computed"
          " in the second line of each):")
    for threshold in thresholds:
        for keep_last in (False, True):
            reference, reused, errors = order[0], [], []
            for step in order[1:]:
                last = step == order[-1]
                if (head[(step, reference)][0] <= threshold and
                        not (keep_last and last)):
                    reused.append(step)
                    errors.append(tail[(step, reference)])
                else:
                    reference = step
            print("  %.2f%s reuses %2d of %d  %s  steps %s" % (
                threshold, " keep-last" if keep_last else "          ",
                len(reused), len(order),
                summary(errors, [share[step] for step in reused]),
                ",".join(str(step) for step in reused) or "-"))

    print("\nfixed-interval core reuse (the last step is always computed):")
    for interval in (2, 3, 4, 6):
        reference, reused, errors = order[0], [], []
        for position, step in enumerate(order[1:], 1):
            if position % interval == 0 or step == order[-1]:
                reference = step
            else:
                reused.append(step)
                errors.append(core[(step, reference)])
        print("  every %d: reuses %2d of %d  %s" % (
            interval, len(reused), len(order),
            summary(errors, [share[step] for step in reused])))

    print("\nthe best choice of as many steps (lowest mean video error, then"
          " lowest weighted):")
    for count in sorted({len(order) // 4, len(order) // 2,
                         len(order) * 2 // 3}):
        for weighted in (False, True):
            found = best_choice(order, tail, share, count, weighted)
            if not found:
                continue
            reused = found[1]
            reference, errors = order[0], []
            for step in order[1:]:
                if step in reused:
                    errors.append(tail[(step, reference)])
                else:
                    reference = step
            print("  %2d steps%s: %s  steps %s" % (
                count, " weighted" if weighted else "         ",
                summary(errors, [share[step] for step in reused]),
                ",".join(str(step) for step in reused)))

    pairs = [key for key in head if key[0] - key[1] <= 4]
    print("\nrank correlation of the indicator with the video error it "
          "stands for:")
    print("  consecutive steps %.2f, steps up to 4 apart %.2f" % (
        rank_correlation([head[(order[i], order[i - 1])][0]
                          for i in range(1, len(order))],
                         [tail[(order[i], order[i - 1])][4]
                          for i in range(1, len(order))]),
        rank_correlation([head[key][0] for key in pairs],
                         [tail[key][4] for key in pairs])))


if __name__ == "__main__":
    main()
