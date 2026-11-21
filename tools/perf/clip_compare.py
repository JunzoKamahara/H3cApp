#!/usr/bin/env python3
"""How far generated clips are from a reference clip of the same request.

Compares the frames written before encoding (H3SPIKE_DUMP: packed RGB24)
and the saved latents (H3SPIKE_SAVE_LATENT), as tools/perf/p0_run.sh leaves
them when those two are given:

    tools/perf/clip_compare.py <width> <height> <reference> <clip> [clip ...]

where each name is a path without its extension (<name>.rgb, <name>.h3lt).
Prints, for each clip: PSNR over all frames and of the worst frame, SSIM
(luma, 7x7 windows, mean over frames and the worst frame), and the latents'
relative error (root mean square of the difference over that of the
reference) for video and audio.
"""
import sys

import numpy as np


def frames(path, width, height):
    data = np.fromfile(path + ".rgb", dtype=np.uint8)
    return data.reshape(-1, height, width, 3)


def latents(path):
    header = np.fromfile(path + ".h3lt", dtype=np.int32, count=9)
    values = np.fromfile(path + ".h3lt", dtype=np.float32, offset=36)
    video = int(header[2]) * int(header[3]) * int(header[4]) * int(header[5])
    return values[:video], values[video:]


def box(image, size):
    """Mean over size x size windows (valid positions only)."""
    total = np.cumsum(np.cumsum(image, axis=0), axis=1)
    total = np.pad(total, ((1, 0), (1, 0)))
    return (total[size:, size:] - total[:-size, size:] -
            total[size:, :-size] + total[:-size, :-size]) / (size * size)


def ssim(a, b, size=7):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mean_a, mean_b = box(a, size), box(b, size)
    var_a = box(a * a, size) - mean_a * mean_a
    var_b = box(b * b, size) - mean_b * mean_b
    cov = box(a * b, size) - mean_a * mean_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(np.mean(((2 * mean_a * mean_b + c1) * (2 * cov + c2)) /
                         ((mean_a ** 2 + mean_b ** 2 + c1) *
                          (var_a + var_b + c2))))


def luma(frame):
    frame = frame.astype(np.float64)
    return (0.299 * frame[..., 0] + 0.587 * frame[..., 1] +
            0.114 * frame[..., 2])


def relative(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    width, height = int(sys.argv[1]), int(sys.argv[2])
    reference = frames(sys.argv[3], width, height)
    reference_video, reference_audio = latents(sys.argv[3])
    print("%-28s %7s %7s %7s %7s %8s %8s" % (
        "clip", "PSNR", "worst", "SSIM", "worst", "video", "audio"))
    for name in sys.argv[4:]:
        clip = frames(name, width, height)
        if clip.shape != reference.shape:
            print("%-28s frames differ in number or size" %
                  name.split("/")[-1])
            continue
        squared = (clip.astype(np.float64) - reference) ** 2
        per_frame = squared.reshape(len(clip), -1).mean(axis=1)
        whole = per_frame.mean()
        similarity = [ssim(luma(clip[index]), luma(reference[index]))
                      for index in range(len(clip))]
        video, audio = latents(name)
        print("%-28s %7.2f %7.2f %7.4f %7.4f %8.4f %8.4f" % (
            name.split("/")[-1],
            10 * np.log10(255.0 ** 2 / whole) if whole else float("inf"),
            10 * np.log10(255.0 ** 2 / per_frame.max())
            if per_frame.max() else float("inf"),
            float(np.mean(similarity)), float(np.min(similarity)),
            relative(video, reference_video),
            relative(audio, reference_audio)))


if __name__ == "__main__":
    main()
