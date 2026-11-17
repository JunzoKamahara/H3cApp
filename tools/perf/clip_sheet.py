#!/usr/bin/env python3
"""Frames of several clips of the same request, side by side.

Reads the frames written before encoding (H3SPIKE_DUMP: packed RGB24) and
writes one image: a row for each clip, the same evenly spaced frames in
each row, the clip's name under its first frame.

    tools/perf/clip_sheet.py <width> <height> <out.png> <clip> [clip ...]

where each clip is a path without its extension (<clip>.rgb). Needs Pillow.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

COLUMNS = 6
SCALE = 2  # each frame is shown at 1/SCALE of its size


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    width, height = int(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[4:]
    cell_width, cell_height = width // SCALE, height // SCALE
    label = 16
    sheet = Image.new("RGB", (cell_width * COLUMNS,
                              (cell_height + label) * len(names)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(names):
        frames = np.fromfile(name + ".rgb", dtype=np.uint8).reshape(
            -1, height, width, 3)
        top = row * (cell_height + label)
        for column in range(COLUMNS):
            index = round(column * (len(frames) - 1) / (COLUMNS - 1))
            frame = Image.fromarray(frames[index]).resize(
                (cell_width, cell_height), Image.LANCZOS)
            sheet.paste(frame, (column * cell_width, top))
        draw.text((4, top + cell_height + 2), name.split("/")[-1],
                  fill="black")
    sheet.save(sys.argv[3])


if __name__ == "__main__":
    main()
