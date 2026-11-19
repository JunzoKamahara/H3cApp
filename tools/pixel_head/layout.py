#!/usr/bin/env python3
"""Which pixels a video token stands for, by the engine's own rules
(src/core/h3_host.c): the latent step i holds h3_frame_per_token[i % 5] frames
(1, 4, 4, 4, 4, ...), and a token is 2x2 latent cells = 32x32 pixels.

  rows of the DiT          [latent_t * token_h * token_w, ...], t outermost
  frames of the video VAE  [frames, token_h * 32, token_w * 32, 3]

Run as a program it checks itself: every frame count the engine accepts splits
into such groups with no frame left out and none taken twice, and frames ->
patches -> frames is the identity."""
import numpy as np

FRAMES_PER_STEP = (1, 4, 4, 4, 4)   # h3_frame_per_token
TOKEN_PIXELS = 32                   # VAE 16x, then the DiT's 2x2 patch


def align_frame_count(requested):   # h3_align_frame_count
    value = max(requested, 5)
    remainder = (value - 5) % 17
    return value + (17 - remainder if remainder else 0)


def latent_steps(frame_count):      # h3_video_latent_t
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def groups(latent_t, pattern=FRAMES_PER_STEP):
    """(first frame, frame count) of every latent step."""
    result, frame = [], 0
    for step in range(latent_t):
        count = pattern[step % len(pattern)]
        result.append((frame, count))
        frame += count
    return result


def patches(frames, latent_t, pattern=FRAMES_PER_STEP):
    """Frames [F, H, W, 3] -> one array per group size: {count: (steps,
    [n, token_h, token_w, count, 32, 32, 3])}, steps being the latent steps
    of that size in order."""
    total, height, width, channels = frames.shape
    token_h, token_w = height // TOKEN_PIXELS, width // TOKEN_PIXELS
    by_count = {}
    for step, (first, count) in enumerate(groups(latent_t, pattern)):
        block = frames[first:first + count].reshape(
            count, token_h, TOKEN_PIXELS, token_w, TOKEN_PIXELS, channels)
        by_count.setdefault(count, ([], []))
        by_count[count][0].append(step)
        by_count[count][1].append(block.transpose(1, 3, 0, 2, 4, 5))
    return {count: (steps, np.stack(blocks))
            for count, (steps, blocks) in by_count.items()}


def frames_of(by_count, latent_t, pattern=FRAMES_PER_STEP):
    """The inverse of patches()."""
    sample = next(iter(by_count.values()))[1]
    _, token_h, token_w, _, _, _, channels = sample.shape
    spans = groups(latent_t, pattern)
    total = spans[-1][0] + spans[-1][1]
    frames = np.empty((total, token_h * TOKEN_PIXELS, token_w * TOKEN_PIXELS,
                       channels), dtype=sample.dtype)
    for count, (steps, blocks) in by_count.items():
        for step, block in zip(steps, blocks):
            first = spans[step][0]
            frames[first:first + count] = block.transpose(2, 0, 3, 1, 4, 5).reshape(
                count, token_h * TOKEN_PIXELS, token_w * TOKEN_PIXELS, channels)
    return frames


def check():
    checked = 0
    for requested in range(1, 400):
        frame_count = align_frame_count(requested)
        latent_t = latent_steps(frame_count)
        spans = groups(latent_t)
        covered = [frame for first, count in spans for frame in range(first, first + count)]
        assert covered == list(range(frame_count)), (requested, frame_count)
        checked += 1
    # 672x384, 120 frames asked for: 124 frames, 21x12 tokens, 37 steps.
    frame_count = align_frame_count(120)
    latent_t = latent_steps(frame_count)
    sizes = [count for _, count in groups(latent_t)]
    assert (frame_count, latent_t, sizes.count(1), sizes.count(4)) == (124, 37, 8, 29)
    rng = np.random.default_rng(0)
    for frame_count, height, width in ((22, 64, 96), (56, 96, 64), (124, 384, 672)):
        latent_t = latent_steps(frame_count)
        frames = rng.integers(0, 256, (frame_count, height, width, 3), dtype=np.uint8)
        split = patches(frames, latent_t)
        assert sum(blocks.shape[0] for _, blocks in split.values()) == latent_t
        assert np.array_equal(frames_of(split, latent_t), frames)
    print(f"ok: {checked} frame counts split into groups of 1 and 4 with no frame "
          "left out or taken twice; 124 frames = 8 groups of 1 + 29 of 4; "
          "frames -> patches -> frames is the identity")


if __name__ == "__main__":
    check()
