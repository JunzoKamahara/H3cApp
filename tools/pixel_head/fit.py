#!/usr/bin/env python3
"""How much of the video VAE's output a linear head reads straight from the
last step's video rows (tools/pixel_head/run.sh, batch.sh): the first thing a
pixel head (DWD, arXiv 2611.22287) does, and a check, on real clips, of which
pixels a token stands for (layout.py).

  python3 tools/pixel_head/fit.py [folder]     default ~/H3cApp-models/pi-h3-prep

Fits ridge regressions from a token's 5376 values to its pixels (RGB in
[-1, 1]) on the fit_NN clips and judges them on the held_NN clips, which have
other prompts and seeds:

  1. the tokens of 4 frames -> their 4x32x32 pixels, averaged 4x4 (768 values)
     and whole (12288 values);
  2. the same with the pixels taken from the wrong place: some frames earlier
     or later, some pixels left, right, up or down. If the layout is right the
     fit is best at no shift. And one frame at a time, counted from the
     token's first frame: the frames a token holds should be the ones read
     best from it;
  3. the tokens of 1 frame: a head of their own, the 4-frame head applied to
     them, and one head fitted on both kinds.

What a linear fit cannot tell: the order of the values inside a patch (any
fixed order fits equally well). Writes results.json and held_NN_sheet.png
(top the VAE's frames, bottom the linear head's) beside the clips."""
import glob, json, os, struct, sys, time, zlib
import numpy as np
from layout import TOKEN_PIXELS, frames_of, groups

HIDDEN = 5376


def load(prefix):
    shape = json.load(open(prefix + ".json"))
    bits = np.fromfile(prefix + ".bf16", dtype=np.uint16)
    rows = (bits.astype(np.uint32) << 16).view(np.float32)
    rows = rows.reshape(shape["latent_t"], shape["token_h"], shape["token_w"], HIDDEN)
    height, width = shape["token_h"] * TOKEN_PIXELS, shape["token_w"] * TOKEN_PIXELS
    frames = np.fromfile(prefix + ".rgb", dtype=np.uint8).reshape(-1, height, width, 3)
    spans = groups(shape["latent_t"])
    assert frames.shape[0] == spans[-1][0] + spans[-1][1], prefix
    return rows, frames.astype(np.float32) / 127.5 - 1.0, spans


def pooled(frames, factor):
    total, height, width, channels = frames.shape
    return frames.reshape(total, height // factor, factor, width // factor, factor,
                          channels).mean(axis=(2, 4))


def targets(frames, spans, steps, count, cell, frame_shift=0, x_shift=0, y_shift=0):
    """[len(steps) * token_h * token_w, count * cell * cell * 3]: the pixels of
    each token of the given steps, `cell` pixels to a token's side, taken
    frame_shift frames later and x_shift / y_shift cells to the right / down
    (edges repeat)."""
    total, height, width, _ = frames.shape
    token_h, token_w = height // cell, width // cell
    ys = np.clip(np.arange(height) + y_shift, 0, height - 1)
    xs = np.clip(np.arange(width) + x_shift, 0, width - 1)
    out = np.empty((len(steps), token_h, token_w, count, cell, cell, 3), np.float32)
    for index, step in enumerate(steps):
        picked = np.clip(np.arange(spans[step][0], spans[step][0] + count) + frame_shift,
                         0, total - 1)
        block = frames[picked][:, ys][:, :, xs]
        out[index] = block.reshape(count, token_h, cell, token_w, cell, 3).transpose(
            1, 3, 0, 2, 4, 5)
    return out.reshape(len(steps) * token_h * token_w, -1)


class Sums:
    """X'X, X'Y and the sums a ridge fit with an intercept needs."""
    def __init__(self):
        self.n, self.xx, self.x, self.xy, self.y = 0, np.zeros((HIDDEN, HIDDEN)), np.zeros(HIDDEN), {}, {}

    def add_x(self, x):
        assert np.isfinite(x).all()
        self.n += x.shape[0]
        self.xx += (x.T @ x).astype(np.float64)
        self.x += x.sum(axis=0, dtype=np.float64)

    def add_y(self, key, x, y):
        if key not in self.xy:
            self.xy[key] = np.zeros((HIDDEN, y.shape[1]))
            self.y[key] = np.zeros(y.shape[1])
        self.xy[key] += (x.T @ y).astype(np.float64)
        self.y[key] += y.sum(axis=0, dtype=np.float64)

    def __add__(self, other):
        total = Sums()
        total.n, total.xx, total.x = self.n + other.n, self.xx + other.xx, self.x + other.x
        for key in set(self.xy) & set(other.xy):     # the targets both have
            if self.xy[key].shape == other.xy[key].shape:
                total.xy[key] = self.xy[key] + other.xy[key]
                total.y[key] = self.y[key] + other.y[key]
        return total


class Ridge:
    def __init__(self, sums):
        self.sums, self.mean = sums, sums.x / sums.n
        centred = sums.xx - sums.n * np.outer(self.mean, self.mean)
        self.values, self.vectors = np.linalg.eigh(centred)
        self.scale = max(self.values.mean(), 1e-12)

    def head(self, key, alpha):
        """(weight [HIDDEN, out] float32, bias [out] float32)."""
        mean_y = self.sums.y[key] / self.sums.n
        centred = self.sums.xy[key] - self.sums.n * np.outer(self.mean, mean_y)
        weight = self.vectors @ ((self.vectors.T @ centred) /
                                 (np.maximum(self.values, 0) + alpha * self.scale)[:, None])
        assert np.isfinite(weight).all()
        return weight.astype(np.float32), (mean_y - self.mean @ weight).astype(np.float32)


def psnr(error_sum, count):
    return 10 * np.log10(4.0 / (error_sum / count)) if count else float("nan")


def png(path, image):
    height, width, _ = image.shape
    raw = b"".join(b"\0" + image[row].tobytes() for row in range(height))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(
        ">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    # Accelerate's matmul raises floating-point flags that mean nothing
    # (numpy 2.0 on Apple silicon); what is fitted is checked to be finite.
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    folder = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/H3cApp-models/pi-h3-prep")
    fit = sorted(path[:-5] for path in glob.glob(folder + "/fit_*.json"))
    held = sorted(path[:-5] for path in glob.glob(folder + "/held_*.json"))
    assert len(fit) >= 5 and held, "run tools/pixel_head/batch.sh first"
    tune = fit[-max(1, len(fit) // 5):]            # the fit clips the ridge strength is chosen on
    frame_shifts = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    cell_shifts = (-32, -16, -8, 0, 8, 16, 32)     # pixels; the sweep's targets are 8x8 means
    frame_offsets = tuple(range(-4, 8))
    started = time.time()

    def steps_of(spans, count):
        return [step for step, (_, size) in enumerate(spans) if size == count]

    def gather(prefixes, whole):
        four, one = Sums(), Sums()
        for prefix in prefixes:
            rows, frames, spans = load(prefix)
            fine, coarse = pooled(frames, 4), pooled(frames, 8)
            s4, s1 = steps_of(spans, 4), steps_of(spans, 1)
            x4, x1 = rows[s4].reshape(-1, HIDDEN), rows[s1].reshape(-1, HIDDEN)
            four.add_x(x4); one.add_x(x1)
            four.add_y("fine", x4, targets(fine, spans, s4, 4, 8))
            one.add_y("fine", x1, targets(fine, spans, s1, 1, 8))
            # one head for both kinds: a 1-frame token's frame stands four times
            one.add_y("fine4", x1, np.tile(targets(fine, spans, s1, 1, 8), (1, 4)))
            four.add_y("fine4", x4, targets(fine, spans, s4, 4, 8))
            for shift in frame_shifts:
                four.add_y(("t", shift), x4, targets(coarse, spans, s4, 4, 4, frame_shift=shift))
            for offset in frame_offsets:    # one frame, counted from the token's first
                four.add_y(("f", offset), x4, targets(coarse, spans, s4, 1, 4, frame_shift=offset))
                one.add_y(("f", offset), x1, targets(coarse, spans, s1, 1, 4, frame_shift=offset))
            for shift in cell_shifts:
                four.add_y(("x", shift), x4, targets(coarse, spans, s4, 4, 4, x_shift=shift // 8))
                four.add_y(("y", shift), x4, targets(coarse, spans, s4, 4, 4, y_shift=shift // 8))
            if whole:
                four.add_y("whole", x4, targets(frames, spans, s4, 4, 32))
                one.add_y("whole", x1, targets(frames, spans, s1, 1, 32))
            print(f"  {os.path.basename(prefix)}  {time.time() - started:.0f}s", flush=True)
        return four, one

    def judge(prefixes, heads):
        """heads: {name: (count, key-like spec, weight, bias)} -> error sums."""
        sums = {name: [0.0, 0] for name in heads}
        for prefix in prefixes:
            rows, frames, spans = load(prefix)
            fine, coarse = pooled(frames, 4), pooled(frames, 8)
            for name, (count, make, weight, bias) in heads.items():
                steps = steps_of(spans, count)
                x = rows[steps].reshape(-1, HIDDEN)
                error = make(x @ weight + bias, frames, fine, coarse, spans, steps)
                sums[name][0] += float(error.sum(dtype=np.float64)); sums[name][1] += error.size
        return {name: psnr(*value) for name, value in sums.items()}

    def against(count, cell_source, cell, **shift):
        def make(predicted, frames, fine, coarse, spans, steps):
            source = {"fine": fine, "coarse": coarse, "whole": frames}[cell_source]
            return (predicted - targets(source, spans, steps, count, cell, **shift)) ** 2
        return make

    print("ridge strength, on", len(fit) - len(tune), "clips, judged on", len(tune))
    four_a, one_a = gather(fit[:-len(tune)], whole=False)
    ridge = Ridge(four_a)
    alphas = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    tuned = judge(tune, {alpha: (4, against(4, "fine", 8), *ridge.head("fine", alpha))
                         for alpha in alphas})
    alpha = max(tuned, key=tuned.get)
    print("  " + ", ".join(f"{a:g}: {tuned[a]:.2f} dB" for a in alphas), "->", alpha)

    print("all", len(fit), "fit clips")
    four_b, one_b = gather(tune, whole=False)
    four, one = four_a + four_b, one_a + one_b
    del four_a, one_a, four_b, one_b
    ridge4, ridge1, ridge_both = Ridge(four), Ridge(one), Ridge(four + one)
    heads = {"4-frame tokens, 4x4 means": (4, against(4, "fine", 8), *ridge4.head("fine", alpha))}
    for shift in frame_shifts:
        heads[f"t{shift:+d}"] = (4, against(4, "coarse", 4, frame_shift=shift),
                                 *ridge4.head(("t", shift), alpha))
    for axis in "xy":
        for shift in cell_shifts:
            heads[f"{axis}{shift:+d}"] = (
                4, against(4, "coarse", 4, **{axis + "_shift": shift // 8}),
                *ridge4.head((axis, shift), alpha))
    for offset in frame_offsets:
        heads[f"4-frame tokens, frame {offset:+d} alone"] = (
            4, against(1, "coarse", 4, frame_shift=offset), *ridge4.head(("f", offset), alpha))
        heads[f"1-frame tokens, frame {offset:+d} alone"] = (
            1, against(1, "coarse", 4, frame_shift=offset), *ridge1.head(("f", offset), alpha))
    heads["1-frame tokens, own head"] = (1, against(1, "fine", 8), *ridge1.head("fine", alpha))
    weight4, bias4 = ridge4.head("fine", alpha)
    for slot in range(4):   # the 4-frame head on a 1-frame token: which of its 4 frames is the frame?
        columns = slice(slot * 192, slot * 192 + 192)
        heads[f"1-frame tokens, 4-frame head, frame {slot + 1}"] = (
            1, against(1, "fine", 8), weight4[:, columns], bias4[columns])
    weight_both, bias_both = ridge_both.head("fine4", alpha)
    heads["4-frame tokens, one head for both"] = (4, against(4, "fine", 8), weight_both, bias_both)
    heads["1-frame tokens, one head for both, frame 1"] = (
        1, against(1, "fine", 8), weight_both[:, :192], bias_both[:192])
    # nothing read from the token: the mean patch of the fit clips
    heads["4-frame tokens, mean patch"] = (
        4, against(4, "fine", 8), np.zeros((HIDDEN, 768), np.float32),
        (four.y["fine"] / four.n).astype(np.float32))
    results = judge(held, heads)
    assert all(np.isfinite(value) for value in results.values())
    for name, value in results.items():
        print(f"  {name:46s} {value:6.2f} dB")

    # by latent step (4x4 means): the steps a VAE chunk boundary blends are 5, 6, 10, 11, ...
    by_step = {}
    for prefix in held:
        rows, frames, spans = load(prefix)
        fine = pooled(frames, 4)
        for step, (_, count) in enumerate(spans):
            weight, bias = (weight4, bias4) if count == 4 else ridge1.head("fine", alpha)
            error = (rows[[step]].reshape(-1, HIDDEN) @ weight + bias -
                     targets(fine, spans, [step], count, 8)) ** 2
            total = by_step.setdefault(step, [0.0, 0, count])
            total[0] += float(error.sum(dtype=np.float64)); total[1] += error.size
    steps = {step: (count, psnr(total, size)) for step, (total, size, count) in by_step.items()}
    print("  by latent step:", " ".join(f"{step}({count}f):{value:.1f}" for step, (count, value) in steps.items()))
    del ridge_both, heads

    print("whole patches (12288 values a token)")
    whole4, whole1 = gather(fit, whole=True)
    four.xy["whole"], four.y["whole"] = whole4.xy["whole"], whole4.y["whole"]
    one.xy["whole"], one.y["whole"] = whole1.xy["whole"], whole1.y["whole"]
    del whole4, whole1
    w4, b4 = ridge4.head("whole", alpha)
    w1, b1 = ridge1.head("whole", alpha)
    whole = judge(held, {
        "4-frame tokens, whole patch": (4, against(4, "whole", 32), w4, b4),
        "1-frame tokens, whole patch, own head": (1, against(1, "whole", 32), w1, b1)})
    video_error, video_count = 0.0, 0
    for prefix in held:
        rows, frames, spans = load(prefix)
        token_t, token_h, token_w, _ = rows.shape
        split = {}
        for count, (weight, bias) in ((4, (w4, b4)), (1, (w1, b1))):
            chosen = steps_of(spans, count)
            split[count] = (chosen, (rows[chosen].reshape(-1, HIDDEN) @ weight + bias).reshape(
                len(chosen), token_h, token_w, count, TOKEN_PIXELS, TOKEN_PIXELS, 3))
        rebuilt = np.clip(frames_of(split, token_t), -1, 1)
        video_error += float(((rebuilt - frames) ** 2).sum(dtype=np.float64)); video_count += frames.size
        shown = [0, frames.shape[0] // 3, 2 * frames.shape[0] // 3, frames.shape[0] - 1]
        sheet = np.concatenate([np.concatenate([frames[i] for i in shown], axis=1),
                                np.concatenate([rebuilt[i] for i in shown], axis=1)], axis=0)
        png(prefix + "_sheet.png", np.round((sheet + 1) * 127.5).astype(np.uint8))
    whole["whole video, both heads"] = psnr(video_error, video_count)
    for name, value in whole.items():
        print(f"  {name:46s} {value:6.2f} dB")
    np.savez(folder + "/linear_head.npz", weight4=w4, bias4=b4, weight1=w1, bias1=b1)
    json.dump({"fit_clips": len(fit), "held_clips": len(held), "alpha": alpha,
               "alpha_sweep_db": {str(a): tuned[a] for a in alphas},
               "tokens_4_frames": four.n, "tokens_1_frame": one.n,
               "held_db": results, "held_by_latent_step_db": {str(k): v for k, v in steps.items()},
               "held_whole_db": whole}, open(folder + "/results.json", "w"), indent=1)
    print(f"done in {time.time() - started:.0f}s: results.json, linear_head.npz, held_NN_sheet.png in {folder}")


if __name__ == "__main__":
    main()
