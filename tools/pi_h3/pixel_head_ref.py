#!/usr/bin/env python3
"""pi-H3's pixel head in numpy, as a reference for the engine's.

It follows LakonLab v0.3.0 (lakonlab/models/architectures/sol_h3/{model,sol_h3}.py
and diffusers/minimax_h3.py, Apache-2.0 parts adapted from the Sol-H3-VAE
reference decoder), not the paper's appendix: the released head is the last
8 layers of the Sol-H3-VAE decoder driven by the DiT's last block.

  rows before the final AdaLN  [latent_t * token_h * token_w, 5376]
    -> dwd_norm_out            LayerNorm (no affine), * (1 + scale) + shift,
                               [scale, shift] = linear(silu(curve(t)))
    -> feature_expand          one row -> 2 x 2 tokens of 2048 (16 px each)
    -> 8 layers                windows of 5 latent steps x 16 x 16 tokens (odd
                               layers shifted by 2 steps and 8 tokens), plus 5
                               global tokens per 5 steps
    -> norm_out, proj_out      4 frames a token; the first step of every 5
                               has first_frame_proj_out, 1 frame
    -> RGB = out * std + mean  clamped to [-1, 1]

  pixel_head_ref.py <adapter.safetensors> <checkpoint.gguf> <dump prefix> <out prefix>

<adapter> is either layout of the adapter: the dwd_* tensors have the same
names in both, plain F16 in the diffusers file and ComfyUI int8 for the
linears of the single-file version. <checkpoint.gguf> gives the AdaLN
curve table. <dump prefix> is what H3_DUMP_FINAL_VIDEO wrote (.hidden.bf16,
.json). Writes <out prefix>.rgb (RGB24 frames) and, with --features, the
tensors between the stages as .npy for the engine to be compared with."""
import json
import math
import struct
import sys

import numpy as np

HEADS, HEAD_DIM = 32, 64
DIM = HEADS * HEAD_DIM
WINDOW, BLOCK, T_SHIFT, TIME_NORM = 16, 5, 2, 5
PATCH, PATCH_T, CHANNELS = 16, 4, 3
REGISTERS = 4
GLOBALS = REGISTERS + 1
ROPE_THETA, ROPE_DIM = 100.0, 48
EPS = 1e-5
FIRST_LAYER = 24 - 8          # the head's layers are 16..23 of Sol's 24
RGB_MEAN = np.array([-0.03, -0.088, -0.188], np.float32)
RGB_STD = np.array([0.458, 0.448, 0.45], np.float32)


class Safetensors:
    def __init__(self, path):
        self.file = open(path, "rb")
        size = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(size))
        self.header.pop("__metadata__", None)
        self.base = 8 + size

    def raw(self, name):
        entry = self.header[name]
        begin, end = entry["data_offsets"]
        self.file.seek(self.base + begin)
        dtype = {"F16": np.float16, "F32": np.float32, "I8": np.int8,
                 "U8": np.uint8}[entry["dtype"]]
        return np.frombuffer(self.file.read(end - begin), dtype).reshape(entry["shape"])

    def __call__(self, name):
        values = self.raw(name)
        if values.dtype != np.int8:
            return values.astype(np.float32)
        # A ComfyUI int8 linear (the engine's h3_comfy_int8_dequantize_row):
        # the integers, rotated back within groups of columns (ConvRot: H4
        # on each base-4 digit of the column index), times the row's scale.
        stem = name[:-len("weight")]
        quant = json.loads(bytes(self.raw(stem + "comfy_quant")).decode())
        values = values.astype(np.float32)
        rows, columns = values.shape
        group = quant.get("convrot_groupsize", 0) if quant.get("convrot") else 0
        stride = 1
        while stride < group:
            v = values.reshape(rows, columns // (4 * stride), 4, stride)
            a, b, c, d = v[:, :, 0], v[:, :, 1], v[:, :, 2], v[:, :, 3]
            t = 0.5 * (a + b + c + d)
            values = np.stack([t - d, t - c, t - b, t - a], axis=2).reshape(rows, columns)
            stride *= 4
        return values * self.raw(stem + "weight_scale").astype(np.float32).reshape(-1, 1)


def gguf_tensor(path, wanted):
    """One F32 tensor of a GGUF file, in numpy's dimension order."""
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def text():
            return f.read(rd("Q")).decode("utf-8", "replace")

        def value(kind):
            plain = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                     7: "?", 10: "Q", 11: "q", 12: "d"}
            if kind in plain:
                return rd(plain[kind])
            if kind == 8:
                return text()
            item, count = rd("I"), rd("Q")
            return [value(item) for _ in range(count)]

        assert f.read(4) == b"GGUF"
        rd("I")
        tensors, pairs = rd("Q"), rd("Q")
        alignment = 32
        for _ in range(pairs):
            key = text()
            got = value(rd("I"))
            if key == "general.alignment":
                alignment = got
        found = None
        for _ in range(tensors):
            name = text()
            dims = [rd("Q") for _ in range(rd("I"))]
            kind, offset = rd("I"), rd("Q")
            if name == wanted:
                assert kind == 0, "an F32 tensor is expected"
                found = (dims[::-1], offset)
        start = (f.tell() + alignment - 1) // alignment * alignment
        shape, offset = found
        f.seek(start + offset)
        return np.frombuffer(f.read(4 * int(np.prod(shape))), np.float32).reshape(shape)


def bf16(path, columns):
    raw = np.fromfile(path, np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(-1, columns)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rms_norm(x, weight=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return y if weight is None else y * weight


def layer_norm(x, weight=None, bias=None):
    y = x - x.mean(axis=-1, keepdims=True)
    y = y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + EPS)
    return y if weight is None else y * weight + bias


def curve(table, t):
    steps = table.shape[0] - 1
    position = min(max(t, 0.0), 1.0) * steps
    lower = min(int(math.floor(position)), steps - 1)
    return table[lower] + (table[lower + 1] - table[lower]) * np.float32(position - lower)


def rope(T, H, W, shifted):
    """cos, sin [T*H*W, 24] of the window-local positions."""
    s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
    t = (((np.arange(T) + st) % BLOCK) + 0.5) / TIME_NORM * 2 - 1
    y = (((np.arange(H) + s) % WINDOW) + 0.5) / WINDOW * 2 - 1
    x = (((np.arange(W) + s) % WINDOW) + 0.5) / WINDOW * 2 - 1
    grid = np.stack(np.meshgrid(t, y, x, indexing="ij"), -1).reshape(-1, 3)
    inv_freq = 1.0 / ROPE_THETA ** np.arange(0, 1, 6 / ROPE_DIM)
    angles = (2.0 * math.pi * grid[:, :, None] * inv_freq[None, None, :]).reshape(len(grid), -1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rotate(x, cos, sin):
    """x [rows, heads, 64]: the first 48 values of a head rotate, in two halves."""
    half = cos.shape[-1]
    first, second = x[..., :half].copy(), x[..., half:2 * half].copy()
    x[..., :half] = first * cos[:, None, :] - second * sin[:, None, :]
    x[..., half:2 * half] = second * cos[:, None, :] + first * sin[:, None, :]


def attend(q, k, v):
    """q [nq, heads, 64], k and v [nk, heads, 64] -> [nq, heads, 64]."""
    scores = q.transpose(1, 0, 2) @ k.transpose(1, 2, 0) / math.sqrt(HEAD_DIM)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ v.transpose(1, 0, 2)).transpose(1, 0, 2)


def windows(T, H, W, shifted):
    """(token rows, global-token blocks) of every window, tokens in (t, y, x) order."""
    s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
    Tp = (T + st + BLOCK - 1) // BLOCK * BLOCK
    Hp = (H + s + WINDOW - 1) // WINDOW * WINDOW
    Wp = (W + s + WINDOW - 1) // WINDOW * WINDOW
    padded = np.full((Tp, Hp, Wp), -1, np.int64)
    padded[st:st + T, s:s + H, s:s + W] = np.arange(T * H * W).reshape(T, H, W)
    result = []
    for j in range(Tp // BLOCK):
        first = max(j * BLOCK - st, 0) // BLOCK
        last = (min((j + 1) * BLOCK - st, T) - 1) // BLOCK
        blocks = [first] if last == first else [first, last]
        for a in range(Hp // WINDOW):
            for b in range(Wp // WINDOW):
                rows = padded[j * BLOCK:(j + 1) * BLOCK, a * WINDOW:(a + 1) * WINDOW,
                              b * WINDOW:(b + 1) * WINDOW].reshape(-1)
                rows = rows[rows >= 0]
                if len(rows):
                    result.append((rows, blocks))
    return result


def layer(weights, index, x, g, shape):
    """x [N, 2048] tokens in (t, y, x) order, g [blocks, 5, 2048]."""
    T, H, W = shape
    N, nb = x.shape[0], g.shape[0]
    prefix = f"dwd_pixel_refiner.blocks.{index}."
    w = lambda name: weights(prefix + name)
    shifted = (FIRST_LAYER + index) % 2 == 1
    both = np.concatenate([x, g.reshape(-1, DIM)])
    qkv = rms_norm(both, w("norm1.weight")) @ w("attn.to_qkv.weight").T + w("attn.to_qkv.bias")
    qkv = qkv.reshape(-1, HEADS, 3, HEAD_DIM)
    q, k, v = rms_norm(qkv[:, :, 0]), rms_norm(qkv[:, :, 1]), qkv[:, :, 2]
    cos, sin = rope(T, H, W, shifted)
    q, k = q.copy(), k.copy()
    rotate(q[:N], cos, sin)
    rotate(k[:N], cos, sin)
    out = np.zeros_like(q)
    for block in range(nb):          # a block's global tokens see all of its tokens
        tokens = np.arange(block * BLOCK * H * W, min((block + 1) * BLOCK * H * W, N))
        own = N + block * GLOBALS + np.arange(GLOBALS)
        keys = np.concatenate([tokens, own])
        out[own] = attend(q[own], k[keys], v[keys])
    for rows, blocks in windows(T, H, W, shifted):
        keys = np.concatenate([rows] + [N + b * GLOBALS + np.arange(GLOBALS) for b in blocks])
        out[rows] = attend(q[rows], k[keys], v[keys])
    both = both + (out.reshape(-1, DIM) @ w("attn.to_out.weight").T + w("attn.to_out.bias")) * w("scale1")
    hidden = rms_norm(both, w("norm2.weight")) @ w("ff.w1.weight").T + w("ff.w1.bias")
    half = hidden.shape[1] // 2
    hidden = silu(hidden[:, :half]) * hidden[:, half:]
    both = both + (hidden @ w("ff.w2.weight").T + w("ff.w2.bias")) * w("scale2")
    return both[:N], both[N:].reshape(nb, GLOBALS, DIM)


def head(weights, table, hidden, shape, sigma, keep=None):
    """hidden [t*h*w, 5376] (before the final AdaLN) -> frames [F, 16H, 16W, 3] in [-1, 1]."""
    t, h, w = shape
    keep = {} if keep is None else keep
    coordinates = curve(table, 1.0 - sigma)
    modulation = weights("dwd_norm_out.linear.weight") @ silu(coordinates) + weights("dwd_norm_out.linear.bias")
    scale, shift = modulation[:hidden.shape[1]], modulation[hidden.shape[1]:]
    features = layer_norm(hidden) * (1.0 + scale) + shift
    keep["modulation"] = modulation
    keep["features"] = features
    x = features @ weights("dwd_pixel_refiner.feature_expand.weight").T + \
        weights("dwd_pixel_refiner.feature_expand.bias")
    e = 32 // PATCH
    H, W = h * e, w * e
    x = x.reshape(t, h, w, e, e, DIM).transpose(0, 1, 3, 2, 4, 5).reshape(t * H * W, DIM)
    keep["expanded"] = x
    nb = (t + BLOCK - 1) // BLOCK
    g = np.concatenate([weights("dwd_pixel_refiner.register_tokens").reshape(REGISTERS, DIM),
                        np.zeros((1, DIM), np.float32)])
    g = np.repeat(g[None], nb, axis=0)
    for index in range(8):
        x, g = layer(weights, index, x, g, (t, H, W))
        keep[f"layer{index}"] = x
        print(f"layer {index}: mean |x| {np.abs(x).mean():.4f}", file=sys.stderr)
    x = layer_norm(x, weights("dwd_pixel_refiner.norm_out.weight"),
                   weights("dwd_pixel_refiner.norm_out.bias")).reshape(t, H, W, DIM)
    frames = []
    for step in range(t):
        if step % BLOCK == 0:
            out = x[step] @ weights("dwd_pixel_refiner.first_frame_proj_out.weight").T + \
                weights("dwd_pixel_refiner.first_frame_proj_out.bias")
            count = PATCH_T - 3
        else:
            out = x[step] @ weights("dwd_pixel_refiner.proj_out.weight").T + \
                weights("dwd_pixel_refiner.proj_out.bias")
            count = PATCH_T
        # a token's values are (channel, frame, y, x)
        out = out.reshape(H, W, CHANNELS, count, PATCH, PATCH).transpose(3, 0, 4, 1, 5, 2)
        frames.append(out.reshape(count, H * PATCH, W * PATCH, CHANNELS))
    frames = np.concatenate(frames)
    keep["raw"] = frames
    return np.clip(frames * RGB_STD + RGB_MEAN, -1.0, 1.0)


def main():
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    adapter, checkpoint, dump, out = arguments
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    shape = json.load(open(dump + ".json"))
    grid = (shape["latent_t"], shape["token_h"], shape["token_w"])
    hidden = bf16(dump + ".hidden.bf16", shape["hidden"])
    assert hidden.shape[0] == grid[0] * grid[1] * grid[2]
    table = gguf_tensor(checkpoint, "adaln_t_table")
    keep = {}
    frames = head(Safetensors(adapter), table, hidden, grid, shape["video_sigma"], keep)
    assert np.isfinite(frames).all()
    np.round((frames + 1.0) * 127.5).astype(np.uint8).tofile(out + ".rgb")
    print(f"{frames.shape[0]} frames of {frames.shape[2]}x{frames.shape[1]} -> {out}.rgb")
    if "--features" in sys.argv:
        for name, values in keep.items():
            np.save(f"{out}.{name}.npy", values.astype(np.float32))
        # the head's AdaLN as tests/pixel_head_probe.c reads it
        keep["modulation"].astype(np.float32).tofile(out + ".modulation.f32")


if __name__ == "__main__":
    main()
