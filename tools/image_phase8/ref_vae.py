# P-H3-Image's VAE decoder in PyTorch, written from diffusers'
# AutoencoderKLPH3Image (43e30028: PH3ImageViTDecoder, PH3ImageAttnProcessor,
# PH3ImageTransformerBlock, _split_tiles / _stitch_tiles) and the pipeline's
# post-processing, reading the ComfyUI single-file weights directly.
#
#   python -I ref_vae.py <vae.safetensors> <config.json> <latent.safetensors> <out.npy>
#        [--bf16] [--crop H W]
#
# The latent file holds "latent" F32 [24, H, W], normalized as the DiT makes
# it. Writes the RGB image in [0, 1] as F32 [H*16, W*16, 3].
#
# ComfyUI's layout differs from diffusers' in two places: attn.to_qkv holds
# each head's q, k and v in turn ([heads, 3, 64] rows), and ff.w1 holds the
# gate half first (diffusers' SwiGLU proj has it second).
import json, math, sys
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

vae_path, config_path, latent_path, out_path = sys.argv[1:5]
dtype = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
crop = None
if "--crop" in sys.argv:
    i = sys.argv.index("--crop")
    crop = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))

config = json.load(open(config_path))
heads = config["decoder_num_attention_heads"]
dim_head = config["decoder_attention_head_dim"]
layers = config["decoder_num_layers"]
rope_dim = int(dim_head * config["decoder_rope_dim_ratio"])
theta = config["decoder_rope_theta"]
mean = torch.tensor(config["latents_mean"]).view(-1, 1, 1)
std = torch.tensor(config["latents_std"]).view(-1, 1, 1)
FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0
PATCH, TILE, OVERLAP = 16, 256, 64

w = {}
with safe_open(vae_path, "pt") as f:
    for k in f.keys():
        if k.startswith("decoder.") or k.startswith("post_quant_conv."):
            w[k] = f.get_tensor(k)
def W(name, keep_f32=False):
    t = w[name].float()
    return t if keep_f32 else t.to(dtype)

dim = heads * dim_head
inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * 3 / rope_dim, dtype=torch.float32)

def rope(position_ids):
    angles = 2.0 * math.pi * position_ids[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)          # [S, rope_dim]
    return angles.cos(), angles.sin()

def rms(x, weight=None, eps=1e-5):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return y * weight.float() if weight is not None else y

def rotate(x, cos, sin):
    r, rest = x[..., :rope_dim], x[..., rope_dim:]
    a, b = r.chunk(2, dim=-1)
    return torch.cat([r * cos + torch.cat([-b, a], dim=-1) * sin, rest], dim=-1)

def block(x, i, cos, sin):
    p = "decoder.transformer_blocks.%d." % i
    h = rms(x, w[p + "norm1.weight"]).to(dtype)
    qkv = F.linear(h, W(p + "attn.to_qkv.weight"), W(p + "attn.to_qkv.bias"))
    qkv = qkv.view(qkv.shape[0], heads, 3, dim_head)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]          # [S, heads, 64]
    q = rms(q).to(dtype); k = rms(k).to(dtype)
    c, s = cos[:, None].to(dtype), sin[:, None].to(dtype)
    q, k = rotate(q, c, s), rotate(k, c, s)
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                       v.transpose(0, 1)[None])[0].transpose(0, 1)
    o = F.linear(o.reshape(o.shape[0], dim), W(p + "attn.to_out.weight"), W(p + "attn.to_out.bias"))
    x = x + o.float() * w[p + "scale1"].float()
    h = rms(x, w[p + "norm2.weight"]).to(dtype)
    g = F.linear(h, W(p + "ff.w1.weight"), W(p + "ff.w1.bias"))
    gate, value = g.chunk(2, dim=-1)
    h = F.linear(value * F.silu(gate), W(p + "ff.w2.weight"), W(p + "ff.w2.bias"))
    return x + h.float() * w[p + "scale2"].float()

def decode_tile(z):                                   # z [24, h, w], denormalized
    c, height, width = z.shape
    z = F.conv2d(z[None].float(), w["post_quant_conv.weight"].float().view(24, 24, 1, 1),
                 w["post_quant_conv.bias"].float())[0]
    x = z.permute(1, 2, 0).reshape(height * width, c)
    x = F.linear(x.to(dtype), W("decoder.x_embedder.weight"), W("decoder.x_embedder.bias")).float()
    n = x.shape[0]
    x = torch.cat([x, w["decoder.register_tokens"][0].float(), torch.zeros(1, dim)], dim=0)
    grids = [torch.full((1,), FRAME_POSITION)] + [
        2.0 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1.0 for size in (height, width)]
    pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).reshape(-1, 3)
    pos = torch.cat([pos, torch.zeros(5, 3)], dim=0)
    cos, sin = rope(pos)
    for i in range(layers):
        x = block(x, i, cos, sin)
    x = F.layer_norm(x, (dim,), w["decoder.norm_out.weight"].float(), w["decoder.norm_out.bias"].float(), 1e-5)
    x = F.linear(x, w["decoder.proj_out.weight"].float(), w["decoder.proj_out.bias"].float())[:n]
    x = x.view(height, width, 3, PATCH, PATCH).permute(2, 0, 3, 1, 4)
    return x.reshape(3, height * PATCH, width * PATCH)

def split(length):
    if TILE >= length:
        return [0], [length], []
    count = math.ceil(length / TILE)
    while TILE * count - OVERLAP * (count - 1) - length < 0:
        count += 1
    overlaps = [OVERLAP] * (count - 1)
    remaining = TILE * count - sum(overlaps) - length
    i = 0
    while remaining > 0:
        overlaps[i % (count - 1)] += PATCH
        remaining -= PATCH
        i += 1
    starts = [0]
    for i in range(count - 1):
        starts.append(starts[-1] + TILE - overlaps[i])
    return starts, [TILE] * count, overlaps

def blend(a, b, extent, dim):
    extent = min(a.shape[dim], b.shape[dim], extent)
    pos = torch.arange(extent, dtype=b.dtype)
    shape = [1] * a.ndim; shape[dim] = extent
    wa = (1 - pos / extent).view(shape); wb = (pos / extent).view(shape)
    sa = [slice(None)] * a.ndim; sa[dim] = slice(-extent, None)
    sb = [slice(None)] * b.ndim; sb[dim] = slice(0, extent)
    mixed = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    if extent == b.shape[dim]:
        return mixed
    rest = [slice(None)] * b.ndim; rest[dim] = slice(extent, None)
    return torch.cat([mixed, b[tuple(rest)]], dim=dim)

with safe_open(latent_path, "pt") as f:
    latent = f.get_tensor("latent").float()
if crop:
    latent = latent[:, :crop[0], :crop[1]]
z = latent * std + mean
height, width = z.shape[1] * PATCH, z.shape[2] * PATCH
ys, yl, yo = split(height)
xs, xl, xo = split(width)
with torch.no_grad():
    tiles = [[decode_tile(z[:, y // PATCH:(y + a) // PATCH, x // PATCH:(x + b) // PATCH])
              for x, b in zip(xs, xl)] for y, a in zip(ys, yl)]
    rows = []
    for i, row in enumerate(tiles):
        out = []
        for j, tile in enumerate(row):
            if i > 0: tile = blend(tiles[i - 1][j], tile, yo[i - 1], -2)
            if j > 0: tile = blend(row[j - 1], tile, xo[j - 1], -1)
            if i < len(tiles) - 1: tile = tile[..., :-yo[i], :]
            if j < len(row) - 1: tile = tile[..., :, :-xo[j]]
            out.append(tile)
        rows.append(torch.cat(out, dim=-1))
    image = torch.cat(rows, dim=-2)
    pm = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    ps = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image * ps + pm).clamp(0, 1).permute(1, 2, 0).contiguous()
np.save(out_path, image.numpy())
print("wrote", out_path, tuple(image.shape), "tiles %d x %d" % (len(ys), len(xs)))
