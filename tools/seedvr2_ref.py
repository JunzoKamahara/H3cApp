"""The official SeedVR2-3B (ByteDance-Seed/SeedVR) on the CPU in float32,
with the very weights H3cApp runs: the MLX int8 export dequantised
(w = scale * q + bias per group of 64), renamed through mflux's table and
the VAE's conv kernels transposed (O, kt, kh, kw, I) -> (O, I, kt, kh, kw)."""
import dataclasses
import json
import os
import sys
import types

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDVR = os.environ.get("SEEDVR_REPO")
MFLUX_MAPPING = os.environ.get("MFLUX_MAPPING")   # .../weights/seedvr2_weight_mapping.py
sys.path.insert(0, os.path.join(HERE, "seedvr2_ref_stubs"))
sys.path.insert(0, SEEDVR)


def _mflux_targets():
    """Executes mflux's mapping file with stand-ins for its imports and returns
    (to_pattern, from_patterns, transform_name, max_blocks) for the
    transformer and the VAE."""
    @dataclasses.dataclass
    class WeightTarget:
        to_pattern: str
        from_pattern: list
        transform: object = None
        required: bool = True
        max_blocks: object = None
        complete_when_present: bool = False

    class _Transforms:
        def __getattr__(self, name):
            return name
    ns = {"WeightTarget": WeightTarget, "WeightMapping": object, "WeightTransforms": _Transforms(),
          "List": list}
    source = open(MFLUX_MAPPING).read()
    source = "\n".join(l for l in source.splitlines() if not l.startswith(("from ", "import ")))
    exec(compile(source, MFLUX_MAPPING, "exec"), ns)
    m = ns["SeedVR2WeightMapping"]
    return m.get_transformer_mapping(), m.get_vae_mapping()


def _expand(target, counts):
    """All concrete (mlx_name, [official candidates]) pairs of one target."""
    names = [(target.to_pattern, list(target.from_pattern))]
    for key, n in counts.items():
        out = []
        for to, frm in names:
            if "{" + key + "}" in to:
                for i in range(n):
                    out.append((to.replace("{" + key + "}", str(i)), [f.replace("{" + key + "}", str(i)) for f in frm]))
            else:
                out.append((to, frm))
        names = out
    return names


def _dequant(f, prefix):
    w = f.get_tensor(prefix + ".weight").view(torch.int32).to(torch.int64) & 0xffffffff
    s = f.get_tensor(prefix + ".scales").float()
    b = f.get_tensor(prefix + ".biases").float()
    out_f, words = w.shape
    q = torch.stack([(w >> (8 * k)) & 0xff for k in range(4)], dim=-1).reshape(out_f, words * 4).float()
    group = q.shape[1] // s.shape[1]
    s = s.repeat_interleave(group, dim=1)
    b = b.repeat_interleave(group, dim=1)
    return s * q + b


def _patch_rope():
    """The official NaMMRotaryEmbedding3d.get_freqs builds the axial table for
    1024 x 128 x 128 positions (about 8 GB) and slices it; the table is a
    function of each position, so building only the positions a clip uses
    and slicing the same rows gives the same values."""
    from models.dit_v2 import rope

    def get_freqs(self, vid_shape, txt_shape):
        vid_list, txt_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freqs = self.rope.get_axial_freqs(l + f, h, w)
            txt_freqs = self.rope.get_axial_freqs(l)
            vid_list.append(vid_freqs[l:l + f, :h, :w].reshape(-1, vid_freqs.size(-1)))
            txt_list.append(txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1)))
        return torch.cat(vid_list, dim=0), torch.cat(txt_list, dim=0)
    rope.NaMMRotaryEmbedding3d.get_freqs = get_freqs


def load_models(model_dir, want_vae=True, want_dit=True):
    from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    from models.dit_v2.nadit import NaDiT
    _patch_rope()
    vae = None if not want_vae else VideoAutoencoderKLWrapper(
        act_fn="silu", block_out_channels=[128, 256, 512, 512],
        down_block_types=["DownEncoderBlock3D"] * 4, in_channels=3, latent_channels=16,
        layers_per_block=2, norm_num_groups=32, out_channels=3, slicing_sample_min_size=4,
        temporal_scale_num=2, inflation_mode="pad", up_block_types=["UpDecoderBlock3D"] * 4,
        spatial_downsample_factor=8, temporal_downsample_factor=4, use_quant_conv=False,
        use_post_quant_conv=False, freeze_encoder=False)
    dit = None if not want_dit else NaDiT(
        vid_in_channels=33, vid_out_channels=16, vid_dim=2560, vid_out_norm="fusedrms",
        txt_in_dim=5120, txt_in_norm="fusedln", txt_dim=2560, emb_dim=6 * 2560, heads=20,
        head_dim=128, expand_ratio=4, norm="fusedrms", norm_eps=1e-5, ada="single",
        qk_bias=False, qk_norm="fusedrms", patch_size=[1, 2, 2], num_layers=32, mm_layers=10,
        mlp_type="swiglu", msa_type=None, block_type=["mmdit_sr"] * 32, window=[(4, 3, 3)] * 32,
        window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16,
        rope_type="mmrope3d", rope_dim=128)
    t_targets, v_targets = _mflux_targets()
    counts = {"block": 32, "res": 3, "i": 3, "j": 3, "up": 4, "down": 4}

    def fill(module, targets, path, quantized):
        """Copies every mapped tensor into the module's own parameters in
        place (no second copy of the 3B transformer). The transformer's
        linear weights are rounded to BF16 as the engine's GPU dequantise
        does; the time embedding (dequantised on the host in F32 by the
        engine) and every bias / norm / modulation stay F32."""
        params = dict(module.named_parameters())
        params.update(dict(module.named_buffers()))
        filled = set()
        with safe_open(path, "pt") as f:
            keys = set(f.keys())
            for t in targets:
                for mlx_name, candidates in _expand(t, counts):
                    hits = [c for c in candidates if c in params and c not in filled]
                    if not hits or mlx_name not in keys:
                        continue
                    prefix = mlx_name[:-len(".weight")] if mlx_name.endswith(".weight") else None
                    if quantized and prefix and prefix + ".scales" in keys:
                        value = _dequant(f, prefix)
                    else:
                        value = f.get_tensor(mlx_name).float()
                    if t.transform == "transpose_conv3d_weight" and value.ndim == 5:
                        value = value.permute(0, 4, 1, 2, 3).contiguous()
                    if quantized and value.ndim == 2 and not mlx_name.startswith("emb_in."):
                        value = value.to(torch.bfloat16).float()
                    for c in hits:
                        params[c].data.copy_(value.reshape(params[c].shape))
                        filled.add(c)
                    del value
        return [k for k in dict(module.named_parameters()) if k not in filled]
    vm = fill(vae, v_targets, os.path.join(model_dir, "vae.safetensors"), False) if vae is not None else []
    tm = fill(dit, t_targets, os.path.join(model_dir, "transformer.safetensors"), True) if dit is not None else []
    if vae is not None: vae.eval()
    if dit is not None: dit.eval()
    with safe_open(os.path.join(model_dir, "pos_emb.safetensors"), "pt") as f:
        pos = f.get_tensor("embedding").float().to(torch.bfloat16).float()   # the engine uploads it as BF16
    return vae, dit, pos, vm, tm


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    vae, dit, pos, vm, tm = load_models(sys.argv[1])
    print("vae params not filled:", vm)
    print("dit params not filled:", tm[:40], len(tm))
    print("pos", tuple(pos.shape))
