#!/usr/bin/env python3
"""Writes the numeric fixture tests/test_refmod.c checks the engine's RefMod
arithmetic against (misc/fixtures/h3_refmod_fixture.safetensors): PyTorch's
adaptive_avg_pool3d, trilinear interpolation (align_corners=False), the
upstream strength blur/mix and 20 Adam steps of the compressed-mode
refinement, written out as the expressions ComfyUI-MiniMaxH3Mod (core.py,
nodes.py) uses. Development only - needs torch and safetensors:

    python3 -I tools/dump_refmod_fixture.py misc/fixtures/h3_refmod_fixture.safetensors
"""
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def blur_latent(z):
    # core.py _blur_latent for a visual latent [1, C, T, H, W]
    _, _, t, h, w = z.shape
    pooled = F.adaptive_avg_pool3d(z.float(), (t, max(1, h // 8), max(1, w // 8)))
    return F.interpolate(pooled, size=(t, h, w), mode="trilinear", align_corners=False).to(z.dtype)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    torch.manual_seed(0)
    C, T, H, W = 24, 3, 16, 24
    PT, PH, PW = 2, 8, 12
    z = torch.randn(1, C, T, H, W, dtype=torch.float32)
    blur = blur_latent(z)
    mix = 0.4 * z + 0.6 * blur
    pooled = F.adaptive_avg_pool3d(z, (PT, PH, PW))
    up = F.interpolate(pooled, size=(T, H, W), mode="trilinear", align_corners=False)

    # nodes.py optimize_latent: Adam(lr=0.02) on the F16-rounded pool,
    # mse against the full latent through a trilinear upsample.
    start = pooled.to(torch.float16).to(torch.float32)
    param = torch.nn.Parameter(start.clone())
    optimizer = torch.optim.Adam([param], lr=0.02)
    grad0 = None
    for step in range(20):
        optimizer.zero_grad()
        upsampled = F.interpolate(param, size=(T, H, W), mode="trilinear", align_corners=False)
        loss = F.mse_loss(upsampled, z)
        loss.backward()
        if step == 0:
            grad0 = param.grad.detach().clone()
        optimizer.step()
    refined = param.detach().clone()

    save_file({
        "z": z.contiguous(),
        "blur": blur.contiguous(),
        "mix_0_4": mix.contiguous(),
        "pooled_2x8x12": pooled.contiguous(),
        "up_from_pooled": up.contiguous(),
        "refined_20": refined.contiguous(),
        "grad_0": grad0.contiguous(),
    }, sys.argv[1], metadata={"made_by": "tools/dump_refmod_fixture.py",
                             "torch": torch.__version__})
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
