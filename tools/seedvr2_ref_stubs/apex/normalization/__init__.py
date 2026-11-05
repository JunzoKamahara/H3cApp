"""Stand-ins for apex's fused norms (same parameters, computed in float32)."""
import torch
from torch import nn


class FusedRMSNorm(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if elementwise_affine else None

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            x = x * self.weight.float()
        return x.to(dtype)


class FusedLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
