"""
Physical priors as differentiable PyTorch modules.
Used as additional feature channels pooled into the degradation conditioning.

Retinex     → illumination / reflectance decomposition (low-light)
DarkChannel → dark channel prior for transmission estimation (haze)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinexPrior(nn.Module):
    """
    Single-Scale Retinex: estimates illumination map via Gaussian smoothing.
    Returns (illumination, reflectance) both in [0, 1].
    """

    def __init__(self, sigma: int = 31) -> None:
        super().__init__()
        self.sigma = sigma
        self.register_buffer("_kernel", self._make_gaussian(sigma))

    @staticmethod
    def _make_gaussian(k: int) -> torch.Tensor:
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * (k / 6) ** 2))
        kernel = g.outer(g)
        kernel /= kernel.sum()
        return kernel.view(1, 1, k, k)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """img: [B, C, H, W] in [0, 1]. Returns (illumination [B,1,H,W], reflectance [B,C,H,W])."""
        luminance = img.max(dim=1, keepdim=True).values          # [B, 1, H, W]
        k = self._kernel
        pad = k.shape[-1] // 2
        illumination = F.conv2d(
            F.pad(luminance, [pad] * 4, mode="reflect"), k
        ).clamp(1e-6, 1.0)
        reflectance = (img / illumination.expand_as(img)).clamp(0, 1)
        return illumination, reflectance


class DarkChannelPrior(nn.Module):
    """
    Dark Channel Prior (He et al.) for haze transmission estimation.
    Returns transmission map in [0, 1] per pixel.
    """

    def __init__(self, patch_size: int = 15, omega: float = 0.95) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.omega = omega

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B, C, H, W] in [0, 1]. Returns transmission [B, 1, H, W]."""
        B, C, H, W = img.shape
        dark = img.min(dim=1, keepdim=True).values               # [B, 1, H, W]
        pad = self.patch_size // 2
        dark_channel = -F.max_pool2d(
            F.pad(-dark, [pad] * 4, mode="reflect"),
            kernel_size=self.patch_size,
            stride=1,
        )

        # Atmospheric light: top-0.1% brightest dark-channel pixels
        flat = dark_channel.view(B, -1)
        A_idx = flat.topk(max(1, flat.shape[1] // 1000), dim=1).indices
        A = (
            img.view(B, C, -1)
            .gather(2, A_idx[:, None, :].expand(-1, C, -1))
            .max(dim=2).values
            .max(dim=1).values
        )                                                         # [B]
        A = A.view(B, 1, 1, 1).clamp(0.5, 1.0)

        return (1 - self.omega * dark_channel / A).clamp(0.1, 1.0)