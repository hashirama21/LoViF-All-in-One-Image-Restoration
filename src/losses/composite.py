"""
Loss modules for LoViF 2026.

DiffusionLoss  — primary DDPM noise-prediction MSE (latent space).
CompositeLoss  — pixel-space L1 + perceptual LPIPS + adversarial PatchGAN.
                 Applied on the x0 estimate decoded from the diffusion forward.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from src.utils.registry import LossRegistry


# ---------------------------------------------------------------------------
# Diffusion loss (primary)
# ---------------------------------------------------------------------------

class DiffusionLoss(nn.Module):
    """MSE between predicted and target noise in latent space."""

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = F.mse_loss(noise_pred, noise_target)
        return loss, {"diffusion_mse": loss.item()}


# ---------------------------------------------------------------------------
# Pixel-space loss components
# ---------------------------------------------------------------------------

class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class PerceptualLoss(nn.Module):
    """LPIPS with AlexNet backbone (fastest; competitive quality)."""

    def __init__(self) -> None:
        super().__init__()
        self._lpips = lpips.LPIPS(net="alex")
        self._lpips.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._lpips(pred * 2 - 1, target * 2 - 1).mean()


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (pix2pix). Input: concat(pred, lq)."""

    def __init__(self, in_channels: int = 3, ndf: int = 64) -> None:
        super().__init__()

        def block(ic: int, oc: int, stride: int = 2, norm: bool = True):
            layers = [nn.Conv2d(ic, oc, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels * 2, ndf, norm=False),
            *block(ndf,     ndf * 2),
            *block(ndf * 2, ndf * 4),
            *block(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, pred: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([lq, pred], dim=1))


class AdversarialLoss(nn.Module):
    """Hinge adversarial loss. Discriminator is updated by the trainer."""

    def __init__(self, ndf: int = 64) -> None:
        super().__init__()
        self.discriminator = PatchGANDiscriminator(ndf=ndf)

    def forward_generator(self, pred: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        return -self.discriminator(pred, lq).mean()

    def forward_discriminator(
        self, pred: torch.Tensor, gt: torch.Tensor, lq: torch.Tensor
    ) -> torch.Tensor:
        real = self.discriminator(gt.detach(), lq)
        fake = self.discriminator(pred.detach(), lq)
        return (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()) * 0.5


# ---------------------------------------------------------------------------
# Composite pixel-space loss
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    l1: float = 1.0
    lpips: float = 0.15
    adversarial: float = 0.01


@LossRegistry.register("composite")
class CompositeLoss(nn.Module):
    """
    Weighted L1 + LPIPS + adversarial on pixel-space x0 predictions.
    Components with weight=0 are not instantiated.
    Breakdown values are raw (unweighted) for interpretable logging.
    """

    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights
        self.l1 = L1Loss()
        self.perceptual = PerceptualLoss() if weights.lpips > 0 else None
        self.adversarial = AdversarialLoss() if weights.adversarial > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        lq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (total_loss, breakdown). Breakdown stores raw unweighted values."""
        breakdown: dict[str, float] = {}

        l1_raw = self.l1(pred, gt)
        total  = self.weights.l1 * l1_raw
        breakdown["l1"] = l1_raw.item()

        if self.perceptual is not None:
            lp_raw = self.perceptual(pred, gt)
            total  = total + self.weights.lpips * lp_raw
            breakdown["lpips"] = lp_raw.item()

        if self.adversarial is not None:
            adv_raw = self.adversarial.forward_generator(pred, lq)
            total   = total + self.weights.adversarial * adv_raw
            breakdown["adversarial_g"] = adv_raw.item()

        breakdown["total"] = total.item()
        return total, breakdown
