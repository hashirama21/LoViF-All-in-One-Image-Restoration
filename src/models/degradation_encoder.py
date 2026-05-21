"""
Degradation Encoder — DCPT-style.
Encodes the LQ image into a degradation embedding injected into the
backbone U-Net cross-attention layers via a learned projection.

Architecture:
  LQ → resize 224×224 → frozen CLIP-ViT-B/32 → linear projection → [B, output_dim]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

# CLIP-ViT-B/32 normalisation constants (ImageNet CLIP stats, NOT [-1, 1])
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class DegradationEncoder(nn.Module):
    """
    Extracts a degradation-aware embedding from an LQ image.

    Args:
        output_dim:  Embedding dimension (matched to the pipeline's encoder_dim).
        freeze_clip: Whether to freeze CLIP weights (always True in practice).
    """

    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, output_dim: int = 512, freeze_clip: bool = True) -> None:
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(self.CLIP_MODEL)

        if freeze_clip:
            self.clip.requires_grad_(False)

        clip_dim = self.clip.config.hidden_size          # 768 for ViT-B/32
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Register normalization stats as buffers so they move with .to(device)
        mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("_clip_mean", mean)
        self.register_buffer("_clip_std",  std)

    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lq: float32 [B, C, H, W] in [0, 1].
        Returns:
            embedding: float32 [B, output_dim]
        """
        lq_224 = F.interpolate(lq, size=(224, 224), mode="bicubic", align_corners=False)
        pixel_values = (lq_224 - self._clip_mean) / self._clip_std

        grad_ctx = torch.no_grad() if self._clip_frozen() else torch.enable_grad()
        with grad_ctx:
            pooled = self.clip(pixel_values=pixel_values).pooler_output  # [B, clip_dim]

        return self.projection(pooled)                  # [B, output_dim]

    def _clip_frozen(self) -> bool:
        return not next(self.clip.parameters()).requires_grad
