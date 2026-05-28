"""
Degradation Encoder — DCPT-style.
Encodes the LQ image into a sequence of degradation-aware tokens injected into
the backbone U-Net cross-attention layers via a learned linear projection.

Architecture:
  LQ → resize 224×224 → frozen CLIP-ViT-B/32 → linear projection → [B, 197, output_dim]

Returns ALL 197 ViT tokens (CLS + 196 patch tokens), preserving spatial structure.
The previous design returned only the pooled CLS token, discarding patch-level locality.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class DegradationEncoder(nn.Module):
    """
    Extracts a sequence of degradation-aware tokens from an LQ image.

    Args:
        output_dim:  Token embedding dimension (matched to the pipeline's encoder_dim).
        freeze_clip: Whether to freeze CLIP weights (always True in practice).
    """

    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, output_dim: int = 512, freeze_clip: bool = True) -> None:
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(self.CLIP_MODEL)
        if freeze_clip:
            self.clip.requires_grad_(False)

        clip_dim = self.clip.config.hidden_size           # 768 for ViT-B/32
        self.projection = nn.Linear(clip_dim, output_dim) # applied token-wise

        self.register_buffer("_clip_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_clip_std",  torch.tensor(_CLIP_STD).view(1, 3, 1, 1))

    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lq: float32 [B, C, H, W] in [0, 1].
        Returns:
            tokens: float32 [B, 197, output_dim]  — CLS + 196 spatial patch tokens.
        """
        lq_224 = F.interpolate(lq, size=(224, 224), mode="bicubic", align_corners=False)
        pixel_values = (lq_224 - self._clip_mean) / self._clip_std

        # torch.enable_grad() cannot override @torch.inference_mode() — use
        # explicit no_grad for the frozen path only; the unfrozen path inherits
        # whatever context is active (training loop's autocast, etc.).
        if self._clip_frozen():
            with torch.no_grad():
                hidden = self.clip(pixel_values=pixel_values).last_hidden_state
        else:
            hidden = self.clip(pixel_values=pixel_values).last_hidden_state

        return self.projection(hidden)  # [B, 197, output_dim]

    def _clip_frozen(self) -> bool:
        return not next(self.clip.parameters()).requires_grad
