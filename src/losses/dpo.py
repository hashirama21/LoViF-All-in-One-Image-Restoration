"""
DPO (Direct Preference Optimization) loss for image restoration.
Given (chosen, rejected, lq) triplets, trains the model to prefer
perceptually better restorations without an explicit reward model at
inference time.

Reference: Rafailov et al., 2023 — adapted for image generation.
"""
from __future__ import annotations
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DPOLoss(nn.Module):
    """
    DPO loss for image restoration preferences.

    beta: KL-penalty coefficient. Higher = stay closer to reference model.
          Typical range [0.05, 0.2].
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        log_prob_chosen:      torch.Tensor,  # [B]
        log_prob_rejected:    torch.Tensor,  # [B]
        ref_log_prob_chosen:  torch.Tensor,  # [B]
        ref_log_prob_rejected: torch.Tensor, # [B]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pi_logratios  = log_prob_chosen   - log_prob_rejected
        ref_logratios = ref_log_prob_chosen - ref_log_prob_rejected
        loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()

        chosen_reward  = (log_prob_chosen  - ref_log_prob_chosen).mean().item()
        rejected_reward = (log_prob_rejected - ref_log_prob_rejected).mean().item()
        breakdown = {
            "dpo_loss":        loss.item(),
            "chosen_reward":   chosen_reward,
            "rejected_reward": rejected_reward,
            "reward_margin":   chosen_reward - rejected_reward,
        }
        return loss, breakdown


class RewardModel(nn.Module):
    """
    Proxy reward model for offline DPO pair generation.
    Scores N generations per image — NOT used at inference time.

    Metrics used (in priority order):
      1. MUSIQ + CLIP-IQA via pyiqa (if installed)
      2. LPIPS (always available, inverted so higher = better)
    """

    def __init__(
        self,
        musiq_weight:    float = 0.5,
        clip_iqa_weight: float = 0.3,
        lpips_weight:    float = 0.2,
    ) -> None:
        super().__init__()
        self.w_musiq    = musiq_weight
        self.w_clip_iqa = clip_iqa_weight
        self.w_lpips    = lpips_weight

        import lpips as lpips_lib
        self._lpips = lpips_lib.LPIPS(net="alex")
        self._lpips.requires_grad_(False)

        self._musiq = self._clip_iqa = None
        try:
            import pyiqa
            self._musiq    = pyiqa.create_metric("musiq",    as_loss=False)
            self._clip_iqa = pyiqa.create_metric("clip_iqa", as_loss=False)
            logger.info("RewardModel: MUSIQ + CLIP-IQA loaded via pyiqa.")
        except ImportError:
            logger.warning(
                "pyiqa not installed — MUSIQ/CLIP-IQA unavailable. "
                "Falling back to LPIPS-only scoring."
            )

    @torch.inference_mode()
    def score(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Returns composite quality score [B] — higher is better."""
        lpips_dist  = self._lpips(pred * 2 - 1, gt * 2 - 1).squeeze().clamp(0, 1)
        lpips_score = 1.0 - lpips_dist

        if self._musiq is not None:
            musiq_score    = self._musiq(pred).squeeze().clamp(0, 1)
            clip_iqa_score = self._clip_iqa(pred).squeeze().clamp(0, 1)
            w_total = self.w_musiq + self.w_clip_iqa + self.w_lpips
            return (
                self.w_musiq    * musiq_score
                + self.w_clip_iqa * clip_iqa_score
                + self.w_lpips    * lpips_score
            ) / w_total

        return lpips_score  # LPIPS is the only available metric

    def select_pairs(
        self,
        generations: list[torch.Tensor],  # N tensors [B, C, H, W]
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given N generations, return (chosen=best, rejected=worst) by composite score."""
        scores  = torch.stack([self.score(g, gt) for g in generations], dim=0)  # [N, B]
        stacked = torch.stack(generations, dim=0)                                # [N, B, C, H, W]
        B = gt.shape[0]
        arange = torch.arange(B)
        chosen   = stacked[scores.argmax(dim=0), arange]
        rejected = stacked[scores.argmin(dim=0), arange]
        return chosen, rejected
