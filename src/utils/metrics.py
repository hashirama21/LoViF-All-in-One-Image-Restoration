"""
Metrics — PSNR, SSIM, LPIPS.
Supports per-degradation-category tracking aligned with LoViF index rules.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import lpips


# LoViF 2026 index → category mapping
LOVIF_INDEX_MAP = {
    range(1, 101):   "blur",
    range(101, 201): "low_light",
    range(201, 301): "haze",
    range(301, 401): "rain",
    range(401, 501): "snow",
}


def get_category(filename: str) -> str:
    """Return degradation category from LoViF filename (e.g. '0253.png')."""
    stem = "".join(filter(str.isdigit, filename.split("/")[-1].split(".")[0]))
    if not stem:
        return "unknown"
    idx = int(stem)
    for r, cat in LOVIF_INDEX_MAP.items():
        if idx in r:
            return cat
    return "unknown"


@dataclass
class MetricBag:
    """Accumulates metrics; call .summary() to get per-category + global means."""
    _records: Dict[str, List[Dict]] = field(default_factory=lambda: defaultdict(list))
    _lpips_fn: Optional[lpips.LPIPS] = field(default=None, init=False)

    def _get_lpips(self, device: torch.device) -> lpips.LPIPS:
        if self._lpips_fn is None:
            self._lpips_fn = lpips.LPIPS(net="alex").to(device)
            self._lpips_fn.eval()
        return self._lpips_fn

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        category: str = "all",
    ) -> Dict[str, float]:
        """
        pred, gt: float tensors [B, C, H, W] in [0, 1].
        Returns per-sample metrics dict.
        """
        assert pred.shape == gt.shape, "pred/gt shape mismatch"

        psnr = peak_signal_noise_ratio(pred, gt, data_range=1.0).item()
        ssim = structural_similarity_index_measure(pred, gt, data_range=1.0).item()

        pred_lp = pred * 2 - 1
        gt_lp = gt * 2 - 1
        lp = self._get_lpips(pred.device)(pred_lp, gt_lp).mean().item()

        record = {"psnr": psnr, "ssim": ssim, "lpips": lp}
        self._records[category].append(record)
        self._records["all"].append(record)
        return record

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for cat, records in self._records.items():
            n = len(records)
            result[cat] = {
                "psnr":  sum(r["psnr"]  for r in records) / n,
                "ssim":  sum(r["ssim"]  for r in records) / n,
                "lpips": sum(r["lpips"] for r in records) / n,
                "n":     n,
            }
        return result

    def reset(self) -> None:
        self._records.clear()
