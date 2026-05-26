"""
Inference engine — TTA + ensemble for LoViF challenge submission.

TTA: 4-way (identity, hflip, vflip, rot90) → inverse-transform then mean.
Ensemble: weighted average over N model checkpoints.
Output images are resized back to the original input resolution before saving
(required for challenge submission scoring).
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from omegaconf import DictConfig

from src.data import InferenceDataset
from src.models import RestorationPipeline, PipelineConfig

logger = logging.getLogger(__name__)


def _tta_augment(img: torch.Tensor) -> List[torch.Tensor]:
    """4-way augmentation of a single image [C, H, W]."""
    return [
        img,
        TF.hflip(img),
        TF.vflip(img),
        torch.rot90(img, k=1, dims=[1, 2]),
    ]


def _tta_deaugment(imgs: List[torch.Tensor]) -> torch.Tensor:
    """Inverse-transform each TTA output, then average."""
    return torch.stack(
        [
            imgs[0],
            TF.hflip(imgs[1]),
            TF.vflip(imgs[2]),
            torch.rot90(imgs[3], k=-1, dims=[1, 2]),
        ],
        dim=0,
    ).mean(dim=0)


class InferenceEngine:
    """
    Loads multiple checkpoints and runs TTA + weighted ensemble.
    Designed for the final LoViF submission (batch_size=1 required for TTA).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.inference.get("device", "cuda"))
        self._dtype = torch.bfloat16

        # Inference hyperparams — read from config, not hardcoded in checkpoint
        self._num_steps      = cfg.inference.get("num_inference_steps", 25)
        self._guidance_scale = cfg.inference.get("guidance_scale", 1.5)

        raw_weights: List[float] = list(cfg.inference.ensemble_weights)
        w_sum = sum(raw_weights)
        if abs(w_sum - 1.0) > 1e-4:
            logger.warning(
                f"Ensemble weights sum to {w_sum:.4f}, not 1.0 — normalising."
            )
        self.weights = [w / w_sum for w in raw_weights]

        self.models: List[RestorationPipeline] = []
        for ckpt_path in cfg.inference.checkpoints:
            self.models.append(self._load_model(ckpt_path))
            logger.info(f"Loaded checkpoint: {ckpt_path}")

        if len(self.models) != len(self.weights):
            raise ValueError(
                f"checkpoints ({len(self.models)}) and ensemble_weights "
                f"({len(self.weights)}) must have the same length."
            )

        self.tta_enabled: bool = cfg.inference.tta.enabled

    def _load_model(self, ckpt_path: str) -> RestorationPipeline:
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_cfg = (
            PipelineConfig(**state["model_config"])
            if "model_config" in state
            else PipelineConfig()
        )
        model = RestorationPipeline(model_cfg).to(self.device)
        model.load_state_dict(state["model_state"], strict=False)
        model.eval()
        return model

    @torch.inference_mode()
    def restore_single(self, lq: torch.Tensor) -> torch.Tensor:
        """
        lq: [1, C, H, W] float32 in [0, 1].
        Returns restored image [1, C, H, W] in [0, 1].
        """
        ensemble_output = torch.zeros_like(lq)
        device_type = self.device.type

        for model, w in zip(self.models, self.weights):
            with torch.autocast(device_type, dtype=self._dtype):
                if self.tta_enabled:
                    augmented = _tta_augment(lq.squeeze(0))          # list of [C, H, W]
                    tta_outs  = [
                        model.restore(
                            a.unsqueeze(0),
                            num_inference_steps=self._num_steps,
                            guidance_scale=self._guidance_scale,
                        ).squeeze(0)
                        for a in augmented
                    ]
                    merged = _tta_deaugment(tta_outs).unsqueeze(0)
                else:
                    merged = model.restore(
                        lq,
                        num_inference_steps=self._num_steps,
                        guidance_scale=self._guidance_scale,
                    )

            ensemble_output = ensemble_output + w * merged

        return ensemble_output.clamp(0, 1)

    def run(self, input_dir: str, output_dir: str) -> None:
        """
        Restores all images in input_dir and writes PNG results to output_dir.
        Output is resized back to the original input resolution before saving.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        dataset = InferenceDataset(input_dir)
        loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        logger.info(
            f"Running inference on {len(dataset)} images "
            f"(TTA={'on' if self.tta_enabled else 'off'}, "
            f"ensemble={len(self.models)} models, "
            f"steps={self._num_steps}, guidance={self._guidance_scale})"
        )

        for batch in tqdm(loader, desc="LoViF inference"):
            lq       = batch["lq"].to(self.device)
            filename = batch["filename"][0]
            orig_h   = int(batch["orig_h"][0])
            orig_w   = int(batch["orig_w"][0])

            restored = self.restore_single(lq)  # [1, C, 512, 512]

            # Resize back to original resolution if different
            if (orig_h, orig_w) != (restored.shape[-2], restored.shape[-1]):
                restored = TF.resize(
                    restored.squeeze(0),
                    [orig_h, orig_w],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ).unsqueeze(0)

            img_pil = Image.fromarray(
                (restored.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
                .clip(0, 255)
                .astype("uint8")
            )
            img_pil.save(Path(output_dir) / filename)

        logger.info(f"Done. Results saved to: {output_dir}")
