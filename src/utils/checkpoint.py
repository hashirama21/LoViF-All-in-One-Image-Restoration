"""
Checkpoint manager and WandB logger wrapper.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Saves best and last checkpoints; handles early stopping counter."""

    def __init__(
        self,
        output_dir: str,
        monitor: str = "lpips",
        mode: str = "min",
        patience: int = 8,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self._best_value = float("inf") if mode == "min" else float("-inf")
        self._no_improve = 0

    def is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value < self._best_value
        return value > self._best_value

    def step(
        self,
        value: float,
        state: Dict[str, Any],
        step: int,
    ) -> bool:
        """
        Save last checkpoint always.
        Save best checkpoint if value improved.
        Returns True if training should stop (early stopping).
        """
        self._save(state, self.output_dir / "last.ckpt", step)

        if self.is_better(value):
            self._best_value = value
            self._no_improve = 0
            self._save(state, self.output_dir / "best.ckpt", step)
            logger.info(f"New best {self.monitor}={value:.4f} at step {step}")
        else:
            self._no_improve += 1
            logger.info(
                f"{self.monitor}={value:.4f} (no improvement {self._no_improve}/{self.patience})"
            )

        return self._no_improve >= self.patience

    @staticmethod
    def _save(state: Dict[str, Any], path: Path, step: int) -> None:
        state["step"] = step
        torch.save(state, path)
        logger.debug(f"Checkpoint saved → {path}")

    @staticmethod
    def load(path: str, device: str = "cpu") -> Dict[str, Any]:
        return torch.load(path, map_location=device)


class WandbLogger:
    """Thin wrapper — no-ops gracefully if wandb not installed / disabled."""

    def __init__(self, project: str, run_name: str, config: Dict, enabled: bool = True) -> None:
        self.enabled = enabled
        if enabled:
            try:
                import wandb
                wandb.init(project=project, name=run_name, config=config)
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed — logging disabled.")
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.enabled:
            self._wandb.log(metrics, step=step)

    def log_images(self, images: Dict[str, Any], step: int) -> None:
        if self.enabled:
            self._wandb.log(
                {k: self._wandb.Image(v) for k, v in images.items()},
                step=step,
            )

    def finish(self) -> None:
        if self.enabled:
            self._wandb.finish()
