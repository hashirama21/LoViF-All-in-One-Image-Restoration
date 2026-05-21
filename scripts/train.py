"""
scripts/train.py — main training entry point.

Usage:
    # Full pipeline (default):
    python scripts/train.py

    # Baseline (FoundIR without LoRA / encoder):
    python scripts/train.py --config-name=train_baseline

    # Override a single key:
    python scripts/train.py training.lr=1e-4
"""
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../configs", config_name="train_full", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.project.seed)
    logger.info(f"Run: {cfg.project.run_name}  |  seed={cfg.project.seed}")
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()