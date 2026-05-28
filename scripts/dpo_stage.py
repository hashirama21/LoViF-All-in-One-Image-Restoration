"""
scripts/dpo_stage.py — preference optimization stage.
Run AFTER train.py has produced outputs/*/best.ckpt.

Usage:
    python scripts/dpo_stage.py                     # default: dpo.yaml
    python scripts/dpo_stage.py --config-name=dpo   # explicit
    python scripts/dpo_stage.py dpo.lr=1e-6         # override a key
"""
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training import DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@hydra.main(config_path="../configs", config_name="dpo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    trainer = DPOTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
