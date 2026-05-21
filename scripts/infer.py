"""
scripts/infer.py — final LoViF 2026 challenge inference.
Produces PNG outputs ready for submission.

Usage:
    python scripts/infer.py config=configs/inference.yaml \
        inference.input_dir=./validation_inputs \
        inference.output_dir=./validation_outputs
"""
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@hydra.main(config_path="../configs", config_name="inference", version_base="1.3")
def main(cfg: DictConfig) -> None:
    engine = InferenceEngine(cfg)
    engine.run(
        input_dir=cfg.inference.input_dir,
        output_dir=cfg.inference.output_dir,
    )


if __name__ == "__main__":
    main()
