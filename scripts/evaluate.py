"""
scripts/evaluate.py — per-category evaluation on validation set.
Prints PSNR / SSIM / LPIPS per degradation type.

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/full_lora_encoder/best.ckpt \
        --val_dir data/validation
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import LoViFValDataset
from src.models import RestorationPipeline, PipelineConfig
from src.utils import MetricBag

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_dir",    required=True)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model — reconstruct config from checkpoint if available
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_config" in state:
        model_cfg = PipelineConfig(**state["model_config"])
    else:
        model_cfg = PipelineConfig()
    model = RestorationPipeline(model_cfg).to(device)
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()
    logger.info(f"Loaded: {args.checkpoint}")

    # Dataset
    ds = LoViFValDataset(args.val_dir, has_gt=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    metrics = MetricBag()

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            lq  = batch["lq"].to(device)
            gt  = batch["gt"].to(device)
            cat = batch["category"][0]

            with torch.autocast(device.type, dtype=torch.bfloat16):
                pred = model.restore(lq)

            metrics.update(pred.float(), gt.float(), category=cat)

    summary = metrics.summary()

    print("\n" + "=" * 55)
    print(f"{'Category':<14} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'N':>5}")
    print("-" * 55)
    for cat in ["blur", "low_light", "haze", "rain", "snow", "all"]:
        if cat not in summary:
            continue
        v = summary[cat]
        marker = " ←" if cat == "all" else ""
        print(
            f"{cat:<14} {v['psnr']:>8.2f} {v['ssim']:>8.4f} "
            f"{v['lpips']:>8.4f} {int(v['n']):>5}{marker}"
        )
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
