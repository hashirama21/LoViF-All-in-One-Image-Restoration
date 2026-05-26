"""
Dataset classes for LoViF 2026.
- LoViFDataset     : paired LQ/GT for training (with augmentation)
- LoViFValDataset  : paired or LQ-only for validation
- DPOPairsDataset  : (lq, chosen, rejected) triplets for preference stage
- InferenceDataset : LQ-only for challenge submission (preserves original resolution)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.augmentations import CompositeDegradationPipeline, DEFAULT_COMPOSITE_PAIRS
from src.utils.registry import DatasetRegistry
from src.utils.metrics import get_category


def _load_image(path: Path) -> torch.Tensor:
    """Load PNG/JPG → float32 [C, H, W] in [0, 1]."""
    return transforms.ToTensor()(Image.open(path).convert("RGB"))


def _resize_512(img: torch.Tensor) -> torch.Tensor:
    if img.shape[-2:] != (512, 512):
        img = TF.resize(
            img, [512, 512],
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
    return img


def _find_gt_dir(lq_root: Path, gt_dir: Optional[str]) -> Path:
    """Resolve GT directory from an explicit path or common naming conventions."""
    if gt_dir is not None:
        p = Path(gt_dir)
        if not p.exists():
            raise FileNotFoundError(f"Explicit gt_dir not found: {p}")
        return p
    for old, new in [("lq", "gt"), ("inputs", "gt"), ("input", "gt")]:
        candidate = lq_root.parent / lq_root.name.replace(old, new)
        if candidate != lq_root and candidate.exists():
            return candidate
    sibling = lq_root.parent / "gt"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        f"Cannot infer GT directory for '{lq_root}'. "
        "Pass gt_dir explicitly or use a recognised naming convention (e.g. 'inputs' → 'gt')."
    )


def _image_files(directory: Path) -> List[Path]:
    """Return sorted list of PNG/JPG files in a directory."""
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


@DatasetRegistry.register("lovif_train")
class LoViFDataset(Dataset):
    """
    Paired LQ/GT training dataset.

    Expected layout:
        root/
            blur/lq/      blur/gt/
            low_light/lq/ low_light/gt/
            haze/lq/      haze/gt/
            rain/lq/      rain/gt/
            snow/lq/      snow/gt/
    """

    CATEGORIES = ["blur", "low_light", "haze", "rain", "snow"]

    def __init__(
        self,
        root: str,
        composite_prob: float = 0.35,
        augment: bool = True,
        crop_size: Optional[int] = None,
        composite_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        self.augment   = augment
        self.crop_size = crop_size
        self.composite = CompositeDegradationPipeline(
            composite_prob=composite_prob,
            composite_pairs=composite_pairs,  # None → uses DEFAULT_COMPOSITE_PAIRS internally
        )
        self.pairs: List[Tuple[Path, Path, str]] = []

        root_path = Path(root)
        for cat in self.CATEGORIES:
            lq_dir = root_path / cat / "lq"
            gt_dir = root_path / cat / "gt"
            if not lq_dir.exists():
                continue
            for lq_path in _image_files(lq_dir):
                gt_path = gt_dir / lq_path.name
                if gt_path.exists():
                    self.pairs.append((lq_path, gt_path, cat))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        lq_path, gt_path, category = self.pairs[idx]
        lq = _resize_512(_load_image(lq_path))
        gt = _resize_512(_load_image(gt_path))

        if self.augment:
            lq = self.composite(lq)

            if torch.rand(1).item() < 0.5:
                lq, gt = TF.hflip(lq), TF.hflip(gt)
            if torch.rand(1).item() < 0.5:
                lq, gt = TF.vflip(lq), TF.vflip(gt)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                lq = torch.rot90(lq, k=k, dims=[1, 2])
                gt = torch.rot90(gt, k=k, dims=[1, 2])

        if self.crop_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(
                lq, output_size=(self.crop_size, self.crop_size)
            )
            lq = TF.crop(lq, i, j, h, w)
            gt = TF.crop(gt, i, j, h, w)

        return {"lq": lq, "gt": gt, "category": category, "filename": lq_path.name}


@DatasetRegistry.register("lovif_val")
class LoViFValDataset(Dataset):
    """
    Validation set — flat directory, filenames encode category via index.
    Layout:  val_dir/*.png  (0001–0500)

    Args:
        root:   Path to directory containing LQ images.
        gt_dir: Path to GT directory. Auto-inferred if None (see _find_gt_dir).
        has_gt: Set False for the test set (no GT available).
    """

    def __init__(
        self,
        root: str,
        gt_dir: Optional[str] = None,
        has_gt: bool = True,
    ) -> None:
        root_path = Path(root)
        self.lq_files = _image_files(root_path)

        self.gt_files: Optional[List[Path]] = None
        if has_gt:
            _gt_root = _find_gt_dir(root_path, gt_dir)
            self.gt_files = [_gt_root / f.name for f in self.lq_files]
            missing = [p for p in self.gt_files if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} GT files not found in {_gt_root} "
                    f"(e.g. {missing[0].name})"
                )

    def __len__(self) -> int:
        return len(self.lq_files)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        lq_path = self.lq_files[idx]
        item: Dict[str, object] = {
            "lq":       _resize_512(_load_image(lq_path)),
            "filename": lq_path.name,
            "category": get_category(lq_path.name),
        }
        if self.gt_files is not None:
            item["gt"] = _resize_512(_load_image(self.gt_files[idx]))
        return item


@DatasetRegistry.register("dpo_pairs")
class DPOPairsDataset(Dataset):
    """
    DPO triplets: (lq, chosen, rejected).
    Layout:
        dpo_dir/lq/  dpo_dir/chosen/  dpo_dir/rejected/
    """

    def __init__(self, root: str) -> None:
        root_path = Path(root)
        self.lq_dir       = root_path / "lq"
        self.chosen_dir   = root_path / "chosen"
        self.rejected_dir = root_path / "rejected"

        self.filenames = [f.name for f in _image_files(self.chosen_dir)]

        missing = [
            n for n in self.filenames
            if not (self.lq_dir / n).exists() or not (self.rejected_dir / n).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} DPO files missing in lq/ or rejected/ "
                f"(e.g. {missing[0]})"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.filenames[idx]
        return {
            "lq":       _resize_512(_load_image(self.lq_dir / name)),
            "chosen":   _resize_512(_load_image(self.chosen_dir / name)),
            "rejected": _resize_512(_load_image(self.rejected_dir / name)),
            "filename": name,
        }


class InferenceDataset(Dataset):
    """
    LQ-only dataset for final challenge submission.
    Stores original resolution so the engine can resize output back before saving.
    """

    def __init__(self, input_dir: str) -> None:
        self.files = _image_files(Path(input_dir))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.files[idx]
        img  = _load_image(path)
        orig_h, orig_w = img.shape[-2], img.shape[-1]
        return {
            "lq":       _resize_512(img),
            "filename": path.name,
            "category": get_category(path.name),
            "orig_h":   orig_h,
            "orig_w":   orig_w,
        }