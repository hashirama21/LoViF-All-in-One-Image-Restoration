# LoViF 2026 — All-in-One Image Restoration

**Second Challenge on Real-World All-in-One Image Restoration @ ECCV 2026**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/LoViF-All-in-One-Image-Restoration/blob/main/lovif2026_pipeline.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-LoViF--All--in--One-181717?logo=github)](https://github.com/hashirama21/LoViF-All-in-One-Image-Restoration)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗-Diffusers-FFD21E)](https://github.com/huggingface/diffusers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

LoViF 2026 is a competitive image restoration pipeline targeting five degradation types simultaneously: **blur · low-light · haze · rain · snow**.

| Component | Details |
|---|---|
| **Backbone** | FoundIR — Stable Diffusion Img2Img (fine-tuned) |
| **PEFT** | LoRA on U-Net attention layers (`to_q/k/v/out`) |
| **Conditioning** | CLIP-ViT-B/32 DegradationEncoder + physical priors |
| **Physical priors** | RetinexPrior (illumination/reflectance) + DarkChannelPrior (transmission) |
| **Training loss** | DDPM noise-prediction MSE + L1 + LPIPS + adversarial |
| **Preference opt.** | DPO stage with MUSIQ / CLIP-IQA reward model |
| **Inference** | TTA (hflip + 4 rotations) × ensemble over multiple checkpoints |

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Baseline (FoundIR on LoViF data, no extras)
python scripts/train.py --config-name=train_baseline

# 3. Full pipeline (LoRA + DegradationEncoder + physical priors + composite loss)
python scripts/train.py --config-name=train_full

# 4. DPO preference stage
python scripts/dpo_stage.py --config-name=dpo

# 5. Final inference (TTA + ensemble)
python scripts/infer.py --config-name=inference \
    inference.input_dir=./validation_inputs \
    inference.output_dir=./validation_outputs
```

---

## End-to-end Notebook

The notebook [`lovif2026_pipeline.ipynb`](lovif2026_pipeline.ipynb) covers the entire workflow in one place:

| Section | Content |
|---|---|
| 0 | Environment setup (Colab / local auto-detection) |
| 1 | Dependency installation |
| 2 | Data download & synthetic data generation |
| 3 | Dataset visualisation (GT / LQ / diff per category) |
| 4 | Dataset verification + augmentation demo |
| 5 | Baseline training (100 steps) |
| 6 | Full pipeline training (LoRA + encoder + priors) |
| 7 | DPO pair generation + preference training |
| 8 | Per-category evaluation (PSNR / SSIM / LPIPS) |
| 9 | TTA + ensemble inference |
| 10 | Summary table + competition recommendations |

**Run it directly on Google Colab:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/LoViF-All-in-One-Image-Restoration/blob/main/lovif2026_pipeline.ipynb)

---

## Project structure

```
lovif2026/
├── configs/
│   ├── train_baseline.yaml   # FoundIR vanilla fine-tune
│   ├── train_full.yaml       # LoRA + encoder + priors + pixel losses
│   ├── dpo.yaml              # DPO preference optimisation
│   └── inference.yaml        # TTA + ensemble settings
├── src/
│   ├── data/                 # Dataset, augmentations, degradation pipeline
│   ├── models/               # RestorationPipeline, DegradationEncoder, priors
│   ├── losses/               # DiffusionLoss, CompositeLoss, DPOLoss
│   ├── training/             # Trainer, DPOTrainer
│   ├── inference/            # InferenceEngine (TTA + ensemble)
│   └── utils/                # MetricBag, CheckpointManager, WandbLogger, Registry
├── scripts/
│   ├── train.py
│   ├── dpo_stage.py
│   ├── evaluate.py
│   └── infer.py
├── tests/                    # pytest unit tests
├── lovif2026_pipeline.ipynb  # End-to-end notebook
└── requirements.txt
```

---

## Challenge dataset layout

Images are indexed as follows across validation & test sets:

| Range | Degradation |
|---|---|
| 0001 – 0100 | Blur |
| 0101 – 0200 | Low-light |
| 0201 – 0300 | Haze |
| 0301 – 0400 | Rain |
| 0401 – 0500 | Snow |

---

## Tests

```bash
pytest tests/ -v
```

---

## References

- **FoundIR**: Hao Li et al., *Unleashing the Power of Large-Scale Diffusion-Based Generative Models for Image Restoration*, ICCV 2025.
- **WeatherBench**: Qiyuan Guan et al., *WeatherBench*, ACM MM 2025.
- **DPO**: Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023.
- **LPIPS**: Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018.