# LoViF 2026 — All-in-One Image Restoration

**Second Challenge on Real-World All-in-One Image Restoration @ ECCV 2026**

## Architecture

```
lovif2026/
├── configs/               # Hydra YAML configs (model, data, training, inference)
├── src/
│   ├── data/              # Dataset, augmentations, degradation pipeline
│   ├── models/            # Backbone, degradation encoder, full pipeline
│   ├── losses/            # Composite loss, perceptual, adversarial, DPO
│   ├── training/          # Trainer, DPO stage, LCM distillation
│   ├── inference/         # Inference engine, TTA, ensemble
│   └── utils/             # Metrics, logging, registry, checkpoint
├── scripts/               # train.py, evaluate.py, infer.py, dpo_stage.py
├── tests/                 # Unit tests per module
└── requirements.txt
```

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Baseline (reproduce FoundIR on LoViF data)
python scripts/train.py config=configs/train_baseline.yaml

# 3. Full pipeline (LoRA + degradation encoder + composite augment)
python scripts/train.py config=configs/train_full.yaml

# 4. DPO preference stage
python scripts/dpo_stage.py config=configs/dpo.yaml

# 5. Final inference (TTA + ensemble)
python scripts/infer.py config=configs/inference.yaml \
    input_dir=./validation_inputs output_dir=./validation_outputs
```

## Challenge dataset layout

Images are indexed as follows (validation & test sets):
- 0001–0100 : Blur
- 0101–0200 : Low-light
- 0201–0300 : Haze
- 0301–0400 : Rain
- 0401–0500 : Snow

## Citation

FoundIR: Hao Li et al., ICCV 2025.
WeatherBench: Qiyuan Guan et al., ACM MM 2025.
