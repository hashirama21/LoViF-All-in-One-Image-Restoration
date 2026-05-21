"""
Trainer — main training loop for LoViF 2026.
Handles: DDPM diffusion loss + optional pixel-space losses,
mixed precision, gradient accumulation, early stopping,
per-category validation, WandB logging.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.data import LoViFDataset, LoViFValDataset
from src.losses import DiffusionLoss, CompositeLoss, LossWeights
from src.models import RestorationPipeline, PipelineConfig
from src.utils import MetricBag, CheckpointManager, WandbLogger

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently for reproducible augmentation."""
    import random
    import numpy as np
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


class Trainer:
    """
    Self-contained trainer. Instantiate with Hydra config, then call .fit().

    Design:
      - Primary loss: DDPM noise-prediction MSE (always, latent space).
      - Secondary loss: pixel-space L1 + LPIPS on decoded x0 estimate
        (activated when cfg.loss.pixel_loss_weight > 0).
      - Discriminator updated every step when adversarial weight > 0.
      - Only LoRA + encoder projection params are trained (backbone frozen).
      - Early stopping monitored on val LPIPS (lower = better).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float16

        # --- Model ---
        model_cfg = PipelineConfig(
            lora_rank=cfg.model.lora_rank,
            use_degradation_encoder=cfg.model.use_degradation_encoder,
            encoder_dim=cfg.model.get("encoder_dim", 512),
            use_physical_priors=cfg.model.get("use_physical_priors", True),
        )
        self.model = RestorationPipeline(model_cfg).to(self.device)

        if cfg.training.get("gradient_checkpointing", False):
            self.model.pipe.unet.enable_gradient_checkpointing()

        # --- Losses ---
        self.diff_criterion = DiffusionLoss()
        self._pixel_weight: float = cfg.loss.get("pixel_loss_weight", 0.0)
        loss_weights = LossWeights(
            l1=cfg.loss.l1_weight,
            lpips=cfg.loss.lpips_weight,
            adversarial=cfg.loss.adversarial_weight,
        )
        self.pixel_criterion: Optional[CompositeLoss] = (
            CompositeLoss(loss_weights).to(self.device)
            if self._pixel_weight > 0
            else None
        )

        # --- Data ---
        train_ds = LoViFDataset(
            cfg.data.train_dir,
            composite_prob=cfg.data.get("composite_prob", 0.35),
            augment=True,
        )
        val_ds = LoViFValDataset(cfg.data.val_dir, has_gt=True)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,     # key lives under `data:`
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )

        # --- Optimizer & scheduler ---
        trainable = list(self.model.trainable_parameters())
        self.optimizer = AdamW(
            trainable,
            lr=cfg.training.lr,
            weight_decay=cfg.training.get("weight_decay", 0.01),
        )
        warmup = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=cfg.training.warmup_steps
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.max_steps - cfg.training.warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine],
            milestones=[cfg.training.warmup_steps],
        )

        # Discriminator optimizer (only when adversarial is active)
        self.disc_optimizer: Optional[torch.optim.Optimizer] = None
        if (
            self.pixel_criterion is not None
            and self.pixel_criterion.adversarial is not None
        ):
            self.disc_optimizer = AdamW(
                self.pixel_criterion.adversarial.discriminator.parameters(),
                lr=cfg.training.lr * 0.5,
            )

        # GradScaler is only needed for fp16; bfloat16 has fp32 dynamic range
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.training.mixed_precision == "fp16")
        )

        # --- Infrastructure ---
        self.ckpt_manager = CheckpointManager(
            cfg.project.output_dir,
            monitor="lpips",
            mode="min",
            patience=cfg.training.get("early_stopping_patience", 8),
        )
        self.wb = WandbLogger(
            project=cfg.project.name,
            run_name=cfg.project.run_name,
            config=dict(cfg),
            enabled=cfg.get("logging", {}).get("use_wandb", False),
        )
        self.metrics = MetricBag()

    # ------------------------------------------------------------------

    def fit(self) -> None:
        cfg = self.cfg.training
        step = 0
        accum_steps = cfg.gradient_accumulation_steps
        device_type = self.device.type

        self.model.train()
        train_iter = iter(self.train_loader)
        logger.info(f"Training for {cfg.max_steps} steps on {self.device}.")

        while step < cfg.max_steps:
            self.optimizer.zero_grad()
            if self.disc_optimizer:
                self.disc_optimizer.zero_grad()

            breakdown: dict[str, float] = {}

            for _ in range(accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                lq = batch["lq"].to(self.device)
                gt = batch["gt"].to(self.device)

                with torch.autocast(device_type, dtype=self._dtype):
                    noise_pred, noise, z0_pred = self.model(lq, gt)

                    diff_loss, diff_bd = self.diff_criterion(noise_pred, noise)
                    breakdown.update(diff_bd)
                    total = diff_loss

                    pred_pixel = None
                    if self.pixel_criterion is not None:
                        pred_pixel = self.model.decode_latent(z0_pred)
                        pixel_loss, pixel_bd = self.pixel_criterion(
                            pred_pixel.float(), gt.float(), lq.float()
                        )
                        breakdown.update({f"px_{k}": v for k, v in pixel_bd.items()})
                        total = total + self._pixel_weight * pixel_loss

                    total = total / accum_steps

                self.scaler.scale(total).backward()

                # Discriminator step within the same accumulation iteration
                if self.disc_optimizer is not None:
                    with torch.autocast(device_type, dtype=self._dtype):
                        disc_loss = (
                            self.pixel_criterion.adversarial.forward_discriminator(
                                pred_pixel.detach(), gt.float(), lq.float()
                            ) / accum_steps
                        )
                    self.scaler.scale(disc_loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.model.trainable_parameters()), 1.0)
            self.scaler.step(self.optimizer)
            if self.disc_optimizer:
                self.scaler.step(self.disc_optimizer)
            self.scaler.update()
            self.scheduler.step()
            step += 1

            if step % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.wb.log({**breakdown, "lr": lr}, step=step)
                logger.info(f"step={step} {breakdown} lr={lr:.2e}")

            if step % cfg.eval_every == 0:
                if self._validate(step):
                    logger.info("Early stopping triggered.")
                    break

            if step % cfg.save_every == 0:
                self._save_periodic(step)

        self.wb.finish()
        logger.info("Training complete.")

    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _validate(self, step: int) -> bool:
        self.model.eval()
        self.metrics.reset()

        device_type = self.device.type
        for batch in self.val_loader:
            lq  = batch["lq"].to(self.device)
            gt  = batch["gt"].to(self.device)
            cat = batch["category"][0]

            with torch.autocast(device_type, dtype=self._dtype):
                pred = self.model.restore(lq)

            self.metrics.update(pred.float(), gt.float(), category=cat)

        summary = self.metrics.summary()
        global_lpips = summary["all"]["lpips"]
        global_psnr  = summary["all"]["psnr"]

        log_payload = {
            f"val/{cat}/{k}": v
            for cat, sub in summary.items()
            for k, v in sub.items()
            if k != "n"
        }
        self.wb.log(log_payload, step=step)
        logger.info(f"[val step={step}] PSNR={global_psnr:.2f} LPIPS={global_lpips:.4f}")
        for cat, vals in summary.items():
            if cat != "all":
                logger.info(f"  {cat:12s}: PSNR={vals['psnr']:.2f} LPIPS={vals['lpips']:.4f}")

        state = {
            "model_state":   self.model.state_dict(),
            "model_config":  self.model.as_config_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics":       summary,
        }
        should_stop = self.ckpt_manager.step(global_lpips, state, step)
        self.model.train()
        return should_stop

    def _save_periodic(self, step: int) -> None:
        state = {
            "model_state":  self.model.state_dict(),
            "model_config": self.model.as_config_dict(),
            "step":         step,
        }
        path = Path(self.cfg.project.output_dir) / f"ckpt_step{step}.pt"
        torch.save(state, path)