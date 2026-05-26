"""
DPO Trainer — preference optimization stage.
Runs AFTER the main fine-tuning has converged.
Loads (chosen, rejected, lq) triplets and applies DPO loss.
"""
from __future__ import annotations
import logging

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.data import DPOPairsDataset
from src.losses import DPOLoss
from src.models import RestorationPipeline, PipelineConfig
from src.utils import CheckpointManager, WandbLogger

logger = logging.getLogger(__name__)


class DPOTrainer:
    """
    Preference optimization stage.

    Reference model = frozen copy of the base checkpoint (no graph built).
    Policy model    = trainable (LoRA + encoder projections only).
    Log-prob proxy  = negative noise-prediction MSE in latent space.
    Breakdown is averaged across all accumulation sub-batches (not last-only).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16

        base_ckpt = cfg.dpo.base_checkpoint

        self.policy    = self._load_model(base_ckpt, trainable=True)
        self.reference = self._load_model(base_ckpt, trainable=False)

        self.dpo_loss = DPOLoss(beta=cfg.dpo.beta)

        ds = DPOPairsDataset(cfg.dpo.preferences_dir)
        self.loader = DataLoader(
            ds,
            batch_size=cfg.dpo.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        self.optimizer = AdamW(
            list(self.policy.trainable_parameters()),
            lr=cfg.dpo.lr,
        )
        warmup_steps = cfg.dpo.get("warmup_steps", 100)
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.dpo.max_steps - warmup_steps)
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[warmup_steps]
        )

        self.ckpt_manager = CheckpointManager(
            cfg.project.output_dir, monitor="reward_margin", mode="max"
        )
        self.wb = WandbLogger(
            project=cfg.project.name,
            run_name=cfg.project.run_name + "_dpo",
            config=dict(cfg),
            enabled=cfg.get("logging", {}).get("use_wandb", False),
        )

    def _load_model(self, ckpt_path: str, trainable: bool) -> RestorationPipeline:
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if "model_config" in state:
            model_cfg = PipelineConfig(**state["model_config"])
        else:
            cfg_m = self.cfg.model
            model_cfg = PipelineConfig(
                lora_rank=cfg_m.lora_rank,
                use_degradation_encoder=cfg_m.use_degradation_encoder,
                encoder_dim=cfg_m.get("encoder_dim", 512),
                use_physical_priors=cfg_m.get("use_physical_priors", True),
            )

        model = RestorationPipeline(model_cfg).to(self.device)
        model.load_state_dict(state["model_state"], strict=False)

        if not trainable:
            model.eval()
            model.requires_grad_(False)

        return model

    # ------------------------------------------------------------------
    # Log-prob proxies — separated so reference always runs under no_grad
    # ------------------------------------------------------------------

    def _compute_policy_log_prob(
        self, lq: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Policy proxy — gradients flow through noise_pred."""
        with torch.autocast(self.device.type, dtype=self._dtype):
            noise_pred, noise, _, _ = self.policy(lq, target)
        return -F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3])

    def _compute_ref_log_prob(
        self, lq: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Reference proxy — no graph constructed, no memory overhead."""
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self._dtype):
            noise_pred, noise, _, _ = self.reference(lq, target)
        return -F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3])

    # ------------------------------------------------------------------

    def fit(self) -> None:
        step     = 0
        max_steps = self.cfg.dpo.max_steps
        accum    = self.cfg.dpo.gradient_accumulation_steps

        logger.info(f"DPO stage — {max_steps} steps on {self.device}.")
        self.policy.train()

        loader_iter = iter(self.loader)
        breakdown: dict[str, float] = {}

        while step < max_steps:
            self.optimizer.zero_grad()

            # Accumulate breakdown across sub-batches for accurate logging
            accum_bd: dict[str, list[float]] = {}

            for _ in range(accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.loader)
                    batch = next(loader_iter)

                lq       = batch["lq"].to(self.device)
                chosen   = batch["chosen"].to(self.device)
                rejected = batch["rejected"].to(self.device)

                pi_chosen    = self._compute_policy_log_prob(lq, chosen)
                pi_rejected  = self._compute_policy_log_prob(lq, rejected)
                ref_chosen   = self._compute_ref_log_prob(lq, chosen)
                ref_rejected = self._compute_ref_log_prob(lq, rejected)

                loss, bd = self.dpo_loss(
                    pi_chosen, pi_rejected, ref_chosen, ref_rejected
                )
                (loss / accum).backward()

                for k, v in bd.items():
                    accum_bd.setdefault(k, []).append(v)

            breakdown = {k: sum(v) / len(v) for k, v in accum_bd.items()}

            torch.nn.utils.clip_grad_norm_(
                list(self.policy.trainable_parameters()), 1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            step += 1

            if step % 50 == 0:
                self.wb.log(breakdown, step=step)
                logger.info(
                    f"DPO step={step} loss={breakdown['dpo_loss']:.4f} "
                    f"margin={breakdown['reward_margin']:.4f}"
                )

            if step % 500 == 0:
                self.ckpt_manager.step(
                    breakdown["reward_margin"],
                    {
                        "model_state":  self.policy.state_dict(),
                        "model_config": self.policy.as_config_dict(),
                        "step":         step,
                    },
                    step,
                )

        self.wb.finish()
        logger.info("DPO stage complete.")
