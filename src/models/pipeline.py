"""
RestorationPipeline — full model wrapping FoundIR backbone with:
  - PEFT LoRA on U-Net attention layers
  - DegradationEncoder cross-attention conditioning (LQ features)
  - Physical priors (Retinex, dark channel) pooled into conditioning
  - DDPM training forward: noise-prediction in latent space (differentiable)
  - Full diffusion inference via restore()
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from diffusers import StableDiffusionImg2ImgPipeline
from peft import LoraConfig, get_peft_model

from src.models.degradation_encoder import DegradationEncoder
from src.models.physical_priors import RetinexPrior, DarkChannelPrior
from src.utils.registry import ModelRegistry


@dataclass
class PipelineConfig:
    backbone_id: str = "House-Leo/FoundIR"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("to_q", "to_k", "to_v", "to_out.0")
    encoder_dim: int = 512
    use_degradation_encoder: bool = True
    use_physical_priors: bool = True
    num_inference_steps: int = 25
    guidance_scale: float = 7.5


@ModelRegistry.register("foundir_lora")
class RestorationPipeline(nn.Module):
    """
    Training  → forward(lq, gt) returns (noise_pred, noise_target, z0_pred).
    Inference → restore(lq) runs the full denoising pipeline.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self.config = config

        # Base diffusion pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.backbone_id,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
        )
        self.pipe.enable_xformers_memory_efficient_attention()

        # Freeze components that are never trained
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        # LoRA on U-Net attention layers
        if config.lora_rank > 0:
            lora_cfg = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules),
            )
            self.pipe.unet = get_peft_model(self.pipe.unet, lora_cfg)

        self._cross_attn_dim: int = self.pipe.unet.config.cross_attention_dim

        # Degradation conditioning modules (all None if encoder disabled)
        self.degradation_encoder: Optional[DegradationEncoder] = None
        self.cond_proj: Optional[nn.Linear] = None
        self.retinex: Optional[RetinexPrior] = None
        self.dark_channel: Optional[DarkChannelPrior] = None
        self.prior_linear: Optional[nn.Linear] = None

        if config.use_degradation_encoder:
            self.degradation_encoder = DegradationEncoder(output_dim=config.encoder_dim)
            # Project encoder_dim → U-Net cross_attention_dim (768 for SD 1.x)
            self.cond_proj = nn.Linear(config.encoder_dim, self._cross_attn_dim)

            if config.use_physical_priors:
                self.retinex = RetinexPrior()
                self.dark_channel = DarkChannelPrior()
                # illum(1) + reflectance(3) + transmission(1) = 5 → encoder_dim
                self.prior_linear = nn.Linear(5, config.encoder_dim)

    # ------------------------------------------------------------------
    # Training forward — DDPM noise-prediction in latent space
    # ------------------------------------------------------------------

    def forward(
        self,
        lq: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        lq: [B, C, H, W] float32 in [0, 1] — degraded input.
        gt: [B, C, H, W] float32 in [0, 1] — clean target.

        Returns:
            noise_pred   [B, 4, H/8, W/8]  U-Net prediction (differentiable)
            noise_target [B, 4, H/8, W/8]  ground-truth noise
            z0_pred      [B, 4, H/8, W/8]  x0 estimate for optional pixel losses
        """
        vae = self.pipe.vae
        scheduler = self.pipe.scheduler

        # Encode GT to latent; VAE is frozen so no grad is needed here
        with torch.no_grad():
            z0 = (
                vae.encode((gt * 2.0 - 1.0).to(vae.dtype))
                .latent_dist.sample()
                * vae.config.scaling_factor
            )

        cond = self._build_conditioning(lq)             # [B, 1, cross_attn_dim]

        B = lq.shape[0]
        t = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=lq.device
        )
        noise = torch.randn_like(z0)
        z_t = scheduler.add_noise(z0, noise, t)

        # LoRA weights receive gradients through this call
        noise_pred = self.pipe.unet(
            z_t.to(cond.dtype),
            t,
            encoder_hidden_states=cond,
        ).sample

        # Approximate x0 for pixel-space loss (TWEEDIE estimate)
        alpha_t = scheduler.alphas_cumprod[t].to(z0.device).view(-1, 1, 1, 1)
        z0_pred = (
            z_t - (1.0 - alpha_t).sqrt() * noise_pred.float()
        ) / alpha_t.sqrt().clamp(min=1e-8)

        return noise_pred.float(), noise.float(), z0_pred

    # ------------------------------------------------------------------
    # Inference — full diffusion denoising
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def restore(self, lq: torch.Tensor) -> torch.Tensor:
        """Full-quality inference. lq: [B, C, H, W] in [0, 1]."""
        cond = self._build_conditioning(lq)
        restored = self.pipe(
            prompt_embeds=cond,
            image=lq,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            strength=0.85,
            output_type="pt",
        ).images
        return restored.clamp(0, 1)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent → pixel image [B, C, H, W] in [0, 1].
        VAE weights are frozen; gradients flow through z.
        """
        img = self.pipe.vae.decode(z / self.pipe.vae.config.scaling_factor).sample
        return ((img.float() + 1.0) / 2.0).clamp(0, 1)

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def _build_conditioning(self, lq: torch.Tensor) -> torch.Tensor:
        """Build cross-attention conditioning [B, 1, cross_attn_dim]."""
        B = lq.shape[0]

        if self.degradation_encoder is None:
            return torch.zeros(
                B, 1, self._cross_attn_dim, device=lq.device, dtype=lq.dtype
            )

        deg_embed = self.degradation_encoder(lq.float())   # [B, encoder_dim]

        if self.retinex is not None:
            illum, reflect = self.retinex(lq.float())       # [B,1,H,W] / [B,3,H,W]
            transmission = self.dark_channel(lq.float())    # [B,1,H,W]
            prior_feat = torch.cat(
                [illum, reflect, transmission], dim=1
            ).mean(dim=[-2, -1])                            # [B, 5]
            deg_embed = deg_embed + self.prior_linear(prior_feat)

        return self.cond_proj(deg_embed).unsqueeze(1)       # [B, 1, cross_attn_dim]

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yields only parameters with requires_grad=True."""
        return (p for p in self.parameters() if p.requires_grad)

    # Alias kept for any external callers
    lora_parameters = trainable_parameters

    def save_lora(self, path: str) -> None:
        self.pipe.unet.save_pretrained(path)

    def as_config_dict(self) -> dict:
        return dataclasses.asdict(self.config)

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "RestorationPipeline":
        return cls(PipelineConfig(**cfg))