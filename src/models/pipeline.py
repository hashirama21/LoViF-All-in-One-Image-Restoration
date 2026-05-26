"""
RestorationPipeline — full model wrapping FoundIR backbone with:
  - PEFT LoRA on U-Net attention layers
  - DegradationEncoder cross-attention conditioning: [B, 197 + P², encoder_dim]
    where P = prior_pool_size (default 7, giving 49 spatial prior tokens)
  - Physical priors spatially pooled to P×P tokens — locality preserved
  - DDPM training forward: noise-prediction in latent space (differentiable)
  - Full diffusion inference via restore() with null-conditioning CFG fix
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    prior_pool_size: int = 7       # spatial prior tokens: 7×7 = 49 per image
    num_inference_steps: int = 25
    guidance_scale: float = 1.5    # low CFG for fidelity (7.5 hallucinates)


@ModelRegistry.register("foundir_lora")
class RestorationPipeline(nn.Module):
    """
    Training  → forward(lq, gt) → (noise_pred, noise_target, z0_pred, alphas_cumprod_t).
    Inference → restore(lq) → denoised image [B, C, H, W] in [0, 1].
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        self.config = config

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.backbone_id,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
        )
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not installed — default attention used

        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        if config.lora_rank > 0:
            lora_cfg = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules),
            )
            self.pipe.unet = get_peft_model(self.pipe.unet, lora_cfg)

        self._cross_attn_dim: int = self.pipe.unet.config.cross_attention_dim

        self.degradation_encoder: Optional[DegradationEncoder] = None
        self.cond_proj: Optional[nn.Linear] = None
        self.retinex: Optional[RetinexPrior] = None
        self.dark_channel: Optional[DarkChannelPrior] = None
        self.prior_linear: Optional[nn.Linear] = None

        if config.use_degradation_encoder:
            self.degradation_encoder = DegradationEncoder(output_dim=config.encoder_dim)
            # Token-wise projection: encoder_dim → U-Net cross_attention_dim (768 for SD 1.x)
            self.cond_proj = nn.Linear(config.encoder_dim, self._cross_attn_dim)

            if config.use_physical_priors:
                self.retinex = RetinexPrior()
                self.dark_channel = DarkChannelPrior()
                # 5 prior channels: illum(1) + reflectance(3) + transmission(1)
                # Applied per spatial token after P×P pooling
                self.prior_linear = nn.Linear(5, config.encoder_dim)

    # ------------------------------------------------------------------
    # Training forward — DDPM noise-prediction in latent space
    # ------------------------------------------------------------------

    def forward(
        self,
        lq: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        lq: [B, C, H, W] float32 in [0, 1] — degraded input.
        gt: [B, C, H, W] float32 in [0, 1] — clean target.

        Returns:
            noise_pred       [B, 4, H/8, W/8]  U-Net prediction (differentiable)
            noise_target     [B, 4, H/8, W/8]  ground-truth noise
            z0_pred          [B, 4, H/8, W/8]  Tweedie x0 estimate for pixel losses
            alphas_cumprod_t [B]                for Min-SNR loss weighting
        """
        vae = self.pipe.vae
        scheduler = self.pipe.scheduler

        with torch.no_grad():
            z0 = (
                vae.encode((gt * 2.0 - 1.0).to(vae.dtype))
                .latent_dist.sample()
                * vae.config.scaling_factor
            )

        cond = self._build_conditioning(lq)    # [B, seq_len, cross_attn_dim]

        B = lq.shape[0]
        t = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=lq.device
        )
        noise  = torch.randn_like(z0)
        z_t    = scheduler.add_noise(z0, noise, t)

        noise_pred = self.pipe.unet(
            z_t.to(cond.dtype), t, encoder_hidden_states=cond,
        ).sample

        # alphas_cumprod is float32; not affected by autocast
        alpha_t = scheduler.alphas_cumprod[t].to(z0.device).view(-1, 1, 1, 1)  # [B,1,1,1]
        z0_pred = (
            z_t - (1.0 - alpha_t).sqrt() * noise_pred.float()
        ) / alpha_t.sqrt().clamp(min=1e-8)

        return noise_pred.float(), noise.float(), z0_pred, alpha_t.view(-1)  # [B]

    # ------------------------------------------------------------------
    # Inference — full diffusion denoising with null-conditioning CFG
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def restore(
        self,
        lq: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Full-quality inference. lq: [B, C, H, W] in [0, 1]."""
        steps = num_inference_steps if num_inference_steps is not None else self.config.num_inference_steps
        scale = guidance_scale      if guidance_scale      is not None else self.config.guidance_scale

        cond     = self._build_conditioning(lq)  # [B, seq_len, cross_attn_dim]
        neg_cond = torch.zeros_like(cond)         # null conditioning for CFG

        restored = self.pipe(
            prompt_embeds=cond,
            negative_prompt_embeds=neg_cond,
            image=lq,
            num_inference_steps=steps,
            guidance_scale=scale,
            strength=0.85,
            output_type="pt",
        ).images
        return restored.clamp(0, 1)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent → pixel image [B, C, H, W] in [0, 1]. Grads flow through z."""
        img = self.pipe.vae.decode(z / self.pipe.vae.config.scaling_factor).sample
        return ((img.float() + 1.0) / 2.0).clamp(0, 1)

    # ------------------------------------------------------------------
    # Conditioning — [B, 197 + P², cross_attn_dim]
    # ------------------------------------------------------------------

    def _build_conditioning(self, lq: torch.Tensor) -> torch.Tensor:
        """
        Build cross-attention conditioning sequence.
        With encoder + priors: [B, 197 + prior_pool_size², cross_attn_dim]
        Without encoder:       [B, 1, cross_attn_dim] (null zeros)
        """
        B = lq.shape[0]

        if self.degradation_encoder is None:
            return torch.zeros(
                B, 1, self._cross_attn_dim, device=lq.device, dtype=lq.dtype
            )

        deg_tokens = self.degradation_encoder(lq.float())  # [B, 197, encoder_dim]

        if self.retinex is not None:
            illum, reflect  = self.retinex(lq.float())      # [B,1,H,W] / [B,3,H,W]
            transmission    = self.dark_channel(lq.float())  # [B,1,H,W]

            priors = torch.cat([illum, reflect, transmission], dim=1)  # [B, 5, H, W]
            P = self.config.prior_pool_size
            priors = F.adaptive_avg_pool2d(priors, P)                  # [B, 5, P, P]
            prior_tokens = priors.permute(0, 2, 3, 1).reshape(B, P * P, 5)  # [B, P², 5]
            prior_tokens = self.prior_linear(prior_tokens)             # [B, P², encoder_dim]

            deg_tokens = torch.cat([deg_tokens, prior_tokens], dim=1)  # [B, 197+P², encoder_dim]

        return self.cond_proj(deg_tokens)  # [B, seq_len, cross_attn_dim]

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yields parameters with requires_grad=True."""
        return (p for p in self.parameters() if p.requires_grad)

    lora_parameters = trainable_parameters  # backward-compat alias

    def save_lora(self, path: str) -> None:
        self.pipe.unet.save_pretrained(path)

    def as_config_dict(self) -> dict:
        return dataclasses.asdict(self.config)

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "RestorationPipeline":
        return cls(PipelineConfig(**cfg))
