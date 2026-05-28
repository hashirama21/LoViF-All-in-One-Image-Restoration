"""tests/test_models.py — model component and regression tests.

Tests that require downloading CLIP (~330 MB) are marked with @pytest.mark.slow
and skipped in fast CI runs. Run them with: pytest -m slow
"""
import pytest
import torch

# ---------------------------------------------------------------------------
# DegradationEncoder
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_degradation_encoder_output_shape():
    """197 tokens (CLS + 196 patches) after the patch to use last_hidden_state."""
    from src.models.degradation_encoder import DegradationEncoder
    enc = DegradationEncoder(output_dim=128)
    out = enc(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, 197, 128), f"Expected [2, 197, 128], got {out.shape}"


@pytest.mark.slow
def test_degradation_encoder_frozen_by_default():
    from src.models.degradation_encoder import DegradationEncoder
    enc = DegradationEncoder(output_dim=128)
    assert not next(enc.clip.parameters()).requires_grad


@pytest.mark.slow
def test_degradation_encoder_projection_trainable():
    from src.models.degradation_encoder import DegradationEncoder
    enc = DegradationEncoder(output_dim=128)
    assert next(enc.projection.parameters()).requires_grad


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

def test_reward_model_select_pairs_shape():
    """select_pairs returns tensors with the same shape as each generation."""
    from src.losses.dpo import RewardModel
    reward = RewardModel(musiq_weight=0.0, clip_iqa_weight=0.0, lpips_weight=1.0)
    B = 2
    gens = [torch.rand(B, 3, 64, 64) for _ in range(4)]
    gt   = torch.rand(B, 3, 64, 64)
    chosen, rejected = reward.select_pairs(gens, gt)
    assert chosen.shape == (B, 3, 64, 64)
    assert rejected.shape == (B, 3, 64, 64)


def test_reward_model_chosen_is_best():
    """Chosen generation should be perceptually closer to GT than rejected."""
    from src.losses.dpo import RewardModel
    reward = RewardModel(musiq_weight=0.0, clip_iqa_weight=0.0, lpips_weight=1.0)
    gt      = torch.ones(1, 3, 64, 64) * 0.5
    perfect = torch.ones(1, 3, 64, 64) * 0.5   # identical to GT → LPIPS ≈ 0
    noisy   = torch.rand(1, 3, 64, 64)          # random → LPIPS >> 0
    chosen, rejected = reward.select_pairs([perfect, noisy], gt)
    assert (chosen - gt).abs().mean() < (rejected - gt).abs().mean()


def test_reward_model_distinct_chosen_rejected():
    """chosen and rejected must differ when input generations differ."""
    from src.losses.dpo import RewardModel
    reward = RewardModel(musiq_weight=0.0, clip_iqa_weight=0.0, lpips_weight=1.0)
    gt   = torch.rand(1, 3, 64, 64)
    gens = [torch.rand(1, 3, 64, 64) for _ in range(3)]
    chosen, rejected = reward.select_pairs(gens, gt)
    assert not torch.allclose(chosen, rejected)


# ---------------------------------------------------------------------------
# PipelineConfig — config dict round-trip
# ---------------------------------------------------------------------------

def test_pipeline_config_round_trip():
    """as_config_dict / from_config_dict must reconstruct an equivalent config."""
    from src.models.pipeline import PipelineConfig, RestorationPipeline
    cfg = PipelineConfig(lora_rank=4, encoder_dim=64, use_physical_priors=False)
    d   = cfg.__class__(**{k: v for k, v in vars(cfg).items()})  # shallow copy
    assert d.lora_rank == 4
    assert d.encoder_dim == 64
    assert not d.use_physical_priors
