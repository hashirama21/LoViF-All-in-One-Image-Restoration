"""tests/test_core.py"""
import pytest
import torch
from src.utils.registry import Registry
from src.utils.metrics import MetricBag, get_category
from src.losses.composite import DiffusionLoss, CompositeLoss, LossWeights
from src.losses.dpo import DPOLoss


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_register_and_build():
    reg = Registry("test")

    @reg.register("dummy")
    class Dummy:
        def __init__(self, x): self.x = x

    assert reg.build("dummy", x=42).x == 42


def test_registry_duplicate_raises():
    reg = Registry("test2")

    @reg.register("dup")
    class A: pass

    with pytest.raises(KeyError):
        @reg.register("dup")
        class B: pass


def test_registry_unknown_key_raises():
    reg = Registry("test3")
    with pytest.raises(KeyError):
        reg.build("nonexistent")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected", [
    ("0001.png",  "blur"),
    ("0150.png",  "low_light"),
    ("0250.png",  "haze"),
    ("0350.png",  "rain"),
    ("0450.png",  "snow"),
    ("9999.png",  "unknown"),
])
def test_get_category(filename, expected):
    assert get_category(filename) == expected


def test_metric_bag_summary_keys():
    bag = MetricBag()
    pred = torch.rand(1, 3, 64, 64)
    gt   = torch.rand(1, 3, 64, 64)
    bag.update(pred, gt, category="blur")
    summary = bag.summary()
    assert "blur" in summary and "all" in summary
    assert {"psnr", "ssim", "lpips"}.issubset(summary["blur"])


def test_metric_bag_reset():
    bag = MetricBag()
    bag.update(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), category="blur")
    bag.reset()
    assert len(bag._records) == 0


# ---------------------------------------------------------------------------
# DiffusionLoss
# ---------------------------------------------------------------------------

def test_diffusion_loss_shape():
    loss_fn = DiffusionLoss()
    B = 2
    noise_pred   = torch.randn(B, 4, 8, 8)
    noise_target = torch.randn(B, 4, 8, 8)
    loss, bd = loss_fn(noise_pred, noise_target)
    assert loss.shape == ()
    assert "diffusion_mse" in bd


def test_diffusion_loss_zero_when_perfect():
    loss_fn = DiffusionLoss()
    x = torch.randn(2, 4, 8, 8)
    loss, _ = loss_fn(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_diffusion_loss_minsnr_weighting():
    """Min-SNR weights should lower loss at high-noise timesteps (low alpha)."""
    loss_fn_snr  = DiffusionLoss(snr_gamma=5.0)
    loss_fn_plain = DiffusionLoss(snr_gamma=0.0)
    pred   = torch.randn(4, 4, 8, 8)
    target = torch.randn(4, 4, 8, 8)
    # Very low alpha_t → very high noise → SNR ≪ 1 → weight < 1 → weighted loss < plain
    alpha_low = torch.full((4,), 0.01)
    loss_snr, _   = loss_fn_snr(pred, target, alphas_cumprod_t=alpha_low)
    loss_plain, _ = loss_fn_plain(pred, target)
    assert loss_snr.item() < loss_plain.item()


def test_composite_loss_default_weights():
    """CompositeLoss(None) must not raise and use default LossWeights."""
    loss_fn = CompositeLoss()
    pred = torch.rand(1, 3, 32, 32)
    gt   = torch.rand(1, 3, 32, 32)
    lq   = torch.rand(1, 3, 32, 32)
    total, bd = loss_fn(pred, gt, lq)
    assert total.item() > 0


# ---------------------------------------------------------------------------
# CompositeLoss — breakdown stores raw unweighted values
# ---------------------------------------------------------------------------

def test_composite_loss_breakdown_is_unweighted():
    weights = LossWeights(l1=2.0, lpips=0.0, adversarial=0.0)
    loss_fn = CompositeLoss(weights)
    pred = torch.rand(1, 3, 32, 32)
    gt   = torch.rand(1, 3, 32, 32)
    lq   = torch.rand(1, 3, 32, 32)
    total, bd = loss_fn(pred, gt, lq)
    import torch.nn.functional as F
    raw_l1 = F.l1_loss(pred, gt).item()
    assert bd["l1"] == pytest.approx(raw_l1, rel=1e-4)
    assert total.item() == pytest.approx(2.0 * raw_l1, rel=1e-4)


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------

def test_dpo_loss_margin_direction():
    dpo = DPOLoss(beta=0.1)
    B = 4
    loss, bd = dpo(
        torch.full((B,), -0.5),
        torch.full((B,), -2.0),
        torch.full((B,), -1.0),
        torch.full((B,), -1.0),
    )
    assert bd["reward_margin"] > 0
    assert loss.item() < 0.7


def test_dpo_loss_is_differentiable():
    dpo = DPOLoss(beta=0.1)
    lp_c = torch.randn(2, requires_grad=True)
    lp_r = torch.randn(2, requires_grad=True)
    loss, _ = dpo(lp_c, lp_r, torch.randn(2), torch.randn(2))
    loss.backward()
    assert lp_c.grad is not None
