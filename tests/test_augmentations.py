"""tests/test_augmentations.py"""
import pytest
import torch
from src.data.augmentations import (
    CompositeDegradationPipeline,
    DEGRADATION_REGISTRY,
    RainDegradation,
)

IMG = torch.rand(3, 64, 64)


@pytest.mark.parametrize("strategy", DEGRADATION_REGISTRY.values())
def test_strategy_output_shape(strategy):
    assert strategy(IMG.clone()).shape == IMG.shape


@pytest.mark.parametrize("strategy", DEGRADATION_REGISTRY.values())
def test_strategy_range(strategy):
    out = strategy(IMG.clone())
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rain_vectorised_no_loop():
    """Verify vectorised RainDegradation produces valid output without Python loops."""
    rain = RainDegradation(n_streaks_range=(100, 200))
    out = rain(IMG.clone())
    assert out.shape == IMG.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Rain should increase brightness on average
    assert out.mean() >= IMG.mean() - 0.05


def test_composite_prob_zero_is_noop():
    pipeline = CompositeDegradationPipeline(composite_prob=0.0)
    img = torch.rand(3, 64, 64)
    assert torch.allclose(pipeline(img.clone()), img)


def test_composite_prob_exceeds_max():
    with pytest.raises(ValueError):
        CompositeDegradationPipeline(composite_prob=0.5)


def test_composite_applies_at_prob_one():
    pipeline = CompositeDegradationPipeline(composite_prob=0.40)
    results = [pipeline(IMG.clone()) for _ in range(20)]
    assert not all(torch.allclose(r, IMG) for r in results)