"""tests/test_physical_priors.py"""
import torch
import pytest
from src.models.physical_priors import RetinexPrior, DarkChannelPrior

B, C, H, W = 2, 3, 64, 64


def test_retinex_output_shapes():
    prior = RetinexPrior(sigma=11)
    illum, reflect = prior(torch.rand(B, C, H, W))
    assert illum.shape   == (B, 1, H, W)
    assert reflect.shape == (B, C, H, W)


def test_retinex_range():
    prior = RetinexPrior(sigma=11)
    illum, reflect = prior(torch.rand(B, C, H, W))
    assert illum.min() >= 0 and illum.max() <= 1.0
    assert reflect.min() >= 0 and reflect.max() <= 1.0


def test_dark_channel_shape():
    prior = DarkChannelPrior()
    assert prior(torch.rand(B, C, H, W)).shape == (B, 1, H, W)


def test_dark_channel_range():
    t = DarkChannelPrior()(torch.rand(B, C, H, W))
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_dark_channel_hazy_image():
    """Uniformly grey (hazy) image should produce low transmission."""
    t = DarkChannelPrior()(torch.full((1, 3, 64, 64), 0.85))
    assert t.mean() < 0.5


def test_dark_channel_channel_agnostic():
    """DarkChannelPrior must not hardcode C=3."""
    prior = DarkChannelPrior()
    img_1ch = torch.rand(2, 1, 64, 64)
    img_4ch = torch.rand(2, 4, 64, 64)
    assert prior(img_1ch).shape == (2, 1, 64, 64)
    assert prior(img_4ch).shape == (2, 1, 64, 64)
