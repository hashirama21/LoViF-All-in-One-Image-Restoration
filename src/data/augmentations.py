"""
Degradation augmentation — Strategy pattern.
Each degradation is a callable strategy returning a degraded image.
CompositeDegradationPipeline chains them based on configured probability.
"""
from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class DegradationStrategy(ABC):
    @abstractmethod
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: float32 [C, H, W] in [0, 1]. Returns degraded image."""


class GaussianNoiseDegradation(DegradationStrategy):
    def __init__(self, sigma_range: Tuple[float, float] = (10.0, 50.0)) -> None:
        self.sigma_range = sigma_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(*self.sigma_range) / 255.0
        return (img + torch.randn_like(img) * sigma).clamp(0, 1)


class MotionBlurDegradation(DegradationStrategy):
    def __init__(self, kernel_range: Tuple[int, int] = (7, 21)) -> None:
        self.kernel_range = kernel_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        k = random.randrange(self.kernel_range[0], self.kernel_range[1] + 1, 2)
        kernel = torch.zeros(k, k, device=img.device)
        kernel[k // 2, :] = 1.0 / k
        angle_rad = random.uniform(0, math.pi)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=torch.float32, device=img.device,
        ).unsqueeze(0)
        grid   = F.affine_grid(theta, (1, 1, k, k), align_corners=False)
        kernel = F.grid_sample(kernel.view(1, 1, k, k), grid, align_corners=False).squeeze()
        kernel = (kernel / kernel.sum().clamp(min=1e-8)).view(1, 1, k, k).expand(img.shape[0], 1, k, k)
        padded = F.pad(img.unsqueeze(0), [k // 2] * 4, mode="reflect")
        return F.conv2d(padded, kernel, groups=img.shape[0]).squeeze(0).clamp(0, 1)


class LowLightDegradation(DegradationStrategy):
    def __init__(self, gamma_range: Tuple[float, float] = (2.0, 4.0)) -> None:
        self.gamma_range = gamma_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img.pow(random.uniform(*self.gamma_range)).clamp(0, 1)


class HazeDegradation(DegradationStrategy):
    """
    Physics-based haze: I = J·t(x) + A·(1 − t(x))
    Depth is spatially varying (vertical gradient + noise) for realism.
    Uniform depth was physically incorrect — haze increases with scene depth.
    """

    def __init__(
        self,
        beta_range: Tuple[float, float] = (0.8, 1.5),
        A: float = 0.9,
    ) -> None:
        self.beta_range = beta_range
        self.A = A

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        beta = random.uniform(*self.beta_range)
        c, h, w = img.shape
        # Vertical gradient (top=near, bottom=far) + random spatial noise
        gradient = torch.linspace(0.3, 1.0, h, device=img.device).view(1, h, 1).expand(1, h, w)
        noise    = torch.rand(1, h, w, device=img.device) * 0.25
        depth    = (gradient + noise).clamp(0.1, 1.0)
        transmission = torch.exp(-beta * depth)               # [1, H, W]
        return (img * transmission + self.A * (1 - transmission)).clamp(0, 1)


class RainDegradation(DegradationStrategy):
    """Vectorised rain streak synthesis — no Python loops over streaks."""

    def __init__(
        self,
        n_streaks_range: Tuple[int, int] = (200, 600),
        max_len: int = 30,
    ) -> None:
        self.n_streaks_range = n_streaks_range
        self.max_len = max_len

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        n = random.randint(*self.n_streaks_range)
        c, h, w = img.shape
        dev = img.device

        xs      = torch.randint(0, w, (n,), device=dev)
        ys      = torch.randint(0, max(1, h - self.max_len), (n,), device=dev)
        lengths = torch.randint(10, self.max_len + 1, (n,), device=dev)
        intens  = torch.empty(n, device=dev).uniform_(0.6, 1.0)

        offsets = torch.arange(self.max_len, device=dev).unsqueeze(0)  # [1, max_len]
        mask    = offsets < lengths.unsqueeze(1)                        # [n, max_len]

        y_coords   = (ys.unsqueeze(1) + offsets).clamp(0, h - 1)       # [n, max_len]
        x_coords   = xs.unsqueeze(1).expand_as(y_coords)
        flat_idx   = (y_coords * w + x_coords).reshape(-1)
        flat_mask  = mask.reshape(-1)
        flat_intens = intens.unsqueeze(1).expand(-1, self.max_len).reshape(-1)

        rain = torch.zeros(h * w, device=dev)
        rain.scatter_add_(0, flat_idx[flat_mask], flat_intens[flat_mask])
        rain = rain.view(1, h, w).clamp(0, 1)

        return (img + rain.expand(c, -1, -1) * 0.4).clamp(0, 1)


class SnowDegradation(DegradationStrategy):
    def __init__(self, density: float = 0.02, brightness: float = 0.8) -> None:
        self.density    = density
        self.brightness = brightness

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        mask = (torch.rand(1, h, w, device=img.device) < self.density).float()
        snow = mask * self.brightness
        return (img * (1 - mask * 0.3) + snow.expand(c, -1, -1) * 0.3).clamp(0, 1)


# Singleton instances are stateless after __init__; safe for DataLoader workers.
DEGRADATION_REGISTRY: dict[str, DegradationStrategy] = {
    "noise":     GaussianNoiseDegradation(),
    "blur":      MotionBlurDegradation(),
    "low_light": LowLightDegradation(),
    "haze":      HazeDegradation(),
    "rain":      RainDegradation(),
    "snow":      SnowDegradation(),
}

# Physically plausible composite pairs
DEFAULT_COMPOSITE_PAIRS: List[Tuple[str, str]] = [
    ("blur",      "noise"),
    ("rain",      "low_light"),
    ("haze",      "noise"),
    ("snow",      "low_light"),
    ("blur",      "low_light"),
]


class CompositeDegradationPipeline:
    """
    Applies a random composite degradation pair with probability `composite_prob`.
    Max composite_prob = 0.40 — above this, per-task performance degrades.
    """

    _MAX_PROB = 0.40

    def __init__(
        self,
        composite_prob: float = 0.35,
        composite_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        if not 0.0 <= composite_prob <= self._MAX_PROB:
            raise ValueError(
                f"composite_prob={composite_prob} exceeds {self._MAX_PROB}; "
                "models lose single-task performance above this threshold."
            )
        self.composite_prob  = composite_prob
        self.composite_pairs = list(composite_pairs) if composite_pairs is not None else list(DEFAULT_COMPOSITE_PAIRS)

    def __call__(self, lq: torch.Tensor) -> torch.Tensor:
        if random.random() < self.composite_prob:
            for name in random.choice(self.composite_pairs):
                lq = DEGRADATION_REGISTRY[name](lq)
        return lq
