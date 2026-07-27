"""Film grain overlay."""

from __future__ import annotations

import cv2
import numpy as np

from processing.ops import luma, midtone_mask


def add_grain(
    rgb: np.ndarray,
    strength: float,
    size: float = 1.0,
    midtone: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add luminance-weighted grain to an RGB float image (0-1).

    strength: 0 = none, ~0.3 = heavy (Tri-X territory)
    size: 1 = per-pixel; larger renders coarser clumps
    midtone: 0 = uniform grain; 1 = grain lives in the midtones only,
             leaving highlights and deep shadows clean (silver-halide-like;
             see docs/hybrid_color_engine.md stage 4)
    """
    if strength <= 0:
        return rgb

    rng = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    sigma = strength * 0.14

    if size > 1.0:
        nh, nw = max(2, round(h / size)), max(2, round(w / size))
        noise = rng.normal(0.0, sigma, (nh, nw)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        noise = rng.normal(0.0, sigma, (h, w)).astype(np.float32)

    if midtone > 0:
        weight = (1.0 - midtone) + midtone * midtone_mask(luma(rgb))
        noise = noise * weight

    return np.clip(rgb + noise[..., None], 0.0, 1.0)
