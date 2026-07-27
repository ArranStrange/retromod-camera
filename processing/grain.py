"""Film grain overlay."""

from __future__ import annotations

import numpy as np


def add_grain(image_bgr: np.ndarray, strength: float, seed: int | None = None) -> np.ndarray:
    """
    Add luminance-weighted grain.

    strength: 0 = none, ~0.3 = heavy (Tri-X territory)
    """
    if strength <= 0:
        return image_bgr

    rng = np.random.default_rng(seed)
    h, w = image_bgr.shape[:2]
    noise = rng.normal(0.0, strength * 35.0, (h, w, 1)).astype(np.float32)

    image = image_bgr.astype(np.float32)
    grainy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return grainy
