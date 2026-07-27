"""Film grain synthesis — see docs/film_grain.md for the research behind it."""

from __future__ import annotations

import cv2
import numpy as np

from processing.ops import luma


def film_grain(
    rgb: np.ndarray,
    strength: float,
    size: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Physically-motivated film grain (Boolean-model statistics).

    A correlated Gaussian field (AV1-style synthesis) scaled by the
    Boolean model's intensity response: zero fluctuation at pure black
    and pure white, peaking just below middle grey and biased slightly
    into the shadows like real silver-halide density.

    strength: amplitude (Fuji "roughness"); ~0.15 weak, ~0.3 strong
    size: clump size relative to the frame (Fuji "size"); 1 fine, 3+ coarse
    """
    if strength <= 0:
        return rgb

    h, w = rgb.shape[:2]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)

    # correlation length scales with resolution so grain has a fixed
    # apparent size on the frame, like grain on a physical negative
    sigma_px = size * min(h, w) / 1500.0
    if sigma_px >= 0.3:
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma_px)
        noise /= max(float(noise.std()), 1e-6)

    y = np.clip(luma(rgb), 0.0, 1.0)
    response = np.sqrt(y) * (1.0 - y) ** 0.8
    grain = noise * response * (strength * 0.5)
    return np.clip(rgb + grain[..., None], 0.0, 1.0)
