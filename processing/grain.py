"""Film grain synthesis — see docs/film_grain.md for the research behind it."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from processing.ops import luma

_BANK_SIZE = 6
_MAX_BANK_PIXELS = 1_500_000  # bank small frames only; captures generate directly


def _make_field(h: int, w: int, sigma_px: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma_px)
    noise /= max(float(noise.std()), 1e-6)
    return noise


@lru_cache(maxsize=4)
def _field_bank(h: int, w: int, sigma_key: int) -> tuple[np.ndarray, ...]:
    """Pre-normalised correlated fields to cycle through in live view."""
    sigma_px = sigma_key / 100.0
    return tuple(_make_field(h, w, sigma_px, 9000 + i) for i in range(_BANK_SIZE))


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
    # correlation length scales with resolution so grain has a fixed
    # apparent size on the frame, like grain on a physical negative;
    # the floor keeps clumps softly resolved at low (preview) resolutions
    # instead of degenerating into harsh single-pixel speckle
    sigma_px = max(size * min(h, w) / 1500.0, 0.6)

    if h * w <= _MAX_BANK_PIXELS:
        bank = _field_bank(h, w, round(sigma_px * 100))
        index = seed if seed is not None else int(np.random.default_rng().integers(_BANK_SIZE))
        noise = bank[index % _BANK_SIZE]
    else:
        noise = _make_field(h, w, sigma_px, seed if seed is not None else None)

    y = np.clip(luma(rgb), 0.0, 1.0)
    response = np.sqrt(y) * (1.0 - y) ** 0.8
    grain = noise * response * (strength * 0.5)
    return np.clip(rgb + grain[..., None], 0.0, 1.0)
