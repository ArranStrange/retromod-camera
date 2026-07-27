"""Film simulation orchestration."""

from __future__ import annotations

import numpy as np

from film_profiles.base import FilmProfile
from processing.color_matrix import (
    adjust_contrast_saturation,
    apply_color_matrix,
    apply_tone_curve,
    to_monochrome,
)
from processing.grain import add_grain


class FilmSimulator:
    """Applies a FilmProfile to a BGR frame."""

    def __init__(self, profile: FilmProfile) -> None:
        self._profile = profile

    @property
    def profile(self) -> FilmProfile:
        return self._profile

    def set_profile(self, profile: FilmProfile) -> None:
        self._profile = profile

    def process(self, frame_bgr: np.ndarray, grain_seed: int | None = None) -> np.ndarray:
        """Run the full film emulation pipeline on one frame."""
        profile = self._profile
        result = apply_color_matrix(frame_bgr, profile.color_matrix)

        if profile.tone_curve is not None:
            result = apply_tone_curve(result, profile.tone_curve)

        result = adjust_contrast_saturation(
            result,
            contrast=profile.contrast,
            saturation=0.0 if profile.monochrome else profile.saturation,
            brightness=profile.brightness,
            shadow_lift=profile.shadow_lift,
        )

        if profile.monochrome:
            result = to_monochrome(result)

        result = add_grain(result, profile.grain_strength, seed=grain_seed)
        return result
