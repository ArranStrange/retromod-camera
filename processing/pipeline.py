"""Film simulation orchestration."""

from __future__ import annotations

import numpy as np

from film_profiles.base import FilmProfile
from processing import ops
from processing.grain import add_grain


class FilmSimulator:
    """Applies a FilmProfile to a BGR uint8 frame."""

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
        rgb = ops.to_float_rgb(frame_bgr)

        rgb = ops.apply_color_matrix(rgb, profile.color_matrix)

        if profile.tone_curve is not None:
            rgb = ops.apply_tone_curve(rgb, profile.tone_curve)

        rgb = ops.adjust_tone(
            rgb,
            contrast=profile.contrast,
            brightness=profile.brightness,
            shadow_lift=profile.shadow_lift,
        )

        if profile.monochrome:
            rgb = ops.to_monochrome(rgb)
        elif profile.saturation != 1.0:
            rgb = ops.adjust_saturation(rgb, profile.saturation)

        rgb = add_grain(rgb, profile.grain_strength, seed=grain_seed)
        return ops.to_bgr_u8(rgb)
