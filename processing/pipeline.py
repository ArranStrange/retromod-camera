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

        rgb = ops.tone_panel(
            rgb,
            exposure=profile.exposure,
            highlights=profile.highlights,
            shadows=profile.shadows,
            whites=profile.whites,
            blacks=profile.blacks,
        )

        if profile.filmic:
            rgb = ops.filmic(rgb, profile.filmic)

        if profile.tone_curve is not None:
            rgb = ops.apply_tone_curve(rgb, profile.tone_curve)

        rgb = ops.apply_channel_curves(
            rgb, profile.curve_red, profile.curve_green, profile.curve_blue
        )

        rgb = ops.adjust_tone(
            rgb,
            contrast=profile.contrast,
            brightness=profile.brightness,
            shadow_lift=profile.shadow_lift,
        )

        if profile.hsl and not profile.monochrome:
            rgb = ops.hsl_mixer(rgb, profile.hsl)

        if profile.monochrome:
            rgb = ops.to_monochrome(rgb)
        elif profile.saturation != 1.0:
            rgb = ops.adjust_saturation(rgb, profile.saturation)

        if profile.grade:
            rgb = ops.color_grade(rgb, profile.grade)

        if profile.micro_contrast:
            rgb = ops.micro_contrast(rgb, profile.micro_contrast)

        if profile.vignette:
            rgb = ops.vignette(rgb, profile.vignette, profile.vignette_mid)

        rgb = add_grain(
            rgb,
            profile.grain_strength,
            size=profile.grain_size,
            midtone=profile.grain_midtone,
            seed=grain_seed,
        )
        return ops.to_bgr_u8(rgb)
