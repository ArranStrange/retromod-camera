"""Film simulation orchestration."""

from __future__ import annotations

import numpy as np

from film_profiles.base import FilmProfile
from processing import ops
from processing.grain import film_grain

LUT_SIZE = 65


def apply_color_stages(rgb: np.ndarray, profile: FilmProfile) -> np.ndarray:
    """The per-pixel colour mapping: every stage except the spatial ones
    (micro-contrast, vignette, grain). Pure function of pixel colour, so it
    can be baked into a 3D LUT."""
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

    return rgb


def bake_color_lut(profile: FilmProfile, size: int = LUT_SIZE) -> np.ndarray:
    """Sample apply_color_stages over an RGB lattice -> (size, size, size, 3)."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    lattice = np.stack([r, g, b], axis=-1).reshape(1, -1, 3)
    return apply_color_stages(lattice, profile).reshape(size, size, size, 3)


class FilmSimulator:
    """Applies a FilmProfile to a BGR uint8 frame."""

    def __init__(self, profile: FilmProfile) -> None:
        self._profile = profile
        self._lut: np.ndarray | None = None

    @property
    def profile(self) -> FilmProfile:
        return self._profile

    def set_profile(self, profile: FilmProfile) -> None:
        if profile is not self._profile:
            self._profile = profile
            self._lut = None

    def process(
        self,
        frame_bgr: np.ndarray,
        grain_seed: int | None = None,
        fast: bool = False,
    ) -> np.ndarray:
        """Run the full film emulation pipeline on one frame.

        fast=True routes the colour stages through a cached 3D LUT
        (baked once per profile) — for live view. The default precise
        path runs every stage exactly — for captures.
        """
        profile = self._profile
        rgb = ops.to_float_rgb(frame_bgr)

        if fast:
            if self._lut is None:
                self._lut = bake_color_lut(profile)
            rgb = ops.apply_color_lut(rgb, self._lut)
        else:
            rgb = apply_color_stages(rgb, profile)

        if profile.micro_contrast:
            rgb = ops.micro_contrast(rgb, profile.micro_contrast)

        if profile.vignette:
            rgb = ops.vignette(rgb, profile.vignette, profile.vignette_mid)

        rgb = film_grain(
            rgb, profile.grain_strength, size=profile.grain_size, seed=grain_seed
        )
        return ops.to_bgr_u8(rgb)
