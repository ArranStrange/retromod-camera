"""Film profile data model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilmProfile:
    """Describes one film simulation recipe."""

    key: str
    name: str
    box_color: tuple[int, int, int]  # RGB for rear-display film box mock
    color_matrix: np.ndarray  # 3x3 RGB transform
    subtitle: str = ""  # short descriptor shown under the sim name
    order: int = 0  # position in the swipe/cycle sequence
    contrast: float = 1.0
    saturation: float = 1.0
    brightness: float = 0.0  # additive offset in 0-1 space after normalize
    exposure: float = 0.0  # stops
    highlights: float = 0.0  # -1..+1
    shadows: float = 0.0  # -1..+1
    whites: float = 0.0  # -1..+1
    blacks: float = 0.0  # -1..+1
    filmic: float = 0.0  # 0..1, per-channel log-sigmoid strength
    micro_contrast: float = 0.0  # -1..+1, midtone-masked local contrast
    color_chrome: float = 0.0  # 0..1, deepen saturated colours (Fuji CCE)
    halation: float = 0.0  # 0..1, warm glow around bright highlights
    shadow_lift: float = 0.0  # lift crushed blacks
    monochrome: bool = False
    hsl: dict | None = None  # {band: {"hue": deg, "sat": -1..1, "lum": -1..1}}
    grade: dict | None = None  # split toning; see processing.ops.color_grade
    grain_strength: float = 0.0  # 0 = off
    grain_size: float = 1.0  # clump size relative to frame; 1 fine, 3+ coarse
    color_smudge: float = 0.0  # 0..1, chroma diffusion (dye-cloud softness)
    vignette: float = 0.0  # -1 dark corners .. +1 bright
    vignette_mid: float = 0.5  # where falloff starts (0..1)
    tone_curve: np.ndarray | None = None  # 256-entry luminance LUT, or None
    curve_red: np.ndarray | None = None  # optional per-channel 256-entry LUTs
    curve_green: np.ndarray | None = None
    curve_blue: np.ndarray | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.color_matrix, dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"{self.key}: color_matrix must be 3x3")

        for label in ("tone_curve", "curve_red", "curve_green", "curve_blue"):
            lut = getattr(self, label)
            if lut is not None:
                curve = np.asarray(lut, dtype=np.float32)
                if curve.shape != (256,):
                    raise ValueError(f"{self.key}: {label} must have 256 entries")
