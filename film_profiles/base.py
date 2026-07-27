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
    shadow_lift: float = 0.0  # lift crushed blacks
    monochrome: bool = False
    grain_strength: float = 0.0  # 0 = off
    tone_curve: np.ndarray | None = None  # 256-entry LUT, or None for linear

    def __post_init__(self) -> None:
        matrix = np.asarray(self.color_matrix, dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"{self.key}: color_matrix must be 3x3")

        if self.tone_curve is not None:
            curve = np.asarray(self.tone_curve, dtype=np.float32)
            if curve.shape != (256,):
                raise ValueError(f"{self.key}: tone_curve must have 256 entries")
