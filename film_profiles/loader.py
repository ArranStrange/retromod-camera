"""Load film profiles from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from film_profiles.base import FilmProfile
from film_profiles.curves import build_tone_lut


def load_profile(path: Path) -> FilmProfile:
    """Read one profile JSON; the filename stem becomes the profile key."""
    data = json.loads(path.read_text())
    curve_points = data.get("tone_curve")

    return FilmProfile(
        key=path.stem,
        name=data["name"],
        subtitle=data.get("subtitle", ""),
        order=data.get("order", 0),
        box_color=tuple(data["box_color"]),
        color_matrix=np.array(data["color_matrix"], dtype=np.float32),
        contrast=data.get("contrast", 1.0),
        saturation=data.get("saturation", 1.0),
        brightness=data.get("brightness", 0.0),
        shadow_lift=data.get("shadow_lift", 0.0),
        monochrome=data.get("monochrome", False),
        grain_strength=data.get("grain_strength", 0.0),
        tone_curve=build_tone_lut(curve_points) if curve_points else None,
    )


def load_profiles(directory: Path) -> dict[str, FilmProfile]:
    """Load every *.json profile in a directory, sorted by its order field."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No profile JSON files found in {directory}")

    profiles = [load_profile(p) for p in paths]
    profiles.sort(key=lambda p: (p.order, p.name))
    return {p.key: p for p in profiles}
