"""Load and save film profiles as JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from film_profiles.base import FilmProfile
from film_profiles.curves import build_tone_lut


def profile_from_dict(key: str, data: dict) -> FilmProfile:
    """Build a FilmProfile from parsed JSON data.

    warmth (red-blue) and tint (green-magenta) are baked into the
    colour matrix here so the pipeline itself stays unchanged.
    """
    matrix = np.array(data["color_matrix"], dtype=np.float32)
    warmth = float(data.get("warmth", 0.0))
    tint = float(data.get("tint", 0.0))
    if warmth:
        matrix[0, 0] += warmth
        matrix[2, 2] -= warmth
    if tint:
        matrix[1, 1] += tint

    curve_points = data.get("tone_curve")

    return FilmProfile(
        key=key,
        name=data["name"],
        subtitle=data.get("subtitle", ""),
        order=data.get("order", 0),
        box_color=tuple(data["box_color"]),
        color_matrix=matrix,
        contrast=data.get("contrast", 1.0),
        saturation=data.get("saturation", 1.0),
        brightness=data.get("brightness", 0.0),
        exposure=data.get("exposure", 0.0),
        highlights=data.get("highlights", 0.0),
        shadows=data.get("shadows", 0.0),
        whites=data.get("whites", 0.0),
        blacks=data.get("blacks", 0.0),
        shadow_lift=data.get("shadow_lift", 0.0),
        monochrome=data.get("monochrome", False),
        grain_strength=data.get("grain_strength", 0.0),
        tone_curve=build_tone_lut(curve_points) if curve_points else None,
    )


def load_profile(path: Path) -> FilmProfile:
    """Read one profile JSON; the filename stem becomes the profile key."""
    return profile_from_dict(path.stem, json.loads(path.read_text()))


def load_profiles(directory: Path) -> dict[str, FilmProfile]:
    """Load every *.json profile in a directory, sorted by its order field."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No profile JSON files found in {directory}")

    profiles = [load_profile(p) for p in paths]
    profiles.sort(key=lambda p: (p.order, p.name))
    return {p.key: p for p in profiles}


def save_profile_data(data: dict, path: Path) -> None:
    """Write profile JSON back to disk (validates it builds first)."""
    profile_from_dict(Path(path).stem, data)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
