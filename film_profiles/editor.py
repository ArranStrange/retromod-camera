"""Keyboard-driven profile editing state for the tuning studio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from film_profiles.base import FilmProfile
from film_profiles.loader import profile_from_dict, save_profile_data


@dataclass(frozen=True)
class ParamSpec:
    label: str
    key: str
    step: float
    lo: float
    hi: float
    default: float = 0.0


PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("contrast", "contrast", 0.02, 0.5, 2.0, default=1.0),
    ParamSpec("saturation", "saturation", 0.02, 0.0, 2.0, default=1.0),
    ParamSpec("brightness", "brightness", 0.005, -0.3, 0.3),
    ParamSpec("shadow lift", "shadow_lift", 0.005, 0.0, 0.3),
    ParamSpec("warmth", "warmth", 0.005, -0.3, 0.3),
    ParamSpec("tint", "tint", 0.005, -0.3, 0.3),
    ParamSpec("grain", "grain_strength", 0.01, 0.0, 0.5),
)


class ProfileEditor:
    """Edits one profile's JSON data in memory, one parameter at a time."""

    def __init__(self, key: str, path: Path) -> None:
        self.key = key
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text())
        self.param_index = 0
        self.dirty = False

    @property
    def param(self) -> ParamSpec:
        return PARAMS[self.param_index]

    def value(self, spec: ParamSpec) -> float:
        return float(self.data.get(spec.key, spec.default))

    def select(self, step: int) -> None:
        self.param_index = (self.param_index + step) % len(PARAMS)

    def adjust(self, steps: int) -> None:
        spec = self.param
        new = self.value(spec) + steps * spec.step
        self.data[spec.key] = round(min(spec.hi, max(spec.lo, new)), 4)
        self.dirty = True

    def build(self) -> FilmProfile:
        """Rebuild the profile from the current (possibly edited) data."""
        return profile_from_dict(self.key, self.data)

    def save(self) -> None:
        save_profile_data(self.data, self.path)
        self.dirty = False
