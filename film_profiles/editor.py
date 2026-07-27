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
    path: str  # dotted path into the profile JSON, e.g. "grade.shadows.hue"
    step: float
    lo: float
    hi: float
    default: float = 0.0


def _hsl_page(component: str, step: float, lo: float, hi: float) -> tuple[ParamSpec, ...]:
    bands = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
    return tuple(
        ParamSpec(band, f"hsl.{band}.{component}", step, lo, hi) for band in bands
    )


PAGES: tuple[tuple[str, tuple[ParamSpec, ...]], ...] = (
    (
        "TONE",
        (
            ParamSpec("exposure", "exposure", 0.05, -2.0, 2.0),
            ParamSpec("contrast", "contrast", 0.02, 0.5, 2.0, default=1.0),
            ParamSpec("brightness", "brightness", 0.005, -0.3, 0.3),
            ParamSpec("highlights", "highlights", 0.02, -1.0, 1.0),
            ParamSpec("shadows", "shadows", 0.02, -1.0, 1.0),
            ParamSpec("whites", "whites", 0.02, -1.0, 1.0),
            ParamSpec("blacks", "blacks", 0.02, -1.0, 1.0),
            ParamSpec("shadow lift", "shadow_lift", 0.005, 0.0, 0.3),
            ParamSpec("filmic", "filmic", 0.02, 0.0, 1.0),
        ),
    ),
    (
        "COLOUR",
        (
            ParamSpec("warmth", "warmth", 0.005, -0.3, 0.3),
            ParamSpec("tint", "tint", 0.005, -0.3, 0.3),
            ParamSpec("saturation", "saturation", 0.02, 0.0, 2.0, default=1.0),
        ),
    ),
    (
        "GRADE",
        (
            ParamSpec("sh hue", "grade.shadows.hue", 5.0, 0.0, 360.0),
            ParamSpec("sh sat", "grade.shadows.sat", 0.02, 0.0, 1.0),
            ParamSpec("mid hue", "grade.midtones.hue", 5.0, 0.0, 360.0),
            ParamSpec("mid sat", "grade.midtones.sat", 0.02, 0.0, 1.0),
            ParamSpec("hi hue", "grade.highlights.hue", 5.0, 0.0, 360.0),
            ParamSpec("hi sat", "grade.highlights.sat", 0.02, 0.0, 1.0),
            ParamSpec("balance", "grade.balance", 0.05, -1.0, 1.0),
        ),
    ),
    ("HSL HUE", _hsl_page("hue", 1.0, -45.0, 45.0)),
    ("HSL SAT", _hsl_page("sat", 0.02, -1.0, 1.0)),
    ("HSL LUM", _hsl_page("lum", 0.02, -1.0, 1.0)),
    (
        "EFFECTS",
        (
            ParamSpec("micro contr", "micro_contrast", 0.01, -1.0, 1.0),
            ParamSpec("grain", "grain_strength", 0.01, 0.0, 0.6),
            ParamSpec("grain size", "grain_size", 0.25, 1.0, 8.0, default=1.0),
            ParamSpec("smudge", "color_smudge", 0.02, 0.0, 1.0),
            ParamSpec("vignette", "vignette", 0.02, -1.0, 1.0),
            ParamSpec("vignette mid", "vignette_mid", 0.02, 0.0, 1.0, default=0.5),
        ),
    ),
)


def _get_path(data: dict, path: str, default: float) -> float:
    node = data
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.get(part) or {}
    return float(node.get(parts[-1], default))


def _set_path(data: dict, path: str, value: float) -> None:
    node = data
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


class ProfileEditor:
    """Edits one profile's JSON data in memory, one parameter at a time."""

    def __init__(self, key: str, path: Path) -> None:
        self.key = key
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text())
        self.page_index = 0
        self.param_index = 0
        self.dirty = False

    @property
    def page_name(self) -> str:
        return PAGES[self.page_index][0]

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        return PAGES[self.page_index][1]

    @property
    def param(self) -> ParamSpec:
        return self.params[self.param_index]

    def value(self, spec: ParamSpec) -> float:
        return _get_path(self.data, spec.path, spec.default)

    def select_page(self, step: int) -> None:
        self.page_index = (self.page_index + step) % len(PAGES)
        self.param_index = 0

    def select(self, step: int) -> None:
        self.param_index = (self.param_index + step) % len(self.params)

    def adjust(self, steps: int) -> None:
        spec = self.param
        new = self.value(spec) + steps * spec.step
        _set_path(self.data, spec.path, round(min(spec.hi, max(spec.lo, new)), 4))
        self.dirty = True

    def build(self) -> FilmProfile:
        """Rebuild the profile from the current (possibly edited) data."""
        return profile_from_dict(self.key, self.data)

    def save(self) -> None:
        save_profile_data(self.data, self.path)
        self.dirty = False
