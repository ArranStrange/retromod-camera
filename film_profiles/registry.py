"""All available film stock profiles."""

import numpy as np

from film_profiles.base import FilmProfile


def _warm_contrast_curve() -> np.ndarray:
    """S-curve with punchy highlights for slide film."""
    x = np.linspace(0, 1, 256, dtype=np.float32)
    return np.clip(1.05 * x**0.88 / (x**0.88 + (1 - x) ** 1.15 * 0.55), 0, 1)


def _soft_portrait_curve() -> np.ndarray:
    """Gentle rolloff for portrait negative film."""
    x = np.linspace(0, 1, 256, dtype=np.float32)
    return np.clip(x**0.92 * 0.97 + 0.03, 0, 1)


def _tungsten_curve() -> np.ndarray:
    """Lifted shadows for tungsten cinema stock."""
    x = np.linspace(0, 1, 256, dtype=np.float32)
    return np.clip(0.08 + x**0.85 * 0.92, 0, 1)


PROFILES: dict[str, FilmProfile] = {
    "kodachrome64": FilmProfile(
        key="kodachrome64",
        name="Kodachrome 64",
        box_color=(200, 45, 35),
        color_matrix=np.array(
            [
                [1.18, 0.06, -0.04],
                [0.02, 1.02, -0.04],
                [-0.06, 0.04, 0.92],
            ],
            dtype=np.float32,
        ),
        contrast=1.22,
        saturation=1.18,
        brightness=0.01,
        grain_strength=0.12,
        tone_curve=_warm_contrast_curve(),
    ),
    "portra400": FilmProfile(
        key="portra400",
        name="Kodak Portra 400",
        box_color=(245, 210, 170),
        color_matrix=np.array(
            [
                [1.04, 0.05, 0.01],
                [0.03, 1.00, 0.02],
                [-0.02, 0.03, 0.96],
            ],
            dtype=np.float32,
        ),
        contrast=0.95,
        saturation=0.88,
        brightness=0.02,
        grain_strength=0.08,
        tone_curve=_soft_portrait_curve(),
    ),
    "superia400": FilmProfile(
        key="superia400",
        name="Fujifilm Superia 400",
        box_color=(30, 120, 70),
        color_matrix=np.array(
            [
                [0.96, 0.02, 0.02],
                [0.04, 1.08, -0.02],
                [-0.04, 0.10, 1.06],
            ],
            dtype=np.float32,
        ),
        contrast=1.05,
        saturation=1.15,
        grain_strength=0.10,
    ),
    "trix400": FilmProfile(
        key="trix400",
        name="Kodak Tri-X 400",
        box_color=(30, 30, 30),
        color_matrix=np.eye(3, dtype=np.float32),
        contrast=1.35,
        saturation=0.0,
        monochrome=True,
        grain_strength=0.28,
    ),
    "hp5plus": FilmProfile(
        key="hp5plus",
        name="Ilford HP5 Plus",
        box_color=(180, 180, 180),
        color_matrix=np.eye(3, dtype=np.float32),
        contrast=1.12,
        saturation=0.0,
        monochrome=True,
        grain_strength=0.18,
    ),
    "cinestill800t": FilmProfile(
        key="cinestill800t",
        name="Cinestill 800T",
        box_color=(40, 80, 160),
        color_matrix=np.array(
            [
                [0.88, 0.04, 0.08],
                [0.02, 0.96, 0.06],
                [0.12, 0.08, 1.14],
            ],
            dtype=np.float32,
        ),
        contrast=1.08,
        saturation=1.05,
        shadow_lift=0.10,
        grain_strength=0.14,
        tone_curve=_tungsten_curve(),
    ),
}


def get_profile(key: str) -> FilmProfile:
    """Return a profile by registry key."""
    try:
        return PROFILES[key]
    except KeyError as exc:
        available = ", ".join(PROFILES)
        raise KeyError(f"Unknown profile {key!r}. Available: {available}") from exc
