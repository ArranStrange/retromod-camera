"""Float-space image operations for the film pipeline.

Every function takes and returns an RGB float32 array with values in 0-1.
Conversion from/to 8-bit BGR happens once, at the pipeline edges, so
intermediate steps never quantise.
"""

from __future__ import annotations

import cv2
import numpy as np

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec. 709


def to_float_rgb(image_bgr_u8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr_u8, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def to_bgr_u8(rgb: np.ndarray) -> np.ndarray:
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA_WEIGHTS


def apply_color_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.clip(rgb @ matrix.T.astype(np.float32), 0.0, 1.0)


def apply_tone_curve(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map luminance through the LUT, preserving chroma ratios."""
    y = luma(rgb)
    y_mapped = np.interp(y, np.linspace(0.0, 1.0, lut.size), lut).astype(np.float32)
    scale = y_mapped / np.maximum(y, 1e-4)
    return np.clip(rgb * scale[..., None], 0.0, 1.0)


def filmic(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Per-channel filmic S-curve (see docs/hybrid_color_engine.md).

    A log-domain sigmoid: steep shadow density, compressive highlight knee.
    Applied per channel rather than on luma, so highlights desaturate
    naturally as channels saturate — the film-like roll-off. The spec's
    piecewise log mapping is discontinuous near black; this uses a
    continuous endpoint-preserving equivalent (x^a / (x^a + m^a), which is
    a logistic curve in log-exposure).
    """
    if strength <= 0:
        return rgb

    a, m = 1.7, 0.5
    x = np.clip(rgb, 0.0, 1.0)
    xa = x**a
    curved = xa * (1.0 + m**a) / (xa + m**a)
    return np.clip(x + (curved - x) * strength, 0.0, 1.0)


def midtone_mask(y: np.ndarray) -> np.ndarray:
    """Gaussian luminance mask peaking at middle grey (spec stage 3/4)."""
    return np.exp(-((y - 0.5) ** 2) / 0.08)


def micro_contrast(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Leica-style midtone micro-contrast ("3D pop").

    Boosts (or, negative, softens) high-frequency detail only where the
    midtone mask is strong; highlights and deep shadows stay untouched.
    The blur radius scales with resolution so the effect matches between
    the 480p preview and a full-res capture.
    """
    if not amount:
        return rgb

    sigma = max(1.0, 2.0 * min(rgb.shape[:2]) / 480.0)
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
    mask = midtone_mask(luma(rgb))
    return np.clip(rgb + (rgb - blurred) * amount * mask[..., None], 0.0, 1.0)


def apply_channel_curves(
    rgb: np.ndarray,
    lut_red: np.ndarray | None = None,
    lut_green: np.ndarray | None = None,
    lut_blue: np.ndarray | None = None,
) -> np.ndarray:
    """Map each channel through its own LUT (None = leave channel linear)."""
    luts = (lut_red, lut_green, lut_blue)
    if all(lut is None for lut in luts):
        return rgb

    out = rgb.copy()
    for channel, lut in enumerate(luts):
        if lut is not None:
            xs = np.linspace(0.0, 1.0, lut.size)
            out[..., channel] = np.interp(rgb[..., channel], xs, lut).astype(np.float32)
    return out


def tone_panel(
    rgb: np.ndarray,
    exposure: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    blacks: float = 0.0,
) -> np.ndarray:
    """Lightroom-style basic tone adjustments, all in -1..+1 (exposure in stops).

    highlights/shadows use bell weights peaked in their range so the other
    end of the tonal scale is untouched; whites/blacks move the endpoints.
    """
    if exposure:
        rgb = np.clip(rgb * 2.0**exposure, 0.0, 1.0)

    if highlights or shadows or whites or blacks:
        y = luma(rgb)
        y_new = y.copy()
        if highlights:
            y_new += highlights * (y**2 * (1.0 - y)) * 2.0
        if shadows:
            y_new += shadows * ((1.0 - y) ** 2 * y) * 2.0
        if whites:
            y_new += whites * y**3 * 0.5
        if blacks:
            y_new += blacks * (1.0 - y) ** 3 * 0.5
        scale = np.clip(y_new, 0.0, 1.0) / np.maximum(y, 1e-4)
        rgb = np.clip(rgb * scale[..., None], 0.0, 1.0)

    return rgb


def adjust_tone(
    rgb: np.ndarray,
    contrast: float = 1.0,
    brightness: float = 0.0,
    shadow_lift: float = 0.0,
) -> np.ndarray:
    if shadow_lift > 0:
        rgb = rgb + shadow_lift * (1.0 - rgb)
    rgb = (rgb - 0.5) * contrast + 0.5 + brightness
    return np.clip(rgb, 0.0, 1.0)


HSL_BAND_CENTERS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 280.0,
    "magenta": 320.0,
}
_HSL_BAND_WIDTH = 60.0  # degrees to zero influence


def hsl_mixer(rgb: np.ndarray, adjustments: dict[str, dict]) -> np.ndarray:
    """Lightroom-style HSL mixer.

    adjustments: {band: {"hue": degrees, "sat": -1..+1, "lum": -1..+1}}
    Each band's influence falls off smoothly over 60 degrees and is
    weighted by pixel saturation so neutrals stay neutral.
    """
    if not adjustments:
        return rgb

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # float32: H 0-360, S/V 0-1
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    hue_shift = np.zeros_like(hue)
    sat_mult = np.ones_like(sat)
    val_mult = np.ones_like(val)

    for band, adj in adjustments.items():
        center = HSL_BAND_CENTERS[band]
        dist = np.abs(((hue - center + 180.0) % 360.0) - 180.0)
        w = np.clip(1.0 - dist / _HSL_BAND_WIDTH, 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)  # smoothstep
        w *= np.minimum(sat * 2.0, 1.0)  # protect near-neutrals

        hue_shift += float(adj.get("hue", 0.0)) * w
        sat_mult += float(adj.get("sat", 0.0)) * w
        val_mult += float(adj.get("lum", 0.0)) * 0.5 * w

    hsv[..., 0] = (hue + hue_shift) % 360.0
    hsv[..., 1] = np.clip(sat * sat_mult, 0.0, 1.0)
    hsv[..., 2] = np.clip(val * val_mult, 0.0, 1.0)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


GRADE_ZONES = ("shadows", "midtones", "highlights")


def _hue_to_rgb(hue_deg: float) -> np.ndarray:
    px = np.array([[[hue_deg % 360.0, 1.0, 1.0]]], dtype=np.float32)
    return cv2.cvtColor(px, cv2.COLOR_HSV2RGB)[0, 0]


def color_grade(rgb: np.ndarray, grade: dict) -> np.ndarray:
    """Split toning: tint shadows/midtones/highlights independently.

    grade: {"shadows": {"hue": deg, "sat": 0..1}, "midtones": ..., "highlights": ...,
            "balance": -1..+1}  (positive balance widens the highlight zone)
    Tints are luma-neutral offsets so overall brightness holds steady.
    """
    if not grade:
        return rgb

    y = luma(rgb)
    balance = float(grade.get("balance", 0.0))
    yb = np.clip(y + balance * 0.25, 0.0, 1.0)
    weights = {
        "shadows": (1.0 - yb) ** 2,
        "midtones": 2.0 * yb * (1.0 - yb),
        "highlights": yb**2,
    }

    out = rgb
    for zone, w in weights.items():
        adj = grade.get(zone)
        if not adj:
            continue
        strength = float(adj.get("sat", 0.0))
        if strength <= 0:
            continue
        tint = _hue_to_rgb(float(adj.get("hue", 0.0)))
        offset = (tint - tint @ LUMA_WEIGHTS) * strength * 0.35
        out = out + w[..., None] * offset[None, None, :]

    return np.clip(out, 0.0, 1.0)


def adjust_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    if saturation == 1.0:
        return rgb
    y = luma(rgb)[..., None]
    return np.clip(y + saturation * (rgb - y), 0.0, 1.0)


def to_monochrome(rgb: np.ndarray) -> np.ndarray:
    return np.repeat(luma(rgb)[..., None], 3, axis=2)


def vignette(rgb: np.ndarray, amount: float, midpoint: float = 0.5) -> np.ndarray:
    """Radial exposure falloff. amount: -1 (dark corners) .. +1 (bright).

    midpoint: 0..1, how far from centre the falloff starts.
    """
    if not amount:
        return rgb

    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / (w - 1)) * 2.0 - 1.0
    ny = (yy / (h - 1)) * 2.0 - 1.0
    dist = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)

    t = np.clip((dist - midpoint) / max(1.0 - midpoint, 1e-4), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep
    return np.clip(rgb * (1.0 + amount * t)[..., None], 0.0, 1.0)
