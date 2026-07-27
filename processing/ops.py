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


def adjust_saturation(rgb: np.ndarray, saturation: float) -> np.ndarray:
    if saturation == 1.0:
        return rgb
    y = luma(rgb)[..., None]
    return np.clip(y + saturation * (rgb - y), 0.0, 1.0)


def to_monochrome(rgb: np.ndarray) -> np.ndarray:
    return np.repeat(luma(rgb)[..., None], 3, axis=2)
