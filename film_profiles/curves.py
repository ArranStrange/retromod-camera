"""Tone curve LUT construction from editable control points."""

from __future__ import annotations

import numpy as np

LUT_SIZE = 256


def build_tone_lut(points: list[list[float]]) -> np.ndarray:
    """
    Build a 256-entry LUT from (x, y) control points in 0-1 space.

    Uses Fritsch-Carlson monotone cubic interpolation so curves stay
    smooth without overshooting between points.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("tone_curve needs at least two [x, y] control points")

    xs, ys = pts[:, 0], pts[:, 1]
    if np.any(np.diff(xs) <= 0):
        raise ValueError("tone_curve x values must be strictly increasing")
    if xs[0] != 0.0 or xs[-1] != 1.0:
        raise ValueError("tone_curve must span x=0 to x=1")

    n = len(xs)
    h = np.diff(xs)
    m = np.diff(ys) / h  # secant slopes

    tangents = np.empty(n, dtype=np.float32)
    tangents[0], tangents[-1] = m[0], m[-1]
    tangents[1:-1] = (m[:-1] + m[1:]) / 2.0

    # Fritsch-Carlson: limit tangents to preserve monotonicity
    for i in range(n - 1):
        if m[i] == 0.0:
            tangents[i] = tangents[i + 1] = 0.0
        else:
            a = tangents[i] / m[i]
            b = tangents[i + 1] / m[i]
            s = a * a + b * b
            if s > 9.0:
                tau = 3.0 / np.sqrt(s)
                tangents[i] = tau * a * m[i]
                tangents[i + 1] = tau * b * m[i]

    x = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
    idx = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, n - 2)
    dx = h[idx]
    t = (x - xs[idx]) / dx

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    y = h00 * ys[idx] + h10 * dx * tangents[idx] + h01 * ys[idx + 1] + h11 * dx * tangents[idx + 1]
    return np.clip(y, 0.0, 1.0).astype(np.float32)
