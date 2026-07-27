"""Per-channel and matrix color transforms."""

import cv2
import numpy as np


def apply_color_matrix(image_bgr: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 RGB color matrix to a BGR uint8 image."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    transformed = rgb @ matrix.T
    transformed = np.clip(transformed, 0.0, 1.0)
    bgr = cv2.cvtColor((transformed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


def apply_tone_curve(image_bgr: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Map luminance through a 256-entry LUT while preserving chroma ratio."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0] / 255.0
    mapped_l = curve[(l_channel * 255).astype(np.uint8)].astype(np.float32)
    lab[:, :, 0] = mapped_l * 255.0
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def adjust_contrast_saturation(
    image_bgr: np.ndarray,
    contrast: float,
    saturation: float,
    brightness: float = 0.0,
    shadow_lift: float = 0.0,
) -> np.ndarray:
    """Global contrast/saturation with optional shadow lift."""
    image = image_bgr.astype(np.float32) / 255.0

    if shadow_lift > 0:
        image = image + shadow_lift * (1.0 - image)

    image = (image - 0.5) * contrast + 0.5 + brightness
    image = np.clip(image, 0.0, 1.0)

    if saturation != 1.0:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        image = gray_bgr + saturation * (image - gray_bgr)

    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def to_monochrome(image_bgr: np.ndarray) -> np.ndarray:
    """Convert to single-channel look rendered as BGR."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
