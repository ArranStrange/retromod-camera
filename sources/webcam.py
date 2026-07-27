"""OpenCV webcam capture for macOS development."""

from typing import Optional

import cv2
import numpy as np

from sources.base import CameraInput


class WebcamInput(CameraInput):
    """Live frames from cv2.VideoCapture."""

    def __init__(self, device_index: int = 0) -> None:
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {device_index}. "
                "Check camera permissions in System Settings > Privacy."
            )

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._capture.read()
        return frame if ok else None

    def release(self) -> None:
        self._capture.release()
