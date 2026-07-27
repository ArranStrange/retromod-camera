"""Raspberry Pi Picamera2 backend (Phase 2 placeholder)."""

from typing import Optional

import numpy as np

from sources.base import CameraInput


class Picamera2Input(CameraInput):
    """
    Stub for Phase 2 Pi deployment.

    Replace this implementation with Picamera2 capture when running on hardware:
        from picamera2 import Picamera2
        picam = Picamera2()
        picam.configure(...)
        picam.start()
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "Picamera2 input is only available on Raspberry Pi OS. "
            "Use INPUT_SOURCE='webcam' or 'folder' during macOS development."
        )

    def read(self) -> Optional[np.ndarray]:
        return None

    def release(self) -> None:
        pass
