"""Static image folder input for offline pipeline testing."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from input.base import CameraInput

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FileFolderInput(CameraInput):
    """Holds one sample image at a time; advances to the next on capture."""

    def __init__(self, folder: Path) -> None:
        self._folder = Path(folder)
        if not self._folder.is_dir():
            raise FileNotFoundError(f"Sample folder not found: {self._folder}")

        self._paths = sorted(
            p for p in self._folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self._paths:
            raise FileNotFoundError(f"No images found in {self._folder}")

        self._index = 0
        self._frame: Optional[np.ndarray] = None
        self._load_current()

    def _load_current(self) -> None:
        path = self._paths[self._index]
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {path}")
        self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        return self._frame

    def advance(self) -> None:
        """Move to the next sample image (call after each capture)."""
        self._index = (self._index + 1) % len(self._paths)
        self._load_current()

    def release(self) -> None:
        pass
