"""Abstract camera input interface."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class CameraInput(ABC):
    """Common interface for all frame sources (webcam, files, Picamera2)."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if unavailable."""

    @abstractmethod
    def release(self) -> None:
        """Release underlying resources."""

    def __enter__(self) -> "CameraInput":
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
