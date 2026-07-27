"""Photo saving pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config


class ImageSaver:
    """Writes raw and film-sim JPEGs to the output directory."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = Path(output_dir or config.OUTPUT_DIR)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._shot_count = 0

    @property
    def shot_count(self) -> int:
        return self._shot_count

    def save_pair(
        self,
        raw_bgr: np.ndarray,
        processed_bgr: np.ndarray,
        profile_key: str,
    ) -> tuple[Path, Path]:
        """Save raw + processed JPEGs; return both paths."""
        self._shot_count += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{stamp}_{profile_key}_{self._shot_count:04d}"

        processed_path = self._output_dir / f"{base}_film.jpg"
        cv2.imwrite(str(processed_path), processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        raw_path = processed_path
        if config.SAVE_RAW:
            raw_path = self._output_dir / f"{base}_raw.jpg"
            cv2.imwrite(str(raw_path), raw_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return raw_path, processed_path
