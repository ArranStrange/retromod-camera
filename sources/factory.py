"""Factory for creating camera input backends."""

from __future__ import annotations

from pathlib import Path

import config
from sources.base import CameraInput
from sources.file_folder import FileFolderInput
from sources.picamera import Picamera2Input
from sources.webcam import WebcamInput


def create_input(source: str | None = None, **kwargs: object) -> CameraInput:
    """Instantiate the configured input backend."""
    source = (source or config.INPUT_SOURCE).lower()

    if source == "webcam":
        index = kwargs.get("index", config.WEBCAM_INDEX)
        return WebcamInput(int(index))  # type: ignore[arg-type]

    if source == "folder":
        folder = kwargs.get("folder", config.SAMPLE_FOLDER)
        return FileFolderInput(Path(folder))  # type: ignore[arg-type]

    if source == "picamera2":
        return Picamera2Input()

    raise ValueError(f"Unknown input source: {source!r}. Use webcam, folder, or picamera2.")
