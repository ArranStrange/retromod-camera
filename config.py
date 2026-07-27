"""Application configuration for Phase 1 (macOS development)."""

from pathlib import Path

# Input source: "webcam", "folder", or "picamera2"
INPUT_SOURCE = "webcam"
WEBCAM_INDEX = 0
SAMPLE_FOLDER = Path(__file__).parent / "samples"

# Output
OUTPUT_DIR = Path(__file__).parent / "output"
SAVE_RAW = True

# Preview window size (live view; rear display mock stays 240x240)
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

# Default film profile key (see film_profiles/registry.py)
DEFAULT_PROFILE = "kodachrome64"
