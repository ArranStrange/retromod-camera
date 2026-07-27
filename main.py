"""Application entry point for Phase 1 macOS development."""

import argparse
import sys
from typing import NoReturn

import cv2
import numpy as np
import pygame

import config
from film_profiles.editor import ProfileEditor
from film_profiles.loader import load_profiles
from processing.pipeline import FilmSimulator
from sources.factory import create_input
from storage.saver import ImageSaver
from ui.display import DISPLAY_SIZE, RearDisplayUI
from ui.overlay import draw_editor_panel, draw_split


def _inset_ui(preview: np.ndarray, ui_surface: pygame.Surface) -> np.ndarray:
    """Blit the rear-display mock into the preview's bottom-right corner."""
    ui_rgb = pygame.surfarray.array3d(ui_surface).swapaxes(0, 1)
    ui_bgr = cv2.cvtColor(ui_rgb, cv2.COLOR_RGB2BGR)

    h, w = preview.shape[:2]
    inset_x = w - DISPLAY_SIZE - 12
    inset_y = h - DISPLAY_SIZE - 12
    preview[inset_y : inset_y + DISPLAY_SIZE, inset_x : inset_x + DISPLAY_SIZE] = ui_bgr

    cv2.rectangle(
        preview,
        (inset_x - 2, inset_y - 2),
        (inset_x + DISPLAY_SIZE + 1, inset_y + DISPLAY_SIZE + 1),
        (200, 200, 200),
        1,
    )
    return preview


def run(source: str | None = None) -> None:
    """Main capture loop."""
    profiles = load_profiles(config.PROFILES_DIR)
    keys = list(profiles)
    profile = profiles[config.DEFAULT_PROFILE]
    simulator = FilmSimulator(profile)
    saver = ImageSaver()
    ui = RearDisplayUI()
    editor: ProfileEditor | None = None
    split = False

    def switch_profile(index: int) -> None:
        nonlocal profile
        profile = profiles[keys[index % len(keys)]]
        simulator.set_profile(profile)
        print(f"Profile: {profile.name}")

    window_name = "Retromod Camera — Phase 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT)

    source = source or config.INPUT_SOURCE
    print("Retromod Camera — Phase 1 (macOS)")
    print(f"  Input:    {source}")
    print(f"  Profiles: {', '.join(profiles[k].name for k in keys)}")
    print(f"  Output:   {config.OUTPUT_DIR}")
    print()
    print("Controls:")
    print("  SPACE     — capture (save raw + film-sim JPEG)")
    print(f"  1-{len(keys)}       — switch film profile")
    print("  , / .     — cycle profiles")
    print("  B         — toggle original/sim split view")
    print("  E         — edit current profile (P page, [ ] select, - = adjust, S save)")
    print("  Q or ESC  — quit")
    print()

    size = (config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT)
    denoise = source == "webcam" and config.DENOISE_WEBCAM
    last_frame: np.ndarray | None = None
    last_profile = None
    raw_small = processed = None

    with create_input(source) as camera:
        while True:
            frame = camera.read()
            if frame is None:
                print("No frame from input source; exiting.")
                break

            # live view runs the LUT fast path at preview resolution, and
            # only reprocesses when the frame or profile actually changed
            if frame is not last_frame or profile is not last_profile:
                raw_small = cv2.resize(frame, size)
                if denoise:
                    raw_small = cv2.bilateralFilter(raw_small, 5, 40, 40)
                # fixed grain seed: stills grain is frozen, not boiling like
                # cine film — each capture still gets its own unique pattern
                processed = simulator.process(raw_small, grain_seed=0, fast=True)
                last_frame, last_profile = frame, profile

            preview = processed.copy()
            if split:
                preview = draw_split(preview, raw_small)
            if editor is not None:
                preview = draw_editor_panel(preview, editor)
            preview = _inset_ui(preview, ui.render(profile, saver.shot_count))
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord(" "):
                src = cv2.bilateralFilter(frame, 5, 40, 40) if denoise else frame
                full = simulator.process(src)  # precise pipeline at full res
                raw_path, film_path = saver.save_pair(frame, full, profile.key)
                print(f"Saved #{saver.shot_count}: {film_path.name}")
                if config.SAVE_RAW:
                    print(f"           raw: {raw_path.name}")
                if hasattr(camera, "advance"):
                    camera.advance()  # type: ignore[union-attr]

            if key == ord("b"):
                split = not split

            if key == ord("e"):
                if editor is None:
                    editor = ProfileEditor(
                        profile.key, config.PROFILES_DIR / f"{profile.key}.json"
                    )
                    print(f"Editing {profile.name}")
                else:
                    if editor.dirty:
                        print("Exited edit mode; unsaved tweaks discarded")
                    profile = profiles[profile.key]
                    simulator.set_profile(profile)
                    editor = None
            elif editor is not None:
                if key == ord("p"):
                    editor.select_page(+1)
                if key == ord("["):
                    editor.select(-1)
                if key == ord("]"):
                    editor.select(+1)
                if key in (ord("-"), ord("=")):
                    editor.adjust(-1 if key == ord("-") else +1)
                    profile = editor.build()
                    simulator.set_profile(profile)
                if key == ord("s"):
                    editor.save()
                    profile = editor.build()
                    profiles[editor.key] = profile
                    simulator.set_profile(profile)
                    print(f"Saved {editor.path.name}")
            else:
                if ord("1") <= key < ord("1") + len(keys):
                    switch_profile(key - ord("1"))

                if key in (ord(","), ord(".")):
                    step = -1 if key == ord(",") else 1
                    switch_profile(keys.index(profile.key) + step)

    cv2.destroyAllWindows()
    ui.quit()


def main() -> NoReturn:
    parser = argparse.ArgumentParser(description="Retromod camera dev preview")
    parser.add_argument(
        "--source",
        choices=["webcam", "folder", "picamera2"],
        default=None,
        help="override INPUT_SOURCE from config.py",
    )
    args = parser.parse_args()
    try:
        run(args.source)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    sys.exit(0)


if __name__ == "__main__":
    main()
