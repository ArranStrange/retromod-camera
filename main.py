"""Application entry point for Phase 1 macOS development."""

import sys
from typing import NoReturn

import cv2
import numpy as np
import pygame

import config
from film_profiles.registry import PROFILES, get_profile
from sources.factory import create_input
from processing.pipeline import FilmSimulator
from storage.saver import ImageSaver
from ui.display import DISPLAY_SIZE, RearDisplayUI

PROFILE_KEYS = list(PROFILES.keys())


def _profile_from_key_index(index: int) -> str:
    return PROFILE_KEYS[index % len(PROFILE_KEYS)]


def _draw_preview(
    frame_bgr: np.ndarray,
    ui_surface: pygame.Surface,
    preview_w: int,
    preview_h: int,
) -> np.ndarray:
    """Compose live preview with inset rear-display mock."""
    preview = cv2.resize(frame_bgr, (preview_w, preview_h))

    ui_rgb = pygame.surfarray.array3d(ui_surface).swapaxes(0, 1)
    ui_bgr = cv2.cvtColor(ui_rgb, cv2.COLOR_RGB2BGR)

    inset_x = preview_w - DISPLAY_SIZE - 12
    inset_y = preview_h - DISPLAY_SIZE - 12
    preview[inset_y : inset_y + DISPLAY_SIZE, inset_x : inset_x + DISPLAY_SIZE] = ui_bgr

    border = cv2.rectangle(
        preview.copy(),
        (inset_x - 2, inset_y - 2),
        (inset_x + DISPLAY_SIZE + 1, inset_y + DISPLAY_SIZE + 1),
        (200, 200, 200),
        1,
    )
    return border


def run() -> None:
    """Main capture loop."""
    profile_key = config.DEFAULT_PROFILE
    profile = get_profile(profile_key)
    simulator = FilmSimulator(profile)
    saver = ImageSaver()
    ui = RearDisplayUI()

    window_name = "Retromod Camera — Phase 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT)

    print("Retromod Camera — Phase 1 (macOS)")
    print(f"  Input:   {config.INPUT_SOURCE}")
    print(f"  Profile: {profile.name}")
    print(f"  Output:  {config.OUTPUT_DIR}")
    print()
    print("Controls:")
    print("  SPACE     — capture (save raw + film-sim JPEG)")
    print("  1-6       — switch film profile")
    print("  , / .     — cycle profiles")
    print("  Q or ESC  — quit")
    print()

    with create_input() as camera:
        while True:
            frame = camera.read()
            if frame is None:
                print("No frame from input source; exiting.")
                break

            processed = simulator.process(frame)
            ui_surface = ui.render(profile, saver.shot_count)
            preview = _draw_preview(
                processed,
                ui_surface,
                config.PREVIEW_WIDTH,
                config.PREVIEW_HEIGHT,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord(" "):
                raw_path, film_path = saver.save_pair(frame, processed, profile.key)
                print(f"Saved #{saver.shot_count}: {film_path.name}")
                if config.SAVE_RAW:
                    print(f"           raw: {raw_path.name}")
                if hasattr(camera, "advance"):
                    camera.advance()  # type: ignore[union-attr]

            if ord("1") <= key <= ord("6"):
                profile_key = _profile_from_key_index(key - ord("1"))
                profile = get_profile(profile_key)
                simulator.set_profile(profile)
                print(f"Profile: {profile.name}")

            if key == ord(","):
                idx = (PROFILE_KEYS.index(profile.key) - 1) % len(PROFILE_KEYS)
                profile = get_profile(PROFILE_KEYS[idx])
                simulator.set_profile(profile)
                print(f"Profile: {profile.name}")

            if key == ord("."):
                idx = (PROFILE_KEYS.index(profile.key) + 1) % len(PROFILE_KEYS)
                profile = get_profile(PROFILE_KEYS[idx])
                simulator.set_profile(profile)
                print(f"Profile: {profile.name}")

    cv2.destroyAllWindows()
    ui.quit()


def main() -> NoReturn:
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    sys.exit(0)


if __name__ == "__main__":
    main()
