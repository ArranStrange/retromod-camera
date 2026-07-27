"""OpenCV overlay drawing for the tuning studio."""

from __future__ import annotations

import cv2
import numpy as np

from film_profiles.editor import PAGES, ProfileEditor

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_YELLOW = (60, 220, 255)  # BGR
_GRAY = (170, 170, 170)


def draw_split(sim_preview: np.ndarray, raw_preview: np.ndarray) -> np.ndarray:
    """Compose original (left) vs simulated (right) with a divider."""
    out = sim_preview.copy()
    half = out.shape[1] // 2
    out[:, :half] = raw_preview[:, :half]
    h = out.shape[0]
    cv2.line(out, (half, 0), (half, h), _WHITE, 1)
    cv2.putText(out, "ORIGINAL", (8, h - 12), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(out, "SIM", (half + 8, h - 12), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)
    return out


def draw_editor_panel(preview: np.ndarray, editor: ProfileEditor) -> np.ndarray:
    """Darkened parameter panel down the left edge."""
    out = preview.copy()
    row_h, top, panel_w = 20, 66, 190
    panel_h = top + len(editor.params) * row_h + 26

    region = out[:panel_h, :panel_w]
    region[:] = (region * 0.25).astype(np.uint8)

    title = f"EDIT {editor.key}" + ("  *" if editor.dirty else "")
    cv2.putText(out, title, (10, 24), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)
    page = f"{editor.page_name}  ({editor.page_index + 1}/{len(PAGES)})"
    cv2.putText(out, page, (10, 44), _FONT, 0.42, _YELLOW, 1, cv2.LINE_AA)

    for i, spec in enumerate(editor.params):
        y = top + i * row_h
        selected = i == editor.param_index
        color = _YELLOW if selected else _GRAY
        marker = "> " if selected else "  "
        text = f"{marker}{spec.label:<12}{editor.value(spec):+.3f}"
        cv2.putText(out, text, (10, y), _FONT, 0.42, color, 1, cv2.LINE_AA)

    hint_y = top + len(editor.params) * row_h + 8
    cv2.putText(out, "P page  [ ] param  - = adjust", (10, hint_y), _FONT, 0.36, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(out, "S save  E exit", (10, hint_y + 14), _FONT, 0.36, _GRAY, 1, cv2.LINE_AA)
    return out
