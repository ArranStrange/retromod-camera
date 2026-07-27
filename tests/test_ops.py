"""Float-space operation tests."""

import numpy as np
import pytest

from processing import ops


@pytest.fixture()
def rgb():
    return np.random.default_rng(0).random((60, 80, 3)).astype(np.float32)


def test_roundtrip_u8_conversion():
    bgr = np.random.default_rng(1).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    np.testing.assert_array_equal(ops.to_bgr_u8(ops.to_float_rgb(bgr)), bgr)


def test_exposure_one_stop_doubles(rgb):
    dark = rgb * 0.3
    out = ops.tone_panel(dark, exposure=1.0)
    np.testing.assert_allclose(out, dark * 2.0, atol=1e-5)


def test_highlights_leave_deep_shadows_alone():
    shadow = np.full((8, 8, 3), 0.05, dtype=np.float32)
    out = ops.tone_panel(shadow, highlights=-1.0)
    np.testing.assert_allclose(out, shadow, atol=0.01)


def test_shadows_leave_highlights_alone():
    bright = np.full((8, 8, 3), 0.95, dtype=np.float32)
    out = ops.tone_panel(bright, shadows=1.0)
    np.testing.assert_allclose(out, bright, atol=0.01)


def test_negative_highlights_darken_highlights():
    bright = np.full((8, 8, 3), 0.7, dtype=np.float32)
    out = ops.tone_panel(bright, highlights=-1.0)
    assert out.mean() < bright.mean()


def test_positive_shadows_lift_shadows():
    dark = np.full((8, 8, 3), 0.25, dtype=np.float32)
    out = ops.tone_panel(dark, shadows=1.0)
    assert out.mean() > dark.mean()


def test_tone_panel_defaults_are_noop(rgb):
    np.testing.assert_array_equal(ops.tone_panel(rgb), rgb)


def test_tone_panel_stays_in_range(rgb):
    out = ops.tone_panel(rgb, exposure=2.0, highlights=1.0, shadows=1.0, whites=1.0, blacks=-1.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
