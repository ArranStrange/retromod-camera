"""Tone curve LUT builder tests."""

import numpy as np
import pytest

from film_profiles.curves import LUT_SIZE, build_tone_lut


def test_linear_points_give_identity():
    lut = build_tone_lut([[0.0, 0.0], [1.0, 1.0]])
    assert lut.shape == (LUT_SIZE,)
    np.testing.assert_allclose(lut, np.linspace(0, 1, LUT_SIZE), atol=1e-5)


def test_curve_passes_through_control_points():
    points = [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]]
    lut = build_tone_lut(points)
    assert lut[0] == pytest.approx(0.0, abs=1e-5)
    assert lut[127] == pytest.approx(0.6, abs=0.01)
    assert lut[255] == pytest.approx(1.0, abs=1e-5)


def test_monotone_input_stays_monotone():
    points = [[0.0, 0.0], [0.2, 0.16], [0.5, 0.52], [0.8, 0.86], [1.0, 1.0]]
    lut = build_tone_lut(points)
    assert np.all(np.diff(lut) >= -1e-6)


def test_output_clipped_to_unit_range():
    lut = build_tone_lut([[0.0, 0.02], [0.5, 0.5], [1.0, 0.98]])
    assert lut.min() >= 0.0
    assert lut.max() <= 1.0


def test_rejects_single_point():
    with pytest.raises(ValueError):
        build_tone_lut([[0.0, 0.0]])


def test_rejects_unsorted_x():
    with pytest.raises(ValueError):
        build_tone_lut([[0.0, 0.0], [0.7, 0.5], [0.3, 0.4], [1.0, 1.0]])


def test_rejects_partial_span():
    with pytest.raises(ValueError):
        build_tone_lut([[0.1, 0.0], [1.0, 1.0]])
