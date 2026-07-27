"""Film simulation pipeline tests."""

import numpy as np
import pytest

import config
from film_profiles.loader import load_profiles, profile_from_dict
from processing.grain import film_grain
from processing.pipeline import FilmSimulator


@pytest.fixture(scope="module")
def profiles():
    return load_profiles(config.PROFILES_DIR)


@pytest.fixture(scope="module")
def mono_profile():
    return profile_from_dict(
        "testmono",
        {
            "name": "Test Mono",
            "box_color": [40, 40, 40],
            "color_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "monochrome": True,
            "grain_strength": 0.2,
        },
    )


@pytest.fixture()
def frame():
    return np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)


def test_process_preserves_shape_and_dtype(profiles, frame):
    simulator = FilmSimulator(profiles["standard"])
    out = simulator.process(frame, grain_seed=1)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_process_is_deterministic_with_seed(profiles, frame):
    simulator = FilmSimulator(profiles["standard"])
    a = simulator.process(frame, grain_seed=7)
    b = simulator.process(frame, grain_seed=7)
    np.testing.assert_array_equal(a, b)


def test_mono_profile_output_is_neutral(mono_profile, frame):
    simulator = FilmSimulator(mono_profile)
    out = simulator.process(frame, grain_seed=1)
    np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
    np.testing.assert_array_equal(out[:, :, 1], out[:, :, 2])


def test_every_shipped_profile_processes(profiles, frame):
    simulator = FilmSimulator(profiles["standard"])
    for profile in profiles.values():
        simulator.set_profile(profile)
        out = simulator.process(frame, grain_seed=1)
        assert out.shape == frame.shape


def test_zero_grain_is_noop(frame):
    rgb = frame.astype(np.float32) / 255.0
    np.testing.assert_array_equal(film_grain(rgb, 0.0), rgb)


def test_grain_is_deterministic_with_seed(frame):
    rgb = frame.astype(np.float32) / 255.0
    a = film_grain(rgb, 0.3, seed=3)
    b = film_grain(rgb, 0.3, seed=3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, rgb)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_grain_vanishes_at_pure_black_and_white():
    img = np.zeros((40, 90, 3), dtype=np.float32)
    img[:, 30:60] = 0.4  # lower midtone
    img[:, 60:] = 1.0    # pure white
    out = film_grain(img, 0.4, seed=5)
    diff = np.abs(out - img)
    assert diff[:, :30].max() == 0.0    # pure black: no grains at all
    assert diff[:, 60:].max() == 0.0    # fully covered: no fluctuation
    assert diff[:, 30:60].mean() > 0.01


def test_grain_response_biased_to_shadows():
    lower = np.full((60, 60, 3), 0.35, dtype=np.float32)
    upper = np.full((60, 60, 3), 0.85, dtype=np.float32)
    lower_noise = np.abs(film_grain(lower, 0.3, seed=7) - lower).mean()
    upper_noise = np.abs(film_grain(upper, 0.3, seed=7) - upper).mean()
    assert lower_noise > upper_noise * 1.5


def test_grain_is_luminance_only():
    flat = np.full((60, 60, 3), 0.5, dtype=np.float32)
    out = film_grain(flat, 0.3, seed=9)
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


def test_grain_size_makes_coarser_clumps():
    rgb = np.full((480, 640, 3), 0.5, dtype=np.float32)
    fine = film_grain(rgb, 0.3, size=1.0, seed=3) - rgb
    coarse = film_grain(rgb, 0.3, size=4.0, seed=3) - rgb
    # coarse grain varies less between adjacent pixels
    assert np.abs(np.diff(coarse[:, :, 0], axis=1)).mean() < np.abs(
        np.diff(fine[:, :, 0], axis=1)
    ).mean()


def test_grain_size_scales_with_resolution():
    small = np.full((120, 160, 3), 0.5, dtype=np.float32)
    large = np.full((480, 640, 3), 0.5, dtype=np.float32)
    g_small = film_grain(small, 0.3, size=3.0, seed=3) - small
    g_large = film_grain(large, 0.3, size=3.0, seed=3) - large
    # relative clump size holds: per-pixel variation is lower at high res
    assert np.abs(np.diff(g_large[:, :, 0], axis=1)).mean() < np.abs(
        np.diff(g_small[:, :, 0], axis=1)
    ).mean()
