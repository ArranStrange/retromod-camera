"""Film simulation pipeline tests."""

import numpy as np
import pytest

import config
from film_profiles.loader import load_profiles
from processing.grain import add_grain
from processing.pipeline import FilmSimulator


@pytest.fixture(scope="module")
def profiles():
    return load_profiles(config.PROFILES_DIR)


@pytest.fixture()
def frame():
    return np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)


def test_process_preserves_shape_and_dtype(profiles, frame):
    simulator = FilmSimulator(profiles["standard"])
    out = simulator.process(frame, grain_seed=1)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_process_is_deterministic_with_seed(profiles, frame):
    simulator = FilmSimulator(profiles["vivid"])
    a = simulator.process(frame, grain_seed=7)
    b = simulator.process(frame, grain_seed=7)
    np.testing.assert_array_equal(a, b)


def test_mono_profile_output_is_neutral(profiles, frame):
    simulator = FilmSimulator(profiles["mono"])
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
    np.testing.assert_array_equal(add_grain(rgb, 0.0), rgb)


def test_grain_changes_image(frame):
    rgb = frame.astype(np.float32) / 255.0
    grainy = add_grain(rgb, 0.2, seed=3)
    assert not np.array_equal(grainy, rgb)
    assert grainy.min() >= 0.0 and grainy.max() <= 1.0


def test_grain_size_makes_coarser_noise(frame):
    rgb = np.full((120, 160, 3), 0.5, dtype=np.float32)
    fine = add_grain(rgb, 0.2, size=1.0, seed=3) - rgb
    coarse = add_grain(rgb, 0.2, size=4.0, seed=3) - rgb
    # coarse grain varies less between adjacent pixels
    assert np.abs(np.diff(coarse[:, :, 0], axis=1)).mean() < np.abs(
        np.diff(fine[:, :, 0], axis=1)
    ).mean()


def test_midtone_grain_protects_highlights_and_shadows():
    img = np.zeros((60, 90, 3), dtype=np.float32)
    img[:, :30] = 0.03   # deep shadow
    img[:, 30:60] = 0.5  # midtone
    img[:, 60:] = 0.97   # highlight
    out = add_grain(img, 0.3, midtone=1.0, seed=5)
    diff = np.abs(out - img)
    mid_noise = diff[:, 30:60].mean()
    assert diff[:, :30].mean() < mid_noise * 0.2
    assert diff[:, 60:].mean() < mid_noise * 0.2
