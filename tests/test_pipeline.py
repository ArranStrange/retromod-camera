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
    np.testing.assert_array_equal(add_grain(frame, 0.0), frame)


def test_grain_changes_image(frame):
    grainy = add_grain(frame, 0.2, seed=3)
    assert not np.array_equal(grainy, frame)
