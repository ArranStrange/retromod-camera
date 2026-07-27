"""3D colour LUT bake/apply tests."""

import dataclasses

import numpy as np
import pytest

import config
from film_profiles.loader import load_profiles, profile_from_dict
from processing import ops
from processing.pipeline import FilmSimulator, apply_color_stages, bake_color_lut


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
            "contrast": 1.2,
        },
    )


@pytest.fixture()
def rgb():
    return np.random.default_rng(0).random((80, 120, 3)).astype(np.float32)


def test_identity_lut_is_identity(rgb):
    n = 17
    axis = np.linspace(0, 1, n, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    identity = np.stack([r, g, b], axis=-1)
    np.testing.assert_allclose(ops.apply_color_lut(rgb, identity), rgb, atol=1e-5)


def test_lut_matches_precise_colour_stages(profiles, rgb):
    for profile in profiles.values():
        precise = apply_color_stages(rgb, profile)
        lut = bake_color_lut(profile)
        fast = ops.apply_color_lut(rgb, lut)
        assert np.abs(fast - precise).max() < 0.02, profile.key


def test_lut_matches_precise_for_mono(mono_profile, rgb):
    precise = apply_color_stages(rgb, mono_profile)
    fast = ops.apply_color_lut(rgb, bake_color_lut(mono_profile))
    assert np.abs(fast - precise).max() < 0.02


def test_fast_process_matches_precise(profiles):
    # grain is deliberately attenuated in the fast path, so compare without it
    frame = np.random.default_rng(1).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    grainless = dataclasses.replace(profiles["standard"], grain_strength=0.0)
    sim = FilmSimulator(grainless)
    precise = sim.process(frame, grain_seed=2).astype(np.int16)
    fast = sim.process(frame, grain_seed=2, fast=True).astype(np.int16)
    assert np.abs(fast - precise).max() <= 8  # within ~3% of 8-bit range


def test_lut_invalidated_on_profile_change(profiles, mono_profile):
    frame = np.full((20, 20, 3), 128, dtype=np.uint8)
    sim = FilmSimulator(profiles["standard"])
    a = sim.process(frame, grain_seed=1, fast=True)
    sim.set_profile(mono_profile)
    b = sim.process(frame, grain_seed=1, fast=True)
    assert not np.array_equal(a, b)
    assert (b[..., 0] == b[..., 1]).all()  # mono LUT took effect
