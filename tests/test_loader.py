"""Profile JSON loader tests."""

import json

import numpy as np
import pytest

import config
from film_profiles.loader import load_profile, load_profiles


def test_shipped_profiles_load():
    profiles = load_profiles(config.PROFILES_DIR)
    assert len(profiles) == 6
    assert config.DEFAULT_PROFILE in profiles


def test_profiles_sorted_by_order_field():
    profiles = load_profiles(config.PROFILES_DIR)
    orders = [p.order for p in profiles.values()]
    assert orders == sorted(orders)


def test_key_comes_from_filename(tmp_path):
    path = tmp_path / "testsim.json"
    path.write_text(
        json.dumps(
            {
                "name": "Test Sim",
                "box_color": [10, 20, 30],
                "color_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        )
    )
    profile = load_profile(path)
    assert profile.key == "testsim"
    assert profile.tone_curve is None
    assert profile.color_matrix.dtype == np.float32


def test_bad_matrix_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(
        json.dumps(
            {
                "name": "Broken",
                "box_color": [0, 0, 0],
                "color_matrix": [[1, 0], [0, 1]],
            }
        )
    )
    with pytest.raises(ValueError):
        load_profile(path)


def test_empty_directory_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profiles(tmp_path)
