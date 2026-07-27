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


def test_warmth_and_tint_bake_into_matrix():
    from film_profiles.loader import profile_from_dict

    data = {
        "name": "Warm",
        "box_color": [0, 0, 0],
        "color_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "warmth": 0.1,
        "tint": 0.05,
    }
    profile = profile_from_dict("warm", data)
    assert profile.color_matrix[0, 0] == pytest.approx(1.1)
    assert profile.color_matrix[2, 2] == pytest.approx(0.9)
    assert profile.color_matrix[1, 1] == pytest.approx(1.05)


def test_save_roundtrip(tmp_path):
    from film_profiles.loader import save_profile_data

    data = {
        "name": "Round Trip",
        "box_color": [1, 2, 3],
        "color_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "contrast": 1.23,
        "tone_curve": [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]],
    }
    path = tmp_path / "roundtrip.json"
    save_profile_data(data, path)
    assert json.loads(path.read_text()) == data
    reloaded = load_profile(path)
    assert reloaded.contrast == pytest.approx(1.23)


def test_save_rejects_invalid_data(tmp_path):
    from film_profiles.loader import save_profile_data

    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        save_profile_data(
            {"name": "Bad", "box_color": [0, 0, 0], "color_matrix": [[1, 0], [0, 1]]},
            path,
        )
    assert not path.exists()
