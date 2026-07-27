"""Profile editor state tests."""

import json

import pytest

from film_profiles.editor import PARAMS, ProfileEditor


@pytest.fixture()
def editor(tmp_path):
    path = tmp_path / "testsim.json"
    path.write_text(
        json.dumps(
            {
                "name": "Test Sim",
                "box_color": [10, 20, 30],
                "color_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "contrast": 1.0,
            }
        )
    )
    return ProfileEditor("testsim", path)


def test_adjust_changes_value_and_marks_dirty(editor):
    assert not editor.dirty
    editor.adjust(+5)
    assert editor.value(editor.param) == pytest.approx(1.1)
    assert editor.dirty


def test_adjust_clamps_to_bounds(editor):
    editor.adjust(-1000)
    assert editor.value(editor.param) == editor.param.lo
    editor.adjust(+10000)
    assert editor.value(editor.param) == editor.param.hi


def test_select_wraps_both_ways(editor):
    editor.select(-1)
    assert editor.param_index == len(PARAMS) - 1
    editor.select(+1)
    assert editor.param_index == 0


def test_unset_param_starts_from_default(editor):
    editor.select(+4)  # warmth, absent from JSON
    assert editor.param.key == "warmth"
    editor.adjust(+2)
    assert editor.value(editor.param) == pytest.approx(0.01)


def test_edit_applies_to_built_profile(editor):
    editor.select(+4)  # warmth
    editor.adjust(+20)  # +0.1
    profile = editor.build()
    assert profile.color_matrix[0, 0] == pytest.approx(1.1)


def test_save_writes_and_clears_dirty(editor):
    editor.adjust(+1)
    editor.save()
    assert not editor.dirty
    on_disk = json.loads(editor.path.read_text())
    assert on_disk["contrast"] == pytest.approx(1.02)
