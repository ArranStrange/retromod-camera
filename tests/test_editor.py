"""Profile editor state tests."""

import json

import pytest

from film_profiles.editor import PAGES, ProfileEditor


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


def _goto(editor, page_name, param_label):
    while editor.page_name != page_name:
        editor.select_page(+1)
    while editor.param.label != param_label:
        editor.select(+1)


def test_adjust_changes_value_and_marks_dirty(editor):
    _goto(editor, "TONE", "contrast")
    editor.adjust(+5)
    assert editor.value(editor.param) == pytest.approx(1.1)
    assert editor.dirty


def test_adjust_clamps_to_bounds(editor):
    _goto(editor, "TONE", "contrast")
    editor.adjust(-1000)
    assert editor.value(editor.param) == editor.param.lo
    editor.adjust(+10000)
    assert editor.value(editor.param) == editor.param.hi


def test_select_wraps_within_page(editor):
    editor.select(-1)
    assert editor.param_index == len(editor.params) - 1
    editor.select(+1)
    assert editor.param_index == 0


def test_page_switch_wraps_and_resets_param(editor):
    editor.select(+2)
    editor.select_page(-1)
    assert editor.page_name == PAGES[-1][0]
    assert editor.param_index == 0


def test_nested_grade_adjust_creates_structure(editor):
    _goto(editor, "GRADE", "sh hue")
    editor.adjust(+44)  # 220 degrees
    _goto(editor, "GRADE", "sh sat")
    editor.adjust(+10)  # 0.2
    assert editor.data["grade"]["shadows"] == {"hue": 220.0, "sat": 0.2}
    profile = editor.build()
    assert profile.grade["shadows"]["sat"] == pytest.approx(0.2)


def test_hsl_adjust_builds_profile(editor):
    _goto(editor, "HSL SAT", "red")
    editor.adjust(-10)  # -0.2
    profile = editor.build()
    assert profile.hsl["red"]["sat"] == pytest.approx(-0.2)


def test_every_param_min_max_still_builds(editor):
    for _ in range(len(PAGES)):
        for _ in range(len(editor.params)):
            editor.adjust(-100000)
            editor.build()
            editor.adjust(+200000)
            editor.build()
            editor.select(+1)
        editor.select_page(+1)


def test_save_writes_and_clears_dirty(editor):
    _goto(editor, "TONE", "contrast")
    editor.adjust(+1)
    editor.save()
    assert not editor.dirty
    on_disk = json.loads(editor.path.read_text())
    assert on_disk["contrast"] == pytest.approx(1.02)
