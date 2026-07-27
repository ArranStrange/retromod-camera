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


def test_channel_curve_affects_only_its_channel(rgb):
    lift_red = np.linspace(0, 1, 256).astype(np.float32) ** 0.5
    out = ops.apply_channel_curves(rgb, lut_red=lift_red)
    assert (out[..., 0] >= rgb[..., 0] - 1e-6).all()
    np.testing.assert_array_equal(out[..., 1], rgb[..., 1])
    np.testing.assert_array_equal(out[..., 2], rgb[..., 2])


def test_channel_curves_all_none_is_noop(rgb):
    assert ops.apply_channel_curves(rgb) is rgb


def _solid(r, g, b):
    px = np.zeros((8, 8, 3), dtype=np.float32)
    px[..., 0], px[..., 1], px[..., 2] = r, g, b
    return px


def test_hsl_empty_is_noop(rgb):
    assert ops.hsl_mixer(rgb, {}) is rgb


def test_hsl_band_desaturates_only_its_band():
    red = _solid(0.8, 0.2, 0.2)
    blue = _solid(0.2, 0.2, 0.8)
    adj = {"red": {"sat": -1.0}}
    red_out = ops.hsl_mixer(red, adj)
    blue_out = ops.hsl_mixer(blue, adj)
    # red pixels lose most saturation, blue untouched
    assert red_out.std(axis=2).mean() < red.std(axis=2).mean() * 0.2
    np.testing.assert_allclose(blue_out, blue, atol=1e-3)


def test_hsl_neutrals_protected():
    gray = _solid(0.5, 0.5, 0.5)
    out = ops.hsl_mixer(gray, {"red": {"hue": 30, "sat": 1.0, "lum": 1.0}})
    np.testing.assert_allclose(out, gray, atol=1e-3)


def test_hsl_hue_shift_moves_hue():
    red = _solid(0.8, 0.1, 0.1)
    out = ops.hsl_mixer(red, {"red": {"hue": 40.0}})
    # shifting red towards orange/yellow raises green channel
    assert out[..., 1].mean() > red[..., 1].mean() + 0.05


def test_hsl_lum_brightens_band():
    green = _solid(0.2, 0.7, 0.2)
    out = ops.hsl_mixer(green, {"green": {"lum": 1.0}})
    assert out[..., 1].mean() > green[..., 1].mean()


def test_grade_empty_is_noop(rgb):
    assert ops.color_grade(rgb, {}) is rgb


def test_grade_tints_shadows_not_highlights():
    dark = _solid(0.15, 0.15, 0.15)
    bright = _solid(0.9, 0.9, 0.9)
    grade = {"shadows": {"hue": 220.0, "sat": 0.8}}
    dark_out = ops.color_grade(dark, grade)
    bright_out = ops.color_grade(bright, grade)
    # shadows go blue (B > R), highlights barely move
    assert dark_out[..., 2].mean() > dark_out[..., 0].mean() + 0.02
    np.testing.assert_allclose(bright_out, bright, atol=0.01)


def test_grade_roughly_preserves_luma():
    mid = _solid(0.5, 0.5, 0.5)
    out = ops.color_grade(mid, {"midtones": {"hue": 40.0, "sat": 1.0}})
    assert abs(ops.luma(out).mean() - 0.5) < 0.05


def test_grade_balance_shifts_zones():
    mid = _solid(0.4, 0.4, 0.4)
    grade = {"highlights": {"hue": 40.0, "sat": 1.0}}
    neutral = ops.color_grade(mid, grade)
    pushed = ops.color_grade(mid, {**grade, "balance": 1.0})
    # positive balance treats midtones more like highlights -> stronger tint
    assert abs(pushed - mid).sum() > abs(neutral - mid).sum()


def test_vignette_zero_is_noop(rgb):
    assert ops.vignette(rgb, 0.0) is rgb


def test_vignette_darkens_corners_not_centre():
    flat = np.full((100, 100, 3), 0.6, dtype=np.float32)
    out = ops.vignette(flat, -0.8, midpoint=0.3)
    assert out[0, 0].mean() < 0.5
    assert out[50, 50].mean() == pytest.approx(0.6, abs=1e-3)


def test_vignette_positive_brightens_corners():
    flat = np.full((100, 100, 3), 0.4, dtype=np.float32)
    out = ops.vignette(flat, 0.8, midpoint=0.3)
    assert out[0, 0].mean() > 0.5


def test_filmic_zero_is_noop(rgb):
    assert ops.filmic(rgb, 0.0) is rgb


def test_filmic_preserves_endpoints():
    black = _solid(0.0, 0.0, 0.0)
    white = _solid(1.0, 1.0, 1.0)
    np.testing.assert_allclose(ops.filmic(black, 1.0), black, atol=1e-5)
    np.testing.assert_allclose(ops.filmic(white, 1.0), white, atol=1e-5)


def test_filmic_is_monotone():
    ramp = np.linspace(0, 1, 256, dtype=np.float32).reshape(1, -1, 1).repeat(3, axis=2)
    out = ops.filmic(ramp, 1.0)
    assert np.all(np.diff(out[0, :, 0]) >= -1e-6)


def test_filmic_densifies_shadows_and_rolls_highlights():
    shadow = _solid(0.08, 0.08, 0.08)
    out = ops.filmic(shadow, 1.0)
    assert out.mean() < shadow.mean()  # steep shadow density
    # slope flattens near white (compression knee)
    hi_a, hi_b = _solid(0.85, 0.85, 0.85), _solid(0.95, 0.95, 0.95)
    gap = ops.filmic(hi_b, 1.0).mean() - ops.filmic(hi_a, 1.0).mean()
    assert gap < 0.1


def test_filmic_desaturates_highlights_per_channel():
    warm_bright = _solid(0.95, 0.75, 0.55)
    out = ops.filmic(warm_bright, 1.0)
    spread_in = warm_bright[0, 0, 0] - warm_bright[0, 0, 2]
    spread_out = out[0, 0, 0] - out[0, 0, 2]
    assert spread_out < spread_in  # channels converge near the knee


def test_micro_contrast_zero_is_noop(rgb):
    assert ops.micro_contrast(rgb, 0.0) is rgb


def _checker(lo, hi):
    tile = np.indices((64, 64)).sum(axis=0) % 2
    img = np.where(tile[..., None] == 1, hi, lo).astype(np.float32)
    return np.repeat(img, 1, axis=2) if img.shape[2] == 3 else np.dstack([img] * 3)


def test_micro_contrast_boosts_midtone_detail():
    img = _checker(0.45, 0.55)
    out = ops.micro_contrast(img, 0.5)
    assert out.std() > img.std()


def test_micro_contrast_leaves_shadows_alone():
    img = _checker(0.02, 0.06)
    out = ops.micro_contrast(img, 0.5)
    assert abs(out.std() - img.std()) < 0.005


def test_micro_contrast_flat_image_unchanged():
    flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
    np.testing.assert_allclose(ops.micro_contrast(flat, 0.5), flat, atol=1e-5)


def test_negative_micro_contrast_softens():
    img = _checker(0.45, 0.55)
    out = ops.micro_contrast(img, -0.5)
    assert out.std() < img.std()


def test_chroma_smudge_zero_is_noop(rgb):
    assert ops.chroma_smudge(rgb, 0.0) is rgb


def test_chroma_smudge_blends_colour_but_keeps_luma_edges():
    # two half-fields with different colour but near-equal luminance
    img = np.zeros((400, 400, 3), dtype=np.float32)
    img[:, :200] = [0.6, 0.35, 0.5]
    img[:, 200:] = [0.3, 0.45, 0.6]
    out = ops.chroma_smudge(img, 1.0)
    mid_in = np.abs(img[:, 199] - img[:, 201]).mean()
    mid_out = np.abs(out[:, 199] - out[:, 201]).mean()
    assert mid_out < mid_in * 0.7  # colour boundary softened
    y_in, y_out = ops.luma(img), ops.luma(out)
    assert np.abs(y_in - y_out).max() < 0.05  # luminance detail held


def test_chroma_smudge_flat_image_unchanged():
    flat = np.full((60, 60, 3), [0.6, 0.4, 0.3], dtype=np.float32)
    np.testing.assert_allclose(ops.chroma_smudge(flat, 0.6), flat, atol=0.01)
