"""Chroma-fringe sanitizer: must kill registration fringes on any rig while
passing OTHER imagers' styles (saturated mineral moons) untouched — the
generality contract of 2026-07-16."""

import numpy as np

from lunation.assemble.render import desaturate_fringes

from .synth import lunar_texture


def _color_moon(sat_boost: float, size: int = 512, seed: int = 3):
    """Lunar disk with smooth large-scale tint (mineral-moon style)."""
    base = lunar_texture(size, seed=seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    tint_r = 1.0 + sat_boost * 0.5 * np.sin(2 * np.pi * xx)
    tint_b = 1.0 + sat_boost * 0.5 * np.cos(2 * np.pi * yy)
    rgb = np.stack([base * tint_r, base, base * tint_b], axis=-1)
    return np.clip(rgb, 0, 1).astype(np.float32)


DISK_R = 0.42 * 512  # lunar_texture disk radius


def test_mineral_moon_untouched():
    """Strong smooth saturation (way beyond any of our frames) must pass."""
    moon = _color_moon(sat_boost=0.8)
    out, n = desaturate_fringes(moon.copy(), DISK_R)
    frac = n / moon[..., 0].size
    assert frac < 0.001, f"{100 * frac:.2f}% of a clean mineral moon touched"
    np.testing.assert_allclose(out, moon, atol=0.02)


def test_neutral_moon_untouched():
    moon = np.repeat(lunar_texture(512)[..., None], 3, axis=2)
    out, n = desaturate_fringes(moon.copy(), DISK_R)
    assert n == 0


def test_fringes_removed_neutral_and_mineral():
    """RGB streak fringes must vanish whether the frame is neutral or
    saturated — the discriminator is spatial frequency, not chroma level."""
    for boost in (0.0, 0.8):
        moon = _color_moon(sat_boost=boost)
        fringed = moon.copy()
        # channel-registration-style streaks near the disk edge
        fringed[100:130, 250:256, 0] += 0.5   # red streak
        fringed[130:160, 250:256, 2] += 0.5   # blue streak
        fringed[330:360, 240:246, 1] += 0.5   # green streak
        fringed = np.clip(fringed, 0, 1)
        out, n = desaturate_fringes(fringed, DISK_R)
        assert n > 100, f"boost {boost}: fringes not detected"
        streak = np.s_[100:160, 250:256]
        chroma_out = out[streak].max(axis=-1) - out[streak].min(axis=-1)
        chroma_in = fringed[streak].max(axis=-1) - fringed[streak].min(axis=-1)
        assert chroma_out.mean() < 0.5 * chroma_in.mean(), (
            f"boost {boost}: fringe chroma not reduced")
        # far from the streaks the frame is untouched
        far = np.s_[380:480, 300:460]
        np.testing.assert_allclose(out[far], fringed[far], atol=0.02)
