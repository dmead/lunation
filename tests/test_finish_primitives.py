"""Finish-chain primitive contracts."""

import numpy as np
import pytest

from lunation.finish.primitives import (curve, histogram_transform,
                                        lab01_to_rgb, mtf, mtf_for,
                                        rgb_to_lab01, rl_deconvolve, starlet,
                                        starlet_sharpen)

from .synth import lunar_texture


def test_mtf_closed_form():
    assert mtf(0.5, 0.3) == pytest.approx(0.3)  # identity at m=0.5
    m = mtf_for(0.2, 0.42)
    assert mtf(m, 0.2) == pytest.approx(0.42, abs=1e-6)


def test_histogram_transform_anchors():
    img = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float32)
    out = histogram_transform(img, shadow=0.1, mid=0.5)
    assert out[1] == pytest.approx(0.0, abs=1e-6)
    assert out[3] == pytest.approx(1.0, abs=1e-6)


def test_curve_monotone_no_overshoot():
    pts = [[0, 0], [0.25, 0.23], [0.75, 0.77], [1, 1]]
    x = np.linspace(0, 1, 101).astype(np.float32)
    y = curve(x, pts)
    assert (np.diff(y) >= -1e-6).all()  # monotone
    assert y.min() >= 0 and y.max() <= 1


def test_starlet_reconstruction_identity():
    img = lunar_texture(128)
    layers = starlet(img, 4)
    rec = np.sum(layers, axis=0)
    np.testing.assert_allclose(rec, img, atol=1e-5)


def test_starlet_sharpen_zero_bias_identity():
    img = lunar_texture(128)
    out = starlet_sharpen(img, [0.0, 0.0, 0.0, 0.0], deringing=False)
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_starlet_sharpen_increases_detail():
    img = lunar_texture(128)
    out = starlet_sharpen(img, [0.0, 0.3, 0.3, 0.1])
    from lunation.core.kernels import laplacian_sharpness

    assert laplacian_sharpness(out) > laplacian_sharpness(img)


def test_rl_deconvolve_recovers_blur():
    """RL must sharpen a gaussian-blurred moon back toward the truth."""
    import cv2

    truth = lunar_texture(192)
    blurred = cv2.GaussianBlur(truth, (0, 0), 1.5)
    restored = rl_deconvolve(blurred, psf_sigma=1.5, iterations=30)
    e_blur = float(np.abs(blurred - truth).mean())
    e_rest = float(np.abs(restored - truth).mean())
    assert e_rest < 0.75 * e_blur, f"{e_rest} !< 0.75*{e_blur}"
    # no wild ringing against the black sky
    assert float(restored.min()) >= 0
    assert float(restored.max()) <= 1


def test_lab_round_trip():
    rng = np.random.default_rng(2)
    rgb = rng.uniform(0.05, 0.95, (32, 32, 3)).astype(np.float32)
    L, a, b = rgb_to_lab01(rgb)
    back = lab01_to_rgb(L, a, b)
    np.testing.assert_allclose(back, rgb, atol=2e-3)
    # neutral gray has centered chroma
    g = np.full((4, 4, 3), 0.4, np.float32)
    _, ga, gb = rgb_to_lab01(g)
    np.testing.assert_allclose(ga, 0.5, atol=1e-3)
    np.testing.assert_allclose(gb, 0.5, atol=1e-3)
