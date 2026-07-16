"""Convention tests for translate/rotate/resample."""

import numpy as np
import pytest

from lunation.core.warp import resample, rotate, translate

from .synth import lunar_texture


def _blob(size, x, y, sigma=3.0):
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)).astype(
        np.float32
    )


def _centroid(img):
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    s = img.sum()
    return float((img * xx).sum() / s), float((img * yy).sum() / s)


def test_translate_moves_content_positive_right_down():
    img = _blob(128, 40.0, 50.0)
    out = translate(img, 7.0, -5.0)
    cx, cy = _centroid(out)
    assert abs(cx - 47.0) < 0.05
    assert abs(cy - 45.0) < 0.05


@pytest.mark.parametrize("method", ["cv2", "ndimage"])
def test_translate_round_trip(method):
    img = lunar_texture()
    out = translate(translate(img, 3.4, -2.7, method=method), -3.4, 2.7,
                    method=method)
    core = np.s_[32:-32, 32:-32]
    assert np.abs(out[core] - img[core]).mean() < 0.005


def test_rotate_positive_is_visual_ccw():
    """Positive angle rotates content visually CCW about the explicit pivot:
    a feature at screen angle theta (math convention on screen axes,
    theta = atan2(cy - y, x - cx)) moves to theta + rot."""
    size, cx, cy, r = 200, 100.0, 100.0, 60.0
    theta0 = 0.0  # feature due right of pivot
    img = _blob(size, cx + r * np.cos(theta0), cy - r * np.sin(theta0))
    rot = np.pi / 6
    out = rotate(img, rot, cx, cy)
    fx, fy = _centroid(out)
    theta1 = np.arctan2(cy - fy, fx - cx)
    assert abs(theta1 - (theta0 + rot)) < 0.01


def test_rotate_explicit_pivot():
    """Rotation about an off-center pivot keeps the pivot fixed."""
    size = 200
    pivot_blob = _blob(size, 60.0, 140.0, sigma=2.0)
    out = rotate(pivot_blob, np.pi / 4, 60.0, 140.0)
    fx, fy = _centroid(out)
    assert abs(fx - 60.0) < 0.05
    assert abs(fy - 140.0) < 0.05


def test_resample_dimensions_and_energy():
    img = lunar_texture(128)
    up = resample(img, 2.0)
    assert up.shape == (256, 256)
    assert abs(up.mean() - img.mean()) < 0.01
    down = resample(up, 0.5)
    assert down.shape == (128, 128)
    core = np.s_[8:-8, 8:-8]
    assert np.abs(down[core] - img[core]).mean() < 0.01
