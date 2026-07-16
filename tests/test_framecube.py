"""FrameCube kappa-sigma rejection semantics (ser-stack.js:596-693)."""

import numpy as np
import pytest

from lunation.core.framecube import SENTINEL, FrameCube, encode_plane


def _cube(tmp_path, planes):
    fc = FrameCube(str(tmp_path / "cube.bin"),
                   width=planes[0].shape[1], height=planes[0].shape[0])
    for i, p in enumerate(planes):
        fc.write_plane(i, p)
    return fc


@pytest.mark.parametrize("engine", ["ported", "astropy"])
def test_planted_outliers_removed(tmp_path, engine):
    """20 well-behaved samples + 2 gross outliers at one pixel: the clipped
    mean must sit at the clean value, and the outliers must not survive."""
    rng = np.random.default_rng(3)
    planes = [np.full((4, 4), 0.5, dtype=np.float32)
              + rng.normal(0, 0.01, (4, 4)).astype(np.float32)
              for _ in range(22)]
    planes[5][2, 2] = 0.95   # hot
    planes[11][2, 2] = 0.02  # cold
    fc = _cube(tmp_path, planes)
    mean, count = fc.combine(kappa=2.5, iters=2, engine=engine)
    assert abs(mean[2, 2] - 0.5) < 0.02
    assert count[2, 2] == 20
    assert count[0, 0] >= 20  # clean pixels keep (almost) everything
    fc.remove()


@pytest.mark.parametrize("engine", ["ported", "astropy"])
def test_sentinel_excluded(tmp_path, engine):
    """Sentinel samples (outside a frame's coverage) never contribute."""
    a = np.full((2, 2), 0.25, dtype=np.float32)
    b = np.full((2, 2), 0.75, dtype=np.float32)
    b_masked = b.copy()
    b_masked[0, 0] = -1.0  # coverage fill -> sentinel (ser-stack.js:497)
    fc = _cube(tmp_path, [a, b_masked, a, b_masked, a, b_masked])
    mean, count = fc.combine(kappa=10.0, iters=1, engine=engine)
    assert count[0, 0] == 3
    assert abs(mean[0, 0] - 0.25) < 1e-3
    assert count[1, 1] == 6
    assert abs(mean[1, 1] - 0.5) < 1e-3
    fc.remove()


def test_low_count_pixels_not_clipped(tmp_path):
    """<5 samples -> bounds (-1, 2), nothing rejected (ser-stack.js:630-635)."""
    planes = [np.array([[0.1]], dtype=np.float32),
              np.array([[0.9]], dtype=np.float32),
              np.array([[0.5]], dtype=np.float32)]
    fc = _cube(tmp_path, planes)
    mean, count = fc.combine(kappa=0.1, iters=3, engine="ported")
    assert count[0, 0] == 3  # aggressive kappa, but n<5 guard holds
    assert abs(mean[0, 0] - 0.5) < 1e-3
    fc.remove()


@pytest.mark.parametrize("engine", ["ported", "astropy"])
def test_never_lose_every_sample(tmp_path, engine):
    """A pixel whose samples would all reject keeps its prior stats
    (ser-stack.js:666-673): bimodal 50/50 far apart, tiny kappa."""
    planes = ([np.full((2, 2), 0.1, dtype=np.float32)] * 5
              + [np.full((2, 2), 0.9, dtype=np.float32)] * 5)
    fc = _cube(tmp_path, planes)
    mean, count = fc.combine(kappa=0.05, iters=4, engine=engine)
    assert count[0, 0] == 10          # everything restored, nothing lost
    assert abs(mean[0, 0] - 0.5) < 1e-3
    fc.remove()


def test_encode_plane_quantization():
    p = np.array([[0.0, 1.0, 0.5, -1.0, 2.0]], dtype=np.float32)
    u = encode_plane(p)
    assert u[0, 0] == 0
    assert u[0, 1] == 65534
    assert u[0, 2] == 32767
    assert u[0, 3] == SENTINEL   # coverage fill
    assert u[0, 4] == 65534      # clamped high
