"""prep-finished normalization: geometry recovery, truncation refusal,
paint-fill removal — all on synthetic finished images."""

import numpy as np
import pytest

from lunation.assemble.prep import TruncatedSource, prep_image
from lunation.io.images import write_png
from lunation.io.xisf_io import read_xisf

from .synth import lunar_texture

CANVAS = 760   # scaled-down canvas for speed
TARGET_R = 323  # keeps the 979/2300 working ratio


def _finished(tmp_path, disk_px=180, at=(300, 260), size=(700, 800),
              bg=0.01, fill=None, seed=6):
    """Synthetic 'finished image': disk somewhere on a larger field."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.clip(rng.normal(bg, 0.002, (h, w)), 0, 1).astype(np.float32)
    disk = lunar_texture(2 * disk_px, seed=seed)  # r = 0.42*2*disk_px
    y0, x0 = at[1] - disk_px, at[0] - disk_px
    # clipped paste (a truncated fixture hangs off the field edge)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1 = min(h, y0 + 2 * disk_px)
    sx1 = min(w, x0 + 2 * disk_px)
    img[sy0:sy1, sx0:sx1] += disk[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
    if fill is not None:
        # ICE-style uniform paint everywhere the moon isn't
        paint = np.full((h, w), fill, np.float32)
        mask = img > bg + 0.05
        img = np.where(mask, img, paint)
    p = str(tmp_path / "finished.png")
    write_png(p, np.clip(img, 0, 1), bit_depth=16)
    return p, 0.42 * 2 * disk_px


def _measure_disk(out_path):
    im = read_xisf(out_path)
    lit = im > 0.05
    ys, xs = np.nonzero(lit)
    r_est = (xs.max() - xs.min()) / 2
    cx = (xs.max() + xs.min()) / 2
    cy = (ys.max() + ys.min()) / 2
    return im, cx, cy, r_est


def test_prep_normalizes_geometry(tmp_path):
    src, true_r = _finished(tmp_path)
    out = str(tmp_path / "prep.xisf")
    prep_image(src, out, TARGET_R, CANVAS, log=lambda s: None)
    im, cx, cy, r_est = _measure_disk(out)
    assert im.shape == (CANVAS, CANVAS)
    assert im.ndim == 2  # mono output
    assert abs(cx - CANVAS / 2) < 6 and abs(cy - CANVAS / 2) < 6
    assert abs(r_est - TARGET_R) / TARGET_R < 0.03
    # sky is zeroed beyond the disk
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    sky = im[np.hypot(xx - CANVAS / 2, yy - CANVAS / 2) > 1.05 * TARGET_R]
    assert float(np.abs(sky).max()) == 0.0
    # tone normalization landed the disk median near 0.45
    disk_v = im[(np.hypot(xx - CANVAS / 2, yy - CANVAS / 2) < TARGET_R)
                & (im > 0.08)]
    assert 0.3 < float(np.median(disk_v)) < 0.6


def test_prep_refuses_truncated(tmp_path):
    # disk hangs off the right edge -> lit terrain along the border
    src, _ = _finished(tmp_path, at=(760, 300))
    out = str(tmp_path / "trunc.xisf")
    with pytest.raises(TruncatedSource):
        prep_image(src, out, TARGET_R, CANVAS, log=lambda s: None)


def test_prep_removes_paint_fill(tmp_path):
    src, _ = _finished(tmp_path, fill=0.06)
    out = str(tmp_path / "paint.xisf")
    prep_image(src, out, TARGET_R, CANVAS, log=lambda s: None)
    im, *_ = _measure_disk(out)
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    sky = im[np.hypot(xx - CANVAS / 2, yy - CANVAS / 2) > 1.05 * TARGET_R]
    assert float(np.abs(sky).max()) == 0.0  # paint gone from the sky
