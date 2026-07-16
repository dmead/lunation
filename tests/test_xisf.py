"""XISF write→read round trip. The PixInsight-side read is the M0 manual
gate (a written file must open in PI itself)."""

import numpy as np

from lunation.io.xisf_io import read_xisf, write_xisf

from .synth import lunar_texture


def test_mono_round_trip(tmp_path):
    p = str(tmp_path / "mono.xisf")
    img = lunar_texture(128).astype(np.float32)
    write_xisf(p, img)
    back = read_xisf(p)
    assert back.shape == img.shape
    np.testing.assert_array_equal(back, img)


def test_rgb_round_trip(tmp_path):
    p = str(tmp_path / "rgb.xisf")
    img = np.stack([lunar_texture(64, seed=s) for s in (1, 2, 3)],
                   axis=-1).astype(np.float32)
    write_xisf(p, img)
    back = read_xisf(p)
    assert back.shape == img.shape
    np.testing.assert_array_equal(back, img)
