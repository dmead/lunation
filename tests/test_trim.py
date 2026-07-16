"""Trim end-to-end: ROI detection + frame selection + SER rewrite."""

import numpy as np

from lunation.io.ser import SerReader, read_header
from lunation.stack.trim import run as trim_run

from .synth import lunar_texture
from .test_ser import build_ser


def test_trim_crops_to_disk_and_keeps_best(tmp_path):
    rng = np.random.default_rng(9)
    size = 256
    # small moon in the top-left quadrant of a big dark frame
    disk = lunar_texture(96, seed=4)
    frames = []
    for k in range(60):
        f = rng.normal(0.003, 0.002, (size, size)).astype(np.float32)
        f = np.clip(f, 0, 1)
        d = disk if k % 3 else disk * 0.6  # every 3rd frame dimmer/softer
        f[20:116, 30:126] += d
        frames.append(np.rint(np.clip(f, 0, 1) * 65535).astype("<u2"))
    src = str(tmp_path / "src.ser")
    out = str(tmp_path / "trim.ser")
    log = str(tmp_path / "trim.log")
    build_ser(src, frames, color_id=0, depth=16)

    assert trim_run(src, out, 0.5, log) is True
    text = open(log).read()
    assert "=== TRIM OK ===" in text

    h = read_header(out)
    assert h.frame_count == 30
    # ROI covers the disk (+PAD, clamped to frame), far smaller than full
    assert h.raw_width < size and h.raw_height < size
    assert h.raw_width >= 96 and h.raw_height >= 96
    # even-aligned offsets preserved CFA phase by construction
    r = SerReader(out)
    f0 = r.read(0)
    assert float(f0.max()) > 0.3  # the disk is inside the crop


def test_trim_degenerate_keeps_full_frame(tmp_path):
    """All-dark capture -> degenerate ROI -> full frame kept."""
    rng = np.random.default_rng(1)
    frames = [np.rint(np.clip(rng.normal(0.002, 0.001, (128, 128)), 0, 1)
                      * 65535).astype("<u2") for _ in range(30)]
    src = str(tmp_path / "dark.ser")
    out = str(tmp_path / "dark_trim.ser")
    log = str(tmp_path / "dark.log")
    build_ser(src, frames, color_id=0, depth=16)
    assert trim_run(src, out, 0.4, log) is True
    h = read_header(out)
    assert (h.raw_width, h.raw_height) == (128, 128)
