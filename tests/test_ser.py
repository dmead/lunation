"""SER container round-trip tests against hand-built files."""

import numpy as np
import pytest

from lunation.io.ser import (CFA_LAYOUT, HEADER_BYTES, SerReader, read_header,
                             write_trimmed)


def build_ser(path, frames, color_id, depth):
    """frames: list of (H,W) or (H,W,3) uint arrays, already mosaiced for
    Bayer ids."""
    f0 = frames[0]
    h, w = f0.shape[0], f0.shape[1]
    header = bytearray(HEADER_BYTES)
    header[0:14] = b"LUCAM-RECORDER"
    header[14:18] = np.int32(0).tobytes()            # LuID
    header[18:22] = np.int32(color_id).tobytes()
    header[22:26] = np.int32(0).tobytes()            # endianness (ignored)
    header[26:30] = np.int32(w).tobytes()
    header[30:34] = np.int32(h).tobytes()
    header[34:38] = np.int32(depth).tobytes()
    header[38:42] = np.int32(len(frames)).tobytes()
    with open(path, "wb") as f:
        f.write(bytes(header))
        for fr in frames:
            f.write(np.ascontiguousarray(fr).tobytes())


def test_mono16(tmp_path):
    p = str(tmp_path / "m16.ser")
    fr = np.arange(64, dtype="<u2").reshape(8, 8) * 1000
    build_ser(p, [fr, fr * 0 + 65535], color_id=0, depth=16)
    r = SerReader(p)
    assert r.frame_count == 2 and r.width == 8 and r.height == 8
    np.testing.assert_allclose(r.read(0), fr / 65535.0, atol=1e-6)
    np.testing.assert_allclose(r.read(1), 1.0, atol=1e-6)


def test_mono8(tmp_path):
    p = str(tmp_path / "m8.ser")
    fr = np.arange(64, dtype="u1").reshape(8, 8)
    build_ser(p, [fr], color_id=0, depth=8)
    r = SerReader(p)
    np.testing.assert_allclose(r.read(0), fr / 255.0, atol=1e-6)


@pytest.mark.parametrize("color_id", [8, 9, 10, 11])
def test_bayer_planes_exact(tmp_path, color_id):
    """Each CFA site gets a distinct constant; every extracted plane must
    return exactly its site's value, and mono the 2x2 average."""
    p = str(tmp_path / f"bayer{color_id}.ser")
    vals = {"R": 10000, "G1": 20000, "G2": 30000, "B": 40000}
    cfa = CFA_LAYOUT[color_id]
    raw = np.zeros((8, 8), dtype="<u2")
    for site, (dx, dy) in cfa.items():
        raw[dy::2, dx::2] = vals[site]
    build_ser(p, [raw], color_id=color_id, depth=16)

    for ch, expected in (("R", vals["R"]), ("B", vals["B"]),
                         ("G", (vals["G1"] + vals["G2"]) / 2)):
        r = SerReader(p, channel=ch)
        assert r.width == 4 and r.height == 4
        np.testing.assert_allclose(r.read(0), expected / 65535.0, atol=1e-6)

    mono = SerReader(p).read(0)
    np.testing.assert_allclose(mono, sum(vals.values()) / 4 / 65535.0,
                               atol=1e-6)


@pytest.mark.parametrize("color_id", [100, 101])
def test_rgb_interleaved(tmp_path, color_id):
    p = str(tmp_path / f"rgb{color_id}.ser")
    fr = np.zeros((4, 4, 3), dtype="<u2")
    fr[:, :, 0] = 10000
    fr[:, :, 1] = 20000
    fr[:, :, 2] = 30000
    build_ser(p, [fr], color_id=color_id, depth=16)
    r_val = 10000 if color_id == 100 else 30000  # 101 = BGR order
    np.testing.assert_allclose(SerReader(p, "R").read(0), r_val / 65535.0,
                               atol=1e-6)
    np.testing.assert_allclose(SerReader(p).read(0),
                               60000 / 3 / 65535.0, atol=1e-6)


def test_truncation_guard(tmp_path):
    p = str(tmp_path / "trunc.ser")
    fr = np.zeros((8, 8), dtype="<u2")
    build_ser(p, [fr, fr, fr], color_id=0, depth=16)
    # chop half of the last frame off
    import os

    size = os.path.getsize(p)
    with open(p, "r+b") as f:
        f.truncate(size - 64)
    r = SerReader(p)
    assert r.truncated and r.frame_count == 2


def test_write_trimmed_round_trip(tmp_path):
    src = str(tmp_path / "src.ser")
    out = str(tmp_path / "out.ser")
    frames = [np.arange(16 * 12, dtype="<u2").reshape(12, 16) + 100 * i
              for i in range(5)]
    build_ser(src, frames, color_id=0, depth=16)
    write_trimmed(src, out, [1, 3], x0=2, y0=4, crop_w=8, crop_h=6)
    h = read_header(out)
    assert (h.raw_width, h.raw_height, h.frame_count) == (8, 6, 2)
    r = SerReader(out)
    np.testing.assert_allclose(
        r.read(0), frames[1][4:10, 2:10] / 65535.0, atol=1e-6)
    np.testing.assert_allclose(
        r.read(1), frames[3][4:10, 2:10] / 65535.0, atol=1e-6)
