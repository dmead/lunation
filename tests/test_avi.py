"""avi2ser round trip: known raw frames → lossless AVI → SER.

Uses ffmpeg (on PATH per project rules) as the AVI muxer. 8-bit mono rides
raw gray; gray16le/rgb24 aren't AVI-representable as rawvideo, so those go
through FFV1 (lossless; RGB is stored as bgr0) — either way the SER must
reproduce the input values bit-exact.
"""

import shutil
import subprocess

import numpy as np
import pytest

from lunation.io.avi import convert
from lunation.io.ser import SerReader, read_header

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH")

W, H, N = 32, 24, 5


def make_avi(tmp_path, frames, in_fmt, codec="ffv1"):
    raw = tmp_path / "in.raw"
    raw.write_bytes(b"".join(np.ascontiguousarray(f).tobytes()
                             for f in frames))
    avi = str(tmp_path / "in.avi")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet", "-f", "rawvideo",
         "-pix_fmt", in_fmt, "-s", f"{W}x{H}", "-r", "30",
         "-i", str(raw), "-c:v", codec, avi],
        check=True)
    return avi


def test_mono8_round_trip(tmp_path):
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, (H, W), dtype=np.uint8)
              for _ in range(N)]
    out = str(tmp_path / "out.ser")
    convert(make_avi(tmp_path, frames, "gray", codec="rawvideo"), out)

    hdr = read_header(out)
    assert (hdr.color_id, hdr.depth, hdr.frame_count) == (0, 8, N)
    r = SerReader(out)
    assert (r.width, r.height, r.frame_count) == (W, H, N)
    for i, f in enumerate(frames):
        np.testing.assert_array_equal(r.read_raw(i), f)


def test_mono16_round_trip(tmp_path):
    rng = np.random.default_rng(11)
    frames = [rng.integers(0, 65536, (H, W)).astype("<u2")
              for _ in range(N)]
    out = str(tmp_path / "out.ser")
    convert(make_avi(tmp_path, frames, "gray16le"), out)

    hdr = read_header(out)
    assert (hdr.color_id, hdr.depth, hdr.frame_count) == (0, 16, N)
    r = SerReader(out)
    for i, f in enumerate(frames):
        np.testing.assert_array_equal(r.read_raw(i), f)


def test_rgb_round_trip(tmp_path):
    rng = np.random.default_rng(13)
    frames = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
              for _ in range(N)]
    out = str(tmp_path / "out.ser")
    convert(make_avi(tmp_path, frames, "rgb24"), out)

    hdr = read_header(out)
    assert (hdr.color_id, hdr.depth, hdr.frame_count) == (100, 8, N)
    r = SerReader(out)
    for i, f in enumerate(frames):
        np.testing.assert_array_equal(r.read_raw(i), f)
