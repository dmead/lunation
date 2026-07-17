"""AVI → SER repackager — ports scripts/avi2ser.mjs.

ffmpeg is used ONLY as a decoder/demuxer (no processing, no frame
selection); output is gray8/gray16 or interleaved-RGB frames in a SER
container. ffmpeg/ffprobe come from PATH. The raw decode goes through a
temp file next to the output (avoids pipe buffering limits on huge files).
"""

import json
import os
import re
import subprocess

from .ser import HEADER_BYTES

_COPY_CHUNK = 64 * 1024 * 1024


def probe(in_path: str) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", in_path],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {in_path}")
    streams = json.loads(r.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream in {in_path}")
    return streams[0]


def convert(in_path: str, out_path: str) -> str:
    st = probe(in_path)
    w, h = int(st["width"]), int(st["height"])
    pix = st.get("pix_fmt") or ""
    # >8-bit sources keep 16 bits; 8-bit sources stay 8-bit (no fake
    # precision). Color sources keep their channels (SER colorId 100 =
    # interleaved RGB); mono policy applies only to genuinely mono captures.
    bits = 16 if re.search(r"p?1[026]le|gray1[026]", pix) else 8
    is_color = not pix.startswith("gray")
    planes = 3 if is_color else 1
    out_fmt = (("rgb48le" if bits == 16 else "rgb24") if is_color
               else ("gray16le" if bits == 16 else "gray"))
    frame_bytes = w * h * (2 if bits == 16 else 1) * planes

    tmp_raw = out_path + ".raw"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet", "-i", in_path,
         "-f", "rawvideo", "-pix_fmt", out_fmt, tmp_raw])
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {in_path}")
    try:
        frames = os.path.getsize(tmp_raw) // frame_bytes
        if frames < 1:
            raise RuntimeError(f"no frames decoded from {in_path}")

        # SER header (178 bytes): FileID(14) LuID(4) ColorID(4)
        # LittleEndian(4) Width(4) Height(4) PixelDepth(4) FrameCount(4)
        # Observer(40) Instrument(40) Telescope(40) DateTime(8) UTC(8)
        hdr = bytearray(HEADER_BYTES)
        hdr[0:14] = b"LUCAM-RECORDER"
        hdr[18:22] = (100 if is_color else 0).to_bytes(4, "little")
        hdr[26:30] = w.to_bytes(4, "little")
        hdr[30:34] = h.to_bytes(4, "little")
        hdr[34:38] = bits.to_bytes(4, "little")
        hdr[38:42] = frames.to_bytes(4, "little")
        hdr[42:49] = b"avi2ser"
        codec = (st.get("codec_name") or "avi")[:39].encode("ascii", "replace")
        hdr[82:82 + len(codec)] = codec

        with open(out_path, "wb") as out, open(tmp_raw, "rb") as raw:
            out.write(bytes(hdr))
            left = frames * frame_bytes
            while left > 0:
                chunk = raw.read(min(_COPY_CHUNK, left))
                if not chunk:
                    break
                out.write(chunk)
                left -= len(chunk)
    finally:
        os.remove(tmp_raw)
    return (f"{out_path}: {w}x{h} {bits}-bit "
            f"{'RGB' if is_color else 'mono'}, {frames} frames")
