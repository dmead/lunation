"""GIF/MP4 encode — ports pjsr/master/Encode.jsh.

Copies the gappy frame_<idx>_*.png sequence to dense seq_%02d.png, writes a
stamps.txt drawtext filter script (one per-frame date stamp gated by
enable='eq(n,K)'), then runs ffmpeg twice: libx264 MP4 (crf 17, yuv420p,
r 30, faststart) and a direct GIF encode (palettegen has crashed before —
never use it). ffmpeg comes from PATH.
"""

import glob
import os
import re
import shutil
import subprocess

FONT = "C\\:/Windows/Fonts/consola.ttf"
FRAMERATE = 1.5


def prep_sequence(frames_dir: str) -> tuple[str, int]:
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png in {frames_dir}")
    stamps = []
    for k, src in enumerate(frames):
        shutil.copyfile(src, os.path.join(frames_dir, f"seq_{k:02d}.png"))
        m = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(src))
        date = m.group(0) if m else ""
        stamps.append(
            f"drawtext=fontfile='{FONT}':text='{date}':x=34:y=h-70:"
            f"fontsize=36:fontcolor=white@0.65:enable='eq(n,{k})'")
    script = os.path.join(frames_dir, "stamps.txt")
    with open(script, "w", encoding="utf-8") as f:
        f.write(",\n".join(stamps) + "\n")
    return script, len(frames)


def encode(frames_dir: str, log=print) -> None:
    script, n = prep_sequence(frames_dir)
    seq = os.path.join(frames_dir, "seq_%02d.png")
    log(f"encode mp4: {n} frames")
    argv = ["ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FRAMERATE), "-i", seq,
            "-filter_script:v", script,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
            "-r", "30", "-movflags", "+faststart",
            os.path.join(frames_dir, "lunation.mp4")]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mp4 failed: {r.stderr.strip()}")
    log("encode mp4 OK")

    # GIF via Pillow with a PER-FRAME adaptive 256-color palette. ffmpeg's
    # paletteless GIF path crushed lunar frames to ~50 colors (posterized
    # terminator gradients); palettegen has crashed before on this box.
    # 256 adaptive levels per frame is visually lossless on near-grayscale
    # moons. Date stamps are burned by ffmpeg only into the MP4; the GIF
    # gets them drawn here so both carry the same info.
    log(f"encode gif: {n} frames (pillow, per-frame palettes)")
    _encode_gif_pillow(frames_dir, n)
    log("encode gif OK")


def run(frames_dir: str) -> bool:
    """Job wrapper: encode.log with standard sentinels, for the master
    scheduler (replaces Encode.jsh's ffmpeg `-progress` file tailing —
    both ends of that contract are ours now)."""
    import time
    import traceback

    from ..stack.logutil import JobLog

    jl = JobLog(os.path.join(frames_dir, "encode.log"))
    t0 = time.time()
    try:
        encode(frames_dir, log=jl.log)
        jl.log(f"=== ENCODE OK ({time.time() - t0:.1f} s) ===")
        return True
    except Exception as e:  # noqa: BLE001 — job boundary
        jl.log(f"*** ENCODE FAILED: {e}")
        jl.log(traceback.format_exc())
        return False
    finally:
        jl.close()


def _encode_gif_pillow(frames_dir: str, n: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    quantized = []
    for k in range(n):
        im = Image.open(os.path.join(frames_dir, f"seq_{k:02d}.png")).convert("RGB")
        src = _seq_source(frames_dir, k)
        m = re.search(r"\d{4}-\d{2}-\d{2}", src or "")
        if m:
            ImageDraw.Draw(im).text((34, im.height - 70), m.group(0),
                                    fill=(166, 166, 166), font=font)
        quantized.append(im.quantize(
            colors=256, method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG))
    quantized[0].save(
        os.path.join(frames_dir, "lunation.gif"), save_all=True,
        append_images=quantized[1:], duration=round(1000 / FRAMERATE),
        loop=0, optimize=False)


def _seq_source(frames_dir: str, k: int) -> str | None:
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    return os.path.basename(frames[k]) if k < len(frames) else None
