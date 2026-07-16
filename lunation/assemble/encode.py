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
    common = ["ffmpeg", "-y", "-loglevel", "error",
              "-framerate", str(FRAMERATE), "-i", seq,
              "-filter_script:v", script]
    jobs = [
        ("mp4", common + ["-c:v", "libx264", "-pix_fmt", "yuv420p",
                          "-crf", "17", "-r", "30",
                          "-movflags", "+faststart",
                          os.path.join(frames_dir, "lunation.mp4")]),
        ("gif", common + [os.path.join(frames_dir, "lunation.gif")]),
    ]
    for name, argv in jobs:
        log(f"encode {name}: {n} frames")
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg {name} failed: {r.stderr.strip()}")
        log(f"encode {name} OK")
