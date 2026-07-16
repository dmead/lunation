"""Rim-physics audit of a lunation render (verify-rot's core check):
on a north-up frame the lit limb axis must be at 0 deg (waxing) or 180
(waning). Reports the residual per frame for one or two run dirs.

  uv run python tools/rim_audit.py <run_dir> [<run_dir2>]
"""

import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunation.core.stats import luminance
from lunation.io.images import read_image
from lunation.assemble.register import norm180, rim_on_small

SMALL = 512


def audit(run_dir):
    rows = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "frame_*.png"))):
        m = re.match(r"frame_(\d+)_", os.path.basename(p))
        idx = int(m.group(1))
        lum = luminance(read_image(p))
        small = cv2.resize(lum, (SMALL, SMALL), interpolation=cv2.INTER_LINEAR)
        # disk radius on these frames: 979/2300 of the canvas scale
        r = 979 / 2300 * SMALL
        rim = rim_on_small(small, r)
        if rim["darkFrac"] < 0.30:
            rows[idx] = None  # near-full: axis unreliable, skip
            continue
        lit = rim["litDeg"]
        # residual to nearest of 0/180 (waxing/waning both acceptable here;
        # a mis-oriented frame shows tens of degrees either way)
        res = min(abs(norm180(lit)), abs(norm180(lit - 180)))
        rows[idx] = (res, lit)
    return rows


def main():
    dirs = sys.argv[1:]
    audits = [audit(d) for d in dirs]
    keys = sorted(set().union(*[a.keys() for a in audits]))
    hdr = f"{'idx':<5}" + "".join(f"{os.path.basename(d.rstrip('/')):>24}" for d in dirs)
    print(hdr)
    sums = [0.0] * len(dirs)
    counts = [0] * len(dirs)
    for k in keys:
        cells = []
        for i, a in enumerate(audits):
            v = a.get(k)
            if v is None:
                cells.append(f"{'(near-full)':>24}")
            else:
                cells.append(f"{f'{v[0]:6.1f} (lit {v[1]:7.1f})':>24}")
                sums[i] += v[0]
                counts[i] += 1
        print(f"{k:<5}" + "".join(cells))
    for i, d in enumerate(dirs):
        if counts[i]:
            print(f"mean |rim residual| {os.path.basename(d.rstrip('/'))}: "
                  f"{sums[i] / counts[i]:.2f} deg over {counts[i]} frames")


if __name__ == "__main__":
    main()
