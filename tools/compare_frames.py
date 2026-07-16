"""Frame-by-frame lunation comparison: rotation residual + NCC between two
render runs (e.g. PJSR baseline vs Python port). Frames must share indices.

  uv run python tools/compare_frames.py <dir_a> <dir_b>
"""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunation.core.stats import luminance
from lunation.core.warp import rotate
from lunation.io.images import read_image
from lunation.assemble.register import ncc_score


def frames(d):
    out = {}
    for p in glob.glob(os.path.join(d, "frame_*.png")):
        m = re.match(r"frame_(\d+)_", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    fa, fb = frames(a_dir), frames(b_dir)
    common = sorted(set(fa) & set(fb))
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    if only_a or only_b:
        print(f"set mismatch: only_a={only_a} only_b={only_b}")
    print(f"{'idx':<5}{'rot resid':>10}{'ncc@0':>8}{'ncc@best':>9}  name")
    worst = (0.0, -1)
    for i in common:
        a = luminance(read_image(fa[i]))
        b = luminance(read_image(fb[i]))
        m = (a > 0.10).astype(np.float32)
        s0 = ncc_score(b, m, a)
        best_a, best_s = 0.0, s0
        for ang in np.arange(-6, 6.01, 0.25):
            br = rotate(b, np.pi * ang / 180, b.shape[1] / 2, b.shape[0] / 2)
            s = ncc_score(br, m, a)
            if s > best_s:
                best_s, best_a = s, float(ang)
        name = os.path.basename(fa[i])[9:-4]
        print(f"{i:<5}{best_a:>10.2f}{s0:>8.3f}{best_s:>9.3f}  {name}")
        if abs(best_a) > abs(worst[0]):
            worst = (best_a, i)
    print(f"\nworst rotation residual: {worst[0]:.2f} deg at frame {worst[1]}")


if __name__ == "__main__":
    main()
