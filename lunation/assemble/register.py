"""Rotation/parity registration primitives — ports gif-frames.js:387-495,
617-644 (flipH, norm180, rimOnSmall, nccScore, bestRotationNCC).

All angles here are degrees in IMAGE coordinates (y down): left=180,
right=0, top=-90, bottom=+90. rotate() positive = visual CCW, so a feature
at screen angle theta renders at theta - rot.
"""

import numpy as np

from ..core.warp import rotate
from .disk import longest_dark_run


def flip_h(img: np.ndarray) -> np.ndarray:
    """Horizontal mirror (parity flip); screen angles map th -> 180-th."""
    return img[:, ::-1].copy() if img.ndim == 2 else img[:, ::-1, :].copy()


def norm180(a: float) -> float:
    while a <= -180:
        a += 360
    while a > 180:
        a -= 360
    return a


def rim_on_small(lum: np.ndarray, r: float) -> dict:
    """Rendered illumination direction of a rotated small luminance
    (disk centered, known radius) — gif-frames.js:419-469."""
    n = 360
    h, w = lum.shape
    cx, cy = w / 2, h / 2
    i = np.arange(n)
    a = 2 * np.pi * i / n
    s = np.zeros(n)
    cnt = np.zeros(n)
    for rr in (0.95, 0.965):
        x = np.rint(cx + rr * r * np.cos(a)).astype(int)
        y = np.rint(cy + rr * r * np.sin(a)).astype(int)
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        s[ok] += lum[y[ok], x[ok]]
        cnt[ok] += 1
    vals = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    has = cnt > 0
    thr = max(0.08, 0.30 * float(np.sort(vals)[int(0.75 * n)]))
    dark = ~has | (vals < thr)
    frac, mid = longest_dark_run(dark)
    return {"litDeg": norm180(np.degrees(mid) + 0.0), "darkFrac": frac}


def ncc_score(f: np.ndarray, m: np.ndarray, ref: np.ndarray) -> float:
    """Mask-weighted normalized correlation of lunar albedo
    (gif-frames.js:474-495)."""
    a = f * m
    s1 = float((a * ref).mean())
    s2 = float((a * a).mean())
    rm = ref * m
    s3 = float((rm * rm).mean())
    return s1 / np.sqrt(max(s2 * s3, 1e-20))


def best_rotation_ncc(f: np.ndarray, m: np.ndarray, ref: np.ndarray,
                      center: float | None = None,
                      half_range: float = 0.0) -> dict:
    """Coarse 2-degree sweep + 0.25-degree refine (gif-frames.js:617-644)."""
    h, w = f.shape
    best, best_score = 0.0, -1.0

    def eval_at(a):
        nonlocal best, best_score
        fr = rotate(f, np.pi * a / 180, w / 2, h / 2)
        mr = rotate(m, np.pi * a / 180, w / 2, h / 2)
        s = ncc_score(fr, mr, ref)
        if s > best_score:
            best_score = s
            best = a

    lo = -178.0 if center is None else center - half_range
    hi = 180.0 if center is None else center + half_range
    a = lo
    while a <= hi + 1e-9:
        eval_at(a)
        a += 2
    coarse = best
    a = coarse - 1.5
    while a <= coarse + 1.5 + 1e-9:
        eval_at(a)
        a += 0.25
    return {"angle": best, "score": best_score}
