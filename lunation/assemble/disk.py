"""Disk geometry + illumination analysis — ports gif-frames.js:26-289.

analyze_disk: Kasa circle fit with radius prior (dual tight/wide fit),
lit-terrain border-clip test, dual-ring rim illumination axis with adaptive
dark threshold, and lit/dark edge containment. `force` skips the fit for
prep-normalized inputs whose geometry is exact by construction.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.stats import luminance_pi as luminance

# mean lunar diameter at TS-70 2x drizzle scale (gif-frames.js:114)
R_EXP = 1865 / 2 / (1.763 / 2)
NR = 360


@dataclass
class Disk:
    cx: float
    cy: float
    r: float
    lit: float = 0.0
    theta: float = 0.0
    off: float = 0.0
    thetaRim: float = 0.0
    rimDark: float = 0.0
    borderRun: int = 0
    fitWide: bool = False
    fitRms: float = 0.0
    fitArc: float = 360.0
    litContain: float = 0.0
    darkContain: float = 0.0
    q: dict = field(default_factory=dict)


def _border_run(m: np.ndarray, thr: float = 0.20) -> int:
    """Longest contiguous bright run within 3 lines of any border
    (gif-frames.js:40-71). Dozens of px = glow kiss, hundreds = capture clip."""
    best = 0
    lines = []
    for d in range(3):
        lines += [m[d, :], m[-1 - d, :], m[:, d], m[:, -1 - d]]
    for line in lines:
        bright = line > thr
        run = 0
        for b in bright:
            run = run + 1 if b else 0
            if run > best:
                best = run
    return int(best)


def _ring_dark(lum, cx, cy, r, radii_in, radii_out):
    """Dual-ring samples + adaptive dark classification (gif-frames.js:188-240)."""
    h, w = lum.shape
    i = np.arange(NR)
    a = 2 * np.pi * i / NR

    def ring(radii):
        s = np.zeros(NR)
        n = np.zeros(NR)
        for rr in radii:
            x = np.rint(cx + rr * r * np.cos(a)).astype(int)
            y = np.rint(cy + rr * r * np.sin(a)).astype(int)
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            s[ok] += lum[y[ok], x[ok]]
            n[ok] += 1
        vals = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        return vals, n > 0

    ring96, has96 = ring(radii_in)
    ring99, has99 = ring(radii_out)
    dark_thr = max(0.08, 0.30 * float(np.sort(ring96)[int(0.75 * NR)]))
    # consensus of inner + extreme-limb rings: a real terminator lune is dark
    # at BOTH; a dark mare crossing the inner ring still has bright rim outside
    dark = ((~has96 | (ring96 < dark_thr))
            & (~has99 | (ring99 < dark_thr)))
    return dark


def longest_dark_run(dark: np.ndarray) -> tuple[float, float]:
    """(fraction, mid_angle_rad) of the longest circular dark run
    (gif-frames.js:242-262)."""
    n = len(dark)
    best_len, best_mid, run0 = 0, 0, -1
    for i in range(2 * n):
        if dark[i % n]:
            if run0 < 0:
                run0 = i
        elif run0 >= 0:
            ln = min(i - run0, n)
            if ln > best_len:
                best_len = ln
                best_mid = ((run0 + i - 1) / 2) % n
            run0 = -1
    return best_len / n, 2 * np.pi * best_mid / n + np.pi


def analyze_disk(img: np.ndarray, force: dict | None = None) -> Disk:
    lum = luminance(img)
    border_run = _border_run(lum)

    # circle-fit points from a LOW threshold (0.04) copy: soft limbs ramp and
    # the 0.08 crossing sits inside the true limb; lit stats from 0.08
    m_fit = lum > 0.04
    m_lit = lum > 0.08
    h, w = lum.shape

    pts = []
    bx0, bx1, by0, by1 = w, 0, h, 0
    ys, xs = np.nonzero(m_lit[::2, ::2])
    ln = len(xs)
    lcx = float(xs.mean()) * 2 if ln else 0.0
    lcy = float(ys.mean()) * 2 if ln else 0.0
    for y in range(0, h, 2):
        row = np.nonzero(m_fit[y, ::2])[0]
        if row.size:
            first, last = int(row[0]) * 2, int(row[-1]) * 2
            pts.append((first, y))
            pts.append((last, y))
            bx0, bx1 = min(bx0, first), max(bx1, last)
            by0, by1 = min(by0, y), max(by1, y)
    p = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))

    def kasa_fit(cx0, cy0, r0, r_lo, r_hi):
        cx, cy, r = cx0, cy0, r0
        for it in range(12):
            tol = 60 if it < 3 else 25
            d = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r)
            sel = p[d <= tol]
            n = len(sel)
            if n < 20:
                break
            x, y = sel[:, 0], sel[:, 1]
            z = x * x + y * y
            A = np.column_stack([x, y, np.ones(n)])
            try:
                sol, *_ = np.linalg.lstsq(A, z, rcond=None)
            except np.linalg.LinAlgError:
                break
            cx, cy = sol[0] / 2, sol[1] / 2
            r = math.sqrt(max(1.0, sol[2] + cx * cx + cy * cy))
            r = max(r_lo, min(r_hi, r))
        d = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r if len(p) else np.zeros(0)
        inl = np.abs(d) < 12
        n = int(inl.sum())
        rms = float(np.sqrt((d[inl] ** 2).mean())) if n else 0.0
        ang = np.arctan2(p[inl, 1] - cy, p[inl, 0] - cx) if n else np.zeros(0)
        bins = np.zeros(36, dtype=bool)
        if n:
            bins[(np.floor((ang + np.pi) / (2 * np.pi) * 36) % 36).astype(int)] = True
        return {"cx": cx, "cy": cy, "r": r, "n": n, "rms": rms,
                "arc": int(bins.sum()) * 10}

    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    bbox_r = max(bx1 - bx0, by1 - by0) / 2
    use_b = False
    if force:
        fit = {"cx": force["cx"], "cy": force["cy"], "r": force["r"],
               "rms": 0.0, "arc": 360, "n": 0}
    else:
        fit_a = kasa_fit(bcx, bcy, R_EXP, 0.9 * R_EXP, 1.1 * R_EXP)
        fit_b = kasa_fit(bcx, bcy, max(bbox_r, 0.4 * R_EXP),
                         0.35 * R_EXP, 1.30 * R_EXP)
        use_b = (fit_b["arc"] >= 140 and fit_b["n"] >= 40
                 and (fit_b["rms"] < 0.7 * fit_a["rms"] or fit_a["n"] < 20))
        fit = fit_b if use_b else fit_a
    cx, cy, r = fit["cx"], fit["cy"], fit["r"]

    dark = _ring_dark(lum, cx, cy, r,
                      radii_in=(0.955, 0.965, 0.975),
                      radii_out=(0.985, 0.992))
    rim_dark, theta_rim = longest_dark_run(dark)

    theta = math.atan2(lcy - cy, lcx - cx)
    edges = [((w - cx - r) / r, 0.0), ((cy - r) / r, -np.pi / 2),
             ((cx - r) / r, np.pi), ((h - cy - r) / r, np.pi / 2)]
    lit_contain = dark_contain = 0.0
    for margin, ang in edges:
        d_a = abs(math.atan2(math.sin(ang - theta_rim),
                             math.cos(ang - theta_rim)))
        if rim_dark < 0.10 or d_a < np.pi * 0.45:
            lit_contain = min(lit_contain, margin)
        else:
            dark_contain = min(dark_contain, margin)

    return Disk(cx=cx, cy=cy, r=r,
                lit=(ln * 4) / (np.pi * r * r),
                theta=theta,
                off=math.hypot(lcx - cx, lcy - cy) / r,
                thetaRim=theta_rim, rimDark=rim_dark, borderRun=border_run,
                fitWide=use_b, fitRms=fit["rms"], fitArc=fit["arc"],
                litContain=lit_contain, darkContain=dark_contain)
