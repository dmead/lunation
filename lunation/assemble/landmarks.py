"""Maria-blob landmarks + rotation Hough — ports gif-frames.js:497-615.
Advisory logging only (never drives the registration decision)."""

import numpy as np
from scipy import ndimage


def maria_blobs(lum: np.ndarray, disk_r: float) -> list[dict]:
    h, w = lum.shape
    cx, cy = w / 2, h / 2
    lit = lum > 0.10
    if int(lit.sum()) < 200:
        return []
    lit_mean = float(lum[lit].mean())
    maria_thr = 0.72 * lit_mean
    yy, xx = np.mgrid[0:h, 0:w]
    inside = np.hypot(xx - cx, yy - cy) <= 0.95 * disk_r
    # NOTE: the JS gates only the flood-fill SEED by radius and lets blobs
    # grow past it; masking uniformly is the clean equivalent (advisory-only)
    cand = (lum > 0.04) & (lum < maria_thr) & inside
    lbl, n = ndimage.label(cand, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    blobs = []
    if n:
        idx = np.arange(1, n + 1)
        areas = ndimage.sum_labels(np.ones_like(lum), lbl, idx)
        cys, cxs = zip(*ndimage.center_of_mass(np.ones_like(lum), lbl, idx))
        for area, bx, by in zip(areas, cxs, cys):
            if area >= 40:
                blobs.append({
                    "th": float(np.arctan2(by - cy, bx - cx)),
                    "rho": float(np.hypot(bx - cx, by - cy) / disk_r),
                    "area": float(area),
                })
    blobs.sort(key=lambda b: -b["area"])
    return blobs[:14]


def hough_rotation(f_blobs: list[dict], a_blobs: list[dict]) -> dict | None:
    BIN, N = 2, 180
    hist = np.zeros(N)
    pairs = 0
    for fb in f_blobs:
        for ab in a_blobs:
            if abs(fb["rho"] - ab["rho"]) > 0.10:
                continue
            ratio = fb["area"] / ab["area"]
            if ratio < 0.35 or ratio > 2.9:
                continue
            d = np.degrees(ab["th"] - fb["th"])
            while d < -180:
                d += 360
            while d >= 180:
                d -= 360
            b = int((d + 180) // BIN) % N
            wgt = min(fb["area"], ab["area"])
            hist[b] += wgt
            hist[(b + 1) % N] += 0.5 * wgt
            hist[(b + N - 1) % N] += 0.5 * wgt
            pairs += 1
    if pairs < 3:
        return None
    best = int(np.argmax(hist))
    total = float(hist.sum())
    if total <= 0 or hist[best] / total < 0.12:
        return None
    return {"angle": best * BIN - 180 + BIN / 2, "support": hist[best] / total}
