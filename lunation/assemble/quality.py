"""Scale-independent quality metrics — ports gif-frames.js:291-385.

Measured on a standard grid (disk radius -> 200 px, nearest-neighbor).
These replace hand exclude lists: junk separates from keepers on numbers.
  detail — mean |Laplacian| per unit brightness over lit pixels
  greenX — mean (G - (R+B)/2)/L over lit pixels (green-veil color failures)
  misreg — best R-vs-B alignment shift (std px at r=200)
"""

import numpy as np

N_HALF = 210
S = 2 * N_HALF + 1


def measure_quality(img: np.ndarray, disk) -> dict:
    k = disk.r / 200.0
    color = img.ndim == 3
    h, w = img.shape[:2]

    sy, sx = np.mgrid[-N_HALF : N_HALF + 1, -N_HALF : N_HALF + 1]
    x = np.rint(disk.cx + sx * k).astype(int)
    y = np.rint(disk.cy + sy * k).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    xc, yc = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)

    if color:
        R = np.where(ok, img[yc, xc, 0], 0.0).astype(np.float32)
        G = np.where(ok, img[yc, xc, 1], 0.0).astype(np.float32)
        B = np.where(ok, img[yc, xc, 2], 0.0).astype(np.float32)
        L = (R + G + B) / 3
    else:
        L = np.where(ok, img[yc, xc], 0.0).astype(np.float32)
        R = G = B = None

    inner = np.s_[1:-1, 1:-1]
    v = L[inner]
    lit_mask = v >= 0.15
    lap = np.abs(4 * v - L[1:-1, :-2] - L[1:-1, 2:]
                 - L[:-2, 1:-1] - L[2:, 1:-1])
    lap_n = int(lit_mask.sum())
    lit = float(v[lit_mask].sum())
    detail = float(lap[lit_mask].sum()) / max(lit, 1e-6) if lap_n else 0.0
    green_x = 0.0
    if color and lap_n:
        gx = (G[inner] - (R[inner] + B[inner]) / 2) / np.maximum(v, 0.05)
        green_x = float(gx[lit_mask].mean())

    misreg = 0.0
    if color and lap_n > 2000:
        best = -2.0
        # subsampled interior grid (gif-frames.js:362-363)
        gy, gx_ = np.mgrid[8 : S - 8 : 2, 8 : S - 8 : 2]
        lit_g = L[gy, gx_] >= 0.15
        a_all = R[gy, gx_]
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                b_all = B[gy + dy, gx_ + dx]
                sel = lit_g
                n = int(sel.sum())
                if n < 500:
                    continue
                a = a_all[sel]
                b = b_all[sel]
                cov = (a * b).mean() - a.mean() * b.mean()
                den = np.sqrt(max((a * a).mean() - a.mean() ** 2, 1e-12)
                              * max((b * b).mean() - b.mean() ** 2, 1e-12))
                ncc = cov / den
                if ncc > best:
                    best = ncc
                    misreg = float(np.hypot(dx, dy))
    return {"detail": detail, "greenX": green_x, "misreg": misreg}
