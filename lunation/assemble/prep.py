"""Normalize arbitrary FINISHED moon images for the lunation pipeline —
ports pjsr/prep-finished.js (2026-07 rev) 1:1.

Border-background subtraction with noise-scaled detection (no per-item
tuning), scale-free two-pass Kasa disk fit, truncation refusal (lit terrain
running along the source border), resample to the standard working radius,
center on the standard canvas, sky zeroed at 1.02r, auto fill-pedestal +
flatness paint removal for stitched panoramas, gray-world + MTF tone
normalization, MONO output (desaturated data gains nothing from RGB and
mono makes residual color artifacts physically impossible).

KNOWN UPSTREAM ISSUE (verified against the live PJSR script 2026-07-16):
on smooth 8-bit sources with a real sky pedestal (nikon-moon-whole.jpg),
the fill gate triggers and the flatness classifier then eats denoised
JPEG-quantized MARIA as "paint" — identical behavior in both
implementations (masks agree within 1%). The FIN_*.xisf archive predates
this JS revision and is unaffected. Fix the discriminator here first
(candidates bounded near the measured fill level, or quantization-aware
flatness), then backport or note for the frozen JS.
"""

import math
import os

import numpy as np

from ..core.stats import luminance_pi as luminance
from ..core.warp import resample
from ..io.images import read_image
from ..io.xisf_io import write_xisf

TARGET_R = 979
CANVAS = 2300


class TruncatedSource(ValueError):
    pass


def _kasa(pts: np.ndarray, cx, cy, r, iters, tol_fn, clamp_fn):
    for it in range(iters):
        tol = tol_fn(it, r)
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        sel = pts[d <= tol]
        if len(sel) < 20:
            break
        x, y = sel[:, 0], sel[:, 1]
        z = x * x + y * y
        try:
            sol, *_ = np.linalg.lstsq(
                np.column_stack([x, y, np.ones(len(sel))]), z, rcond=None)
        except np.linalg.LinAlgError:
            break
        cx, cy = sol[0] / 2, sol[1] / 2
        rn = math.sqrt(max(1.0, sol[2] + cx * cx + cy * cy))
        r = clamp_fn(r, rn)
    return cx, cy, r


def prep_image(src: str, out: str, target_r: int = TARGET_R,
               canvas: int = CANVAS, log=print) -> None:
    img = read_image(src)
    lum = luminance(img)
    H, W = lum.shape

    # background: median of a border ring (10px in from each edge)
    border = np.concatenate([
        lum[min(10, H - 1), ::3], lum[max(H - 11, 0), ::3],
        lum[::3, min(10, W - 1)], lum[::3, max(W - 11, 0)]])
    bg = float(np.median(border))
    sigma = 1.4826 * float(np.median(np.abs(border - bg)))

    stats = lum[::4, ::4].ravel()
    p995 = float(np.sort(stats)[int(0.995 * stats.size)])
    # detection floor rides the measured noise, not a fixed constant
    thr = bg + max(0.01, 6 * sigma, 0.30 * (p995 - bg))

    def limb_points(threshold, ring=None):
        pts, bbox = [], [W, 0, H, 0]
        for y in range(0, H, 2):
            xs = np.nonzero(lum[y, ::2] > threshold)[0]
            if ring is not None:
                cx0, cy0, r0_, frac = ring
                xr = xs * 2
                keep = np.abs(np.hypot(xr - cx0, y - cy0) - r0_) < frac * r0_
                xs = xs[keep]
            if xs.size:
                first, last = int(xs[0]) * 2, int(xs[-1]) * 2
                pts += [(first, y), (last, y)]
                bbox = [min(bbox[0], first), max(bbox[1], last),
                        min(bbox[2], y), max(bbox[3], y)]
        return np.array(pts, dtype=np.float64), bbox

    pts, (bx0, bx1, by0, by1) = limb_points(thr)
    if len(pts) < 60:
        raise ValueError(f"no lunar disk found in the image"
                         f" (detection thr {thr:.3f})")

    # scale-free Kasa: r0 from the lit bbox, generous clamp
    r0 = max(bx1 - bx0, by1 - by0) / 2
    cx, cy, r = _kasa(
        pts, (bx0 + bx1) / 2, (by0 + by1) / 2, r0, 14,
        lambda it, r_: max(40, 0.15 * r_) if it < 3 else max(18, 0.05 * r_),
        lambda r_, rn: max(0.5 * r0, min(1.6 * r0, rn)))

    # pass 2: low threshold on bg-subtracted levels — soft limbs cross a
    # high threshold well inside the true limb
    thr2 = bg + max(0.02, 0.10 * (p995 - bg))
    pts2, _ = limb_points(thr2, ring=(cx, cy, r, 0.35))
    if len(pts2) > 60:
        cx, cy, r = _kasa(
            pts2, cx, cy, r, 10,
            lambda it, r_: max(30, 0.10 * r_) if it < 3 else max(14, 0.04 * r_),
            lambda r_, rn: max(0.8 * r_, min(1.25 * r_, rn)))

    # truncation refusal: lit terrain running along the SOURCE border means
    # the source does not hold the whole disk
    lit_thr = bg + max(0.06, 0.30 * (p995 - bg))
    max_run = 0
    for band in (lum[0:3, :], lum[H - 3 : H, :],
                 lum[:, 0:3].T, lum[:, W - 3 : W].T):
        lit_line = (band > lit_thr).any(axis=0)
        run = 0
        for b in lit_line:
            run = run + 1 if b else 0
            max_run = max(max_run, run)
    if max_run >= 0.16 * r:
        raise TruncatedSource(
            f"TRUNCATED source — lit terrain runs {max_run}px along the"
            f" border (disk r {r:.0f}); the source does not contain the"
            " full lunar disk")

    log(f"{os.path.basename(src)}: {W}x{H} bg {bg:.4f} sigma {sigma:.4f}"
        f" thr {thr:.3f}/{thr2:.3f} disk cx {cx:.0f} cy {cy:.0f} r {r:.1f}")

    # background subtract, rescale to target_r, center on canvas
    work = img.astype(np.float32)
    if bg > 0.004:
        work = np.clip(work - bg, 0, 1)
    s = target_r / r
    work = resample(work, s, "bicubic")
    icx, icy = round(cx * s), round(cy * s)
    ch = () if work.ndim == 2 else (work.shape[2],)
    half = canvas >> 1
    canv = np.zeros((canvas, canvas, *ch), dtype=np.float32)
    sx0, sy0 = max(0, icx - half), max(0, icy - half)
    sx1, sy1 = min(work.shape[1], icx + half), min(work.shape[0], icy + half)
    dx0, dy0 = half - (icx - sx0), half - (icy - sy0)
    canv[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = \
        work[sy0:sy1, sx0:sx1]

    # ---- bring in line with the stacks: sky zero, gray-world, MTF tone ----
    nch = 1 if canv.ndim == 2 else canv.shape[2]
    chans = ([canv] if nch == 1
             else [canv[:, :, c] for c in range(nch)])
    cc = canvas >> 1
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    dist = np.hypot(xx - cc, yy - cc)
    r_lim = 1.02 * target_r  # stitch staircase edges hug the limb

    med = []
    sub = np.s_[::4, ::4]
    for a in chans:
        v = a[sub][(dist[sub] <= target_r) & (a[sub] > 0.08)]
        med.append(float(np.median(v)) if v.size else 0.4)
    gray = sum(med) / nch
    x0 = max(0.03, min(0.9, gray))
    y0 = 0.45
    m = (x0 * (y0 - 1)) / (2 * x0 * y0 - x0 - y0)
    m = max(0.28, min(0.75, m))  # thin-crescent boost clamp

    # synthetic canvas fill (ICE panoramas): its level just OUTSIDE the disk
    # tells us what "non-moon" looks like — floor out anything at/below it.
    # Only sample canvas the source copy actually covered.
    floor_abs = 0.0
    a0 = chans[0]
    cvx0, cvy0 = dx0 + 5, dy0 + 5
    cvx1, cvy1 = dx0 + (sx1 - sx0) - 10, dy0 + (sy1 - sy0) - 10
    ys = slice(max(0, cvy0), min(canvas, cvy1), 2)
    xs = slice(max(0, cvx0), min(canvas, cvx1), 2)
    dpatch = dist[ys, xs]
    ann = a0[ys, xs][(dpatch > 1.05 * target_r) & (dpatch < 1.35 * target_r)]
    if ann.size > 500:
        fill = float(np.sort(ann)[int(0.75 * ann.size)])  # rides over zeros
        if fill > 0.02:
            nz = np.sort(ann[ann > 0.5 * fill])
            p95 = float(nz[int(0.95 * nz.size)])
            floor_abs = p95 + 0.02
            log(f"  fill pedestal {fill:.3f} (p95 {p95:.3f}) outside disk"
                f" -> floor {floor_abs:.3f}")

    # paint is FLAT where real terrain textures: classify sub-lit flat
    # pixels as synthetic and zero them (dilated). Gated on fill detection.
    syn = None
    if floor_abs > 0:
        import cv2

        lit_cut = 0.5 * (p995 - bg)
        cand = (a0 > 0) & (a0 < lit_cut)
        mx = np.full_like(a0, -1e9)
        mn = np.full_like(a0, 1e9)
        for dy in (-3, -1, 1, 3):
            for dx in (-3, -1, 1, 3):
                sh = np.roll(np.roll(a0, dy, axis=0), dx, axis=1)
                np.maximum(mx, sh, out=mx)
                np.minimum(mn, sh, out=mn)
        flat = cand & ((mx - mn) < 0.012)
        syn = cv2.dilate(flat.astype(np.uint8), np.ones((5, 5), np.uint8))
        syn = syn.astype(bool)
        log(f"  synthetic-paint mask: {int(flat.sum())} flat px (dilated)")

    out_chans = []
    for a, mc in zip(chans, med):
        g = gray / max(mc, 1e-4)
        v = np.clip(a * g, 0, 1)
        toned = ((m - 1) * v) / ((2 * m - 1) * v - m)
        kill = (dist > r_lim) | (a < floor_abs)
        if syn is not None:
            kill |= syn
        out_chans.append(np.where(kill, 0.0, toned).astype(np.float32))
    mono = (sum(out_chans) / nch).astype(np.float32)
    log(f"  tone: med {'/'.join(f'{v:.3f}' for v in med)} -> gray {gray:.3f}"
        f" mtf m {m:.3f}"
        + (f" floor {floor_abs:.3f}" if floor_abs > 0 else "") + " -> MONO")
    write_xisf(out, mono)
    log(f"  -> {out}")


def run(cfg: dict) -> bool:
    """Config runner: {targetR, canvas, log, items:[{src,out}]}."""
    lines = []

    def log(s):
        lines.append(str(s))
        print(s)

    ok = True
    for it in cfg["items"]:
        try:
            prep_image(it["src"], it["out"], cfg.get("targetR", TARGET_R),
                       cfg.get("canvas", CANVAS), log)
        except Exception as e:  # noqa: BLE001 — per-item boundary
            log(f"PREP FAILED {it['src']}: {e}")
            ok = False
    log("PREP DONE")
    if cfg.get("log"):
        with open(cfg["log"], "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return ok
