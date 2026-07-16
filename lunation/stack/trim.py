"""Stage-side SER reducer — ports pjsr/ser-trim.js.

Read a SER once from its (slow) source drive and write a much smaller SER
for stacking:
  1. quality-rank all frames (Laplacian stdDev), keep the best fraction
  2. ROI-crop every kept frame to the union lit bounding box (+margin),
     discarding empty sky — offsets even-aligned to preserve CFA phase
Frame order, bit depth and colorId are preserved; W/H/frameCount updated.
Log sentinels: `=== TRIM OK ===` / `*** TRIM FAILED:`.
"""

import time
import traceback

import numpy as np

from ..core.kernels import laplacian_sharpness
from ..io.ser import SerReader, write_trimmed
from .logutil import JobLog

PAD = 72
MIN_KEEP = 25


def run(in_path: str, out_path: str, keep: float, log_path: str) -> bool:
    jl = JobLog(log_path)
    log = jl.log
    try:
        t0 = time.time()
        # mono view: Bayer superpixel / RGB green plane (ser-trim.js:85-90)
        from ..io.ser import read_header

        channel = "G" if read_header(in_path).rgb else "mono"
        ser = SerReader(in_path, channel)
        h = ser.header
        usable = ser.frame_count
        sc = 2 if h.bayer else 1
        mw, mh = ser.width, ser.height
        log(f"trim {in_path}")
        log(f"  {h.raw_width}x{h.raw_height} x{usable} frames,"
            f" colorId {h.color_id}, keep {100 * keep:.0f}%")

        # ---- pass 1: rank quality + track lit bbox (decimated mono) ----
        q = np.empty(usable, dtype=np.float64)
        bx0, bx1, by0, by1 = mw, 0, mh, 0
        thr = -1.0
        for i in range(usable):
            mono = ser.read(i)
            q[i] = laplacian_sharpness(mono)
            # union lit bbox every 20th frame (drift coverage at low cost)
            if i % 20 == 0:
                if thr < 0:
                    thr = max(3.0 * float(np.median(mono)),
                              float(mono.max()) * 0.12)
                lit = mono[::3, ::3] > thr
                ys, xs = np.nonzero(lit)
                if xs.size:
                    bx0 = min(bx0, int(xs.min()) * 3)
                    bx1 = max(bx1, int(xs.max()) * 3)
                    by0 = min(by0, int(ys.min()) * 3)
                    by1 = max(by1, int(ys.max()) * 3)
            if (i + 1) % 500 == 0:
                log(f"  ranked {i + 1}/{usable}")
                jl.progress(i + 1, usable, "rank")

        # ---- ROI in full-res coords: pad for drift + processing margin ----
        W, H = h.raw_width, h.raw_height
        x0 = max(0, sc * bx0 - PAD)
        x1 = min(W, sc * bx1 + PAD)
        y0 = max(0, sc * by0 - PAD)
        y1 = min(H, sc * by1 + PAD)
        x0 &= ~1
        y0 &= ~1  # even-align: preserve CFA phase
        cw = min(W - x0, (x1 - x0 + 2) & ~1)
        ch = min(H - y0, (y1 - y0 + 2) & ~1)
        if bx1 <= bx0 or cw < 64 or ch < 64:
            x0, y0, cw, ch = 0, 0, W, H  # degenerate: keep full frame
            log("  WARNING: ROI detection degenerate — keeping full frame")
        log(f"  ROI {cw}x{ch} at ({x0},{y0}) — "
            f"{100 * (1 - cw * ch / (W * H)):.0f}% area discarded")

        n = min(usable, max(MIN_KEEP, int(np.ceil(keep * usable))))
        selected = sorted(np.argsort(-q)[:n].tolist())

        # ---- pass 2: write selected frames, cropped ----
        write_trimmed(in_path, out_path, selected, x0, y0, cw, ch)
        import os

        out_size = os.path.getsize(out_path)
        in_size = os.path.getsize(in_path)
        log(f"  kept {n}/{usable} frames, {out_size / 1e9:.2f} GB"
            f" (was {in_size / 1e9:.2f} GB), {time.time() - t0:.0f} s")
        log("=== TRIM OK ===")
        return True
    except Exception as e:  # noqa: BLE001 — job boundary, always log+fail
        log(f"*** TRIM FAILED: {e}")
        log(traceback.format_exc())
        return False
    finally:
        jl.close()
