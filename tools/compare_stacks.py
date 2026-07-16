"""Better-than validation harness: PI baseline stack vs Python port stack.

Registers the pair, computes disk-detail / sky-noise / level metrics plus
report-side agreement (frame-selection overlap, shifts), and renders a
side-by-side sheet with the full disk and the highest-detail crops.

  uv run python tools/compare_stacks.py <pi_stack.xisf> <py_stack.xisf> \
      [--sheet cmp.png] [--pi-report R.json --py-report R.json]

Pass = detail >= PI at noise <= PI and no visual regressions on the sheet
(never trust a registration without reading the rendered frames).
"""

import argparse
import json

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunation.core.fftreg import PhaseCorrelator
from lunation.core.kernels import LAPLACIAN
from lunation.core.warp import translate
from lunation.io.images import read_image
from lunation.core.stats import luminance


def lap_std(img, mask=None):
    r = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, LAPLACIAN)
    return float(r[mask].std()) if mask is not None else float(r.std())


def disk_mask(img):
    med = float(np.median(img))
    thr = max(3.0 * med, 0.08 * float(img.max()))
    m = (img > thr).astype(np.uint8)
    m = cv2.erode(m, np.ones((25, 25), np.uint8))  # stay off the limb
    return m.astype(bool)


def sky_mask(img):
    thr = max(2.0 * float(np.median(img)), 0.02 * float(img.max()))
    m = (img < thr).astype(np.uint8)
    m = cv2.erode(m, np.ones((25, 25), np.uint8))  # stay off the limb
    return m.astype(bool)


def auto_stretch(img, ref_med, ref_mad):
    """Simple MTF display stretch shared by both panels."""
    shadows = max(0.0, ref_med - 2.8 * ref_mad)
    m = 0.25
    x = np.clip((img - shadows) / max(1e-6, 1.0 - shadows), 0, 1)
    y = (m - 1) * x / ((2 * m - 1) * x - m)
    return np.clip(y, 0, 1)


def top_detail_tiles(img, mask, tile=384, n=3):
    r = np.abs(cv2.filter2D(img, cv2.CV_32F, LAPLACIAN))
    r[~mask] = 0
    tiles = []
    h, w = img.shape
    for y in range(0, h - tile, tile // 2):
        for x in range(0, w - tile, tile // 2):
            tiles.append((float(r[y : y + tile, x : x + tile].mean()), x, y))
    tiles.sort(reverse=True)
    picked = []
    for s, x, y in tiles:
        if all(abs(x - px) > tile or abs(y - py) > tile for _, px, py in picked):
            picked.append((s, x, y))
        if len(picked) == n:
            break
    return [(x, y) for _, x, y in picked]


def label(img8, text):
    cv2.putText(img8, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2,
                cv2.LINE_AA)
    return img8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pi")
    ap.add_argument("py")
    ap.add_argument("--sheet")
    ap.add_argument("--pi-report")
    ap.add_argument("--py-report")
    ap.add_argument("--tile", type=int, default=384)
    a = ap.parse_args()

    pi = luminance(read_image(a.pi))
    py = luminance(read_image(a.py))
    if pi.shape != py.shape:
        print(f"NOTE: shapes differ pi={pi.shape} py={py.shape}; "
              "cropping to common")
        h = min(pi.shape[0], py.shape[0])
        w = min(pi.shape[1], py.shape[1])
        pi, py = pi[:h, :w], py[:h, :w]

    pc = PhaseCorrelator()
    pc.initialize(pi)
    dx, dy = pc.evaluate(py)
    py_r = translate(py, -dx, -dy)
    print(f"registration: py is offset ({dx:+.2f}, {dy:+.2f}) px vs pi")

    # exclude the registration's border fill — a fill/sky step inside a mask
    # reads as huge fake "noise" (found on the H_2025-03-04_osc validation)
    covered = translate(np.ones_like(py), -dx, -dy) > 0.5
    covered = cv2.erode(covered.astype(np.uint8),
                        np.ones((9, 9), np.uint8)).astype(bool)
    dm = disk_mask(pi) & covered
    sm = sky_mask(pi) & covered

    # radiometric normalization: different tools scale output levels
    # differently (e.g. PSS ~2x); match disk means so detail/noise compare
    gain = float(pi[dm].mean()) / max(1e-9, float(py_r[dm].mean()))
    if abs(gain - 1.0) > 0.02:
        print(f"level normalization: py x{gain:.4f} (disk-mean match)")
    py_r = py_r * gain
    rows = [
        ("disk detail (lap std)", lap_std(pi, dm), lap_std(py_r, dm), "higher"),
        ("sky noise sigma", float(pi[sm].std()), float(py_r[sm].std()), "lower"),
        ("disk mean level", float(pi[dm].mean()), float(py_r[dm].mean()), "~equal"),
        ("p99.9 highlight", float(np.percentile(pi[dm], 99.9)),
         float(np.percentile(py_r[dm], 99.9)), "~equal"),
    ]
    print(f"\n{'metric':<26}{'PI':>12}{'PY':>12}   want")
    for name, vpi, vpy, want in rows:
        mark = ""
        if want == "higher":
            mark = "PY WINS" if vpy > vpi else "pi wins"
        elif want == "lower":
            mark = "PY WINS" if vpy < vpi else "pi wins"
        print(f"{name:<26}{vpi:>12.5g}{vpy:>12.5g}   {want:<7} {mark}")

    if a.pi_report and a.py_report:
        rpi = json.load(open(a.pi_report))
        rpy = json.load(open(a.py_report))
        n = rpi["stacked"]
        top_pi = set(np.argsort(-np.array(rpi["quality"]))[:n].tolist())
        top_py = set(np.argsort(-np.array(rpy["quality"]))[:n].tolist())
        ov = len(top_pi & top_py) / max(1, n)
        print(f"\nselection overlap: {100 * ov:.1f}% of {n}"
              f" | refIndex pi={rpi['refIndex']} py={rpy['refIndex']}"
              f" | maxShift pi={rpi['maxShift']:.2f} py={rpy['maxShift']:.2f}")

    if a.sheet:
        med, mad = float(np.median(pi)), float(np.median(np.abs(pi - np.median(pi))))
        panels = []
        t = a.tile
        crops = top_detail_tiles(pi, dm, t)
        for name, img in (("PI", pi), ("PY", py_r)):
            s = auto_stretch(img, med, mad)
            full = cv2.resize(s, (t, round(t * s.shape[0] / s.shape[1])))
            row = [label((full * 255).astype(np.uint8), f"{name} full")]
            for k, (x, y) in enumerate(crops):
                c = (s[y : y + t, x : x + t] * 255).astype(np.uint8)
                row.append(label(c, f"{name} crop{k + 1}"))
            hmax = max(p.shape[0] for p in row)
            row = [np.pad(p, ((0, hmax - p.shape[0]), (0, 0))) for p in row]
            panels.append(np.hstack(row))
        wmax = max(p.shape[1] for p in panels)
        panels = [np.pad(p, ((0, 0), (0, wmax - p.shape[1]))) for p in panels]
        sheet = np.vstack([panels[0], np.full((8, wmax), 128, np.uint8),
                           panels[1]])
        cv2.imwrite(a.sheet, sheet)
        print(f"\nsheet: {a.sheet}  (crops at {crops})")


if __name__ == "__main__":
    main()
