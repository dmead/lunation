"""Lunar finishing chain — ports pjsr/lunar-finish.js.

Stage order, config schema, log protocol (PROGRESS k/9, === FINISH OK) and
artifact naming match the PJSR original. PI-process replacements live in
primitives.py; better-than decisions: skimage-free hand-rolled sRGB/CIELab
(no PI RGBWS), PCHIP curves, RL deconvolution with wavelet-regularized
corrections.
"""

import math
import os
import time
import traceback

import numpy as np

from ..core.fftreg import PhaseCorrelator
from ..core.kernels import gradient_magnitude
from ..core.stats import luminance_pi
from ..core.warp import translate
from ..io.xisf_io import read_xisf, write_xisf
from ..stack.logutil import JobLog
from .primitives import (curve, gaussian, histogram_transform, mtf, mtf_for,
                         lab01_to_rgb, rgb_to_lab01, rl_deconvolve,
                         starlet_sharpen)


def coarse_centroid(img: np.ndarray) -> tuple[float, float]:
    m = img > 0.15 * float(img.max())
    ys, xs = np.nonzero(m[::4, ::4])
    if not len(xs):
        return 0.0, 0.0
    return float(xs.mean()) * 4, float(ys.mean()) * 4


def masked_apply(orig: np.ndarray, processed: np.ndarray,
                 mask: np.ndarray | None) -> np.ndarray:
    if mask is None or mask.shape != orig.shape:
        return processed
    return orig * (1 - mask) + processed * mask


class Finisher:
    def __init__(self, cfg: dict, log, progress):
        self.cfg = cfg
        self.log = log
        self.progress = progress
        self.tx = [0.0, 0.0]  # min/max applied Tx
        self.ty = [0.0, 0.0]

    def _track(self, dx, dy):
        self.tx = [min(self.tx[0], -dx), max(self.tx[1], -dx)]
        self.ty = [min(self.ty[0], -dy), max(self.ty[1], -dy)]

    def _robust_shift(self, grad_pc, plain_pc, ref_c, img, label):
        """Gradient + plain phase correlation with centroid arbitration
        (lunar-finish.js:101-117)."""
        gdx, gdy = grad_pc.evaluate(img)
        pdx, pdy = plain_pc.evaluate(img)
        if math.hypot(gdx - pdx, gdy - pdy) <= 3:
            return gdx, gdy
        cx, cy = coarse_centroid(img)
        cdx, cdy = cx - ref_c[0], cy - ref_c[1]
        eg = math.hypot(gdx - cdx, gdy - cdy)
        ep = math.hypot(pdx - cdx, pdy - cdy)
        self.log(f"{label}: grad ({gdx:.1f},{gdy:.1f}) vs plain"
                 f" ({pdx:.1f},{pdy:.1f}) disagree; centroid delta"
                 f" ({cdx:.0f},{cdy:.0f}) -> using"
                 f" {'gradient' if eg <= ep else 'plain'}")
        return (gdx, gdy) if eg <= ep else (pdx, pdy)

    def open_merged(self, file_or_list, label) -> np.ndarray:
        sd = self.cfg["stacksDir"]
        lst = file_or_list if isinstance(file_or_list, list) else [file_or_list]
        img = read_xisf(f"{sd}/{lst[0]}")
        if img.ndim == 3:
            img = luminance_pi(img)
        if len(lst) > 1:
            plain = PhaseCorrelator(use_gradient=False)
            grad = PhaseCorrelator(use_gradient=True)
            plain.initialize(img)
            grad.initialize(img)
            ref_c = coarse_centroid(img)
            acc = img.astype(np.float64)
            for i, name in enumerate(lst[1:], start=2):
                w = read_xisf(f"{sd}/{name}")
                if w.ndim == 3:
                    w = luminance_pi(w)
                dx, dy = self._robust_shift(grad, plain, ref_c, w,
                                            f"{label}{i}")
                self._track(dx, dy)
                self.log(f"{label}{i} -> {label}1 shift dx={dx:.2f}"
                         f" dy={dy:.2f}")
                acc += translate(w, -dx, -dy)
            img = (acc / len(lst)).astype(np.float32)
            self.progress("merge")
            self.log(f"merged {len(lst)} {label} stacks")
        return img


def run(cfg: dict, config_path: str = "<inline>") -> bool:
    out_dir = cfg["outDir"]
    os.makedirs(out_dir, exist_ok=True)
    jl = JobLog(cfg.get("log") or f"{out_dir}/finish.log")
    prog_k = [0]

    def progress(stage):
        prog_k[0] += 1
        jl.log(f"PROGRESS {prog_k[0]}/9 {stage}")

    t0 = time.time()
    try:
        _run(cfg, jl.log, progress)
        jl.log(f"=== FINISH OK ({time.time() - t0:.1f} s) ===")
        return True
    except Exception as e:  # noqa: BLE001 — job boundary
        jl.log(f"*** FINISH FAILED: {e}")
        jl.log(traceback.format_exc())
        return False
    finally:
        jl.close()


def _run(cfg, log, progress):
    log(f"lunar-finish starting: {cfg['name']}")
    fin = Finisher(cfg, log, progress)

    # ---------------- open + merge, register channels to L ----------------
    L = fin.open_merged(cfg["L"], "ch_L")
    plain = PhaseCorrelator(use_gradient=False)
    grad = PhaseCorrelator(use_gradient=True)
    plain.initialize(L)
    grad.initialize(L)
    ref_c = coarse_centroid(L)
    names = {"R": cfg.get("R"), "G": cfg.get("G"), "B": cfg.get("B")}
    extras = list((cfg.get("extras") or {}).keys())
    for k, v in (cfg.get("extras") or {}).items():
        names[k] = v
    chans: dict[str, np.ndarray] = {}
    for ch, name in names.items():
        if not name:
            continue
        img = fin.open_merged(name, f"ch_{ch}")
        plain.initialize(L)
        grad.initialize(L)
        dx, dy = fin._robust_shift(grad, plain, ref_c, img, ch)
        fin._track(dx, dy)
        log(f"{ch} -> L shift dx={dx:.2f} dy={dy:.2f}")
        chans[ch] = translate(img, -dx, -dy)

    # ---------------- sky pedestal via isodata ----------------
    t_iso = float(L.mean())
    disk_mean = t_iso
    for _ in range(8):
        mk = L > t_iso
        f = float(mk.mean())
        if f <= 0 or f >= 1:
            break
        disk_mean = float(L[mk].mean())
        sky_mean = (float(L.mean()) - disk_mean * f) / (1 - f)
        t_new = 0.5 * (sky_mean + disk_mean)
        if abs(t_new - t_iso) < 1e-6:
            break
        t_iso = t_new
    sky_mask = L <= t_iso
    sky_frac = float(sky_mask.mean())
    sky_l = 0.0
    all_keys = ["L"] + list(chans.keys())

    def get(k):
        return L if k == "L" else chans[k]

    def put(k, v):
        nonlocal L
        if k == "L":
            L = v
        else:
            chans[k] = v

    if sky_frac > 0.02:
        for k in all_keys:
            img = get(k)
            bg = float(img[sky_mask].mean())
            if k == "L":
                sky_l = bg
            if bg > 0.002:
                put(k, np.clip(img - bg, 0, 1))
            log(f"sky pedestal ch_{k}: {bg:.5f}"
                + (" (subtracted)" if bg > 0.002 else " (negligible)"))

    # ---------------- veiling-glare (haze) subtraction ----------------
    if cfg.get("dehaze"):
        k_d = cfg["dehaze"] if isinstance(cfg["dehaze"], (int, float)) \
            and cfg["dehaze"] is not True else 0.55
        sigma_d = round(0.08 * max(L.shape))
        for k in all_keys:
            img = get(k)
            put(k, np.clip(img - k_d * gaussian(img, sigma_d), 0, 1))
            log(f"dehaze ch_{k}: subtracted {k_d} x blur(sigma {sigma_d}px)")

    thr = max(0.15 * (disk_mean - sky_l), 0.01)
    log(f"auto disk threshold: {thr:.4f} (isodata split {t_iso:.4f},"
        f" disk mean {disk_mean:.4f})")

    # ---------------- smooth luminance sharpening mask ----------------
    sharp_mask = None
    if cfg.get("deconvolve") or cfg.get("sharpen") is not False:
        m = gaussian(L, 24)
        rng = float(m.max()) - float(m.min())
        m = (m - m.min()) / max(rng, 1e-6)
        sharp_mask = np.sqrt(m).astype(np.float32)
        log("smooth luminance sharpening mask built")

    # ---------------- RL deconvolution on L (linear) ----------------
    if cfg.get("deconvolve"):
        dc = cfg["deconvolve"]
        kw = dict(psf_sigma=dc.get("psfSigma", 1.5),
                  iterations=dc.get("iterations", 25),
                  dark_dering=dc.get("deringingDark", 0.1))
        L = masked_apply(L, rl_deconvolve(L, **kw), sharp_mask)
        progress("deconvolve")
        log(f"RL deconvolution on L: psfSigma={kw['psf_sigma']}"
            f" iters={kw['iterations']} (disk-masked)")
        if extras and dc.get("extras") is not False:
            for ch in extras:
                if ch in chans:
                    chans[ch] = masked_apply(
                        chans[ch], rl_deconvolve(chans[ch], **kw), sharp_mask)

    # ---------------- per-channel illumination flattening ----------------
    if cfg.get("channelFlatten") is not False:
        h, w = L.shape
        step = 8
        lum_floor = 3 * thr
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        l_s = L[::step, ::step]
        for ch in list(chans.keys()):
            img = chans[ch]
            if img.shape != L.shape:
                continue
            r = img[::step, ::step] / np.maximum(l_s, 1e-9)
            ok = (l_s >= lum_floor) & np.isfinite(r) & (r > 0) & (r <= 10)
            n = int(ok.sum())
            if n < 500:
                log(f"{ch}: flatten skipped (only {n} disk samples)")
                continue
            A = np.column_stack([xs[ok], ys[ok], np.ones(n)])
            sol, *_ = np.linalg.lstsq(A, r[ok], rcond=None)
            a, b, c = (float(v) for v in sol)
            mx, my = float(xs[ok].mean()), float(ys[ok].mean())
            p_mean = a * mx + b * my + c
            span = max(abs(a) * w, abs(b) * h) / max(p_mean, 1e-9)
            if span < 0.01:
                log(f"{ch}: flatten skipped (tilt {100 * span:.2f}%"
                    " — negligible)")
                continue
            yy, xx = np.mgrid[0:h, 0:w]
            plane = a * xx + b * yy + c
            chans[ch] = np.clip(img * p_mean / np.maximum(plane, 1e-9),
                                0, 1).astype(np.float32)
            log(f"{ch}: flattened {100 * span:.1f}% illumination tilt vs L")

    # ---------------- disk mask + white balance (linear) ----------------
    disk = L > thr
    log(f"disk fraction: {100 * float(disk.mean()):.1f}%")
    if cfg.get("whiteBalance") is not False and all(
            k in chans for k in "RGB"):
        m_r = float(chans["R"][disk].mean())
        m_g = float(chans["G"][disk].mean())
        m_b = float(chans["B"][disk].mean())
        log(f"disk means R={m_r:.5f} G={m_g:.5f} B={m_b:.5f}")
        warm = cfg.get("warmth", 1.0)
        chans["R"] = np.clip(chans["R"] * (warm * m_g / m_r), 0, 1)
        chans["B"] = np.clip(chans["B"] * ((m_g / m_b) / warm), 0, 1)
        progress("whitebalance")
        log(f"white balance: R*{warm * m_g / m_r:.4f}"
            f" B*{(m_g / m_b) / warm:.4f}"
            + (f" (warmth {warm})" if warm != 1 else ""))

    # ---------------- joint soft stretch ----------------
    st = cfg.get("stretch") or {}
    target_median = st.get("targetMedian", 0.42)
    disk_med = float(L[disk].mean())
    mad = float(np.median(np.abs(L - np.median(L))))
    shadow = max(0.0, float(np.median(L))
                 - st.get("shadowSigma", 2.0) * 1.4826 * mad)
    x_anchor = (disk_med - shadow) / (1 - shadow)
    mid = mtf_for(x_anchor, target_median)
    progress("stretch")
    log(f"stretch: shadow={shadow:.5f} diskMean={disk_med:.5f}"
        f" -> midtones={mid:.5f}")
    L = histogram_transform(L, shadow, mid)
    for ch in chans:
        chans[ch] = histogram_transform(chans[ch], shadow, mid)

    # ---------------- MLT wavelet sharpening on L ----------------
    if cfg.get("sharpen") is not False:
        sh = cfg.get("sharpen") or {}
        biases = sh.get("biases", [0.0, 0.05, 0.08, 0.03])
        dering = sh.get("deringing", True) is not False
        L = masked_apply(L, starlet_sharpen(L, biases, dering), sharp_mask)
        progress("sharpen")
        log(f"MLT sharpening on L: biases [{', '.join(map(str, biases))}]"
            " (disk-masked)")
        for ch in extras:
            if ch in chans and chans[ch].shape == L.shape:
                chans[ch] = masked_apply(
                    chans[ch], starlet_sharpen(chans[ch], biases, dering),
                    sharp_mask)

    # ---------------- RGB combine + LRGB + chroma work ----------------
    final = None
    if all(k in chans for k in "RGB"):
        rgb = np.stack([chans["R"], chans["G"], chans["B"]], axis=-1)
        progress("combine")
        log("RGB combined")
        lp = cfg.get("lrgb") or {}
        m_l = lp.get("mL", 0.5)
        m_c = lp.get("mc", 0.35)
        lab_l, lab_a, lab_b = rgb_to_lab01(rgb)
        # LRGB: lightness replaced by the (MTF-balanced) L image; chroma
        # deviations balanced by mc (mc < 0.5 boosts, > 0.5 mutes)
        lab_l = mtf(m_l, L)
        ca = lab_a - 0.5
        cb = lab_b - 0.5
        sat = np.hypot(ca, cb)
        sat_t = mtf(m_c, np.clip(sat * 2, 0, 1)) / 2  # normalize ~[0,0.5]
        gain = sat_t / np.maximum(sat, 1e-6)
        lab_a = 0.5 + ca * gain
        lab_b = 0.5 + cb * gain
        log(f"LRGB combined (mL={m_l} mc={m_c})")

        if cfg.get("chromaSmooth") or cfg.get("chromaBoost"):
            boost = cfg.get("chromaBoost") or 1
            ba, bb = (boost if isinstance(boost, list) else (boost, boost))
            if cfg.get("chromaSmooth"):
                lab_a = gaussian(lab_a, cfg["chromaSmooth"])
                lab_b = gaussian(lab_b, cfg["chromaSmooth"])
            lab_a = np.clip(0.5 + (lab_a - 0.5) * ba, 0, 1)
            lab_b = np.clip(0.5 + (lab_b - 0.5) * bb, 0, 1)
            log(f"chroma: smooth sigma {cfg.get('chromaSmooth', 0)},"
                f" boost x{boost}")

            if cfg.get("edgeDesat") != 0:
                strength = cfg.get("edgeDesat", 0.85)
                g = gradient_magnitude(lab_l)
                g = cv2_box(g, 5)
                rng = float(g.max()) - float(g.min())
                g = (g - g.min()) / max(rng, 1e-9)
                g = np.clip(g**1.5 * strength, 0, 1)
                lab_a = lab_a - (lab_a - 0.5) * g
                lab_b = lab_b - (lab_b - 0.5) * g
                log(f"edge-weighted chroma desat (strength {strength})")

            if cfg.get("limbDesat") is not False:
                sky = (~(lab_l > 0.15)).astype(np.float32)
                sky = (cv2_box(sky, 17) > 0.005).astype(np.float32)
                bright = (lab_l > 0.12).astype(np.float32)
                ring = cv2_box(sky * bright, 17)
                lab_a = lab_a - (lab_a - 0.5) * ring
                lab_b = lab_b - (lab_b - 0.5) * ring
                log("limb ring desaturated (bright-limb only)")

        final = lab01_to_rgb(lab_l, lab_a, lab_b)

        if cfg.get("satCurve"):
            import cv2 as _cv2

            hsv = _cv2.cvtColor(final, _cv2.COLOR_RGB2HSV)
            hsv[..., 1] = curve(hsv[..., 1], cfg["satCurve"])
            final = _cv2.cvtColor(hsv, _cv2.COLOR_HSV2RGB)
            log("saturation curve applied")
        if cfg.get("contrastCurve"):
            final = curve(final, cfg["contrastCurve"])
            log("contrast curve applied")
    else:
        log("no RGB channels — luminance-only output")
        final = L

    # ---------------- crop registration borders (directional) ----------------
    crop_l = math.ceil(max(0, fin.tx[1])) + 2
    crop_r = math.ceil(max(0, -fin.tx[0])) + 2
    crop_t = math.ceil(max(0, fin.ty[1])) + 2
    crop_b = math.ceil(max(0, -fin.ty[0])) + 2

    def crop(a):
        return a[crop_t : a.shape[0] - crop_b, crop_l : a.shape[1] - crop_r]

    final = crop(final)
    if final is not L:
        L = crop(L)
    for ch in extras:
        if ch in chans:
            chans[ch] = crop(chans[ch])
    log(f"cropped registration margins L={crop_l} T={crop_t} R={crop_r}"
        f" B={crop_b} px")

    # ---------------- save ----------------
    out_base = f"{cfg['outDir']}/{cfg['name']}"
    write_xisf(f"{out_base}.xisf", np.clip(final, 0, 1))
    progress("save")
    log(f"saved {out_base}.xisf")
    if final is not L:
        write_xisf(f"{out_base}_L.xisf", np.clip(L, 0, 1))
        log(f"saved {out_base}_L.xisf")
    for ch in extras:
        if ch in chans:
            write_xisf(f"{out_base}_{ch}.xisf", np.clip(chans[ch], 0, 1))
            log(f"saved {out_base}_{ch}.xisf")


def cv2_box(img: np.ndarray, k: int) -> np.ndarray:
    import cv2

    return cv2.boxFilter(img, -1, (k, k))
