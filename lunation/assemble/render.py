"""Lunation frame renderer — ports the gif-frames.js main flow (747-1340).

Registration model (hard-won; see gif-frames.js header): disk-centered,
common-scale frames carry one rotation + one parity unknown. Rotation is
seeded by physics (lunar age -> waxing/waning -> lit limb maps to east/west,
anchor rotated so east=+x, north up), NCC-refined against phase neighbors in
both parities with a multi-neighbor vote and median rotation, then physics
snap-back on rim-residual disagreement. Prep-normalized ingests are
self-detected by exact-canvas geometry and never anchor or serve as chain
targets.
"""

import math
import os
import time
import traceback

import cv2
import numpy as np

from ..core.stats import luminance_pi as luminance
from ..core.warp import resample, rotate
from ..io.images import read_image, write_png
from ..stack.logutil import JobLog
from .disk import analyze_disk
from .landmarks import maria_blobs
from .quality import measure_quality
from .register import best_rotation_ncc, flip_h, norm180, rim_on_small

SMALL = 512
PREP_R_FRAC = 979 / 2300  # prep-finished working radius fraction of canvas
SYNODIC = 29.530588


def phase_name(age: float) -> str:
    if not math.isfinite(age):
        return "?"
    if age < 1.0 or age >= 28.5:
        return "new"
    if age < 6.4:
        return "waxing crescent"
    if age < 8.4:
        return "first quarter"
    if age < 13.8:
        return "waxing gibbous"
    if age < 15.8:
        return "full"
    if age < 21.1:
        return "waning gibbous"
    if age < 23.1:
        return "last quarter"
    return "waning crescent"


def center_on_canvas(img: np.ndarray, disk, canvas: int) -> np.ndarray:
    """Paste img disk-centered onto a square float canvas (gif-frames.js:647)."""
    ch = () if img.ndim == 2 else (img.shape[2],)
    out = np.zeros((canvas, canvas, *ch), dtype=np.float32)
    icx, icy = round(disk.cx), round(disk.cy)
    half = canvas >> 1
    sx0, sy0 = max(0, icx - half), max(0, icy - half)
    sx1 = min(img.shape[1], icx + half)
    sy1 = min(img.shape[0], icy + half)
    dx0, dy0 = half - (icx - sx0), half - (icy - sy0)
    out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = \
        img[sy0:sy1, sx0:sx1]
    return out


def center_rescale(img: np.ndarray, s: float, canvas: int) -> np.ndarray:
    """Resample about center, then center-crop/pad back to canvas
    (gif-frames.js:960-968 cropBy semantics)."""
    up = resample(img, s, "bicubic")
    d = up.shape[1] - canvas
    l = d >> 1
    if d >= 0:
        return np.ascontiguousarray(up[l : l + canvas, l : l + canvas])
    pad = [( -l, -(d - l)), (-l, -(d - l))] + ([(0, 0)] if up.ndim == 3 else [])
    return np.pad(up, pad)


def small_lum(img: np.ndarray, canvas: int) -> np.ndarray:
    lum = luminance(img)
    return cv2.resize(lum, (SMALL, SMALL), interpolation=cv2.INTER_LINEAR)


def desaturate_fringes(frame: np.ndarray,
                       disk_r: float) -> tuple[np.ndarray, int]:
    """Clamp channel-registration fringes toward the LOCAL ambient chroma.

    Style-independent discriminator (ground rule: no constants tuned to
    one imager's data — other rigs and saturated processing styles must pass
    through untouched): genuine color, however strong, is spatially SMOOTH;
    registration fringes are high-frequency chroma outliers. We compare each
    pixel's chroma to the local ambient chroma field (Gaussian at a scale
    tied to the disk radius) and act only on residuals beyond a robust-sigma
    threshold of the frame's OWN residual distribution. Offenders have their
    color deviation scaled back to the ambient level — never to gray — so a
    mineral moon keeps its saturation while R/G/B streaks vanish on any rig.
    Smooth sigma-ramped blend, never a binarized mask."""
    Z0, Z1 = 8.0, 16.0  # robust-sigma ramp (units of the frame's own noise)
    if frame.ndim != 3:
        return frame, 0
    lum = frame.mean(axis=2)
    chroma = frame.max(axis=2) - frame.min(axis=2)
    # lit terrain, self-normalized (finals may be stretched differently)
    lit = lum > max(0.05, 0.15 * float(np.percentile(lum, 99)))
    if not np.any(lit):
        return frame, 0
    # SATURATION field (chroma/L): raw chroma inherits albedo texture on a
    # tinted frame (chroma = tint x luminance), drowning fringes in its own
    # variance; saturation cancels the texture and stays smooth for any
    # genuine style while fringes still spike
    sat = np.where(lit, chroma / np.maximum(lum, 1e-6), 0.0).astype(np.float32)
    sigma_px = max(5.0, disk_r / 12.0)

    def masked_ambient(weight):
        # normalized (masked) blur: a plain Gaussian dilutes the ambient
        # level toward zero at the limb and fakes an edge ring
        num = cv2.GaussianBlur(sat * weight, (0, 0), sigma_px)
        den = cv2.GaussianBlur(weight, (0, 0), sigma_px)
        return num / np.maximum(den, 1e-3)

    def zscore(ambient):
        resid = sat - ambient
        r_lit = resid[lit]
        mad = float(np.median(np.abs(r_lit - np.median(r_lit))))
        return resid / max(1.4826 * mad, 1e-4)

    # single pass: catches sharp registration streaks on any style. A
    # mid-scale chroma WASH is damaged source data, not a render problem —
    # chasing it here risks eating genuine styles (tested and rejected
    # 2026-07-16); such finals get fixed at the source or dropped.
    litf = lit.astype(np.float32)
    ambient = masked_ambient(litf)
    z = zscore(ambient)
    w = np.clip((z - Z0) / (Z1 - Z0), 0.0, 1.0) * lit
    n = int((w > 0).sum())
    if n == 0:
        return frame, 0
    gray = lum[..., None]
    # target: same hue direction, saturation reduced to the ambient level
    target_chroma = np.maximum(ambient, 0.0) * lum
    scale = (target_chroma / np.maximum(chroma, 1e-6))[..., None]
    target = gray + (frame - gray) * np.clip(scale, 0.0, 1.0)
    w3 = w[..., None]
    return frame * (1 - w3) + target * w3, n


def make_sky_mask(canvas: int, ref_r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    c = canvas / 2
    d = np.hypot(xx - c, yy - c)
    c_r = ref_r * 1.05
    fade = ref_r * 0.02
    return np.clip(1 - (d - c_r) / fade, 0, 1).astype(np.float32)


def run(out_dir: str, canvas: int, out_px: int, entries: list[tuple[str, float]],
        explicit_order: bool = True) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    jl = JobLog(os.path.join(out_dir, "gif-frames.log"))
    glog = jl.log
    t0 = time.time()
    try:
        _run(out_dir, canvas, out_px, entries, explicit_order, jl)
        glog(f"=== GIF OK ({time.time() - t0:.1f} s) ===")
        return True
    except Exception as e:  # noqa: BLE001 — job boundary
        glog(f"*** GIF FAILED: {e}")
        glog(traceback.format_exc())
        return False
    finally:
        jl.close()


def _run(out_dir, canvas, out_px, entries, explicit_order, jl):
    glog = jl.log
    files = [e[0] for e in entries]
    ages = [e[1] for e in entries]
    glog(f"gif-frames starting: {len(files)} inputs, outPx {out_px}")

    # ---- pass 1: disk analysis for all frames ----
    info: list[str] = []
    disks: list = []
    prepped: list[bool] = []
    for idx, path in enumerate(files):
        try:
            img = read_image(path)
            is_prepped = (img.shape[0] == canvas and img.shape[1] == canvas)
            prepped.append(is_prepped)
            force = ({"cx": img.shape[1] / 2, "cy": img.shape[0] / 2,
                      "r": round(PREP_R_FRAC * canvas)} if is_prepped else None)
            d = analyze_disk(img, force)
            d.q = measure_quality(img, d)
            q_line = (f"quality {_name(path)} detail {d.q['detail']:.4f}"
                      f" greenX {d.q['greenX']:.4f} misreg {d.q['misreg']:.1f}")
            info.append(q_line)
            disks.append(d)
            glog(f"PROGRESS {idx + 1}/{len(files)} analyze")
        except Exception as e:  # noqa: BLE001 — drop the frame, not the run
            glog(f"ANALYZE FAILED {path}: {e}")
            disks.append(None)
            prepped.append(False)

    # ---- measured junk gates (thresholds sit mid-gap between measured
    # junk and keeper clusters across the 38-input survey, 2026-07-14) ----
    ds = sorted(d.q["detail"] for d in disks if d and d.q)
    detail_median = ds[len(ds) >> 1] if ds else 0.0
    for i, d in enumerate(disks):
        if not d or not d.q:
            continue
        q, why = d.q, None
        if q["detail"] < 0.065:
            why = f"detail {q['detail']:.4f} below floor 0.065 (soft/upscaled mush)"
        elif abs(q["greenX"]) > 0.02:
            why = f"greenX {q['greenX']:.4f} beyond 0.02 (color failure)"
        elif q["misreg"] >= 2:
            why = f"channel misregistration {q['misreg']:.1f} std px (>=2)"
        if why:
            info.append(f"EXCLUDED {_name(files[i])} — {why}")
            disks[i] = None

    if out_px == 0:  # measure-only: gate calibration, no render
        _write(out_dir, "frames-debug.txt", info)
        glog("MEASURE-ONLY done")
        return

    # ---- processing order: anchor at the fullest REAL frame, walk outward ----
    ref_idx_of: dict[int, int] = {}
    if explicit_order:
        F = -1
        for i, d in enumerate(disks):
            if d and not prepped[i] and (F < 0 or d.lit > disks[F].lit):
                F = i
        if F < 0:
            for i, d in enumerate(disks):
                if d and (F < 0 or d.lit > disks[F].lit):
                    F = i
        if F < 0:
            raise RuntimeError("no analyzable inputs")
        order = [F]
        for i in range(F + 1, len(files)):
            order.append(i)
            ref_idx_of[i] = i - 1
        for i in range(F - 1, -1, -1):
            order.append(i)
            ref_idx_of[i] = i + 1
    else:
        order = sorted((i for i, d in enumerate(disks) if d),
                       key=lambda i: -disks[i].lit)
        for k in range(1, len(order)):
            ref_idx_of[order[k]] = order[k - 1]
    ref_r = disks[order[0]].r

    # ---- pass 2: chain registration + render ----
    anchor_lum = None
    rendered_lum: dict[int, np.ndarray] = {}
    rendered_score: dict[int, float] = {}
    rows = []
    sky_mask = make_sky_mask(canvas, ref_r)

    for k, idx in enumerate(order):
        if not disks[idx]:
            info.append(f"EXCLUDED {_name(files[idx])} — unreadable/analysis failed")
            continue
        disk = disks[idx]
        if disk.borderRun >= 40:
            info.append(
                f"EXCLUDED {_name(files[idx])} — lit terrain clipped at border"
                f" (run {disk.borderRun}px; contain lit {disk.litContain:.3f}"
                f" dark {disk.darkContain:.3f})")
            continue
        img = read_image(files[idx])
        frame = center_on_canvas(img, disk, canvas)
        s_norm = ref_r / disk.r
        if abs(s_norm - 1) > 0.005:
            frame = center_rescale(frame, s_norm, canvas)

        # physics: age says waxing (lit limb = lunar east) or waning (= west);
        # target east at +x so north is up. Rendered lit direction after
        # rotate(rot) is theta - rot, so rot_seed = theta - T.
        age = ages[idx]
        waxing = not (age >= SYNODIC / 2)  # NaN -> waxing default
        T = 0.0 if waxing else 180.0
        theta_deg = norm180(math.degrees(disk.thetaRim))
        rot_deg, score, mirrored = 0.0, 1.0, False

        if k > 0:
            # register against <=3 already-rendered phase neighbors
            t_list, t_names = [], []
            t_idx = ref_idx_of.get(idx)
            while t_idx is not None and len(t_list) < 3:
                if t_idx in rendered_lum and rendered_score.get(t_idx, 0) >= 0.60:
                    t_list.append(rendered_lum[t_idx])
                    t_names.append(_name(files[t_idx]))
                t_idx = ref_idx_of.get(t_idx)
            if not t_list:
                t_list.append(anchor_lum)
                t_names.append("anchor")

            f = small_lum(frame, canvas)
            m = (f > 0.10).astype(np.float32)
            f_m, m_m = flip_h(f), flip_h(m)
            r_is, r_ms, vote_margin = [], [], 0.0
            for tgt in t_list:
                if disk.rimDark >= 0.30:
                    mode = "seeded"
                    v_i = best_rotation_ncc(f, m, tgt, norm180(theta_deg - T), 20)
                    v_m = best_rotation_ncc(f_m, m_m, tgt,
                                            norm180((180 - theta_deg) - T), 20)
                else:
                    mode = "full-circle"
                    v_i = best_rotation_ncc(f, m, tgt)
                    v_m = best_rotation_ncc(f_m, m_m, tgt)
                r_is.append(v_i)
                r_ms.append(v_m)
                vote_margin += v_m["score"] - v_i["score"]
            r_i, r_m = r_is[0], r_ms[0]
            # parity flips only on a decisive aggregate margin
            mirrored = vote_margin > 0.03 * len(t_list)
            r = r_m if mirrored else r_i
            est = [v["angle"] for v in (r_ms if mirrored else r_is)]
            if len(est) >= 3:
                rel = sorted(norm180(a - est[0]) for a in est)
                rot_deg = norm180(est[0] + rel[len(rel) >> 1])
            else:
                rot_deg = r["angle"]
            score = r["score"]

            # physics snap-back: rotated rim must show its lit limb at T
            snap_note = ""
            rs = rotate(f_m if mirrored else f, np.pi * rot_deg / 180,
                        SMALL / 2, SMALL / 2)
            rim = rim_on_small(rs, ref_r * SMALL / canvas)
            if rim["darkFrac"] >= 0.30:
                res = norm180(rim["litDeg"] - T)
                if 15 < abs(res) <= 30:
                    rot_deg += res
                    snap_note = f" SNAP {res:.1f}"
                elif abs(res) > 30:
                    snap_note = f" rimRes {res:.1f} (uncorrected)"

            info.append(
                f"  {_name(files[idx])} vs {t_names[0]}: age {age:.1f}d"
                f" {'wax' if waxing else 'wane'} rimDark {disk.rimDark:.2f}"
                f" {mode} id {r_i['angle']:.1f}/{r_i['score']:.3f}"
                f" mir {r_m['angle']:.1f}/{r_m['score']:.3f}"
                f" vote {vote_margin:.3f}x{len(t_list)}"
                f" -> {'MIRROR ' if mirrored else ''}{rot_deg:.2f}{snap_note}")
            if score < 0.55:
                if (disk.rimDark >= 0.30 and disk.q
                        and disk.q["detail"] >= 0.6 * detail_median):
                    # physics seed alone is good to ~±10°; keep, never target
                    rot_deg = norm180(theta_deg - T)
                    mirrored = False
                    info.append(f"  {_name(files[idx])} — link {score:.3f}"
                                f" unusable, PHYSICS-ONLY rot {rot_deg:.2f}")
                else:
                    info.append(f"EXCLUDED {_name(files[idx])} — anchor"
                                f" correlation {score:.3f} (unreliable registration)")
                    continue
            if mirrored:
                frame = flip_h(frame)
        elif disk.rimDark >= 0.30:
            # anchor with a true terminator: its own axis fixes orientation
            rot_deg = norm180(theta_deg - T)
            info.append(f"anchor(physics) {_name(files[idx])}: age {age:.1f}d"
                        f" {'wax' if waxing else 'wane'} theta {theta_deg:.1f}"
                        f" -> rot {rot_deg:.2f}")
        else:
            # near-full anchor: orient by NCC against the richest frame with a
            # REAL terminator, preferring the anchor's own rig (radius ±8%)
            o_idx = -1
            for pass_ in range(2):
                if o_idx >= 0:
                    break
                for i, d in enumerate(disks):
                    if (i != idx and d and not prepped[i]
                            and d.rimDark >= 0.30 and d.borderRun < 40
                            and (pass_ > 0 or abs(d.r - disk.r) / disk.r < 0.08)
                            and (o_idx < 0 or d.lit > disks[o_idx].lit)):
                        o_idx = i
            if o_idx < 0:
                rot_deg = norm180(theta_deg - T)
                info.append(f"anchor(fallback) {_name(files[idx])}: rot {rot_deg:.2f}")
            else:
                o_img = read_image(files[o_idx])
                o_can = center_on_canvas(o_img, disks[o_idx], canvas)
                o_norm = ref_r / disks[o_idx].r
                if abs(o_norm - 1) > 0.005:
                    o_can = center_rescale(o_can, o_norm, canvas)
                o_wax = not (ages[o_idx] >= SYNODIC / 2)
                o_rot = norm180(norm180(math.degrees(disks[o_idx].thetaRim))
                                - (0 if o_wax else 180))
                o_can = rotate(luminance(o_can), np.pi * o_rot / 180,
                               canvas / 2, canvas / 2)
                o_lum = cv2.resize(o_can, (SMALL, SMALL),
                                   interpolation=cv2.INTER_LINEAR)
                a_l = small_lum(frame, canvas)
                a_m = (a_l > 0.10).astype(np.float32)
                r_a = best_rotation_ncc(a_l, a_m, o_lum)
                r_am = best_rotation_ncc(flip_h(a_l), flip_h(a_m), o_lum)
                mirrored = r_am["score"] > r_a["score"] + 0.03
                r_best = r_am if mirrored else r_a
                rot_deg, score = r_best["angle"], r_best["score"]
                info.append(
                    f"anchor(NCC->{_name(files[o_idx])} rot {o_rot:.1f})"
                    f" {_name(files[idx])}: id {r_a['angle']:.1f}/{r_a['score']:.3f}"
                    f" mir {r_am['angle']:.1f}/{r_am['score']:.3f}"
                    f" -> {'MIRROR ' if mirrored else ''}{rot_deg:.2f}")
                if mirrored:
                    frame = flip_h(frame)

        if abs(rot_deg) > 0.2:
            frame = rotate(frame, np.pi * rot_deg / 180, canvas / 2, canvas / 2)

        # cache rotated small luminance; ingests never serve as chain targets
        rl = small_lum(frame, canvas)
        rendered_lum[idx] = rl
        rendered_score[idx] = 0.0 if prepped[idx] else (1.0 if k == 0 else score)
        if k == 0:
            anchor_lum = rl
            maria_blobs(rl, ref_r * SMALL / canvas)  # advisory only

        # chroma sanitizer: high-frequency chroma outliers are registration
        # fringes on any rig; smooth color of any strength passes through
        frame, n_desat = desaturate_fringes(frame, ref_r)
        if n_desat:
            info.append(f"  {_name(files[idx])}: desaturated {n_desat}"
                        " chroma-fringe px (adaptive)")

        # black sky beyond the disk, then final resample
        frame = frame * (sky_mask if frame.ndim == 2 else sky_mask[..., None])
        out = resample(frame, out_px / canvas, "mitchell")
        name = _name(files[idx])
        out_path = os.path.join(out_dir, f"frame_{idx:02d}_{name}.png")
        write_png(out_path, out, bit_depth=8)
        glog(f"PROGRESS {k + 1}/{len(order)} render")
        info.append(
            f"frame {idx} {name} lit={100 * disk.lit:.1f}% r={disk.r:.1f}"
            f" ({'wide-fit' if disk.fitWide else 'prior-fit'}"
            f" rms {disk.fitRms:.1f} arc {disk.fitArc})"
            f" theta={math.degrees(disk.theta):.1f} rot={rot_deg:.2f}")
        rows.append({"idx": idx, "name": name, "age": age})

    # frames-info.txt: human table; raw chain diagnostics to frames-debug.txt
    rows.sort(key=lambda r: r["idx"])
    name_w = max([4] + [len(r["name"]) for r in rows])
    table = [f"{'frame':<7}{'file':<{name_w + 2}}{'phase':<17}{'age':<7}"
             f"{'illum':<7}date"]
    import re

    for r in rows:
        m = re.search(r"\d{4}-\d{2}-\d{2}", r["name"])
        age = r["age"]
        illum = (f"{50 * (1 - math.cos(2 * np.pi * age / SYNODIC)):.0f}%"
                 if math.isfinite(age) else "?")
        table.append(
            f"{r['idx']:02d}{'':5}{r['name']:<{name_w + 2}}"
            f"{phase_name(age):<17}"
            f"{(f'{age:.1f}d' if math.isfinite(age) else '?'):<7}"
            f"{illum:<7}{m.group(0) if m else '?'}")
    _write(out_dir, "frames-info.txt", table)
    _write(out_dir, "frames-debug.txt", info)


def _name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _write(out_dir: str, name: str, lines: list[str]) -> None:
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
