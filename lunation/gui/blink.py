"""Blink preview set — one representative frame per date group,
disk-aligned, in phase order. PixInsight-Blink-like, but one frame per
day: no stacking, no noise processing — pick a decent frame from each
group's best SER, fit the lunar disk, and center it, so cycling frames
holds the moon still while the phase advances.

Frame choice per group: luminance if the night is LRGB, otherwise the
OSC/mono capture (chroma channels only as a last resort). "Decent" =
best Laplacian sharpness among a small evenly-spaced sample of frames.
"""

import math
import re

import numpy as np

from ..assemble.collect import SYNODIC, lunar_age
from ..assemble.disk import R_EXP, analyze_disk
from ..assemble.register import (best_rotation_ncc, flip_h, norm180,
                                 rim_on_small)
from ..assemble.render import SMALL, center_on_canvas, center_rescale, \
    small_lum
from ..core.kernels import laplacian_sharpness
from ..core.warp import rotate
from ..io.ser import SerReader
from .preview import autostretch

# lum first, chroma last (fixed: "lum if it's lrgb, whichever decent
# frame if it's osc")
_CHANNEL_PREF = {"L": 0, "MONO": 1, "OSC": 2, "G": 3, "R": 4, "B": 5,
                 "S": 6, "H": 7}

SAMPLES = 10


def pick_group_ser(ser_entries: list[dict]) -> dict | None:
    """The group's most luminance-like SER entry (needs 'label')."""
    if not ser_entries:
        return None
    return min(ser_entries,
               key=lambda e: _CHANNEL_PREF.get(e.get("label", ""), 9))


def pick_frame(ser_path: str, samples: int = SAMPLES) -> np.ndarray:
    """Sharpest of `samples` evenly spaced frames, float32 [0,1]."""
    r = SerReader(ser_path)
    try:
        n = r.frame_count
        idxs = sorted({int(k * (n - 1) / max(1, min(samples, n) - 1))
                       for k in range(min(samples, n))})
        best, best_q = None, -1.0
        for i in idxs:
            f = r.read(i)
            q = laplacian_sharpness(f)
            if q > best_q:
                best, best_q = f, q
        return best
    finally:
        r.close()


def _age_of(key: str) -> float:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", key):
        return lunar_age(key)
    return float("nan")


def _texture(f: np.ndarray) -> np.ndarray:
    """Rotation-signal preconditioning for terminator-less (near-full)
    frames: a raw full disk is dominated by its rotation-INVARIANT
    brightness profile (limb darkening), so full-circle NCC plateaus
    ~0.9 at any angle and weak maria texture picks a near-arbitrary
    winner. Bandpass the luminance so craters/maria — the only rotation
    signal a full moon has — drive the score instead. Same principle as
    the stacker's gradient preconditioning (M0)."""
    import cv2

    return (cv2.GaussianBlur(f, (0, 0), 1.5)
            - cv2.GaussianBlur(f, (0, 0), 8.0))


def _median_angle(angles: list[float]) -> float:
    rel = sorted(norm180(a - angles[0]) for a in angles)
    return norm180(angles[0] + rel[len(rel) >> 1])


def _full_circle_consensus(f, m, f_m, m_m, targets, trace):
    """Terminator-less rotation by texture CONSENSUS. Two field-found
    facts (2026-06-30): crater relief reverses contrast across the
    wax/wane illumination boundary, so the true angle can appear as
    ANTI-correlation against waxing targets; and the absolute NCC level
    of weak full-moon texture is untrustworthy — scattered wrong answers
    can outscore the consistent right one. The one thing wrong answers
    can't do is AGREE: evaluate mirror x polarity hypotheses against
    every target and prefer the tightest multi-target consensus.

    Mirrored hypotheses rank below unmirrored ones no matter the score:
    a near-full disk is symmetric enough that mirror+rotation can
    pseudo-match maria, but every blink frame is a raw SER from the
    same rig — parity does not flip between nights. Mirror only wins
    when nothing unmirrored agrees. Returns (rot, mirrored, score)."""
    # NOTE: do NOT mask off the limb band here — tried and reverted
    # (2026-07-17): on a real near-full moon the strongest texture lives
    # NEAR the limb (the terminator hugs it), and interior-only scoring
    # collapsed every real hypothesis to noise
    ft, fmt = _texture(f), _texture(f_m)
    hyps: dict[str, list[dict]] = {}
    for tgt in targets:
        tt = _texture(tgt)
        for name, (img, msk, ref) in {
                "id+": (ft, m, tt), "id-": (ft, m, -tt),
                "mir+": (fmt, m_m, tt), "mir-": (fmt, m_m, -tt)}.items():
            hyps.setdefault(name, []).append(
                best_rotation_ncc(img, msk, ref))
    best = None
    for name, vs in hyps.items():
        angles = [v["angle"] for v in vs]
        a = _median_angle(angles)
        spread = max(abs(norm180(x - a)) for x in angles)
        score = float(np.mean([abs(v["score"]) for v in vs]))
        tight = len(vs) >= 2 and spread <= 15
        trace(f"  {name}: rot {a:.1f} spread {spread:.1f}"
              f" |s| {score:.3f}{' TIGHT' if tight else ''}")
        rank = (tight, not name.startswith("mir"), score)
        if best is None or rank > best[0]:
            best = (rank, a, name.startswith("mir"), score)
    _, rot, mirrored, score = best
    return rot, mirrored, score


def _chain_rotation(frame, disk, age, ref_r, canvas, targets,
                    trace=lambda msg: None):
    """One link of the render chain (render.py:280-370): physics seed
    (lit limb east for waxing / west for waning), dual-parity NCC voted
    across the <=3 most recent aligned frames, rim snap-back, physics
    fallback below the 0.55 link score. Returns (rot_deg, mirrored).

    Preview deviation: the render caps the snap-back at 30° and trusts its
    multi-neighbor vote beyond that; here a frame with a real terminator
    gets its lit limb snapped to the physics target NO MATTER how far off
    the NCC answer was — raw single frames earn less trust, and a
    backwards moon in a draft preview is worse than a slightly re-seated
    one."""
    waxing = not (age >= SYNODIC / 2)  # NaN -> waxing default
    t = 0.0 if waxing else 180.0
    theta = norm180(math.degrees(disk.thetaRim))
    if not targets:
        # anchor: its own terminator axis fixes orientation
        trace(f"anchor rimDark {disk.rimDark:.2f} theta {theta:.1f}"
              f" T {t:.0f}")
        return (norm180(theta - t), False) if disk.rimDark >= 0.30 \
            else (0.0, False)
    f = small_lum(frame, canvas)
    m = (f > 0.10).astype(np.float32)
    f_m, m_m = flip_h(f), flip_h(m)
    if disk.rimDark >= 0.30:
        r_is, r_ms, vote_margin = [], [], 0.0
        for tgt in targets:
            r_is.append(best_rotation_ncc(f, m, tgt,
                                          norm180(theta - t), 20))
            r_ms.append(best_rotation_ncc(f_m, m_m, tgt,
                                          norm180((180 - theta) - t), 20))
            vote_margin += r_ms[-1]["score"] - r_is[-1]["score"]
        # parity flips only on a decisive aggregate margin (render.py:321)
        mirrored = vote_margin > 0.03 * len(targets)
        est = [v["angle"] for v in (r_ms if mirrored else r_is)]
        rot = _median_angle(est) if len(est) >= 3 else est[0]
        score = (r_ms if mirrored else r_is)[0]["score"]
        trace(f"rimDark {disk.rimDark:.2f} theta {theta:.1f} T {t:.0f}"
              f" seeded id {r_is[0]['angle']:.1f}/{r_is[0]['score']:.3f}"
              f" mir {r_ms[0]['angle']:.1f}/{r_ms[0]['score']:.3f}"
              f" vote {vote_margin:.4f}x{len(targets)}"
              f" -> {'MIRROR ' if mirrored else ''}rot {rot:.1f}"
              f" score {score:.3f}")
    else:
        trace(f"rimDark {disk.rimDark:.2f} T {t:.0f} full-circle"
              f" consensus x{len(targets)}:")
        rot, mirrored, score = _full_circle_consensus(
            f, m, f_m, m_m, targets, trace)
    # physics snap-back: the rotated rim must show its lit limb at T
    rs = rotate(f_m if mirrored else f, np.pi * rot / 180,
                SMALL / 2, SMALL / 2)
    rim = rim_on_small(rs, ref_r * SMALL / canvas)
    if rim["darkFrac"] >= 0.30:
        res = norm180(rim["litDeg"] - t)
        if abs(res) > 15:  # full-range: never leave a moon backwards
            rot = norm180(rot + res)
            trace(f"  SNAP {res:.1f} -> rot {rot:.1f}")
    if score < 0.55 and disk.rimDark >= 0.30:
        # link unusable — physics-only, like the render (but the preview
        # never excludes: it exists to LOOK at weak frames)
        trace(f"  FALLBACK physics -> rot {norm180(theta - t):.1f}")
        return norm180(theta - t), False
    return rot, mirrored


def normalize_pedestal(frame: np.ndarray) -> np.ndarray:
    """Kill the sky pedestal of a raw capture (twilight glow, gain
    offset). Every consumer downstream — the disk fit's 0.04/0.08
    thresholds, the NCC masks (>0.10), the rim dark classification —
    assumes finished-style black sky; a pedestal above threshold turns
    the whole frame 'lit', the limb fit finds no rim, and the physics
    axis (and everything seeded from it) is garbage. Robust and
    style-independent: black point at the 2nd percentile, unit range at
    the 99.5th."""
    lo = float(np.percentile(frame, 2.0))
    hi = float(np.percentile(frame, 99.5))
    if hi - lo < 1e-6:
        return frame
    return np.clip((frame - lo) / (hi - lo), 0.0, 1.0)


def analyze_disk_raw(frame: np.ndarray):
    """analyze_disk with a scale bridge: its Kasa fit is calibrated for
    drizzled/prep-normalized disks (R_EXP ≈ 1058 px radius), while raw SER
    frames run ~half that (OSC superpixel: a quarter) — a crescent's short
    limb arc then fails the fit gates and falls back to a clamped radius,
    wrecking the shared canvas. Rough-size the disk from the lit bbox (a
    crescent's lit limb still spans the full diameter), resample into the
    calibrated regime, fit there, and map the circle back."""
    import cv2

    ys, xs = np.nonzero(frame > 0.08)
    if not len(xs):
        return analyze_disk(frame)
    rough_r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    s = R_EXP / max(rough_r, 1.0)
    if 0.85 <= s <= 1.2:
        return analyze_disk(frame)
    scaled = cv2.resize(
        frame, (round(frame.shape[1] * s), round(frame.shape[0] * s)),
        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    d = analyze_disk(scaled)
    d.cx /= s
    d.cy /= s
    d.r /= s
    return d


def build_blink(group_sers: list[tuple[str, str]], out_px: int = 680,
                progress=lambda step, total, label: None,
                align: bool = False,
                trace=lambda msg: None) -> list[dict]:
    """[(key, ser_path)] -> [{key, image}] disk-centered on a common
    canvas (sized by the largest disk), per-frame autostretched.

    align=True additionally runs the output's rotation chain over the set
    in phase order — a draft of how the gif will play: registration only,
    no stacking, no noise processing, no finishing.

    `progress(step, total, label)` fires before each unit of work across
    BOTH phases (pick + align/render), so a UI can keep a bar moving
    through the expensive NCC pass instead of appearing hung."""
    import cv2

    n = len(group_sers)
    total = 2 * n
    picked = []
    for k, (key, path) in enumerate(group_sers):
        progress(k, total, f"picking {key}")
        frame = normalize_pedestal(pick_frame(path))
        picked.append((key, frame, analyze_disk_raw(frame)))
    if not picked:
        return []
    ref_r = max(d.r for _, _, d in picked)
    canvas = int(2 * ref_r * 1.06) & ~1
    out, targets = [], []
    for i, (key, frame, disk) in enumerate(picked):
        progress(n + i, total,
                 f"{'aligning' if align else 'rendering'} {key}")
        c = center_on_canvas(frame, disk, canvas)
        s_norm = ref_r / disk.r
        if abs(s_norm - 1) > 0.005:  # radius-normalize mixed rigs
            c = center_rescale(c, s_norm, canvas)
        if align:
            rot, mirrored = _chain_rotation(
                c, disk, _age_of(key), ref_r, canvas, targets,
                trace=lambda msg, key=key: trace(f"{key}: {msg}"))
            if mirrored:
                c = flip_h(c)
            if rot:
                c = rotate(c, np.pi * rot / 180, canvas / 2, canvas / 2)
            targets.insert(0, small_lum(c, canvas))  # newest first
            del targets[3:]
        if canvas > out_px:
            c = cv2.resize(c, (out_px, out_px),
                           interpolation=cv2.INTER_AREA)
        out.append({"key": key, "image": autostretch(c)})
    progress(total, total, "done")
    return out
