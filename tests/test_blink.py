"""Blink set builder: frame choice, channel preference, disk alignment."""

import numpy as np
import pytest

from lunation.gui.blink import build_blink, pick_frame, pick_group_ser

from .test_ser import build_ser

# analyze_disk's Kasa fit is calibrated for production-scale disks
# (R_EXP ~979px, lower fit bound ~343px) — synthetic disks must be
# realistically sized or the fit rejects the limb
SIZE, R = 1024, 450


def disk_frames(dx, dy, n=5, sharp_at=None, seed=3):
    """n frames of a speckled disk at (center+dx, center+dy); all but
    `sharp_at` are blurred when it's set."""
    import cv2

    rng = np.random.default_rng(seed)
    # multi-scale texture like a real moon (maria + craters + grain):
    # rotation search needs a WIDE correlation basin, which per-pixel or
    # single-scale noise cannot give at realistic disk radii
    speckle = np.zeros((SIZE, SIZE), np.float32)
    for sigma, amp in ((64, 0.5), (16, 0.3), (2, 0.2)):
        layer = cv2.GaussianBlur(
            rng.uniform(-1, 1, (SIZE, SIZE)).astype(np.float32),
            (0, 0), sigma)
        speckle += amp * layer / max(1e-6, np.abs(layer).max())
    speckle = np.clip(0.55 + 0.45 * speckle, 0.05, 1)
    out = []
    for k in range(n):
        m = np.zeros((SIZE, SIZE), np.float32)
        cv2.circle(m, (SIZE // 2 + dx, SIZE // 2 + dy), R, 1.0, -1)
        f = m * speckle
        if sharp_at is not None and k != sharp_at:
            f = cv2.GaussianBlur(f, (0, 0), 3.0)
        out.append(np.rint(np.clip(f, 0, 1) * 65535).astype("<u2"))
    return out


def disk_ser(path, dx=0, dy=0, **kw):
    build_ser(str(path), disk_frames(dx, dy, **kw), color_id=0, depth=16)
    return str(path)


def test_pick_frame_prefers_sharp(tmp_path):
    p = disk_ser(tmp_path / "a.ser", sharp_at=3)
    from lunation.io.ser import SerReader

    picked = pick_frame(p, samples=5)
    np.testing.assert_array_almost_equal(
        picked, SerReader(p).read(3), decimal=6)


def test_pick_group_ser_preference():
    assert pick_group_ser([]) is None
    entries = [{"label": "R", "path": "r"}, {"label": "L", "path": "l"},
               {"label": "B", "path": "b"}]
    assert pick_group_ser(entries)["path"] == "l"  # lum wins in LRGB
    assert pick_group_ser([{"label": "OSC", "path": "o"},
                           {"label": "R", "path": "r"}])["path"] == "o"


def test_align_recovers_planted_rotation(tmp_path):
    """Group 2 is group 1's disk rotated 25° — the chain rotation must
    recover ~-25° to de-rotate it, unmirrored."""
    from lunation.assemble.disk import analyze_disk
    from lunation.assemble.render import center_on_canvas, small_lum
    from lunation.assemble.register import norm180
    from lunation.core.warp import rotate
    from lunation.gui.blink import _chain_rotation

    f1 = disk_frames(0, 0, n=1, seed=7)[0].astype(np.float32) / 65535.0
    f2 = rotate(f1, np.pi * 25 / 180, SIZE / 2, SIZE / 2)
    d1, d2 = analyze_disk(f1), analyze_disk(f2)
    ref_r = max(d1.r, d2.r)
    canvas = int(2 * ref_r * 1.06) & ~1
    c1 = center_on_canvas(f1, d1, canvas)
    c2 = center_on_canvas(f2, d2, canvas)

    rot, mirrored = _chain_rotation(c2, d2, float("nan"), ref_r, canvas,
                                    targets=[small_lum(c1, canvas)])
    assert not mirrored
    assert abs(norm180(rot + 25)) < 3.0


def test_analyze_disk_raw_recovers_true_radius(tmp_path):
    """Raw-scale disks (half the drizzled R_EXP and below) must fit their
    true radius, not the fallback clamp."""
    from lunation.gui.blink import analyze_disk_raw

    f = disk_frames(0, 0, n=1, seed=5)[0].astype(np.float32) / 65535.0
    d = analyze_disk_raw(f)
    assert abs(d.r - R) < 0.03 * R
    # OSC-superpixel scale (quarter of drizzled)
    import cv2

    half = cv2.resize(f, (SIZE // 2, SIZE // 2),
                      interpolation=cv2.INTER_AREA)
    d2 = analyze_disk_raw(half)
    assert abs(d2.r - R / 2) < 0.03 * R


def test_build_blink_matches_mixed_scales(tmp_path):
    """Same moon captured at two rigs/scales: after radius-normalize the
    output disks must be the SAME size (the crescent-vs-quarter bug)."""
    import cv2

    frames = disk_frames(0, 0, n=1, seed=9)
    small = [np.rint(cv2.resize(f.astype(np.float32), (0, 0), fx=0.55,
                                fy=0.55, interpolation=cv2.INTER_AREA))
             .astype("<u2") for f in frames]
    p1, p2 = str(tmp_path / "1.ser"), str(tmp_path / "2.ser")
    build_ser(p1, frames, color_id=0, depth=16)
    build_ser(p2, small, color_id=0, depth=16)

    out = build_blink([("d1", p1), ("d2", p2)], out_px=128)
    widths = []
    for f in out:
        m = f["image"] > 0.02
        xs = np.nonzero(m)[1]
        widths.append(xs.max() - xs.min())
    assert abs(widths[0] - widths[1]) <= 3  # size-matched despite rigs


@pytest.mark.parametrize("pedestal", [0.0, 0.12])
def test_align_never_leaves_a_moon_backwards(tmp_path, pedestal):
    """A waxing night captured lit-WEST (mirrored/derotated rig) must come
    out lit-EAST after align — the full-range physics snap-back. The
    pedestal variant is the 2026-06-20 regression: a twilight/gain sky
    offset above the disk-fit thresholds turned the whole frame 'lit',
    wrecked the limb fit and the physics axis, and the frame rendered
    backwards with every guard blinded by the same bad geometry."""
    import cv2

    rng = np.random.default_rng(11)
    tex = cv2.GaussianBlur(
        rng.uniform(0.2, 1.0, (SIZE, SIZE)).astype(np.float32), (0, 0), 8)
    tex = np.clip(0.4 + 0.6 * tex, 0, 1)
    m = np.zeros((SIZE, SIZE), np.float32)
    cv2.circle(m, (SIZE // 2, SIZE // 2), R, 1.0, -1)
    xx = np.arange(SIZE)[None, :]

    def half(lit_right):
        lit = (xx >= SIZE // 2) if lit_right else (xx < SIZE // 2)
        f = m * tex * np.where(lit, 1.0, 0.015)
        f = pedestal + (1 - pedestal) * f  # sky glow / gain offset
        return [np.rint(np.clip(f, 0, 1) * 65535).astype("<u2")]

    p1, p2 = str(tmp_path / "1.ser"), str(tmp_path / "2.ser")
    build_ser(p1, half(lit_right=True), color_id=0, depth=16)
    build_ser(p2, half(lit_right=False), color_id=0, depth=16)  # backwards

    # both waxing dates -> physics demands lit limb EAST (+x)
    out = build_blink([("2026-04-21", p1), ("2026-04-23", p2)],
                      out_px=128, align=True)
    for f in out:
        img = f["image"]
        bright = img > 0.5 * img.max()
        xs = np.nonzero(bright)[1]
        assert xs.mean() > img.shape[1] / 2 + 5, \
            f"{f['key']} rendered lit-west (backwards)"


def test_full_circle_consensus_survives_relief_reversal(tmp_path):
    """The 2026-06-30 regression, mechanism-level: a terminator-less
    (near-full) frame whose crater relief REVERSES contrast against its
    targets (wax->wane illumination flip) must still register — the true
    angle only shows as ANTI-correlation, garbage positive-correlation
    answers scatter across targets, and a mirrored pseudo-match must not
    win (same-rig SERs don't flip parity)."""
    import cv2

    from lunation.assemble.register import norm180
    from lunation.assemble.render import center_on_canvas, small_lum
    from lunation.core.warp import rotate
    from lunation.gui.blink import _chain_rotation, analyze_disk_raw

    # physical relief lighting: texture = directional derivative of a
    # height map along the sun azimuth. Different nights = different
    # azimuths (decorrelates spurious matches); wax->wane = azimuth
    # flipped 180 = texture exactly negated.
    rng = np.random.default_rng(21)
    height = np.zeros((SIZE, SIZE), np.float32)
    for sigma, amp in ((64, 0.5), (16, 0.3), (4, 0.2)):
        layer = cv2.GaussianBlur(
            rng.uniform(-1, 1, (SIZE, SIZE)).astype(np.float32),
            (0, 0), sigma)
        height += amp * layer / max(1e-6, np.abs(layer).max())
    gy, gx = np.gradient(height)
    m = np.zeros((SIZE, SIZE), np.float32)
    cv2.circle(m, (SIZE // 2, SIZE // 2), R, 1.0, -1)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    d2 = ((xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2) / R ** 2
    # strong limb darkening + soft edge: the limb ring must not dominate
    # the texture (a sharp synthetic edge creates a rotation-invariant
    # NCC baseline no real moon has)
    profile = np.clip(1 - 0.9 * d2, 0, 1) ** 0.8

    def night(sun_az_deg, rot_deg, seed):
        import math

        a = math.radians(sun_az_deg)
        relief = math.cos(a) * gx + math.sin(a) * gy
        relief = relief / max(1e-6, np.abs(relief).max())
        f = m * profile * (0.5 + 0.5 * (0.5 + 0.5 * relief))
        f = cv2.GaussianBlur(f, (0, 0), 3.0)
        if rot_deg:
            f = rotate(f, np.pi * rot_deg / 180, SIZE / 2, SIZE / 2)
        n = np.random.default_rng(seed).normal(0, 0.01, f.shape)
        return np.clip(f + n, 0, 1).astype(np.float32)

    f1 = night(0, 0, 31)     # waxing, sun from the east
    f2 = night(20, 15, 32)   # waxing, sun moved on; other camera angle
    f3 = night(180, 40, 33)  # WANING: relief reversed, rotated +40

    disks = [analyze_disk_raw(f) for f in (f1, f2, f3)]
    ref_r = max(d.r for d in disks)
    canvas = int(2 * ref_r * 1.06) & ~1
    c1 = center_on_canvas(f1, disks[0], canvas)
    c2 = rotate(center_on_canvas(f2, disks[1], canvas),
                np.pi * -15 / 180, canvas / 2, canvas / 2)  # de-rotated
    targets = [small_lum(c2, canvas), small_lum(c1, canvas)]
    c3 = center_on_canvas(f3, disks[2], canvas)
    rot, mirrored = _chain_rotation(c3, disks[2], float("nan"), ref_r,
                                    canvas, targets)
    assert not mirrored
    assert abs(norm180(rot + 40)) < 5.0


def test_pedestal_does_not_break_disk_fit(tmp_path):
    """Sky pedestal above the fit thresholds (the 06-20 failure): radius
    must still be recovered through normalize_pedestal."""
    from lunation.gui.blink import analyze_disk_raw, normalize_pedestal

    f = disk_frames(0, 0, n=1, seed=5)[0].astype(np.float32) / 65535.0
    hazy = 0.12 + 0.88 * f
    # unnormalized, the fit sees a fully-'lit' frame and degenerates
    d_fixed = analyze_disk_raw(normalize_pedestal(hazy))
    assert abs(d_fixed.r - R) < 0.03 * R
    assert d_fixed.fitArc >= 140  # a real limb arc was found


def test_build_blink_reports_both_phases(tmp_path):
    """progress must tick through pick AND align (the NCC pass is the
    slow part — a bar that stops there reads as a hang)."""
    p1 = disk_ser(tmp_path / "1.ser", dx=-10)
    p2 = disk_ser(tmp_path / "2.ser", dx=8)
    calls = []
    build_blink([("2026-04-21", p1), ("2026-04-23", p2)], out_px=96,
                align=True,
                progress=lambda s, t, label: calls.append((s, t, label)))
    steps = [c[0] for c in calls]
    assert steps == sorted(steps)  # monotonic
    assert calls[0][:2] == (0, 4)
    assert calls[-1] == (4, 4, "done")
    labels = " ".join(c[2] for c in calls)
    assert "picking" in labels and "aligning" in labels


def test_build_blink_centers_disks(tmp_path):
    groups = [("d1", disk_ser(tmp_path / "1.ser", dx=-30, dy=18)),
              ("d2", disk_ser(tmp_path / "2.ser", dx=24, dy=-22))]
    frames = build_blink(groups, out_px=96)
    assert [f["key"] for f in frames] == ["d1", "d2"]
    shapes = {f["image"].shape for f in frames}
    assert len(shapes) == 1  # common canvas
    for f in frames:
        img = f["image"]
        h, w = img.shape
        m = img > 0.02  # whole disk support (sky stays ~0 after stretch)
        ys, xs = np.nonzero(m)
        # disk centered despite different capture offsets
        assert abs(xs.mean() - w / 2) < 3
        assert abs(ys.mean() - h / 2) < 3
