"""End-to-end synthetic stacking tests.

A SER of noisy, randomly-shifted copies of a known ground truth goes
through the full stacker; the stack must come out sharper/cleaner than any
single frame and honor the log/report contracts. Small sizes keep this
fast; the pool path is exercised implicitly in the real-session validation
(workers>1 needs >=32 frames and real capture sizes to be worth it).
"""

import json

import numpy as np
import pytest

from lunation.core.kernels import laplacian_sharpness
from lunation.io.xisf_io import read_xisf
from lunation.stack import stacker

from .synth import fourier_shift, lunar_texture
from .test_ser import build_ser

SIZE = 128
N_FRAMES = 40


def _make_capture(tmp_path, seed=11, n=N_FRAMES, blur_bad=True):
    """SER of shifted noisy frames; returns (path, truth, shifts)."""
    import cv2

    rng = np.random.default_rng(seed)
    truth = lunar_texture(SIZE, seed=5)
    frames, shifts = [], []
    for k in range(n):
        dx, dy = rng.uniform(-4, 4, 2)
        f = fourier_shift(truth, dx, dy)
        if blur_bad and k % 4 == 0:  # every 4th frame is bad-seeing
            f = cv2.GaussianBlur(f, (0, 0), 2.5)
        f = f + rng.normal(0, 0.01, f.shape).astype(np.float32)
        frames.append(np.clip(f, 0, 1))
        shifts.append((dx, dy))
    raw = [np.rint(f * 65535).astype("<u2") for f in frames]
    p = str(tmp_path / "capture.ser")
    build_ser(p, raw, color_id=0, depth=16)
    return p, truth, shifts


@pytest.mark.parametrize("engine", ["stsci", "ported"])
def test_stack_recovers_detail(tmp_path, engine):
    ser_path, truth, _ = _make_capture(tmp_path)
    out = str(tmp_path / f"stack_{engine}.xisf")
    cfg = {
        "ser": ser_path, "out": out, "report": out + ".json",
        "bestFraction": 0.5, "minFrames": 10, "drizzle": 2,
        "drizzleMargin": 8, "rejection": "minmax",  # few frames
        "drizzleEngine": engine, "workers": 1,
    }
    assert stacker.run(cfg) is True

    stacked = read_xisf(out)
    assert stacked.shape == (SIZE * 2, SIZE * 2)
    # mean level preserved (drizzle must average, not sum)
    assert abs(stacked.mean() - truth.mean()) < 0.02
    # noise beaten: single-frame sky std ~0.01, stack should be well below
    sky = stacked[:20, :20]
    assert float(sky.std()) < 0.006

    rep = json.load(open(out + ".json"))
    assert rep["frames"] == N_FRAMES
    assert rep["drizzle"] == 2
    assert rep["stacked"] >= 10
    assert len(rep["quality"]) == N_FRAMES
    assert len(rep["shifts"]) == rep["stacked"] + rep["dropped"]
    assert rep["shifts"][0]["dx"] == 0.0  # reference first

    log_text = open(out + ".log").read()
    assert "=== STACK OK" in log_text
    assert "PROGRESS" in log_text


def test_registration_accuracy_against_known_shifts(tmp_path):
    """Report shifts must match the planted shifts to ~0.05 px (skimage
    engine promise) for the sharp frames."""
    ser_path, _, planted = _make_capture(tmp_path, blur_bad=False)
    out = str(tmp_path / "reg.xisf")
    cfg = {"ser": ser_path, "out": out, "report": out + ".json",
           "bestFraction": 1.0, "minFrames": 10, "drizzle": 1,
           "rejection": "none", "workers": 1}
    assert stacker.run(cfg) is True
    rep = json.load(open(out + ".json"))
    ref = rep["refIndex"]
    ref_dx, ref_dy = planted[ref]
    errs = []
    for s in rep["shifts"][1:]:
        pdx, pdy = planted[s["i"]]
        # shifts are relative to the reference frame
        errs.append(np.hypot(s["dx"] - (pdx - ref_dx),
                             s["dy"] - (pdy - ref_dy)))
    assert float(np.median(errs)) < 0.05
    assert float(np.max(errs)) < 0.2


def test_sigma_rejection_kills_transient(tmp_path):
    """A satellite-streak-like transient in a few frames must vanish under
    sigma rejection (needs >=25 frames + drizzle)."""
    import cv2

    rng = np.random.default_rng(2)
    truth = lunar_texture(SIZE, seed=5)
    frames = []
    for k in range(30):
        f = fourier_shift(truth, *rng.uniform(-2, 2, 2))
        f = f + rng.normal(0, 0.008, f.shape).astype(np.float32)
        if k in (3, 4):  # transient bright streak
            cv2.line(f, (10, 20), (110, 100), 1.0, 2)
        frames.append(np.clip(f, 0, 1))
    p = str(tmp_path / "streak.ser")
    build_ser(p, [np.rint(f * 65535).astype("<u2") for f in frames],
              color_id=0, depth=16)

    outs = {}
    for rej in ("sigma", "none"):
        out = str(tmp_path / f"s_{rej}.xisf")
        cfg = {"ser": p, "out": out, "bestFraction": 1.0, "minFrames": 25,
               "drizzle": 2, "drizzleMargin": 8, "rejection": rej,
               "workers": 1}
        assert stacker.run(cfg) is True
        outs[rej] = read_xisf(out)
    # streak pixels: bright in the plain mean, gone in the sigma stack
    diff = outs["none"] - outs["sigma"]
    assert float(diff.max()) > 0.02          # streak visible in plain mean
    assert float(np.abs(outs["sigma"] - outs["none"]).mean()) < 0.005


def test_localalign_runs_and_improves_nothing_synthetic(tmp_path):
    """LocalWarp on pure-translation data must at least not hurt (its win
    shows on real seeing warps): stack completes, output sane."""
    ser_path, truth, _ = _make_capture(tmp_path, blur_bad=False)
    out = str(tmp_path / "warp.xisf")
    cfg = {"ser": ser_path, "out": out, "bestFraction": 0.5, "minFrames": 10,
           "drizzle": 1, "rejection": "none", "localAlign": True,
           "tileSize": 64, "tileStep": 32, "workers": 1}
    assert stacker.run(cfg) is True
    stacked = read_xisf(out)
    assert abs(stacked.mean() - truth.mean()) < 0.03
    core = np.s_[16:-16, 16:-16]
    assert laplacian_sharpness(stacked[core]) > 0
