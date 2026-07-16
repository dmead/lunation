"""SER lucky-imaging stacker — ports pjsr/ser-stack.js:326-800.

Stage order, config schema, log protocol (`PROGRESS k/n stage`,
`=== STACK OK` / `*** STACK FAILED`) and the report JSON field set match
the PJSR original exactly; PI's lunar-finish.js consumes our output.

What changed (better-than decisions):
- registration: skimage upsampled-DFT phase correlation (~0.01 px) instead
  of PI's 3x3 peak estimate (~0.5 px);
- drizzle: STScI drizzle kernel driven by per-frame pixmaps
  (drizzleEngine "stsci", default) with the 1:1 upsample-and-place port
  behind drizzleEngine "ported" for A/B;
- parallelism: a process pool over FRAMES inside the job (replaces child
  PixInsight instances; no launch mutex, one memory budget).
"""

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..core.fftreg import PhaseCorrelator
from ..core.framecube import SENTINEL, FrameCube
from ..core.kernels import laplacian_sharpness
from ..core.warp import resample, translate
from ..io.ser import SerReader
from ..io.xisf_io import write_xisf
from .localwarp import LocalWarp
from .logutil import JobLog

DEFAULTS = {
    "bestFraction": 0.10, "maxFrames": 1_000_000, "minFrames": 25,
    "drizzle": 1, "drizzleMargin": 16, "channel": "mono",
    "alignOnGradient": False, "localAlign": False,
    "tileSize": 256, "tileStep": 128, "clampPx": 3,
    "rejectKappa": 3.0, "rejectIterations": 3,
    "drizzleInterpolation": "bicubic", "drizzleEngine": "stsci",
    "workers": None,
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _cfg(cfg: dict, key: str):
    v = cfg.get(key)
    return DEFAULTS[key] if v is None else v


# ---------------------------------------------------------------- workers

_W: dict = {}


def _init_quality(ser_path: str, channel: str) -> None:
    import cv2

    cv2.setNumThreads(1)
    _W["ser"] = SerReader(ser_path, channel)


def _quality_chunk(idxs: list[int]) -> list[tuple[int, float]]:
    ser = _W["ser"]
    return [(i, laplacian_sharpness(ser.read(i))) for i in idxs]


def _drizzle_plane_ported(src, scale, sdx, sdy, margin, acc_shape, interp):
    """1:1 port of ser-stack.js upsampled()+placement (435-448, 550-569).
    Returns (plane with -1 outside coverage, ox, oy) or None if the shift
    exceeds the drizzle headroom."""
    tx, ty = -scale * sdx, -scale * sdy
    ox, oy = margin + round(tx), margin + round(ty)
    if ox < 0 or oy < 0 or ox > 2 * margin or oy > 2 * margin:
        return None
    up = resample(src, scale, interp)
    fx, fy = tx - round(tx), ty - round(ty)
    if interp != "nearest" and (fx or fy):
        up = translate(up, fx, fy, interp)
    plane = np.full(acc_shape, -1.0, dtype=np.float32)
    plane[oy : oy + up.shape[0], ox : ox + up.shape[1]] = up
    return plane


def _drizzle_plane_stsci(src, scale, sdx, sdy, margin, acc_shape):
    """STScI drizzle kernel via raw pixmap: out = scale*(in - shift) + margin.
    Same headroom drop rule as the ported engine so 'dropped' semantics
    match."""
    from drizzle.resample import Drizzle

    tx, ty = -scale * sdx, -scale * sdy
    ox, oy = margin + round(tx), margin + round(ty)
    if ox < 0 or oy < 0 or ox > 2 * margin or oy > 2 * margin:
        return None
    pm = _W["pixmap_base"].copy()
    pm[..., 0] += tx
    pm[..., 1] += ty
    d = Drizzle(kernel="square", out_shape=acc_shape, fillval=0.0)
    d.add_image(data=np.ascontiguousarray(src, dtype=np.float32),
                exptime=1.0, pixmap=pm, pixfrac=1.0)
    plane = np.asarray(d.out_img, dtype=np.float32)
    valid = np.asarray(d.out_wht) > 0
    plane[~valid] = -1.0
    return plane


def _make_pixmap_base(w, h, scale, margin):
    ys, xs = np.indices((h, w), dtype=np.float64)
    half = (scale - 1) / 2.0
    return np.dstack([xs * scale + margin + half, ys * scale + margin + half])


def _init_stack(ser_path, channel, ref_index, p) -> None:
    import cv2

    cv2.setNumThreads(1)
    ser = SerReader(ser_path, channel)
    ref = ser.read(ref_index)
    aligner = PhaseCorrelator(use_gradient=p["alignOnGradient"])
    aligner.initialize(ref)
    warper = None
    if p["localAlign"]:
        warper = LocalWarp(ref, p["tileSize"], p["tileStep"], p["clampPx"])
    _W.update(ser=ser, aligner=aligner, warper=warper, p=p)
    if p["scale"] > 1 and p["engine"] == "stsci":
        _W["pixmap_base"] = _make_pixmap_base(
            ser.width, ser.height, p["scale"], p["margin"])
    if p["cache_path"]:
        _W["cube"] = FrameCube(p["cache_path"], p["accW"], p["accH"])


def _frame_plane(src, sdx, sdy, p):
    if p["engine"] == "stsci":
        return _drizzle_plane_stsci(src, p["scale"], sdx, sdy, p["margin"],
                                    (p["accH"], p["accW"]))
    return _drizzle_plane_ported(src, p["scale"], sdx, sdy, p["margin"],
                                 (p["accH"], p["accW"]), p["interp"])


def _stack_chunk(items: list[tuple[int, int]]) -> dict:
    """items: (plane_index, frame_index). Returns partial accumulators."""
    ser, aligner, warper, p = _W["ser"], _W["aligner"], _W["warper"], _W["p"]
    acc = np.zeros((p["accH"], p["accW"]), dtype=np.float64)
    cover = np.zeros((p["accH"], p["accW"]), dtype=np.int32)
    mn = np.full((p["accH"], p["accW"]), 1e10, dtype=np.float32)
    mx = np.full((p["accH"], p["accW"]), -1e10, dtype=np.float32)
    shifts, dropped, stacked = [], 0, 0
    for plane_idx, i in items:
        img = ser.read(i)
        dx, dy = aligner.evaluate(img)
        shifts.append({"i": i, "dx": dx, "dy": dy})
        if warper is not None:
            src, sdx, sdy = warper.apply(img, dx, dy), 0.0, 0.0
        else:
            src, sdx, sdy = img, dx, dy
        if p["scale"] > 1:
            plane = _frame_plane(src, sdx, sdy, p)
            if plane is None:
                dropped += 1  # outlier shift beyond drizzle headroom
                if "cube" in _W:  # keep the plane slot explicit: all-sentinel
                    _W["cube"].write_plane(
                        plane_idx, np.full((p["accH"], p["accW"]), -1.0,
                                           dtype=np.float32))
                continue
            valid = plane > -0.5
            acc[valid] += plane[valid].astype(np.float64)
            cover += valid
            if p["doTrim"]:
                np.minimum(mn, np.where(valid, plane, 1e10), out=mn)
                np.maximum(mx, np.where(valid, plane, -1e10), out=mx)
            if "cube" in _W:
                _W["cube"].write_plane(plane_idx, plane)
        else:
            if warper is None:
                src = translate(src, -sdx, -sdy)
            acc += src
            cover += 1
            if p["doTrim"]:
                np.minimum(mn, src, out=mn)
                np.maximum(mx, src, out=mx)
        stacked += 1
    return {"acc": acc, "cover": cover, "min": mn, "max": mx,
            "shifts": shifts, "dropped": dropped, "stacked": stacked}


# ------------------------------------------------------------------ main

def run(cfg: dict, config_path: str = "<inline>") -> bool:
    log_path = cfg.get("log") or (cfg["out"] + ".log")
    jl = JobLog(log_path)
    log, progress = jl.log, jl.progress
    t0 = time.time()
    cache_path = cfg["out"] + ".framecache"
    try:
        best_fraction = _cfg(cfg, "bestFraction")
        max_frames = _cfg(cfg, "maxFrames")
        min_frames = _cfg(cfg, "minFrames")
        scale = _cfg(cfg, "drizzle")
        channel = _cfg(cfg, "channel")
        local_align = bool(_cfg(cfg, "localAlign"))
        engine = _cfg(cfg, "drizzleEngine")
        interp = ("nearest" if _cfg(cfg, "drizzleInterpolation") == "nearest"
                  else "bicubic")
        margin = round(_cfg(cfg, "drizzleMargin") * scale) if scale > 1 else 0

        log("ser-stack starting (lunation python port)")
        log(f"config: {config_path}")
        log(f"input:  {cfg['ser']}")
        log(f"output: {cfg['out']}")
        log(f"params: bestFraction={best_fraction} minFrames={min_frames}"
            f" maxFrames={max_frames} drizzle={scale} margin={margin}"
            f" channel={channel}"
            f" alignOnGradient={bool(cfg.get('alignOnGradient'))}"
            f" localAlign={local_align} drizzleEngine={engine}")

        ser = SerReader(cfg["ser"], channel)
        h = ser.header
        log(f"SER: {ser.width}x{ser.height} depth={h.depth}"
            f" frames={ser.frame_count} colorId={h.color_id}"
            + (" (Bayer CFA)" if h.bayer else
               " (interleaved RGB)" if h.rgb else " (mono)"))
        n_frames = ser.frame_count

        workers = cfg.get("workers") or min(6, max(1, (os.cpu_count() or 4) - 2))

        # ---------------- pass 1: sharpness ranking ----------------
        tq = time.time()
        quality = [0.0] * n_frames
        idx_chunks = [list(range(a, min(a + 64, n_frames)))
                      for a in range(0, n_frames, 64)]
        if workers > 1 and n_frames >= 128:
            with ProcessPoolExecutor(
                    max_workers=workers, initializer=_init_quality,
                    initargs=(cfg["ser"], channel)) as pool:
                done = 0
                futs = [pool.submit(_quality_chunk, c) for c in idx_chunks]
                for fu in as_completed(futs):
                    for i, q in fu.result():
                        quality[i] = q
                    done += 1
                    k = min(done * 64, n_frames)
                    progress(k, n_frames, "quality")
        else:
            _init_quality(cfg["ser"], channel)
            for c in idx_chunks:
                for i, q in _quality_chunk(c):
                    quality[i] = q
                progress(min(c[-1] + 1, n_frames), n_frames, "quality")
        log(f"quality pass done in {time.time() - tq:.1f} s")

        ranked = sorted(range(n_frames), key=lambda i: -quality[i])
        n_select = min(max(int(np.ceil(best_fraction * n_frames)),
                           min_frames), max_frames, n_frames)
        selected = ranked[:n_select]
        log(f"selected best {n_select} frames; q range"
            f" {quality[selected[-1]]:.3e} .. {quality[selected[0]]:.3e}"
            f" (worst in file: {quality[ranked[-1]]:.3e})")

        # ---------------- pass 2: align + stack ----------------
        ref_index = selected[0]
        ref = ser.read(ref_index)
        log(f"reference frame: {ref_index} (q={quality[ref_index]:.3e})")

        rejection = cfg.get("rejection") or (
            "none" if cfg.get("rejectExtremes") is False else "sigma")
        if rejection == "sigma" and (n_select < 25 or scale <= 1):
            log("rejection: sigma requested but "
                + ("too few frames" if n_select < 25 else "not drizzling")
                + " — falling back to minmax")
            rejection = "minmax"
        do_trim = rejection == "minmax" and n_select >= 25
        do_sigma = rejection == "sigma"
        kappa = cfg.get("rejectKappa") or DEFAULTS["rejectKappa"]
        rej_iters = cfg.get("rejectIterations") or DEFAULTS["rejectIterations"]

        acc_w = ser.width * scale + 2 * margin
        acc_h = ser.height * scale + 2 * margin
        p = {"scale": scale, "margin": margin, "accW": acc_w, "accH": acc_h,
             "interp": interp, "engine": engine, "doTrim": do_trim,
             "alignOnGradient": bool(cfg.get("alignOnGradient")),
             "localAlign": local_align, "tileSize": _cfg(cfg, "tileSize"),
             "tileStep": _cfg(cfg, "tileStep"), "clampPx": _cfg(cfg, "clampPx"),
             "cache_path": cache_path if do_sigma else None}

        # reference plane (plane 0)
        _init_stack(cfg["ser"], channel, ref_index, p)
        if local_align:
            log(f"local warp correction: {len(_W['warper'].tiles)} tiles"
                f" ({p['tileSize']}px, step {p['tileStep']},"
                f" clamp {p['clampPx']}px)")
        if scale > 1:
            ref_plane = _frame_plane(ref, 0.0, 0.0, p)
            valid = ref_plane > -0.5
            acc = np.where(valid, ref_plane, 0.0).astype(np.float64)
            cover = valid.astype(np.int32)
            mn = np.where(valid, ref_plane, 1e10).astype(np.float32)
            mx = np.where(valid, ref_plane, -1e10).astype(np.float32)
        else:
            acc = ref.astype(np.float64)
            cover = np.ones((acc_h, acc_w), dtype=np.int32)
            mn = ref.copy()
            mx = ref.copy()
        if do_sigma:
            # preallocate so parallel writers can open r+b at any offset
            with open(cache_path, "wb") as f:
                f.truncate(n_select * acc_w * acc_h * 2)
            cube = FrameCube(cache_path, acc_w, acc_h)
            cube.write_plane(0, ref_plane)
            log(f"rejection: sigma-clip kappa={kappa} x{rej_iters}"
                f" — caching aligned frames to {cache_path}"
                f" (~{acc_w * acc_h * 2 * n_select / 1e9:.1f} GB)")

        order = sorted(selected[1:])
        ta = time.time()
        shifts = [{"i": ref_index, "dx": 0.0, "dy": 0.0}]
        n_stacked, n_dropped, chunk_results = 1, 0, []
        items = [(1 + k, i) for k, i in enumerate(order)]
        chunk_size = max(8, int(np.ceil(len(items) / (workers * 4))))
        chunks = [items[a : a + chunk_size]
                  for a in range(0, len(items), chunk_size)]
        if workers > 1 and len(order) >= 32:
            with ProcessPoolExecutor(
                    max_workers=workers, initializer=_init_stack,
                    initargs=(cfg["ser"], channel, ref_index, p)) as pool:
                futs = {pool.submit(_stack_chunk, c): len(c) for c in chunks}
                done = 0
                for fu in as_completed(futs):
                    chunk_results.append(fu.result())
                    done += futs[fu]
                    progress(1 + done, n_select, "stack")
                    log(f"stacked {1 + done}/{n_select}"
                        f" ({(time.time() - ta) / max(done, 1) * 1000:.0f}"
                        " ms/frame)")
        else:
            for c in chunks:
                chunk_results.append(_stack_chunk(c))
                progress(1 + sum(r['stacked'] + r['dropped']
                                 for r in chunk_results), n_select, "stack")

        max_shift = 0.0
        by_index = []
        for r in chunk_results:
            acc += r["acc"]
            cover += r["cover"]
            np.minimum(mn, r["min"], out=mn)
            np.maximum(mx, r["max"], out=mx)
            n_stacked += r["stacked"]
            n_dropped += r["dropped"]
            by_index.extend(r["shifts"])
        by_index.sort(key=lambda s: s["i"])
        shifts.extend(by_index)
        for s in shifts:
            max_shift = max(max_shift, abs(s["dx"]), abs(s["dy"]))

        # ---------------- combine / rejection ----------------
        c = np.s_[margin : acc_h - margin, margin : acc_w - margin] \
            if margin else np.s_[:, :]
        if scale > 1 and do_sigma:
            mean, cnt = cube.combine(kappa, rej_iters, nplanes=n_select)
            cube.remove()
            result = mean[c]
            progress(rej_iters, rej_iters, "reject")
            log(f"rejection: sigma-clip kappa={kappa} x{rej_iters} done")
        elif scale > 1:
            a, cv = acc[c], cover[c]
            if do_trim:
                m3 = cv >= 3
                trimmed = ((a - mn[c] - mx[c])
                           / np.clip(cv - 2, 1, None))
                plain = a / np.clip(cv, 1, None)
                result = np.where(m3, trimmed, plain).astype(np.float32)
                log("extreme-pixel rejection: min/max trimmed (coverage >= 3)")
            else:
                result = (a / np.clip(cv, 1, None)).astype(np.float32)
        elif do_trim:
            result = ((acc - mn - mx)
                      / max(1, n_stacked - 2)).astype(np.float32)
            log("extreme-pixel rejection: min/max trimmed")
        else:
            result = (acc / n_stacked).astype(np.float32)
        if n_dropped > 0:
            log(f"WARNING: dropped {n_dropped} frame(s) with shifts beyond"
                " drizzle margin")
        log(f"align+stack done in {time.time() - ta:.1f} s;"
            f" drizzle x{scale}; max |shift| = {max_shift:.2f} px")

        # ---------------- save ----------------
        write_xisf(cfg["out"], np.clip(result, 0.0, 1.0))
        log(f"saved {cfg['out']}")

        if cfg.get("report"):
            rep = {
                "ser": cfg["ser"], "out": cfg["out"], "frames": n_frames,
                "stacked": n_stacked, "dropped": n_dropped, "drizzle": scale,
                "refIndex": ref_index, "maxShift": max_shift,
                "elapsedSec": time.time() - t0,
                "quality": [float(f"{q:.4e}") for q in quality],
                "shifts": shifts,
            }
            with open(cfg["report"], "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1)
            log(f"report {cfg['report']}")

        progress(1, 1, "done")
        log(f"=== STACK OK ({time.time() - t0:.1f} s) ===")
        return True
    except Exception as e:  # noqa: BLE001 — job boundary, always log+fail
        jl.log(f"*** STACK FAILED: {e}")
        jl.log(traceback.format_exc())
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except OSError:
            pass
        return False
    finally:
        jl.close()
