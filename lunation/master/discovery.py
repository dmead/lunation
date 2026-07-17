"""Capture discovery + config generation — ports pjsr/master/Discovery.jsh
and the drizzle-side of Optics.jsh.

Walks search paths for SER captures and finished single images, infers
channels (filename `_[LRGBSH].ser` letter, else the SER colorId header),
assembles per-date sessions with the fixed job-id rules, and generates
dataset configs under `<root>/configs/auto/`. An existing config always
wins (Discovery.jsh:180-181) — hand-tuned configs are never clobbered.
"""

import glob
import json
import os
import re

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".xisf"}
MIN_IMAGE_BYTES = 300 * 1024      # smaller = thumbnail/byproduct
MAX_WALK_DEPTH = 64               # loop backstop (Discovery.jsh:35-54)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# pipeline byproducts that must not be re-ingested (Discovery.jsh:114-119)
_BYPRODUCT_RE = re.compile(
    r"^\d+_(accept|reject)_|^(frame|seq|lab)_|\.ser\.png$|\.pre\w*\.xisf$",
    re.IGNORECASE)

_SKIP_DIR_RE = re.compile(r"^\.|^#|_files$", re.IGNORECASE)
_SKIP_DIR_NAMES = {"$recycle.bin", "system volume information"}


# ---- optics (Optics.jsh:12-59; full derivations live in optics.py) ------

EQUIPMENT_DEFAULTS = {"aperture": 70, "focalLength": 440, "pixelSize": 3.76}
_GREEN_NM = 550  # representative wavelength for Q (Optics.jsh:13)


def load_equipment(root: str) -> dict:
    try:
        with open(os.path.join(root, "equipment.json"),
                  encoding="utf-8-sig") as f:
            return {**EQUIPMENT_DEFAULTS, **json.load(f)}
    except (OSError, ValueError):
        return dict(EQUIPMENT_DEFAULTS)


def plate_scale(focal_mm: float, pixel_um: float) -> float:
    return 206265 * pixel_um / (focal_mm * 1000)


def derive_drizzle(focal_mm: float, pixel_um: float,
                   aperture_mm: float) -> int:
    f_ratio = focal_mm / aperture_mm
    critical_pixel = _GREEN_NM * 1e-3 * f_ratio / 2
    q = pixel_um / critical_pixel
    return max(1, min(3, round(q)))


def default_optics(root: str) -> dict:
    eq = load_equipment(root)
    return {"focalLength": eq["focalLength"], "pixelSize": eq["pixelSize"],
            "aperture": eq["aperture"],
            "drizzle": derive_drizzle(eq["focalLength"], eq["pixelSize"],
                                      eq["aperture"])}


# ---- filesystem scan ------------------------------------------------------


def _skip_dir(name: str) -> bool:
    return bool(_SKIP_DIR_RE.search(name)) \
        or name.lower() in _SKIP_DIR_NAMES


def walk(top: str):
    """Recursive dir walk with the pipeline's skip rules; yields
    (dirpath, filenames)."""
    top = top.replace("\\", "/").rstrip("/")
    stack = [(top, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            names = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        files = []
        for e in names:
            if e.is_dir(follow_symlinks=False):
                if depth < MAX_WALK_DEPTH and not _skip_dir(e.name):
                    stack.append((f"{d}/{e.name}", depth + 1))
            else:
                files.append(e.name)
        if files:
            yield d, files


def ser_channel(path: str) -> str:
    """Filter letter from `_<X>.ser`, else header colorId → OSC/MONO
    (Discovery.jsh:65-83)."""
    m = re.search(r"_([LRGBSH])\.ser$", os.path.basename(path),
                  re.IGNORECASE)
    if m:
        return m.group(1).upper()
    try:
        with open(path, "rb") as f:
            f.seek(18)
            color_id = int.from_bytes(f.read(4), "little", signed=True)
        return "OSC" if color_id >= 8 else "MONO"
    except OSError:
        return "MONO"


def _date_of(path: str) -> str:
    m = DATE_RE.search(path.replace("\\", "/"))
    return m.group(1) if m else ""


def scan_search_paths(paths: list[str]) -> dict:
    """{"sers": [{path, dir, date, channel}], "images": [{path, date,
    name}]} — SERs grouped per directory, finished images filtered of
    pipeline byproducts (Discovery.jsh:87-144)."""
    sers, images = [], []
    for top in paths:
        for d, files in walk(top):
            for name in files:
                p = f"{d}/{name}"
                low = name.lower()
                if low.endswith(".ser"):
                    sers.append({"path": p, "dir": d, "date": _date_of(p),
                                 "channel": ser_channel(p)})
                    continue
                if os.path.splitext(low)[1] not in IMAGE_EXT:
                    continue
                if _BYPRODUCT_RE.search(name):
                    continue
                try:
                    if os.path.getsize(p) < MIN_IMAGE_BYTES:
                        continue
                except OSError:
                    continue
                images.append({"path": p, "date": _date_of(p),
                               "name": os.path.splitext(name)[0]})
    return {"sers": sers, "images": images}


# ---- session assembly + config generation --------------------------------


def session_from_sers(date: str, ser_paths: list[str]) -> dict:
    """Job list with the fixed id rules (Discovery.jsh:149-170): R/G/B/S/H
    keep bare letters (one per night), L/MONO/OSC dedup by numeric suffix
    (L1/L2, M1, OSC1); chroma gets bestFraction 0.35, luminance-like
    channels get localAlign."""
    counts: dict[str, int] = {}
    jobs = []
    for p in ser_paths:
        ch = ser_channel(p)
        counts[ch] = counts.get(ch, 0) + 1
        if ch in ("R", "G", "B"):
            jobs.append({"id": ch, "ser": p, "channel": ch,
                         "bestFraction": 0.35})
        elif ch in ("S", "H"):
            jobs.append({"id": ch, "ser": p, "localAlign": True})
        elif ch == "OSC":
            jobs.append({"id": f"OSC{counts[ch]}", "ser": p,
                         "channel": "osc"})
        elif ch == "L":
            jobs.append({"id": f"L{counts[ch]}", "ser": p,
                         "localAlign": True})
        else:  # MONO
            jobs.append({"id": f"M{counts[ch]}", "ser": p,
                         "localAlign": True})
    return {"name": date, "jobs": jobs}


def write_discovered_config(session: dict, config_dir: str,
                            output_root: str, drizzle: int = 2) -> str:
    """`<config_dir>/<name>.json`; an existing config is returned untouched
    (hand-tuned configs win, Discovery.jsh:174-194). The GUI passes a
    per-run `<output>/lunation_<id>/` dir so the configuration is stored
    with the output; the CLI passes `<root>/configs/auto` (old contract)."""
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir,
                        f"{session['name']}.json").replace("\\", "/")
    if os.path.exists(path):
        return path
    cfg = {
        "name": session["name"],
        "outDir": f"{output_root.rstrip('/')}/{session['name']}",
        "defaults": {"bestFraction": 0.10, "maxFrames": 400,
                     "minFrames": 20, "alignOnGradient": True,
                     "drizzle": drizzle or 2, "drizzleMargin": 16},
        "concurrency": 2,
        "jobs": session["jobs"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return path


def finish_config_for(root: str, name: str) -> str | None:
    p = os.path.join(root, "configs", "auto",
                     f"finish-{name}.json").replace("\\", "/")
    return p if os.path.exists(p) else None


def new_run_dir(output_root: str) -> str:
    """Mint `<output>/lunation_<id>/` for one GUI run — the PIPP/
    AutoStakkert model: machinery files live in a collision-proof dir
    stored WITH the output, not at a fixed pipeline root."""
    import uuid

    d = f"{output_root.rstrip('/')}/lunation_{uuid.uuid4().hex[:8]}"
    os.makedirs(d, exist_ok=True)
    return d


def find_finish_config(output_root: str, name: str,
                       near: str | None = None) -> str | None:
    """`finish-<name>.json` next to an explicitly added config (`near`),
    else the newest one in any previous `lunation_*` run dir under the
    output."""
    if near:
        p = os.path.join(os.path.dirname(near),
                         f"finish-{name}.json").replace("\\", "/")
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(output_root, "lunation_*",
                                  f"finish-{name}.json"))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime).replace("\\", "/")


def write_prep_config(image: dict, config_dir: str,
                      output_root: str) -> dict:
    """Prep-normalize config for one finished image (Discovery.jsh:197-229).
    Returns {config, out, log, name}."""
    date = image.get("date") or "undated"
    base = re.sub(r"[^\w.-]+", "_", image["name"])
    out_name = f"FIN_{date}_{base}"
    fin_dir = f"{output_root.rstrip('/')}/finished"
    out_path = f"{fin_dir}/{out_name}.xisf"
    log_path = f"{fin_dir}/prep-{out_name}.log"
    os.makedirs(config_dir, exist_ok=True)
    cfg_path = os.path.join(config_dir,
                            f"prep-{out_name}.json").replace("\\", "/")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"targetR": 979, "canvas": 2300, "log": log_path,
                   "items": [{"src": image["path"], "out": out_path}]},
                  f, indent=2)
    return {"config": cfg_path, "out": out_path, "log": log_path,
            "name": out_name}


# ---- lunar phase (Lunation.jsh:8-23, lunar-master.js:264-277) -------------

def phase_name(date: str) -> str:
    from ..assemble.collect import lunar_age

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
        return ""
    age = lunar_age(date)
    if age < 1.0 or age >= 28.5:
        return "new"
    for limit, name in ((6.4, "waxing crescent"), (8.4, "first quarter"),
                        (13.8, "waxing gibbous"), (15.8, "full"),
                        (21.1, "waning gibbous"), (23.1, "last quarter")):
        if age < limit:
            return name
    return "waning crescent"


def list_config_sessions(root: str) -> list[dict]:
    """Existing configs/auto sessions, for adding .json entries."""
    auto = os.path.join(root, "configs", "auto")
    out = []
    for p in sorted(glob.glob(os.path.join(auto, "*.json"))):
        base = os.path.basename(p)[:-len(".json")]
        if base.startswith(("finish-", "prep-")):
            continue
        out.append({"name": base, "config": p.replace("\\", "/"),
                    "finish_config": finish_config_for(root, base)})
    return out
