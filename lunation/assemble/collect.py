"""Lunation input collector — ports scripts/render-lunation.mjs:34-96.

Scans production roots (pipeline sessions, survey stacks, finished ingests)
or explicit --inputs dirs, dedupes to one frame per capture date with the
fixed preference (mono stack > ingest > OSC, then drive), and orders by
synodic lunar age.
"""

import json
import math
import os
import re

SYNODIC = 29.530588
EPOCH_JD = 2451550.26


def julian_day(date_str: str) -> float:
    y, m, d = (int(t) for t in date_str.split("-"))
    a = (14 - m) // 12
    yy = y + 4800 - a
    mm = m + 12 * a - 3
    return (d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100
            + yy // 400 - 32045 + 0.4)  # ~21:00 UTC capture-ish


def lunar_age(date_str: str) -> float:
    return ((julian_day(date_str) - EPOCH_JD) % SYNODIC + SYNODIC) % SYNODIC


KIND_RANK = {"stack": 0, "ingest": 1, "osc": 2}
DRIVE_PREF = {"Z": 0, "H": 1, "S": 2, "V": 3, "Y": 4, "F": 5}


def _rank(e: dict) -> int:
    return (KIND_RANK.get(e["kind"], 9) * 10
            + DRIVE_PREF.get(e["key"][:1], 9))


def _push_xisf(entries, dir_, file, kind):
    if not file.endswith(".xisf") or re.search(r"\.pre\w+\.xisf$", file):
        return
    m = re.search(r"(\d{4}-\d{2}-\d{2})", file)
    if m:
        entries.append({"date": m.group(1),
                        "final": f"{dir_}/{file}",
                        "key": file[: -len(".xisf")], "kind": kind})


def collect(root: str | None = None,
            input_dirs: list[str] | None = None,
            log=print, out_dir: str | None = None) -> list[tuple[str, float]]:
    """Returns [(final_path, age_days)] thinnest -> fullest ('ordered').
    The production tree is `out_dir` if given, else `<root>/out`."""
    entries: list[dict] = []
    if input_dirs:
        for raw in input_dirs:
            d = raw.replace("\\", "/").rstrip("/")
            for f in sorted(os.listdir(d)):
                _push_xisf(entries, d, f, "ingest")
    else:
        if not (root or out_dir):
            raise ValueError("collect needs root, out_dir or input_dirs")
        out = (out_dir or f"{root}/out").replace("\\", "/").rstrip("/")
        # pipeline sessions: out/<date>/final/moon_<date>[_sharp3].xisf
        for d in sorted(os.listdir(out)):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                continue
            base = f"{out}/{d}/final/moon_{d}"
            final = (f"{base}_sharp3.xisf"
                     if os.path.exists(f"{base}_sharp3.xisf")
                     else f"{base}.xisf" if os.path.exists(f"{base}.xisf")
                     else None)
            if final:
                entries.append({"date": d, "final": final,
                                "key": f"Z_{d}", "kind": "stack"})
        # survey sessions (external-drive stacks)
        with open(f"{out}/survey/manifest.json", encoding="utf-8") as f:
            man = json.load(f)
        for e in man:
            if not e.get("ok") or not os.path.exists(e.get("final", "")):
                continue
            d = e.get("date")
            if not d:
                m = re.search(r"_(\d{4}-\d{2}-\d{2})", e["key"])
                d = m.group(1) if m else None
            if d:
                entries.append({
                    "date": d, "final": e["final"], "key": e["key"],
                    "kind": "osc" if e["key"].endswith("_osc") else "stack"})
        # finished-image ingests (prep-normalized)
        for f in sorted(os.listdir(f"{out}/finished")):
            _push_xisf(entries, f"{out}/finished", f, "ingest")

    # one frame per capture date; fixed preference, never per-item tuning
    by_date: dict[str, dict] = {}
    for e in entries:
        cur = by_date.get(e["date"])
        if cur is None:
            by_date[e["date"]] = e
            continue
        win, lose = (e, cur) if _rank(e) < _rank(cur) else (cur, e)
        by_date[e["date"]] = win
        log(f"dedup {e['date']}: {lose['key']} dropped (kept {win['key']})")
    unique = list(by_date.values())
    for e in unique:
        e["age"] = lunar_age(e["date"])
    unique.sort(key=lambda e: e["age"])
    for e in unique:
        log(f"age {e['age']:5.1f}d  {e['key']}  {e['date']}")
    log(f"{len(unique)} inputs")
    return [(e["final"], e["age"]) for e in unique]
