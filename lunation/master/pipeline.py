"""DAG builder — ports pjsr/master/Pipeline.jsh + lunar-master.js:52-76.

Sessions come from `<root>/configs/auto/*.json` (skipping `finish-*.json`),
each pairing a `finish-<base>.json` when present (Pipeline.jsh:89-129).
Edges: stack jobs are roots; finish hard-deps ALL of its session's stacks
(one failed stack skips the finish); the gif render soft-deps every finish
— it becomes ready once they are merely TERMINAL, success not required,
because it collects whatever finals actually landed on disk at launch
time; encode hard-deps the gif.

Deviation from the original (both ends of the contract are now ours):
encode is ONE job producing lunation.mp4 + lunation.gif via
assemble/encode.py, not two ffmpeg jobs tailing `-progress` files —
its log carries ordinary `=== ENCODE OK` sentinels like every other stage.
The survey pipeline stays out of scope (port plan).
"""

import datetime
import glob
import json
import os
import sys

from .job import NEVER, Job

# frozen app (PyInstaller): the exe IS the CLI when given args — there is
# no python/-m inside a bundle
_CLI = ([sys.executable] if getattr(sys, "frozen", False)
        else [sys.executable, "-m", "lunation"])


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_sessions(root: str,
                      only: list[str] | None = None) -> list[dict]:
    """[{name, config, finish_config|None}] from configs/auto (the
    gen-configs contract, Pipeline.jsh:89-129)."""
    auto = os.path.join(root, "configs", "auto")
    sessions = []
    for p in sorted(glob.glob(os.path.join(auto, "*.json"))):
        base = os.path.basename(p)[:-len(".json")]
        if base.startswith(("finish-", "prep-")):
            continue
        if only and base not in only:
            continue
        fin = os.path.join(auto, f"finish-{base}.json")
        sessions.append({"name": base, "config": p.replace("\\", "/"),
                         "finish_config": fin.replace("\\", "/")
                         if os.path.exists(fin) else None})
    return sessions


def session_jobs(name: str, config_path: str,
                 finish_config: str | None = None,
                 workers: int | None = None) -> list[Job]:
    """Stack jobs (+ finish when configured) for one session — the shared
    builder behind both `lunation run` and the GUI's Start
    (Pipeline.jsh:16-85)."""
    ds = _load(config_path)
    out_dir = ds["outDir"].replace("\\", "/")
    jobs = []
    for spec in ds["jobs"]:
        jid = spec["id"]
        cfg = {**ds.get("defaults", {}), **spec,
               "out": f"{out_dir}/{jid}_stack.xisf",
               "log": f"{out_dir}/{jid}_stack.log",
               "report": f"{out_dir}/{jid}_stack.json"}
        cfg.pop("id")
        if workers:
            cfg["workers"] = workers
        cfg_path = f"{out_dir}/configs/{jid}.json"

        def prepare(j, cfg=cfg, cfg_path=cfg_path):
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        out_xisf = cfg["out"]
        jobs.append(Job(
            id=f"{name}:{jid}", kind="stack", pool="heavy",
            argv=[*_CLI, "stack-one", cfg_path],
            log_path=cfg["log"],
            sentinel_ok="=== STACK OK", sentinel_fail="*** STACK FAILED",
            prepare=prepare,
            output_check=lambda p=out_xisf: os.path.exists(p),
            meta={"ser": spec.get("ser", "").replace("\\", "/")}))
    if finish_config:
        fin = _load(finish_config)
        fin_dir = (fin.get("outDir") or f"{out_dir}/final").replace("\\", "/")
        fin_name = fin.get("name") or f"moon_{ds.get('name', name)}"
        final = f"{fin_dir}/{fin_name}.xisf"
        jobs.append(Job(
            id=f"{name}:finish", kind="finish", pool="heavy",
            argv=[*_CLI, "finish", finish_config],
            log_path=fin.get("log") or f"{fin_dir}/finish.log",
            sentinel_ok="=== FINISH OK", sentinel_fail="*** FINISH FAILED",
            deps=[j.id for j in jobs],
            output_check=lambda p=final: os.path.exists(p)))
    return jobs


def prep_job(prep: dict) -> Job:
    """Prep-normalize one finished image (Discovery.jsh:197-229). Never
    fails the run: per-item failures log `PREP FAILED` without the
    sentinel, so the only failure mode is exit-without-sentinel."""
    return Job(
        id=f"prep:{prep['name']}", kind="prep", pool="heavy",
        argv=[*_CLI, "prep", "--config", prep["config"]],
        log_path=prep["log"],
        sentinel_ok="PREP DONE", sentinel_fail=NEVER,
        output_check=lambda p=prep["out"]: os.path.exists(p))


def lunation_jobs(out_root: str, finish_ids: list[str], out_px: int = 1080,
                  canvas: int = 2300) -> list[Job]:
    """gif render (soft-deps the finishes) + encode (lunar-master.js:52-81,
    Encode.jsh). `out_root` is the production out/ tree the render scans
    for finals."""
    out_root = out_root.replace("\\", "/").rstrip("/")
    stamp = datetime.datetime.now().strftime("run-%Y%m%d-%H%M")
    frames_dir = f"{out_root}/lunation/{stamp}"
    gif = Job(
        id="gif", kind="gif", pool="heavy",
        argv=[*_CLI, "render", frames_dir, "--scan-out", out_root,
              "--out-px", str(out_px), "--canvas", str(canvas),
              "--no-stamp"],
        log_path=f"{frames_dir}/gif-frames.log",
        sentinel_ok="=== GIF OK", sentinel_fail="*** GIF FAILED",
        soft_deps=list(finish_ids))
    encode = Job(
        id="encode", kind="encode", pool="native",
        argv=[*_CLI, "encode", frames_dir],
        log_path=f"{frames_dir}/encode.log",
        sentinel_ok="=== ENCODE OK", sentinel_fail="*** ENCODE FAILED",
        deps=["gif"],
        output_check=lambda: os.path.exists(f"{frames_dir}/lunation.mp4")
        and os.path.exists(f"{frames_dir}/lunation.gif"))
    return [gif, encode]


def build_dag(root: str, only: list[str] | None = None,
              workers: int | None = None, gif: bool = True,
              out_px: int = 1080, canvas: int = 2300) -> list[Job]:
    root = root.replace("\\", "/").rstrip("/")
    sessions = discover_sessions(root, only)
    if not sessions:
        raise SystemExit(f"no sessions under {root}/configs/auto")
    jobs: list[Job] = []
    finish_ids: list[str] = []
    for s in sessions:
        sj = session_jobs(s["name"], s["config"], s["finish_config"],
                          workers)
        jobs += sj
        finish_ids += [j.id for j in sj if j.kind == "finish"]
    if gif:
        jobs += lunation_jobs(f"{root}/out", finish_ids, out_px, canvas)
    return jobs
