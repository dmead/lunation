"""Dataset fan-out — ports scripts/run-planetary.mjs from the old repo.

Consumes the same dataset config schema, writes the same per-job expanded
configs to <outDir>/configs/<id>.json, and produces the same artifact paths
(<outDir>/<id>_stack.{xisf,log,json}). Jobs run sequentially by default —
parallelism lives INSIDE each job (frame workers), which replaces the old
child-PixInsight-instance concurrency.
"""

import json
import os

from . import stacker


def run_dataset(config_path: str, only: list[str] | None = None,
                out_root: str | None = None,
                workers: int | None = None) -> bool:
    ds = stacker.load_config(config_path)
    out_dir = (out_root or ds["outDir"]).replace("\\", "/")
    cfg_dir = os.path.join(out_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)

    jobs = ds["jobs"]
    if only:
        jobs = [j for j in jobs if j["id"] in only]
    if not jobs:
        raise SystemExit("no jobs selected")
    for j in jobs:
        if not os.path.exists(j["ser"]):
            raise SystemExit(f"missing SER for {j['id']}: {j['ser']}")

    print(f"{ds.get('name', config_path)}: {len(jobs)} job(s)")
    results = []
    for job in jobs:
        job_cfg = {**ds.get("defaults", {}), **job,
                   "out": f"{out_dir}/{job['id']}_stack.xisf",
                   "log": f"{out_dir}/{job['id']}_stack.log",
                   "report": f"{out_dir}/{job['id']}_stack.json"}
        job_id = job_cfg.pop("id")
        if workers:
            job_cfg["workers"] = workers
        cfg_path = os.path.join(cfg_dir, f"{job_id}.json").replace("\\", "/")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(job_cfg, f, indent=2)
        print(f"[{job_id}] starting: {job_cfg['ser']}")
        import time

        t = time.time()
        ok = stacker.run(job_cfg, cfg_path)
        print(f"[{job_id}] {'OK' if ok else 'FAILED'}"
              f" in {time.time() - t:.0f}s")
        results.append((job_id, ok))

    failed = [jid for jid, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} jobs OK")
    if failed:
        print(f"failed: {', '.join(failed)}")
        return False
    return True
