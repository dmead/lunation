"""Master scheduler end-to-end: a real `python -m lunation stack-one`
child over a synthetic SER, driven through build_dag + Scheduler."""

import json

from lunation.master.pipeline import build_dag
from lunation.master.scheduler import Scheduler

from .test_stack_e2e import _make_capture


def test_run_dag_stacks_real_child(tmp_path):
    ser_path, _, _ = _make_capture(tmp_path)
    auto = tmp_path / "configs" / "auto"
    auto.mkdir(parents=True)
    out_dir = str(tmp_path / "out" / "2026-01-01").replace("\\", "/")
    (auto / "2026-01-01.json").write_text(json.dumps({
        "name": "2026-01-01", "outDir": out_dir,
        "defaults": {"bestFraction": 0.5, "minFrames": 10, "drizzle": 1,
                     "rejection": "minmax"},
        "jobs": [{"id": "L1", "ser": ser_path}]}))

    dag = build_dag(str(tmp_path), workers=1, gif=False)
    assert [j.id for j in dag] == ["2026-01-01:L1"]
    ok = Scheduler(dag, tick_s=0.1, log=lambda *_: None).run()
    assert ok

    log_text = open(f"{out_dir}/L1_stack.log").read()
    assert "=== STACK OK" in log_text
    rep = json.load(open(f"{out_dir}/L1_stack.json"))
    assert rep["stacked"] >= 10
    assert dag[0].progress is not None  # PROGRESS lines were tailed
