"""Master scheduler state machine + DAG builder, on synthetic child
processes (python -c one-liners that write job logs)."""

import json
import os
import sys
import time

from lunation.master.job import Job, State
from lunation.master.pipeline import build_dag, discover_sessions
from lunation.master.scheduler import Scheduler

QUIET = {"log": lambda *_: None}


def child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def writer(log, *lines, hang=False):
    body = ";".join(
        [f"f=open({str(log)!r},'a')"]
        + [f"f.write({ln + chr(10)!r});f.flush()" for ln in lines]
        + (["import time;time.sleep(60)"] if hang else ["f.close()"]))
    return child(body)


def sched(jobs, **kw):
    return Scheduler(jobs, tick_s=0.02, startup_log_s=0.8, stall_s=0.8,
                     **QUIET, **kw)


def test_ok_with_output_check(tmp_path):
    log = str(tmp_path / "a.log")
    out = tmp_path / "a.xisf"
    j = Job(id="a", kind="stack", pool="heavy",
            argv=writer(log, "hello", "PROGRESS 3/9 stack", "=== STACK OK"),
            log_path=log, sentinel_ok="=== STACK OK",
            output_check=lambda: out.exists())
    out.write_bytes(b"x")
    assert sched([j]).run()
    assert j.state == State.OK
    assert j.progress == (3, 9, "stack")


def test_fail_sentinel_skips_dependents(tmp_path):
    log = str(tmp_path / "a.log")
    j = Job(id="a", kind="stack", pool="heavy",
            argv=writer(log, "*** STACK FAILED: bad SER"),
            log_path=log, sentinel_ok="=== STACK OK",
            sentinel_fail="*** STACK FAILED")
    dep = Job(id="fin", kind="finish", pool="heavy", deps=["a"])
    assert not sched([j, dep]).run()
    assert j.state == State.FAILED
    assert "bad SER" in j.reason
    assert dep.state == State.SKIPPED and dep.reason == "dependency failed"


def test_soft_dep_runs_after_failure(tmp_path):
    bad = str(tmp_path / "bad.log")
    gif = str(tmp_path / "gif.log")
    j_bad = Job(id="fin", kind="finish", pool="heavy",
                argv=writer(bad, "*** FINISH FAILED: nope"), log_path=bad,
                sentinel_ok="=== FINISH OK",
                sentinel_fail="*** FINISH FAILED")
    j_gif = Job(id="gif", kind="gif", pool="heavy",
                argv=writer(gif, "=== GIF OK"), log_path=gif,
                sentinel_ok="=== GIF OK", soft_deps=["fin"])
    sched([j_bad, j_gif]).run()
    assert j_bad.state == State.FAILED
    assert j_gif.state == State.OK  # terminal soft dep suffices


def test_exit_without_sentinel(tmp_path):
    log = str(tmp_path / "a.log")
    j = Job(id="a", kind="stack", pool="heavy",
            argv=writer(log, "started but no verdict"),
            log_path=log, sentinel_ok="=== STACK OK")
    assert not sched([j]).run()
    assert j.state == State.FAILED
    assert j.reason == "exited without sentinel"


def test_output_missing(tmp_path):
    log = str(tmp_path / "a.log")
    j = Job(id="a", kind="stack", pool="heavy",
            argv=writer(log, "=== STACK OK"), log_path=log,
            sentinel_ok="=== STACK OK", output_check=lambda: False)
    assert not sched([j]).run()
    assert j.state == State.FAILED and j.reason == "output missing"


def test_stall_watchdog_kills(tmp_path):
    log = str(tmp_path / "a.log")
    j = Job(id="a", kind="stack", pool="heavy",
            argv=writer(log, "one line then freeze", hang=True),
            log_path=log, sentinel_ok="=== STACK OK")
    assert not sched([j]).run()
    assert j.state == State.FAILED and j.reason.startswith("stalled")
    assert j.proc.poll() is not None  # tree actually killed


def test_startup_hang_retries_once(tmp_path):
    j = Job(id="a", kind="stack", pool="heavy",
            argv=child("import time;time.sleep(60)"),
            log_path=str(tmp_path / "never.log"),
            sentinel_ok="=== STACK OK")
    assert not sched([j]).run()
    assert j.state == State.FAILED
    assert j.reason == "no log after startup"
    assert j.retries == 1  # exactly one retry happened


def test_stale_log_deleted_before_launch(tmp_path):
    log = tmp_path / "a.log"
    log.write_text("=== STACK OK\n")  # leftover from a previous run
    j = Job(id="a", kind="stack", pool="heavy",
            argv=writer(str(log), "*** STACK FAILED: real verdict"),
            log_path=str(log), sentinel_ok="=== STACK OK",
            sentinel_fail="*** STACK FAILED")
    assert not sched([j]).run()
    assert j.state == State.FAILED  # stale OK never seen


def test_heavy_pool_serial(tmp_path):
    """cap 1: second job must not start before the first ends."""
    marks = str(tmp_path / "marks.txt")

    def jb(i):
        log = str(tmp_path / f"{i}.log")
        code = (f"import time;m=open({marks!r},'a');m.write('s');m.flush();"
                f"time.sleep(0.3);m.write('e');m.close();"
                f"open({log!r},'a').write('=== STACK OK')")
        return Job(id=f"j{i}", kind="stack", pool="heavy",
                   argv=child(code), log_path=log,
                   sentinel_ok="=== STACK OK")
    jobs = [jb(0), jb(1)]
    assert sched(jobs).run()
    with open(marks) as f:
        assert f.read() == "sese"  # never interleaved


def test_live_cap_suspends_and_resumes(tmp_path):
    import psutil

    def hang_job(i):
        log = str(tmp_path / f"{i}.log")
        return Job(id=f"j{i}", kind="stack", pool="heavy",
                   argv=writer(log, "working", hang=True), log_path=log,
                   sentinel_ok="=== STACK OK")

    jobs = [hang_job(0), hang_job(1)]
    s = Scheduler(jobs, heavy_cap=2, tick_s=0.02, startup_log_s=30,
                  stall_s=30, **QUIET)
    deadline = time.time() + 30
    while not all(j.state == State.RUNNING for j in jobs):
        s.tick()
        assert time.time() < deadline
        time.sleep(0.05)

    s.set_cap("heavy", 1)
    s.tick()
    frozen = [j for j in jobs if j.suspended]
    assert len(frozen) == 1
    assert frozen[0] is jobs[1]  # newest launched suspends first
    assert psutil.Process(frozen[0].proc.pid).status() \
        == psutil.STATUS_STOPPED

    s.set_cap("heavy", 2)
    s.tick()
    assert not any(j.suspended for j in jobs)
    assert psutil.Process(jobs[1].proc.pid).status() \
        != psutil.STATUS_STOPPED

    s.cancel_all()  # kills suspended and live trees alike
    assert all(j.state == State.CANCELLED for j in jobs)
    deadline = time.time() + 10
    while any(j.proc.poll() is None for j in jobs):
        assert time.time() < deadline, "children survived cancel"
        time.sleep(0.05)


def make_root(tmp_path):
    auto = tmp_path / "configs" / "auto"
    auto.mkdir(parents=True)
    ds = {"name": "2026-01-01", "outDir": str(tmp_path / "out/2026-01-01"),
          "defaults": {"bestFraction": 0.1},
          "jobs": [{"id": "L1", "ser": "x_L.ser"},
                   {"id": "R", "ser": "x_R.ser"}]}
    (auto / "2026-01-01.json").write_text(json.dumps(ds))
    fin = {"name": "moon_2026-01-01",
           "outDir": str(tmp_path / "out/2026-01-01/final"),
           "stacksDir": str(tmp_path / "out/2026-01-01")}
    (auto / "finish-2026-01-01.json").write_text(json.dumps(fin))
    ds2 = {"name": "nofin", "outDir": str(tmp_path / "out/nofin"),
           "jobs": [{"id": "OSC1", "ser": "y.ser"}]}
    (auto / "nofin.json").write_text(json.dumps(ds2))
    return str(tmp_path)


def test_discover_sessions(tmp_path):
    root = make_root(tmp_path)
    s = discover_sessions(root)
    assert [x["name"] for x in s] == ["2026-01-01", "nofin"]
    assert s[0]["finish_config"] and not s[1]["finish_config"]
    assert [x["name"] for x in discover_sessions(root, ["nofin"])] \
        == ["nofin"]


def test_build_dag_edges(tmp_path):
    root = make_root(tmp_path)
    dag = {j.id: j for j in build_dag(root)}
    assert set(dag) == {"2026-01-01:L1", "2026-01-01:R",
                        "2026-01-01:finish", "nofin:OSC1", "gif", "encode"}
    fin = dag["2026-01-01:finish"]
    assert sorted(fin.deps) == ["2026-01-01:L1", "2026-01-01:R"]
    assert dag["gif"].soft_deps == ["2026-01-01:finish"]
    assert dag["gif"].deps == []          # best-effort: soft only
    assert dag["encode"].deps == ["gif"]  # hard: no frames, no encode
    assert dag["encode"].pool == "native"

    # stack prepare writes the expanded per-job config (runner.py contract)
    l1 = dag["2026-01-01:L1"]
    l1.prepare(l1)
    cfg_path = f"{tmp_path}/out/2026-01-01/configs/L1.json".replace("\\", "/")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["bestFraction"] == 0.1 and "id" not in cfg
    assert cfg["out"].endswith("L1_stack.xisf")
    assert cfg["log"].endswith("L1_stack.log")
    assert cfg["report"].endswith("L1_stack.json")


def test_build_dag_no_gif(tmp_path):
    dag = build_dag(make_root(tmp_path), gif=False)
    assert all(j.kind in ("stack", "finish") for j in dag)
