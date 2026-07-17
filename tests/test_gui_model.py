"""Entry rollup math for the GUI table (Qt-free)."""

from lunation.gui.model import fmt_elapsed, overall_fraction, rollup
from lunation.master.job import Job, State


def J(kind, state, progress=None, reason="", t0=0.0, t1=0.0):
    j = Job(id=f"s:{kind}", kind=kind, pool="heavy")
    j.state = state
    j.progress = progress
    j.reason = reason
    j.started_at = t0
    j.finished_at = t1
    return j


def test_rollup_running_label_and_fraction():
    jobs = [J("stack", State.OK), J("finish", State.RUNNING, (3, 9, "deconv"))]
    r = rollup(jobs, now=100.0)
    # stack(10) done + finish(6) at 3/9 -> (10+2)/16
    assert abs(r.fraction - 12 / 16) < 1e-9
    assert r.label == "75%  finish deconv"
    assert r.state == State.RUNNING


def test_rollup_failed_truncates_reason():
    long = "x" * 80
    jobs = [J("stack", State.FAILED, reason=long), J("stack", State.FAILED)]
    r = rollup(jobs)
    assert r.label.startswith("FAILED: " + "x" * 57 + "…")
    assert r.label.endswith("(+1 more)")
    assert r.tooltip == long
    assert r.state == State.FAILED


def test_rollup_done_cancelled_waiting():
    assert rollup([J("stack", State.OK)]).label == "done"
    assert rollup([J("stack", State.CANCELLED)]).label == "cancelled"
    assert rollup([J("stack", State.PENDING)]).label == "waiting"
    assert rollup([J("stack", State.PENDING)]).state is None


def test_rollup_elapsed_span():
    jobs = [J("stack", State.OK, t0=100.0, t1=160.0),
            J("finish", State.RUNNING, t0=150.0)]
    r = rollup(jobs, now=222.0)
    assert r.elapsed_s == 122.0  # 100 -> now
    assert fmt_elapsed(r.elapsed_s) == "02:02"


def test_overall_fraction_cost_weighted():
    jobs = [J("stack", State.OK), J("encode", State.PENDING)]
    assert abs(overall_fraction(jobs) - 10 / 12) < 1e-9
    # failed consumes its slice: bar reaches 1.0 on all-terminal
    jobs = [J("stack", State.OK), J("gif", State.FAILED)]
    assert overall_fraction(jobs) == 1.0
