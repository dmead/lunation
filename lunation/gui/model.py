"""Entry/job aggregation for the table rows — ports the pure logic of
UI.jsh lpUpdateEntryNode/lpUpdateOverall (Qt-free, tested headless).

An entry row (one capture date item) binds the jobs it spawned; its
progress cell shows a cost-weighted blend of their states. Costs reflect
typical stage runtimes (UI.jsh:390)."""

import time
from dataclasses import dataclass

from ..master.job import Job, State

COST = {"stack": 10, "finish": 6, "gif": 8, "encode": 2, "trim": 1,
        "prep": 2}

_REASON_MAX = 60


@dataclass
class Rollup:
    fraction: float
    label: str
    tooltip: str
    state: State | None  # dominant state for coloring (None = idle/waiting)
    elapsed_s: float     # wall span over the entry's jobs (0 = not started)


def _fraction(j: Job) -> float:
    if j.state == State.OK or j.state.terminal:
        return 1.0
    if j.state == State.RUNNING and j.progress:
        k, n, _ = j.progress
        return k / n if n else 0.0
    return 0.0


def rollup(jobs: list[Job], now: float | None = None) -> Rollup:
    """Cost-weighted aggregate of an entry's jobs (UI.jsh:388-461)."""
    now = time.time() if now is None else now
    total = done = 0.0
    running: Job | None = None
    failed: list[Job] = []
    terminal = cancelled = 0
    t0 = t1 = 0.0
    for j in jobs:
        c = COST.get(j.kind, 1)
        total += c
        done += c * _fraction(j)
        if j.state == State.RUNNING and running is None:
            running = j
        if j.state == State.FAILED:
            failed.append(j)
        if j.state == State.CANCELLED:
            cancelled += 1
        if j.state.terminal:
            terminal += 1
        if j.started_at > 0:
            t0 = j.started_at if t0 == 0 else min(t0, j.started_at)
            end = j.finished_at if j.state.terminal else now
            t1 = max(t1, end)
    frac = done / total if total else 0.0

    tip = ""
    if failed:
        why = failed[0].reason or ""
        tip = why
        if len(why) > _REASON_MAX:
            why = why[:_REASON_MAX - 3] + "…"
        label = "FAILED" + (f": {why}" if why else "")
        if len(failed) > 1:
            label += f"  (+{len(failed) - 1} more)"
        state = State.FAILED
    elif running:
        short = running.id.split(":", 1)[-1]
        stage = f" {running.progress[2]}" if running.progress else ""
        label = f"{round(100 * frac)}%  {short}{stage}"
        state = State.RUNNING
    elif jobs and terminal == len(jobs):
        label = "cancelled" if cancelled == len(jobs) else "done"
        state = State.CANCELLED if cancelled == len(jobs) else State.OK
    else:
        label = "waiting"
        state = None
    return Rollup(frac, label, tip, state,
                  (t1 - t0) if t0 > 0 else 0.0)


def overall_fraction(jobs: list[Job]) -> float:
    """Cost-weighted whole-run progress (UI.jsh:488-505); failed/skipped
    consume their slice so the bar always reaches 100%."""
    total = done = 0.0
    for j in jobs:
        c = COST.get(j.kind, 1)
        total += c
        done += c * _fraction(j)
    return done / total if total else 0.0


def fmt_elapsed(seconds: float) -> str:
    if seconds <= 0:
        return ""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
