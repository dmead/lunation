"""Tick scheduler — ports pjsr/master/Scheduler.jsh.

Each tick polls running jobs, promotes pending ones whose deps resolved,
and launches at most one heavy + one native job (Scheduler.jsh:132-244).
Deliberately DELETED from the original: launch stagger, pi-launch file
lock, spawn-timeout, PID resolve — all existed only because two PixInsight
startups race a shared-memory instance slot (Scheduler.jsh:7-9). Our
children are plain CLI processes owned by this one scheduler.

Watchdogs (in-process timeouts, same thresholds as Scheduler.jsh:12-21):
- startup: no log bytes STARTUP_LOG_S after spawn → kill; retry once
  (the machine keeps the master's single-retry semantics), then FAILED.
- stall: log frozen STALL_S while running → kill, FAILED, no retry.
- exit without sentinel → one last poll, then OK/FAILED.
"""

import os
import subprocess
import sys
import time

from .job import NEVER, Job, State

TICK_S = 0.75          # UI.jsh:18 LP_TICK_SEC
STARTUP_LOG_S = 120.0  # LP_STARTUP_LOG_MS
STALL_S = 600.0        # LP_STALL_MS
POOL_CAPS = {"heavy": 1, "native": 1}  # jobs mostly serial: frame workers
                                       # inside a job replace child-PI fanout


def _kill_tree(proc) -> None:
    """Ports kill-tree.cmd: a job's own worker pool dies with it."""
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _tree(proc):
    import psutil

    try:
        p = psutil.Process(proc.pid)
        return [p, *p.children(recursive=True)], psutil.Error
    except psutil.Error:
        return [], psutil.Error


def _suspend_tree(proc) -> None:
    """Freeze a job AND its frame workers (a stack job is a process
    tree — suspending only the parent leaves the workers crunching)."""
    procs, err = _tree(proc)
    for p in procs:
        try:
            p.suspend()
        except err:
            pass


def _resume_tree(proc) -> None:
    procs, err = _tree(proc)
    for p in procs:
        try:
            p.resume()
        except err:
            pass


class Scheduler:
    def __init__(self, jobs: list[Job], heavy_cap: int | None = None,
                 tick_s: float = TICK_S,
                 startup_log_s: float = STARTUP_LOG_S,
                 stall_s: float = STALL_S, log=print):
        self.jobs = jobs
        self.by_id = {j.id: j for j in jobs}
        self.caps = dict(POOL_CAPS)
        if heavy_cap:
            self.caps["heavy"] = heavy_cap
        self.tick_s = tick_s
        self.startup_log_s = startup_log_s
        self.stall_s = stall_s
        self.log = log
        self.paused = False  # UI Pause: polls continue, launches don't

    # -- dependency resolution (Scheduler.jsh:100-129) ------------------

    def promote(self) -> None:
        for j in self.jobs:
            if j.state != State.PENDING:
                continue
            deps = [self.by_id.get(d) for d in j.deps]
            if any(d is None or d.state in
                   (State.FAILED, State.CANCELLED, State.SKIPPED)
                   for d in deps):
                j.set_state(State.SKIPPED, "dependency failed")
                self._note(j)
            elif (all(d.state == State.OK for d in deps)
                  and all(s is None or s.state.terminal
                          for s in (self.by_id.get(x)
                                    for x in j.soft_deps))):
                j.set_state(State.READY)

    # -- launch (Scheduler.jsh:226-277) ----------------------------------

    def set_cap(self, pool: str, n: int) -> None:
        """Live concurrency change: takes effect next tick — excess
        running jobs get suspended, headroom resumes/launches."""
        self.caps[pool] = max(1, n)

    def _running_count(self, pool: str) -> int:
        return sum(1 for j in self.jobs if j.pool == pool
                   and j.state in (State.LAUNCHING, State.RUNNING)
                   and not j.suspended)

    def _rebalance(self, pool: str) -> None:
        """Enforce the (possibly just-changed) cap on live jobs: suspend
        the newest excess trees, resume suspended ones as slots free up —
        resumed jobs take priority over launching new ones."""
        cap = self.caps[pool]
        live = [j for j in self.jobs
                if j.pool == pool and j.state == State.RUNNING]
        active = [j for j in live if not j.suspended]
        if len(active) > cap:
            for j in sorted(active, key=lambda x: x.started_at,
                            reverse=True)[:len(active) - cap]:
                _suspend_tree(j.proc)
                j.suspended = True
                self.log(f"[{j.id}] suspended (cap {cap})")
        elif len(active) < cap:
            for j in [x for x in live if x.suspended][:cap - len(active)]:
                _resume_tree(j.proc)
                j.suspended = False
                # the log was frozen with the process — don't let the
                # stall watchdog count the suspension against it
                j.log_last_growth = time.time()
                self.log(f"[{j.id}] resumed")

    def _launch(self, j: Job) -> None:
        try:
            if j.prepare:
                j.prepare(j)
            j.set_state(State.LAUNCHING)
            # stale-log defense (Pipeline.jsh:38-39): a leftover sentinel
            # must never satisfy the poller before the stage truncates it
            if j.log_path:
                os.makedirs(os.path.dirname(os.path.abspath(j.log_path)),
                            exist_ok=True)
                if os.path.exists(j.log_path):
                    os.remove(j.log_path)
            out = open(f"{j.log_path}.out", "w") if j.log_path else \
                subprocess.DEVNULL
            j.proc = subprocess.Popen(j.argv, stdout=out, stderr=out)
            if out is not subprocess.DEVNULL:
                out.close()
        except Exception as e:  # noqa: BLE001 — launch boundary
            j.set_state(State.FAILED, f"launch threw: {e}")
            self._note(j)
            return
        j.started_at = j.log_last_growth = time.time()
        j.set_state(State.RUNNING)
        self.log(f"[{j.id}] launched: {' '.join(j.argv)}")

    # -- poll (Scheduler.jsh:140-224) -------------------------------------

    def _poll(self, j: Job) -> None:
        if j.proc is None:
            j.set_state(State.FAILED, "spawn failed")
            self._note(j)
            return
        alive = j.proc.poll() is None
        verdict = j.poll_log()
        now = time.time()

        if verdict == "ok":
            if j.output_ok():
                j.set_state(State.OK)
            else:
                j.set_state(State.FAILED, "output missing")
            self._note(j)
        elif verdict == "failed":
            _kill_tree(j.proc)
            j.set_state(State.FAILED,
                        j.fail_message or "job reported failure")
            self._note(j)
        elif not alive:
            j.set_state(State.FAILED, "exited without sentinel")
            self._note(j)
        elif j.log_offset == 0 and now - j.started_at > self.startup_log_s:
            _kill_tree(j.proc)
            if j.retries < 1:
                j.retries += 1
                j.reset()
                j.set_state(State.READY, "retry after startup hang")
                self.log(f"[{j.id}] no log after startup — retrying")
            else:
                j.set_state(State.FAILED, "no log after startup")
                self._note(j)
        elif j.log_offset > 0 and now - j.log_last_growth > self.stall_s:
            _kill_tree(j.proc)
            j.set_state(State.FAILED,
                        f"stalled (log frozen {self.stall_s:.0f} s)")
            self._note(j)

    # -- main loop --------------------------------------------------------

    def tick(self) -> None:
        for j in self.jobs:
            # suspended jobs are frozen: no polling, no watchdogs
            if j.state in (State.LAUNCHING, State.RUNNING) \
                    and not j.suspended:
                self._poll(j)
        self.promote()
        if self.paused:
            return
        for pool in self.caps:
            self._rebalance(pool)
        launched = set()
        for j in self.jobs:  # one launch per pool per tick
            if (j.state == State.READY and j.pool not in launched
                    and self._running_count(j.pool) < self.caps[j.pool]):
                launched.add(j.pool)
                self._launch(j)

    def done(self) -> bool:
        return all(j.state.terminal for j in self.jobs)

    def run(self) -> bool:
        try:
            while not self.done():
                self.tick()
                if not self.done():
                    time.sleep(self.tick_s)
        except KeyboardInterrupt:
            self.cancel_all("user cancel")
        counts: dict[str, int] = {}
        for j in self.jobs:
            counts[j.state.value] = counts.get(j.state.value, 0) + 1
        self.log("summary: " + ", ".join(f"{n} {s}"
                                         for s, n in sorted(counts.items())))
        # skips only ever cascade from a failure/cancel, so all-OK is the
        # one success shape
        return all(j.state == State.OK for j in self.jobs)

    def cancel(self, j: Job, reason: str = "user cancel") -> None:
        if not j.state.terminal:
            if j.state in (State.LAUNCHING, State.RUNNING):
                _kill_tree(j.proc)
            j.set_state(State.CANCELLED, reason)
            self._note(j)

    def cancel_all(self, reason: str = "user cancel") -> None:
        for j in self.jobs:
            self.cancel(j, reason)

    def _note(self, j: Job) -> None:
        extra = f" — {j.reason}" if j.reason else ""
        self.log(f"[{j.id}] {j.state.value}{extra}")
