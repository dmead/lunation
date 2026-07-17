"""Job state machine + log poller — ports pjsr/master/Job.jsh.

Success and failure are read from the job's LOG FILE via sentinel
substrings, never from the exit code (Job.jsh:6-8: that contract predates
us — automation-mode children never forwarded a console, so every tailer
in the old pipeline watches logs). `poll_log()` reads only newly appended
bytes; any growth feeds the stall watchdog (Job.jsh:134-204).
"""

import enum
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable


class State(enum.Enum):
    PENDING = "pending"
    READY = "ready"
    LAUNCHING = "launching"
    RUNNING = "running"
    OK = "ok"
    FAILED = "FAILED"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self in (State.OK, State.FAILED, State.CANCELLED,
                        State.SKIPPED)


_PROGRESS_RE = re.compile(r"PROGRESS (\d+)/(\d+) (\S+)")
_FAIL_MSG_RE = re.compile(r"FAILED[^:]*:\s*(.+)")

# sentinel that never matches — for jobs whose failure mode is only
# "exited without sentinel" (Job.jsh's @@never@@ idiom, Encode.jsh:67)
NEVER = "@@never@@"


@dataclass
class Job:
    id: str
    kind: str
    pool: str  # "heavy" | "native"
    argv: list[str] = field(default_factory=list)
    log_path: str = ""
    sentinel_ok: str = ""
    sentinel_fail: str = "***"
    deps: list[str] = field(default_factory=list)       # must ALL be OK
    soft_deps: list[str] = field(default_factory=list)  # must be terminal
    # runs synchronously just before spawn (Scheduler.jsh:250-252) — this
    # deferral is load-bearing: inputs of gif/encode only exist once their
    # upstream jobs have finished, so argv/inputs are computed at launch
    prepare: Callable[["Job"], None] | None = None
    output_check: Callable[[], bool] | None = None
    meta: dict = field(default_factory=dict)  # UI binding (e.g. ser path)

    state: State = State.PENDING
    reason: str = ""
    proc: object = None  # subprocess.Popen
    suspended: bool = False  # frozen by a live cap decrease, not dead
    retries: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    log_offset: int = 0
    log_last_growth: float = 0.0
    progress: tuple[int, int, str] | None = None
    fail_message: str = ""
    log_tail: list[str] = field(default_factory=list)  # last N lines (UI)
    _verdict: str | None = None

    TAIL_LINES = 200

    def set_state(self, state: State, reason: str = "") -> None:
        if state == self.state:
            return
        self.state = state
        self.reason = reason
        if state.terminal:
            self.finished_at = time.time()

    def reset(self) -> None:
        """Clear run residue for a retry (Job.jsh:72-84). `retries` is the
        retry counter itself and survives — the startup-hang retry fires at
        most once."""
        self.proc = None
        self.started_at = 0.0
        self.finished_at = 0.0
        self.log_offset = 0
        self.log_last_growth = 0.0
        self.progress = None
        self.fail_message = ""
        self.log_tail = []
        self._verdict = None

    def poll_log(self) -> str | None:
        """Scan newly appended log bytes; returns sticky verdict
        "ok" | "failed" | None (Job.jsh:134-204)."""
        try:
            size = os.path.getsize(self.log_path)
        except OSError:
            return self._verdict
        if size < self.log_offset:  # stage truncated the stale log
            self.log_offset = 0
        if size == self.log_offset:
            return self._verdict
        with open(self.log_path, "rb") as f:
            f.seek(self.log_offset)
            text = f.read(size - self.log_offset).decode("utf-8", "replace")
        self.log_offset = size
        self.log_last_growth = time.time()
        self.log_tail = (self.log_tail
                         + text.splitlines())[-self.TAIL_LINES:]
        for m in _PROGRESS_RE.finditer(text):
            self.progress = (int(m.group(1)), int(m.group(2)), m.group(3))
        if self._verdict is None:
            if self.sentinel_ok and self.sentinel_ok in text:
                self._verdict = "ok"
            elif self.sentinel_fail != NEVER and self.sentinel_fail in text:
                self._verdict = "failed"
                m = _FAIL_MSG_RE.search(text)
                if m:
                    self.fail_message = m.group(1).strip()
        return self._verdict

    def output_ok(self) -> bool:
        return self.output_check is None or self.output_check()
