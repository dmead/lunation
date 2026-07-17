"""Pipeline orchestration — ports pjsr/master/ (Scheduler/Job/Pipeline).

One in-process scheduler replaces the Node launchers and the PJSR master
dialog. Jobs are child processes of THIS package's own CLI (no PixInsight,
no cmd wrappers), so the instance-slot machinery of the original — launch
stagger, pi-launch file lock, PID resolve/minimize — is deleted by
construction. Everything contract-shaped survives: the 8-state job machine,
hard deps (all-OK-else-skip) vs soft deps (all-terminal, outcome ignored),
log-sentinel verdicts, log-growth watchdogs, and the artifact paths.
"""

from .job import Job, State
from .scheduler import Scheduler

__all__ = ["Job", "State", "Scheduler"]
