"""Desktop GUI — ports pjsr/master/UI.jsh (the Lunation master dialog).

PySide6/Qt, installed via the `gui` extra (`uv tool install lunation[gui]`).
Same architecture as the original dialog: a periodic timer drives
Scheduler.tick() inside the GUI event loop (no worker threads — the
scheduler is non-blocking by design), one checkable entry table whose rows
aggregate their bound jobs, a preview pane, and a log tail following the
selected entry's most interesting job.
"""
