"""`python -m lunation` — how the master scheduler launches its job
children (same interpreter/venv as the scheduler, no PATH lookup)."""

from .cli import main

main()
