# Lunation (standalone)

PixInsight-free port of [Lunation](https://github.com/dmead/lunation) — a
lunar lucky-imaging pipeline (SER decode → quality ranking → sub-pixel
registration → drizzle stacking → finishing → phase-ordered lunation
animation) as a plain Python package.

Status: **M0 (numeric core) complete** — registration, warping, frame-cube
rejection, SER/XISF/TIFF I/O, optics derivations, all under synthetic tests;
XISF output verified bit-exact against PixInsight's own reader. Stacking
(M1) is next. See `docs/PORT-PLAN.md` for the full milestone plan.

## Setup

```bash
# uv manages everything incl. Python itself (no system Python needed)
. scripts/py.sh      # keeps uv caches off C: locally
uv sync
uv run pytest -q
```
