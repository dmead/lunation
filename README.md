# Lunation (standalone)

PixInsight-free port of [Lunation for PixInsight](https://github.com/dmead/lunation-pixinsight) — a
lunar lucky-imaging pipeline (SER decode → quality ranking → sub-pixel
registration → drizzle stacking → finishing → phase-ordered lunation
animation) as a plain Python package.

Status: **M0–M4 complete** — the whole pipeline runs without PixInsight or
Node: numeric core (M0), drizzle stacking (M1), lunation assembly + encode
(M2), finishing chain (M3), and the orchestrating scheduler + packaging
(M4). See `docs/PORT-PLAN.md` for the milestone plan and validation bars.

## Install

```bash
uv tool install lunation          # CLI only
uv tool install 'lunation[gui]'   # + desktop GUI (Qt)
lunation --help
```

One command, no PixInsight, no Node. `ffmpeg` on PATH is needed for
`avi2ser` and the mp4 encode.

## Desktop GUI

```bash
lunation gui [--root <dir>]
```

Port of the old in-PixInsight master dialog: add search paths and **Scan**
for SER captures and finished images (channel inference, date grouping,
moon-phase labels), review/edit optics per row (double-click FL/px cells —
drizzle is derived from sampling), check what to run, **Start**. Per-row
progress bars aggregate each entry's jobs, the preview pane shows SER
frame 0 / finished images (auto-stretched), a log pane tails the selected
job, and Pause / Cancel entry / Cancel all control the run. Generated
session configs land in `<root>/configs/auto/` — an existing (hand-tuned)
config always wins and is never overwritten.

## Run

```bash
# full pipeline over a production root (configs/auto/*.json sessions):
# stack -> finish -> gif frames (soft deps) -> encode
lunation run <root> [--sessions 2026-06-30] [--jobs 1] [--no-gif]

# individual stages (same configs/artifacts as the old pipeline)
lunation stack --config configs/<dataset>.json [--out-root out/py/...]
lunation finish configs/auto/finish-<name>.json
lunation render <framesDir> --root <root>
lunation encode <framesDir>
lunation trim <in.ser> <out.ser> <keep> <log>
lunation avi2ser <in.avi> <out.ser>
```

Job success/failure is read from the per-job logs (`=== STACK OK`,
`*** FINISH FAILED: …`), the same sentinel contract the PixInsight-era
tailers used, so stages still mix freely with the old pipeline.

## Development

```bash
# uv manages everything incl. Python itself (no system Python needed)
. scripts/py.sh      # keeps uv caches off C: locally
uv sync
uv run pytest -q
```
