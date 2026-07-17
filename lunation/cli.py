"""Lunation CLI."""

import typer

app = typer.Typer(help="Lunation — standalone lunar lucky-imaging pipeline.")


@app.command()
def version() -> None:
    """Print the package version."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as v

    try:
        typer.echo(v("lunation"))
    except PackageNotFoundError:  # frozen bundle without dist metadata
        typer.echo("unknown (no package metadata)")


@app.command()
def stack(
    config: str = typer.Option(..., help="Dataset config JSON (old schema)"),
    only: str = typer.Option(None, help="Comma-separated job ids"),
    out_root: str = typer.Option(None, help="Override outDir (keeps PI baselines safe)"),
    workers: int = typer.Option(None, help="Frame workers per job (default: cores-2, max 6)"),
) -> None:
    """Stack every SER job of a dataset config."""
    from .stack.runner import run_dataset

    ok = run_dataset(config, only.split(",") if only else None,
                     out_root, workers)
    raise typer.Exit(0 if ok else 1)


@app.command("stack-one")
def stack_one(
    job_config: str = typer.Argument(..., help="Expanded per-job config JSON"),
) -> None:
    """Run a single stack job from an expanded config (ser-stack.js parity)."""
    from .stack import stacker

    ok = stacker.run(stacker.load_config(job_config), job_config)
    raise typer.Exit(0 if ok else 1)


@app.command()
def avi2ser(
    in_avi: str = typer.Argument(...),
    out_ser: str = typer.Argument(...),
) -> None:
    """Repackage an AVI capture as a SER (decode only, no processing)."""
    from .io.avi import convert

    typer.echo(convert(in_avi, out_ser))


@app.command()
def trim(
    in_ser: str = typer.Argument(...),
    out_ser: str = typer.Argument(...),
    keep: float = typer.Argument(..., help="Fraction of frames to keep"),
    log: str = typer.Argument(...),
) -> None:
    """Frame-select + ROI-crop a SER (ser-trim.js parity)."""
    from .stack.trim import run

    ok = run(in_ser, out_ser, keep, log)
    raise typer.Exit(0 if ok else 1)


@app.command()
def render(
    out_dir: str = typer.Argument(..., help="Output dir for lunation frames ('dry' = list inputs only)"),
    out_px: int = typer.Option(1080, help="Frame size (0 = measure-only)"),
    canvas: int = typer.Option(2300, help="Working canvas size"),
    root: str = typer.Option(None, help="Production root to scan (out/ tree)"),
    scan_out: str = typer.Option(None, help="Production out/ tree to scan directly"),
    inputs: str = typer.Option(None, help="Comma-separated dirs of dated .xisf finals"),
    stamp: bool = typer.Option(True, help="Render into a fresh <out_dir>/run-<stamp> subdir"),
) -> None:
    """Render phase-ordered, disk-stable lunation frames (gif-frames parity)."""
    import datetime
    import os

    from .assemble.collect import collect
    from .assemble.render import run

    entries = collect(root=root, out_dir=scan_out,
                      input_dirs=inputs.split(",") if inputs else None)
    if out_dir == "dry":
        raise typer.Exit(0)
    if stamp:
        # fresh dir per run: input sets shift frame indices, and stale
        # frame_NN files from a previous run would poison the encode
        out_dir = os.path.join(
            out_dir,
            datetime.datetime.now().strftime("run-%Y%m%d-%H%M"))
    ok = run(out_dir, canvas, out_px, entries, explicit_order=True)
    typer.echo(f"frames: {out_dir}")
    raise typer.Exit(0 if ok else 1)


@app.command()
def prep(
    config: str = typer.Option(None, help="Prep config JSON {targetR,canvas,log,items}"),
    src: str = typer.Option(None, help="Single finished image to normalize"),
    out: str = typer.Option(None, help="Output .xisf for --src"),
) -> None:
    """Normalize finished moon images for the lunation (prep-finished parity)."""
    from .assemble import prep as prep_mod

    if config:
        from .stack.stacker import load_config

        ok = prep_mod.run(load_config(config))
    elif src and out:
        try:
            prep_mod.prep_image(src, out)
            ok = True
        except Exception as e:  # noqa: BLE001 — CLI boundary
            typer.echo(f"PREP FAILED {src}: {e}")
            ok = False
    else:
        typer.echo("need --config, or --src with --out")
        ok = False
    raise typer.Exit(0 if ok else 1)


@app.command()
def finish(
    config: str = typer.Argument(..., help="Finish config JSON (old schema)"),
    stacks_dir: str = typer.Option(None, help="Override stacksDir"),
    out_dir: str = typer.Option(None, help="Override outDir"),
) -> None:
    """Finish per-filter stacks into a color final (lunar-finish parity)."""
    from .finish.chain import run
    from .stack.stacker import load_config

    cfg = load_config(config)
    if stacks_dir:
        cfg["stacksDir"] = stacks_dir
    if out_dir:
        cfg["outDir"] = out_dir
        cfg["log"] = f"{out_dir}/finish.log"
    ok = run(cfg, config)
    raise typer.Exit(0 if ok else 1)


@app.command()
def encode(
    frames_dir: str = typer.Argument(..., help="Dir of frame_*.png renders"),
) -> None:
    """Encode rendered lunation frames to lunation.{mp4,gif}."""
    from .assemble import encode as encode_mod

    raise typer.Exit(0 if encode_mod.run(frames_dir) else 1)


@app.command()
def run(
    root: str = typer.Argument(..., help="Pipeline root (configs/auto + out/)"),
    sessions: str = typer.Option(None, help="Comma-separated session names"),
    workers: int = typer.Option(None, help="Frame workers per stack job"),
    jobs: int = typer.Option(1, help="Concurrent heavy jobs"),
    gif: bool = typer.Option(True, help="Render+encode the lunation after finishes"),
    out_px: int = typer.Option(1080, help="Lunation frame size"),
    canvas: int = typer.Option(2300, help="Lunation working canvas"),
) -> None:
    """Run the full DAG: stack -> finish -> gif (soft deps) -> encode."""
    from .master.pipeline import build_dag
    from .master.scheduler import Scheduler

    dag = build_dag(root, sessions.split(",") if sessions else None,
                    workers, gif, out_px, canvas)
    typer.echo(f"{len(dag)} job(s): "
               + ", ".join(j.id for j in dag))
    ok = Scheduler(dag, heavy_cap=jobs).run()
    raise typer.Exit(0 if ok else 1)


@app.command()
def gui(
    output: str = typer.Option(None, help="Output tree (configs are stored"
                               " with it, under lunation_<id> run dirs)"),
) -> None:
    """Open the Lunation master window (needs the [gui] extra)."""
    from .gui.app import run as gui_run

    raise typer.Exit(gui_run(output))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
