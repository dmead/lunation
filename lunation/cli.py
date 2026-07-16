"""Lunation CLI."""

import typer

app = typer.Typer(help="Lunation — standalone lunar lucky-imaging pipeline.")


@app.command()
def version() -> None:
    """Print the package version."""
    from importlib.metadata import version as v

    typer.echo(v("lunation"))


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
    inputs: str = typer.Option(None, help="Comma-separated dirs of dated .xisf finals"),
) -> None:
    """Render phase-ordered, disk-stable lunation frames (gif-frames parity)."""
    from .assemble.collect import collect
    from .assemble.render import run

    entries = collect(root=root,
                      input_dirs=inputs.split(",") if inputs else None)
    if out_dir == "dry":
        raise typer.Exit(0)
    ok = run(out_dir, canvas, out_px, entries, explicit_order=True)
    raise typer.Exit(0 if ok else 1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
