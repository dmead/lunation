"""Lunation CLI. M0 ships the skeleton; stack/trim commands land in M1."""

import typer

app = typer.Typer(help="Lunation — standalone lunar lucky-imaging pipeline.")


@app.command()
def version() -> None:
    """Print the package version."""
    from importlib.metadata import version as v

    typer.echo(v("lunation"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
