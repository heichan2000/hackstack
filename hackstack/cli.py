"""The ``hackstack`` command. Later tickets add ``ask``, ``ingest``, ``reset`` and friends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from hackstack.config import config_hash, load_settings

app = typer.Typer(help="Hackstack: a hand-built starter kit for AI agent hackathons.")

FileOpt = Annotated[
    Path | None, typer.Option("--file", "-f", help="Config file (default: ./hackstack.yaml).")
]
ProfileOpt = Annotated[
    str | None, typer.Option("--profile", "-p", help="Profile (default: $HACKSTACK_PROFILE).")
]


@app.callback()
def main() -> None:
    """Hackstack: a hand-built starter kit for AI agent hackathons."""


@app.command()
def config(file: FileOpt = None, profile: ProfileOpt = None) -> None:
    """Print the active profile, its config hash and the resolved settings as JSON."""
    try:
        settings = load_settings(file, profile=profile)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    out = {
        "profile": settings.profile,
        "config_hash": config_hash(settings),
        "settings": settings.model_dump(mode="json"),
    }
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    app()
