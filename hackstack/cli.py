"""The ``hackstack`` command. Later tickets add ``ask``, ``ingest``, ``reset`` and friends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from hackstack.apps import DEFAULT_APPS_DIR, load_app
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


AppsDirOpt = Annotated[
    Path, typer.Option("--apps-dir", help="Folder holding the apps (default: ./apps).")
]


@app.command(name="app")
def show_app(
    file: FileOpt = None, profile: ProfileOpt = None, apps_dir: AppsDirOpt = DEFAULT_APPS_DIR
) -> None:
    """Print the active app (``app:`` in hackstack.yaml): title, tools, users and connectors."""
    try:
        loaded = load_app(load_settings(file, profile=profile), apps_dir=apps_dir)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    out = loaded.model_dump(mode="json", exclude={"system_prompt"})
    out["system_prompt_chars"] = len(loaded.system_prompt)
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    app()
