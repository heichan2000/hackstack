"""The ``hackstack`` command. Later tickets add ``ask`` and friends."""

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


@app.command()
def reset(
    file: FileOpt = None,
    profile: ProfileOpt = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Drop and recreate the index for the configured embedder (then re-ingest)."""
    import psycopg

    from hackstack.store.postgres import PostgresStore

    try:
        settings = load_settings(file, profile=profile)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    target = f"{settings.embed.model} ({settings.embed.dim} dim)"
    if not yes:
        typer.confirm(
            f"Drop every indexed document and chunk and rebuild the index for {target}?",
            abort=True,
        )
    try:
        with PostgresStore.from_settings(settings) as store:
            store.reset()
    except psycopg.OperationalError as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        typer.echo(f"error: cannot reach Postgres ({first}). Is it running? `make up`", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"index reset for {target}; re-ingest to fill it")


@app.command()
def ingest(
    file: FileOpt = None,
    profile: ProfileOpt = None,
    connector: Annotated[
        list[str] | None,
        typer.Option("--connector", "-c", help="Connector(s) to run (default: all configured)."),
    ] = None,
    path: Annotated[
        str | None, typer.Option(help="Override the connector's path (local_files).")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Parse and chunk only: no embedding, no store.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the reports as JSON.")] = False,
    apps_dir: AppsDirOpt = DEFAULT_APPS_DIR,
) -> None:
    """Ingest the active app's sources: connector -> sanitize -> parse -> chunk -> embed -> store.

    Connector configs come from the app's ``app.yaml``, overridden by ``hackstack.yaml``. With no
    app folder, ``hackstack.yaml``'s ``connectors:`` alone.
    """
    from hackstack.config import load_secrets
    from hackstack.ingest import ingestor_from_settings

    try:
        settings = load_settings(file, profile=profile)
        if (Path(apps_dir) / settings.app / "app.yaml").is_file():
            configs = load_app(settings, apps_dir=apps_dir).connectors
        else:
            configs = dict(settings.connectors)
        names = connector or list(configs) or ["local_files"]
        secrets = {
            k: (v.get_secret_value() if hasattr(v, "get_secret_value") else v)
            for k, v in load_secrets().model_dump().items()
            if v is not None
        }
        reports = [
            ingestor_from_settings(
                settings, name, secrets, cfg=configs.get(name), path=path, dry_run=dry_run
            ).run()
            for name in names
        ]
    except (KeyError, ValueError) as exc:
        typer.echo(f"error: {exc.args[0] if exc.args else exc}", err=True)
        raise typer.Exit(code=2) from exc

    if as_json:
        out = [r.model_dump(mode="json") for r in reports]
        typer.echo(json.dumps(out[0] if len(out) == 1 else out, indent=2))
        return
    for r in reports:
        mode = " (dry run)" if r.dry_run else ""
        typer.echo(
            f"{r.connector}{mode}: {r.docs} docs, {r.chunks} chunks, "
            f"{len(r.failures)} failures in {r.seconds}s"
        )
        for f in r.failures:
            typer.echo(f"  ! {f.item_id} [{f.stage}] {f.error_type}: {f.error}", err=True)


if __name__ == "__main__":
    app()
