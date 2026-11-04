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
    full: Annotated[
        bool,
        typer.Option("--full", help="Ignore saved cursors and re-read every source item."),
    ] = False,
    apps_dir: AppsDirOpt = DEFAULT_APPS_DIR,
) -> None:
    """Ingest the active app's sources: connector -> sanitize -> parse -> chunk -> embed -> store.

    Connector configs come from the app's ``app.yaml``, overridden by ``hackstack.yaml``. With no
    app folder, ``hackstack.yaml``'s ``connectors:`` alone.

    Runs are incremental: unchanged documents are skipped, only changed chunks are re-embedded,
    deleted source items leave the index, and each connector resumes from its saved cursor.
    """
    from hackstack.config import load_secrets
    from hackstack.ingest import ingestor_from_settings
    from hackstack.sync.manager import SyncManager, SyncReport, SyncStore

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
        reports = []
        for name in names:
            ingestor = ingestor_from_settings(
                settings, name, secrets, cfg=configs.get(name), path=path, dry_run=dry_run
            )
            try:
                if isinstance(ingestor.store, SyncStore):
                    reports.append(SyncManager(ingestor).run(full=full))
                else:
                    reports.append(ingestor.run())
            finally:
                ingestor.close()
    except (KeyError, ValueError) as exc:
        typer.echo(f"error: {exc.args[0] if exc.args else exc}", err=True)
        raise typer.Exit(code=2) from exc

    if as_json:
        out = [r.model_dump(mode="json") for r in reports]
        typer.echo(json.dumps(out[0] if len(out) == 1 else out, indent=2))
        return
    for r in reports:
        if isinstance(r, SyncReport):
            typer.echo(
                f"{r.connector} ({r.mode}): {r.docs} docs updated, {r.unchanged} unchanged, "
                f"{r.deleted} deleted; {r.chunks_embedded} chunks embedded, "
                f"{r.chunks_reused} reused; {len(r.failures)} failures in {r.seconds}s"
            )
        else:
            mode = " (dry run)" if r.dry_run else ""
            typer.echo(
                f"{r.connector}{mode}: {r.docs} docs, {r.chunks} chunks, "
                f"{len(r.failures)} failures in {r.seconds}s"
            )
        if r.enrich is not None:
            e = r.enrich
            usage = ", ".join(f"{k}={v}" for k, v in sorted(e.usage.items())) or "no usage"
            typer.echo(
                f"  contextual enrichment: {e.enriched}/{e.chunks} chunks in {e.calls} calls, "
                f"{e.failed} failed, {e.skipped} over budget, {e.seconds}s; {usage}"
            )
        for f in r.failures:
            typer.echo(f"  ! {f.item_id} [{f.stage}] {f.error_type}: {f.error}", err=True)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    as_user: Annotated[
        str | None,
        typer.Option(
            "--as",
            help="Search as this user from the app's users.yaml (default: its default_user).",
        ),
    ] = None,
    source: Annotated[
        list[str] | None, typer.Option("--source", "-s", help="Only these sources.")
    ] = None,
    since: Annotated[
        str | None, typer.Option(help="Only chunks updated on/after this ISO date.")
    ] = None,
    top_n: Annotated[int | None, typer.Option("--top-n", "-n", help="Hits to keep.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print every stage as JSON.")] = False,
    file: FileOpt = None,
    profile: ProfileOpt = None,
    apps_dir: AppsDirOpt = DEFAULT_APPS_DIR,
) -> None:
    """Hybrid search as a user: dense + lexical -> RRF -> rerank -> neighbour expansion."""
    import psycopg

    from hackstack.retrieval.acl import UnknownUserError, resolve_principals
    from hackstack.retrieval.hybrid import HybridRetriever
    from hackstack.retrieval.rerank import RerankerUnavailable

    try:
        settings = load_settings(file, profile=profile)
        user = as_user or load_app(settings, apps_dir=apps_dir).default_user
        if not user:
            raise ValueError("no --as user and the app has no default_user")
        principals = resolve_principals(settings, user, apps_dir=apps_dir)
        filters = {"source": source or None, "since": since}
        result = HybridRetriever.from_settings(settings).retrieve(
            query, principals, filters, top_n=top_n
        )
    except (UnknownUserError, RerankerUnavailable, ValueError) as exc:
        typer.echo(f"error: {exc.args[0] if exc.args else exc}", err=True)
        raise typer.Exit(code=2) from exc
    except psycopg.OperationalError as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        typer.echo(f"error: cannot reach Postgres ({first}). Is it running? `make up`", err=True)
        raise typer.Exit(code=2) from exc

    if as_json:
        out = {
            "query": query,
            "user": user,
            "principals": principals,
            "counts": {
                stage: len(getattr(result, stage))
                for stage in ("dense", "lexical", "fused", "reranked", "hits")
            },
            "hits": [h.model_dump(mode="json") for h in result.hits],
        }
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(
        f"{user}: dense {len(result.dense)}, lexical {len(result.lexical)}, fused "
        f"{len(result.fused)} -> reranked {len(result.reranked)} -> {len(result.hits)} passages"
    )
    for i, h in enumerate(result.hits, start=1):
        where = " > ".join(h.section_path) or h.title
        extra = f" +{len(h.neighbor_ids)} neighbours" if h.neighbor_ids else ""
        typer.echo(f"{i:2}. [{h.score:.3f}] {h.title} | {where} ({h.chunk_id}){extra}")
        snippet = " ".join(h.text.split())
        typer.echo(f"    {snippet[:200]}{'...' if len(snippet) > 200 else ''}")


if __name__ == "__main__":
    app()
