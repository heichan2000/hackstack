"""``make eval``: the eval suite, stage by stage.

    HACKSTACK_PROFILE=local make eval          # retrieval comparison, gated on baseline.json
    HACKSTACK_PROFILE=local make eval FULL=1   # + contextual enrichment on/off + generation

Stages (``--stages`` picks any of them; ``--full`` means all three):

1. **retrieval** — dense / lexical / hybrid / hybrid+rerank on the golden set against the
   ingested index (``evals/run_retrieval.py``); exits 1 below ``baseline.json``. Deterministic,
   seconds to minutes: this is the per-change gate.
2. **enrich** — the same comparison on a *second* index built with contextual enrichment on, in
   its own database (``<db>_eval_enriched`` next to the dev one, never the dev index itself).
   Building it costs one aux-model call per chunk (about 10 minutes for the Northwind corpus on
   an RTX 3070 Laptop); later runs only re-enrich changed documents.
3. **generation** — agent vs single-shot answers with the PASS/FAIL judges, refusal checks and
   the red-team items (``evals/run_generation.py``). Expensive: one agent turn plus up to two
   judge calls per question per mode, so ``--per-kind N`` runs a subset.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

from evals import run_generation, run_retrieval
from evals.golden import GOLDEN_PATH, dump_report, golden_sha256, load_golden

if TYPE_CHECKING:
    from hackstack.config import Settings

STAGES = ("retrieval", "enrich", "generation")
ENRICHED_SUFFIX = "_eval_enriched"


def enriched_settings(settings: Settings) -> Settings:
    """``settings`` with contextual enrichment on, pointed at its own throwaway database."""
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    params = conninfo_to_dict(settings.store.dsn)
    params["dbname"] = f"{params.get('dbname', 'hackstack')}{ENRICHED_SUFFIX}"
    out = settings.model_copy(
        update={
            "store": settings.store.model_copy(update={"dsn": make_conninfo(**params)}),
            "enrich": settings.enrich.model_copy(update={"contextual": True}),
        }
    )
    out._profile = settings.profile
    return out


def ensure_database(dsn: str) -> bool:
    """Create ``dsn``'s database if it does not exist (on the same server); True if created."""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    params = conninfo_to_dict(dsn)
    name = params["dbname"]
    admin = make_conninfo(**{**params, "dbname": "postgres"})
    with psycopg.connect(admin, autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            return False
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return True


def stage_enrich(settings: Settings, items: list, sha: str, principals) -> dict[str, Any]:
    """Build (or sync) the enriched index, then run the retrieval comparison on it."""
    from hackstack.embed import build_embedder
    from hackstack.ingest import ingest_app
    from hackstack.store.postgres import PostgresStore

    es = enriched_settings(settings)
    created = ensure_database(es.store.dsn)
    print(f"enriched index: {'creating' if created else 'syncing'} {es.store.dsn.split('@')[-1]}"
          " (one aux-model call per new or changed chunk)")  # fmt: skip
    for report in ingest_app(es):
        print(f"  {report}")
    store, embedder = PostgresStore.from_settings(es), build_embedder(es)
    try:
        variants = [v for v in run_retrieval.VARIANTS
                    if v != "hybrid_rerank" or es.rerank.provider != "noop"]  # fmt: skip
        report = run_retrieval.run(es, items, variants=variants, store=store, embedder=embedder,
                                   principals_of=principals, golden_sha=sha)  # fmt: skip
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    report["enriched"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    from datetime import UTC, datetime

    from hackstack.config import load_settings
    from hackstack.retrieval.acl import resolve_principals

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", "-p", help="config profile (default: $HACKSTACK_PROFILE)")
    ap.add_argument("--full", action="store_true", help="all stages")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=None)
    ap.add_argument("--per-kind", type=int, default=None, help="generation: N questions per kind")
    ap.add_argument("--modes", nargs="+", default=list(run_generation.MODES),
                    choices=run_generation.MODES)  # fmt: skip
    args = ap.parse_args(argv)
    stages = list(STAGES) if args.full else (args.stages or ["retrieval"])
    profile = ["--profile", args.profile] if args.profile else []
    status = 0

    if "retrieval" in stages:
        print("== retrieval: plain index ==")
        status = run_retrieval.main([*profile, "--check"])

    if "enrich" in stages:
        print("\n== retrieval: contextual enrichment on (separate index) ==")
        settings = load_settings(profile=args.profile)
        items = load_golden(GOLDEN_PATH)
        cache: dict[str, list[str]] = {}

        def principals(user: str) -> list[str]:
            if user not in cache:
                cache[user] = resolve_principals(settings, user)
            return cache[user]

        report = stage_enrich(settings, items, golden_sha256(GOLDEN_PATH), principals)
        print(run_retrieval.format_table(report))
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}-{settings.profile}-enriched-{report['golden_sha256']}.json"
        out = run_retrieval.RUNS_DIR / name
        dump_report(out, report)
        print(f"\nsaved {out}")

    if "generation" in stages:
        print("\n== generation: agent vs single-shot, judges, refusals, red team ==")
        gen = [*profile, "--modes", *args.modes]
        if args.per_kind is not None:
            gen += ["--per-kind", str(args.per_kind)]
        run_generation.main(gen)
    return status


if __name__ == "__main__":
    sys.exit(main())
