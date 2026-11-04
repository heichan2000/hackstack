"""The committed golden set still fits the corpus, and recall@5 / MRR hold the baseline.

- Offline (always): ``golden.jsonl`` labels exist in the corpus as chunked today, every asker can
  see their labels and cannot see their forbidden docs, and the fake-embedder index scores at
  or above ``baseline.json``'s ``offline`` numbers. Deterministic, no services.
- Live (``pg`` + ``ollama``): the ``local`` profile against the ingested dev index
  (``make ingest-sample``), dense and hybrid variants, against the ``local`` baseline. The
  reranked variant is checked by ``make eval`` (it costs about a second a question on CPU).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.golden import GOLDEN_PATH, corpus_chunks, golden_sha256, load_golden
from evals.run_retrieval import (
    BASELINE_PATH,
    check_baseline,
    offline_index,
    offline_settings,
    run,
)
from hackstack.apps import load_app
from hackstack.config import load_settings
from hackstack.parsing import router as router_mod
from hackstack.retrieval.acl import principals_for

REPO = Path(__file__).resolve().parents[1]
APPS = REPO / "apps"


@pytest.fixture(scope="module")
def no_docling():
    saved = router_mod.docling_available
    router_mod.docling_available = lambda: False
    yield
    router_mod.docling_available = saved


@pytest.fixture(scope="module")
def settings():
    return load_settings(REPO / "hackstack.yaml", env_file=None)


@pytest.fixture(scope="module")
def items():
    return load_golden(GOLDEN_PATH)


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def principals_of(settings):
    app = load_app(settings, apps_dir=APPS)
    return lambda user: principals_for(app.principal(user))


def test_golden_labels_exist_in_the_corpus_as_chunked_today(settings, items, no_docling):
    chunk_ids = {c.chunk_id for _, c in corpus_chunks(settings, apps_dir=APPS)}
    stale = {cid for it in items for cid in it.relevant if cid not in chunk_ids}
    assert not stale, (
        f"{len(stale)} golden labels no longer exist (did the chunker change?); "
        "rerun `make golden` and `--write-baseline`"
    )


def test_golden_set_is_well_formed(settings, items, principals_of, no_docling):
    chunks = {c.chunk_id: c for _, c in corpus_chunks(settings, apps_dir=APPS)}
    docs_acl = {c.doc_id: set(c.acl_principals) for c in chunks.values()}
    assert len({it.id for it in items}) == len(items)
    kinds = {it.kind for it in items}
    assert {"synthetic", "handwritten", "unanswerable", "forbidden"} <= kinds
    for it in items:
        who = set(principals_of(it.as_user))  # raises for an unknown persona
        for cid in it.relevant:
            assert who & set(chunks[cid].acl_principals), f"{it.id}: {it.as_user} can't see {cid}"
            assert chunks[cid].doc_id in it.relevant_docs
        for doc in it.forbidden_docs:
            assert not who & docs_acl[doc], f"{it.id}: {it.as_user} can see {doc}"
        if it.kind in ("unanswerable", "forbidden"):
            assert not it.relevant


def test_baseline_matches_the_committed_golden_set(baseline):
    assert baseline["golden_sha256"] == golden_sha256(GOLDEN_PATH), (
        "golden.jsonl changed since baseline.json was written: review, then "
        "`uv run python -m evals.run_retrieval --offline --write-baseline` (and the local one)"
    )


def test_offline_retrieval_holds_the_baseline(settings, items, baseline, principals_of, no_docling):
    offline = offline_settings(settings)
    store, embedder = offline_index(offline, apps_dir=APPS)
    report = run(offline, items, variants=("dense", "lexical", "hybrid"), store=store,
                 embedder=embedder, principals_of=principals_of,
                 golden_sha=golden_sha256(GOLDEN_PATH))  # fmt: skip
    assert check_baseline(report, baseline, "offline") == []
    for v in report["variants"].values():
        assert v["forbidden"]["leaks"] == 0


@pytest.mark.pg
@pytest.mark.ollama
def test_local_retrieval_holds_the_baseline(items, baseline):
    psycopg = pytest.importorskip("psycopg")
    import httpx2

    from hackstack.embed import build_embedder
    from hackstack.retrieval.acl import resolve_principals
    from hackstack.store.postgres import IndexMismatchError, PostgresStore

    local = load_settings(REPO / "hackstack.yaml", profile="local", env_file=None)
    if "local" not in baseline["profiles"]:
        pytest.skip("no local baseline yet")
    try:
        httpx2.get(local.embed.base_url + "/api/tags", timeout=2).raise_for_status()
    except httpx2.HTTPError as exc:
        pytest.skip(f"Ollama not reachable: {exc}")
    try:
        store = PostgresStore.from_settings(local)
        n = store.count_chunks()
    except (psycopg.OperationalError, IndexMismatchError) as exc:
        pytest.skip(f"no usable local index: {str(exc).splitlines()[0]}")
    if n == 0:
        store.close()
        pytest.skip("local index is empty: run `HACKSTACK_PROFILE=local make ingest-sample`")
    try:
        report = run(local, items, variants=("dense", "hybrid"), store=store,
                     embedder=build_embedder(local),
                     principals_of=lambda u: resolve_principals(local, u, apps_dir=APPS),
                     golden_sha=golden_sha256(GOLDEN_PATH))  # fmt: skip
    finally:
        store.close()
    assert check_baseline(report, baseline, "local") == []
    for v in report["variants"].values():
        assert v["forbidden"]["leaks"] == 0
