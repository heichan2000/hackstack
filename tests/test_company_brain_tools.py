"""company_brain's own tools, offline: find_experts, whats_changed, and the dated-conflict prompt.

The Northwind corpus is ingested for real (connector, parser, chunker) into the in-memory store
with the fake embedder, so owners, dates and ACLs are the corpus's own. Ranking is the one thing
the fakes cannot do, so find_experts' aggregation is tested with a scripted retriever whose hit
order is fixed, over that real store.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from hackstack.agent.citations import CitationRegistry
from hackstack.agent.loop import _to_result_content
from hackstack.agent.tools import Passages, ToolContext, ToolInputError, load_tools
from hackstack.apps import load_app
from hackstack.config import load_settings
from hackstack.models import SearchHit
from hackstack.retrieval.acl import principals_for
from hackstack.retrieval.hybrid import HybridRetriever
from hackstack.retrieval.rerank import NoopReranker
from tests.fakes import FakeEmbedder
from tests.test_ingest_sample import ingest_corpus

REPO = Path(__file__).resolve().parents[1]
APP_ROOT = REPO / "apps" / "company_brain"
TODAY = date(2026, 11, 4)  # Wednesday: the demo day the corpus is written for

NW = "local_files:northwind/"
RUNBOOK = NW + "engineering/runbooks/postgres-migrations.md"
POSTMORTEM = NW + "engineering/postmortems/2026-11-02-checkout-latency.md"
CODE_REVIEW = NW + "handbook/code-review-and-deploys.md"
EDR = NW + "engineering/decisions/2026-10-30-review-and-friday-deploys.md"
ONCALL = NW + "engineering/oncall-rotation.md"
RELEASE = NW + "product/release-notes-pulse-3-4.md"
PRICING = NW + "finance/2027-pricing-change.md"
PERF = NW + "hr/h2-2026-performance-cycle.md"
COMP = NW + "hr/compensation-bands-2026.md"

SAM = "sam.okafor@northwind.test"
TOMAS = "tomas.silva@northwind.test"
PRIYA = "priya.raman@northwind.test"


@pytest.fixture(scope="module")
def store():
    _, store = ingest_corpus()
    return store


@pytest.fixture(scope="module")
def tools():
    loaded = load_tools(["find_experts", "whats_changed"], app_root=APP_ROOT)
    assert [t.name for t in loaded] == ["find_experts", "whats_changed"]
    return {t.name: t for t in loaded}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return load_app(load_settings(REPO / "hackstack.yaml"), apps_dir=REPO / "apps")


def principals(app, user):
    return principals_for(app.principal(user))


class ScriptedRetriever:
    """A retriever whose hits are fixed: one per listed chunk id, in this order, ACL-filtered."""

    def __init__(self, store, chunk_ids):
        self.store = store
        self.chunk_ids = chunk_ids
        self.calls: list[tuple[str, dict | None, int | None]] = []

    def search(self, query, principals, filters=None, *, top_n=None):
        self.calls.append((query, filters, top_n))
        allowed = set(principals)
        out = []
        for rank, cid in enumerate(self.chunk_ids):
            c = self.store.chunks[cid]
            if allowed & set(c.acl_principals):
                doc = self.store.docs[c.doc_id]
                out.append(
                    SearchHit(
                        chunk_id=cid,
                        doc_id=c.doc_id,
                        text=c.text,
                        score=1.0 / (rank + 1),
                        title=doc.title,
                        source=c.source,
                        section_path=c.section_path,
                        updated_at=c.updated_at,
                        ordinal=c.ordinal,
                    )
                )
        return out


def ctx_for(app, user, retriever, **extra):
    return ToolContext(
        principals=principals(app, user),
        retriever=retriever,
        user=user,
        app=app,
        extra={"today": TODAY, **extra},
    )


def first_chunk(store, doc_id):
    return min((c for c in store.chunks.values() if c.doc_id == doc_id), key=lambda c: c.ordinal)


def chunk_ids(store, *doc_ids):
    return [first_chunk(store, d).chunk_id for d in doc_ids]


# --- registration ------------------------------------------------------------------------------


def test_the_app_lists_and_registers_all_its_tools(app, tools):
    assert app.tools == ["search_kb", "get_document", "find_experts", "whats_changed"]
    names = [t.name for t in load_tools(app.tools, app_root=app.root)]
    assert names == app.tools  # nothing "listed but not registered"
    for t in tools.values():
        assert t.description.count(". ") >= 3
        assert t.definition()["input_schema"]["required"]


# --- whats_changed -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        ("eng@northwind.test", {POSTMORTEM, RELEASE, ONCALL}),
        ("hr@northwind.test", {RELEASE, PERF}),
        ("finance@northwind.test", {RELEASE, PRICING}),
        ("ceo@northwind.test", {POSTMORTEM, RELEASE, ONCALL, PRICING, PERF}),
    ],
)
def test_whats_changed_since_monday_differs_by_role(app, store, tools, user, expected):
    retriever = ScriptedRetriever(store, [])
    out = tools["whats_changed"].handler({"since": "2026-11-02"}, ctx_for(app, user, retriever))
    assert isinstance(out, Passages)
    assert {h.doc_id for h in out.hits} == expected
    dates = [h.updated_at for h in out.hits]
    assert dates == sorted(dates, reverse=True)  # newest first
    assert "2026-11-02" in out.preamble
    for h in out.hits:
        assert h.title in out.preamble


def test_whats_changed_understands_weekday_names_and_relative_days(app, store, tools):
    retriever = ScriptedRetriever(store, [])
    ctx = ctx_for(app, "ceo@northwind.test", retriever)
    monday = tools["whats_changed"].handler({"since": "Monday"}, ctx)
    assert "2026-11-02" in monday.preamble and len(monday.hits) == 5
    yesterday = tools["whats_changed"].handler({"since": "yesterday"}, ctx)
    assert {h.doc_id for h in yesterday.hits} == {ONCALL, PRICING, PERF}
    week = tools["whats_changed"].handler({"since": "last week"}, ctx)
    assert "2026-10-28" in week.preamble


@pytest.mark.parametrize("bad", ["", "soon", "2026-13-40", "2026-12-01"])
def test_whats_changed_rejects_dates_it_cannot_use(app, store, tools, bad):
    ctx = ctx_for(app, "eng@northwind.test", ScriptedRetriever(store, []))
    with pytest.raises(ToolInputError):
        tools["whats_changed"].handler({"since": bad}, ctx)


def test_whats_changed_on_a_topic_searches_with_the_date_filter(app, store, tools):
    retriever = ScriptedRetriever(store, chunk_ids(store, POSTMORTEM, ONCALL))
    ctx = ctx_for(app, "eng@northwind.test", retriever)
    out = tools["whats_changed"].handler({"since": "2026-11-02", "topic": "incidents"}, ctx)
    query, filters, _ = retriever.calls[0]
    assert query == "incidents" and filters["since"] == "2026-11-02"
    assert [h.doc_id for h in out.hits] == [ONCALL, POSTMORTEM]  # 3 Nov before 2 Nov


def test_whats_changed_with_nothing_new_says_so(app, store, tools):
    ctx = ctx_for(app, "eng@northwind.test", ScriptedRetriever(store, []))
    out = tools["whats_changed"].handler({"since": "2026-11-04"}, ctx)
    assert out.hits == [] and "2026-11-04" in out.empty


# --- find_experts ------------------------------------------------------------------------------


def test_find_experts_ranks_owners_by_relevance_and_recency(app, store, tools):
    runbook = [
        c.chunk_id
        for c in sorted(store.chunks.values(), key=lambda c: c.ordinal)
        if c.doc_id == RUNBOOK
    ]
    order = [runbook[0], *chunk_ids(store, CODE_REVIEW, EDR), runbook[1]]
    order += chunk_ids(store, POSTMORTEM)
    ctx = ctx_for(app, "eng@northwind.test", ScriptedRetriever(store, order))
    out = tools["find_experts"].handler({"topic": "database migrations"}, ctx)
    assert isinstance(out, Passages)
    lines = [ln for ln in out.preamble.splitlines() if ln[:2] in {"1.", "2.", "3."}]
    assert [ln.split(" (")[0][3:] for ln in lines] == ["Sam Okafor", "Priya Raman", "Tomás Silva"]
    assert "Staff Engineer, Data" in lines[0] and SAM in lines[0]
    assert "Runbook: Postgres schema migrations (updated 2026-10-21)" in lines[0]
    assert "Postmortem: checkout latency spike" in lines[0]
    assert {h.doc_id for h in out.hits} == {RUNBOOK, CODE_REVIEW, EDR, POSTMORTEM}
    assert len(out.hits) == 4  # one evidence passage per document, the best-ranked one


def test_recency_can_outweigh_a_small_rank_lead(app, store, tools):
    # The March code-review page ranks first, the 3 November on-call page second: with a
    # 180-day half-life the newer page's owner comes out ahead.
    order = chunk_ids(store, CODE_REVIEW, ONCALL)
    out = tools["find_experts"].handler(
        {"topic": "x"}, ctx_for(app, "eng@northwind.test", ScriptedRetriever(store, order))
    )
    ranked = [ln.split(" (")[0][3:] for ln in out.preamble.splitlines() if ln[:2] in {"1.", "2."}]
    assert ranked == ["Tomás Silva", "Priya Raman"]


def test_find_experts_never_names_owners_of_documents_the_person_cannot_read(app, store, tools):
    # The retriever "leaks" engineering-only and HR-only chunks; owners are looked up under the
    # asker's principals too, so nobody from those documents is named.
    leaky = chunk_ids(store, RUNBOOK, POSTMORTEM, COMP, CODE_REVIEW)

    class LeakyRetriever(ScriptedRetriever):
        def search(self, query, principals, filters=None, *, top_n=None):
            return super().search(query, ["group:everyone", "group:engineering", "group:hr"])

    out = tools["find_experts"].handler(
        {"topic": "database migrations"},
        ctx_for(app, "finance@northwind.test", LeakyRetriever(store, leaky)),
    )
    assert SAM not in out.preamble and TOMAS not in out.preamble
    assert "maria.santos" not in out.preamble
    assert PRIYA in out.preamble
    assert {h.doc_id for h in out.hits} == {CODE_REVIEW}


def test_find_experts_with_no_owned_documents_names_no_one(app, store, tools):
    handbook_only = [c.chunk_id for c in store.chunks.values() if "northwind/" not in c.doc_id][:3]
    out = tools["find_experts"].handler(
        {"topic": "values"},
        ctx_for(app, "eng@northwind.test", ScriptedRetriever(store, handbook_only)),
    )
    assert out.hits == [] and out.preamble == "" and "values" in out.empty


def test_find_experts_needs_a_topic(app, store, tools):
    with pytest.raises(ToolInputError):
        tools["find_experts"].handler(
            {"topic": " "}, ctx_for(app, "eng@northwind.test", ScriptedRetriever(store, []))
        )


def test_find_experts_over_the_real_retriever_is_acl_safe(app, store, tools):
    retriever = HybridRetriever(store, FakeEmbedder(), NoopReranker(), window=20, top_n=8)
    out = tools["find_experts"].handler(
        {"topic": "database migrations runbook"}, ctx_for(app, "finance@northwind.test", retriever)
    )
    assert SAM not in out.preamble and TOMAS not in out.preamble
    finance = set(principals(app, "finance@northwind.test"))
    for h in out.hits:
        assert finance & set(store.docs[h.doc_id].acl.principals)


# --- the preamble reaches the model ------------------------------------------------------------


def test_a_preamble_is_sent_before_the_numbered_passages_and_survives_dedup(store):
    hit = SearchHit(chunk_id="d#0", doc_id="d", text="Body.", score=1.0, title="Doc")
    reg = CitationRegistry()
    content, n, _ = _to_result_content(Passages(hits=[hit], preamble="People: Sam."), reg)
    assert n == 1
    assert content[0] == {"type": "text", "text": "People: Sam."}
    assert content[1]["type"] == "search_result"
    again, n2, _ = _to_result_content(Passages(hits=[hit], preamble="People: Sam."), reg)
    assert n2 == 0 and again[0]["text"] == "People: Sam."


# --- the prompt asks for dated conflicts -------------------------------------------------------


def test_the_prompt_asks_for_both_dated_sources_and_the_newer_one(app):
    prompt = app.render_system_prompt("eng@northwind.test", today=TODAY)
    assert "2026-11-04" in prompt and "Wednesday" in prompt
    assert "find_experts" in prompt and "whats_changed" in prompt
    assert "newer" in prompt
