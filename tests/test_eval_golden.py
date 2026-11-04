"""The golden set: corpus chunks offline, hand-written labels, sampling and LLM generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.golden import (
    GoldenItem,
    corpus_chunks,
    golden_sha256,
    load_golden,
    resolve_handwritten,
    save_golden,
)
from evals.make_golden import (
    asker_for,
    generate,
    lookalike_pairs,
    parse_json_object,
    sample_chunks,
)
from hackstack.apps import load_app
from hackstack.config import load_settings
from hackstack.models import TextDelta
from hackstack.parsing import router as router_mod
from hackstack.retrieval.acl import principals_for

REPO = Path(__file__).resolve().parents[1]
APPS = REPO / "apps"
COMP = "local_files:northwind/hr/compensation-bands-2026.md"
PTO = "local_files:northwind/handbook/pto-policy.md"


@pytest.fixture(autouse=True)
def _no_docling(monkeypatch):
    monkeypatch.setattr(router_mod, "docling_available", lambda: False)


@pytest.fixture(scope="module")
def settings():
    return load_settings(REPO / "hackstack.yaml", env_file=None)


@pytest.fixture(scope="module")
def app(settings):
    return load_app(settings, apps_dir=APPS)


@pytest.fixture(scope="module")
def pairs(settings):
    import hackstack.parsing.router as r

    saved = r.docling_available
    r.docling_available = lambda: False
    try:
        return corpus_chunks(settings, apps_dir=APPS)
    finally:
        r.docling_available = saved


# --- corpus -----------------------------------------------------------------------------------


def test_corpus_chunks_are_the_ingested_chunks_offline(pairs):
    docs = {raw.doc_id for raw, _ in pairs}
    assert len(docs) == 30
    assert len(pairs) == 265  # what `make ingest-sample` stores
    raw, chunk = pairs[0]
    assert chunk.doc_id == raw.doc_id
    assert "#md1-" in chunk.chunk_id  # chunker version is part of every id


# --- hand-written items -----------------------------------------------------------------------


def _hw(tmp_path, items) -> Path:
    import yaml

    path = tmp_path / "hw.yaml"
    path.write_text(yaml.safe_dump(items), encoding="utf-8")
    return path


def test_handwritten_contains_resolves_to_the_matching_chunk(tmp_path, pairs):
    path = _hw(
        tmp_path,
        [
            {
                "id": "q1",
                "question": "carryover?",
                "as_user": "finance@northwind.test",
                "kind": "handwritten",
                "relevant": [{"doc": PTO, "contains": "UP TO 5 UNUSED DAYS"}],
            }
        ],
    )
    (item,) = resolve_handwritten(path, pairs)
    assert len(item.relevant) == 1
    (cid,) = item.relevant
    chunk = next(c for _, c in pairs if c.chunk_id == cid)
    assert "up to 5 unused days" in chunk.text.lower()
    assert item.relevant_docs == [PTO]


def test_handwritten_without_contains_takes_every_chunk_of_the_doc(tmp_path, pairs):
    path = _hw(
        tmp_path,
        [{"id": "q", "question": "?", "as_user": "x", "kind": "handwritten",
          "relevant": [{"doc": PTO}]}],
    )  # fmt: skip
    (item,) = resolve_handwritten(path, pairs)
    assert sorted(item.relevant) == sorted(c.chunk_id for _, c in pairs if c.doc_id == PTO)


@pytest.mark.parametrize(
    "ref", [{"doc": PTO, "contains": "no such text anywhere"}, {"doc": "local_files:nope.md"}]
)
def test_a_label_that_matches_nothing_fails_loudly(tmp_path, pairs, ref):
    path = _hw(
        tmp_path,
        [{"id": "q", "question": "?", "as_user": "x", "kind": "handwritten", "relevant": [ref]}],
    )
    with pytest.raises(ValueError, match="q"):
        resolve_handwritten(path, pairs)


def test_unanswerable_and_forbidden_items_have_no_relevant_chunks(pairs):
    items = resolve_handwritten(REPO / "evals" / "handwritten.yaml", pairs)
    by_kind = {}
    for it in items:
        by_kind.setdefault(it.kind, []).append(it)
    assert by_kind["unanswerable"] and by_kind["forbidden"] and by_kind["handwritten"]
    for it in by_kind["unanswerable"] + by_kind["forbidden"]:
        assert it.relevant == [] and not it.answerable
    for it in by_kind["forbidden"]:
        assert it.forbidden_docs


# --- jsonl ------------------------------------------------------------------------------------


def test_golden_round_trips_through_jsonl_and_hashes_stably(tmp_path):
    items = [
        GoldenItem(id="a", question="q?", as_user="u", kind="synthetic", relevant=["d#1"],
                   relevant_docs=["d"], source_chunk="d#1", realism=5),
        GoldenItem(id="b", question="none?", as_user="u", kind="unanswerable"),
    ]  # fmt: skip
    path = tmp_path / "golden.jsonl"
    save_golden(path, items)
    assert load_golden(path) == items
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    h = golden_sha256(path)
    save_golden(path, items)
    assert golden_sha256(path) == h
    save_golden(path, items[:1])
    assert golden_sha256(path) != h


# --- sampling ---------------------------------------------------------------------------------


def test_sampling_is_deterministic_distinct_and_spread_across_docs(pairs):
    a = sample_chunks(pairs, 20, seed=7)
    assert [c.chunk_id for _, c in a] == [c.chunk_id for _, c in sample_chunks(pairs, 20, seed=7)]
    assert [c.chunk_id for _, c in a] != [c.chunk_id for _, c in sample_chunks(pairs, 20, seed=8)]
    ids = [c.chunk_id for _, c in a]
    assert len(set(ids)) == 20
    assert len({c.doc_id for _, c in a}) == 20  # 30 docs: one chunk per doc before any repeats


def test_sampling_skips_short_chunks(pairs):
    for _, c in sample_chunks(pairs, 60, seed=1, min_chars=300):
        assert len(c.text) >= 300


def test_the_asker_can_always_see_the_chunk(pairs, app):
    seen_users = set()
    for _, c in pairs:
        user = asker_for(c, app)
        assert user is not None
        assert set(principals_for(app.principal(user))) & set(c.acl_principals)
        seen_users.add(user)
        assert asker_for(c, app) == user  # deterministic
    assert len(seen_users) > 1  # spread over the personas, not always the first one


def test_restricted_chunks_are_asked_by_someone_allowed(pairs, app):
    for _, c in pairs:
        if c.doc_id == COMP:
            assert asker_for(c, app) in {"hr@northwind.test", "ceo@northwind.test"}


def test_lookalike_pairs_are_the_most_similar_chunks_of_different_docs(pairs):
    found = lookalike_pairs(pairs, 5)
    assert len(found) == 5
    sims = [s for _, _, s in found]
    assert sims == sorted(sims, reverse=True)
    a_docs = [a.doc_id for (_, a), _, _ in found]
    assert len(set(a_docs)) == 5
    for (_, a), (_, b), _ in found:
        assert a.doc_id != b.doc_id


# --- generation -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"question": "Why?"}', {"question": "Why?"}),
        ('Sure! ```json\n{"realism": 4, "answered": true}\n``` done', {"realism": 4,
                                                                       "answered": True}),
        ("no json here", None),
        ('{"broken": ', None),
    ],
)  # fmt: skip
def test_parse_json_object_is_forgiving_about_wrapping(text, expected):
    assert parse_json_object(text) == expected


class ResponderLLM:
    """Answers each prompt with ``respond(prompt)``; records every prompt."""

    def __init__(self, respond):
        self.respond = respond
        self.prompts: list[str] = []

    async def stream(self, system, messages, tools, **opts):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        yield TextDelta(text=self.respond(prompt))


class Responder:
    """Distinct questions per prompt; a lookalike's second rating (passage B) says "no"."""

    def __init__(self):
        self.n = 0
        self.ratings: dict[str, int] = {}

    def __call__(self, prompt: str) -> str:
        if prompt.startswith("Rate this question"):
            question = prompt.split("Question: ", 1)[1].split("\n", 1)[0]
            self.ratings[question] = self.ratings.get(question, 0) + 1
            if "UNREALISTIC" in question:  # one the realism filter should drop
                return json.dumps({"realism": 2, "answered": True})
            if question.startswith("Lookalike") and self.ratings[question] == 2:
                return json.dumps({"realism": 5, "answered": False})  # B does not answer it
            return json.dumps({"realism": 5, "answered": True})
        self.n += 1
        if prompt.startswith("Passage A and passage B"):
            return json.dumps({"question": f"Lookalike question {self.n}?"})
        if "Paid time off" in prompt:
            return json.dumps({"question": f"UNREALISTIC question about leave {self.n}"})
        return json.dumps({"question": f"What does section {self.n} say?"})


def test_generate_labels_each_question_with_its_source_chunk(settings, app, pairs):
    llm = ResponderLLM(Responder())
    items, stats = generate(
        pairs, app, llm, n_synthetic=30, n_lookalike=3, seed=7,
        handwritten=REPO / "evals" / "handwritten.yaml",
    )  # fmt: skip
    synthetic = [i for i in items if i.kind == "synthetic"]
    lookalikes = [i for i in items if i.kind == "lookalike"]
    assert synthetic and len(lookalikes) == 3
    by_id = {c.chunk_id: c for _, c in pairs}
    for it in synthetic + lookalikes:
        assert it.relevant == [it.source_chunk]
        assert it.relevant_docs == [by_id[it.source_chunk].doc_id]
        assert it.realism >= 4
        assert set(principals_for(app.principal(it.as_user))) & set(
            by_id[it.source_chunk].acl_principals
        )
    for it in lookalikes:
        assert it.confusable_chunk and it.confusable_chunk != it.source_chunk
    # the PTO-policy questions were rated 2 and dropped
    assert stats["dropped_realism"] >= 1
    assert all(PTO not in (i.relevant_docs or []) for i in synthetic)
    assert stats["kept"] == len(synthetic) + len(lookalikes)
    # hand-written items come along, ids are unique
    assert any(i.kind == "handwritten" for i in items)
    assert len({i.id for i in items}) == len(items)


def test_generate_drops_unparseable_and_unanswered(settings, app, pairs):
    def respond(prompt):
        if prompt.startswith("Rate this question"):
            return json.dumps({"realism": 5, "answered": False})
        return "I cannot help with that"

    items, stats = generate(pairs, app, ResponderLLM(respond), n_synthetic=5, n_lookalike=0)
    assert [i for i in items if i.kind == "synthetic"] == []
    assert stats["dropped_unparseable"] == 5


def test_a_lookalike_that_passage_b_also_answers_is_dropped(settings, app, pairs):
    # Every rating says "answered": so B answers the lookalike too, and it is not a lookalike.
    def respond(prompt):
        if prompt.startswith("Rate this question"):
            return json.dumps({"realism": 5, "answered": True})
        return json.dumps({"question": f"Question {len(prompt)}?"})

    items, stats = generate(pairs, app, ResponderLLM(respond), n_synthetic=0, n_lookalike=3,
                            handwritten=None)  # fmt: skip
    assert items == []
    assert stats["dropped_ambiguous"] == 3


def test_repeated_questions_are_kept_once(settings, app, pairs):
    def respond(prompt):
        if prompt.startswith("Rate this question"):
            return json.dumps({"realism": 5, "answered": True})
        return json.dumps({"question": "What  is the PTO policy?"})

    items, stats = generate(pairs, app, ResponderLLM(respond), n_synthetic=4, n_lookalike=0,
                            handwritten=None)  # fmt: skip
    assert [i.question for i in items] == ["What is the PTO policy?"]
    assert stats["dropped_duplicate"] == 3
