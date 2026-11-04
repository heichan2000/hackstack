"""Generation evals: agent vs single-shot, judges by question kind, red team, resumable records."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evals.golden import GoldenItem
from evals.run_generation import (
    REDTEAM_PATH,
    Case,
    agreement,
    cases_from,
    check_redteam,
    format_table,
    load_redteam,
    run,
    summarize,
)
from hackstack.agent.runtime import Agent
from hackstack.apps import load_app
from hackstack.config import load_settings
from hackstack.models import Stop, TextDelta
from hackstack.parsing import router as router_mod
from tests.fakes import FakeLLM

REPO = Path(__file__).resolve().parents[1]
APPS = REPO / "apps"
CORPUS = APPS / "company_brain" / "sample_corpus"


class RuleJudge(FakeLLM):
    """PASS for everything, except a faithfulness check on an answer containing "WRONG"."""

    def _next_turn(self, messages):
        prompt = messages[-1]["content"]
        answer = prompt.split("<answer>")[-1]
        verdict = "FAIL" if "Is every factual claim" in prompt and "WRONG" in answer else "PASS"
        return [TextDelta(text=json.dumps({"reason": "rule", "verdict": verdict})),
                Stop(reason="end_turn")]  # fmt: skip


# --- red team -------------------------------------------------------------------------------------


def _rt(**kw) -> Case:
    base = {"id": "rt-x", "kind": "redteam", "question": "q", "as_user": "eng@northwind.test"}
    return Case(**{**base, **kw})


def test_check_redteam_catches_canaries_and_forbidden_passages():
    case = _rt(must_not_contain=["146,000"], forbidden_docs=["hr/bands"])
    assert check_redteam(case, "I can't share that.", shown_docs=["pto"]) == {
        "passed": True, "failures": []
    }  # fmt: skip
    leaked = check_redteam(case, "The band midpoint is $146,000.", shown_docs=[])
    assert leaked["passed"] is False and "146,000" in leaked["failures"][0]
    shown = check_redteam(case, "No.", shown_docs=["hr/bands"])
    assert shown["passed"] is False and "hr/bands" in shown["failures"][0]
    assert (
        check_redteam(_rt(must_not_contain=["Evil.Test"]), "see evil.test", [])["passed"] is False
    )


def test_redteam_file_is_well_formed():
    items = load_redteam(REDTEAM_PATH)
    assert len(items) >= 5 and len({i.id for i in items}) == len(items)
    settings = load_settings(REPO / "hackstack.yaml", env_file=None)
    app = load_app(settings, apps_dir=APPS)
    users = {u.id for u in app.users}
    for it in items:
        assert it.as_user in users
        assert it.must_not_contain or it.forbidden_docs
        for doc in it.forbidden_docs:
            assert (CORPUS / doc.removeprefix("local_files:")).is_file(), doc


# --- summary --------------------------------------------------------------------------------------


def _rec(mode, kind, **kw):
    rec = {"id": f"{kind}-{len(kw)}", "mode": mode, "kind": kind, "seconds": 1.0,
           "iterations": 1, "tool_calls": [], "sources": [], "judges": {}}  # fmt: skip
    rec.update(kw)
    return rec


def _v(passed):
    return {"passed": passed, "reason": ""}


def test_summarize_rates_per_mode_and_kind():
    records = [
        _rec("agent", "synthetic", judges={"faithfulness": _v(True), "relevance": _v(True)},
             source_doc_hit=True, cited_doc_hit=True, seconds=2.0),
        _rec("agent", "handwritten", judges={"faithfulness": _v(False), "relevance": _v(True)},
             source_doc_hit=True, cited_doc_hit=False, seconds=4.0),
        _rec("agent", "handwritten", judges={"faithfulness": _v(None), "relevance": _v(False)},
             source_doc_hit=False, cited_doc_hit=False, seconds=6.0),
        _rec("agent", "unanswerable", judges={"refusal": _v(True)}),
        _rec("agent", "forbidden", judges={"refusal": _v(False)}, leak=False),
        _rec("agent", "redteam", redteam={"passed": False, "failures": ["x"]}, id="rt-1"),
        _rec("single_shot", "synthetic", judges={"faithfulness": _v(True), "relevance": _v(True)},
             source_doc_hit=True, cited_doc_hit=True),
        _rec("single_shot", "synthetic", error="boom"),
    ]  # fmt: skip
    s = summarize(records)
    a = s["agent"]
    assert a["answerable"]["n"] == 3
    assert a["answerable"]["faithfulness"] == pytest.approx(1 / 2)  # the None is excluded...
    assert a["answerable"]["judge_errors"] == 1  # ...and counted
    assert a["answerable"]["relevance"] == pytest.approx(2 / 3)
    assert a["answerable"]["both"] == pytest.approx(1 / 3)  # faithful AND relevant
    assert a["answerable"]["cited_doc_hit"] == pytest.approx(1 / 3)
    assert a["by_kind"]["handwritten"]["n"] == 2
    assert a["no_answer"] == {"n": 2, "refusal": 0.5, "judge_errors": 0, "leaks": 0}
    assert a["redteam"] == {"n": 1, "passed": 0.0, "failed": ["rt-1"]}
    assert a["latency_s"]["p50"] == 2.0 and a["errors"] == 0  # [1, 1, 1, 2, 4, 6]
    assert s["single_shot"]["errors"] == 1 and s["single_shot"]["answerable"]["n"] == 1
    table = format_table(s)
    assert "| agent |" in table and "| single_shot |" in table


def test_agreement_and_kappa():
    records = [
        {"id": "a", "mode": "agent", "judges": {"faithfulness": _v(True)}},
        {"id": "b", "mode": "agent", "judges": {"faithfulness": _v(False)}},
        {"id": "c", "mode": "agent", "judges": {"faithfulness": _v(True)}},
        {"id": "d", "mode": "agent", "judges": {"faithfulness": _v(True)}},
    ]
    labels = [
        {"id": "a", "mode": "agent", "judge": "faithfulness", "label": True},
        {"id": "b", "mode": "agent", "judge": "faithfulness", "label": False},
        {"id": "c", "mode": "agent", "judge": "faithfulness", "label": False},
        {"id": "d", "mode": "agent", "judge": "faithfulness", "label": True},
        {"id": "zz", "mode": "agent", "judge": "faithfulness", "label": True},  # no record: skipped
    ]
    got = agreement(records, labels)["faithfulness"]
    assert got["n"] == 4 and got["agreement"] == 0.75
    assert got["kappa"] == pytest.approx(0.5)  # p_o 0.75, p_e (3/4*2/4 + 1/4*2/4) = 0.5
    # the hand labels FAIL b and c; the judge said FAIL on b only: it caught 1 of 2 failures
    assert got["fail_caught"] == "1/2"


# --- end to end on the fakes ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def no_docling():
    saved = router_mod.docling_available
    router_mod.docling_available = lambda: False
    yield
    router_mod.docling_available = saved


@pytest.fixture(scope="module")
def offline_agent(no_docling):
    from evals.run_retrieval import _retriever, offline_index, offline_settings

    settings = offline_settings(load_settings(REPO / "hackstack.yaml", profile="local",
                                              env_file=None))  # fmt: skip
    settings = settings.model_copy(
        update={"rerank": settings.rerank.model_copy(update={"provider": "noop"})}
    )
    store, embedder = offline_index(settings, apps_dir=APPS)
    retriever = _retriever(settings, store, embedder)
    return Agent.from_settings(settings, apps_dir=APPS, llm=FakeLLM(), retriever=retriever)


ENG = "eng@northwind.test"


def test_run_judges_by_kind_and_resumes(offline_agent, tmp_path):
    golden = [
        GoldenItem(id="hw-1", question="How many PTO days carry over?", as_user=ENG,
                   kind="handwritten", relevant=["x#0"],
                   relevant_docs=["local_files:northwind/handbook/pto-policy.md"]),
        GoldenItem(id="na-1", question="What's the wifi password?", as_user="eng@northwind.test",
                   kind="unanswerable"),
    ]  # fmt: skip
    cases = cases_from(golden, [_rt(id="rt-1", must_not_contain=["evil.test"])])
    out = tmp_path / "gen.jsonl"
    judge = RuleJudge()
    records = asyncio.run(
        run(offline_agent, judge, cases, ["agent", "single_shot"], records_path=out)
    )
    assert len(records) == 6
    by = {(r["id"], r["mode"]): r for r in records}
    hw_ss = by[("hw-1", "single_shot")]
    assert set(hw_ss["judges"]) == {"faithfulness", "relevance"}
    assert hw_ss["answer"].startswith("echo:")  # the fake answerer
    assert hw_ss["tool_calls"] == ["retrieve"] and hw_ss["sources"]  # single-shot retrieved
    assert hw_ss["source_doc_hit"] in (True, False)
    assert set(by[("na-1", "agent")]["judges"]) == {"refusal"}
    assert by[("rt-1", "agent")]["redteam"]["passed"] is True
    assert by[("rt-1", "agent")]["judges"] == {}  # red team is string checks, no judge
    assert len(out.read_text(encoding="utf-8").splitlines()) == 6
    calls = len(judge.calls)
    assert calls == 2 * (2 + 1)  # per mode: faithfulness + relevance + refusal

    again = asyncio.run(
        run(offline_agent, judge, cases, ["agent", "single_shot"], records_path=out)
    )
    assert len(again) == 6 and len(judge.calls) == calls  # all resumed from the file


def test_enriched_settings_use_their_own_database_and_turn_enrichment_on():
    from evals.suite import enriched_settings

    settings = load_settings(REPO / "hackstack.yaml", profile="local", env_file=None)
    es = enriched_settings(settings)
    assert es.enrich.contextual is True and settings.enrich.contextual is False
    assert "dbname=hackstack_eval_enriched" in es.store.dsn
    assert "hackstack_eval_enriched" not in settings.store.dsn
    assert es.profile == "local"


def test_rejudge_replaces_verdicts_without_answering_again():
    from evals.run_generation import rejudge

    stored = [
        {"id": "hw-1", "mode": "agent", "kind": "handwritten", "question": "q", "answer": "WRONG",
         "sources": [], "passages": [], "judges": {"faithfulness": _v(True)}},
        {"id": "na-1", "mode": "agent", "kind": "unanswerable", "question": "q", "answer": "no",
         "sources": [], "passages": [], "judges": {}},
    ]  # fmt: skip
    judge = RuleJudge()
    out = asyncio.run(rejudge(judge, stored))
    assert out[0]["judges"]["faithfulness"]["passed"] is False  # the new judge's verdict
    assert out[0]["judges"]["relevance"]["passed"] is True
    assert set(out[1]["judges"]) == {"refusal"}
    assert stored[0]["judges"] == {"faithfulness": _v(True)}  # the input is not modified
    assert len(judge.calls) == 3
