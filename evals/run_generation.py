"""Generation evals: answer the golden set in agent and single-shot mode, then judge the answers.

    HACKSTACK_PROFILE=local uv run python -m evals.run_generation            # everything
    HACKSTACK_PROFILE=local uv run python -m evals.run_generation --per-kind 6   # a subset

Each question is asked as its ``as_user`` through the real :class:`~hackstack.agent.runtime.Agent`
(tools, citations, ACL), once per mode. What is scored depends on the question's kind:

- **answerable** (synthetic, lookalike, handwritten): the faithfulness and relevance judges
  (``evals/judges.py``), plus two deterministic checks: was a passage from a labelled document
  *shown* to the model (``source_doc_hit``), and did the answer *cite* one (``cited_doc_hit``).
- **unanswerable / forbidden:** the refusal judge; for forbidden ones, whether a restricted
  document reached the model at all (``leak``, which the ACL filter should make impossible).
- **red team** (``evals/redteam.yaml``): prompt injections in the user's turn, checked with
  string canaries and the forbidden-document list, no judge.

Records are appended to a JSONL file in ``evals/runs/`` as they finish, keyed by question and
mode, so an interrupted run resumes where it stopped. ``--labels`` compares the judges with hand
labels (agreement, Cohen's kappa, and how many labelled failures each judge caught).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from evals.golden import EVALS_DIR, GOLDEN_PATH, GoldenItem, dump_report, golden_sha256
from evals.judges import (
    Passage,
    judge_faithfulness,
    judge_refusal,
    judge_relevance,
    passages_from_messages,
)
from hackstack.config import config_hash

if TYPE_CHECKING:
    from hackstack.agent.runtime import Agent
    from hackstack.llm.base import LLMProvider

MODES = ("agent", "single_shot")
ANSWERABLE = ("synthetic", "lookalike", "handwritten")
NO_ANSWER = ("unanswerable", "forbidden")
REDTEAM_PATH = EVALS_DIR / "redteam.yaml"
LABELS_PATH = EVALS_DIR / "judge_labels.jsonl"
RUNS_DIR = EVALS_DIR / "runs"


class Case(BaseModel):
    """One question to answer: a golden item or a red-team item."""

    id: str
    kind: str  # a golden kind, or "redteam"
    question: str
    as_user: str
    relevant_docs: list[str] = Field(default_factory=list)
    forbidden_docs: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)  # red team canaries
    attack: str | None = None


def load_redteam(path: str | Path = REDTEAM_PATH) -> list[Case]:
    entries = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [Case(kind="redteam", **{**e, "question": " ".join(e["question"].split())})
            for e in entries]  # fmt: skip


def cases_from(golden: Iterable[GoldenItem], redteam: Iterable[Case] = ()) -> list[Case]:
    cases = [
        Case(
            id=it.id,
            kind=it.kind,
            question=it.question,
            as_user=it.as_user,
            relevant_docs=it.relevant_docs,
            forbidden_docs=it.forbidden_docs,
        )
        for it in golden
    ]
    return cases + list(redteam)


def select(cases: list[Case], *, per_kind: int | None = None, kinds: Sequence[str] = ()) -> list:
    """The first ``per_kind`` questions of each kind (golden order), of ``kinds`` only."""
    out, seen = [], {}
    for c in cases:
        if kinds and c.kind not in kinds:
            continue
        seen[c.kind] = seen.get(c.kind, 0) + 1
        if per_kind is None or seen[c.kind] <= per_kind:
            out.append(c)
    return out


def check_redteam(case: Case, answer: str, shown_docs: Iterable[str]) -> dict[str, Any]:
    """PASS when no canary is in the answer and no forbidden document reached the model."""
    low = answer.lower()
    failures = [f"answer contains {s!r}" for s in case.must_not_contain if s.lower() in low]
    failures += [f"model was shown {d}" for d in sorted(set(shown_docs) & set(case.forbidden_docs))]
    return {"passed": not failures, "failures": failures}


# --- answering and judging ------------------------------------------------------------------------


async def answer_case(agent: Agent, case: Case, mode: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": case.id,
        "kind": case.kind,
        "mode": mode,
        "as_user": case.as_user,
        "question": case.question,
    }
    try:
        result = await agent.ask(case.question, user=case.as_user, mode=mode)  # type: ignore[arg-type]
    except Exception as exc:  # an eval run records failures and moves on
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    shown = [s.doc_id for s in result.sources]
    cited = {s.doc_id for s in result.sources if s.cited}
    record.update(
        answer=result.answer,
        seconds=result.seconds,
        iterations=result.iterations,
        tool_calls=[t.name for t in result.tool_calls],
        stop_reason=result.stop_reason,
        low_grounding=result.low_grounding,
        sources=[
            {"number": s.number, "doc_id": s.doc_id, "chunk_id": s.chunk_id, "cited": s.cited}
            for s in result.sources
        ],  # fmt: skip
        passages=[p.__dict__ for p in passages_from_messages(result.messages)],
        judges={},
    )
    if case.kind in ANSWERABLE:
        record["source_doc_hit"] = bool(set(shown) & set(case.relevant_docs))
        record["cited_doc_hit"] = bool(cited & set(case.relevant_docs))
    if case.forbidden_docs and case.kind != "redteam":
        record["leak"] = bool(set(shown) & set(case.forbidden_docs))
    if case.kind == "redteam":
        record["redteam"] = check_redteam(case, result.answer, shown)
    return record


async def judge_record(
    llm: LLMProvider, record: dict[str, Any], *, opts: dict[str, Any] | None = None
) -> dict[str, Any]:
    if "error" in record:
        return record
    q, a = record["question"], record["answer"]
    judges: dict[str, Any] = {}
    if record["kind"] in ANSWERABLE:
        passages = [Passage(**p) for p in record["passages"]]
        cited = {s["number"] for s in record["sources"] if s["cited"]}
        judges["faithfulness"] = (
            await judge_faithfulness(llm, q, passages, a, cited=cited, opts=opts)
        ).as_dict()
        judges["relevance"] = (await judge_relevance(llm, q, a, opts=opts)).as_dict()
    elif record["kind"] in NO_ANSWER:
        judges["refusal"] = (await judge_refusal(llm, q, a, opts=opts)).as_dict()
    record["judges"] = judges
    return record


def _read_records(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if "error" not in rec:  # errors are retried
                out[(rec["id"], rec["mode"])] = rec
    return out


async def run(
    agent: Agent,
    judge_llm: LLMProvider | None,
    cases: Sequence[Case],
    modes: Sequence[str] = MODES,
    *,
    records_path: Path | None = None,
    judge_opts: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Answer (and judge) every case in every mode; cases alternate modes so a cut run is fair."""
    done = _read_records(records_path)
    if records_path is not None:
        records_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for case in cases:
        for mode in modes:
            rec = done.get((case.id, mode))
            if rec is None:
                rec = await answer_case(agent, case, mode)
                if judge_llm is not None:
                    rec = await judge_record(judge_llm, rec, opts=judge_opts)
                if records_path is not None:
                    with records_path.open("a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(rec) + "\n")
                if progress:
                    progress(_progress_line(rec))
            out.append(rec)
    return out


async def rejudge(
    judge_llm: LLMProvider,
    records: Sequence[dict[str, Any]],
    *,
    judge_opts: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Judge stored answers again (new judge prompt or model) without re-answering anything."""
    out = []
    for rec in records:
        new = await judge_record(judge_llm, dict(rec), opts=judge_opts)
        if progress:
            progress(_progress_line(new))
        out.append(new)
    return out


def _progress_line(rec: dict[str, Any]) -> str:
    if "error" in rec:
        return f"{rec['id']:<24} {rec['mode']:<11} ERROR {rec['error'][:80]}"
    verdicts = " ".join(
        f"{k}={'-' if v['passed'] is None else 'PASS' if v['passed'] else 'FAIL'}"
        for k, v in rec["judges"].items()
    )
    if "redteam" in rec:
        verdicts = "redteam=" + ("PASS" if rec["redteam"]["passed"] else "FAIL")
    return f"{rec['id']:<24} {rec['mode']:<11} {rec['seconds']:6.1f}s {verdicts}"


# --- scoring --------------------------------------------------------------------------------------


def _rate(values: Iterable[bool | None]) -> float | None:
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else None


def _verdict(rec: dict[str, Any], judge: str) -> bool | None:
    return (rec.get("judges", {}).get(judge) or {}).get("passed")


def _both(faithful: bool | None, relevant: bool | None) -> bool | None:
    """Faithful AND relevant: one known FAIL decides it, even if the other verdict is missing."""
    if faithful is False or relevant is False:
        return False
    return None if faithful is None or relevant is None else True


def _answerable_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    faith = [_verdict(r, "faithfulness") for r in recs]
    rel = [_verdict(r, "relevance") for r in recs]
    both = [_both(f, r) for f, r in zip(faith, rel, strict=True)]
    return {
        "n": len(recs),
        "faithfulness": _rate(faith),
        "relevance": _rate(rel),
        "both": _rate(both),
        "source_doc_hit": _rate(r.get("source_doc_hit") for r in recs),
        "cited_doc_hit": _rate(r.get("cited_doc_hit") for r in recs),
        "judge_errors": sum(v is None for v in faith + rel),
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per mode: answerable judge rates (overall and per kind), refusals, red team, latency."""
    out: dict[str, Any] = {}
    for mode in dict.fromkeys(r["mode"] for r in records):
        recs = [r for r in records if r["mode"] == mode]
        ok = [r for r in recs if "error" not in r]
        ans = [r for r in ok if r["kind"] in ANSWERABLE]
        na = [r for r in ok if r["kind"] in NO_ANSWER]
        rt = [r for r in ok if r["kind"] == "redteam"]
        secs = sorted(r["seconds"] for r in ok)
        refusal = [_verdict(r, "refusal") for r in na]
        stats = _answerable_stats(ans)
        out[mode] = {
            "answerable": stats,
            "by_kind": {
                k: {
                    key: v
                    for key, v in _answerable_stats([r for r in ans if r["kind"] == k]).items()
                    if key in ("n", "faithfulness", "relevance", "both", "cited_doc_hit")
                }
                for k in ANSWERABLE
                if any(r["kind"] == k for r in ans)
            },  # fmt: skip
            "no_answer": {
                "n": len(na),
                "refusal": _rate(refusal),
                "judge_errors": sum(v is None for v in refusal),
                "leaks": sum(bool(r.get("leak")) for r in na),
            },
            "redteam": {
                "n": len(rt),
                "passed": _rate(r["redteam"]["passed"] for r in rt),
                "failed": [r["id"] for r in rt if not r["redteam"]["passed"]],
            },
            "latency_s": {
                "p50": secs[len(secs) // 2] if secs else None,
                "p90": secs[min(len(secs) - 1, int(len(secs) * 0.9))] if secs else None,
            },
            "llm_calls_mean": statistics.fmean(r["iterations"] for r in ok) if ok else None,
            "errors": len(recs) - len(ok),
        }
    return out


def agreement(records: Sequence[dict[str, Any]], labels: Sequence[dict[str, Any]]) -> dict:
    """Judge vs hand labels, per judge: raw agreement, Cohen's kappa, labelled failures caught."""
    by_key = {(r["id"], r["mode"]): r for r in records}
    pairs: dict[str, list[tuple[bool, bool]]] = {}
    for lab in labels:
        rec = by_key.get((lab["id"], lab["mode"]))
        got = _verdict(rec, lab["judge"]) if rec is not None else None
        if got is not None:
            pairs.setdefault(lab["judge"], []).append((got, bool(lab["label"])))
    out = {}
    for judge, ps in pairs.items():
        n = len(ps)
        p_o = sum(j == h for j, h in ps) / n
        pj, ph = sum(j for j, _ in ps) / n, sum(h for _, h in ps) / n
        p_e = pj * ph + (1 - pj) * (1 - ph)
        human_fails = [(j, h) for j, h in ps if not h]
        out[judge] = {
            "n": n,
            "agreement": p_o,
            "kappa": (p_o - p_e) / (1 - p_e) if p_e < 1 else None,
            "fail_caught": f"{sum(not j for j, _ in human_fails)}/{len(human_fails)}",
            "false_fail": sum((not j) and h for j, h in ps),
        }
    return out


def _f(x: Any, pct: bool = True) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.0%}" if pct else f"{x:.1f}"
    return str(x)


def format_table(summary: dict[str, Any]) -> str:
    head = ["mode", "answerable", "faithful", "relevant", "faithful & relevant", "source doc shown",
            "cited labelled doc", "refused no-answer", "red team passed", "p50 s", "p90 s",
            "LLM calls"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for mode, s in summary.items():
        a, na, rt = s["answerable"], s["no_answer"], s["redteam"]
        row = [mode, a["n"], _f(a["faithfulness"]), _f(a["relevance"]), _f(a["both"]),
               _f(a["source_doc_hit"]), _f(a["cited_doc_hit"]),
               f"{_f(na['refusal'])} of {na['n']}", f"{_f(rt['passed'])} of {rt['n']}",
               _f(s["latency_s"]["p50"], pct=False), _f(s["latency_s"]["p90"], pct=False),
               _f(s["llm_calls_mean"], pct=False)]  # fmt: skip
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    kinds = list(dict.fromkeys(k for s in summary.values() for k in s["by_kind"]))
    if kinds:
        lines += ["", "| mode | " + " | ".join(f"{k} (n, faithful & relevant)" for k in kinds)
                  + " |", "|" + "---|" * (len(kinds) + 1)]  # fmt: skip
        for mode, s in summary.items():
            cells = [f"{s['by_kind'][k]['n']}, {_f(s['by_kind'][k]['both'])}"
                     if k in s["by_kind"] else "-" for k in kinds]  # fmt: skip
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    notes = []
    for mode, s in summary.items():
        if s["redteam"]["failed"]:
            notes.append(f"{mode} red-team failures: {', '.join(s['redteam']['failed'])}")
        if s["no_answer"]["leaks"]:
            notes.append(f"{mode}: {s['no_answer']['leaks']} forbidden documents LEAKED")
        if s["errors"] or s["answerable"]["judge_errors"]:
            notes.append(f"{mode}: {s['errors']} answer errors, "
                         f"{s['answerable']['judge_errors']} unparseable verdicts")  # fmt: skip
    return "\n".join(lines + ([""] + notes if notes else []))


def format_agreement(agr: dict[str, Any]) -> str:
    if not agr:
        return "(no hand labels matched these records)"
    head = "| judge | labelled | agreement | Cohen's kappa | labelled FAILs caught | false FAILs |"
    lines = [head,
             "|---|---|---|---|---|---|"]  # fmt: skip
    for judge, a in agr.items():
        kappa = "-" if a["kappa"] is None else f"{a['kappa']:.2f}"
        lines.append(f"| {judge} | {a['n']} | {a['agreement']:.0%} | {kappa} | "
                     f"{a['fail_caught']} | {a['false_fail']} |")  # fmt: skip
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------------------


def default_records_path(settings: Any, golden_sha: str) -> Path:
    return (
        RUNS_DIR / f"generation-{settings.profile}-{golden_sha}-{config_hash(settings)[:10]}.jsonl"
    )


def build(settings: Any) -> tuple[Agent, LLMProvider, dict[str, Any]]:
    """The agent under test, the judge, and the judge's call options."""
    from hackstack.agent.runtime import Agent
    from hackstack.llm import make_llm

    agent = Agent.from_settings(settings)
    pinned = {"seed": 7} if settings.llm.provider == "openai_compat" else {}
    agent.llm_opts = {"temperature": 0, **pinned}
    return agent, make_llm(settings, aux=True), pinned


def main(argv: list[str] | None = None) -> int:
    from evals.golden import load_golden
    from hackstack.config import load_settings

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", "-p", help="config profile (default: $HACKSTACK_PROFILE)")
    ap.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--kinds", nargs="+", default=[], help="only these kinds (and/or redteam)")
    ap.add_argument("--per-kind", type=int, default=None, help="first N questions of each kind")
    ap.add_argument("--no-redteam", action="store_true")
    ap.add_argument("--no-judges", action="store_true", help="answer only")
    ap.add_argument("--records", type=Path, default=None, help="JSONL to resume/append to")
    ap.add_argument("--labels", type=Path, default=LABELS_PATH, help="hand labels for agreement")
    ap.add_argument(
        "--rejudge", action="store_true", help="re-judge stored answers into *-rejudged.jsonl"
    )
    args = ap.parse_args(argv)

    settings = load_settings(profile=args.profile)
    sha = golden_sha256(args.golden)
    cases = cases_from(load_golden(args.golden), [] if args.no_redteam else load_redteam())
    cases = select(cases, per_kind=args.per_kind, kinds=args.kinds)
    path = args.records or default_records_path(settings, sha)
    if args.rejudge:
        from hackstack.llm import make_llm

        stored = _read_records(path)
        todo = [stored[(c.id, m)] for c in cases for m in args.modes if (c.id, m) in stored]
        print(f"re-judging {len(todo)} stored answers from {path} with {settings.llm.aux_model}")
        pinned = {"seed": 7} if settings.llm.provider == "openai_compat" else {}
        records = asyncio.run(
            rejudge(make_llm(settings, aux=True), todo, judge_opts=pinned, progress=print)
        )
        path = path.with_name(path.stem + "-rejudged.jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    else:
        agent, judge, judge_opts = build(settings)
        models = f"{settings.llm.answer_model} / {settings.llm.aux_model}"
        print(f"{len(cases)} questions x {len(args.modes)} modes; answers and judge: {models}; "
              f"records {path}")  # fmt: skip
        try:
            records = asyncio.run(
                run(agent, None if args.no_judges else judge, cases, args.modes,
                    records_path=path, judge_opts=judge_opts, progress=print)
            )  # fmt: skip
        finally:
            agent.close()
    summary = summarize(records)
    print()
    print(format_table(summary))
    labels = []
    if args.labels.exists():
        labels = [json.loads(x) for x in args.labels.read_text(encoding="utf-8").splitlines() if x]
        print()
        print(format_agreement(agreement(records, labels)))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS_DIR / f"{stamp}-generation-{settings.profile}-{sha}.json"
    dump_report(out, {"profile": settings.profile, "golden_sha256": sha,
                      "config_hash": config_hash(settings), "records": str(path),
                      "summary": summary, "agreement": agreement(records, labels)})  # fmt: skip
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
