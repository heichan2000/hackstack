"""Compare retrieval configs on the golden set: ``make eval``.

    HACKSTACK_PROFILE=local make eval                 # against the ingested index
    uv run python -m evals.run_retrieval --offline    # fake embedder + in-memory store, no services

Variants, all on the same index and the same golden set:

- ``dense``         vector search only (``rrf.weights: [1, 0]``), no reranker
- ``lexical``       Postgres full-text search only (``[0, 1]``), no reranker
- ``hybrid``        both, fused with RRF, no reranker
- ``hybrid_rerank`` both, fused, then the profile's reranker over ``rerank.window`` candidates

Scoring uses the ranked list *before* context expansion (expansion folds neighbours into a
passage but does not reorder), cut at ``k <= 10``. Each question is asked as its ``as_user``, so
the ACL filter is in the loop. Unanswerable and forbidden questions have no recall; they report
the top score (the "is there a not-found threshold?" question) and, for forbidden ones, whether
any restricted document leaked.

Every run is saved to ``evals/runs/`` with each variant's ``config_hash``. ``--check`` exits 1
when recall@5 or MRR falls below ``baseline.json`` for this profile; ``--write-baseline``
records the current numbers there.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.golden import EVALS_DIR, GOLDEN_PATH, GoldenItem, dump_report, golden_sha256
from evals.metrics import DEFAULT_KS, hit_rate_at_k, mean_metrics
from hackstack.config import config_hash

if TYPE_CHECKING:
    from hackstack.config import Settings
    from hackstack.embed.base import Embedder
    from hackstack.store.base import Store

VARIANTS = ("dense", "lexical", "hybrid", "hybrid_rerank")
BASELINE_PATH = EVALS_DIR / "baseline.json"
RUNS_DIR = EVALS_DIR / "runs"
K_MAX = max(DEFAULT_KS)
GATED = ("recall@5", "mrr")  # what the regression check holds the line on


def variant_settings(settings: Settings, name: str) -> Settings:
    """A copy of ``settings`` configured as one variant; its ``config_hash`` names the run."""
    noop = settings.rerank.model_copy(update={"provider": "noop"})
    if name == "dense":
        weights, rerank = [1.0, 0.0], noop
    elif name == "lexical":
        weights, rerank = [0.0, 1.0], noop
    elif name == "hybrid":
        weights, rerank = [1.0, 1.0], noop
    elif name == "hybrid_rerank":
        weights, rerank = list(settings.rrf.weights), settings.rerank
    else:
        raise ValueError(f"unknown variant {name!r}; known: {', '.join(VARIANTS)}")
    out = settings.model_copy(
        update={"rrf": settings.rrf.model_copy(update={"weights": weights}), "rerank": rerank}
    )
    out._profile = settings.profile
    return out


def _retriever(vs: Settings, store: Store, embedder: Embedder):
    from hackstack.retrieval.hybrid import HybridRetriever
    from hackstack.retrieval.rerank import build_reranker

    return HybridRetriever(
        store=store,
        embedder=embedder,
        reranker=build_reranker(vs),
        expander=None,
        window=vs.retrieval.window,
        k=vs.rrf.k,
        weights=tuple(vs.rrf.weights),
        rerank_window=vs.rerank.window,
        top_n=K_MAX,
    )


def no_answer_stats(
    answerable: Sequence[float | None], no_answer: Sequence[float | None]
) -> dict[str, Any]:
    """How well "answer only if the top score >= t" separates answerable from no-answer queries.

    ``None`` is a query with no hits at all: it is "no answer" at every threshold. The best
    threshold is the lowest one with the highest accuracy over both groups.
    """
    if not no_answer:
        return {"n": 0}
    ans = [s for s in answerable if s is not None]
    na = [s for s in no_answer if s is not None]
    total = len(answerable) + len(no_answer)
    best_t, best_acc = None, -1.0
    for t in sorted(set(ans) | set(na)):
        correct = sum(s >= t for s in ans) + (len(no_answer) - sum(s >= t for s in na))
        if correct / total > best_acc:
            best_t, best_acc = t, correct / total
    max_na = max(na) if na else None
    above = sum(s > max_na for s in ans) if max_na is not None else len(ans)
    return {
        "n": len(no_answer),
        "top_score_mean_answerable": statistics.fmean(ans) if ans else None,
        "top_score_mean_no_answer": statistics.fmean(na) if na else None,
        "max_no_answer": max_na,
        "answerable_above_max_no_answer": above / len(answerable) if answerable else None,
        "best_threshold": best_t,
        "best_threshold_accuracy": best_acc if best_t is not None else None,
    }


def run_variant(
    name: str,
    vs: Settings,
    items: list[GoldenItem],
    *,
    store: Store,
    embedder: Embedder,
    principals_of: Callable[[str], list[str]],
) -> dict[str, Any]:
    retriever = _retriever(vs, store, embedder)
    queries = []
    for it in items:
        started = time.perf_counter()
        hits = retriever.retrieve(it.question, principals_of(it.as_user)).reranked[:K_MAX]
        queries.append(
            {
                "id": it.id,
                "kind": it.kind,
                "ranked": [h.chunk_id for h in hits],
                "ranked_docs": [h.doc_id for h in hits],
                "top_score": hits[0].score if hits else None,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    by_id = {q["id"]: q for q in queries}
    answerable = [it for it in items if it.answerable]

    def metrics_for(subset: list[GoldenItem]) -> dict[str, float]:
        return mean_metrics([(by_id[it.id]["ranked"], set(it.relevant)) for it in subset])

    kinds = sorted({it.kind for it in answerable})
    forbidden = [it for it in items if it.kind == "forbidden"]
    leaks = sum(bool(set(by_id[it.id]["ranked_docs"]) & set(it.forbidden_docs)) for it in forbidden)
    ms = sorted(q["ms"] for q in queries)
    return {
        "config_hash": config_hash(vs),
        "reranker": vs.rerank.provider,
        "metrics": metrics_for(answerable) if answerable else {"n": 0},
        "by_kind": {
            k: {
                key: v
                for key, v in metrics_for([it for it in answerable if it.kind == k]).items()
                if key in ("n", "recall@5", "recall@10", "mrr", "hit@1")
            }
            for k in kinds
        },
        "doc_hit@5": (
            statistics.fmean(
                hit_rate_at_k(by_id[it.id]["ranked_docs"], set(it.relevant_docs), 5)
                for it in answerable
            )
            if answerable
            else None
        ),
        "no_answer": no_answer_stats(
            [by_id[it.id]["top_score"] for it in answerable],
            [by_id[it.id]["top_score"] for it in items if not it.answerable],
        ),
        "forbidden": {"n": len(forbidden), "leaks": leaks},
        "latency_ms": {"p50": ms[len(ms) // 2], "max": ms[-1]} if ms else {},
        "queries": queries,
    }


def run(
    settings: Settings,
    items: list[GoldenItem],
    *,
    variants: Sequence[str] = VARIANTS,
    store: Store,
    embedder: Embedder,
    principals_of: Callable[[str], list[str]],
    golden_sha: str,
) -> dict[str, Any]:
    """Every variant over ``items``; the report that is printed, saved and checked."""
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile": settings.profile,
        "golden_sha256": golden_sha,
        "items": len(items),
        "by_kind": {k: sum(it.kind == k for it in items) for k in sorted({i.kind for i in items})},
        "variants": {},
    }
    for name in variants:
        vs = variant_settings(settings, name)
        report["variants"][name] = run_variant(
            name, vs, items, store=store, embedder=embedder, principals_of=principals_of
        )
    return report


def _f(x: Any) -> str:
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def format_table(report: dict[str, Any]) -> str:
    """The comparison as a Markdown table, plus the per-kind recall and the no-answer stats."""
    kinds = sorted({k for v in report["variants"].values() for k in v["by_kind"]})
    head = ["variant", "recall@5", "recall@10", "MRR", "nDCG@10", "hit@1", "doc hit@5"]
    head += [f"recall@5 {k}" for k in kinds] + ["leaks", "p50 ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, v in report["variants"].items():
        m = v["metrics"]
        row = [name, m.get("recall@5"), m.get("recall@10"), m.get("mrr"), m.get("ndcg@10"),
               m.get("hit@1"), v["doc_hit@5"]]  # fmt: skip
        row += [v["by_kind"].get(k, {}).get("recall@5") for k in kinds]
        row += [f"{v['forbidden']['leaks']}/{v['forbidden']['n']}", v["latency_ms"].get("p50")]
        lines.append("| " + " | ".join(_f(x) for x in row) + " |")
    n = {k: v.get("n") for k, v in next(iter(report["variants"].values()))["by_kind"].items()}
    lines.append("")
    lines.append(f"answerable questions per kind: {n}; golden {report['golden_sha256']}")
    for name, v in report["variants"].items():
        na = v["no_answer"]
        # Without a reranker the top score is RRF's 1/(k + rank): the same for every query.
        if na.get("n") and v["reranker"] != "noop":
            lines.append(
                f"{name}: top score answerable {_f(na['top_score_mean_answerable'])} vs "
                f"no-answer {_f(na['top_score_mean_no_answer'])}; "
                f"{_f(na['answerable_above_max_no_answer'])} of answerable above the best "
                f"no-answer; best threshold {_f(na['best_threshold'])} -> accuracy "
                f"{_f(na['best_threshold_accuracy'])}"
            )
    return "\n".join(lines)


def check_baseline(report: dict[str, Any], baseline: dict[str, Any], profile: str) -> list[str]:
    """Regressions against ``baseline.json``: empty when every gated metric holds the line."""
    if baseline.get("golden_sha256") != report["golden_sha256"]:
        return [
            f"golden set changed ({report['golden_sha256']} vs baseline "
            f"{baseline.get('golden_sha256')}): rerun with --write-baseline after reviewing"
        ]
    expected = baseline["profiles"][profile]  # KeyError: no baseline for this profile yet
    tol = float(baseline.get("tolerance", 0.0))
    failures = []
    for name, base in expected.items():
        got = report["variants"].get(name)
        if got is None:
            continue
        for key in GATED:
            if key in base and got["metrics"].get(key, 0.0) < base[key] - tol - 1e-9:
                failures.append(
                    f"{name}: {key} {got['metrics'].get(key, 0.0):.3f} < baseline {base[key]:.3f}"
                )
    return failures


def baseline_entry(report: dict[str, Any]) -> dict[str, Any]:
    # Floor, not round: a baseline rounded up would fail against the very run that wrote it.
    return {
        name: {
            "config_hash": v["config_hash"],
            **{
                k: math.floor(v["metrics"][k] * 10_000) / 10_000
                for k in (*GATED, "recall@10", "ndcg@10")
            },
        }
        for name, v in report["variants"].items()
    }


# --- wiring ---------------------------------------------------------------------------------------


def offline_settings(settings: Settings) -> Settings:
    """``settings`` with the fake embedder, as profile ``offline`` (its own baseline entry)."""
    out = settings.model_copy(
        update={"embed": settings.embed.model_copy(update={"provider": "fake", "model": "fake"})}
    )
    out._profile = "offline"
    return out


def offline_index(settings: Settings, *, apps_dir: str | Path = "apps"):
    """The corpus in the in-memory test store with the fake embedder: no services, deterministic."""
    import tests.fake_store  # noqa: F401 - registers store "memory"
    from evals.golden import corpus_chunks
    from tests.fake_store import InMemoryStore
    from tests.fakes import FakeEmbedder

    store, embedder = InMemoryStore(), FakeEmbedder()
    docs: dict[str, Any] = {}
    chunks: dict[str, list] = {}
    for raw, c in corpus_chunks(settings, apps_dir=apps_dir):
        docs[raw.doc_id] = raw
        chunks.setdefault(raw.doc_id, []).append(c)
    for doc_id, cs in chunks.items():
        vectors = embedder.embed_documents([c.embed_text for c in cs])
        for c, v in zip(cs, vectors, strict=True):
            c.embedding = v
        store.upsert(docs[doc_id], cs)
    return store, embedder


def main(argv: list[str] | None = None) -> int:
    import json

    from evals.golden import load_golden
    from hackstack.config import load_settings
    from hackstack.retrieval.acl import resolve_principals

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", "-p", help="config profile (default: $HACKSTACK_PROFILE)")
    ap.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    ap.add_argument("--variants", nargs="+", default=None, choices=VARIANTS)
    ap.add_argument("--offline", action="store_true",
                    help="fake embedder + in-memory store, noop reranker")  # fmt: skip
    ap.add_argument("--check", action="store_true", help="exit 1 below baseline.json")
    ap.add_argument("--write-baseline", action="store_true", help="record these numbers")
    args = ap.parse_args(argv)

    if not args.golden.exists():
        print(f"no golden set at {args.golden}: run `make golden` first", file=sys.stderr)
        return 2
    items = load_golden(args.golden)
    sha = golden_sha256(args.golden)
    settings = load_settings(profile=args.profile)
    principals_of = {}

    def principals(user: str) -> list[str]:
        if user not in principals_of:
            principals_of[user] = resolve_principals(settings, user)
        return principals_of[user]

    if args.offline:
        settings = offline_settings(settings)
        store, embedder = offline_index(settings)
        variants = args.variants or ["dense", "lexical", "hybrid"]
    else:
        from hackstack.embed import build_embedder
        from hackstack.store.postgres import PostgresStore

        store, embedder = PostgresStore.from_settings(settings), build_embedder(settings)
        variants = args.variants or list(VARIANTS)
        if "hybrid_rerank" in variants and settings.rerank.provider == "noop":
            variants.remove("hybrid_rerank")  # identical to hybrid

    report = run(settings, items, variants=variants, store=store, embedder=embedder,
                 principals_of=principals, golden_sha=sha)  # fmt: skip
    close = getattr(store, "close", None)
    if callable(close):
        close()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS_DIR / f"{stamp}-{report['profile']}-{sha}.json"
    dump_report(out, report)
    print(format_table(report))
    print(f"\nsaved {out}")

    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else None
    if args.write_baseline:
        if baseline is None or baseline.get("golden_sha256") != sha:
            # 0.005 absorbs float noise in GPU embeddings; one lost query (1/46) still fails.
            baseline = {"golden_sha256": sha, "tolerance": 0.005, "profiles": {}}
        baseline["profiles"][report["profile"]] = baseline_entry(report)
        dump_report(BASELINE_PATH, baseline)
        print(f"baseline for {report['profile']!r} written to {BASELINE_PATH}")
        return 0
    if baseline is None or report["profile"] not in baseline.get("profiles", {}):
        print(f"no baseline for profile {report['profile']!r} yet (--write-baseline)")
        return 0
    failures = check_baseline(report, baseline, report["profile"])
    for f in failures:
        print(f"REGRESSION {f}", file=sys.stderr)
    if not failures:
        print(f"at or above the {report['profile']!r} baseline")
    return 1 if failures and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
