"""Generate ``golden.jsonl`` from the active app's corpus with the aux LLM.

    uv run python -m evals.make_golden            # HACKSTACK_PROFILE=local for Ollama
    make golden

1. **Synthetic:** sample chunks (spread over documents, short ones skipped) and ask the model
   for one question each chunk answers. The chunk id is the label.
2. **Lookalikes:** pair each of the most similar chunks of *different* documents and ask for a
   question that chunk A answers and chunk B does not. The label is A; B is the trap.
3. **Rating:** a second call rates each question's realism (1-5) and whether the chunk answers
   it; only realism >= 4 and answered survive (Hamel Husain's evals recipe). A lookalike is
   also rated against B and dropped if B answers it too, and repeated questions are dropped.
   The rater is the same small model, so it is a filter, not ground truth: spot-check labels.
4. **Hand-written** items from ``handwritten.yaml`` are added: realistic wording, the planted
   moments, unanswerable and ACL-forbidden questions.

Each question is asked as a persona who can see its chunk, picked deterministically from the
app's ``users.yaml``. Synthetic questions echo their chunk's wording, which flatters full-text
search: compare configs with them, and read the hand-written numbers for the absolute level.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.golden import (
    GOLDEN_PATH,
    HANDWRITTEN_PATH,
    GoldenItem,
    corpus_chunks,
    resolve_handwritten,
    save_golden,
)
from hackstack.models import TextDelta
from hackstack.retrieval.acl import principals_for

if TYPE_CHECKING:
    from hackstack.apps import App
    from hackstack.llm.base import LLMProvider
    from hackstack.models import Chunk, RawDoc

Pair = tuple["RawDoc", "Chunk"]

MIN_REALISM = 4
_WORD = re.compile(r"[a-z0-9]{4,}")  # 4+ letters: a cheap stop-word filter for similarity

SYSTEM = "You write test questions for a company's internal search engine. Reply with JSON only."

QUESTION_PROMPT = """\
Here is a passage from the company document "{title}" (section: {section}).

<passage>
{text}
</passage>

Write one question that an employee might type into the company search box, which this passage \
answers. Use your own words: do not copy phrases from the passage. Under 25 words.
Reply as JSON: {{"question": "..."}}"""

LOOKALIKE_PROMPT = """\
Passage A and passage B come from the same company's documents and look alike.

<passage_a title="{title_a}">
{text_a}
</passage_a>

<passage_b title="{title_b}">
{text_b}
</passage_b>

Write one question that passage A answers and passage B does not, phrased so that a careless \
search engine might return passage B. Use your own words. Under 25 words.
Reply as JSON: {{"question": "..."}}"""

RATE_PROMPT = """\
Rate this question for a company search engine test set.

Question: {question}

<passage>
{text}
</passage>

1. realism, 1-5: how likely is a real employee to ask exactly this, in these words? \
(1 = nobody would, 5 = very natural)
2. answered: does the passage alone fully answer the question?
Reply as JSON: {{"realism": <1-5>, "answered": true or false}}"""


# --- pure helpers -------------------------------------------------------------------------------


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first ``{...}`` in ``text`` as a dict, or None. Models like to wrap JSON in prose."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _stable(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def sample_chunks(pairs: list[Pair], n: int, *, seed: int = 7, min_chars: int = 200) -> list[Pair]:
    """``n`` distinct chunks, round-robin over documents in a seeded order; short ones skipped."""
    rng = random.Random(seed)
    by_doc: dict[str, list[Pair]] = {}
    for raw, c in pairs:
        if len(c.text) >= min_chars:
            by_doc.setdefault(c.doc_id, []).append((raw, c))
    for chunks in by_doc.values():
        rng.shuffle(chunks)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    picked: list[Pair] = []
    while len(picked) < n and any(by_doc.values()):
        for doc in docs:
            if by_doc[doc] and len(picked) < n:
                picked.append(by_doc[doc].pop())
    return picked


def asker_for(chunk: Chunk, app: App) -> str | None:
    """A persona who may see ``chunk``, chosen by the chunk id's hash (spreads, deterministic)."""
    acl = set(chunk.acl_principals)
    allowed = [p.id for p in app.users if acl & set(principals_for(p.principal()))]
    if not allowed:
        return None
    return allowed[_stable(chunk.chunk_id) % len(allowed)]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def lookalike_pairs(
    pairs: list[Pair], n: int, *, min_chars: int = 200
) -> list[tuple[Pair, Pair, float]]:
    """The ``n`` most similar (A, B) chunk pairs from different documents, one per A document.

    Similarity is Jaccard over 4+ letter words: cheap, offline and good enough to find chunks a
    retriever could mix up (the two PTO pages, the two code-review pages ...).
    """
    if n <= 0:
        return []
    pool = [(p, _words(p[1].text)) for p in pairs if len(p[1].text) >= min_chars]
    scored: list[tuple[float, int, int]] = []
    for i, (pa, wa) in enumerate(pool):
        for j, (pb, wb) in enumerate(pool):
            if pa[1].doc_id == pb[1].doc_id or not wa or not wb:
                continue
            scored.append((len(wa & wb) / len(wa | wb), i, j))
    scored.sort(key=lambda t: (-t[0], pool[t[1]][0][1].chunk_id, pool[t[2]][0][1].chunk_id))
    out: list[tuple[Pair, Pair, float]] = []
    used_docs: set[str] = set()
    for sim, i, j in scored:
        a, b = pool[i][0], pool[j][0]
        if a[1].doc_id in used_docs:
            continue
        used_docs.add(a[1].doc_id)
        out.append((a, b, sim))
        if len(out) == n:
            break
    return out


# --- LLM calls ----------------------------------------------------------------------------------


async def _complete(llm: LLMProvider, prompt: str, opts: dict[str, Any]) -> str:
    parts = []
    async for event in llm.stream(SYSTEM, [{"role": "user", "content": prompt}], [], **opts):
        if isinstance(event, TextDelta):
            parts.append(event.text)
    return "".join(parts)


def _section(chunk: Chunk) -> str:
    return " > ".join(chunk.section_path) or "(top)"


def generate(
    pairs: list[Pair],
    app: App,
    llm: LLMProvider,
    *,
    n_synthetic: int = 40,
    n_lookalike: int = 10,
    seed: int = 7,
    handwritten: str | Path | None = HANDWRITTEN_PATH,
    opts: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[GoldenItem], dict[str, int]]:
    """Build the golden set; returns the items and counts of what was kept and dropped."""
    opts = {"temperature": 0, "max_tokens": 300, **(opts or {})}
    stats = {"asked": 0, "kept": 0, "dropped_unparseable": 0, "dropped_duplicate": 0,
             "dropped_unanswered": 0, "dropped_realism": 0, "dropped_ambiguous": 0,
             "dropped_no_asker": 0}  # fmt: skip
    items: list[GoldenItem] = []
    seen: set[str] = set()

    async def ask(prompt: str) -> dict[str, Any] | None:
        return parse_json_object(await _complete(llm, prompt, opts))

    async def rate(question: str, chunk: Chunk) -> tuple[bool, int]:
        rating = await ask(RATE_PROMPT.format(question=question, text=chunk.text)) or {}
        try:
            realism = int(rating.get("realism", 0))
        except (TypeError, ValueError):
            realism = 0
        return rating.get("answered") is True, realism

    async def one(kind: str, a: Pair, b: Pair | None, n: int) -> None:
        raw, chunk = a
        stats["asked"] += 1
        user = asker_for(chunk, app)
        if user is None:
            stats["dropped_no_asker"] += 1
            return
        if b is None:
            prompt = QUESTION_PROMPT.format(
                title=raw.title, section=_section(chunk), text=chunk.text
            )
        else:
            prompt = LOOKALIKE_PROMPT.format(
                title_a=raw.title, text_a=chunk.text, title_b=b[0].title, text_b=b[1].text
            )
        got = await ask(prompt)
        question = (got or {}).get("question")
        if not isinstance(question, str) or not question.strip():
            stats["dropped_unparseable"] += 1
            return
        question = " ".join(question.split())
        if question.lower() in seen:
            stats["dropped_duplicate"] += 1
            return
        answered, realism = await rate(question, chunk)
        if not answered:
            stats["dropped_unanswered"] += 1
            return
        if realism < MIN_REALISM:
            stats["dropped_realism"] += 1
            return
        # A lookalike is only a trap if B does *not* answer it; else its label is ambiguous.
        if b is not None and (await rate(question, b[1]))[0]:
            stats["dropped_ambiguous"] += 1
            return
        seen.add(question.lower())
        stats["kept"] += 1
        prefix = "syn" if kind == "synthetic" else "look"
        items.append(
            GoldenItem(
                id=f"{prefix}-{n:03d}",
                question=question,
                as_user=user,
                kind=kind,
                relevant=[chunk.chunk_id],
                relevant_docs=[chunk.doc_id],
                source_chunk=chunk.chunk_id,
                confusable_chunk=b[1].chunk_id if b is not None else None,
                realism=realism,
            )
        )
        if progress:
            progress(f"{prefix}-{n:03d} [{user}] {question}")

    async def run_all() -> None:
        for n, pair in enumerate(sample_chunks(pairs, n_synthetic, seed=seed), start=1):
            await one("synthetic", pair, None, n)
        for n, (a, b, _) in enumerate(lookalike_pairs(pairs, n_lookalike), start=1):
            await one("lookalike", a, b, n)

    asyncio.run(run_all())
    if handwritten is not None:
        items.extend(resolve_handwritten(handwritten, pairs))
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate golden ids")
    return items, stats


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from hackstack.apps import load_app
    from hackstack.config import load_settings
    from hackstack.llm import make_llm

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", "-p", help="config profile (default: $HACKSTACK_PROFILE)")
    ap.add_argument("--synthetic", type=int, default=40, help="chunks to sample (default 40)")
    ap.add_argument("--lookalikes", type=int, default=10, help="lookalike pairs (default 10)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=GOLDEN_PATH)
    args = ap.parse_args(argv)

    settings = load_settings(profile=args.profile)
    app = load_app(settings)
    pairs = corpus_chunks(settings)
    llm = make_llm(settings, aux=True)
    opts = {"seed": args.seed} if settings.llm.provider == "openai_compat" else {}
    print(f"{len(pairs)} chunks; asking {settings.llm.aux_model} ({settings.profile} profile)")
    items, stats = generate(
        pairs, app, llm, n_synthetic=args.synthetic, n_lookalike=args.lookalikes,
        seed=args.seed, opts=opts, progress=print,
    )  # fmt: skip
    save_golden(args.out, items)
    by_kind: dict[str, int] = {}
    for it in items:
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
    print(f"wrote {len(items)} items to {args.out}: {by_kind}")
    print(f"generation: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
