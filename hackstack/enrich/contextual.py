"""Contextual retrieval: a 50-100 token blurb per chunk that situates it in its document.

Anthropic's recipe: give the aux model the whole document and one chunk, ask for a short
context, and prepend it before *both* embedding and BM25. Here that means ``chunk.context``
(its own column, so it can be shown, regenerated or dropped) and ``embed_text`` rebuilt as
``breadcrumb + context + text``; the store's ``tsvector`` is generated from ``embed_text``.

Every call for one document shares a byte-identical prefix (system prompt, then the document
block marked ``cache_control``), with the chunk after it:

* on Claude the document is written to the prompt cache once and read by every other chunk.
  Below the model's minimum (512 tokens on current models) nothing is cached and no error is
  raised, so a large document whose calls never report ``cache_read_input_tokens`` is logged;
* on Ollama the same prefix lets the server reuse its KV cache between calls.

A document over ``max_doc_tokens`` is split by its headings: each chunk is situated in the
smallest enclosing section group that fits. A chunk whose own section is still over budget is
skipped rather than sent truncated. Turning enrichment on, or changing the model, prompt or
budget, changes :attr:`ContextualEnricher.fingerprint`, which the sync manager hashes, so
already-indexed documents are re-processed.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from hackstack.chunking.markdown import count_tokens
from hackstack.models import Chunk, ParsedDoc, RawDoc, Stop, TextDelta
from hackstack.parsing.sections import split_sections

if TYPE_CHECKING:
    from hackstack.config import Secrets, Settings
    from hackstack.llm.base import LLMProvider

log = logging.getLogger(__name__)

#: Bump when the prompt changes: it is part of the fingerprint, so the index is rebuilt.
PROMPT_VERSION = "ctx-v1"

#: Claude's smallest cacheable prefix on current models; shorter prefixes silently miss.
MIN_CACHEABLE_TOKENS = 512

SYSTEM = (
    "You write short retrieval context for chunks of company documents. Given a document "
    "and one chunk of it, say in 50-100 tokens where the chunk sits in the document and what "
    "it is about: name the document, the section, and the people, teams, products, dates or "
    "policies it refers to, so the chunk can be found by search on its own. Use only facts "
    "from the document. Do not repeat the chunk. Answer with the context and nothing else."
)

ASK = (
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document "
    "for the purposes of improving search retrieval of the chunk. Answer only with the "
    "succinct context and nothing else."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LABEL = re.compile(r"^\s*(?:context|succinct context)\s*:\s*", re.IGNORECASE)


class EnrichStats(BaseModel):
    """What one or more ``enrich`` calls did; ``usage`` sums the providers' token counts."""

    chunks: int = 0
    enriched: int = 0
    failed: int = 0  # empty answer, refusal or truncation: the chunk is left as it was
    skipped: int = 0  # its own section is over the budget
    split_docs: int = 0  # documents over the budget, situated section by section
    calls: int = 0
    seconds: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)

    def add(self, other: EnrichStats) -> None:
        for name in ("chunks", "enriched", "failed", "skipped", "split_docs", "calls"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.seconds = round(self.seconds + other.seconds, 3)
        for key, value in other.usage.items():
            self.usage[key] = self.usage.get(key, 0) + value


class ContextualEnricher:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        model: str | None = None,
        max_doc_tokens: int = 50_000,
        max_context_tokens: int = 150,
        max_output_tokens: int = 1024,
        concurrency: int = 1,
        domain_hint: str | None = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.max_doc_tokens = max_doc_tokens
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.concurrency = max(1, concurrency)
        self.domain_hint = domain_hint
        self.system = SYSTEM if not domain_hint else f"{SYSTEM}\n\nAbout the company: {domain_hint}"

    @classmethod
    def from_settings(
        cls, settings: Settings, secrets: Secrets | None = None
    ) -> ContextualEnricher:
        from hackstack.llm import make_llm

        e = settings.enrich
        return cls(
            make_llm(settings, secrets, aux=True),
            model=e.model or settings.llm.aux_model,
            max_doc_tokens=e.max_doc_tokens,
            max_context_tokens=e.max_context_tokens,
            concurrency=e.concurrency,
            domain_hint=e.domain_hint,
        )

    @property
    def fingerprint(self) -> str:
        """Everything that decides the contexts; part of the sync manager's doc hash."""
        hint = ""
        if self.domain_hint:
            hint = ":hint=" + hashlib.sha256(self.domain_hint.encode("utf-8")).hexdigest()[:12]
        return (
            f"contextual:{PROMPT_VERSION}:{self.model or ''}:{self.max_doc_tokens}:"
            f"{self.max_context_tokens}{hint}"
        )

    # --- public ------------------------------------------------------------------------------

    def enrich(self, raw: RawDoc, parsed: ParsedDoc, chunks: list[Chunk]) -> EnrichStats:
        """Set ``context`` on each chunk and rebuild its ``embed_text``; returns what it did.

        Raises the provider's error if the LLM itself fails (unreachable, bad model): like the
        embedder, that is infrastructure, not a per-document problem.
        """
        return asyncio.run(self.aenrich(raw, parsed, chunks))

    async def aenrich(self, raw: RawDoc, parsed: ParsedDoc, chunks: list[Chunk]) -> EnrichStats:
        started = time.perf_counter()
        stats = EnrichStats(chunks=len(chunks))
        jobs: list[tuple[Chunk, str]] = []
        whole = parsed.markdown
        if count_tokens(whole) <= self.max_doc_tokens:
            jobs = [(c, whole) for c in chunks]
        else:
            stats.split_docs = 1
            sections = split_sections(whole)
            for chunk in chunks:
                part = _enclosing_part(sections, chunk.section_path, self.max_doc_tokens)
                if part is None:
                    stats.skipped += 1
                else:
                    jobs.append((chunk, part))

        if jobs:
            # The first call writes the document to the cache; the rest can then read it.
            await self._one(raw, *jobs[0], stats)
            sem = asyncio.Semaphore(self.concurrency)

            async def bounded(job: tuple[Chunk, str]) -> None:
                async with sem:
                    await self._one(raw, *job, stats)

            await asyncio.gather(*(bounded(j) for j in jobs[1:]))
        stats.seconds = round(time.perf_counter() - started, 3)
        self._check_cache(raw, jobs, stats)
        return stats

    # --- internals ---------------------------------------------------------------------------

    async def _one(self, raw: RawDoc, chunk: Chunk, document: str, stats: EnrichStats) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'<document title="{raw.title}">\n{document}\n</document>',
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": ASK.format(chunk=chunk.text)},
                ],
            }
        ]
        opts: dict[str, Any] = {"max_tokens": self.max_output_tokens, "temperature": 0}
        if self.model:
            opts["model"] = self.model
        parts: list[str] = []
        stop: Stop | None = None
        stats.calls += 1
        async for event in self.llm.stream(self.system, messages, [], **opts):
            if isinstance(event, TextDelta):
                parts.append(event.text)
            elif isinstance(event, Stop):
                stop = event
        if stop is not None:
            for key, value in stop.usage.items():
                stats.usage[key] = stats.usage.get(key, 0) + value
        context = clean_context("".join(parts), max_tokens=self.max_context_tokens)
        if not context or stop is None or stop.reason in ("refusal", "max_tokens"):
            stats.failed += 1
            return
        _apply(chunk, context)
        stats.enriched += 1

    def _check_cache(self, raw: RawDoc, jobs: list[tuple[Chunk, str]], stats: EnrichStats) -> None:
        """Warn when a cacheable document was sent several times and never read from cache."""
        if "cache_read_input_tokens" not in stats.usage or stats.usage["cache_read_input_tokens"]:
            return
        docs = {doc for _, doc in jobs}
        repeated = len(jobs) > len(docs)
        if repeated and max((count_tokens(d) for d in docs), default=0) >= MIN_CACHEABLE_TOKENS:
            log.warning(
                "contextual enrichment of %s: %d calls, cache_read_input_tokens stayed 0 - "
                "prompt caching is not working, so every chunk pays for the whole document",
                raw.doc_id,
                stats.calls,
            )


def clean_context(text: str, *, max_tokens: int) -> str:
    """The model's answer as one line of plain text: no thinking, label, quotes or padding."""
    text = _THINK.sub("", text)
    text = " ".join(text.split())
    text = _LABEL.sub("", text).strip().strip('"').strip("'").strip()
    words = text.split()
    if count_tokens(text) > max_tokens:
        kept: list[str] = []
        for word in words:
            if count_tokens(" ".join([*kept, word])) > max_tokens:
                break
            kept.append(word)
        text = " ".join(kept)
    return text


def _apply(chunk: Chunk, context: str) -> None:
    """Insert the context between the breadcrumb and the chunk text in ``embed_text``."""
    chunk.context = context
    if chunk.embed_text.endswith(chunk.text):
        head = chunk.embed_text[: len(chunk.embed_text) - len(chunk.text)]
        chunk.embed_text = f"{head}{context}\n\n{chunk.text}"
    else:
        chunk.embed_text = f"{context}\n\n{chunk.embed_text}"


def _enclosing_part(sections: list, path: list[str], budget: int) -> str | None:
    """The smallest-depth section group containing ``path`` that fits ``budget`` tokens.

    Tries the chunk's top-level section with everything under it, then one level deeper, down
    to the chunk's own section. ``None`` if even that is over budget.
    """
    for depth in range(1, max(len(path), 1) + 1):
        prefix = path[:depth]
        group = [s for s in sections if s.path[:depth] == prefix]
        text = "\n\n".join(_render(s) for s in group)
        if group and count_tokens(text) <= budget:
            return text
    return None


def _render(section: Any) -> str:
    if not section.path:
        return section.text
    heading = "#" * min(len(section.path), 6) + " " + section.path[-1]
    return f"{heading}\n\n{section.text}" if section.text else heading
