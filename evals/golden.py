"""The golden set: questions labelled with the chunk ids that answer them.

``golden.jsonl`` holds one :class:`GoldenItem` per line. Labels are chunk ids, and chunk ids
carry the chunker version (``{doc_id}#md1-s400-...-{ordinal}``), so re-chunking changes them;
``tests/test_eval_regression.py`` re-chunks the corpus offline and fails if any label no longer
exists, rather than letting the golden set rot silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

from hackstack.apps import DEFAULT_APPS_DIR, load_app

if TYPE_CHECKING:
    from hackstack.config import Settings
    from hackstack.models import Chunk, RawDoc

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
HANDWRITTEN_PATH = EVALS_DIR / "handwritten.yaml"

Kind = Literal["synthetic", "lookalike", "handwritten", "unanswerable", "forbidden"]


class GoldenItem(BaseModel):
    id: str
    question: str
    as_user: str  # asked as this persona (users.yaml); retrieval runs under their principals
    kind: Kind
    relevant: list[str] = Field(default_factory=list)  # chunk ids; empty = nothing should match
    relevant_docs: list[str] = Field(default_factory=list)
    forbidden_docs: list[str] = Field(default_factory=list)  # must never be returned
    source_chunk: str | None = None  # synthetic / lookalike: the chunk the question came from
    confusable_chunk: str | None = None  # lookalike: the similar chunk that does not answer it
    realism: int | None = None  # 1-5 from the rating pass (synthetic / lookalike)

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def load_golden(path: str | Path = GOLDEN_PATH) -> list[GoldenItem]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [GoldenItem.model_validate_json(line) for line in lines if line.strip()]


def save_golden(path: str | Path, items: list[GoldenItem]) -> None:
    text = "".join(it.model_dump_json(exclude_defaults=True) + "\n" for it in items)
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def golden_sha256(path: str | Path = GOLDEN_PATH) -> str:
    """Hash of the golden set's content (line endings normalized), for results and baselines."""
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def corpus_chunks(
    settings: Settings, *, apps_dir: str | Path = DEFAULT_APPS_DIR, connector: str = "local_files"
) -> list[tuple[RawDoc, Chunk]]:
    """The active app's corpus, parsed and chunked exactly as ingest does, without a store.

    Offline and deterministic: the chunk ids match what ``make ingest-sample`` stored.
    """
    from hackstack.connectors.base import ConnectorFailure
    from hackstack.ingest import ingestor_from_settings

    cfg = load_app(settings, apps_dir=apps_dir).connectors.get(connector)
    ingestor = ingestor_from_settings(settings, connector, {}, cfg=cfg, dry_run=True)
    out: list[tuple[RawDoc, Chunk]] = []
    for raw in ingestor.connector.load_all():
        if isinstance(raw, ConnectorFailure):
            raise ValueError(f"corpus item failed to load: {raw.item_id}: {raw.error}")
        chunks = ingestor.prepare(raw)
        if isinstance(chunks, ConnectorFailure):
            raise ValueError(f"corpus item failed to chunk: {chunks.item_id}: {chunks.error}")
        out.extend((raw, c) for c in chunks)
    return out


def resolve_handwritten(path: str | Path, pairs: list[tuple[RawDoc, Chunk]]) -> list[GoldenItem]:
    """Hand-written items with ``{doc, contains}`` references resolved to chunk ids."""
    entries = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    by_doc: dict[str, list[Chunk]] = {}
    for _, c in pairs:
        by_doc.setdefault(c.doc_id, []).append(c)
    items = []
    for entry in entries:
        refs = entry.pop("relevant", None) or []
        relevant: list[str] = []
        docs: list[str] = []
        for ref in refs:
            doc = ref["doc"]
            if doc not in by_doc:
                raise ValueError(f"{entry['id']}: no document {doc!r} in the corpus")
            needle = (ref.get("contains") or "").lower()
            hits = [c.chunk_id for c in by_doc[doc] if needle in c.text.lower()]
            if not hits:
                raise ValueError(f"{entry['id']}: no chunk of {doc} contains {ref['contains']!r}")
            relevant.extend(h for h in hits if h not in relevant)
            if doc not in docs:
                docs.append(doc)
        items.append(GoldenItem(**entry, relevant=relevant, relevant_docs=docs))
    return items


def dump_report(path: str | Path, report: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
