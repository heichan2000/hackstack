"""Data that flows between the kit's components.

Connector -> RawDoc -> Parser -> ParsedDoc -> Chunker -> Chunk -> Store
Store/retrieval -> SearchHit -> agent -> Citation;  LLMProvider -> LLMEvent stream
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# --- permissions -----------------------------------------------------------------------------


class Principal(BaseModel):
    """Who is asking: a user and the access groups they belong to."""

    user: str
    groups: list[str] = Field(default_factory=list)

    def principals(self) -> list[str]:
        """The principal strings matched against ``ACL.principals`` (and in SQL)."""
        return [f"user:{self.user}", *(f"group:{g}" for g in self.groups)]


class ACL(BaseModel):
    """Who may see a document. Copied onto every chunk. Empty means nobody: fail closed."""

    principals: list[str]

    def allows(self, who: Principal) -> bool:
        return not set(self.principals).isdisjoint(who.principals())


# --- ingest ----------------------------------------------------------------------------------


class RawDoc(BaseModel):
    """One source item, normalized by a connector, permissions included."""

    doc_id: str
    source: str  # connector name, e.g. "local_files", "slack"
    source_type: str = "document"  # document | thread | email | code | page | issue ...
    title: str
    content: str | bytes
    mime: str = "text/markdown"
    url: str | None = None
    owners: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    parent_id: str | None = None
    acl: ACL  # required: a connector that cannot say who may read a doc must not emit it
    extra: dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    path: list[str]  # heading trail, e.g. ["Handbook", "PTO", "Carryover"]
    text: str


class ParsedDoc(BaseModel):
    """A parser's output: Markdown plus its structure."""

    markdown: str
    sections: list[Section] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """The retrieval unit. ``embed_text`` is what gets embedded and full-text indexed."""

    chunk_id: str
    doc_id: str
    ordinal: int
    section_path: list[str] = Field(default_factory=list)
    text: str
    context: str | None = None  # contextual-retrieval blurb, if enrichment ran
    embed_text: str
    source: str
    updated_at: datetime | None = None
    acl_principals: list[str]
    embedding: list[float] | None = Field(default=None, repr=False)  # set by the embedder

    @classmethod
    def for_doc(
        cls,
        raw: RawDoc,
        *,
        ordinal: int,
        text: str,
        section_path: list[str] | None = None,
        embed_text: str | None = None,
    ) -> Chunk:
        """Build a chunk that inherits the document's id, source, timestamp and ACL."""
        return cls(
            chunk_id=f"{raw.doc_id}#{ordinal}",
            doc_id=raw.doc_id,
            ordinal=ordinal,
            section_path=section_path or [],
            text=text,
            embed_text=embed_text if embed_text is not None else text,
            source=raw.source,
            updated_at=raw.updated_at,
            acl_principals=list(raw.acl.principals),
        )


# --- query -----------------------------------------------------------------------------------


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    title: str = ""
    url: str | None = None
    source: str = ""
    section_path: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None
    retriever: str = ""  # dense | lexical | hybrid | rerank


class Citation(BaseModel):
    """A span of the answer grounded in one search result."""

    doc_id: str
    chunk_id: str | None = None
    title: str = ""
    url: str | None = None
    cited_text: str = ""


# --- LLM stream events (normalized across providers) -----------------------------------------


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class CitationEvent(BaseModel):
    type: Literal["citation"] = "citation"
    citation: Citation


class Stop(BaseModel):
    type: Literal["stop"] = "stop"
    reason: str = "end_turn"  # end_turn | tool_use | max_tokens | refusal | ...
    usage: dict[str, int] = Field(default_factory=dict)


LLMEvent = Annotated[TextDelta | ToolCall | CitationEvent | Stop, Field(discriminator="type")]
