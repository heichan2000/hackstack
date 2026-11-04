"""company_brain's own agent tools: ``find_experts`` and ``whats_changed``.

Both read through the asker's principals only (fail closed), like the core tools:

- ``find_experts(topic)`` searches, then ranks the **owners** of the documents that came back,
  weighted by how high each document ranked and how recently it was updated. It returns the
  ranked people as a preamble and one citable passage per document as evidence. Owners are
  looked up under the asker's principals too, so a document they cannot read never puts its
  owner's name in front of them.
- ``whats_changed(since, topic?)`` lists the documents updated on or after a date, newest first,
  with each one's opening passage (or, with a topic, a date-filtered search), so the model can
  summarize the changes with dated citations.

Conflicts between an old and a new document are handled by the prompt (``prompts/system.md``):
every passage title carries its "(updated YYYY-MM-DD)" date, and the prompt asks for both dated
sources and a recommendation of the newer one.

Loaded by ``Agent.from_settings`` because ``app.yaml`` lists these names.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from hackstack.agent.tools import Passages, ToolContext, ToolInputError, tool
from hackstack.models import SearchHit

PEOPLE_FILE = Path(__file__).with_name("people.yaml")
EXPERT_HITS = 10  # passages whose owners find_experts weighs
HALF_LIFE_DAYS = 180  # a document's weight halves every 6 months
UNDATED_WEIGHT = 0.5
MAX_PEOPLE = 5
MAX_EVIDENCE = 6  # citable passages find_experts returns
CHANGED_LIMIT = 12  # documents whats_changed lists
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# --- helpers -----------------------------------------------------------------------------------


def _today(ctx: ToolContext) -> date:
    today = ctx.extra.get("today") if ctx.extra else None
    return today if isinstance(today, date) else date.today()


def _day(value: datetime | date | None) -> date | None:
    if value is None:
        return None
    return value.date() if isinstance(value, datetime) else value


def _dated_title(hit: SearchHit) -> str:
    day = _day(hit.updated_at)
    title = hit.title or hit.doc_id
    return f"{title} (updated {day.isoformat()})" if day else title


def _people() -> dict[str, dict[str, str]]:
    try:
        data = yaml.safe_load(PEOPLE_FILE.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    return {str(k).lower(): v or {} for k, v in (data.get("people") or {}).items()}


def _name_from_email(email: str) -> str:
    local = email.split("@", 1)[0]
    return " ".join(p.capitalize() for p in re.split(r"[._-]+", local) if p) or email


def recency_weight(updated: date | None, today: date) -> float:
    """1.0 today, 0.5 after ``HALF_LIFE_DAYS``; undated documents get ``UNDATED_WEIGHT``."""
    if updated is None:
        return UNDATED_WEIGHT
    age = max((today - updated).days, 0)
    return 0.5 ** (age / HALF_LIFE_DAYS)


def parse_since(value: Any, today: date) -> date:
    """A ``since`` argument as a date: ``YYYY-MM-DD``, a weekday name ("monday" = the most recent
    Monday, today included), ``today``, ``yesterday``, ``this week``, ``last week`` or
    ``N days ago``. Raises :class:`ToolInputError` for anything else or a future date."""
    raw = str(value or "").strip()
    s = raw.lower().removeprefix("since ").strip()
    if s.startswith("last ") and s[5:] in WEEKDAYS:
        s = s[5:]
    day: date | None = None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", s):
        try:
            day = date.fromisoformat(s[:10])
        except ValueError:
            day = None
    elif s == "today":
        day = today
    elif s == "yesterday":
        day = today - timedelta(days=1)
    elif s in WEEKDAYS:
        day = today - timedelta(days=(today.weekday() - WEEKDAYS.index(s)) % 7)
    elif s == "this week":
        day = today - timedelta(days=today.weekday())
    elif s in {"last week", "a week ago", "past week"}:
        day = today - timedelta(days=7)
    elif m := re.fullmatch(r"(?:(\d+) days? ago|(?:last|past) (\d+) days)", s):
        day = today - timedelta(days=int(m.group(1) or m.group(2)))
    if day is None:
        raise ToolInputError(
            f"`since` must be a date like {today.isoformat()} or a weekday name like 'monday'; "
            f"got {raw!r}."
        )
    if day > today:
        raise ToolInputError(
            f"`since` {day.isoformat()} is in the future (today is {today.isoformat()})."
        )
    return day


# --- find_experts ------------------------------------------------------------------------------

_FIND_EXPERTS_DESCRIPTION = """\
Find the people to ask about a topic: the owners of the company documents most relevant to it, \
ranked by how relevant their documents are and how recently they were updated. Use it for \
"who should I ask about X", "who owns X" or "who knows about X" questions, instead of guessing \
names from search results. It returns a ranked list of people with their job titles and the \
documents they own, followed by those documents' passages, which you should cite as the \
evidence for each person. Only documents the person asking may read are used, so an empty \
result means no readable document on the topic lists an owner."""


def find_experts(args: dict[str, Any], ctx: ToolContext) -> Passages:
    topic = str(args.get("topic") or "").strip()
    if not topic:
        raise ToolInputError("find_experts needs a non-empty `topic`.")
    try:
        limit = int(args.get("limit") or 3)
    except (TypeError, ValueError) as exc:
        raise ToolInputError("`limit` must be a whole number from 1 to 5.") from exc
    limit = min(max(limit, 1), MAX_PEOPLE)
    today = _today(ctx)

    hits = ctx.retriever.search(topic, ctx.principals, None, top_n=EXPERT_HITS)
    # Per document: relevance = sum of 1/(rank+1) over its passages, and its best passage.
    relevance: dict[str, float] = defaultdict(float)
    best: dict[str, SearchHit] = {}
    for rank, hit in enumerate(hits):
        relevance[hit.doc_id] += 1.0 / (rank + 1)
        best.setdefault(hit.doc_id, hit)
    # Owners only for documents the asker may read, whatever the retriever returned.
    owners = ctx.retriever.store.doc_owners(list(best), ctx.principals) if best else {}

    score: dict[str, float] = defaultdict(float)
    owned: dict[str, list[str]] = defaultdict(list)
    for doc_id, rel in relevance.items():
        weight = rel * recency_weight(_day(best[doc_id].updated_at), today)
        for person in owners.get(doc_id, []):
            score[person.lower()] += weight
            owned[person.lower()].append(doc_id)
    ranked = sorted(score, key=lambda p: (-score[p], p))[:limit]
    if not ranked:
        return Passages(
            hits=[],
            empty=(
                f"No one found for {topic!r}: none of the documents about it that this person "
                "can read lists an owner. Try search_kb instead."
            ),
        )

    directory = _people()
    lines = [
        f'People to ask about "{topic}", best match first. Each owns the documents listed; '
        "cite the passages below as the evidence."
    ]
    evidence: list[SearchHit] = []
    for i, person in enumerate(ranked, start=1):
        info = directory.get(person, {})
        name = info.get("name") or _name_from_email(person)
        title = f", {info['title']}" if info.get("title") else ""
        docs = sorted(owned[person], key=lambda d: -relevance[d])
        listed = "; ".join(_dated_title(best[d]) for d in docs)
        lines.append(f"{i}. {name} ({person}){title}. Owns: {listed}.")
        for d in docs:
            if best[d] not in evidence and len(evidence) < MAX_EVIDENCE:
                evidence.append(best[d])
    return Passages(hits=evidence, preamble="\n".join(lines))


# --- whats_changed -----------------------------------------------------------------------------

_WHATS_CHANGED_DESCRIPTION = """\
List what changed in the company documents on or after a date, newest first, with each changed \
document's opening passage so you can summarize the changes and cite them. Use it for "what \
changed since Monday", "what's new this week" or "anything new about X lately" questions. Pass \
`since` exactly as the person said it when it is relative ("monday", "yesterday", "this week", \
"last week"), and the tool works out the date; otherwise pass a date (YYYY-MM-DD). Optionally \
pass a `topic` to only include changes about it. Only documents the person asking may \
read are listed, so different people see different changes; use get_document to read a \
changed document in full."""


def whats_changed(args: dict[str, Any], ctx: ToolContext) -> Passages:
    today = _today(ctx)
    since = parse_since(args.get("since"), today)
    topic = str(args.get("topic") or "").strip()
    source = args.get("source") or None

    if topic:
        hits = ctx.retriever.search(
            topic, ctx.principals, {"since": since.isoformat(), "source": source}, top_n=ctx.top_n
        )
    else:
        hits = ctx.retriever.store.changed_since(
            ctx.principals, since, source=source, limit=CHANGED_LIMIT
        )
    hits = sorted(hits, key=lambda h: _day(h.updated_at) or date.min, reverse=True)

    when = f"{since.isoformat()} ({since.strftime('%A')})"
    about = f' about "{topic}"' if topic else ""
    if not hits:
        return Passages(
            hits=[],
            empty=f"Nothing{about} that this person can read changed on or after {when}.",
        )
    docs: dict[str, SearchHit] = {}
    for h in hits:
        docs.setdefault(h.doc_id, h)
    lines = [
        f"{len(docs)} document(s){about} that this person can read changed on or after {when}, "
        "newest first:"
    ]
    for h in docs.values():
        day = _day(h.updated_at)
        when_doc = f"{day.isoformat()} ({day.strftime('%A')})" if day else "undated"
        lines.append(f"- {when_doc}: {h.title or h.doc_id}")
    lines.append(
        "Their passages follow. Summarize what changed with each date, citing the passages."
    )
    return Passages(hits=hits, preamble="\n".join(lines))


tool(
    "find_experts",
    _FIND_EXPERTS_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "What the person needs help with."},
            "limit": {"type": "integer", "description": "How many people, 1-5 (default 3)."},
        },
        "required": ["topic"],
    },
)(find_experts)

tool(
    "whats_changed",
    _WHATS_CHANGED_DESCRIPTION,
    {
        "type": "object",
        "properties": {
            "since": {
                "type": "string",
                "description": "As said: 'monday', 'yesterday', 'this week', or YYYY-MM-DD.",
            },
            "topic": {"type": "string", "description": "Only changes about this, optional."},
            "source": {
                "type": "string",
                "description": "Only this source, e.g. local_files, github, slack, notion.",
            },
        },
        "required": ["since"],
    },
)(whats_changed)
