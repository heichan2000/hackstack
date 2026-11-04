"""Context expansion: give each final hit its neighbouring chunks from the same document.

Small chunks retrieve precisely but read badly on their own (a list item without its lead-in, an
answer without its question). After reranking, each hit's text becomes the run of chunks around
it, ``radius`` on each side, joined in document order. The anchor stays the hit's ``chunk_id``, so
citations still point at the chunk that matched; ``neighbor_ids`` records what was folded in.

Rules:

- Neighbours come from the store under the *same principals*, so expansion can never surface a
  chunk the user could not have retrieved directly.
- ``same_section`` (default) stops at a sibling heading: a neighbour qualifies when its
  ``section_path`` and the hit's are in one lineage (one is a prefix of the other). The chunker
  merges small sections, so a chunk's path is the heading it *starts* under; exact equality
  would block nearly every expansion on a real corpus, while lineage still keeps "Parental
  leave" out of a "Carryover" answer.
- ``max_chars`` bounds a passage. Neighbours are added nearest first, previous before next, and
  a side stops at the first one that does not fit; the anchor is never cut.
- A hit whose chunk is already inside a higher-ranked passage is dropped, and a passage never
  repeats a chunk an earlier passage showed, so the model never reads the same text twice.

Parent-document expansion (whole doc / thread) is left to the agent's ``get_document`` tool.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hackstack.models import SearchHit

SEPARATOR = "\n\n"


@runtime_checkable
class NeighborStore(Protocol):
    """A store that can fetch chunks by position (``PostgresStore``, the test store)."""

    def neighbors(
        self, windows: list[tuple[str, int, int]], principals: list[str]
    ) -> list[SearchHit]:
        """Chunks with ``lo <= ordinal <= hi`` of each ``(doc_id, lo, hi)`` the principals see."""
        ...


class Expander:
    def __init__(
        self, store: object, *, radius: int = 1, same_section: bool = True, max_chars: int = 4000
    ) -> None:
        self.store = store
        self.radius = max(radius, 0)
        self.same_section = same_section
        self.max_chars = max_chars

    def expand(self, hits: list[SearchHit], principals: list[str]) -> list[SearchHit]:
        if self.radius == 0 or not hits or not isinstance(self.store, NeighborStore):
            return list(hits)
        windows = [
            (h.doc_id, h.ordinal - self.radius, h.ordinal + self.radius)
            for h in hits
            if h.ordinal is not None
        ]
        by_pos: dict[tuple[str, int], SearchHit] = {}
        if windows and principals:
            for n in self.store.neighbors(windows, principals):
                if n.ordinal is not None:
                    by_pos[(n.doc_id, n.ordinal)] = n

        shown: set[tuple[str, int]] = set()
        out: list[SearchHit] = []
        for hit in hits:
            if hit.ordinal is None:
                out.append(hit)
                continue
            if (hit.doc_id, hit.ordinal) in shown:
                continue  # already inside a higher-ranked passage
            shown.add((hit.doc_id, hit.ordinal))
            before, after = self._grow(hit, by_pos, shown)
            if not before and not after:
                out.append(hit)
                continue
            run = [*reversed(before), hit, *after]
            out.append(
                hit.model_copy(
                    update={
                        "text": SEPARATOR.join(c.text for c in run),
                        "neighbor_ids": [c.chunk_id for c in run if c is not hit],
                    }
                )
            )
        return out

    def _grow(
        self,
        hit: SearchHit,
        by_pos: dict[tuple[str, int], SearchHit],
        shown: set[tuple[str, int]],
    ) -> tuple[list[SearchHit], list[SearchHit]]:
        """Neighbours before (nearest first) and after the hit that fit the budget."""
        assert hit.ordinal is not None
        size = len(hit.text)
        sides: dict[int, list[SearchHit]] = {-1: [], 1: []}
        open_sides = {-1, 1}
        for dist in range(1, self.radius + 1):
            for direction in (-1, 1):
                if direction not in open_sides:
                    continue
                pos = (hit.doc_id, hit.ordinal + direction * dist)
                n = by_pos.get(pos)
                if (
                    n is None
                    or pos in shown
                    or (self.same_section and not _same_lineage(n.section_path, hit.section_path))
                    or size + len(SEPARATOR) + len(n.text) > self.max_chars
                ):
                    open_sides.discard(direction)  # keep the run contiguous
                    continue
                size += len(SEPARATOR) + len(n.text)
                sides[direction].append(n)
                shown.add(pos)
        return sides[-1], sides[1]


def _same_lineage(a: list[str], b: list[str]) -> bool:
    """One heading path is a prefix of the other (parent/child, or the same section)."""
    n = min(len(a), len(b))
    return a[:n] == b[:n]
