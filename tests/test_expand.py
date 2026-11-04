"""Context expansion: neighbouring chunks of the same document, under the same ACL."""

from __future__ import annotations

from hackstack.models import ACL, Chunk, RawDoc, SearchHit
from hackstack.retrieval.expand import Expander
from tests.fake_store import InMemoryStore

EVERYONE = ["group:everyone"]


def seed(store, doc_id, texts, *, acl=None, sections=None, chunk_acls=None):
    raw = RawDoc(
        doc_id=doc_id,
        source="local_files",
        title=doc_id.title(),
        content="",
        acl=ACL(principals=acl or EVERYONE),
    )
    chunks = []
    for i, text in enumerate(texts):
        c = Chunk.for_doc(
            raw,
            ordinal=i,
            text=text,
            section_path=(sections[i] if sections else ["Doc"]),
        )
        if chunk_acls and chunk_acls[i] is not None:
            c.acl_principals = chunk_acls[i]
        c.embedding = [1.0, 0.0]
        chunks.append(c)
    store.upsert(raw, chunks)
    return chunks


def anchor(store, chunk_id):
    c = store.chunks[chunk_id]
    return SearchHit(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        text=c.text,
        score=0.5,
        title="T",
        section_path=c.section_path,
        ordinal=c.ordinal,
        retriever="rerank",
    )


def test_a_hit_gains_its_previous_and_next_chunks_in_order():
    store = InMemoryStore()
    seed(store, "pto", ["zero", "one", "two", "three", "four"])
    [out] = Expander(store, radius=1).expand([anchor(store, "pto#2")], EVERYONE)
    assert out.text == "one\n\ntwo\n\nthree"
    assert out.chunk_id == "pto#2"  # the anchor stays the citation target
    assert out.neighbor_ids == ["pto#1", "pto#3"]
    assert out.score == 0.5 and out.retriever == "rerank"


def test_radius_two_and_document_edges():
    store = InMemoryStore()
    seed(store, "pto", ["zero", "one", "two"])
    [out] = Expander(store, radius=2).expand([anchor(store, "pto#0")], EVERYONE)
    assert out.text == "zero\n\none\n\ntwo"
    assert out.neighbor_ids == ["pto#1", "pto#2"]


def test_radius_zero_is_a_no_op():
    store = InMemoryStore()
    seed(store, "pto", ["zero", "one", "two"])
    hits = [anchor(store, "pto#1")]
    assert Expander(store, radius=0).expand(hits, EVERYONE) == hits


def test_same_section_keeps_expansion_inside_the_hits_heading():
    store = InMemoryStore()
    seed(
        store,
        "pto",
        ["intro", "carryover a", "carryover b", "parental"],
        sections=[["PTO"], ["PTO", "Carryover"], ["PTO", "Carryover"], ["PTO", "Parental"]],
    )
    [out] = Expander(store, radius=1, same_section=True).expand([anchor(store, "pto#2")], EVERYONE)
    assert out.text == "carryover a\n\ncarryover b"
    [loose] = Expander(store, radius=1, same_section=False).expand(
        [anchor(store, "pto#2")], EVERYONE
    )
    assert loose.text == "carryover a\n\ncarryover b\n\nparental"


def test_same_section_allows_parent_and_child_headings_but_not_siblings():
    # The chunker merges small sections, so a chunk's path is where it *starts*: a parent's
    # path is a prefix of its children's. Lineage counts as the same section; siblings do not.
    store = InMemoryStore()
    seed(
        store,
        "pto",
        ["intro", "carryover", "carryover detail", "parental"],
        sections=[
            ["PTO"],
            ["PTO", "Carryover"],
            ["PTO", "Carryover", "Limits"],
            ["PTO", "Parental"],
        ],
    )
    [out] = Expander(store, radius=2).expand([anchor(store, "pto#1")], EVERYONE)
    assert out.text == "intro\n\ncarryover\n\ncarryover detail"
    [from_parent] = Expander(store, radius=1).expand([anchor(store, "pto#0")], EVERYONE)
    assert from_parent.neighbor_ids == ["pto#1"]


def test_neighbours_the_principal_cannot_see_are_never_added():
    store = InMemoryStore()
    seed(
        store,
        "mixed",
        ["visible before", "anchor", "hr only"],
        chunk_acls=[None, None, ["group:hr"]],
    )
    [out] = Expander(store, radius=1).expand([anchor(store, "mixed#1")], EVERYONE)
    assert "hr only" not in out.text
    assert out.neighbor_ids == ["mixed#0"]


def test_hits_already_inside_a_higher_ranked_passage_are_dropped():
    store = InMemoryStore()
    seed(store, "pto", ["zero", "one", "two", "three", "four", "five"])
    hits = [anchor(store, "pto#2"), anchor(store, "pto#3"), anchor(store, "pto#5")]
    out = Expander(store, radius=1).expand(hits, EVERYONE)
    # pto#3 is inside pto#2's window; pto#5 keeps only the chunks nobody showed yet.
    assert [h.chunk_id for h in out] == ["pto#2", "pto#5"]
    assert out[0].text == "one\n\ntwo\n\nthree"
    assert out[1].text == "four\n\nfive"
    assert out[1].neighbor_ids == ["pto#4"]


def test_hits_from_different_documents_expand_independently():
    store = InMemoryStore()
    seed(store, "a", ["a0", "a1", "a2"])
    seed(store, "b", ["b0", "b1", "b2"])
    out = Expander(store, radius=1).expand([anchor(store, "b#1"), anchor(store, "a#1")], EVERYONE)
    assert [h.text for h in out] == ["b0\n\nb1\n\nb2", "a0\n\na1\n\na2"]


def test_max_chars_bounds_the_passage_but_never_cuts_the_anchor():
    store = InMemoryStore()
    seed(store, "pto", ["x" * 30, "anchor " * 10, "y" * 30])
    a = anchor(store, "pto#1")
    [tight] = Expander(store, radius=1, max_chars=len(a.text) + 5).expand([a], EVERYONE)
    assert tight.text == a.text and tight.neighbor_ids == []
    [one_side] = Expander(store, radius=1, max_chars=len(a.text) + 35).expand([a], EVERYONE)
    assert one_side.neighbor_ids == ["pto#0"]  # the nearer-first order prefers the previous one


def test_hits_without_an_ordinal_pass_through():
    store = InMemoryStore()
    seed(store, "pto", ["zero", "one"])
    bare = anchor(store, "pto#1").model_copy(update={"ordinal": None})
    assert Expander(store, radius=1).expand([bare], EVERYONE) == [bare]


def test_a_store_without_neighbor_lookup_disables_expansion():
    class DenseOnly:
        pass

    hit = SearchHit(chunk_id="d#0", doc_id="d", text="t", score=1.0, ordinal=0)
    assert Expander(DenseOnly(), radius=1).expand([hit], EVERYONE) == [hit]


def test_no_principals_means_no_neighbours():
    store = InMemoryStore()
    seed(store, "pto", ["zero", "one", "two"])
    hit = anchor(store, "pto#1")
    [out] = Expander(store, radius=1).expand([hit], [])
    assert out.text == "one" and out.neighbor_ids == []
