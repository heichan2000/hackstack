from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from hackstack.models import ACL, Chunk, LLMEvent, Principal, RawDoc, TextDelta, ToolCall


def test_principal_expands_to_user_and_group_principals():
    p = Principal(user="finance@northwind.test", groups=["finance", "all"])
    assert p.principals() == ["user:finance@northwind.test", "group:finance", "group:all"]


def test_acl_allows_a_principal_in_one_of_its_groups():
    acl = ACL(principals=["group:finance"])
    assert acl.allows(Principal(user="a@x.test", groups=["finance"]))


def test_acl_denies_a_principal_outside_it():
    acl = ACL(principals=["group:finance"])
    assert not acl.allows(Principal(user="a@x.test", groups=["eng"]))


def test_empty_acl_fails_closed():
    assert not ACL(principals=[]).allows(Principal(user="a@x.test", groups=["all"]))


def test_raw_doc_requires_an_acl():
    with pytest.raises(ValidationError):
        RawDoc(doc_id="d1", source="local_files", title="t", content="x")


def test_chunk_carries_the_documents_acl():
    when = datetime(2026, 11, 2, tzinfo=UTC)
    raw = RawDoc(
        doc_id="d1",
        source="local_files",
        title="PTO policy",
        content="# PTO",
        acl=ACL(principals=["group:all"]),
        updated_at=when,
    )
    chunk = Chunk.for_doc(raw, ordinal=0, text="Carry over 5 days.", section_path=["PTO"])
    assert chunk.acl_principals == ["group:all"]
    assert chunk.doc_id == "d1"
    assert chunk.embed_text == "Carry over 5 days."
    assert chunk.updated_at == when


def test_chunk_ids_are_deterministic_and_distinct_per_ordinal():
    raw = RawDoc(doc_id="d1", source="s", title="t", content="", acl=ACL(principals=["g"]))
    a0 = Chunk.for_doc(raw, ordinal=0, text="a")
    b0 = Chunk.for_doc(raw, ordinal=0, text="a")
    a1 = Chunk.for_doc(raw, ordinal=1, text="a")
    assert a0.chunk_id == b0.chunk_id
    assert a0.chunk_id != a1.chunk_id


def test_llm_events_round_trip_through_json_by_type():
    adapter = TypeAdapter(list[LLMEvent])
    events = [TextDelta(text="Hi"), ToolCall(id="t1", name="search_kb", input={"query": "pto"})]
    assert adapter.validate_json(adapter.dump_json(events)) == events
