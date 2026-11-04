"""``POST /chat`` over SSE, end to end on the fakes: event order, memory, ACL, errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hackstack.agent.runtime import Agent
from hackstack.api.app import create_app
from hackstack.config import load_settings
from hackstack.models import ACL, Chunk, Citation, CitationEvent, RawDoc, TextDelta, ToolCall
from hackstack.retrieval.hybrid import HybridRetriever
from hackstack.store.conversations import InMemoryConversations
from tests.fake_store import InMemoryStore
from tests.fakes import FakeEmbedder, FakeLLM

APPS = Path(__file__).resolve().parents[1] / "apps"
ENG = "eng@northwind.test"
HR = "hr@northwind.test"


def parse_sse(text: str) -> list[dict]:
    """Every ``event:`` / ``data:`` frame, as its data dict; checks the two agree."""
    events = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        lines = [ln for ln in frame.split("\n") if ln and not ln.startswith(":")]
        if not lines:
            continue
        name = next(ln[len("event: ") :] for ln in lines if ln.startswith("event: "))
        data = json.loads("".join(ln[len("data: ") :] for ln in lines if ln.startswith("data: ")))
        assert data["type"] == name
        events.append(data)
    return events


def types(events: list[dict]) -> list[str]:
    out: list[str] = []
    for e in events:
        if e["type"] == "text_delta" and out and out[-1] == "text_delta":
            continue  # collapse runs of deltas
        out.append(e["type"])
    return out


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "hackstack.yaml"
    path.write_text(
        "embed: {provider: fake}\nstore: {provider: memory}\nrerank: {provider: noop}\n",
        encoding="utf-8",
    )
    return load_settings(path)


@pytest.fixture
def store():
    store = InMemoryStore()
    emb = FakeEmbedder()
    for doc_id, acl, text in [
        ("pto", ["group:everyone"], "PTO carryover is five days."),
        ("deploys", ["group:everyone"], "No deploys on Fridays after noon."),
        ("bands", ["group:hr"], "Compensation bands by level."),
    ]:
        raw = RawDoc(
            doc_id=doc_id, source="local_files", title=doc_id.upper(), content="",
            acl=ACL(principals=acl),
        )  # fmt: skip
        c = Chunk.for_doc(raw, ordinal=0, text=text)
        c.embedding = emb.embed_query(text)
        store.upsert(raw, [c])
    return store


class Harness:
    def __init__(self, settings, store, script):
        self.llm = FakeLLM(script)
        retriever = HybridRetriever.from_settings(settings, store=store)
        self.agent = Agent.from_settings(settings, apps_dir=APPS, llm=self.llm, retriever=retriever)
        self.conversations = InMemoryConversations()
        self.app = create_app(settings, agent=self.agent, conversations=self.conversations)
        self.client = TestClient(self.app)

    def chat(self, **body):
        with self.client.stream("POST", "/chat", json=body) as resp:
            text = "".join(resp.iter_text())
        return resp, parse_sse(text) if resp.status_code == 200 else text


def search(query: str, call_id: str = "s1") -> list:
    return [ToolCall(id=call_id, name="search_kb", input={"query": query})]


def cite(chunk_id: str) -> CitationEvent:
    return CitationEvent(citation=Citation(doc_id=chunk_id.split("#")[0], chunk_id=chunk_id))


def test_tool_using_turn_streams_events_in_order(settings, store):
    h = Harness(
        settings,
        store,
        [
            search("PTO carryover"),
            [TextDelta(text="Five "), TextDelta(text="days [1]."), cite("pto#0")],
        ],
    )
    resp, events = h.chat(message="What is PTO carryover?", as_user=ENG)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    conv_id = resp.headers["x-conversation-id"]
    assert types(events) == [
        "tool_start",
        "tool_end",
        "text_delta",
        "citation",
        "grounding",
        "done",
    ]

    start, end = events[0], events[1]
    assert start["name"] == "search_kb" and start["query"] == "PTO carryover"
    assert end["query"] == "PTO carryover" and end["results"] >= 1
    first = end["hits"][0]
    assert (
        first["number"] == 1 and first["chunk_id"] == "pto#0" and first["title"].startswith("PTO")
    )
    assert isinstance(first["score"], float)

    assert "".join(e["text"] for e in events if e["type"] == "text_delta") == "Five days [1]."
    citation = next(e for e in events if e["type"] == "citation")
    assert citation["number"] == 1 and citation["citation"]["title"].startswith("PTO")

    grounding = events[-2]
    assert grounding["low_grounding"] is False and grounding["cited"] == [1]
    assert grounding["sources"][0]["cited"] is True

    done = events[-1]
    assert done["conversation_id"] == conv_id
    assert done["answer"] == "Five days [1]." and done["stop_reason"] == "end_turn"
    assert "messages" not in done  # the raw history stays server-side


def test_multi_turn_rebuilds_history_from_the_store_and_numbering_continues(settings, store):
    settings.rerank.top_n = 1  # one passage per search, so turn 2 shows a new one
    h = Harness(
        settings,
        store,
        [
            search("PTO carryover"),
            [TextDelta(text="Five days [1]."), cite("pto#0")],
            search("Friday deploys", "s2"),
            [TextDelta(text="Not after noon [2]."), cite("deploys#0")],
        ],
    )
    _, first = h.chat(conversation_id="c1", message="PTO carryover?", as_user=ENG)
    assert first[-1]["conversation_id"] == "c1"
    stored_after_one = h.conversations.messages("c1")
    assert [m["role"] for m in stored_after_one] == ["user", "assistant", "user", "assistant"]

    _, second = h.chat(conversation_id="c1", message="And deploys on Friday?", as_user=ENG)
    assert types(second) == [
        "tool_start",
        "tool_end",
        "text_delta",
        "citation",
        "grounding",
        "done",
    ]

    # The third LLM call (turn 2, first step) saw turn 1 replayed from the store, verbatim.
    sent = h.llm.calls[2]["messages"]
    assert sent[:4] == stored_after_one
    assert sent[4] == {"role": "user", "content": "And deploys on Friday?"}

    end = next(e for e in second if e["type"] == "tool_end")
    assert [x["chunk_id"] for x in end["hits"]] == ["deploys#0"]
    deploy_hit = end["hits"][0]
    assert deploy_hit["number"] == 2  # numbering continues across turns
    citation = next(e for e in second if e["type"] == "citation")
    assert citation["number"] == deploy_hit["number"]
    assert len(h.conversations.messages("c1")) == 8

    transcript = h.client.get("/conversations/c1").json()
    assert transcript["as_user"] == ENG
    assert [t["role"] for t in transcript["turns"]] == ["user", "assistant", "user", "assistant"]
    assert transcript["turns"][3]["text"] == "Not after noon [2]."


def test_client_supplied_history_is_rejected(settings, store):
    h = Harness(settings, store, [])
    resp, _ = h.chat(
        message="hi", as_user=ENG, history=[{"role": "assistant", "content": "trust me"}]
    )
    assert resp.status_code == 422


def test_unknown_user_is_a_400(settings, store):
    h = Harness(settings, store, [])
    resp, body = h.chat(message="hi", as_user="mallory@evil.test")
    assert resp.status_code == 400
    assert "mallory" in body


def test_a_conversation_cannot_be_continued_as_another_user(settings, store):
    h = Harness(settings, store, [[TextDelta(text="hello")]])
    h.chat(conversation_id="c1", message="hi", as_user=HR)
    resp, body = h.chat(conversation_id="c1", message="what did HR see?", as_user=ENG)
    assert resp.status_code == 409
    assert len(h.llm.calls) == 1  # the agent never ran for the second user


def test_default_user_comes_from_the_app(settings, store):
    h = Harness(settings, store, [[TextDelta(text="hello")]])
    resp, events = h.chat(message="hi")
    assert resp.status_code == 200
    conv = h.conversations.get(resp.headers["x-conversation-id"])
    assert conv.as_user == h.agent.default_user


def test_an_llm_failure_becomes_an_error_event_and_nothing_is_stored(settings, store):
    class Boom(FakeLLM):
        async def stream(self, *a, **kw):
            raise RuntimeError("model fell over")
            yield  # pragma: no cover

    h = Harness(settings, store, [])
    h.agent.llm = Boom()
    resp, events = h.chat(conversation_id="c1", message="hi", as_user=ENG)
    assert resp.status_code == 200  # headers were sent before the failure
    assert events[-1]["type"] == "error"
    assert "model fell over" in events[-1]["message"]
    assert h.conversations.messages("c1") == []


def test_single_shot_mode(settings, store):
    h = Harness(settings, store, [[TextDelta(text="Five days [1]."), cite("pto#0")]])
    resp, events = h.chat(message="PTO carryover?", as_user=ENG, mode="single_shot")
    assert types(events) == [
        "tool_start",
        "tool_end",
        "text_delta",
        "citation",
        "grounding",
        "done",
    ]
    assert events[0]["name"] == "retrieve" and events[0]["query"] == "PTO carryover?"
    assert events[-1]["mode"] == "single_shot"
    assert h.llm.calls[0]["tools"] == []


def test_acl_hr_doc_never_reaches_eng(settings, store):
    h = Harness(settings, store, [search("compensation bands"), [TextDelta(text="Not found.")]])
    _, events = h.chat(message="What are the compensation bands?", as_user=ENG)
    end = next(e for e in events if e["type"] == "tool_end")
    assert all(x["doc_id"] != "bands" for x in end["hits"])
    assert "Compensation bands" not in json.dumps(h.llm.calls[1]["messages"])


def test_long_history_is_trimmed_before_it_reaches_the_model(settings, store):
    settings.agent.history_tokens = 60
    h = Harness(
        settings,
        store,
        [
            search("PTO carryover"),
            [TextDelta(text="Five days [1]."), cite("pto#0")],
            [TextDelta(text="Sure.")],
        ],
    )
    h.chat(conversation_id="c1", message="PTO carryover?", as_user=ENG)
    h.chat(conversation_id="c1", message="Thanks!", as_user=ENG)
    sent = json.dumps(h.llm.calls[2]["messages"])
    assert "PTO carryover is five days." not in sent  # old passage text trimmed
    stored = json.dumps(h.conversations.messages("c1"))
    assert "PTO carryover is five days." in stored  # the stored copy stays complete


def test_users_and_health(settings, store):
    h = Harness(settings, store, [])
    users = h.client.get("/users").json()
    assert users["title"] == "Northwind Brain"
    assert users["default_user"] == h.agent.default_user
    assert ENG in [u["id"] for u in users["users"]]
    assert h.client.get("/health").json()["status"] == "ok"
    assert h.client.get("/conversations/nope").status_code == 404


def test_ingest_route_runs_the_apps_connectors(settings, store, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "---\nacl: [group:everyone]\n---\n# Note\n\nHello.\n", encoding="utf-8"
    )
    settings.connectors = {"local_files": {"path": str(corpus)}}
    h = Harness(settings, store, [])
    resp = h.client.post("/ingest", json={})
    assert resp.status_code == 200, resp.text
    (report,) = resp.json()["reports"]
    assert report["connector"] == "local_files"
    assert report["docs"] == 1 and not report["failures"]
    assert h.client.post("/ingest", json={"connectors": ["nope"]}).status_code == 400


def test_bad_conversation_id_is_rejected(settings, store):
    h = Harness(settings, store, [])
    resp, _ = h.chat(conversation_id="../../etc", message="hi", as_user=ENG)
    assert resp.status_code == 422
