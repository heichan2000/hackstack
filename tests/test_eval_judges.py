"""Binary LLM judges: prompts, verdict parsing and the passages they are shown (fake LLM)."""

from __future__ import annotations

import asyncio

from evals.judges import (
    Verdict,
    format_passages,
    judge_faithfulness,
    judge_refusal,
    judge_relevance,
    parse_verdict,
    passages_from_messages,
)
from hackstack.models import Stop, TextDelta
from tests.fakes import FakeLLM


def _says(text: str) -> FakeLLM:
    return FakeLLM([[TextDelta(text=text), Stop(reason="end_turn")]])


def _run(coro):
    return asyncio.run(coro)


# --- parsing --------------------------------------------------------------------------------------


def test_parse_verdict_reads_json_with_reason():
    v = parse_verdict('{"reason": "all claims are in [1]", "verdict": "PASS"}')
    assert v == Verdict(passed=True, reason="all claims are in [1]", raw=v.raw)


def test_parse_verdict_takes_the_faithfulness_judges_unsupported_claim_as_reason():
    v = parse_verdict('{"unsupported": "receipts within 14 days", "verdict": "FAIL"}')
    assert v.passed is False and v.reason == "receipts within 14 days"


def test_parse_verdict_tolerates_prose_case_and_fences():
    assert parse_verdict('Sure!\n```json\n{"verdict": "fail", "reason": "x"}\n```').passed is False
    assert parse_verdict("The answer is grounded. Verdict: PASS").passed is True


def test_parse_verdict_last_word_wins_in_prose():
    # "could PASS ... but FAIL": a reasoning model talks itself round; the last verdict counts.
    assert (
        parse_verdict("It might PASS at first sight, but the date is wrong: FAIL").passed is False
    )


def test_parse_verdict_unparseable_is_none_not_a_guess():
    v = parse_verdict("I am not sure.")
    assert v.passed is None
    assert v.reason == "unparseable judge reply"
    assert parse_verdict('{"verdict": "MAYBE"}').passed is None


# --- passages -------------------------------------------------------------------------------------


def _search_result(chunk_id: str, title: str, text: str) -> dict:
    return {
        "type": "search_result",
        "source": f"kb://{chunk_id}",
        "title": title,
        "content": [{"type": "text", "text": text}],
        "citations": {"enabled": True},
    }


MESSAGES = [
    {"role": "user", "content": "How many PTO days carry over?"},
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "search_kb",
                "input": {"query": "pto carryover"},
            }
        ],
    },  # fmt: skip
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [
                    _search_result("d1#0", "PTO policy (updated 2026-03-01)", "Up to 5 days."),
                    _search_result("d2#0", "Carryover change (updated 2026-10-29)", "Now 10 days."),
                ],
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t2", "name": "whats_changed", "input": {"since": "monday"}}
        ],
    },  # fmt: skip
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t2",
                "content": "3 documents changed since 2026-11-02.",
            }
        ],
    },  # fmt: skip
]


def test_passages_come_from_tool_results_numbered_like_the_conversation():
    ps = passages_from_messages(MESSAGES)
    assert [(p.number, p.title, p.text) for p in ps[:2]] == [
        (1, "PTO policy (updated 2026-03-01)", "Up to 5 days."),
        (2, "Carryover change (updated 2026-10-29)", "Now 10 days."),
    ]
    assert ps[2].number is None and "3 documents changed" in ps[2].text  # plain-text tool output


def test_single_shot_passages_sit_in_the_user_turn():
    messages = [{"role": "user", "content": [_search_result("d1#0", "PTO", "Up to 5 days."),
                                             {"type": "text", "text": "q?"}]}]  # fmt: skip
    assert [(p.number, p.text) for p in passages_from_messages(messages)] == [(1, "Up to 5 days.")]


def test_format_passages_puts_cited_first_and_caps_length():
    ps = passages_from_messages(MESSAGES)
    text = format_passages(ps, cited={2})
    assert text.index("[2]") < text.index("[1]")
    capped = format_passages(ps, cited=set(), max_chars=40)
    assert "[1]" in capped and "[2]" not in capped and "omitted" in capped
    assert format_passages([], cited=set()) == "(no passages: the assistant searched nothing)"


# --- judges ---------------------------------------------------------------------------------------


def test_faithfulness_prompt_has_question_passages_and_answer():
    llm = _says('{"reason": "supported by [2]", "verdict": "PASS"}')
    v = _run(
        judge_faithfulness(
            llm, "How many days carry over?", passages_from_messages(MESSAGES),
            "Ten days now [2].", cited={2},
        )
    )  # fmt: skip
    assert v.passed is True
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "How many days carry over?" in prompt
    assert "Now 10 days." in prompt and "Ten days now [2]." in prompt
    assert "PASS" in prompt and "FAIL" in prompt
    assert llm.calls[0]["tools"] == []  # a judge never gets tools
    assert llm.calls[0]["opts"]["temperature"] == 0


def test_relevance_and_refusal_judges_parse_fail():
    rel = _run(judge_relevance(_says('{"verdict": "FAIL", "reason": "evasive"}'), "q", "a"))
    assert rel.passed is False and rel.reason == "evasive"
    llm = _says('{"reason": "it says it found nothing", "verdict": "PASS"}')
    ref = _run(judge_refusal(llm, "What's the wifi password?", "I couldn't find that."))
    assert ref.passed is True
    assert "wifi password" in llm.calls[0]["messages"][0]["content"]


def test_judge_opts_pass_through():
    llm = _says('{"verdict": "PASS"}')
    _run(judge_relevance(llm, "q", "a", opts={"seed": 7}))
    assert llm.calls[0]["opts"]["seed"] == 7
