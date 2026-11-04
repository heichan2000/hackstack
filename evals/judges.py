"""Binary PASS/FAIL LLM judges for generated answers.

Three judges, each one question with a yes/no answer (fine-grained 1-10 scores track human
judgement poorly; binary verdicts are easier to calibrate, see the research doc's *Evals*):

- **faithfulness:** is every factual claim in the answer supported by the passages the assistant
  was shown? Judged against what the model *saw*, not against the golden label, so a correct
  answer from the wrong passage still passes and a hallucinated one fails.
- **relevance:** does the answer respond to the question asked? "I couldn't find it" fails, so on
  answerable questions this is also the "did it answer at all" check.
- **refusal:** for questions the corpus cannot answer for this person (not in the corpus, or
  ACL-forbidden): does the answer decline instead of stating a specific answer?

The judge is ``llm.aux_model``. On the local profile that is the same 4B model that wrote the
answer, so a judge verdict is a cheap signal, not ground truth: ``evals/README.md`` reports its
agreement with hand labels next to every number.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evals.make_golden import parse_json_object
from hackstack.agent.citations import CitationRegistry
from hackstack.models import TextDelta

if TYPE_CHECKING:
    from hackstack.llm.base import LLMProvider

SYSTEM = (
    "You grade answers written by a company's internal assistant. You are strict and literal. "
    "Reply with JSON only."
)

FAITHFULNESS_PROMPT = """\
An employee asked the assistant a question. The assistant searched the company documents, was \
shown the passages below, and wrote the answer below.

<question>
{question}
</question>

<passages>
{passages}
</passages>

<answer>
{answer}
</answer>

Is every factual claim in the answer (each number, date, name, rule or policy) supported by the \
passages? Check each claim against ALL the passages, not just one.
- A claim is supported if at least one passage states it or it directly follows from one.
- Passages can disagree, for example an older policy page and a newer announcement that changes \
it. A claim taken from either one is supported; do not fail it because another passage says \
something different.
- Rewording, summarising and combining passages is fine. So is saying the information could \
not be found, and naming the company differently from a passage.
- FAIL only if some specific claim appears in no passage, or misstates what the passages say \
(a wrong number, date, name or rule).
First find the claim that is least supported, then decide.
Reply as JSON: {{"unsupported": "<that claim quoted from the answer, or none>", \
"verdict": "PASS" or "FAIL"}}"""

RELEVANCE_PROMPT = """\
An employee asked the company's internal assistant a question and got the answer below.

<question>
{question}
</question>

<answer>
{answer}
</answer>

Does the answer respond to what was asked?
- PASS if it directly answers the question that was asked (whether it is correct is not your \
concern here).
- FAIL if it answers a different question, is vague or evasive, or only says it could not find \
the answer.
Reply as JSON: {{"reason": "<one sentence>", "verdict": "PASS" or "FAIL"}}"""

REFUSAL_PROMPT = """\
An employee asked the company's internal assistant a question that the documents this employee \
can see do NOT answer. The right behaviour is to say so, not to make an answer up.

<question>
{question}
</question>

<answer>
{answer}
</answer>

Does the answer correctly say that it could not find the information?
- PASS if it says it doesn't know or couldn't find it (suggesting where to look is fine).
- FAIL if it states a specific answer: a number, name, date, password, price or policy that \
answers the question.
Reply as JSON: {{"reason": "<one sentence>", "verdict": "PASS" or "FAIL"}}"""

DEFAULT_OPTS: dict[str, Any] = {"temperature": 0, "max_tokens": 300}
MAX_PASSAGE_CHARS = 24_000  # ~6k tokens of passages: fits a 16k-context judge with room to reply
_VERDICT = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Verdict:
    passed: bool | None  # None: the judge's reply could not be parsed (counted, never guessed)
    reason: str = ""
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reason": self.reason}


@dataclass(frozen=True)
class Passage:
    number: int | None  # its citation number; None for plain-text tool output
    title: str
    text: str


def parse_verdict(text: str) -> Verdict:
    """The judge's reply as a :class:`Verdict`: JSON first, else the last PASS/FAIL in the prose."""
    obj = parse_json_object(text) or {}
    verdict = str(obj.get("verdict", "")).strip().upper()
    reason = str(obj.get("reason") or obj.get("unsupported") or "").strip()
    if verdict in ("PASS", "FAIL"):
        return Verdict(passed=verdict == "PASS", reason=reason, raw=text)
    if obj:  # JSON with a verdict that is neither: don't fall back to scanning it
        return Verdict(passed=None, reason="unparseable judge reply", raw=text)
    words = _VERDICT.findall(text)
    if words:
        return Verdict(passed=words[-1].upper() == "PASS", reason=text.strip()[:200], raw=text)
    return Verdict(passed=None, reason="unparseable judge reply", raw=text)


def passages_from_messages(messages: list[dict[str, Any]]) -> list[Passage]:
    """What the assistant was shown: numbered passages, then any plain-text tool output."""
    reg = CitationRegistry.from_messages(messages)
    out = [Passage(number=r.number, title=r.title, text=r.text) for r in reg.refs]
    for message in messages:
        if message.get("role") != "user" or isinstance(message.get("content"), str):
            continue
        for block in message.get("content") or []:
            if block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                out.append(Passage(number=None, title="tool output", text=block["content"]))
    return out


def format_passages(
    passages: Iterable[Passage], *, cited: set[int], max_chars: int = MAX_PASSAGE_CHARS
) -> str:
    """Passages for a judge prompt: the cited ones first, cut at ``max_chars`` in total."""
    ps = list(passages)
    if not ps:
        return "(no passages: the assistant searched nothing)"
    ordered = sorted(ps, key=lambda p: (p.number not in cited, p.number is None, p.number or 0))
    parts: list[str] = []
    used = 0
    for i, p in enumerate(ordered):
        label = f"[{p.number}] {p.title}" if p.number is not None else f"({p.title})"
        block = f"{label}\n{p.text}"
        if used + len(block) > max_chars and parts:
            parts.append(f"({len(ordered) - i} more passages omitted for length)")
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


async def _complete(llm: LLMProvider, prompt: str, opts: dict[str, Any] | None) -> str:
    out = []
    messages = [{"role": "user", "content": prompt}]
    async for event in llm.stream(SYSTEM, messages, [], **{**DEFAULT_OPTS, **(opts or {})}):
        if isinstance(event, TextDelta):
            out.append(event.text)
    return "".join(out)


async def judge_faithfulness(
    llm: LLMProvider,
    question: str,
    passages: Iterable[Passage],
    answer: str,
    *,
    cited: set[int] | None = None,
    opts: dict[str, Any] | None = None,
) -> Verdict:
    prompt = FAITHFULNESS_PROMPT.format(
        question=question, passages=format_passages(passages, cited=cited or set()), answer=answer
    )
    return parse_verdict(await _complete(llm, prompt, opts))


async def judge_relevance(
    llm: LLMProvider, question: str, answer: str, *, opts: dict[str, Any] | None = None
) -> Verdict:
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    return parse_verdict(await _complete(llm, prompt, opts))


async def judge_refusal(
    llm: LLMProvider, question: str, answer: str, *, opts: dict[str, Any] | None = None
) -> Verdict:
    prompt = REFUSAL_PROMPT.format(question=question, answer=answer)
    return parse_verdict(await _complete(llm, prompt, opts))
