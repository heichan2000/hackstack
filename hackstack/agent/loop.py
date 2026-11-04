"""The agent loop: a hand-written tool-use loop over any :class:`LLMProvider`.

One turn::

    llm.stream(system, history, tools) -> text deltas, tool calls, citations, one Stop
    append {"role": "assistant", "content": stop.content}   # unchanged (signatures, citations)
    stop.reason == "tool_use" -> run every call at once (asyncio.gather), then append ONE user
                                 message whose content is all the tool_results, in call order

and repeat, at most ``max_iterations`` LLM calls. The last allowed call runs with
``tool_choice: none`` so the model has to answer with what it has. Separate user messages per
result "teach" Claude to stop calling tools in parallel, hence the single message.

Every stop reason is handled:

- ``end_turn`` / ``stop_sequence``: done.
- ``tool_use``: run the tools and loop.
- ``pause_turn``: the turn is appended as-is and resent.
- ``max_tokens``: the answer is marked truncated; a tool call cut off mid-way is dropped, not run.
- ``refusal``: done, with the provider's ``stop.details``.
- ``model_context_window_exceeded``: done, flagged; compacting history is the caller's job.

``single_shot`` mode skips the loop: retrieve once for the question, then one call without tools.
It is the fast path and the eval baseline.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, Literal

from opentelemetry.trace import Span, SpanKind
from pydantic import BaseModel, Field

from hackstack import semconv as sc
from hackstack import tracing
from hackstack.agent.citations import CitationRegistry, SourceRef, is_low_grounding
from hackstack.agent.guard import OutputGuard, guard_text
from hackstack.agent.tools import Passages, Tool, ToolContext, ToolInputError, ToolOutput
from hackstack.llm.base import LLMProvider
from hackstack.models import Citation, CitationEvent, Stop, TextDelta, ToolCall

log = logging.getLogger(__name__)

DONE_REASONS = {"end_turn", "stop_sequence"}


class ToolStart(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolHit(BaseModel):
    """One passage a tool call returned, for the live search panel."""

    number: int  # its citation number in the conversation
    chunk_id: str
    doc_id: str
    title: str
    score: float
    url: str | None = None


class ToolEnd(BaseModel):
    type: Literal["tool_end"] = "tool_end"
    id: str
    name: str
    is_error: bool = False
    results: int = 0  # passages returned (0 for plain-text results)
    seconds: float = 0.0
    summary: str = ""  # the error message, or the first passage titles
    hits: list[ToolHit] = Field(default_factory=list)


class SourceInfo(BaseModel):
    """A passage the model was shown, for the sources panel and evals."""

    number: int
    chunk_id: str
    doc_id: str
    title: str
    url: str | None = None
    cited: bool = False


class AgentResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceInfo] = Field(default_factory=list)
    low_grounding: bool = True
    stop_reason: str = "end_turn"  # a Stop reason, or "max_iterations"
    truncated: bool = False  # max_tokens cut the answer
    refusal: dict[str, Any] | None = None
    iterations: int = 0  # LLM calls made
    tool_calls: list[ToolEnd] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    seconds: float = 0.0
    mode: Literal["agent", "single_shot"] = "agent"
    messages: list[dict[str, Any]] = Field(default_factory=list)  # the history to store


class AgentDone(BaseModel):
    type: Literal["done"] = "done"
    result: AgentResult


AgentEvent = Annotated[
    TextDelta | CitationEvent | ToolStart | ToolEnd | AgentDone, Field(discriminator="type")
]


class _Turn:
    """What one LLM call produced."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.calls: list[ToolCall] = []
        self.stop = Stop()


async def _stream_turn(
    llm: LLMProvider,
    system: str,
    history: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    opts: dict[str, Any],
    registry: CitationRegistry,
    cited: dict[str, Citation],
    turn: _Turn,
    *,
    parent: Span | None = None,
    iteration: int = 1,
) -> AsyncIterator[TextDelta | CitationEvent]:
    s = _llm_span(llm, opts, parent, iteration)
    tracing.content(s, sc.OI_INPUT_VALUE, {"system": system, "messages": history})
    guard = _guard(system, registry)
    # The guard holds text back; a citation waits until the text before it has come out.
    waiting: list[tuple[int, CitationEvent]] = []

    def due() -> list[CitationEvent]:
        ready = [c for at, c in waiting if at <= guard.released]
        waiting[:] = [(at, c) for at, c in waiting if at > guard.released]
        return ready

    try:
        async for event in llm.stream(system, history, tools, **opts):
            if isinstance(event, TextDelta):
                if safe := guard.feed(event.text):
                    turn.text.append(safe)
                    yield event.model_copy(update={"text": safe})
                for c in due():
                    yield c
            elif isinstance(event, ToolCall):
                turn.calls.append(event)
            elif isinstance(event, CitationEvent):
                citation = registry.resolve(event.citation)
                cited.setdefault(citation.chunk_id or citation.doc_id, citation)
                waiting.append((guard.fed, CitationEvent(citation=citation)))
                for c in due():
                    yield c
            elif isinstance(event, Stop):
                turn.stop = event
        if tail := guard.finish():
            turn.text.append(tail)
            yield TextDelta(text=tail)
        for _, c in waiting:
            yield c
        _guard_content(turn, system, registry)
    except (GeneratorExit, asyncio.CancelledError):
        raise
    except BaseException as exc:
        tracing.fail(s, exc)
        raise
    else:
        _record_llm(s, turn)
    finally:
        s.end()


def _guard(system: str, registry: CitationRegistry) -> OutputGuard:
    """The output guard for one LLM call: links may point only at sources shown so far."""
    return OutputGuard(
        system=system,
        allowed_urls=lambda: {r.url for r in registry.refs if r.url},
        passages=lambda: [r.text for r in registry.refs],
    )


def _guard_content(turn: _Turn, system: str, registry: CitationRegistry) -> None:
    """Guard the provider's verbatim content too: it is what gets stored and reloaded."""
    if not turn.stop.content:
        return
    urls = {r.url for r in registry.refs if r.url}
    shown = [r.text for r in registry.refs]
    blocks = []
    for block in turn.stop.content:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text = guard_text(block["text"], system=system, allowed_urls=urls, passages=shown)
            block = {**block, "text": text}
        blocks.append(block)
    turn.stop = turn.stop.model_copy(update={"content": blocks})


def _llm_span(llm: LLMProvider, opts: dict[str, Any], parent: Span | None, iteration: int) -> Span:
    model = opts.get("model") or getattr(llm, "model", None)
    return tracing.start_span(
        f"{sc.SPAN_CHAT} {model}" if model else sc.SPAN_CHAT,
        {
            sc.OPERATION_NAME: sc.OP_CHAT,
            sc.OI_SPAN_KIND: sc.KIND_LLM,
            sc.PROVIDER_NAME: getattr(llm, "provider_name", None),
            sc.REQUEST_MODEL: model,
            sc.HS_ITERATION: iteration,
            **tracing.server_attributes(getattr(llm, "base_url", None)),
        },
        kind=SpanKind.CLIENT,
        parent=parent,
    )


def _record_llm(s: Span, turn: _Turn) -> None:
    usage = turn.stop.usage
    tracing.set_attributes(
        s,
        {
            sc.USAGE_INPUT_TOKENS: usage.get("input_tokens"),
            sc.USAGE_OUTPUT_TOKENS: usage.get("output_tokens"),
            sc.USAGE_CACHE_READ_TOKENS: usage.get("cache_read_input_tokens"),
            sc.USAGE_CACHE_CREATION_TOKENS: usage.get("cache_creation_input_tokens"),
            sc.FINISH_REASONS: (turn.stop.reason,),
        },
    )
    tracing.content(s, sc.OI_OUTPUT_VALUE, "".join(turn.text))


def _content(turn: _Turn) -> list[dict[str, Any]]:
    """The assistant turn to append: the provider's, verbatim; rebuilt only if it sent none."""
    if turn.stop.content:
        return list(turn.stop.content)
    content: list[dict[str, Any]] = []
    if turn.text:
        content.append({"type": "text", "text": "".join(turn.text)})
    content += [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in turn.calls
    ]
    return content


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for k, v in usage.items():
        total[k] = total.get(k, 0) + v


def _error_result(call_id: str, message: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": call_id, "content": message, "is_error": True}


async def _call(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
    if inspect.iscoroutinefunction(tool.handler):
        return await tool.handler(args, ctx)
    return await asyncio.to_thread(tool.handler, args, ctx)  # parallel calls really overlap


async def _run_tool(
    call: ToolCall, tools: dict[str, Tool], ctx: ToolContext
) -> tuple[ToolOutput | None, str | None, float]:
    """``(output, error, seconds)``. Never raises: failures become error results."""
    attrs = {
        sc.OPERATION_NAME: sc.OP_EXECUTE_TOOL,
        sc.OI_SPAN_KIND: sc.KIND_TOOL,
        sc.TOOL_NAME: call.name,
        sc.TOOL_CALL_ID: call.id,
        sc.TOOL_TYPE: "function",
    }
    with tracing.span(f"{sc.SPAN_EXECUTE_TOOL} {call.name}", attrs) as s:
        tracing.content(s, sc.TOOL_CALL_ARGUMENTS, json.dumps(call.input, ensure_ascii=False))
        out, error, seconds = await _run_tool_untraced(call, tools, ctx)
        if error is not None:
            tracing.fail(s, error_type="tool_error")
            tracing.content(s, sc.TOOL_CALL_RESULT, error)
        elif isinstance(out, Passages):
            s.set_attribute(sc.HS_TOOL_RESULTS, len(out.hits))
        return out, error, seconds


async def _run_tool_untraced(
    call: ToolCall, tools: dict[str, Tool], ctx: ToolContext
) -> tuple[ToolOutput | None, str | None, float]:
    t0 = time.perf_counter()
    tool = tools.get(call.name)
    if tool is None:
        known = ", ".join(sorted(tools)) or "none"
        return None, f"Unknown tool {call.name!r}. Available tools: {known}.", 0.0
    if "_invalid_arguments" in call.input:
        raw = str(call.input["_invalid_arguments"])[:200]
        return (
            None,
            (
                f"The arguments were not valid JSON ({raw!r}). Call {call.name} again with a JSON "
                "object that matches its schema."
            ),
            0.0,
        )
    try:
        out = await _call(tool, call.input, ctx)
    except ToolInputError as exc:
        return None, str(exc), time.perf_counter() - t0
    except Exception as exc:  # a tool bug must not kill the conversation
        log.exception("tool %s failed", call.name)
        return None, f"{call.name} failed: {type(exc).__name__}: {exc}", time.perf_counter() - t0
    return out, None, time.perf_counter() - t0


def _to_result_content(
    out: ToolOutput, registry: CitationRegistry, shown: list[ToolHit] | None = None
) -> tuple[str | list[dict[str, Any]], int, str]:
    """Tool output as ``tool_result`` content, plus (passage count, summary).

    Passages handed to the model are appended to ``shown`` with their citation numbers.
    """
    if isinstance(out, Passages):
        hits = [h for h in out.hits if not (out.skip_seen and registry.seen(h.chunk_id))]
        lead = [{"type": "text", "text": out.preamble}] if out.preamble else []
        if not hits:
            empty = out.empty
            if out.hits:  # everything matched was shown before
                empty = "No new results: every match was already shown above."
            if lead:
                return lead, 0, out.preamble[:120]
            return [{"type": "text", "text": empty}], 0, empty
        blocks = []
        for h in hits:
            blocks.append(registry.add(h))
            if shown is not None:
                ref = registry.refs[-1]
                shown.append(
                    ToolHit(
                        number=ref.number,
                        chunk_id=h.chunk_id,
                        doc_id=h.doc_id,
                        title=ref.title,
                        score=round(float(h.score), 4),
                        url=h.url,
                    )
                )
        return [*lead, *blocks], len(blocks), "; ".join(b["title"] for b in blocks[:3])
    if isinstance(out, str):
        return out, 0, out[:120]
    return list(out), 0, ""


async def _execute(
    calls: list[ToolCall], tools: dict[str, Tool], ctx: ToolContext, parent: Span | None = None
) -> tuple[list[dict[str, Any]], list[ToolEnd]]:
    if parent is None:
        outcomes = await asyncio.gather(*(_run_tool(c, tools, ctx) for c in calls))
    else:
        with tracing.use(parent):  # each gathered task copies this context: tool spans nest
            outcomes = await asyncio.gather(*(_run_tool(c, tools, ctx) for c in calls))
    results: list[dict[str, Any]] = []
    ends: list[ToolEnd] = []
    for call, (out, error, seconds) in zip(calls, outcomes, strict=True):
        if error is not None or out is None:
            message = error or "The tool returned nothing."
            results.append(_error_result(call.id, message))
            ends.append(
                ToolEnd(id=call.id, name=call.name, is_error=True, seconds=seconds, summary=message)
            )
            continue
        # Registered in call order, so numbering matches the history the provider sees.
        shown: list[ToolHit] = []
        content, n, summary = _to_result_content(out, ctx.citations, shown)
        results.append({"type": "tool_result", "tool_use_id": call.id, "content": content})
        ends.append(
            ToolEnd(
                id=call.id,
                name=call.name,
                results=n,
                seconds=round(seconds, 3),
                summary=summary,
                hits=shown,
            )
        )
    return results, ends


def _sources(registry: CitationRegistry, cited: dict[str, Citation]) -> list[SourceInfo]:
    def info(ref: SourceRef) -> SourceInfo:
        return SourceInfo(
            number=ref.number,
            chunk_id=ref.chunk_id,
            doc_id=ref.doc_id,
            title=ref.title,
            url=ref.url,
            cited=ref.chunk_id in cited,
        )

    return [info(r) for r in registry.refs]


def _agent_span(
    mode: str, messages: list[dict[str, Any]], ctx: ToolContext, attrs: dict[str, Any] | None
) -> Span:
    """The request's root span: one trace per agent turn."""
    attrs = dict(attrs or {})
    name = attrs.get(sc.AGENT_NAME)
    s = tracing.start_span(
        f"{sc.SPAN_INVOKE_AGENT} {name}" if name else sc.SPAN_INVOKE_AGENT,
        {
            sc.OPERATION_NAME: sc.OP_INVOKE_AGENT,
            sc.OI_SPAN_KIND: sc.KIND_AGENT,
            sc.HS_MODE: mode,
            sc.HS_PRINCIPALS: len(ctx.principals),
            **attrs,
        },
    )
    if messages:
        tracing.content(s, sc.OI_INPUT_VALUE, _question(messages[-1]))
    return s


def _record_result(s: Span, result: AgentResult) -> None:
    usage = result.usage
    tracing.set_attributes(
        s,
        {
            sc.HS_ITERATIONS: result.iterations,
            sc.HS_STOP_REASON: result.stop_reason,
            sc.HS_LOW_GROUNDING: result.low_grounding,
            sc.HS_CITATIONS: len(result.citations),
            sc.USAGE_INPUT_TOKENS: usage.get("input_tokens"),
            sc.USAGE_OUTPUT_TOKENS: usage.get("output_tokens"),
        },
    )
    tracing.content(s, sc.OI_OUTPUT_VALUE, result.answer)


async def _traced(
    events: AsyncIterator[Any], root: Span
) -> AsyncIterator[TextDelta | CitationEvent | ToolStart | ToolEnd | AgentDone]:
    """Pass ``events`` through, ending ``root`` however the stream ends (even client gone)."""
    try:
        async for event in events:
            if isinstance(event, AgentDone):
                _record_result(root, event.result)
            yield event
    except (GeneratorExit, asyncio.CancelledError):
        raise
    except BaseException as exc:
        tracing.fail(root, exc)
        raise
    finally:
        await events.aclose()  # type: ignore[attr-defined]
        root.end()


async def stream_agent(
    llm: LLMProvider,
    *,
    system: str,
    messages: list[dict[str, Any]],
    tools: Sequence[Tool],
    ctx: ToolContext,
    max_iterations: int = 6,
    llm_opts: dict[str, Any] | None = None,
    trace_attrs: dict[str, Any] | None = None,
) -> AsyncIterator[TextDelta | CitationEvent | ToolStart | ToolEnd | AgentDone]:
    """Run the tool loop on ``messages`` (ending with the user's turn), streaming events.

    The last event is always :class:`AgentDone`, whose ``result.messages`` is the full history
    to store for the next turn. ``ctx.citations`` should be built from ``messages`` with
    :meth:`CitationRegistry.from_messages` so numbering continues across turns.
    ``trace_attrs`` go on the turn's root span (``gen_ai.agent.name``, ``gen_ai.conversation.id``).
    """
    root = _agent_span("agent", messages, ctx, trace_attrs)
    events = _agent_events(
        llm,
        system=system,
        messages=messages,
        tools=tools,
        ctx=ctx,
        max_iterations=max_iterations,
        llm_opts=llm_opts,
        root=root,
    )
    async for event in _traced(events, root):
        yield event


async def _agent_events(
    llm: LLMProvider,
    *,
    system: str,
    messages: list[dict[str, Any]],
    tools: Sequence[Tool],
    ctx: ToolContext,
    max_iterations: int,
    llm_opts: dict[str, Any] | None,
    root: Span,
) -> AsyncIterator[TextDelta | CitationEvent | ToolStart | ToolEnd | AgentDone]:
    if max_iterations < 1:
        raise ValueError(f"agent.max_iterations must be >= 1, got {max_iterations}")
    t0 = time.perf_counter()
    history = list(messages)
    by_name = {t.name: t for t in tools}
    defs = [t.definition() for t in tools]
    cited: dict[str, Citation] = {}
    usage: dict[str, int] = {}
    tool_log: list[ToolEnd] = []
    answer, reason, refusal, truncated = "", "max_iterations", None, False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        opts = dict(llm_opts or {})
        if iteration == max_iterations and defs:
            opts["tool_choice"] = {"type": "none"}  # out of budget: answer now
        turn = _Turn()
        async for event in _stream_turn(
            llm,
            system,
            history,
            defs,
            opts,
            ctx.citations,
            cited,
            turn,
            parent=root,
            iteration=iteration,
        ):
            yield event
        _add_usage(usage, turn.stop.usage)
        content = _content(turn)
        reason = turn.stop.reason
        if reason == "max_tokens" and turn.calls:
            # A call cut off mid-arguments must not run, and must not stay unanswered in history.
            content = [b for b in content if b.get("type") != "tool_use"]
            turn.calls = []
        if content:
            history.append({"role": "assistant", "content": content})
        answer = "".join(turn.text)

        if reason == "tool_use" and turn.calls:
            if iteration == max_iterations:
                # The model ignored tool_choice none. Keep the history valid, don't run them.
                history.append(
                    {
                        "role": "user",
                        "content": [
                            _error_result(c.id, "Iteration limit reached; the tool was not run.")
                            for c in turn.calls
                        ],
                    }
                )
                reason = "max_iterations"
                break
            for c in turn.calls:
                yield ToolStart(id=c.id, name=c.name, input=c.input)
            results, ends = await _execute(turn.calls, by_name, ctx, parent=root)
            history.append({"role": "user", "content": results})  # ONE message, results first
            tool_log += ends
            for end in ends:
                yield end
            reason = "max_iterations"
            continue
        if reason == "pause_turn":
            reason = "max_iterations"
            continue  # appended as-is; resend
        truncated = reason == "max_tokens"
        refusal = turn.stop.details if reason == "refusal" else None
        break

    citations = list(cited.values())
    yield AgentDone(
        result=AgentResult(
            answer=answer,
            citations=citations,
            sources=_sources(ctx.citations, cited),
            low_grounding=is_low_grounding(citations),
            stop_reason=reason,
            truncated=truncated,
            refusal=refusal,
            iterations=iterations,
            tool_calls=tool_log,
            usage=usage,
            seconds=round(time.perf_counter() - t0, 3),
            mode="agent",
            messages=history,
        )
    )


def _question(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if b.get("type") == "text")


async def stream_single_shot(
    llm: LLMProvider,
    *,
    system: str,
    messages: list[dict[str, Any]],
    ctx: ToolContext,
    llm_opts: dict[str, Any] | None = None,
    trace_attrs: dict[str, Any] | None = None,
) -> AsyncIterator[TextDelta | CitationEvent | ToolStart | ToolEnd | AgentDone]:
    """Retrieve once for the last user message, then answer in one call without tools."""
    root = _agent_span("single_shot", messages, ctx, trace_attrs)
    events = _single_shot_events(
        llm, system=system, messages=messages, ctx=ctx, llm_opts=llm_opts, root=root
    )
    async for event in _traced(events, root):
        yield event


async def _single_shot_events(
    llm: LLMProvider,
    *,
    system: str,
    messages: list[dict[str, Any]],
    ctx: ToolContext,
    llm_opts: dict[str, Any] | None,
    root: Span,
) -> AsyncIterator[TextDelta | CitationEvent | ToolStart | ToolEnd | AgentDone]:
    t0 = time.perf_counter()
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("single_shot needs the conversation to end with a user message")
    question = _question(messages[-1])
    yield ToolStart(id="retrieve", name="retrieve", input={"query": question})
    r0 = time.perf_counter()
    with tracing.use(root):  # to_thread copies the context: retrieval spans nest under root
        hits = await asyncio.to_thread(
            ctx.retriever.search, question, ctx.principals, None, top_n=ctx.top_n
        )
    shown: list[ToolHit] = []
    content, n, summary = _to_result_content(Passages(hits=list(hits)), ctx.citations, shown)
    end = ToolEnd(
        id="retrieve",
        name="retrieve",
        results=n,
        seconds=round(time.perf_counter() - r0, 3),
        summary=summary,
        hits=shown,
    )
    yield end
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    history = [
        *messages[:-1],
        {"role": "user", "content": [*blocks, {"type": "text", "text": question}]},
    ]

    cited: dict[str, Citation] = {}
    turn = _Turn()
    async for event in _stream_turn(
        llm, system, history, [], dict(llm_opts or {}), ctx.citations, cited, turn, parent=root
    ):
        yield event
    content_out = _content(turn)
    content_out = [b for b in content_out if b.get("type") != "tool_use"]
    if content_out:
        history.append({"role": "assistant", "content": content_out})
    citations = list(cited.values())
    reason = turn.stop.reason
    yield AgentDone(
        result=AgentResult(
            answer="".join(turn.text),
            citations=citations,
            sources=_sources(ctx.citations, cited),
            low_grounding=is_low_grounding(citations),
            stop_reason=reason,
            truncated=reason == "max_tokens",
            refusal=turn.stop.details if reason == "refusal" else None,
            iterations=1,
            tool_calls=[end],
            usage=dict(turn.stop.usage),
            seconds=round(time.perf_counter() - t0, 3),
            mode="single_shot",
            messages=history,
        )
    )


async def collect(events: AsyncIterator[Any]) -> AgentResult:
    """Drain an event stream and return its :class:`AgentResult`."""
    result = None
    async for event in events:
        if isinstance(event, AgentDone):
            result = event.result
    if result is None:  # pragma: no cover - both streams always end with AgentDone
        raise RuntimeError("agent stream ended without a result")
    return result
