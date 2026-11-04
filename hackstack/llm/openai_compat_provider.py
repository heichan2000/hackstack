"""OpenAI-compatible chat completions: Ollama, OpenAI, vLLM, Groq, OpenRouter.

Talks plain HTTP (``POST {base_url}/chat/completions`` with ``stream: true``) and translates
both ways, so the rest of the kit only ever sees the Anthropic Messages shape and
:data:`~hackstack.models.LLMEvent` streams:

- tools: Anthropic ``{name, description, input_schema}`` -> OpenAI function tools;
- history: ``tool_use`` blocks -> ``tool_calls``, ``tool_result`` blocks -> ``tool`` messages,
  thinking blocks dropped;
- citations: there are no native citations, so ``search_result`` blocks are shown numbered and
  the model is asked to cite with ``[n]``; markers in the answer become ``CitationEvent``s.

``tool_choice`` is ``auto`` only, as with current Claude models (and Ollama ignores it anyway).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx2

if TYPE_CHECKING:
    from hackstack.config import LLMSettings, Secrets

from hackstack.llm.base import LLMError
from hackstack.llm.citations import citation_from_source, search_result_text
from hackstack.models import Citation, CitationEvent, LLMEvent, Stop, TextDelta, ToolCall
from hackstack.registry import LLM, register

CITE_INSTRUCTION = "Search results. Cite each claim with its number in square brackets, e.g. [1]."
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1]  [1, 3]

_FINISH = {"length": "max_tokens", "content_filter": "refusal"}
# Messages-API options with no OpenAI equivalent; dropped so one opts dict fits both adapters.
_ANTHROPIC_ONLY = {"cache_control", "thinking", "output_config", "effort", "metadata", "betas"}
_TIMEOUT = httpx2.Timeout(300.0, connect=10.0)  # a cold local model can take a while to load


@register(LLM, "openai_compat")
class OpenAICompatProvider:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort  # Ollama: "none" turns qwen3.5 thinking off
        self._http = http_client

    @classmethod
    def from_settings(
        cls, llm: LLMSettings, secrets: Secrets, *, model: str
    ) -> OpenAICompatProvider:
        if not llm.base_url:
            raise ValueError(
                "llm.provider openai_compat needs llm.base_url, e.g. Ollama's "
                "http://localhost:11434/v1"
            )
        key = secrets.openai_api_key
        return cls(
            model=model,
            base_url=llm.base_url,
            api_key=key.get_secret_value() if key else None,
            reasoning_effort=llm.reasoning_effort,
        )

    async def stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **opts: Any,
    ) -> AsyncIterator[LLMEvent]:
        sources: list[Citation] = []
        body = self._request(system, _to_openai(messages, sources), tools, opts)
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        text: list[str] = []
        scanned = 0  # how far into the answer text markers have been looked for
        calls: dict[int, dict[str, str]] = {}  # tool-call fragments by stream index
        usage: dict[str, int] = {}
        finish: str | None = None
        async with AsyncExitStack() as stack:
            http = self._http or await stack.enter_async_context(
                httpx2.AsyncClient(timeout=_TIMEOUT)
            )
            url = f"{self.base_url}/chat/completions"
            try:
                resp = await stack.enter_async_context(
                    http.stream("POST", url, json=body, headers=headers)
                )
            except httpx2.TransportError as exc:
                raise LLMError(f"cannot reach {url}: {exc}. Is the server running?") from exc
            if resp.status_code >= 400:
                raise LLMError(_error_message(resp.status_code, await resp.aread(), body))
            async for chunk in _sse_json(resp):
                if chunk.get("usage"):
                    usage = {
                        "input_tokens": chunk["usage"].get("prompt_tokens", 0),
                        "output_tokens": chunk["usage"].get("completion_tokens", 0),
                    }
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if piece := delta.get("content"):
                        text.append(piece)
                        yield TextDelta(text=piece)
                        for marker in _MARKER.finditer("".join(text), scanned):
                            scanned = marker.end()
                            for n in map(int, marker.group(1).split(",")):
                                if 1 <= n <= len(sources):  # ignore invented numbers
                                    yield CitationEvent(citation=sources[n - 1])
                    for frag in delta.get("tool_calls") or []:
                        call = calls.setdefault(frag.get("index", len(calls)), _new_call())
                        call["id"] = frag.get("id") or call["id"]
                        fn = frag.get("function") or {}
                        call["name"] += fn.get("name") or ""
                        call["arguments"] += fn.get("arguments") or ""
                    finish = choice.get("finish_reason") or finish

        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": "".join(text)})
        for index in sorted(calls):
            event = _tool_call(calls[index], index)
            yield event
            content.append(
                {"type": "tool_use", "id": event.id, "name": event.name, "input": event.input}
            )
        reason = _FINISH.get(finish or "", "tool_use" if calls else "end_turn")
        yield Stop(reason=reason, usage=usage, content=content)

    def _request(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        opts: dict[str, Any],
    ) -> dict[str, Any]:
        opts = {k: v for k, v in opts.items() if k not in _ANTHROPIC_ONLY}
        choice = (opts.pop("tool_choice", None) or {"type": "auto"}).get("type")
        if choice == "none":
            tools = []
        elif choice != "auto":
            raise ValueError(f"tool_choice {choice!r} is not supported; use 'auto' (the default)")
        body: dict[str, Any] = {
            "model": opts.pop("model", None) or self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = [_to_function(t) for t in tools]
        effort = opts.pop("reasoning_effort", self.reasoning_effort)
        if effort is not None:
            body["reasoning_effort"] = effort
        body.update(opts)  # max_tokens, temperature, seed, top_p, stop ...
        return body


def _to_openai(messages: list[dict[str, Any]], sources: list[Citation]) -> list[dict[str, Any]]:
    """Messages-shape history as OpenAI chat messages.

    Search-result blocks are numbered across the whole conversation, in order, and each one is
    appended to ``sources`` so ``[n]`` maps back to ``sources[n - 1]``.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role, content = message["role"], message.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            turn: dict[str, Any] = {"role": "assistant", "content": _joined_text(content)}
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
                }
                for b in content
                if b.get("type") == "tool_use"
            ]
            if calls:
                turn["tool_calls"] = calls
            out.append(turn)
            continue
        # A user turn: tool results first (as `tool` messages), then any other content.
        rest: list[str] = []
        for block in content:
            if block.get("type") == "tool_result":
                body = _render(block.get("content", ""), sources)
                if block.get("is_error"):
                    body = f"Error: {body}"
                out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": body})
            else:
                rest.append(_render([block], sources))
        if any(rest):
            out.append({"role": role, "content": "\n\n".join(r for r in rest if r)})
    return out


def _render(content: str | list[dict[str, Any]], sources: list[Citation]) -> str:
    """Tool-result or user content as text, numbering any search results."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    numbered: list[str] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "search_result":
            body = search_result_text(block)
            sources.append(
                citation_from_source(block["source"], title=block.get("title", ""), cited_text=body)
            )
            numbered.append(f"[{len(sources)}] {block.get('title', '')}\n{body}")
    if numbered:
        parts.append("\n\n".join([CITE_INSTRUCTION, *numbered]))
    return "\n\n".join(p for p in parts if p)


def _joined_text(content: list[dict[str, Any]]) -> str:
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def _new_call() -> dict[str, str]:
    return {"id": "", "name": "", "arguments": ""}


def _tool_call(call: dict[str, str], index: int) -> ToolCall:
    try:
        args = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        # Small local models occasionally emit broken JSON. Keep the raw text so the agent can
        # answer with an error tool_result instead of crashing the turn.
        args = {"_invalid_arguments": call["arguments"]}
    return ToolCall(id=call["id"] or f"call_{index}", name=call["name"], input=args)


def _to_function(tool: dict[str, Any]) -> dict[str, Any]:
    """An Anthropic tool definition as an OpenAI function tool."""
    fn: dict[str, Any] = {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
    }
    if "strict" in tool:
        fn["strict"] = tool["strict"]
    return {"type": "function", "function": fn}


def _error_message(status: int, raw: bytes, body: dict[str, Any]) -> str:
    try:
        detail = json.loads(raw)["error"]
        detail = detail.get("message", detail) if isinstance(detail, dict) else detail
    except (ValueError, KeyError, TypeError):
        detail = raw.decode("utf-8", "replace")[:500]
    message = f"HTTP {status} from the LLM server for model {body['model']!r}: {detail}"
    if status == 404 and "not found" in str(detail):
        message += " (for the local profile, run `make ollama-models`)"
    return message


async def _sse_json(resp: httpx2.Response) -> AsyncIterator[dict[str, Any]]:
    """The JSON payloads of an OpenAI-style SSE stream, up to ``data: [DONE]``."""
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield json.loads(data)
