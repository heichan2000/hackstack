"""Claude through the official ``anthropic`` SDK, normalized to LLMEvents.

- Native citations: ``search_result`` blocks in tool results come back as
  ``search_result_location`` citations, mapped to :class:`~hackstack.models.Citation` through
  the block's ``source`` (see :mod:`hackstack.llm.citations`).
- Top-level ``cache_control`` (automatic prompt caching) on every request.
- Every stop reason is passed through on :class:`~hackstack.models.Stop`, with the assistant
  turn's content blocks unchanged (thinking signatures, citations, server-tool blocks) so the
  agent can append them as-is, and ``stop_details`` on a refusal. ``pause_turn`` means "append
  the content and resend"; that is the agent loop's job.
- ``tool_choice`` is ``auto`` only: current Claude models return a 400 for ``any``/``tool``.
- On the Claude API, models that support it get the server-side refusal fallback
  (``fallbacks: "default"``), which re-runs a declined request on a fallback model.

Also works against Ollama's Anthropic-compatible ``/v1/messages`` (``base_url`` without
``/v1``), which supports tool use but not ``search_result`` blocks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anthropic
import httpx2

if TYPE_CHECKING:
    from hackstack.config import LLMSettings, Secrets

from hackstack.llm.base import LLMError
from hackstack.llm.citations import citation_from_source
from hackstack.models import CitationEvent, LLMEvent, Stop, TextDelta, ToolCall
from hackstack.registry import LLM, register

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side `fallbacks: "default"` parameter on the Claude API.
_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
# OpenAI-side options with no Messages equivalent; dropped so one opts dict fits both adapters.
_OPENAI_ONLY = {"seed", "temperature", "top_p", "presence_penalty", "frequency_penalty"}


@register(LLM, "anthropic")
class AnthropicProvider:
    provider_name = "anthropic"  # gen_ai.provider.name on trace spans

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16000,
        reasoning_effort: str | None = None,
        refusal_fallback: bool | None = None,
        max_retries: int = 2,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        # Only the Claude API itself knows `fallbacks`; a custom base_url (Ollama, a proxy) gets
        # plain requests unless asked otherwise.
        self.refusal_fallback = base_url is None if refusal_fallback is None else refusal_fallback
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            http_client=http_client,
        )

    @classmethod
    def from_settings(cls, llm: LLMSettings, secrets: Secrets, *, model: str) -> AnthropicProvider:
        key = secrets.anthropic_api_key
        if key is None and llm.base_url is None:
            raise ValueError(
                "llm.provider anthropic needs ANTHROPIC_API_KEY in .env; "
                "or run fully locally with HACKSTACK_PROFILE=local"
            )
        return cls(
            model=model,
            # Ollama's /v1/messages ignores the key but the SDK wants one.
            api_key=key.get_secret_value() if key else "unused",
            base_url=llm.base_url,
            reasoning_effort=llm.reasoning_effort,
        )

    async def stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **opts: Any,
    ) -> AsyncIterator[LLMEvent]:
        params = self._params(system, messages, tools, opts)
        if self.refusal_fallback and params["model"].startswith(_FALLBACK_MODELS):
            manager = self._client.beta.messages.stream(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            )
        else:
            manager = self._client.messages.stream(**params)

        try:
            async with manager as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield TextDelta(text=delta.text)
                        elif (
                            delta.type == "citations_delta"
                            and delta.citation.type == "search_result_location"
                        ):
                            c = delta.citation
                            yield CitationEvent(
                                citation=citation_from_source(
                                    c.source, title=c.title or "", cited_text=c.cited_text
                                )
                            )
                    elif event.type == "content_block_stop":
                        block = event.content_block
                        if block.type == "tool_use":
                            yield ToolCall(id=block.id, name=block.name, input=block.input or {})
                final = await stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise LLMError(f"HTTP {exc.status_code} from the Anthropic API: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"cannot reach the Anthropic API: {exc}") from exc

        usage = final.usage.model_dump()
        details = getattr(final, "stop_details", None)
        yield Stop(
            reason=final.stop_reason or "end_turn",
            usage={k: usage[k] for k in _USAGE_KEYS if isinstance(usage.get(k), int)},
            content=[b.model_dump(mode="json", exclude_none=True) for b in final.content],
            details=details.model_dump(mode="json", exclude_none=True) if details else None,
        )

    def _params(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        opts: dict[str, Any],
    ) -> dict[str, Any]:
        opts = {k: v for k, v in opts.items() if k not in _OPENAI_ONLY}
        choice = opts.pop("tool_choice", None)
        if choice is not None and choice.get("type") not in ("auto", "none"):
            raise ValueError(
                f"tool_choice {choice.get('type')!r} is not supported (current Claude models "
                "return a 400); use 'auto' and name the tool in the prompt"
            )
        params: dict[str, Any] = {
            "model": opts.pop("model", None) or self.model,
            "max_tokens": opts.pop("max_tokens", self.max_tokens),
            "system": system,
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            params["tools"] = tools
        if choice is not None:
            params["tool_choice"] = choice
        effort = opts.pop("reasoning_effort", self.reasoning_effort)
        if effort in _EFFORTS:
            params["output_config"] = {**opts.pop("output_config", {}), "effort": effort}
        params.update(opts)  # thinking, output_config, stop_sequences, metadata ...
        return params
