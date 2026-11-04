"""One small JSON-over-HTTP helper for the model adapters that talk REST (Ollama, Voyage, ...).

It retries rate limits (429) and transient server/network errors with exponential backoff,
honouring ``Retry-After``, and turns everything else into a :class:`ProviderError` whose message
says which provider failed, the HTTP status and the server's own error text.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class ProviderError(RuntimeError):
    """A model provider refused or failed a request."""


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    provider: str,
    headers: dict[str, str] | None = None,
    hint: Callable[[int | None, str], str] | None = None,
    retries: int = 3,
    backoff: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON response.

    ``hint(status, detail)`` may add a how-to-fix line to the error (``status`` is ``None`` when
    the server could not be reached).
    """
    for attempt in range(retries + 1):
        last = attempt == retries
        try:
            response = client.post(url, json=payload, headers=headers)
        except httpx.TransportError as exc:
            if not last:
                sleep(backoff * 2**attempt)
                continue
            raise ProviderError(
                _message(provider, f"cannot reach {url}: {exc}", None, "", hint)
            ) from exc

        if response.status_code in RETRY_STATUSES and not last:
            sleep(_retry_after(response) or backoff * 2**attempt)
            continue
        if response.status_code >= 400:
            detail = _detail(response)
            text = f"HTTP {response.status_code} from {url}: {detail}"
            raise ProviderError(_message(provider, text, response.status_code, detail, hint))
        return response.json()
    raise AssertionError("unreachable")  # pragma: no cover


def _message(provider, text, status, detail, hint) -> str:
    message = f"{provider}: {text}"
    if hint is not None and (extra := hint(status, detail)):
        message += f"\n  {extra}"
    return message


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        for key in ("error", "detail", "message"):
            if key in body:
                value = body[key]
                if isinstance(value, dict):  # OpenAI style: {"error": {"message": ...}}
                    value = value.get("message", value)
                return str(value)
    return str(body)[:500]
