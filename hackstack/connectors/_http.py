"""``RateLimitedClient``: the one HTTP client every API connector shares.

It spaces requests to ``max_rps``, retries 429 / 5xx / 529 and transport errors, and honours
``Retry-After`` (seconds or an HTTP date) before falling back to exponential backoff. Notion
(~3 rps, 429/529), Slack and GitHub secondary limits all speak this dialect.

After ``max_retries`` the last response is returned as is, so the caller decides whether to
``raise_for_status()``. Transport errors re-raise after the last attempt.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


class RateLimitedClient:
    def __init__(
        self,
        *,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        max_rps: float | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        max_wait: float = 120.0,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )
        self._interval = 1.0 / max_rps if max_rps else 0.0
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_wait = max_wait
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            self._throttle()
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError:
                if attempt >= self._max_retries:
                    raise
                self._wait(self._backoff(attempt))
                attempt += 1
                continue
            if response.status_code not in RETRY_STATUSES or attempt >= self._max_retries:
                return response
            delay = _retry_after(response, now=self._clock)
            self._wait(self._backoff(attempt) if delay is None else delay)
            attempt += 1

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RateLimitedClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- internals ---------------------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        return min(self._backoff_base * (2**attempt), self._max_wait)

    def _wait(self, seconds: float) -> None:
        seconds = max(0.0, min(seconds, self._max_wait))
        if seconds:
            self._sleep(seconds)
        # a deliberate wait also satisfies the rate limit
        self._last = self._clock()

    def _throttle(self) -> None:
        now = self._clock()
        if self._interval and self._last is not None:
            gap = self._interval - (now - self._last)
            if gap > 0:
                self._sleep(gap)
                now = self._clock()
        self._last = now


def _retry_after(response: httpx.Response, *, now: Callable[[], float]) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, when.timestamp() - now())
