from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from hackstack.connectors._http import RateLimitedClient


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def client(responses, clock=None, **kw):
    """A client whose transport plays ``responses`` (status, headers) in order."""
    seen = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        status, headers = item
        return httpx.Response(status, headers=headers, json={"n": len(seen)})

    clock = clock or Clock()
    c = RateLimitedClient(
        transport=httpx.MockTransport(handler), sleep=clock.sleep, clock=clock.time, **kw
    )
    return c, seen, clock


def test_retry_after_seconds_is_honoured():
    c, seen, clock = client([(429, {"Retry-After": "2"}), (200, {})])
    r = c.get("https://api.example.com/x")
    assert r.status_code == 200 and len(seen) == 2
    assert clock.sleeps == [2.0]


def test_retry_after_http_date_is_honoured():
    clock = Clock()
    when = datetime.fromtimestamp(clock.now, UTC) + timedelta(seconds=7)
    c, _, _ = client([(503, {"Retry-After": format_datetime(when, usegmt=True)}), (200, {})], clock)
    c.get("https://api.example.com/x")
    assert clock.sleeps == [pytest.approx(7.0, abs=1.0)]


def test_backoff_is_exponential_without_retry_after():
    c, _, clock = client([(503, {}), (502, {}), (529, {}), (200, {})], backoff_base=1.0)
    assert c.get("https://x.test/").status_code == 200
    assert clock.sleeps == [1.0, 2.0, 4.0]


def test_gives_up_after_max_retries_and_returns_last_response():
    c, seen, _ = client([(429, {})] * 4, max_retries=3)
    r = c.get("https://x.test/")
    assert r.status_code == 429 and len(seen) == 4


def test_client_errors_are_not_retried():
    c, seen, clock = client([(404, {})])
    assert c.get("https://x.test/").status_code == 404
    assert len(seen) == 1 and clock.sleeps == []


def test_transport_errors_are_retried():
    c, seen, _ = client([httpx.ConnectError("boom"), (200, {})])
    assert c.get("https://x.test/").status_code == 200 and len(seen) == 2


def test_transport_errors_raise_after_max_retries():
    c, _, _ = client([httpx.ConnectError("boom")] * 3, max_retries=2)
    with pytest.raises(httpx.ConnectError):
        c.get("https://x.test/")


def test_requests_are_spaced_to_max_rps():
    c, _, clock = client([(200, {})] * 3, max_rps=2.0)
    for _ in range(3):
        c.get("https://x.test/")
    assert clock.sleeps == [0.5, 0.5]


def test_retry_after_is_capped():
    c, _, clock = client([(429, {"Retry-After": "9999"}), (200, {})], max_wait=30.0)
    c.get("https://x.test/")
    assert clock.sleeps == [30.0]
