"""Load driver: 50 concurrent users against the real streaming chat path (5.1, NFR-1).

NFR-1 is "p95 time-to-first-token <= 5 s". Two things have to be measured honestly
for that number to mean anything.

**TTFT is measured to the first `token` the client can read, not to the response
arriving.** The endpoint sends `sources` before any token (FR-20), so a driver that
stops its clock on the first SSE frame would report the time to reach the sources
panel and call it TTFT. The clock stops on the first `token` event, and the driver
reads the response incrementally — a driver using a buffered `response.text` would
have no first-token timestamp at all.

**The server's own `ttft_ms` is recorded but not trusted as the only number.** It is
measured inside the request, before serialization and socket write, so it excludes
exactly the part of the latency a user waits through. Both are reported: the
client-observed figure drives the assertion, and the delta between them is a
direct measure of time spent in the transport, which is what a reverse proxy or a
buffering misconfiguration would inflate.

The driver uses threads rather than asyncio because httpx's sync client releases the
GIL on socket reads; with 50 workers the cost of the driver's own code is small next
to the request, and a threaded driver has no event-loop scheduling artifact to
explain away in the results.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx

#: NFR-1. Exceeding this fails the test.
TTFT_BUDGET_MS = 5_000.0

#: implementation.md 5.1 specifies 50 concurrent users.
DEFAULT_VUS = 50

#: NFR-1's second clause, checked while the same run is in flight.
FULL_ANSWER_BUDGET_MS = 20_000.0

#: Each worker gets its own identity, and a quota above the requests it will make.
#: See conftest's module docstring: the limiter is per-client, and a shared address
#: would make this a test of the limiter.
RATE_LIMIT_PER_VU = 20

_QUESTIONS = (
    "What is the refund window for digital products?",
    "How long do I have to return a physical good?",
    "What are the standard shipping times?",
    "How does the warranty period work?",
    "What happens if a customer disputes a charge?",
    "How long is personal data retained?",
    "When are invoices due?",
    "What does the escalation policy say?",
    "Which items are non-refundable?",
    "How quickly are refunds processed?",
)


@dataclass
class RequestResult:
    """One measured request."""

    vu: int
    ok: bool
    status_code: int
    client_ttft_ms: float | None
    server_ttft_ms: int | None
    total_ms: float
    source_count: int
    token_events: int
    abstained: bool
    #: True when the stream produced tokens but never sent `done`.
    #:
    #: This is not a cosmetic case. A truncated stream is a *successful-looking*
    #: answer to the user: the sentence arrived, and then the connection closed
    #: without an error event. It happens when `persist()` raises after the tokens
    #: are flushed -- SQLite write contention under load, for instance -- and the
    #: failure escapes the `except AppError` handler. Counting these as passes is
    #: how a benchmark ends up reporting a healthy p95 for a server that is dropping
    #: half its query logs, so they are separated out and failed explicitly.
    truncated: bool = False
    error_code: str | None = None
    failure: str | None = None

    @property
    def transport_ms(self) -> float | None:
        """Client-observed TTFT minus the server's own measurement.

        The gap the server does not see: serialization, socket buffering, and
        anything between the process and the client. A large value under low
        concurrency points at the transport or a proxy buffering the stream, which
        is a direct NFR-1 violation that a server-side-only metric cannot detect.
        """
        if self.client_ttft_ms is None or self.server_ttft_ms is None:
            return None
        return self.client_ttft_ms - float(self.server_ttft_ms)


@dataclass
class LoadReport:
    """Aggregate of a run, in the shape the perf report needs."""

    vus: int
    requests: int
    duration_s: float
    results: list[RequestResult] = field(default_factory=list)
    #: Peak in-flight requests the *server* reported, or None when no probe ran.
    #: Client-attempted concurrency is not the same number: a request waiting for a
    #: threadpool slot, a socket the OS has not yet accepted, or a client that gave up
    #: early all make the two disagree. The probe counts in-flight at the ASGI
    #: boundary, so it is the user-facing number (open connections and live streams),
    #: not the number of threads executing.
    observed_peak_concurrency: int | None = None

    @property
    def successes(self) -> list[RequestResult]:
        return [r for r in self.results if r.ok]

    @property
    def answered(self) -> list[RequestResult]:
        """Successful requests that produced a real answer.

        Split out because a refusal short-circuits before the LLM call (architecture.md
        3.3): it streams a canned sentence and never pays generation. Averaging
        refusals into the TTFT sample makes the p95 look better than the experience of
        a user who got an answer, so the report carries both figures and the gate
        asserts on this one.
        """
        return [r for r in self.successes if not r.abstained]

    @property
    def refused(self) -> list[RequestResult]:
        return [r for r in self.successes if r.abstained]

    @staticmethod
    def _ttfts(results: list[RequestResult]) -> list[float]:
        return [r.client_ttft_ms for r in results if r.client_ttft_ms is not None]

    @property
    def client_ttfts(self) -> list[float]:
        return [r.client_ttft_ms for r in self.successes if r.client_ttft_ms is not None]

    @property
    def server_ttfts(self) -> list[float]:
        return [r.server_ttft_ms for r in self.successes if r.server_ttft_ms is not None]

    @property
    def totals(self) -> list[float]:
        return [r.total_ms for r in self.successes]

    def percentile(self, values: list[float], fraction: float) -> float:
        """Linear-interpolated percentile.

        `bench_ttft.py` uses a nearest-rank index; interpolation is used here because
        a p95 computed from 50 samples is sensitive to which sample sits on the
        boundary, and the nearest-rank method can only ever report a value that was
        actually observed. Interpolating between the two bracketing samples is what
        makes a 50-request p95 comparable to a 5000-request p95.
        """
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[int(position)]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def as_dict(self) -> dict[str, Any]:
        client = self.client_ttfts
        server = self.server_ttfts
        totals = self.totals
        transport = [r.transport_ms for r in self.successes if r.transport_ms is not None]
        failures = Counter(
            r.failure or r.error_code or f"http_{r.status_code}"
            for r in self.results
            if not r.ok
        )
        return {
            "vus": self.vus,
            "observed_peak_concurrency": self.observed_peak_concurrency,
            "requests": self.requests,
            "ok": len(self.successes),
            "failed": self.requests - len(self.successes),
            "truncated": sum(1 for r in self.results if r.truncated),
            "duration_s": round(self.duration_s, 2),
            "throughput_rps": round(
                len(self.successes) / self.duration_s if self.duration_s else 0.0, 2
            ),
            "client_ttft_ms": {
                "p50": round(self.percentile(client, 0.50), 1),
                "p95": round(self.percentile(client, 0.95), 1),
                "p99": round(self.percentile(client, 0.99), 1),
                "mean": round(statistics.fmean(client), 1) if client else 0.0,
                "max": round(max(client), 1) if client else 0.0,
                "n": len(client),
            },
            "server_ttft_ms": {
                "p50": round(self.percentile(server, 0.50), 1),
                "p95": round(self.percentile(server, 0.95), 1),
                "n": len(server),
            },
            "transport_gap_ms": {
                "p50": round(self.percentile(transport, 0.50), 1),
                "p95": round(self.percentile(transport, 0.95), 1),
                "n": len(transport),
            },
            "full_answer_ms": {
                "p50": round(self.percentile(totals, 0.50), 1),
                "p95": round(self.percentile(totals, 0.95), 1),
                "max": round(max(totals), 1) if totals else 0.0,
            },
            "answered_ttft_ms": {
                "p50": round(self.percentile(self._ttfts(self.answered), 0.50), 1),
                "p95": round(self.percentile(self._ttfts(self.answered), 0.95), 1),
                "max": round(max(self._ttfts(self.answered), default=0.0), 1),
                "n": len(self.answered),
            },
            "refused_ttft_ms": {
                "p50": round(self.percentile(self._ttfts(self.refused), 0.50), 1),
                "p95": round(self.percentile(self._ttfts(self.refused), 0.95), 1),
                "n": len(self.refused),
            },
            "abstained": sum(1 for r in self.successes if r.abstained),
            "sources_total": sum(r.source_count for r in self.successes),
            "failures": dict(failures),
        }


def _iter_sse_events(buffer: str) -> tuple[list[tuple[str, dict]], str]:
    """Split complete SSE frames out of `buffer`, returning them and the remainder.

    Hand-rolled rather than using a library because the driver must timestamp the
    moment the first `token` frame is *read off the socket*. A parser that consumed
    the whole body first would destroy the only measurement that matters.
    """
    events: list[tuple[str, dict]] = []
    remainder = buffer
    while "\n\n" in remainder:
        frame, _, remainder = remainder.partition("\n\n")
        name = ""
        data: dict = {}
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[len("data: ") :])
                except json.JSONDecodeError:
                    data = {}
        if name:
            events.append((name, data))
    return events, remainder


def _one_request(
    client: httpx.Client, url: str, message: str, vu: int, timeout: float
) -> RequestResult:
    """Stream one chat request, stopping the clock on the first visible token."""
    started = time.perf_counter()
    status_code = 0
    first_token_ms: float | None = None
    server_ttft: int | None = None
    source_count = 0
    token_events = 0
    abstained = False
    saw_done = False
    error_code: str | None = None
    buffer = ""

    headers = {
        # Distinct identity per virtual user, so the per-client rate limit applies to
        # each user separately rather than to the whole run as one abusive address.
        "X-Forwarded-For": f"10.0.{vu // 250}.{vu % 250 + 1}",
    }

    try:
        with client.stream(
            "POST",
            url,
            json={"message": message, "answer_style": "concise"},
            headers=headers,
            timeout=timeout,
        ) as response:
            status_code = response.status_code
            if response.status_code != 200:
                response.read()
                return RequestResult(
                    vu=vu,
                    ok=False,
                    status_code=status_code,
                    client_ttft_ms=None,
                    server_ttft_ms=None,
                    total_ms=(time.perf_counter() - started) * 1000,
                    source_count=0,
                    token_events=0,
                    abstained=False,
                    error_code=f"http_{status_code}",
                )
            for chunk in response.iter_text():
                buffer += chunk
                events, buffer = _iter_sse_events(buffer)
                for name, data in events:
                    if name == "sources":
                        source_count = len(data.get("sources") or [])
                    elif name == "token":
                        if first_token_ms is None:
                            # The first token the user could see. A refusal's text is
                            # a token too, so an abstention still has a real TTFT
                            # rather than being dropped from the sample.
                            first_token_ms = (time.perf_counter() - started) * 1000
                        token_events += 1
                    elif name == "done":
                        server_ttft = data.get("ttft_ms")
                        abstained = bool(data.get("abstained"))
                        saw_done = True
                    elif name == "error":
                        error_code = data.get("code")
    except (httpx.HTTPError, OSError) as exc:
        return RequestResult(
            vu=vu,
            ok=False,
            status_code=status_code,
            client_ttft_ms=None,
            server_ttft_ms=None,
            total_ms=(time.perf_counter() - started) * 1000,
            source_count=0,
            token_events=0,
            abstained=False,
            failure=f"{type(exc).__name__}: {exc}",
        )

    total_ms = (time.perf_counter() - started) * 1000
    truncated = not saw_done and error_code is None and status_code == 200
    ok = error_code is None and first_token_ms is not None and not truncated
    if truncated:
        error_code = error_code or "truncated_stream"
    return RequestResult(
        vu=vu,
        ok=ok,
        status_code=status_code,
        client_ttft_ms=first_token_ms,
        server_ttft_ms=server_ttft,
        total_ms=total_ms,
        source_count=source_count,
        token_events=token_events,
        abstained=abstained,
        truncated=truncated,
        error_code=error_code,
    )


def run_load(
    base_url: str,
    *,
    vus: int = DEFAULT_VUS,
    requests_per_vu: int = 2,
    warmup_requests: int = 5,
    timeout: float = 60.0,
) -> LoadReport:
    """Drive `vus` concurrent workers and return the aggregate.

    The warm-up runs first and is discarded: the first request through the process
    pays for import, connection-pool growth, and page-cache misses, and including it
    would attribute startup cost to steady-state latency.
    """
    url = f"{base_url}/chat/stream"

    # The client spans warm-up *and* the measured run. Scoping it to the warm-up
    # would close the connection pool before the workers start, and every worker
    # would then fail with "client has been closed" -- a harness bug that looks
    # exactly like a server refusing load.
    with httpx.Client() as client:
        for i in range(warmup_requests):
            _one_request(client, url, _QUESTIONS[i % len(_QUESTIONS)], 900 + i, timeout)

        results: list[RequestResult] = []
        lock = threading.Lock()
        start_barrier = threading.Barrier(vus)
        counter = threading.Lock()
        issued = 0

        def worker(vu: int) -> None:
            nonlocal issued
            # All workers start their first request at the same moment. Without the
            # barrier they would stagger by however long each thread took to spin up,
            # and a run whose concurrency never actually reached 50 would still be
            # reported as a 50-VU test.
            start_barrier.wait()
            for _n in range(requests_per_vu):
                with counter:
                    issued += 1
                    index = issued
                message = _QUESTIONS[index % len(_QUESTIONS)]
                result = _one_request(client, url, message, vu, timeout)
                with lock:
                    results.append(result)

        started = time.perf_counter()
        threads = [
            threading.Thread(target=worker, args=(vu,), name=f"vu-{vu}", daemon=True)
            for vu in range(vus)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=timeout * (requests_per_vu + 2))
        duration = time.perf_counter() - started

    still_running = [t.name for t in threads if t.is_alive()]
    if still_running:
        # A hang is a result, not something to average away. Reporting a p95 from
        # the requests that did return would hide exactly the failure that matters.
        raise RuntimeError(
            f"{len(still_running)} workers did not finish within the timeout "
            f"(e.g. {still_running[:3]})"
        )

    return LoadReport(vus=vus, requests=len(results), duration_s=duration, results=results)
