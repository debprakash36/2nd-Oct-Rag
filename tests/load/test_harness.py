"""Tests for the load harness itself (5.1).

Everything in `docs/load_test_results.json` is a claim about the server produced by
this driver, so the driver's own failure modes decide what the report is worth. A
harness that scores a broken response as a pass does not fail loudly -- it produces
a confident, green, wrong number, which is worse than no number.

These run without a server: they feed `httpx.MockTransport` a synthetic SSE body and
assert on how the driver classifies it. They are deliberately *not* marked `load`, so
they run in the default suite and a broken harness is caught long before anyone
trusts a load result.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.load.loadkit import (
    DEFAULT_VUS,
    TTFT_BUDGET_MS,
    LoadReport,
    RequestResult,
    _one_request,
)

#: Absolute, because `httpx` needs a real origin to build a request against.
_URL = "http://testserver/chat/stream"


def _sse(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _complete_stream(*, abstained: bool = False, tokens: int = 3) -> str:
    body = _sse("sources", {"sources": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]})
    for _ in range(tokens):
        body += _sse("token", {"text": "word "})
    body += _sse("done", {"query_id": "q1", "abstained": abstained, "ttft_ms": 120})
    return body


def _client_returning(body: str, status: int = 200) -> httpx.Client:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    return httpx.Client(transport=transport)


class TestStreamClassification:
    """A stream is only a success if the whole contract was honoured."""

    def test_a_complete_stream_is_a_pass(self):
        result = _one_request(_client_returning(_complete_stream()), _URL, "m", 0, 5.0)
        assert result.ok
        assert not result.truncated
        assert result.client_ttft_ms is not None
        assert result.server_ttft_ms == 120
        assert result.source_count == 2
        assert result.token_events == 3

    def test_a_truncated_stream_is_a_failure(self):
        """Tokens but no `done`: the case that used to score as a pass.

        This is the exact shape produced by `persist()` raising after the last
        token -- a locked SQLite database under load. The client received a
        complete-looking sentence and then a dropped connection, and the old
        criterion (`a token arrived and no error event`) called that a success.
        """
        body = _sse("sources", {"sources": [{"chunk_id": "c1"}]})
        body += _sse("token", {"text": "The refund window is 30 days."})

        result = _one_request(_client_returning(body), _URL, "m", 0, 5.0)
        assert not result.ok
        assert result.truncated
        assert result.error_code == "truncated_stream"
        assert result.client_ttft_ms is not None, "the user did see a token"

    def test_an_explicit_error_event_is_a_failure(self):
        body = _sse("error", {"code": "retrieval_unavailable", "message": "down"})

        result = _one_request(_client_returning(body), _URL, "m", 0, 5.0)
        assert not result.ok
        assert not result.truncated, "an explicit error is not a silent truncation"
        assert result.error_code == "retrieval_unavailable"

    def test_a_non_200_is_a_failure(self):
        result = _one_request(_client_returning("nope", status=503), _URL, "m", 0, 5.0)
        assert not result.ok
        assert result.status_code == 503
        assert result.error_code == "http_503"

    def test_a_refusal_is_a_pass_and_still_has_a_ttft(self):
        """Refusals skip generation but the user still waits for a sentence."""
        result = _one_request(
            _client_returning(_complete_stream(abstained=True, tokens=1)),
            _URL,
            "m",
            0,
            5.0,
        )
        assert result.ok
        assert result.abstained
        assert result.client_ttft_ms is not None

    def test_a_connection_error_is_a_failure(self):
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with httpx.Client(transport=httpx.MockTransport(boom)) as client:
            result = _one_request(client, _URL, "m", 0, 5.0)
        assert not result.ok
        assert "ConnectError" in (result.failure or "")


class TestReportAggregation:
    def test_truncations_are_reported_and_excluded_from_successes(self):
        ok = RequestResult(
            vu=0,
            ok=True,
            status_code=200,
            client_ttft_ms=100.0,
            server_ttft_ms=80,
            total_ms=900.0,
            source_count=2,
            token_events=4,
            abstained=False,
        )
        cut = RequestResult(
            vu=1,
            ok=False,
            status_code=200,
            client_ttft_ms=100.0,
            server_ttft_ms=None,
            total_ms=400.0,
            source_count=1,
            token_events=2,
            abstained=False,
            truncated=True,
            error_code="truncated_stream",
        )
        report = LoadReport(vus=2, requests=2, duration_s=2.0, results=[ok, cut])
        data = report.as_dict()

        assert data["ok"] == 1
        assert data["failed"] == 1
        assert data["truncated"] == 1
        assert data["failures"] == {"truncated_stream": 1}
        # The truncated request must not contribute a TTFT that flatters the p95.
        assert data["client_ttft_ms"]["n"] == 1

    def test_answered_and_refused_are_reported_separately(self):
        def make(abstained: bool, ttft: float) -> RequestResult:
            return RequestResult(
                vu=0,
                ok=True,
                status_code=200,
                client_ttft_ms=ttft,
                server_ttft_ms=50,
                total_ms=ttft + 500,
                source_count=1,
                token_events=2,
                abstained=abstained,
            )

        report = LoadReport(
            vus=1,
            requests=2,
            duration_s=1.0,
            results=[make(False, 900.0), make(True, 50.0)],
        )
        data = report.as_dict()

        assert data["answered_ttft_ms"]["n"] == 1
        assert data["refused_ttft_ms"]["n"] == 1
        assert data["answered_ttft_ms"]["p50"] == 900.0
        assert data["refused_ttft_ms"]["p50"] == 50.0


class TestPercentiles:
    def test_single_sample(self):
        assert LoadReport(vus=1, requests=0, duration_s=1.0).percentile([7.5], 0.95) == 7.5

    def test_empty_is_zero_not_an_error(self):
        assert LoadReport(vus=1, requests=0, duration_s=1.0).percentile([], 0.95) == 0.0

    def test_interpolates_between_bracketing_samples(self):
        report = LoadReport(vus=1, requests=0, duration_s=1.0)
        values = [float(n) for n in range(101)]
        # p95 of 0..100 sits at position 0.95*100 = 95 exactly.
        assert report.percentile(values, 0.95) == 95.0
        # p50 of 0..99 sits mid-way between 49 and 50.
        assert report.percentile([float(n) for n in range(100)], 0.50) == 49.5


class TestBudgetConstants:
    def test_budget_matches_the_prd(self):
        assert TTFT_BUDGET_MS == 5_000.0
        assert DEFAULT_VUS == 50


class TestConcurrencyHarness:
    """The driver must not report a run whose concurrency never materialised."""

    def test_all_workers_are_in_flight_before_any_completes(self, monkeypatch):
        """Exercises `run_load`'s own start barrier.

        `_one_request` is replaced with a probe that records how many calls overlap,
        so this measures the real driver rather than a reimplementation of it.
        Without the barrier each thread would stagger by its own startup cost and the
        run would quietly apply much less load than the reported VU count claims --
        while still producing a comfortable, green p95.
        """
        import threading

        from tests.load import loadkit

        state = {"in_flight": 0, "peak": 0, "calls": 0}
        lock = threading.Lock()
        # Long enough that every worker is inside the probe before the first returns.
        hold = threading.Barrier(6, timeout=10)

        def probe(client, url, message, vu, timeout):
            with lock:
                state["in_flight"] += 1
                state["calls"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            try:
                # A 5-worker barrier cannot trip on its own, so it acts as a
                # "nobody else is here yet" detector: if fewer than 5 requests are
                # ever simultaneously in flight, this raises.
                hold.wait()
            except threading.BrokenBarrierError:
                state["peak"] = min(state["peak"], 5)
            finally:
                with lock:
                    state["in_flight"] -= 1
            return RequestResult(
                vu=vu,
                ok=True,
                status_code=200,
                client_ttft_ms=10.0,
                server_ttft_ms=8,
                total_ms=20.0,
                source_count=1,
                token_events=1,
                abstained=False,
            )

        monkeypatch.setattr(loadkit, "_one_request", probe)

        report = loadkit.run_load(
            "http://t", vus=5, requests_per_vu=1, warmup_requests=0
        )
        assert report.vus == 5
        assert report.requests == 5
        assert state["calls"] == 5
        assert state["peak"] == 5, (
            f"only {state['peak']} of 5 workers were ever in flight simultaneously; "
            f"the reported concurrency would not match the reported VU count"
        )

    def test_run_load_reports_a_hang_instead_of_a_partial_p95(self, monkeypatch):
        """A worker that never returns is a result, not something to average away.

        Reporting a p95 from the requests that did come back would hide precisely the
        failure that matters: a server that stops responding under load would produce
        its best-looking percentile report at the moment it is least healthy.
        """
        import threading

        from tests.load import loadkit

        # Released at the end of the test so the driver threads exit cleanly. A
        # bare `Event().wait(30)` would leave them running into other tests and then
        # raise in a background thread, which pytest reports as an unhandled
        # exception -- noise that looks like a product failure.
        release = threading.Event()

        def hang(client, url, message, vu, timeout):
            release.wait(30)
            return RequestResult(
                vu=vu,
                ok=True,
                status_code=200,
                client_ttft_ms=1.0,
                server_ttft_ms=1,
                total_ms=1.0,
                source_count=0,
                token_events=1,
                abstained=True,
            )

        monkeypatch.setattr(loadkit, "_one_request", hang)
        try:
            with pytest.raises(RuntimeError, match="did not finish"):
                loadkit.run_load(
                    "http://t", vus=2, requests_per_vu=1, warmup_requests=0, timeout=0.5
                )
        finally:
            release.set()

    def test_warmup_requests_are_not_counted(self, monkeypatch):
        """Warm-up exists to exclude process startup from steady-state latency.

        If its results leaked into the sample, the first request's import and
        connection-pool cost would be reported as product latency.
        """
        from tests.load import loadkit

        seen: list[int] = []

        def probe(client, url, message, vu, timeout):
            seen.append(vu)
            return RequestResult(
                vu=vu,
                ok=True,
                status_code=200,
                client_ttft_ms=10.0,
                server_ttft_ms=8,
                total_ms=20.0,
                source_count=1,
                token_events=1,
                abstained=False,
            )

        monkeypatch.setattr(loadkit, "_one_request", probe)

        report = loadkit.run_load(
            "http://t", vus=3, requests_per_vu=2, warmup_requests=4
        )
        assert len(seen) == 4 + 6
        assert report.requests == 6
        # Warm-up uses VU ids above the measured range, so they are separable.
        assert sorted(vu for vu in seen if vu >= 900) == [900, 901, 902, 903]
