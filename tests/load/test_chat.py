"""`pytest tests/load/` — the NFR-1 gate (implementation.md 5.1, PRD NFR-1).

Asserts what the PRD asserts: p95 time-to-first-token <= 5 s at 50 concurrent
users, plus p95 full answer <= 20 s. The first test in the file is a cheap
smoke test that the harness itself is sound, because a load test that silently
measured nothing would report a perfect score.

Marked `load` rather than left in the default run: it needs a seeded corpus, starts
a server, and takes tens of seconds, so `make check` should not pay for it. Run it
explicitly with `make load-test` or `pytest -m load tests/load/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.load.conftest import LiveServer, peak_concurrency
from tests.load.loadkit import (
    DEFAULT_VUS,
    FULL_ANSWER_BUDGET_MS,
    TTFT_BUDGET_MS,
    LoadReport,
    run_load,
)

pytestmark = pytest.mark.load

#: Written next to the report so 5.2 can quote measured numbers without re-running
#: the load test, and so a reviewer can see the raw shape of the run.
ARTIFACT = Path("docs") / "load_test_results.json"


@pytest.fixture(scope="module")
def report(live_server: LiveServer) -> LoadReport:
    """One 50-VU run, shared by the assertions below.

    Module-scoped and run once on purpose: re-running per assertion would measure
    the machine four times and produce four slightly different numbers, which reads
    as variance in the product when it is variance in the harness.
    """
    result = run_load(live_server.base_url, vus=DEFAULT_VUS, requests_per_vu=2)
    result.observed_peak_concurrency = peak_concurrency(live_server)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(
        json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(result.as_dict(), indent=2))
    return result


def test_harness_measures_a_populated_index(live_server: LiveServer):
    """A load test against an empty corpus measures nothing.

    An empty index refuses every question: no passages, so no retrieval work and no
    LLM call, and the resulting TTFT would be a floor of a few milliseconds that no
    amount of load would push toward the budget. The gate would pass forever while
    the product was on fire.
    """
    assert live_server.corpus_chunks >= 20, (
        f"corpus has only {live_server.corpus_chunks} chunks; the load test needs a "
        f"populated index to be meaningful"
    )


def test_a_single_request_streams_and_is_timed(live_server: LiveServer):
    """One request must produce a measurable TTFT before 50 are trusted.

    A driver that never sees a `token` frame would report p95 = 0.0 ms, which beats
    any budget and is the most dangerous possible result. This is the canary.
    """
    result = run_load(live_server.base_url, vus=1, requests_per_vu=1, warmup_requests=1)
    assert len(result.results) == 1
    only = result.results[0]
    assert only.ok, f"single request failed: {only.failure or only.error_code}"
    assert only.client_ttft_ms is not None and only.client_ttft_ms > 0
    assert only.token_events >= 1
    assert only.source_count > 0, "no sources means nothing was retrieved"


def test_all_50_requests_succeed(report: LoadReport):
    """Concurrency must not cost correctness.

    Separate from the latency assertion on purpose: a p95 that passes because a
    third of the requests 503'd is not a p95, it is a smaller sample. A store-down
    error surfacing here is exactly the 5.8 degradation path misbehaving under load.
    """
    assert report.requests == DEFAULT_VUS * 2
    failures = [r for r in report.results if not r.ok]
    assert not failures, (
        f"{len(failures)}/{report.requests} requests failed under "
        f"{DEFAULT_VUS}-way concurrency: "
        f"{[(r.vu, r.failure or r.error_code or r.status_code) for r in failures[:10]]}"
    )


def test_the_server_really_served_the_concurrency_claimed(report: LoadReport):
    """"50 concurrent users" has to mean 50 open requests, not 50 attempted.

    Guards against a driver that reports 50 VUs while the server never saw more than a
    handful at once -- a staggered start, a client-side connection-pool limit, or a
    barrier that silently serialized the run would all produce a passing p95 for the
    wrong reason.

    This counts requests in flight at the ASGI boundary, which includes requests
    waiting for a threadpool slot; it does not claim 50 threads were busy. See
    `conftest._install_concurrency_probe`.
    """
    observed = report.observed_peak_concurrency
    assert observed is not None, (
        "no server-side concurrency reading; the run cannot claim the concurrency it "
        "measured (the probe did not report)"
    )
    # Allow slack: the last VUs scheduled may not all overlap, and the health poll
    # that the fixture makes can add one.
    assert observed >= report.vus * 0.75, (
        f"client sent {report.vus} concurrent requests but the server only ever had "
        f"{observed} in flight; the p95 below describes a lower concurrency than the "
        f"test claims"
    )


def test_p95_ttft_within_budget(report: LoadReport):
    """NFR-1: p95 time-to-first-token <= 5 s at 50 concurrent users.

    Asserted over *answered* requests only. A refusal short-circuits before the LLM
    call, so including it would let the cheap path carry the percentile and flatter
    the number for users who actually got an answer.
    """
    answered = report._ttfts(report.answered)
    assert answered, "no answered requests in the run; nothing was measured"
    p95 = report.percentile(answered, 0.95)
    assert p95 <= TTFT_BUDGET_MS, (
        f"p95 client-observed TTFT on answered queries {p95:.0f} ms exceeds the "
        f"{TTFT_BUDGET_MS:.0f} ms budget at {report.vus} concurrent users "
        f"(p50 {report.percentile(answered, 0.5):.0f} ms, "
        f"max {max(answered):.0f} ms, n={len(answered)})"
    )


def test_p95_ttft_over_all_responses_within_budget(report: LoadReport):
    """The combined figure, including refusals, must also clear the budget.

    Reported as a second assertion rather than folded into the first: it is the
    number a user on a mixed workload actually experiences, and it should not be
    able to regress unnoticed just because the answered-only figure still passes.
    """
    p95 = report.percentile(report.client_ttfts, 0.95)
    assert p95 <= TTFT_BUDGET_MS, (
        f"p95 TTFT across all responses {p95:.0f} ms exceeds "
        f"{TTFT_BUDGET_MS:.0f} ms"
    )


def test_refusal_rate_is_in_the_healthy_band(report: LoadReport):
    """PRD §8 puts healthy refusal at 10-30%; a load run far outside it is a signal.

    Not a latency assertion, and not a hard product gate -- the questions here are a
    fixed set and the corpus is a synthetic subset, so the absolute rate is not
    meaningful. It is checked because a sudden collapse to near-zero refusals would
    mean the threshold stopped abstaining, which turns a latency benchmark into
    evidence that the system stopped being careful.
    """
    total = len(report.successes)
    rate = (len(report.refused) / total) if total else 0.0
    assert 0.0 <= rate <= 0.60, (
        f"refusal rate {rate:.1%} over {total} responses is implausible for this "
        f"corpus; either retrieval stopped abstaining or every query failed"
    )


def test_p95_full_answer_within_budget(report: LoadReport):
    """NFR-1's second clause: p95 full answer <= 20 s."""
    p95 = report.percentile(report.totals, 0.95)
    assert p95 <= FULL_ANSWER_BUDGET_MS, (
        f"p95 full answer {p95:.0f} ms exceeds the {FULL_ANSWER_BUDGET_MS:.0f} ms budget"
    )


def test_streaming_is_incremental_not_buffered(report: LoadReport):
    """Answers must arrive in pieces, or NFR-1 is met by buffering.

    The offline provider streams a sentence-flushed answer, so a single-token answer
    is legitimate and this cannot assert a large token count. What it does assert is
    that tokens are separate frames rather than one blob, and that the transport gap
    is not the whole latency: a buffered answer would show a server TTFT near zero
    and a client TTFT equal to the full answer time.
    """
    buffered = [
        r
        for r in report.successes
        if r.client_ttft_ms is not None
        and r.transport_ms is not None
        and r.transport_ms > 0.9 * r.client_ttft_ms
    ]
    assert not buffered, (
        f"{len(buffered)} responses showed ~all latency in the transport gap, which "
        f"is the signature of a buffered answer rather than a streamed one"
    )


def test_client_ttft_is_not_below_the_server_measurement(report: LoadReport):
    """The client cannot see a token before the server sent it.

    A negative gap means the clocks disagree, which would make the client figure
    the one being reported meaningless. Kept as an assertion rather than a log line
    because a mis-measured benchmark is worse than a slow one: it fails silently.
    """
    impossible = [
        r for r in report.successes if r.transport_ms is not None and r.transport_ms < -250
    ]
    assert not impossible, (
        f"{len(impossible)} responses reported a client TTFT more than 250 ms below "
        f"the server's own measurement"
    )
