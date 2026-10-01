# Performance report (implementation.md 5.2, NFR-1)

Replaces the indicative allocation in `architecture.md` §7.4 with measurement. §7.4
says of its own table that these are "starting estimates to be replaced by
measurement, not derived numbers" — this is that measurement.

## What was measured, and what it is worth

Two different things, and the distinction decides how much any of this can be
trusted:

| Measurement | What it includes | What it excludes |
| --- | --- | --- |
| `scripts/profile_stages.py` (below) | The work each retrieval stage does | Concurrency, generation |
| `tests/load/` (`docs/load_test_results.json`) | The full HTTP + SSE path under 50 concurrent users | Real generation |

**Neither includes a real language model.** The suite runs the offline provider, so
generation is effectively free. §7.4 budgets LLM time-to-first-token at 3,500 ms and
calls it the dominant term; that term is absent from every number in this document.
What is measured here is the application's own overhead — which turns out to be
small — and the honest conclusion follows in [§4](#4-what-this-cannot-tell-you).

Reproduce with:

```
.venv\Scripts\python.exe scripts\profile_stages.py --iterations 200
.venv\Scripts\python.exe -m pytest -m load tests/load/ -v
```

## 1. Retrieval stage profile

200 iterations after 20 warm-up requests, in-process, SQLite vector store, 40-document
corpus (107 chunks). Full data in `docs/perf_stages.json`.

| Stage | §7.4 budget | p50 | p95 | max | Verdict |
| --- | --- | --- | --- | --- | --- |
| Vector search | 150 ms | 15.0 | 19.5 | 25.2 | within |
| Keyword search | 150 ms | 15.1 | 19.1 | 29.0 | within |
| Rerank | 300 ms | 10.3 | 12.8 | 18.7 | within |
| Fusion | (in rerank budget) | 0.1 | 0.1 | 0.2 | within |
| Query rewrite | 400 ms | 0.0 | 0.0 | 0.1 | within |
| Threshold | 50 ms | 0.0 | 0.0 | 0.0 | within |
| Context budget | 50 ms | 0.0 | 0.0 | 0.0 | within |
| **Retrieval total** | ~1,000 ms | **40.3** | **50.6** | 66.2 | **within, ~20× under** |

Three findings worth acting on:

**Retrieval is not the bottleneck, by an order of magnitude.** The whole retrieval
pipeline costs 50.6 ms at p95 against roughly 1,000 ms of budget across §7.4's
non-LLM stages. Any future work to speed up TTFT should not start here.

**Query rewrite is 0.0 ms because the cheap path is working.** §7.4 budgets 400 ms
and warns that the rewrite is "an LLM call on the critical path; budget it and make
it skippable". The rule-based anaphora resolution resolves every query in this corpus
without an LLM call, so the expensive path is never taken. This is the design
working as intended, and it is load-bearing: 400 ms is 8% of the entire TTFT budget
spent on a stage that in practice costs nothing.

**Rerank is cheap enough that §7.4's stated first thing to cut is not needed.**
§7.4 says reranking "is the one stage whose budget is genuinely a trade: 300 ms buys
a large quality gain, and it is the first thing to cut if TTFT misses." At 12.8 ms
p95 it is not a candidate for cutting on these measurements.

### Caveat: SQLite is not pgvector

The vector search above is `SqliteVectorStore`, an exact O(n) Python cosine scan over
107 chunks. Production is required to use pgvector (`Settings.validate_production`
rejects SQLite outside local/test), which is an ANN index. Two consequences, in
opposite directions:

- These figures are a **floor**, not a prediction. pgvector will be faster at scale.
- The comparison is not free. SQLite's O(n) scan is pure Python and holds the GIL,
  so its 15 ms is disproportionately expensive under concurrency. pgvector does its
  work in Postgres, off the Python process. The concurrency behaviour in §2 will
  therefore be **better** in production than these numbers suggest.

## 2. Load behaviour, 50 concurrent users

Full data in `docs/load_test_results.json`. Real uvicorn process, real SSE, 50 VUs
issuing 100 requests, rate limiting enabled with a distinct client identity per VU.

| Metric | Value | Budget |
| --- | --- | --- |
| Requests / failures / truncations | 100 / 0 / 0 | 0 failures |
| Server-observed peak in-flight | 54 | ≥ 50 |
| Client TTFT p50 / p95 / p99 | 1,057.0 / 1,733.5 / 1,918.1 ms | p95 ≤ 5,000 ms |
| **Answered-only TTFT p95** | **1,806.0 ms** | **p95 ≤ 5,000 ms** |
| Refused-only TTFT p95 | 1,725.9 ms | — |
| Server-reported TTFT p95 | 1,437.9 ms | — |
| Transport gap p95 | 455.7 ms | — |
| Full answer p95 | 2,998.6 ms | ≤ 20,000 ms |
| Throughput | 22.92 req/s | — |

**NFR-1 passes with 2.8× headroom**, and the gate is asserted on answered queries
only (§3 explains why that distinction is load-bearing).

The transport gap — client-observed minus server-reported TTFT — is ~450 ms at p95.
This is the number a server-side-only metric cannot see: serialization, socket
buffering, and anything a reverse proxy adds. It is ~9% of the budget here with no
proxy in the path, which is the finding: a deployment that adds a buffering
misconfiguration has roughly 450 ms of slack to lose before NFR-1 is at risk, and
this harness would show it.

Percentiles move by a couple of hundred milliseconds between runs, so treat the
figures above as one run rather than a precise constant. `docs/load_test_results.json`
is rewritten by every load run and is the authoritative copy of the most recent one;
if the two ever disagree, the JSON is the measurement and this table is the summary.

## 3. Why answered-only is the gate

The 10 load questions produce a 30% abstention rate. A refusal short-circuits before
generation (`architecture.md` §3.3) and streams a canned sentence, so it never pays
TTFT cost. Averaging refusals into the sample would report p95 ≈ 1,862 ms while
understating what a user who got a real answer waits.

Both distributions are in the artifact so the difference is visible rather than
asserted. On the recorded run the two are within ~80 ms of each other (1,806 vs
1,726 ms) because generation is free offline; against a real model the gap would
widen, since only answered queries would carry the 3,500 ms LLM term.

## 4. What this cannot tell you

**The 5-second target is not validated, and cannot be with an offline provider.**

The arithmetic: measured application overhead is ~51 ms of retrieval plus ~180 ms of
HTTP/SSE/gateway overhead. Add §7.4's 3,500 ms LLM budget and the total is ~3,731 ms
against a 5,000 ms target — roughly 1.27 s of headroom, or 25%.

That headroom is real but modest, and it is entirely consumed by the one term not
measured. Concretely, the offline provider reports TTFT in *microseconds*; a real
provider's p95 first-token latency is the number that decides whether NFR-1 holds. If
it exceeds ~4.8 s, no amount of application tuning recovers the target.

**Therefore: re-run `tests/load/` against the real provider before treating NFR-1 as
met.** `scripts/bench_ttft.py` already carries this warning. It is restated here
because a green load test on the offline provider is the most likely way for this
project to believe something untrue about its own latency.

## 5. Saturation: where this deployment tops out

A single 50-VU point cannot distinguish "comfortably within budget" from "saturated,
with the numbers saved by having only just enough users". `scripts/load_sweep.py`
sweeps the load to find out. Full data in `docs/load_sweep_results.json`.

| VUs | TTFT p50 | TTFT p95 | Throughput | Failed | Truncated |
| --- | --- | --- | --- | --- | --- |
| 1 | 32.6 | 33.4 | 27.56 | 0 | 0 |
| 5 | 129.1 | 192.4 | 23.51 | 0 | 0 |
| 10 | 128.7 | 323.9 | 24.90 | 0 | 0 |
| 20 | 409.6 | 764.3 | 23.75 | 0 | 0 |
| 30 | 465.5 | 1,088.2 | 24.05 | 0 | 0 |
| 40 | 638.6 | 1,548.9 | 24.23 | 0 | 0 |
| **50** | **919.3** | **1,807.3** | **24.32** | **0** | **0** |
| 75 | 1,724.8 | 2,708.3 | 25.20 | 0 | 0 |
| 100 | 2,515.6 | 3,653.3 | 24.43 | 0 | 0 |

Throughput is flat at ~24 req/s from 1 VU to 100 VUs while latency grows roughly
linearly. Flat throughput under rising concurrency is the signature of a saturated
single process: requests queue, and each additional user adds their wait to everyone
else's. It is *not* a capacity limit reached at 50 users — 50 is nowhere near the
edge, and the 5,000 ms budget is not breached anywhere in this sweep, including at
twice the specified load.

Two limits follow, and they are different:

- **~24 req/s is the single-process ceiling** on this hardware with SQLite. It is set
  by CPU and GIL contention in the Python vector scan, and it would rise with
  pgvector (§1).
- **This ceiling is not the product's real ceiling.** With a 3,500 ms LLM term, a
  request spends almost all of its time waiting on a network call rather than
  consuming CPU, so a real deployment would be I/O-bound and could serve far more
  concurrent users per process. The 40-token anyio threadpool that bounds
  concurrent *CPU* work would bind before the CPU does.

So: scale horizontally, and do not read 24 req/s as a capacity plan.

## 6. A product bug this profiling found

Sweeping beyond 50 VUs surfaced a defect that the 50-VU gate did not, and which the
original harness *reported as passing*.

`persist()` writes the `QueryLog` row and runs after the last token has been flushed
to the client. When it raised — SQLite write contention under concurrency — the
exception was not an `AppError`, so it escaped the `except AppError` handler and
became an unhandled ASGI exception. The stream ended with no `done` and no `error`:
the user had the complete answer and then lost the connection.

Two things were wrong, and both are now fixed and tested (`app/api/chat.py`):

1. **A logging failure destroyed a delivered answer.** A lost telemetry row is not
   worth truncating text the user has already read. The stream now completes with a
   null `query_id` and the loss is logged. The one visible consequence is honest:
   FR-29 feedback buttons need a query id, so that turn has none.
2. **The load driver scored it as a success.** The driver treated "a token arrived
   and no `error` event" as a pass, so a truncated stream inflated the p95 and
   `failed` stayed at 0. The driver now requires a `done` event and reports
   truncations separately — a harness that cannot detect a broken response produces
   a confident, green, wrong number.

The driver fix is covered by `tests/load/test_harness.py`, which feeds synthetic SSE
bodies to it. The truncated-stream case is asserted directly: this exact shape must
count as a failure, or the next one will not be caught.

Post-fix, every point in the sweep above shows 0 failures and 0 truncations.

## 7. Recommendations

1. **Re-run the load suite against the real provider.** This is the only outstanding
   item for NFR-1, and it is the one that matters. Everything else here is
   application overhead that is already small.
2. **Keep SQLite out of any capacity planning** (§1). It inflates CPU cost and
   understates what pgvector will do.
3. **Do not spend effort on retrieval latency** (§1). 50.6 ms at p95, ~20× under
   budget.
4. **Watch the transport gap as a deployment canary** (§2). 443 ms p95 with no proxy
   in the path is the slack a reverse proxy can consume.
5. **If the provider's first-token p95 approaches 4.8 s**, NFR-1 fails on the LLM
   term and the only remaining lever is §7.4's stated one: cut rerank. It is cheap
   to cut — 12.8 ms — but it is the largest quality regression available, so it
   should be a decision, not a reflex.
