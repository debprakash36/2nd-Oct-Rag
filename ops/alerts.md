# Alerts (implementation.md 5.7, NFR-8)

Two alerts are specified for day one, and they are chosen for a specific reason
(`architecture.md` §7.2): **both catch failures that are otherwise invisible.**

TTFT breaching its target announces itself — users complain. Strip-rate drift does
not. Nobody notices an answer that quietly lost two of its three citations; they just
trust it less and cannot say why. The second alert exists because the first class of
bug is loud and the second is silent.

Alert rules are expressed as SQL against `query_logs` and `documents`
(`app/db/models.py`). They are written to be portable, and each is annotated with what
it looks like when it is *wrong* — a rule nobody has seen fire during a real incident
is a rule that may be broken.

`query_path.json` (dashboards) carries the graphs these thresholds are read against.

---

## Alert 1 — TTFT p95 above 4 s

| | |
| --- | --- |
| **Severity** | Warning at 4 s, **critical at 5 s** |
| **Window** | 5 minutes, evaluated every minute |
| **Data** | `query_logs.ttft_ms` |
| **Rationale** | NFR-1 target is 5 s. The alert fires at 4 s so there is lead time to react before the user-visible target is breached — degrading is worth catching, breaching is not worth waiting for. |

**Expression**

```sql
SELECT
  $__timeGroup(created_at, '5m'),
  percentile_cont(0.95) WITHIN GROUP (ORDER BY ttft_ms) AS p95_ttft_ms
FROM query_logs
WHERE $__timeFilter(created_at)
  AND ttft_ms IS NOT NULL
GROUP BY 1
```

**Rules**

| Name | Condition for | For |
| --- | --- | --- |
| `RagTTFTP95Warning` | `p95_ttft_ms > 4000` | 5 min |
| `RagTTFTP95Critical` | `p95_ttft_ms > 5000` | 5 min |
| `RagTTFTNoData` | `count(*) = 0` over 15 min | 15 min |

**Why `NoData` is its own alert.** A p95 computed from two queries is noise, and a
rule that only fires on a breach will happily stay green all afternoon while the
service is down and returning no traffic at all. Silence is a distinct failure mode
from slowness and needs its own signal. See also the volume panel on the dashboard:
read p95 *next to* query count, because a low-latency p95 over six requests is not
evidence of health.

**Triage, in order**

1. **Is it the model?** Check the LLM provider's own first-token latency for the same
   window. `architecture.md` §7.4 budgets generation at 3,500 ms and calls it the
   dominant term; if the provider is slow, nothing in this application is the cause.
2. **Is it retrieval?** Compare against `docs/perf_stages.json` — measured retrieval
   is ~51 ms p95. If retrieval has grown by hundreds of milliseconds, something in the
   pipeline changed (index size, fetch_k, a reranker swap). Re-run
   `scripts/profile_stages.py`.
3. **Is it the transport?** The load harness reports a `transport_gap_ms`
   (client-observed minus server-reported TTFT). A large gap means buffering between
   the app and the user — a proxy, a missing `X-Accel-Buffering` handling, not the
   application.
4. **Is it saturation?** Run `scripts/load_sweep.py` against the deployment. Flat
   throughput with latency rising per VU means queueing, and the fix is capacity
   rather than tuning.

**The measurement caveat, repeated because it is the most likely wrong conclusion.**
All local numbers come from the offline provider, so generation is effectively free and
`docs/perf_report.md` shows measured p95 of ~1.9 s at 50 VUs. A real provider adds its
own first-token latency, budgeted at 3,500 ms. If the alert fires only after a real
provider is connected, that is the expected behaviour of the budget, not a regression.
Do not tune retrieval in response to it; §1 of the perf report shows retrieval is ~20×
under its allocation.

---

## Alert 2 — Citation strip-rate spike

| | |
| --- | --- |
| **Severity** | Warning |
| **Window** | 15 minutes, evaluated every 5 minutes |
| **Data** | `query_logs.citations_stripped` |
| **Rationale** | `architecture.md` §7.2: a strip-rate spike "indicates model or prompt drift, and it degrades answer quality *silently* from the user's perspective." |

**Expression**

```sql
SELECT
  $__timeGroup(created_at, '15m'),
  avg(citations_stripped) AS strip_mean,
  count(*) AS queries
FROM query_logs
WHERE $__timeFilter(created_at)
GROUP BY 1
```

**Rules**

| Name | Condition for | For |
| --- | --- | --- |
| `RagStripRateSpike` | `strip_mean > 0.2` **and** `queries >= 20` | 15 min |
| `RagStripRateZero` | `strip_mean = 0` **and** `queries >= 50` over 24 h | 30 min |

**Why the volume guard on the first rule.** The absolute mean is unstable at low
traffic: two queries that each lost a marker is a mean of 0.5, which would page
someone for nothing. Requiring 20 queries in the window means the signal is a rate
over a real sample. A ratio-based variant (`sum(stripped) / count(*)`) is equivalent
and is left to the deployment if per-window volume is spiky.

**Why `StripRateZero` is included.** The opposite failure is a real one and the
default rule cannot see it. If the strip count is exactly zero across 50+ queries,
either citations are no longer being validated — the validator stopped running, and
invalid markers are reaching users as if they were real — or the model has stopped
citing. Both are defects, and both are invisible in a panel that only draws strip rate
upward. This rule exists because an all-zero line is the most suspicious line on the
chart.

**Triage**

1. **Did the prompt change?** Strip rate responds directly to `PROMPT_VERSION`
   (`app/api/chat.py`, logged per query). A jump that lines up with a prompt deploy is
   the prompt, not the model.
2. **Did the model change?** `QueryLog.model` records the model per query. Group strip
   rate by model to separate a model rollout from a prompt rollout.
3. **Is it the corpus?** If ingestion changed, chunk boundaries moved and the
   assembler's passage numbering may no longer match what the model saw. Check the
   ingest dashboard alongside this one.
4. **Is it real or is it retrieval degrading?** A rise in refusals *and* strips
   together usually means fewer, poorer passages were retrieved — one upstream cause
   showing up in two panels. Read the refusal-rate panel before blaming the model.

---

## Not implemented, deliberately

- **No alert on refusal rate.** The PRD calls 10–30% a *healthy band*, and refusals
  are a health signal rather than an error. An alert threshold here would train
  operators to ignore it, and a system that refuses everything can look perfectly calm
  on a refusal-rate-only dashboard. Refusal rate is on the dashboard for a human to
  read in context. If it must be automated, alert on the *band being exited in either
  direction* — 0% and 100% are both broken.
- **No alert on ingest failure rate** in this file. Ingestion is an offline pipeline
  and a failure there is not user-visible; the ingest dashboard carries the failure
  table. Promoting it to an alert depends on whether ingestion is expected to complete
  within an SLO, which is a deployment decision not yet made.
- **No per-user or per-conversation alerting.** There is no authentication (see
  `docs/security_review.md`), so there is no user identity to alert on.

## Before trusting any of these

The rules have been checked against the schema, not fired in anger. Before relying on
them, confirm each fires by breaking it on purpose:

- Break TTFT: set the generation provider to a high-latency stub, confirm
  `RagTTFTP95Warning` fires and `Critical` does not.
- Break strip rate: lower the citation validator's tolerance in a test deployment,
  confirm `RagStripRateSpike` fires.
- Break the pipeline: stop the app, confirm `RagTTFTNoData` fires. **A degradation
  alert that has never been observed firing is not known to work.**
