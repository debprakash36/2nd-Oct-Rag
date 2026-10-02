# Runbook (implementation.md 5.8, NFR-2)

Operational procedures for the failure modes this system actually has. Each entry
states the symptom, how to confirm it, and what to do — in that order, because during
an incident the symptom is what you have.

The governing rule, from NFR-2 and `architecture.md` §7.1: **never serve a partial
index as if it were whole.** A degraded-but-200 response is worse than a 503, because
it looks like success to both the load balancer and the user while quietly returning
fewer sources than the corpus contains.

---

## 1. Retrieval store unavailable

### Symptom

`/health` returns **503** with `"status": "degraded"`, and one or both of
`vector_store` / `keyword_index` failing in `checks`.

`POST /chat/stream` returns an SSE `error` event with code
`retrieval_unavailable` rather than an answer.

### Why it is a 503 and not an answer

Retrieval is the only source of grounded text in this system (FR-13, FR-14). With the
store down, the system has no evidence — and an ungrounded answer from a RAG chatbot
is a confident fabrication, which is the specific failure FR-14 and the refusal path
exist to prevent.

The distinction the code maintains, and the reason it matters operationally:

| Condition | Response | Meaning |
| --- | --- | --- |
| Store reachable, top score below threshold | **200**, refusal text, `abstained: true` | Working as designed. The corpus genuinely has no answer. |
| Store unreachable | **503**, SSE `error`, `retrieval_unavailable` | We cannot tell whether there is an answer. |

These look similar to a user ("the bot said it couldn't help") and mean opposite
things. Before treating a refusal as a bug, check the status code.

### Diagnosis

```bash
curl -s localhost:8000/health | python -m json.tool
```

- `vector_store: fail` — the vector backend (pgvector/Chroma) is unreachable.
- `keyword_index: fail` — the `chunk_terms` table is unreachable. This usually means
  the database itself is down, since both checks hit the same database; if both fail
  together, go to §2.
- Either failing while the process is otherwise healthy usually means a **pool
  exhaustion** or a **hung connection**, not a down database: the process can still
  serve `/health`'s own request.

Confirm from the logs, which carry `trace_id` end to end (§7.2):

```
"msg": "chat stream error", "detail": ...      # typed AppError
"exc": "... StoreUnavailableError ..."          # underlying driver error
```

### Remediation

1. **Confirm the database is up** and accepting connections:
   `pg_isready` (or the equivalent). If not, see §2.
2. **If only the vector store failed**, check that backend's own status. A Chroma
   collection out of sync with the SQL schema produces this — the health check counts
   rows, so an empty or unsynced collection fails the check even though SQL is fine.
3. **Do not bypass the check.** Serving 200 while degraded is the failure NFR-2
   exists to prevent. The check is a count query against `chunks` / `chunk_terms`;
   it is cheap and it is the only thing distinguishing "no answer" from "no data".
4. **After recovery**, confirm `/health` is 200 and that answer quality is restored —
   a store that came back empty is healthy by the status code and wrong in practice.
   Spot-check a question you know has an answer in the corpus, and confirm sources are
   returned rather than a refusal.

### Related alert

`RagTTFTNoData` (`ops/alerts.md`) covers the related case where the process is up but
no queries are arriving at all.

---

## 2. Database down

### Symptom

`/health` 503 with both `vector_store` and `keyword_index` failing, or the process
fails to start with a connection error.

### Remediation

1. Restore the database. This system does not manage its own schema migrations at
   runtime; confirm the expected schema is present after recovery.
2. Confirm the process reconnects. SQLAlchemy pools may hold dead connections; the
   first request after recovery can fail while the pool recycles. If it does not
   recover, restart the process — there is no in-process reconnect logic beyond pool
   recycling.
3. `DOCUMENT_URL` / `DATABASE_URL` are read at startup. A changed connection string
   needs a restart, not a reload.

**Data loss expectation:** in-flight chat queries are lost. The streaming path writes
the `QueryLog` row *after* the answer is delivered, so a query interrupted by the
outage has no log row. This is a deliberate ordering choice (the answer must not wait
on telemetry) and is visible in the load-test harness as a `truncated` count — see
`app/api/chat.py::_safe_persist` and the note in `docs/perf_report.md` §6.

---

## 3. Query-log write failures under load

### Symptom

Answers stream to the user but arrive **truncated** — text appears and the stream
ends with no `done` event. Or: `failed to record query log` in the logs, and answers
complete with a `null` `query_id`.

### What is happening

The `QueryLog` insert happens after the last token is flushed. Under write contention
(for example a locked SQLite database, which the 50-VU load sweep reproduced
repeatedly) that insert can fail. The system then completes the stream with a null
`query_id` rather than truncating an answer the user has already read.

### Impact

- **Users see their full answer.** This is the designed behaviour.
- **FR-29 feedback buttons are absent** for those turns, because feedback attaches to
  a query id and there is none.
- **Those queries are missing from analytics**, so dashboards and the TTFT alert are
  under-reporting during the incident. This matters: low volume is not the same as
  low latency, and a p95 over a thinned sample is not a p95.

### Diagnosis

Search the logs for `failed to record query log`. The `delivered` field distinguishes
the harmless case (answer already sent) from the case where nothing was sent and the
error is real.

### Remediation

1. This is usually **database write contention**, not a bug. Check for long-running
   transactions, a stuck migration, or a writer on the same tables.
2. SQLite is not for production (`Settings.validate_production` rejects it outside
   local/test), so this specific cause should not occur in a conforming deployment.
3. If the rate of these is non-trivial, treat it as a data-completeness incident, not
   only an availability one — the improvement loop (FR-30) depends on these rows.

---

## 4. Extraction sandbox failures (ingestion)

### Symptom

Documents land in state `failed` with an `error_reason`. The ingest dashboard's
"Most recent failures" table shows the reason text.

### Common causes

| Symptom in `error_reason` | Cause | Action |
| --- | --- | --- |
| `sandbox produced no result for op='extract' (exit code N)` | Child crashed or was killed — memory cap, or a parser bug in a malformed file | Check file size and whether it reproduces on that file alone. A single file that always crashes is a parser bug worth an isolated report. |
| Timeout / exit code 124 | Document exceeded the CPU or wall-clock cap | Legitimate enforcement. Large PDFs need splitting upstream, not a higher cap. |
| `unsupported MIME type` | Format not in the extraction allow-list | Expected. Convert upstream. |

### Remediation

1. Group failures by `error_reason`. Repeated identical reasons are one systemic cause,
   not many documents.
2. The sandbox is the security boundary (NFR-5) — **do not raise the resource caps to
   make a failure go away** without understanding what the document was doing. That
   cap is what stops a malformed file from exhausting the host.
3. A document stuck in `extracting` or `indexing` after a process restart will not
   advance on its own; re-upload it.

### Platform caveat

On Windows the POSIX resource limits (CPU, address space, file size) are **not
enforced** — the sandbox still isolates the process and enforces timeout and
read-only filesystem, but not memory or CPU caps. If a Windows host ingests untrusted
files, do not treat it as equivalent to the Linux deployment described in
`architecture.md` §8.

---

## 5. Provider errors (generation)

### Symptom

SSE `error` events during generation, or elevated TTFT with no retrieval cost
increase.

### Diagnosis

- The offline provider returns instantly; real latency is provider latency. Read the
  TTFT alert triage order in `ops/alerts.md` — step one is "is it the model", and it
  is the right first question because generation dominates the budget.
- Rate-limit errors from the provider surface as `error` events; sustained rate
  limiting looks like latency, not like errors.

### Remediation

Check provider status and account limits before touching this application. The
generation path has no retry logic by design: a retry that fires mid-stream would
either duplicate text the user has already read or stall a partially-delivered answer.

---

## Quick reference

| Symptom | First check | Section |
| --- | --- | --- |
| `/health` 503 | Which of `vector_store` / `keyword_index` failed | §1 |
| Both failed | Is the database up? | §2 |
| Answers truncated, no `done` | `failed to record query log` in logs | §3 |
| Documents in `failed` | `error_reason` on the ingest dashboard | §4 |
| Slow answers | Provider latency first, then retrieval | §5 |
| "It says it can't help" | 200 (refusal, by design) or 503 (store down) | §1 |

---

## 6. Before a public URL

Local `/health` 200 is not a production clearance. SQLite, fake providers, and
localhost CORS are allowed when `ENVIRONMENT=local` and refused in staging/production.

```bash
python scripts/check_launch.py
```

A FAIL row must be fixed before the process is reachable from the internet.
`Phase 6 traffic gate` staying BLOCKED is expected until real users exist.

First cloud deploy also needs `CREATE EXTENSION vector;` on Postgres before
`alembic upgrade head` (`docs/known_issues.md` item 4), `API_TOKEN` set, and
`NEXT_PUBLIC_API_BASE` present at the **web build**, not only at runtime.

