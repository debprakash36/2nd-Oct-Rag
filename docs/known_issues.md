# Known issues

Everything here was reproduced against this checkout, not inferred. Each entry says how
to reproduce it and what the evidence was. Where something is a deliberate design
decision rather than a defect, it says so.

Severity is about impact on a deployment, not about how hard the code is to change.

- **Blocking** — will fail a deploy, lose data, or leave the service reporting healthy
  when it cannot answer.
- **Degraded** — works, but reports something untrue or refuses a legitimate config.
- **Incomplete** — the feature is honest about being unfinished.

---

## 1. `/health` reports "ok" on a corpus retrieval cannot search

**Severity: blocking. This is the most dangerous item on this page.**

`rag.db` contains 11 chunks whose stored vector is 64-dimensional while the query
embedding is 384. Any retrieval against it raises:

```
ValueError: embedding dimension mismatch for chunk 035b17089843476b97ef5121c9553919:
stored 64, query 384
```

The same query against the clean copy returns 3 candidates and does not abstain.

`/health` on that same `rag.db` returns:

```json
{"status":"ok","checks":{"database":"ok","vector_store":"ok","keyword_index":"ok",
 "vector_index":"308/308 live chunks","environment":"local","embedding_dim":384}}
```

`status: ok`, and `308/308` — implying every chunk is retrievable when 11 cannot be.

**Why it happens.** `measure_divergence()` in `app/retrieval/vector_store.py` compares
only *counts*:

```python
sql_chunks = live_chunk_count(session)
return IndexDivergence(sql_chunks=sql_chunks, store_chunks=store.count())
```

`live_chunk_count()` filters on document state and never inspects vector dimension. The
SQL store returns the same 308 the count query sees, so the two agree and divergence is
reported as zero. A corpus can be uniformly wrong in a way a count cannot detect.

This defeats the stated purpose of the check. `app/api/health.py` documents the probe as
existing because "the store answered a count query" and "the store can return results"
are different questions — but dimension is a third question that neither answers.

**Repro.**

```powershell
$env:DATABASE_URL="sqlite:///rag.db"; $env:VECTOR_STORE="sqlite"; $env:EMBEDDING_DIM="384"
python -c "from app.db.session import get_session_factory; from app.core.config import get_settings; from app.retrieval.retriever import Retriever; s=get_settings(); sf=get_session_factory(s); sess=sf(); Retriever(sess).retrieve('escalation policy')"
# -> ValueError: embedding dimension mismatch
```

**Fix.** Have the health probe compare a stored vector's dimension against
`settings.embedding_dim`, and treat a mismatch as `is_empty`-class failure — unhealthy,
not merely stale. A count cannot stand in for this.

Worth noting what the test suite already covers, because it shows the gap is narrow and
specific rather than general: dimension mismatch **is** tested at the store level
(`tests/retrieval/test_vector_store.py`, `test_chroma_store.py` — "fail on dimension
mismatch, never score silently wrong"), and `/health` is exercised incidentally in
`test_cors.py` and `test_admin_documents.py`. But nothing asserts that a dimension
mismatch makes `/health` unhealthy, and every divergence test in
`test_vector_store_selection.py` moves a *count*. A regression test that seeds one
short-vector chunk and asserts a 503 would close this.

**Relation to item 2.** Fixing the data (item 2) removes the current instance. It does
not remove the blind spot, so a future bad ingest can recreate this silently.

---

## 2. `rag.db` holds 11 short-vector rows and cannot be repaired in place

**Severity: blocking for local work. Not urgent in production — production is Postgres.**

Confirmed state: 124 documents, 308 chunks, of which **297 are 384-dim and 11 are
64-dim**, all `embedding_model = fake-embed-v1`.

The repair is **blocked on an API key**, not on anything else:

- `HF_TOKEN` is empty in `.env`.
- Re-embedding requires `EMBEDDING_PROVIDER=huggingface` and a valid token.
- `make chroma-sync` and `pytest -m indexcheck` were deliberately **not** run, because
  they would operate on this corpus.

`rag.db.pre_purge_backup` is retained and gitignored.

A clean copy exists at `rag_eval.db` (113 documents, 297 chunks) with the 11 rows
removed. Retrieval, the eval harness, and the threshold re-sweep all run against it.
**It is a derived artifact and is gitignored — anyone else cloning gets a fresh
`rag.db` with the same 11 bad rows.**

This is documented at length in `docs/investigation_duplicates.md`; it is repeated here
because item 1 is a consequence of it and neither is discoverable from the other.

**Do not** try to fix this by editing `rag.db` directly. The vectors are wrong, not the
counts; a row edit would make the count agree while leaving retrieval broken, which is
strictly worse than the current honest failure.

---

## 3. No authentication on any endpoint

**Severity: blocking for a public deployment.**

Every endpoint is unauthenticated, including `/admin/documents`, which uploads **and
deletes** documents. A public URL is an open delete button.

`docs/implementation.md` records this as a deliberate scope decision for local use. It
becomes a vulnerability the moment the service is reachable by anyone else.

**If deploying to Render:** set services to **Private**, or add authorization first.
`allow_credentials=False` in the CORS setup is not a mitigation — it is unrelated to
authentication, and the comment next to it says so.

---

## 4. Render first deploy fails unless pgvector is enabled on the database

**Severity: blocking, but only on first deploy.**

`alembic/env.py` calls `_assert_pgvector()` and raises:

```
RuntimeError: pgvector is not installed in the target database.
Run CREATE EXTENSION vector; ...
```

pgvector is not in Render's default Postgres image. Enable it in the database's
settings, or run `CREATE EXTENSION vector;` as owner, before the first
`alembic upgrade head`.

Two further first-deploy traps, both verified:

- **`ENVIRONMENT` must be `production`.** `create_all()` in the app lifespan runs only
  when it is `local` or `test`, so a misconfigured deploy starts against an empty schema
  and every request fails with a missing-table error. Setting `production` also enables
  the `validate_production()` checks.
- **`LOG_LEVEL` is case-sensitive.** It is `Literal["DEBUG","INFO","WARNING","ERROR"]`,
  not a case-insensitive enum. `LOG_LEVEL=info` — the obvious spelling — fails at import
  with a pydantic `literal_error` naming `log_level`. Use `INFO`.

---

## 5. Phase 6 cannot be closed: no real pilot traffic

**Severity: incomplete, and not fixable by writing more code.**

All four traffic metrics are `UNMEASURED`. `query_logs` holds 6 rows across 2 distinct
queries — local test residue, not a pilot. `scripts/pilot_metrics.py` reports them
`UNMEASURED` rather than passing them, which is correct and is the intended behaviour.

`scripts/content_gaps.py` has nothing to classify and says so, noting that a near-zero
refusal rate is itself the failure mode PRD §8.2 warns about.

`scripts/retune_threshold.py` has one recorded baseline snapshot
(`docs/eval/threshold_history.jsonl`). One snapshot is not a trend. Activity 1 asks for a
**monthly** re-sweep of **real** traffic, and the eval-set distribution is not that.

**Consequence:** the Phase 6 exit gate is genuinely outstanding. It needs an audience,
not a script.

---

## 6. Threshold and model defaults are fake in every checked-in path

**Severity: degraded if deployed accidentally.**

`EMBEDDING_PROVIDER`, `GENERATION_PROVIDER`, and `EMBEDDING_MODEL` all default to fake
values in `app/core/config.py`:

```python
embedding_provider: Literal["fake", "huggingface"] = "fake"
embedding_model: str = "fake-embed-v1"
generation_provider: Literal["fake", "groq"] = "fake"
```

Nothing rejects `fake` when `ENVIRONMENT=production`. A deploy that forgets
`EMBEDDING_PROVIDER` and `GENERATION_PROVIDER` starts, answers from fixtures, and
produces quality numbers that look real. The recorded baseline recall of 0.892 is from
fake embeddings on a 113-document synthetic corpus and is **not** a production-quality
measurement.

`validate_production()` already blocks sqlite in production. Rejecting fake providers
there is the same shape of check and would close this.

---

## 7. Deployment values that could not be verified from the repo

**Severity: incomplete — these are unknowns, not confirmed defects.**

| Value | Status |
|---|---|
| `NODE_VERSION=22` | **Guess.** `web/package.json` declares no `engines` field, so nothing in the repo constrains it. `npm run build` was confirmed working on Node 24 locally. |
| `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` | **Unverified.** The repo default is `fake-embed-v1`. This model name comes from the 384-dim decision in `docs/architecture.md`, not from code. |
| `postgresMajorVersion: 16` | **Choice.** The code only requires the `vector` extension; no version is pinned anywhere. |
| `Pre-Deploy Command` | **Plan-dependent.** Render exposes this only on paid plans. On free, keep `alembic upgrade head` in the Start Command. |
| `npm start -- --hostname 0.0.0.0 --port $PORT` | **Untested.** `next start` already binds `0.0.0.0` by default; plain `npm start` works. |

`PYTHON_VERSION=3.12` is also a judgement call, though a better-supported one:
`pyproject.toml` says `requires-python = ">=3.11"`, while its own mypy configuration
documents that 3.11 fails on numpy's bundled stubs and that raising `requires-python` is
"a project decision." Pin 3.12 to match what the project actually develops against.

---

## 8. A stale test-failure log is committed to the repo

**Severity: degraded — noise and a misleading signal, not a secret.**

`web/chatfail.txt` is a 13 KB vitest failure log from an earlier debugging session, and
it is **tracked in git and not ignored**. It records two failures in
`web/app/chat/page.test.tsx`:

```
× refuses to send an over-long message before any request (FR-34)
× re-enables the composer after the stream ends
```

Both of those tests **pass now** — the full web suite is 76/76 green. So the committed
artifact asserts failures that no longer exist. Anyone auditing the repo, or any tooling
that greps for failures, sees this file and reasonably concludes the frontend is broken.

It contains no credentials and no absolute paths, so it is not a leak. It is dead
weight that is actively misleading.

**Fix:** `git rm --cached web/chatfail.txt` and add it to `.gitignore`.

Build outputs and caches are already handled correctly — `web/.next/`,
`web/tsconfig.tsbuildinfo`, `pytest.out`, `html.html`, `new.html`, and the
`.mypy_cache`/`.pytest_cache`/`.ruff_cache` directories are all ignored and untracked.
That was checked rather than assumed.

---

## Verified healthy

So the list above is not read as "this does not work":

- 619 backend tests pass, 0 failures, 3 skips (POSIX-only). 10 load tests pass.
  76 web tests pass; lint, typecheck and build clean.
- `ruff` and `mypy` clean across `app tests scripts alembic`.
- `/health` responds and correctly reports `ok` on a **consistent** corpus.
- `alembic` has a single head (`0003_conversations`) with no branch or missing revision.
- `app.main:app` imports as a FastAPI ASGI instance under production settings.
- Retrieval works end to end against `rag_eval.db` (3 candidates, no spurious abstain).
- `rag.db` is unmodified by all of the above; its 11 bad rows are unchanged.
- The live `GROQ_API_KEY` in `.env` has **never** been committed. `.env` is gitignored
  and untracked; `.env.example` is the tracked, secret-free copy.
- The `gsk_` strings in `tests/providers/test_hosted_providers.py` are fixtures
  (`gsk_secret`, `gsk_x`, `gsk_ok`), not credentials.
