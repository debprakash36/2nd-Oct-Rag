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

## 1. Mixed-dimension vectors in a local `rag.db` (health probe is fixed)

**Severity: blocking for that database file. The `/health` blind spot is closed.**

`measure_divergence(..., dim=settings.embedding_dim)` counts wrong-width vectors.
`/health` returns 503 with `status: degraded` when any live chunk's stored width
does not match `EMBEDDING_DIM`. Regression:
`tests/retrieval/test_vector_store_selection.py::TestHealthReportsAMalformedCorpus`.

A copy of `rag.db` can still hold mixed 64-dim / 384-dim rows (item 2). The probe
now reports that instead of `ok`. Re-embed with `scripts/reembed_chunks.py`.

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

## 3. Authentication is a shared secret, not per-user accounts

**Severity: blocking for a public deployment if `API_TOKEN` is left empty locally.
Staging/production refuse to boot without it.**

`AuthMiddleware` requires `Authorization: Bearer <API_TOKEN>` on every route except
`/health`, `/auth/status`, and `/auth/login`. The web UI stores the token in
`sessionStorage` and sends it on `apiFetch` and `/chat/stream`. An empty token still
leaves the API open so tests and a fresh checkout work; `validate_production()`
rejects that combination in staging and production.

This is a single operator secret, not user accounts or roles. Admin upload/delete
and the pilot console share it. Do not put the value in `NEXT_PUBLIC_*`.

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

## 6. Fake provider defaults are for local/test only

**Severity: closed at startup for staging/production. Still the local default.**

`EMBEDDING_PROVIDER`, `GENERATION_PROVIDER`, and `EMBEDDING_MODEL` still default to
fake values so tests and a fresh checkout stay offline. `validate_production()`
rejects `fake` providers, a `fake-*` embedding model name, sqlite, and an empty
`API_TOKEN` when `ENVIRONMENT` is staging or production.

The recorded baseline recall of 0.892 was from fake embeddings and is not a
production-quality measurement. On the same corpus with `all-MiniLM-L6-v2` it is
**0.8378**. The gate itself is still to be set from pilot traffic.

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
