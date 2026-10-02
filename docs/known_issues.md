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

## 1. Mixed-dimension vectors (health probe closed; this corpus is consistent)

**Severity: closed on the current local `rag.db`.**

`/health` 503s when any live chunk's stored width disagrees with `EMBEDDING_DIM`.
A check on this checkout reported `mismatched_dims = 0` and `308/308 live chunks`.
If a future ingest writes the wrong width, the probe fails rather than returning `ok`.
Re-embed with `scripts/reembed_chunks.py` if it happens again.


---

## 2. Re-embedding a hash-vector corpus

**Severity: closed on this checkout's serving DB. Still the rule for a clone.**

This machine's `rag.db` is uniformly 384-dim. A fresh clone that still has
`fake-embed-v1` / mixed-width rows must re-embed with `EMBEDDING_PROVIDER=huggingface`
and a valid `HF_TOKEN` via `scripts/reembed_chunks.py`. Do not edit vectors in place.


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
| `NODE_VERSION=22` | **Pinned.** `web/package.json` `engines.node` is `>=22`. |
| `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` | **Required in production** by `validate_production()` when not using fake. |
| `postgresMajorVersion: 16` | **Choice.** The code only requires the `vector` extension. |
| `NEXT_PUBLIC_API_BASE` | **Build-time.** Runtime-only does nothing; `scripts/check_launch.py` cannot see the Next bundle. |
| `API_TOKEN` | **Required in production.** Listed in `render.yaml` as `sync: false`. |

`PYTHON_VERSION=3.12` is also a judgement call, though a better-supported one:
`pyproject.toml` says `requires-python = ">=3.11"`, while its own mypy configuration
documents that 3.11 fails on numpy's bundled stubs and that raising `requires-python` is
"a project decision." Pin 3.12 to match what the project actually develops against.

---

## 8. Stale vitest failure log

**Severity: closed.** `web/chatfail.txt` is gitignored (`*fail*.txt`) and is not
tracked. Failing-run captures belong in CI logs, not the tree.


---

## Verified healthy

So the list above is not read as "this does not work":

- 619 backend tests pass, 0 failures, 3 skips (POSIX-only). 10 load tests pass.
  76 web tests pass; lint, typecheck and build clean.
- `ruff` and `mypy` clean across `app tests scripts alembic`.
- `/health` responds and correctly reports `ok` on a **consistent** corpus.
- `alembic` has a single head (`0003_conversations`) with no branch or missing revision.
- `app.main:app` imports as a FastAPI ASGI instance under production settings.
- Retrieval works end to end against the local corpus.
- `rag.db` on this checkout is uniformly 384-dim (`mismatched_dims = 0`).
- The live `GROQ_API_KEY` in `.env` has **never** been committed. `.env` is gitignored
  and untracked; `.env.example` is the tracked, secret-free copy.
- The `gsk_` strings in `tests/providers/test_hosted_providers.py` are fixtures
  (`gsk_secret`, `gsk_x`, `gsk_ok`), not credentials.
