# Investigation: duplicate `policy.md` rows and the broken `rag.db`

> **Status: fixes 1, 3, 4 and 5 landed. Fix 2 (the data repair) is BLOCKED on a
> missing API key. The 11 corrupt rows are still present and retrieval against
> `rag.db` still aborts.** See [Status of each fix](#status-of-each-fix) below.
>
> One claim in the original write-up was **wrong** and is corrected in §2: the
> 297/308 Chroma gap was reported as "time-based staleness, the documented
> behaviour". It was not. It was corruption, and `make chroma-sync` cannot repair it.

## Summary

Three distinct defects, stacked. They look like one problem ("why are there 11
duplicates?") but only the third is a dedup bug, and the one that actually breaks
retrieval is the first.

1. **`rag.db` is being written by the test suite.** Not a hypothetical — measured.
2. **Those test writes use `embedding_dim=64`, production uses 384.** One bad row
   aborts every retrieval query.
3. **Duplicates are flagged but still indexed and served.** `DocumentState.DUPLICATE`
   is a dead enum member — nothing ever assigns it.

## Status of each fix

| # | Fix | State | Verification |
| --- | --- | --- | --- |
| 1 | `get_session_factory()` refuses an implicit database; app lifespan and test `engine` fixture register the factory | **Landed** | `tests/test_isolation.py` — 6 tests, including a subprocess run of `test_conversations.py` asserting `rag.db` is byte-identical. Verified stable across a full suite run. |
| 3 | Duplicates transition to `DUPLICATE` and are not indexed | **Landed** | `tests/ingest/test_idempotency.py` — state, exclusion from `search()`, no chunks written, and `DUPLICATE` reachable in the transition table. |
| 4 | Dead code removed from `check_content_hash` | **Landed** | `app/ingest/dedupe.py:116-128` deleted. Suite green. |
| 5 | `DocumentState.DUPLICATE` added to the enum and transition table | **Landed** | `app/db/models.py`, `app/ingest/states.py`. Note: `web/lib/types.ts` already expected `"duplicate"`, so the frontend had been waiting on a state the backend could not produce. |
| 2 | Repair the 11 short-embedding rows | **BLOCKED** | See below. |

Gate at time of writing: ruff clean, mypy clean (56 files), 574 tests passed / 0
failed / 3 skipped (POSIX sandbox only), load suite 10 passed.

### Why fix 2 is blocked

The 11 rows must be re-embedded, not deleted — deleting all 11 would remove the only
copy of that document, since the root of the `duplicate_of` chain is itself 64-dim.
Re-embedding needs a real embedder, and the decision is now
`all-MiniLM-L6-v2` via the HuggingFace Inference API (`architecture.md` §13). That
requires `HF_TOKEN`, which is **empty**.

Two consequences worth stating plainly:

- **`make chroma-sync` cannot run yet.** `sync_chroma_index` validates every live
  chunk's width and raises `ValueError` on mismatch, by design, so a bad vector can
  never enter the index. With 11 short rows present it aborts. This is the same
  guard that caught the corruption.
- **A partial re-embed would be worse than none.** The 297 "good" vectors are
  hash-based, not model-based. Re-embedding only the 11 would leave two vector
  spaces in one index, where every cosine score is meaningless while still returning
  plausible neighbours. It has to be all 308 in one pass.

`scripts/purge_bad_dimension_chunks.py` is the repair tool. It is dry-run by default
and **refuses to delete any row that has no healthy copy of the same content hash**,
which is all 11 of them. It is a safety net, not the fix.

`rag.db.pre_purge_backup` is retained.

## 1. The test suite writes to the real database

`tests/conftest.py:3` claims: *"Every test runs against a fresh temp SQLite file."*
That invariant is not enforced, and it is currently false.

`app/db/session.py:33` caches a process-wide engine:

```python
def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(settings), ...)
    return _session_factory
```

`settings=None` falls through to `get_settings()` — the module-level singleton, which
is **not** overridden by the `client` fixture (which only overrides the FastAPI
dependency). Measured, in a fresh process with `ENVIRONMENT=test`:

```
engine url when called with NO settings: sqlite:///./rag.db
```

So `get_session_factory()` is safe only while another test has already warmed
`_engine` to a temp path. `settings_env` teardown calls `reset_engine()`, clearing the
cache between tests — so the first code path in a test that reaches for the global
factory without first touching `engine`/`settings_env` gets the real database.

`tests/api/test_conversations.py:26` and `:151` do exactly that:

```python
session = get_session_factory()()   # no settings argument
```

Observed effect: running `tests/api` as a directory added a document to `rag.db`
(123 → 124). Running each file individually added nothing — it is order-dependent,
which is why it survived this long.

**Why it was never noticed:** the writes are small, the suite is green, and `rag.db`
is a working-tree artifact nobody diffs. The 11 duplicate `policy.md` rows are
timestamps from 18:23 to 20:57 — spread across many test runs.

**Outcome — landed.** `get_session_factory()` now raises rather than falling through
to `get_settings()`, and both the app lifespan (`app/main.py`) and the test `engine`
fixture (`tests/conftest.py`) register the factory explicitly. The guard caught
**three further bare call sites** that the original bisect had missed:
`tests/api/test_cors.py` built a `TestClient` outside a context manager, so the
lifespan never ran and the factory was never registered. Those tests now use the
shared `client` fixture.

## 2. The 64-dimension pollution (this is what breaks retrieval)

`tests/conftest.py:37` pins `embedding_dim=64` deliberately, for determinism:

```python
settings = Settings(..., embedding_dim=64)
```

Production default is 384 (`app/core/config.py:40`). So every document a test
ingests through the fixture gets 64-wide vectors. When those land in the real
`rag.db`, the corpus is left with two incompatible vector widths:

```
embedding dimension histogram: {384: 297, 64: 11}
```

`SqliteVectorStore.search` validates width and raises, and the exception propagates
out of the parallel search stage — so **one malformed row fails every query**, not
just queries that would have matched it:

```
ValueError: embedding dimension mismatch for chunk 035b1708...: stored 64, query 384
  retriever.py:279  future.result()
  vector_store.py:399  raise ValueError
```

**Correction to what I told you earlier.** I previously reported the 297-vs-304
Chroma gap as "time-based staleness, the documented behaviour." That was wrong. The
297 vectors in Chroma are exactly the 297 at 384 dims. The 11 short rows are the gap,
and they are corruption, not lag. `make chroma-sync` would not fix this — it raises
on the dimension check by design.

The duplicate chain is intact, and the root document is itself 64-dim — uploaded
through the test path, so every copy inherited short vectors. There has never been a
healthy copy of this content:

```
cca705d7… state=LIVE dup_of=None            <- 18:23:55  <- the root, 64-dim
598a1333… state=LIVE dup_of=cca705d7        <- 18:24:53
02801d21… state=LIVE dup_of=598a1333        <- 18:59:56
…  8 more, all 1 chunk, all embed_len=1240
```

This is why deletion was rejected. The eval does not reference any of the 11 chunk
IDs, so deletion would not have moved the metrics — but "the metrics don't move" is
not the same as "nothing is lost", and it would have removed the document.

Verified consequence: with the 11 rows removed (on a **copy**; `rag.db` was not
modified), the full 200-question eval passes:

```
 fake embeddings — see caveat below, these are not gate results
 recall@10   0.892  (target >= 0.85 NOT ESTABLISHED)
 refusal     21.0%  (healthy band 10-30%)
 latency     mean 57.6 ms, p95 70.9 ms
```

> **The 0.892 here is not a gate result.** This run predates the re-embed, so it used
> `FakeEmbeddingProvider` — a SHA-256 hash bucket, not a sentence encoder. With
> `all-MiniLM-L6-v2` on the same corpus the figure is **0.8378**, below the 0.85 target;
> **0.8378 is the first real baseline**, and the gate is **to be set from pilot traffic**.
> The eval set is lexically biased (templates quote document titles and scopes
> verbatim), which flatters a bag-of-words vectoriser. These numbers remain valid as
> evidence that the 11 rows were irrelevant to scoring — which is what this section
> needed them for — and not as evidence about retrieval quality.

**Outcome — BLOCKED, not fixed.** The leak that caused this is closed, so no *new*
64-dim rows can appear. The 11 existing rows still abort every query against
`rag.db`. Repair requires a full re-embed with `all-MiniLM-L6-v2`, which needs
`HF_TOKEN` (currently empty). See [Why fix 2 is blocked](#why-fix-2-is-blocked).

## 3. Duplicates are flagged, then served anyway

`app/ingest/worker.py:217-226`:

```python
if verdict.verdict == HashVerdict.DUPLICATE:
    doc.duplicate_of = verdict.existing_doc_id
    log.warning("duplicate content detected", ...)
session.flush()

# --- embed and persist ---
transition(session, doc, DocumentState.EMBEDDING)
chunks = write_chunks(...)
write_indexes(session, doc)
finalize_document(session, doc)
transition(session, doc, DocumentState.LIVE)   # <- duplicate goes live
```

The verdict is recorded and execution falls straight through to embedding, indexing
and `LIVE`. `check_content_hash` correctly returns `DUPLICATE` — the classification
is right; nothing acts on it.

**Correction to the original write-up.** This section originally read
"`DocumentState.DUPLICATE` exists in the enum and is **never assigned anywhere**",
with a `grep` returning no matches. Both halves were wrong. The member did **not**
exist — the `DUPLICATE = "duplicate"` found during the original investigation was in
`app/ingest/chunk.py`, a different enum. It was added as part of this fix, and
`web/lib/types.ts` had been listing `"duplicate"` in its `DocumentState` union the
whole time: the frontend was typed against a state the backend could not produce.
The substantive finding stands unchanged — nothing was acting on a duplicate.

So at retrieval time a duplicate is indistinguishable from a genuine document. All 11
`policy.md` copies are `LIVE` and retrievable, and a query matching that content
returns the same passage up to 11 times — inflating the `sources` panel and spending
context budget on copies.

The comment at `worker.py:218-220` argues both sides and resolves it wrongly:

> "silently dropping looks like data loss, and silently ingesting two identical
> documents produces two identical answers. An admin decides (FR-4)."

"Two identical answers" is the reason *not* to index the duplicate, not a reason to
defer to an admin — and no admin is in the loop. FR-4 is satisfied by the flag, but
the retrieval consequence was never addressed.

**Outcome — landed.** The duplicate branch now transitions to `DUPLICATE` and returns
before embedding. `DUPLICATE` was added to `DocumentState` and to the transition
table (`chunking`/`extracting` → `duplicate` → `pending`/`deleted`, so an admin can
promote it). Retrieval filters on `live` in both the SQL scan and the derived ANN
projection, so no search-path code changed. This is a deliberate change from
`architecture.md` §4.1, which is now updated to record the `duplicate` state and the
flagging-versus-neutralising distinction.

Note the design change: §4.1 originally said "offer to skip", and the implementation
records the upload and makes it unretrievable. Both halves of "offer" are preserved —
the upload is acknowledged rather than dropped, and the row is visible in the admin
listing with its `duplicate_of` pointer.

## 4. Dead code

`app/ingest/dedupe.py:116-128` is unreachable. The function returns at line 114;
everything after is a leftover from an earlier single-row implementation, superseded
by the loop at lines 97-114. It is a plausible-looking duplicate of the real logic,
which is exactly the kind of thing that gets "fixed" by editing the wrong copy.

**Outcome — landed.** Removed.

---

# Fixes

Five changes. **1, 3, 4 and 5 are landed; 2 is blocked** on a missing API key. The
diffs below are kept as the record of what was approved and why. They are the
as-applied text except where a section notes a deviation made during implementation.

## Fix 1 — make the session factory refuse an implicit database

Stop the leak at the source rather than auditing every call site.

```diff
--- a/app/db/session.py
+++ b/app/db/session.py
@@
 def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
-    """Return the process-wide session factory."""
+    """Return the process-wide session factory.
+
+    `settings` may be omitted only when an engine is already cached. Falling
+    through to `get_settings()` when it is not is how the test suite came to
+    write into the real `rag.db`: the global singleton is not the same object
+    the `client` fixture overrides, so a no-argument call silently reached
+    `sqlite:///./rag.db` and left 64-wide embeddings in the corpus that
+    production reads at 384.
+
+    Raising is the correct failure. A test that reaches for the global factory
+    before warming `_engine` has a bug -- it is writing somewhere it did not
+    choose -- and the alternative is corrupting a file that no assertion
+    covers.
+    """
     global _session_factory
     if _session_factory is None:
+        if settings is None:
+            raise RuntimeError(
+                "get_session_factory() called without settings and no engine is "
+                "cached. Pass the test's settings explicitly, or request the "
+                "`engine`/`session` fixture so a temp engine is registered first. "
+                "Refusing here prevents tests writing to the configured database."
+            )
         _session_factory = sessionmaker(bind=get_engine(settings), expire_on_commit=False,
                                         future=True)
     return _session_factory
```

Then fix the two offenders:

```diff
--- a/tests/api/test_conversations.py
+++ b/tests/api/test_conversations.py
-def _add_turn(client: TestClient, conversation_id: str, role: str, content: str) -> None:
+def _add_turn(
+    session_factory, conversation_id: str, role: str, content: str
+) -> None:
     """Append a turn directly in the store, bypassing the streaming endpoint.
 
     Most of these tests are about the conversation API, and driving a full
     generation to obtain a turn would make them fail for retrieval reasons whenever
     the corpus or threshold changes.
+
+    Takes the session factory from the `session` fixture rather than calling
+    `get_session_factory()` with no argument, which resolved to the configured
+    database (rag.db) rather than the per-test temp file.
     """
     from app.db import conversation as store
-    from app.db.session import get_session_factory
-
-    session = get_session_factory()()
+
+    session = session_factory()
```

*(`session_factory` threaded in from a fixture that exposes the temp factory; the
exact fixture shape is the only part of this diff I'd want to confirm against the
call sites — there are four, in `_add_turn` and `test_turns_are_removed`.)*

## Fix 2 — repair `rag.db` and stop a bad row killing all retrieval

Two parts, because they address different problems.

**2a. One bad row must not abort every query.** Currently `search` raises on the
first wrong-width vector. A single corrupted row makes the whole system unavailable,
which is the wrong blast radius — the dimension guard is right to complain and wrong
to be fatal for unrelated queries.

```diff
--- a/app/retrieval/vector_store.py
+++ b/app/retrieval/vector_store.py
@@ class SqliteVectorStore:
             if len(embedding) != self._dim:
-                # Checked here rather than left to the store's own error so the message
-                # names the pinned config value, which is the actionable part. The
-                # existing test for this invariant asserts on "dimension mismatch".
-                raise ValueError(
-                    f"embedding dimension mismatch: query has {len(query_vector)}, "
-                    f"the collection is built for {self._dim}. The embedding model has "
-                    f"probably changed; the Chroma index must be rebuilt."
-                )
+                # Logged and skipped, not raised. This check exists to catch an
+                # embedding model or dimension changing under a populated index, and
+                # a 384-wide query against 64-wide stored vectors is
+                # unscoreable -- but so is only *that row*. Raising here took down
+                # every query in the system, because the exception propagated out of
+                # the parallel search stage and aborted the request before any
+                # result was assembled. One corrupt row is a data problem to
+                # surface loudly in logs and health; it is not grounds for the
+                # service to report that it cannot answer anything.
+                #
+                # `sync_chroma_index` still raises on the same condition: re-projecting
+                # is an explicit repair action, and half-writing a collection would
+                # be worse than refusing.
+                _LOG.warning(
+                    "skipping chunk with mismatched embedding dimension: "
+                    "chunk_id=%s stored=%d expected=%d. The index needs rebuilding "
+                    "with the configured embedding model.",
+                    chunk.chunk_id, len(embedding), self._dim,
+                )
+                continue
```

This needs the `continue` to sit inside the candidate loop, and a count surfaced so
"11 rows skipped" is visible rather than merely logged. I would also add
`skipped_dim_mismatch` to the `vector_index` health check so the corruption is
visible without reading logs.

**2b. Clean the existing data.** I have *not* done this — it mutates your database
and you should decide. Two options:

- **Delete the 11 documents** and their chunks. They are all `policy.md` with
  identical content hash `bfb6ee238238`, so a genuine copy of that content already
  exists among the 297 healthy rows and retrieval loses nothing.
- **Re-embed** them at 384. Preserves the rows, costs one embed pass.

I recommend deletion, and can script it as a dry-run-first migration.

## Fix 3 — make duplicates actually stop being served

This is the answer to your original question. The classification is already correct;
the fix is to act on it.

```diff
--- a/app/ingest/worker.py
+++ b/app/ingest/worker.py
     if verdict.verdict == HashVerdict.DUPLICATE:
-        # Recorded, not rejected: silently dropping looks like data loss, and
-        # silently ingesting two identical documents produces two identical
-        # answers. An admin decides (FR-4).
+        # Recorded, not rejected: the upload is acknowledged and the admin decides
+        # (FR-4). Silently dropping it would look like data loss from the uploader's
+        # side, and the admin list needs to show what arrived.
+        #
+        # But it must not also be *served*. Indexing it produced eleven identical
+        # `policy.md` chunks, all LIVE, all retrievable: a query matching that
+        # content returned the same passage up to eleven times, inflating the
+        # sources panel and spending the context budget on copies. Flagging is not
+        # the same as neutralising, and nothing downstream read the flag --
+        # `DocumentState.DUPLICATE` was never assigned by any code path.
+        #
+        # So the row is recorded and made unretrievable in one step. The text stays
+        # in the object store and the row keeps its `duplicate_of` pointer, so
+        # nothing is lost and an admin can promote it if the original is removed.
         doc.duplicate_of = verdict.existing_doc_id
         log.warning(
             "duplicate content detected",
             extra={"doc_id": doc.doc_id, "duplicate_of": verdict.existing_doc_id},
         )
+        transition(session, doc, DocumentState.DUPLICATE)
+        session.commit()
+        return IngestOutcome(
+            doc.doc_id,
+            DocumentState.DUPLICATE,
+            duplicate_of=verdict.existing_doc_id,
+            detail=f"duplicate of {verdict.existing_doc_id}; not indexed",
+        )
     session.flush()
```

Retrieval already filters on `Document.state == LIVE`
(`vector_store.py:148`, `sync_chroma_index:493`), so `DUPLICATE` is excluded from both
the SQL scan and the Chroma projection with no change to the search path. The upload
response already reports `duplicate_of` separately from accepted/rejected
(`admin_documents.py`), so the UI needs no change.

**This needs one design decision from you.** FR-4 says "let an admin decide." Two
readings:

- **(a) Not retrievable until an admin promotes it** (what the diff above does).
  Duplicate text never reaches a user. An admin who wants the copy can flip it to
  `LIVE` from the existing admin surface.
- **(b) Retrievable, as today**, and the 11 copies are accepted as the cost of not
  silently dropping uploads.

Option (a) was chosen. The retrieval consequence of serving eleven identical
documents is worse than the cost of one extra admin click, and (a) is the only
option in which `DocumentState.DUPLICATE` is ever actually used.

## Fix 5 — add the state itself (not in the original proposal)

Fix 3 could not work as written. `DocumentState` had no `DUPLICATE` member to assign
to, and `DocumentState.DUPLICATE` in the original investigation was a misread of an
unrelated enum in `app/ingest/chunk.py`.

```diff
--- a/app/db/models.py
+++ b/app/db/models.py
     LIVE = "live"
     FAILED = "failed"
     SUPERSEDED = "superseded"
     DISABLED = "disabled"
     DELETED = "deleted"
+    DUPLICATE = "duplicate"

--- a/app/ingest/states.py
+++ b/app/ingest/states.py
     DocumentState.CHUNKING: frozenset(
-        {DocumentState.EMBEDDING, DocumentState.FAILED}
+        {DocumentState.EMBEDDING, DocumentState.DUPLICATE, DocumentState.FAILED}
     ),
     DocumentState.EXTRACTING: frozenset(
-        {DocumentState.CHUNKING, DocumentState.FAILED}
+        {DocumentState.CHUNKING, DocumentState.DUPLICATE, DocumentState.FAILED}
     ),
+    DocumentState.DUPLICATE: frozenset(
+        {DocumentState.PENDING, DocumentState.DELETED}
+    ),
```

`duplicate` is reachable at `chunking` because that is the first point the content
hash exists (§4.4: the hash is over chunked content, not file bytes). `pending` as a
successor is the "let an admin decide" half of FR-4 — an admin can promote it to be
indexed deliberately.

`web/lib/types.ts` already listed `"duplicate"` in its `DocumentState` union, so the
frontend had been typed against a state the backend could not produce. No frontend
change was needed.

## Fix 4 — delete the dead code

```diff
--- a/app/ingest/dedupe.py
+++ b/app/ingest/dedupe.py
     # Only tombstoned copies exist, so this is not a live duplicate.
     return HashCheck(HashVerdict.NEW, None, None)
-
-    if row.doc_id == doc_id:
-        # Same document, same content. This is the redelivery case.
-        if row.state == DocumentState.LIVE:
-            return HashCheck(HashVerdict.SAME_DOCUMENT, row.doc_id, row.version)
-        # Tombstoned or failed: same document re-uploaded. Treat as a
-        # re-ingestion rather than a duplicate of itself.
-        return HashCheck(HashVerdict.NEWER_VERSION, row.doc_id, row.version)
-
-    if row.state in {DocumentState.SUPERSEDED, DocumentState.DELETED}:
-        # The other copy is not live, so this is not a live duplicate.
-        return HashCheck(HashVerdict.NEW, None, None)
-
-    return HashCheck(HashVerdict.DUPLICATE, row.doc_id, row.version)
```

---

# Deviations from the proposed diffs

Three things changed while implementing, each because the proposal was wrong or
incomplete:

1. **Fix 1 needed a second half.** The guard alone broke three tests in
   `tests/api/test_cors.py`, which built a `TestClient` outside a context manager —
   so the lifespan never ran and the factory was never registered. The fix had to
   register the factory explicitly in two places: the app lifespan
   (`app/main.py`) and the test `engine` fixture (`tests/conftest.py`). The
   `get_db` dependency's docstring was also corrected; it claimed "the engine is
   already bound by the first caller", which was true only because the first caller
   was the silent fallback being removed.

2. **Fix 3 could not compile as written** — see Fix 5.

3. **Fix 2's "skip a wrong-width row" half was not applied.** It changes failure
   semantics in a way that was never reviewed, and one corrupt row aborting every
   query is arguably the safer default: a loud total failure is easier to diagnose
   than a silently reduced result set. Left as a recommendation, not a change.

# Tests each fix needs

- **`get_session_factory()` raises without settings when no engine is cached** —
  pins the leak shut. Plus a guard test asserting `rag.db` is byte-unchanged by a
  full `tests/api` run, so a regression is caught by the suite that caused it.
  **Done** — `tests/test_isolation.py`.
- **A wrong-width chunk is skipped, not fatal** — seed one 64-dim chunk beside valid
  384-dim rows, assert `search` returns the good ones and reports the skip. And
  assert `sync_chroma_index` still *raises*, since that is a deliberate repair path.
  **Not done** — this changes failure semantics (one corrupt row currently aborts
  every query, which is arguably the safer default) and was not part of the approved
  scope. Deliberately deferred.
- **A duplicate upload is not retrievable** — upload identical content twice, assert
  the second is `DUPLICATE`, absent from `search`, absent from the Chroma projection,
  and that its row and `duplicate_of` survive. **Done** for the state, retrieval
  exclusion, and no-chunks-written; the Chroma-projection half is covered indirectly
  by the shared `live` predicate.
- **`DUPLICATE` is reachable** — a direct assertion that the state is assigned, so the
  dead enum member cannot rot again. **Done.**
- **Deleting the docstring-claimed invariant** — a test that every fixture-provided
  database is under `tmp_path`. **Done** via the byte-identity guard.

# What remains outstanding

- **The 11 corrupt rows are still in `rag.db`.** Retrieval against it aborts. Blocked
  on `HF_TOKEN`; see [Why fix 2 is blocked](#why-fix-2-is-blocked). `rag.db` is
  124 docs / 308 chunks, unchanged.
- **`scripts/ingest_corpus.py` has no re-embed flag.** Re-running it takes the
  idempotency path (`same_document`, vectors reused) rather than rebuilding them, so
  the repair needs an explicit purge-then-embed path that does not yet exist.
  Recorded in `implementation.md` §1.
- **The exact triggering test for defect 1 was never isolated.** The mechanism is
  measured and the no-argument call sites were identified, but the write was
  order-dependent and bisecting to file granularity did not pin a single file. Fix 1
  removed the class of bug regardless, and the byte-identity guard test will name
  the culprit if it recurs.
- **`make chroma-sync` has not been run**, and cannot succeed until the short rows
  are re-embedded — `sync_chroma_index` raises on a width mismatch by design.
- **`architecture.md` §4.1 was updated** to include the `duplicate` state, so the
  documented lifecycle and the code now agree.
