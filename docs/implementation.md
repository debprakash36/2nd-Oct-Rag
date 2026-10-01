# Implementation Guide — RAG Chatbot

**Purpose:** Phase-by-phase build instructions for an AI coding agent (Cursor). Each phase is self-contained, verifiable, and safe to stop at.
**Source of truth:** `docs/architecture.md` (design) and `docs/PRD.md` (requirements).
**How to use:** Work phases in order. Do not start a phase until the previous phase's exit gate passes. Each phase ends with a commit boundary.

---

## 0. How to use this document

**The loop, every phase:**
1. Give the agent the phase prompt block (§1.8, §2.8, …) verbatim.
2. Agent implements. It must not move on to the next task before the current one is verified.
3. Run the phase's verification commands yourself. Do not accept the agent's claim that something works.
4. Only then proceed.

**Rules the agent is given in every phase prompt** (repeated at the top of each block so it survives context compaction):

> - Implement only what this phase specifies. Do not add features, endpoints, or abstractions that belong to a later phase.
> - When the architecture says a choice is deferred or configurable, implement it as configuration with a documented default. Do not invent a second way to do it.
> - Every module gets type hints and docstrings explaining *why*, not *what*.
> - No secrets in code. All credentials via environment variables, loaded through the config module.
> - No `print()` for logging outside scripts. Use the structured logger.
> - If a task cannot be completed as written, stop and report why. Do not substitute a different design silently.

**Anti-patterns to watch for.** These are the specific failure modes that show up when an agent implements this kind of system, and they are called out per phase below:

| Pattern | Why it is wrong |
| --- | --- |
| Provider SDK types leaking past the interface | Defeats NFR-10. The orchestrator must not import a vendor SDK. |
| Keyword index written independently of the vector index | Silent drift → confidently wrong answers. The chunk store is the only membership source. |
| Post-generation access filtering | Unauthorized text already reached the model. It can leak by paraphrase. |
| Buffered (non-streaming) answer assembly | Violates NFR-1 TTFT. See `architecture.md` §3.1. |
| An answer shown with no valid citations | Violates NFR-3. Converts to refusal instead. |
| Chunk strategy hardcoded | Phase 2 must compare strategies. Must be config. |
| Shell-interpolated filenames during extraction | Injection vector. NFR-5. |

---

## 1. Reference stack

The architecture deliberately defers product choices (`architecture.md` §13). This document needs *a* concrete stack to write file paths against. Use the table below; if you have already chosen differently, keep the structure and substitute products.

| Concern | Reference choice | Interface to keep |
| --- | --- | --- |
| Language | Python 3.11+ | - |
| API framework | FastAPI | SSE via `StreamingResponse` |
| Data models | Pydantic v2 | Used for API schemas and internal models |
| Vector store | pgvector (Postgres) | `VectorStore` protocol |
| Keyword index | Postgres full-text / ParadeDB | `KeywordIndex` protocol |
| Chunk + document state | Postgres | Schema in `architecture.md` §5 |
| Object storage | S3-compatible | `ObjectStore` protocol |
| Job queue | Postgres-backed queue (or Celery) | At-least-once, idempotent workers |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, **384 dimensions**, via the HuggingFace Inference API | `EmbeddingProvider` — `architecture.md` §7.3 |
| Generation | **Groq** (`llama-3.3-70b-versatile`), streaming | `GenerationProvider` — `architecture.md` §7.3 |
| Reranking | Provider-agnostic, behind `RerankerProvider` | `RerankerProvider` — `architecture.md` §7.3 |
| Frontend | Next.js + React, SSE via `fetch` + `ReadableStream` | Escapes all document-derived text |
| Tests | pytest | - |

**Why Postgres for three things.** One database holding chunks, document state, and the keyword index means membership consistency is a transaction rather than a reconciliation job. That is the single most valuable property in this architecture. Splitting them later is a migration; starting split means building reconciliation first.

**Why MiniLM at 384 dimensions.** Chosen and recorded in `architecture.md` §13. Three properties:

1. **384 matches the existing `EMBEDDING_DIM`**, so the pgvector vector column does
   not change width. 768-dim alternatives (`all-mpnet-base-v2`, `e5-base-v2`) would
   have required a wider column in the initial migration.
2. **Hosted rather than local weights.** The Inference API avoids a ~2.5 GB
   `torch` + `transformers` install. The cost is a network round-trip per embed, and
   because embedding runs on the *query* path as well as at ingestion, that latency
   is spent out of the NFR-1 budget on every request. Measured offline retrieval is
   ~50 ms p95 (`docs/perf_stages.json`) — this is what that headroom buys.
3. **Small enough to be cheap.** MiniLM is a distilled sentence-transformer; quality
   is below a 768-dim base model, which is the acceptable trade at this corpus size
   and the one to revisit if groundedness falls short of the §8 gate.

**Why Groq for generation, and why that is not a preference.** Groq does not offer an
embeddings endpoint, so the generation and embedding providers *must* be different
vendors. Given that, Groq's OpenAI-compatible streaming API drops into the
sentence-buffered citation validator unchanged — the consumer depends on
`GenerationProvider`, not on the wire format.

**Re-embedding is mandatory, and is a schema change.** No vector produced by the
former `fake` hash embedder is usable with a real model: they are the wrong kind of
thing, not merely the wrong width. The full corpus must be re-ingested in a single
pass. A *partial* re-embed is worse than either extreme — it leaves two vector
spaces in one index, so every cosine score becomes meaningless while still returning
plausible-looking neighbours. The dimension guard (§7.3) catches the width case; it
cannot catch a mixed-provenance corpus, which is why re-embedding is all-or-nothing.

**`scripts/ingest_corpus.py` has no re-embed flag.** Re-running it against documents
that already exist takes the idempotency path: the content hash is unchanged, so
`check_content_hash` returns `same_document` and existing vectors are reused rather
than rebuilt. A forced re-embed needs an explicit purge-then-embed path, which is
not yet written.

---

## 2. Phase 0 — Foundations

**Goal:** A running skeleton with secrets managed, provider interfaces defined, and trace IDs flowing. No features.
**Exit gate:** `pytest` green, `/health` responds, one embedding call succeeds against a real provider, logs are structured with a trace ID.
**Time:** 0.5–1 day. **Commits:** 4

### 2.1 Tasks

| # | Task | Deliverable |
| --- | --- | --- |
| 0.1 | Repo skeleton, `pyproject.toml`, lint + type-check + pytest wired to one command | `pyproject.toml`, `Makefile` |
| 0.2 | Config module: all settings from env, validated at startup, fail loudly on missing | `app/core/config.py` |
| 0.3 | Structured logging with `trace_id` context var | `app/core/logging.py` |
| 0.4 | Middleware: assign/propagate `trace_id`, emit one access log line | `app/core/middleware.py` |
| 0.5 | **Provider interfaces** — the NFR-10 seam | `app/providers/base.py` |
| 0.6 | One concrete embedding provider + a `FakeEmbeddingProvider` for tests | `app/providers/embedding.py` |
| 0.7 | DB session management, migrations wired, `/health` checks DB + providers | `app/db/session.py`, `app/api/health.py` |
| 0.8 | Docker Compose: app + Postgres (pgvector) | `docker-compose.yml`, `Dockerfile` |

### 2.2 Provider interfaces — the one thing to get exactly right

This is the most-reversed decision in the system. Write it as specified:

```python
# app/providers/base.py
from typing import Protocol, Iterator, Sequence
from dataclasses import dataclass

@dataclass(frozen=True)
class ScoredDoc:
    chunk_id: str
    score: float
    text: str

class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        """Batch-embed. Must preserve input order in output."""
        ...

class GenerationProvider(Protocol):
    def stream(self, messages: list[dict], *, model: str,
               max_tokens: int, temperature: float) -> Iterator[str]:
        """Yield text deltas. Must be lazy; do not buffer the full response."""
        ...

class RerankerProvider(Protocol):
    def rerank(self, query: str, docs: Sequence[str], *, top_k: int) -> list[ScoredDoc]:
        ...
```

Instruct the agent: **no module outside `app/providers/` may import a vendor SDK.** Enforce it in CI with a grep-based test (§2.4) rather than trusting review — this is exactly the kind of rule that erodes silently.

### 2.3 Config rules

- Every setting has a documented default except secrets, which are required.
- `EMBEDDING_DIM` is required and pinned. A dimension mismatch against the vector store must raise at startup, not degrade silently into wrong similarity scores (`architecture.md` §7.3).
- `PROMPT_VERSION` is a config value from day one. Traceability of every prompt change depends on it.

### 2.4 Verification

```bash
make check          # lint + typecheck + tests, must be green
docker compose up -d
curl localhost:8000/health          # db ok, providers ok
pytest tests/test_providers.py     # includes the no-vendor-import guard
```

The guard test:

```python
def test_no_vendor_sdk_imports_outside_providers():
    """NFR-10: only app/providers/ may import vendor model SDKs."""
    offenders = []
    for path in Path("app").rglob("*.py"):
        if "providers" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for sdk in ("openai", "anthropic", "cohere", "google.generativeai"):
            if re.search(rf"^\s*(import|from)\s+{sdk}\b", text, re.M):
                offenders.append(f"{path}: {sdk}")
    assert not offenders, f"vendor SDK leaked outside providers/: {offenders}"
```

### 2.5 Phase 0 prompt block

```
Implement Phase 0 (Foundations) of the RAG chatbot per docs/implementation.md §2.
Design reference: docs/architecture.md. Requirements: docs/PRD.md.

Before writing code, read docs/architecture.md §1 (drivers), §5 (data model),
§7.2 (observability), §7.3 (provider abstraction).

Standing rules:
- Implement only Phase 0. No ingestion, no retrieval, no generation endpoints.
- Provider interfaces exactly as specified in implementation.md §2.2.
- No vendor SDK imports outside app/providers/ (enforced by test).
- No secrets in code; env only, validated at startup.
- Structured logging with trace_id propagated via middleware.
- Type hints and docstrings on everything; docstrings explain why.

Do not proceed to task 0.N+1 until 0.N is verified. Report what you built and
what you could not complete, with reasons.
```

---

## 3. Phase 1 — Ingestion path

**Goal:** Documents go in and become retrievable, correctly-addressed chunks. This is `architecture.md` §4.
**Exit gate:** 100 sample documents ingest; `chunk.char_start/char_end` resolve to the exact source passage; state machine transitions are legal and idempotent under redelivery.
**Time:** 1–2 weeks. **Commits:** 6 (one per subsection below)

### 3.1 Task breakdown

| # | Task | Satisfies | Deliverable |
| --- | --- | --- | --- |
| 1.1 | Schema + migrations: `Document`, `Chunk` | FR-6, FR-10 | `alembic/versions/0001_init.py` |
| 1.2 | Upload endpoint, validation, per-file errors | FR-1, FR-2 | `app/api/admin/documents.py` |
| 1.3 | Sandboxed extraction + cleaning | FR-3, NFR-5 | `app/ingest/extract.py`, `app/ingest/sandbox.py` |
| 1.4 | Chunking, config-driven, with `char_start/char_end` | FR-9, FR-10 | `app/ingest/chunk.py` |
| 1.5 | Breadcrumb construction | FR-11 | `app/ingest/breadcrumb.py` |
| 1.6 | Content hash, dedupe, version supersede | FR-4, FR-8 | `app/ingest/dedupe.py` |
| 1.7 | Embedding + index write (vector **and** keyword **and** chunk store, one transaction boundary per chunk batch) | FR-12 prep | `app/ingest/index.py` |
| 1.8 | State machine + queue worker, idempotent | FR-5, G4 | `app/ingest/worker.py`, `app/ingest/states.py` |
| 1.9 | Tombstone semantics: disable / supersede / delete | FR-7 | `app/ingest/tombstone.py` |
| 1.10 | Admin API: list, status, chunk count, disable, delete | FR-27 | `app/api/admin/documents.py` |

### 3.2 Chunking — the details that matter

Per `architecture.md` §4.2. Instruct the agent on the specifics; the PRD's one-line summary is not sufficient to implement against.

```
Configuration (app/ingest/chunk_config.py) — a Pydantic model, not constants:
  target_tokens: int = 1000
  overlap_tokens: int = 200
  strategy: Literal["heading", "paragraph", "fixed"] = "heading"
  min_chunk_tokens: int = 50
```

Rules to encode:
- **Split on headings first**, then accumulate paragraphs into `target_tokens` windows with `overlap_tokens` overlap.
- **Never split mid-sentence.** A single paragraph longer than the target becomes its own oversized chunk rather than being cut.
- **Drop degenerate chunks** below `min_chunk_tokens` (page numbers, stray headers).
- **Record exact offsets.** `char_start`/`char_end` are offsets into the *cleaned* text, and the cleaned text must be persisted so offsets stay resolvable. This is what makes citations land on a passage instead of a file.
- **Breadcrumb prepended to the embedded text, not the stored text.** The stored chunk keeps raw text; embedding input is `breadcrumb + "\n" + text`. Otherwise the breadcrumb pollutes search results and shows up in citations.

**Anti-pattern to call out:** agents commonly emit offsets that count whitespace-normalized text while the offsets were computed on the raw text. Pick one canonical cleaned text, persist it, and derive all offsets from it. Add a test asserting a known phrase's offsets round-trip.

### 3.3 Idempotency — the subtle part

`architecture.md` §4.4. Queue delivery is at-least-once, so every stage must converge rather than accumulate.

The failure this prevents: a redelivered message embeds the same chunks twice, producing duplicate vectors that return the same passage twice in one answer.

```
hash on CHUNKED CONTENT, not file bytes
  ├─ new hash                  → ingest as new document
  ├─ existing hash, same doc_id, newer source → supersede old version, re-index (FR-8)
  └─ existing hash, different doc_id          → flag near-duplicate, offer skip (FR-4)
```

Hashing chunked content rather than file bytes is deliberate: re-saving a PDF changes the bytes without changing the text, and byte hashing re-embeds for nothing. Make the agent implement it this way and say why in a comment.

**Required test:** process the same document 3× via the queue; assert exactly one `Document` row, one set of `Chunk` rows, one vector set, and `state = live`.

### 3.4 State machine

```
pending → extracting → chunking → embedding → indexing → live
   │           │            │           │           │
   └───────────┴────────────┴───────────┴───────────┴──→ failed (with reason)

live → superseded (FR-8)      live → disabled (FR-7) → live
live|disabled → deleted (FR-7)
```

Encode as an explicit transition table, not scattered `if` statements. Two invariants:
1. Only `live` is retrievable.
2. A `failed` document leaves **no partial index entries**. Clean up on failure, or mark all its chunks non-retrievable.

A half-indexed document is worse than a missing one: the system answers confidently from a source the user cannot fully read. Say this in a comment at the cleanup site so it is not "optimized away" later.

`superseded` and `disabled` are **tombstones, not deletes.** Retrieval filters on document state, so a disabled doc stops being served immediately even though its chunks remain. Physical removal is a separate background job.

### 3.5 Verification

```bash
make check
pytest tests/ingest/ -v
pytest tests/ingest/test_idempotency.py -v      # the 3× test above
pytest tests/ingest/test_state_machine.py -v   # transitions exhaustive + legal
pytest tests/ingest/test_offsets.py -v          # offset round-trip

# end-to-end: ingest the 100-doc sample corpus
python scripts/ingest_corpus.py --dir ./samples --stats
```

The stats output must show `documents=100 live=100 failed=0` and a chunk count consistent with the configured strategy. A silent drop in chunk count usually means the min-size filter is too aggressive — check that before moving on.

### 3.6 Phase 1 prompt block

```
Implement Phase 1 (Ingestion path) of the RAG chatbot per docs/implementation.md §3.
Design reference: docs/architecture.md §4, §4.2, §4.3, §4.4, §5.
Requirements: docs/PRD.md FR-1 through FR-11, FR-27.

Standing rules:
- Only Phase 1. No retrieval, no generation, no chat endpoint.
- Extraction runs sandboxed: no network, read-only FS, CPU/memory/time caps,
  no shell interpolation on filenames.
- Chunking is config-driven. Do not hardcode chunk sizes.
- Offsets derive from ONE canonical cleaned text, which must be persisted.
- Breadcrumb is prepended to the EMBEDDING input only, not stored text.
- Hash chunked content, not file bytes.
- Only state=live is retrievable. Failed docs leave no partial index entries.
- Every ingestion stage must be idempotent under queue redelivery.

Build the state machine as an explicit transition table. Add the three required
tests named in §3.5 by name.

Report: files created, test results, and anything you could not complete.
```

---

## 4. Phase 2 — Retrieval + eval harness

**Goal:** Retrieve the right passages, and prove it. This is `architecture.md` §3.3. **This phase gates the quality of everything downstream.**
**Exit gate:** Recall@10 ≥ 0.85 on the eval set; the four validation checks in `architecture.md` §10 have been run and their results recorded.
**Time:** 2–3 weeks — the longest phase. **Commits:** 5

> **Do not compress this phase.** Prompts are cheap to iterate and retrieval is not. Phase 3 tuning generation against an unmeasured retriever produces prompt work that compensates for a defect it cannot fix.

### 4.1 Task breakdown

| # | Task | Satisfies | Deliverable |
| --- | --- | --- | --- |
| 2.1 | **Eval set**: 150–300 hand-labelled questions, each with expected source doc/chunks | A-5, NFR-3 | `eval/dataset.jsonl`, `docs/eval_guide.md` |
| 2.2 | `VectorStore` protocol + pgvector implementation, filters pushed down | FR-16 | `app/retrieval/vector_store.py` |
| 2.3 | `KeywordIndex` protocol + BM25 implementation, filters pushed down | FR-12 | `app/retrieval/keyword_index.py` |
| 2.4 | Parallel fetch + RRF fusion + near-duplicate dedupe | FR-12, FR-15 | `app/retrieval/fusion.py` |
| 2.5 | Rerank stage, config-driven on/off | FR-13 | `app/retrieval/rerank.py` |
| 2.6 | Threshold calibration + context token cap + eviction | FR-14, FR-15 | `app/retrieval/threshold.py` |
| 2.7 | Query rewrite: anaphora resolution (rule-based first), optional LLM | G3, US-3 | `app/retrieval/rewrite.py` |
| 2.8 | `Retriever` orchestrating the above; logs original **and** rewritten query | FR-28 | `app/retrieval/retriever.py` |
| 2.9 | Eval harness: recall@k, MRR, per-stage score dump | §8 metrics | `eval/run_eval.py` |

### 4.2 The eval set comes first

Build it **before** any tuning. An agent asked to "improve retrieval" without labelled data will optimize against its own guesses, and will report improvement that is not there.

`eval/dataset.jsonl`, one object per line:

```json
{
  "id": "q001",
  "question": "What is the refund window for digital products?",
  "relevant_chunk_ids": ["..."],
  "relevant_doc_ids": ["..."],
  "expected_abstain": false,
  "category": "policy",
  "notes": "Follow-up style: depends on prior turn for 'digital products'."
}
```

Include at minimum:
- ~60% directly answerable, spread across document types and topics.
- ~20% **should abstain** — not in the corpus. These are the only way to calibrate the threshold honestly; a threshold tuned only on answerable questions always ends up too permissive.
- ~10% multi-turn, requiring rewrite.
- ~10% exact-identifier lookups (codes, dates, proper nouns) — the set that exposes a missing keyword index.

`expected_abstain` is the field most often omitted and the most valuable. Insist on it.

### 4.3 Fusion — why RRF and not score averaging

Instruct the agent to implement Reciprocal Rank Fusion, and give it the reason so it does not "improve" it into weighted score combination:

> Vector cosine similarity and BM25 are not on comparable scales. RRF uses only ranks (`Σ 1/(k + rank_i)`, k≈60), so it needs no calibration. Any normalization applied today is silently invalidated when the corpus distribution shifts — which it will, every time documents are added.

### 4.4 Threshold calibration

Per `architecture.md` §3.3, the threshold is measured, not guessed. Two targets must hold **simultaneously**:
- recall@10 ≥ 0.85
- refusal rate 10–30%

Sweep the threshold across the eval set and plot both. If **no** threshold satisfies both, stop and report it. That means the PRD's target pair is wrong, and it is a PRD-level conversation, not something to tune around.

### 4.5 Verification

```bash
make check
python eval/run_eval.py --dataset eval/dataset.jsonl --stage all
python eval/run_eval.py --ablate keyword    # §10 check 1
python eval/run_eval.py --ablate rerank     # §10 check 2
python eval/run_eval.py --ablate rewrite    # rule-based vs LLM rewrite
python eval/run_eval.py --sweep-threshold   # §10 check 4
```

Record every run's output in `docs/eval_results.md`, including the ablations. These numbers are the evidence base for the PRD §8 metrics and for the Phase 3 prompt work.

### 4.6 Phase 2 prompt block

```
Implement Phase 2 (Retrieval + eval harness) per docs/implementation.md §4.
Design reference: docs/architecture.md §3.2, §3.3, §10.
Requirements: docs/PRD.md FR-12 through FR-16.

Standing rules:
- BUILD THE EVAL DATASET FIRST, before any tuning. 150-300 questions.
  At least 20% must be expected_abstain=true. At least 10% must require
  multi-turn rewrite. At least 10% must be exact-identifier lookups.
- Do not tune anything against unlabelled data. If you cannot measure an
  improvement, do not claim one.
- Fusion is RRF (rank-based). Do not implement weighted score combination.
- Access filters are pushed INTO both stores. Never filter after generation.
- Threshold must be calibrated, not guessed. It must satisfy recall@10 >= 0.85
  AND refusal rate in [0.10, 0.30] simultaneously.
- Log the original AND rewritten query separately.
- Rerank must be toggleable by config so it can be ablated.

Run the four validation checks in architecture.md §10 and write results to
docs/eval_results.md. If no threshold satisfies both targets, STOP and report
that rather than picking the closest one.

Report: recall@10, refusal rate, ablation deltas, and the threshold chosen
with its justification.
```

---

## 5. Phase 3 — Generation, streaming, citation validation

**Goal:** Turn retrieved passages into grounded, cited, streaming answers. This is `architecture.md` §3.1, §3.4, §3.5.
**Exit gate:** Groundedness ≥ 0.90 on the eval set; 100% of emitted citations resolvable; TTFT p95 ≤ 5 s against a local provider.
**Time:** 1–2 weeks. **Commits:** 5

### 5.1 Task breakdown

| # | Task | Satisfies | Deliverable |
| --- | --- | --- | --- |
| 3.1 | Fixed, versioned system prompt with delimiters around context | FR-17, FR-32 | `app/generation/prompt.py` |
| 3.2 | `GenerationProvider` + streaming implementation | FR-19, NFR-10 | `app/providers/generation.py` |
| 3.3 | Sentence-level sentence splitter for citation validation | FR-18 | `app/generation/sentences.py` |
| 3.4 | Citation validator: closed ID set, strip invalid, zero-valid → refusal | FR-18, G6 | `app/generation/validator.py` |
| 3.5 | Streaming response assembly: `sources` → flushed tokens → `done` | FR-19, FR-20 | `app/api/chat.py` |
| 3.6 | Refusal path with its own prompt and sources panel | G2, US-2 | `app/generation/refusal.py` |
| 3.7 | Answer length presets | FR-21 | `app/generation/prompt.py` |
| 3.8 | `QueryLog` schema + write path with all instrumented fields | FR-28 | `app/db/query_log.py` |

### 5.2 The citation scheme

`architecture.md` §3.5. Give the agent the mechanism explicitly — this is the part most likely to be built wrong:

```
Context is presented to the model as a numbered list, 1..K.
The model emits inline markers [1], [2].
Markers are integers indexing the retrieved set — never internal chunk IDs.
The server maps marker n → real chunk_id after generation.
```

Two rules that make it hold:
1. **The model never sees or emits internal `chunk_id` values.** Storage identifiers must not reach the prompt or the response.
2. **The retrieved set is closed.** A marker outside `1..K` is fabricated by definition.

Validator logic, in order:

```python
for sentence in split_sentences(streamed_text):
    markers = extract_markers(sentence)
    valid = [m for m in markers if 1 <= m <= len(retrieved)]
    if not markers:
        flush(sentence); continue                    # transitional text, no claim
    if valid:
        flush(sentence.remap(valid)); continue
    if not valid:
        flush(sentence.remap([])); flag_stripped()
    if NO_SENTENCE_HAS_A_VALID_MARKER:
        emit_refusal_instead_of_answer()             # zero valid citations
```

The last rule is the deliberate deviation from the PRD's literal "strip" wording, documented in `architecture.md` §3.5: an answer with **zero** valid citations is ungrounded, and displaying it fails the PRD's own NFR-3. Instruct the agent to implement it and to comment why, so it is not "simplified" away.

### 5.3 Streaming — the conflict made concrete

FR-19 (stream) and FR-18 (validate citations) cannot both hold if tokens reach the user before validation. `architecture.md` §3.1 resolves this with **sentence-level buffering**:

- Buffer one sentence at a time, not the whole answer.
- Flush each clean sentence immediately. TTFT is unaffected because the first sentence is usually clean.
- Do **not** buffer the full response "to be safe." That reintroduces the NFR-1 violation the design is avoiding.

Instruct the agent: the sentence buffer is a deliberate, bounded trade. Add a comment stating the condition under which it should escalate to full buffering (a measurable rate of early claims invalidated by later ones), so the decision is documented rather than accidental.

### 5.4 Prompt rules

- **System prompt is fixed and versioned.** Not per-tenant, not assembled from retrieved content. `PROMPT_VERSION` is logged on every query (`QueryLog.prompt_version`).
- **Retrieved text sits inside explicit delimiters** in a data block. This is a functional part of the injection defense (FR-32), not formatting.
- **Instruction hierarchy is stated in the prompt:** system instructions govern; retrieved documents are reference data; nothing in a document can change the system prompt.
- **Abstention instruction is explicit** (FR-17): if the context does not contain the answer, say so and do not speculate.
- Length preset is a parameter, not a second prompt (FR-21).

### 5.5 `QueryLog` fields

`architecture.md` §5. Persist all of them now — retroactive instrumentation is not possible, and FR-30 plus the PRD §11 improvement loop depend on this table.

```
query_id, trace_id, conversation_id, original_query (post-redaction),
rewritten_query, retrieved_ids, scores{vector,bm25,fused,reranked},
threshold_applied, abstained, model, prompt_version, tokens_in, tokens_out,
ttft_ms, total_ms, citations_stripped, feedback
```

`scores` and `rewritten_query` are the two most commonly omitted and the two most valuable: without them you cannot distinguish a **content gap** (right document absent) from a **retrieval gap** (document present, not retrieved) — the classification the PRD's improvement loop requires.

### 5.6 Verification

```bash
make check
python eval/run_eval.py --stage generation    # groundedness, citation correctness
pytest tests/generation/test_validator.py -v  # incl. zero-valid → refusal
pytest tests/generation/test_streaming.py -v  # sentence flush, not full buffer
pytest tests/generation/test_injection.py -v  # document-embedded prompt injection
python scripts/bench_ttft.py                   # p50/p95 TTFT
```

Required tests, named so the agent cannot skip them:

| Test | Asserts |
| --- | --- |
| `test_invalid_marker_stripped` | `[9]` with K=3 is removed, sentence still shown |
| `test_zero_valid_markers_becomes_refusal` | No valid marker anywhere → refusal, not an answer |
| `test_internal_ids_never_leak` | No `chunk_id` string appears in prompt or response |
| `test_first_token_before_completion` | First flush arrives while the generator is still producing |
| `test_document_injection_ignored` | A document containing "ignore previous instructions" does not change behaviour |

### 5.7 Phase 3 prompt block

```
Implement Phase 3 (Generation, streaming, citation validation) per
docs/implementation.md §5. Design reference: docs/architecture.md §3.1,
§3.4, §3.5. Requirements: docs/PRD.md FR-17 through FR-22.

Standing rules:
- Citation markers are integers 1..K indexing the retrieved set. Internal
  chunk_id values must never reach the prompt or the response.
- Buffer SENTENCE-wise, not answer-wise. Do not assemble the full response
  before flushing. This preserves TTFT and is a deliberate trade — comment
  the condition under which it should escalate.
- An answer with ZERO valid citations becomes a refusal. This is a deliberate
  deviation from the PRD's "strip" wording; comment why at the code site.
- Stripped markers are logged, never silently dropped.
- System prompt is fixed and versioned. Retrieved text goes inside explicit
  delimiters and is treated as untrusted data, never instructions.
- Persist every QueryLog field in §5.5. No field is optional.
- Log original AND rewritten query, plus per-stage scores.

Write the five tests in §5.6 by exact name. Do not proceed until they pass.

Report: groundedness score, citation correctness, TTFT p50/p95, strip rate.
```

---

## 6. Phase 4 — Product surface

**Goal:** A usable product. Chat UI, conversation state, sources panel, feedback, admin console.
**Exit gate:** End-to-end happy path works; a bad answer is diagnosable from logs alone.
**Time:** 2–3 weeks. **Commits:** 6

### 6.1 Task breakdown

| # | Task | Satisfies | Deliverable |
| --- | --- | --- | --- |
| 4.1 | Conversation store; last-N turns into context | FR-23 | `app/db/conversation.py` |
| 4.2 | New chat / delete conversation | FR-25 | `app/api/conversations.py` |
| 4.3 | `POST /chat/stream` SSE endpoint, all event types | FR-19, FR-20, FR-34 | `app/api/chat.py` |
| 4.4 | Guardrails: size cap, rate limit, error sanitization | FR-31, FR-34, NFR-5 | `app/core/guardrails.py` |
| 4.5 | Chat UI: streaming render, message history | FR-19, FR-23 | `web/app/chat/page.tsx` |
| 4.6 | Sources panel: clickable citations → exact passage | FR-18, FR-20, NFR-9 | `web/components/SourceViewer.tsx` |
| 4.7 | Feedback endpoint + thumbs UI | FR-29 | `app/api/feedback.py` |
| 4.8 | Admin console: doc list, state, chunk count, disable/delete | FR-27 | `web/app/admin/page.tsx` |
| 4.9 | Content-gap report (P2 — build only if time permits) | FR-30 | `scripts/content_gaps.py` |

### 6.2 Frontend rules

These are the frontend requirements that an agent will otherwise get wrong. State each explicitly:

- **Escape everything.** All document-derived and model-derived text renders as text, never HTML. `dangerouslySetInnerHTML` is prohibited anywhere in the UI (FR-22). A document containing `<script>` must render as visible text.
- **`sources` event arrives before tokens.** Render the sources panel immediately so the user sees what the system is reading while it reads it. Do not wait for the answer.
- **Refusals still show sources** (FR-20). A refusal with no sources gives the user nowhere to go.
- **Keyboard navigable, ARIA-labelled** citation markers and source panel, WCAG 2.1 AA contrast (NFR-9).
- **Errors are user-safe.** Provider errors, stack traces, and internal IDs never reach the browser. The API returns a code and a message only.

### 6.3 Guardrails

| Guard | Behaviour | Requirement |
| --- | --- | --- |
| Input size cap | Reject above the cap with a clear message, before any provider call | FR-34 |
| Rate limit | Per user/IP at the gateway; clear message on `429` | FR-31 |
| Output sanitization | Model output escaped at render, not filtered by regex | FR-22 |
| Error sanitization | Typed internal errors → safe external codes | NFR-5 |

Say clearly: regex filtering of model output for "sensitive words" is not a security control and must not be implemented as one. Sanitization happens at the render boundary.

### 6.4 Verification

```bash
make check
pytest tests/api/ -v
pytest tests/e2e/ -v                 # full flow, mocked providers
npm --prefix web run build && npm --prefix web run test

# manual
curl -N -X POST localhost:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{"message":"What is the refund policy?","answer_style":"concise"}'
```

`curl -N` is the fast check that streaming is real: tokens should arrive incrementally, not in one burst at the end. If output appears all at once, sentence buffering has regressed into answer buffering — a direct NFR-1 violation.

### 6.5 Status

Tasks 4.1–4.8 shipped. 4.9 (content-gap report) is deferred: it is P2 and the logging it reads from (`QueryLog`) only became queryable in this phase.

Verified:

| Check | Result |
| --- | --- |
| `make lint` (ruff) | clean |
| `make typecheck` (mypy, py3.12 target) | clean, 56 files |
| `make test` | 467 passed |
| `npm --prefix web run lint` | clean |
| `npm --prefix web run typecheck` | clean |
| `npm --prefix web test` | 67 passed |
| `npm --prefix web run build` | 6 routes, static prerender |
| `curl -N /chat/stream` | `sources` → incremental `token` events → `done`, verified against the live corpus |

Things worth knowing before building on this:

- **The web UI is a separate origin.** CORS is configured through `cors_allow_origins`, defaulting to localhost:3000 only. A deployment that does not set it will refuse browser requests rather than serve any origin. `NEXT_PUBLIC_API_BASE` is inlined at build time, so it is a build argument, not a runtime env var.
- **`next lint` is deprecated** and prompts interactively on first run. Linting runs through the ESLint CLI (`eslint.config.mjs`, `FlatCompat` over `eslint-config-next`).
- **Chroma is a derived index and nothing writes to it during ingestion.** It is empty until `make chroma-sync` runs. Re-run it after corpus changes; it is idempotent. At the current corpus size (297 chunks) the brute-force SQLite store is likely faster than the ANN path, so Chroma earns its place only as the corpus grows — not as a default.
- **No authentication.** Every conversation, feedback, chunk, and admin endpoint is unauthenticated. None of them may be exposed publicly without authorization being added.

Fixed since this was written:

- Citation markers inside answer text (`[2]`) are now real focusable controls. `web/components/CitedAnswer.tsx` renders each marker as a `<button>` carrying `aria-label` (naming the source document and page), `aria-expanded`, and `aria-controls` pointing at the passage panel that actually appears. A marker with no matching source stays literal text rather than becoming a dead button. The known limitation is that sources are only in client state for the turn being streamed, so markers in a *reloaded* conversation render as plain text — the passage is still readable, which is the pre-fix behaviour.

### 6.6 Phase 4 prompt block

```
Implement Phase 4 (Product surface) per docs/implementation.md §6.
Design reference: docs/architecture.md §5, §6, §7.1, §8.
Requirements: docs/PRD.md FR-20, FR-23, FR-25, FR-27, FR-29, FR-31, FR-34, NFR-9.

Standing rules:
- The 'sources' SSE event is emitted BEFORE the first token. Do not batch it
  to the end of the response.
- Refusals render the sources panel too.
- dangerouslySetInnerHTML is prohibited in the entire UI. Escape all
  document- and model-derived text.
- Errors crossing the API boundary carry a code and a message only. No stack
  traces, no provider names, no internal IDs.
- Do NOT implement output filtering by keyword/regex. It is not a security
  control. Sanitization happens at the render boundary.
- Citation markers must be keyboard-focusable and screen-reader labelled.
- Build FR-30 only after FR-27 ships; it is P2.

Verify with `curl -N` that tokens stream incrementally, not in one burst.
Report: files created, and confirmation that streaming is incremental.
```

---

## 7. Phase 5 — Hardening

**Goal:** Meet the NFRs that the earlier phases only approximated.
**Exit gate:** NFR-1, NFR-5, NFR-9 met with evidence.
**Time:** 1 week. **Commits:** 4

### 7.1 Task breakdown

| # | Task | Target | Deliverable |
| --- | --- | --- | --- |
| 5.1 | Load test: 50 concurrent users, p95 TTFT | NFR-1 | `tests/load/test_chat.py` |
| 5.2 | Optimize against the §7.4 stage budget; profile the request path | NFR-1 | `docs/perf_report.md` |
| 5.3 | Security review: injection, path traversal, SSRF, XSS, secrets | NFR-5, FR-32 | `docs/security_review.md` |
| 5.4 | Extraction sandbox verification (resource caps actually enforced) | NFR-5 | `tests/ingest/test_sandbox.py` |
| 5.5 | Accessibility audit + fixes | NFR-9 | `docs/a11y_audit.md` |
| 5.6 | Dashboards: TTFT, refusal rate, strip rate, ingest throughput | NFR-8 | `ops/dashboards/` |
| 5.7 | Alerts: TTFT p95 > 4 s; strip-rate spike | NFR-8 | `ops/alerts.md` |
| 5.8 | Degradation: store-down behaviour | NFR-2 | `app/api/health.py`, `docs/runbook.md` |

### 7.2 The two alerts worth having first

They are listed separately because they catch the two failures that are otherwise invisible:

- **TTFT p95 > 4 s.** Fires *before* the 5 s target breaks, buying lead time.
- **Citation strip rate spike.** Model or prompt drift degrades answer quality silently — the UI still looks fine while answers get worse. This is the only early signal.

### 7.3 Degradation behaviour

Per `architecture.md` §8: if the vector or keyword store is unavailable, the system cannot answer. It must return a clear "search is temporarily unavailable" — **not** an answer from a partial index. A degraded-but-plausible answer is the exact failure mode this system exists to prevent. Encode this in the health check and the runbook, and test it by stopping Postgres.

### 7.4 Verification

```bash
make check
k6 run tests/load/test_chat.js              # 50 VUs; assert p95 TTFT <= 5s
pytest tests/security/ -v
pytest tests/ingest/test_sandbox.py -v
docker compose stop postgres               # assert graceful degradation, not a wrong answer
```

### 7.5 Phase 5 prompt block

```
Implement Phase 5 (Hardening) per docs/implementation.md §7.
Design reference: docs/architecture.md §7, §8. Requirements: docs/PRD.md §6.

Standing rules:
- Profile before optimizing. Record actual stage timings in
  docs/perf_report.md against the budget in architecture.md §7.4. The budget
  is a starting estimate, not a target to hit by assertion.
- If the store is down, return a clear unavailable error. NEVER answer from
  a partial index. This is a correctness requirement, not a UX preference.
- Verify sandbox resource caps are actually enforced, not merely configured.
- The only two alerts: TTFT p95 > 4s (early warning) and citation strip-rate
  spike (silent degradation).

Report: p95 TTFT under load, sandbox verification result, and the security
review findings with severity.
```

---

## 8. Phase 6 — Pilot and tuning

**Goal:** Measure against PRD §8 with real queries, and feed the improvement loop.
**Exit gate:** PRD §8 metrics measured on real traffic; at least one full improvement-loop cycle completed.
**Time:** 2–4 weeks of elapsed time, low effort.
**Commits:** ongoing

### 8.1 What happens here

No new features. Three activities:

1. **Threshold re-tuning on real queries.** The Phase 2 threshold was calibrated on 150–300 hand-picked questions. Real traffic has a different distribution. Re-sweep monthly and record the delta.
2. **The improvement loop** (PRD §11). Weekly, review lowest-rated and most-refused queries and classify each:
   - **Content gap** → the document is missing. Fix: write the document.
   - **Retrieval gap** → the document exists but was not retrieved. Fix: retune chunking, add keyword terms, adjust the threshold.
   
   These need different fixes. Classifying before acting is the entire point; an agent that "improves retrieval" for a content gap will change nothing and report progress.
3. **Metric review** against PRD §8.

### 8.1a Tooling for these activities

Activity 2 is the only one that is *buildable* — activities 1 and 3 need real
traffic, which a pilot has not yet produced. Two scripts cover the work:

| Script | Covers | Notes |
| --- | --- | --- |
| `scripts/content_gaps.py` | Activity 2, and FR-30 (deferred from 4.9) | Classifies a failed query before you act on it |
| `scripts/pilot_metrics.py` | Activity 3 | Reports the §8.2 gate as PASS / FAIL / **UNMEASURED** |

**`content_gaps.py` — classify before acting.** §8.1 is explicit that a content gap
and a retrieval gap need different fixes and that an agent which "improves retrieval"
for a content gap "will change nothing and report progress". The classifier decides
which you have, by differencing the production config against progressively more
permissive ones and isolating the two halves of hybrid search:

| Classification | Fix |
| --- | --- |
| `content_gap` | Write the document |
| `threshold_gap` | Re-sweep `retrieval_threshold` |
| `lexical_gap` | Chunking or the keyword index |
| `semantic_gap` | Embedding model or chunk size |
| `ranking_gap` | Fusion weights, `rerank_k`, `top_k` |
| `answer_gap` | Generation or prompt — *not* retrieval |

```bash
python scripts/content_gaps.py                                   # worst queries in the log
python scripts/content_gaps.py --query "What is the escalation policy?"
python scripts/content_gaps.py --from eval/dataset.jsonl --json
```

**`pilot_metrics.py` — the gate that refuses unearned passes.** Every §8.2 metric is
`PASS`, `FAIL`, or `UNMEASURED`, and the third is the point. A rate computed from
too few *distinct* queries is reported with its number but marked unmeasured, because
a tool that prints "100% thumbs up" from four votes on one question converts an
absence of evidence into a gate clearance. The sample-size floor is
`--min-samples` (default 20). It also flags a *near-zero* refusal rate as a failure,
per §8.2's warning that this is "the most damaging failure mode here, because it is
invisible to the user".

```bash
python scripts/pilot_metrics.py
python scripts/pilot_metrics.py --min-samples 30 --json
```

**Current state: all four traffic metrics are UNMEASURED.** `query_logs` holds 6 rows
across 2 distinct queries — local test residue, not a pilot. The exit gate for this
phase is genuinely outstanding and cannot be closed by tooling; it needs a real
audience. The two eval-set metrics (citation correctness, groundedness) are computed
by `eval/run_eval.py` and are not re-run by these scripts.

### 8.2 Pilot gate

Ship to a wider audience only when:

| Metric | Target | Source |
| --- | --- | --- |
| Answered-helpfully (thumbs up) | ≥ 70% | `QueryLog.feedback` |
| Refusal rate | 10–30% | `QueryLog.abstained` |
| Unanswered-question rate | ≤ 15% | `QueryLog` |
| TTFT p95 | ≤ 5 s | `QueryLog.ttft_ms` |
| Citation correctness | ≥ 0.95 | eval set, re-run |
| Groundedness | ≥ 0.90 | eval set, re-run |

**Refusal rate is a health signal, not an error.** The PRD puts the healthy band at 10–30%. A rate near zero usually means the threshold is too permissive and the system is answering from irrelevant text — the most damaging failure mode here, because it is invisible to the user. Alert on it, do not celebrate it.

---

## 9. Cross-phase reference

### 9.1 Requirement → phase

| Phase | Requirements |
| --- | --- |
| 0 | NFR-8 (partial), NFR-10 (seam) |
| 1 | FR-1…FR-11, FR-27 |
| 2 | FR-12…FR-16, G3, A-5 |
| 3 | FR-17…FR-22, FR-28, G2, G6, NFR-3 |
| 4 | FR-20, FR-23, FR-25, FR-27, FR-29, FR-31, FR-34, NFR-9 |
| 5 | NFR-1, NFR-2, NFR-5, NFR-8 |
| 6 | FR-30, all PRD §8 metrics |

**Not built in any phase:** FR-24 (turn condensation, P2) and FR-30 (content gaps, P2) — both deferred per `architecture.md` §9. Build them only if pilot feedback shows they matter.

**Unresolvable without input:** NFR-6 (no third-party training) and NFR-7 (cost ceiling). Both depend on provider and compliance answers — PRD open questions 7 and 8.

### 9.2 Phase dependencies

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
           corpus      labelled   generation  UI         NFRs      real
           indexed     data       + cites                verified  traffic
```

The two orderings worth defending:

- **Phase 2 before Phase 3.** Retrieval before generation. Otherwise prompt work compensates for a retrieval defect it cannot fix, and you never learn whether the retriever or the prompt was at fault.
- **Eval set before tuning.** `implementation.md` §4.2. The most common way this project fails quietly.

### 9.3 What to do when a phase gate fails

1. **Do not proceed to the next phase.** The gate exists because downstream work built on a failing foundation is wasted effort.
2. **Get the measurement first.** "Retrieval feels bad" is not actionable; `recall@10 = 0.62` is.
3. **Check whether the gate or the target is wrong.** If no threshold satisfies both §4.4 targets, the PRD target pair is the problem. Escalate as a PRD change rather than shipping the closest miss.
4. **Record the failure** in the phase's doc. A gate that has never failed has not been tested.

---

## 10. Quick start

```bash
# Phase 0
git init && make bootstrap
docker compose up -d
cp .env.example .env        # fill in provider keys
make check
curl localhost:8000/health

# Phase 1
python scripts/ingest_corpus.py --dir ./samples --stats

# Phase 2
python eval/build_dataset.py --dir ./samples --questions 200
python eval/run_eval.py --stage all

# Phase 3+
python eval/run_eval.py --stage generation

# Phase 4 (web UI)
npm --prefix web install
# Chroma is a derived index: ingestion never writes to it, so a fresh checkout
# needs an explicit sync before VECTOR_STORE=chroma returns anything.
python scripts/sync_chroma_index.py
npm --prefix web run dev
```

The API must be reachable from the browser as a separate origin. It allows only
`http://localhost:3000` by default; set `CORS_ALLOW_ORIGINS` for any other host.

Then feed each phase's prompt block from §2.5, §3.6, §4.6, §5.7, §6.6, §7.5, and verify the gate before continuing.

---

## Appendix — Source documents

| Document | Role |
| --- | --- |
| `docs/PRD.md` | What to build and why. Requirements, metrics, open questions. |
| `docs/architecture.md` | How it is built. Design decisions and their rationale. |
| `docs/implementation.md` | This file. Phase order, tasks, verification. |
| `docs/eval_results.md` | Created in Phase 2. Evidence for every quality claim. |
| `docs/perf_report.md` | Created in Phase 5. Measured vs. `architecture.md` §7.4 budget. |

**Still open.** The PRD's `§13` anchor is broken — assumptions live in `architecture.md` §12 (A-1…A-7), and PRD open questions 5, 6, 7, 8 remain unanswered. Stack choices in §1 above are the reference set, not decisions. Product selection (vector store, embedding model, generation model) is the most expensive thing to reverse, so settle it before Phase 1.
