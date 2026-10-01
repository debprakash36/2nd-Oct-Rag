# Architecture — RAG Chatbot

**Status:** Draft
**Derived from:** `docs/PRD.md` (v1)
**Scope:** Technical design satisfying the PRD's functional (FR-1…FR-34) and non-functional (NFR-1…NFR-10) requirements.

> **Note on the PRD:** §4 of the PRD line references "[§13](#13-assumptions--open-questions)" for assumptions, but the assumptions landed in **§10 Open Questions**. The anchor is broken. Assumptions A-1…A-5 referenced throughout this document are listed in [§12](#12-assumptions-carried-from-the-prd), and they are inferences, not confirmed decisions.

---

## 1. Architecture Drivers

Design pressure comes from six places in the PRD. Every significant decision below traces to one of them.

| Driver | Source | Architectural consequence |
| --- | --- | --- |
| **Groundedness is a release gate, not a nice-to-have** | NFR-3, G2, FR-17/18 | Retrieval and generation are separately tunable; citation validation is a hard gate between the model and the user |
| **TTFT ≤ 5 s p95** | NFR-1, G5 | Streaming end-to-end, not buffered; wide-then-narrow retrieval; no synchronous work in the request path that could be avoided |
| **1M chunks, 10k docs** | NFR-4 | Vector store and keyword index are separate systems with different strengths; filters must be pushed down into them |
| **Providers must be swappable** | NFR-10, NG3 | Every model call goes through a provider interface. No vendor SDK types leak past the interface boundary |
| **Corpus is untrusted input** | FR-32, FR-22, NFR-5 | Extraction is sandboxed; retrieved text is data, never instructions; rendering escapes everything |
| **New docs ship continuously** | G4, FR-7/8 | Indexing is an async job; index and corpus state are eventually consistent, and the UI must show that state honestly (FR-5) |

---

## 2. System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                        User's browser                           │
│   Chat UI · Source viewer · Admin console · SSE client          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Edge / Gateway                            │
│   TLS termination · rate limit (FR-31) · request ID (NFR-8)   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴─────────────────────┐
        ▼                                          ▼
┌───────────────────────┐              ┌──────────────────────────┐
│  Query path (sync)    │              │  Ingestion path (async)  │
│  Chat Orchestrator    │              │  Ingestion Worker pool   │
└───────────┬───────────┘              └──────────┬───────────────┘
            │                                         │
            ├──► Conversation Store                   ├──► Extract (sandboxed)
            ├──► Retriever ──► Vector Store          ├──► Clean
            │             └──► Keyword Index         ├──► Chunk
            ├──► Generator (LLM provider)            ├──► Embed
            ├──► Citation Validator                  ├──► Index
            └──► Query Log ◄──────────────────────────┘
```

Two paths, deliberately independent. Ingestion failing must never take the chat endpoint down, and chat traffic must never block on an index rebuild.

---

## 3. Component Design — Query Path

### 3.1 Request flow

```
Client                Orchestrator        Retriever         Store        Generator        Validator
  │                        │                  │                │             │                │
  │─ POST /chat/stream ───►│                  │                │             │                │
  │                        │─ guardrails: size cap (FR-34)     │             │                │
  │                        │─ load conversation (FR-23)        │             │                │
  │                        │─ resolve access filters (FR-16)   │             │                │
  │                        │─────────────────►│                │             │                │
  │                        │                  │─ rewrite query (3.2)          │                │
  │                        │                  │─ vector search ──────────────►│                │
  │                        │                  │─ keyword search ─────────────►│                │
  │                        │                  │─ fuse (RRF) + dedupe         │                │
  │                        │                  │─ rerank (FR-13)              │                │
  │                        │                  │─ threshold check (FR-14)      │                │
  │                        │◄─────────────────│                │             │                │
  │                        │                  │  [below threshold → refusal]  │                │
  │                        │───────────────────────────────────────────────────►│                │
  │                        │                  │  context + system prompt      │                │
  │                        │◄───────────────────────────────────────────────────│ streaming      │
  │                        │─ validate citations (FR-18) ─────────────────────────────────────► │
  │◄═ SSE token stream ═══►│                  │                │             │                │
  │   (buffered for validation)               │                │             │                │
```

**The streaming tension.** FR-19 requires streaming; FR-18 requires citation validation; NFR-1 requires fast TTFT. These conflict — you cannot stream a token to the user and then retract it because its citation was fabricated.

Resolution, in order of preference:
1. **Constrain generation so fabrication is rare.** The model may only emit citation IDs drawn from the retrieved set; invalid IDs are structurally impossible if the provider supports constrained decoding.
2. **Validate citations as they stream.** Buffer sentence-wise rather than token-wise. A sentence with an unresolvable citation is held back; a sentence that is clean flushes immediately. TTFT is unaffected because the first sentence is usually clean.
3. **Hold the whole answer only if sentence-level validation is insufficient** (e.g. an answer where a *later* sentence makes an *earlier* one unsupported).

Sentence-level is the right default: it preserves perceived streaming, and the failure it cannot catch — an earlier claim retroactively invalidated by a later one — is rare enough that a whole-answer buffer is a disproportionate latency cost. If the eval set (Phase 2) shows this is not rare, escalate to option 3 knowingly rather than by accident.

### 3.2 Query rewrite

Multi-turn support (G3, US-3) means the raw question is frequently unanswerable in isolation. *"What about refunds?"* has no retrievable content on its own.

Rewrite step, in order:
1. **Anaphora resolution** — replace pronouns and elliptical references with their antecedents from prior turns. Rule-based first; an LLM call only when rules are insufficient. The rule-based path keeps the common case off the critical path.
2. **Query expansion** — add synonyms and domain terms from the glossary, if one exists (open question 9).
3. **Standalone rewrite** — produce a self-contained question. This is an LLM call on the critical path; budget it and make it skippable.

Rewrite is logged verbatim alongside the original question (FR-28). A bad rewrite is otherwise invisible in production and looks like a retrieval failure.

### 3.3 Retrieval

Three stages, in the PRD's terms:

**Stage 1 — Wide candidate fetch (FR-12).**

Two searches run in parallel against the same logical corpus, each carrying the access filter pushed down into the store:

- *Vector:* ANN over chunk embeddings. Recalls paraphrases and concept matches.
- *Keyword:* BM25 over chunk text. Recalls exact identifiers — error codes, part numbers, surnames, dates, "Article 12". This is the failure mode that makes pure-vector search unacceptable for document Q&A, and it is why FR-12 is P0.

BM25 requires a lexical index maintained in step with the vector store. The two must not be allowed to drift: a chunk present in one and absent from the other produces confidently wrong answers. The chunk store (below) is the single source of truth for membership, and both indexes are written from it.

**Stage 2 — Fusion.** Reciprocal Rank Fusion. Chosen over score-combination methods (e.g. weighted min-max of normalized scores) because vector cosine similarity and BM25 are not on comparable scales; RRF only needs ranks, so no calibration is required or silently invalidated when the corpus changes.

**Stage 3 — Rerank (FR-13).** A cross-encoder scores each fused candidate against the query. Wide fetch (20–50), narrow final context (4–8). This is the single largest quality-per-unit-of-effort lever in the system, and the reason retrieval can afford to over-fetch.

**Threshold and budget (FR-14, FR-15).**
- If the top reranked score falls below a configured threshold, the pipeline short-circuits to refusal. No LLM call, no tokens, no chance of a plausible invention.
- Dedup by near-identical content — boilerplate repeated across documents will otherwise occupy the entire context window with one fact.
- Hard cap on total context tokens, with eviction of lowest-scoring chunks until it fits.

**The threshold is a measured parameter, not a guess.** It is calibrated against the eval set (150–300 labelled questions) to hit two simultaneous targets: recall@10 ≥ 0.85 and refusal rate in the 10–30% band. Set it too low and the system answers confidently from irrelevant text; too high and it refuses answerable questions and the support lead's deflection goal collapses. Both failure modes are visible only on labelled data, which is why the eval harness is a Phase 2 exit criterion rather than a later task.

### 3.4 Generation

- System prompt is **fixed and versioned** (FR-17). Not editable per-tenant, not assembled from retrieved content.
- Retrieved context is placed in a clearly delimited data block. Delimiters are part of the injection defense, not decoration (FR-32).
- **Access filters are applied before generation, never after (FR-16).** Filtering the answer after the fact is a security bug: the unauthorized text has already been read into the model's context and can leak through paraphrase.
- Output length preset per FR-21, selected by a parameter rather than a separate prompt.
- Streaming from the provider, forwarded rather than buffered end-to-end.

### 3.5 Citation validation

The mechanism that makes FR-18 and G6 true rather than aspirational.

```
for each sentence emitted:
    extract citation markers [n]
    if all n ∈ retrieved_chunk_ids:  flush to client
    else:                              strip invalid markers, flush, flag
```

Two rules make this hold:
1. **Closed ID set.** Markers are integers 1..K indexing the retrieved chunks, mapped server-side to real chunk IDs. The model never sees or emits internal IDs, so it cannot leak storage details.
2. **Fabricated references are removed, not displayed.** An answer with an invalid marker loses that marker. An answer where *no* marker resolves is treated as ungrounded and converted to a refusal. This is stricter than the PRD's literal "strip" wording and is a deliberate interpretation — an answer with zero valid citations is a hallucination wearing a citation costume, and the PRD's own NFR-3 (groundedness ≥ 0.9) is not met by showing it.

Stripped markers are logged, not silently dropped. A rising strip rate is the earliest available signal of model or prompt degradation (NFR-8).

---

## 4. Component Design — Ingestion Path

```
Upload ──► Validate (FR-2) ──► Extract (FR-3) ──► Chunk (FR-9) ──► Embed ──► Index ──► Mark live
  │            │                  │                 │              │         │          │
  │            │                  │                 │              │         │          └─ doc state
  │            │                  │                 │              │         └─ vector + BM25 + chunk store
  │            │                  │                 │              └─ batched
  │            │                  │                 └─ config-driven strategy
  │            │                  └─ sandboxed
  │            └─ reject with per-file error
  └─ content hash → dedupe (FR-4) → version supersede (FR-8)
```

### 4.1 Document lifecycle

The state machine FR-5 implies, made explicit because the admin UI (FR-27) and the index depend on it:

```
pending ──► extracting ──► chunking ──► embedding ──► indexing ──► live
   │            │             │             │            │
   └────────────┴─────────────┴─────────────┴────────────┴──► failed (with reason)
                                                                      │
live ──► superseded (FR-8)      live ──► disabled (FR-7) ──► live
live/disabled ──► deleted (FR-7, index cleanup)

chunking ──► duplicate (FR-4)   duplicate ──► pending (admin promotes)
duplicate ──► deleted
```

Only `live` documents are retrievable. A `failed` document leaves no partial index entries — a half-indexed document produces answers that appear to come from a source that is not fully readable, which is worse than a missing document.

`superseded` and `disabled` are **tombstones, not deletes.** Retrieval filters on document state, so the old version stops being served immediately even though its chunks remain in the index. Physical removal is a separate background operation. This makes FR-8 (idempotent re-ingest) cheap and makes a bad ingest reversible without a full rebuild.

`duplicate` was added during implementation; the diagram originally omitted it. FR-4 said "flag as near-duplicate, offer to skip", and that was built as a *column* and nothing more: the upload was still chunked, embedded, indexed, and set to `live`. Eleven identical uploads of one document were all retrievable, so a single query matched the same passage up to eleven times — inflating the sources panel and spending the context budget on copies.

The distinction worth recording is between **flagging** and **neutralising**. `duplicate_of` was correctly populated on every duplicate; nothing downstream read it. `duplicate` is a tombstone in the same sense as `superseded`: the row and its `duplicate_of` pointer are kept, an admin can promote it to `pending` to index it deliberately, and retrieval excludes it meanwhile because both the SQL scan and the derived ANN projection filter on `live`.

A duplicate is detected at `chunking` because that is the first point the content hash exists — it is computed over chunked content, not file bytes (§4.4). This is also why a same-content re-upload is classified `same_document` rather than `duplicate`: a redelivered queue message must converge on itself, not record itself as a duplicate of itself.

### 4.2 Chunking

Semantic-first, config-driven (FR-9, PRD risk row 1):
1. Split on heading boundaries from the document outline.
2. Within a section, accumulate paragraphs into ~1000-token windows with ~200-token overlap.
3. Never split mid-sentence; allow a window to exceed the target if a single paragraph does.
4. Prepend a breadcrumb — document title + heading path — to the embedded text (FR-11). This is a cheap, large win: a chunk that says "must be filed within 30 days" is near-useless in isolation, and the breadcrumb restores its subject at embedding time.
5. A short parent-document summary is generated for long documents so a single chunk can be expanded into more context on demand. This addresses the PRD's context-overflow risk without adding it to every request.

Every chunk records `doc_id, chunk_index, page/section, char_start, char_end` (FR-6, FR-10). `char_start/char_end` is what makes citations land on the exact passage rather than the whole document — without it, FR-18 degrades into "links to a file."

The strategy is configuration, not code, so Phase 2 can compare strategies against the eval set without a code change per experiment.

### 4.3 Storage layout

Five stores, each with a distinct job. They are not interchangeable and collapsing them loses capability.

| Store | Contents | Why separate |
| --- | --- | --- |
| **Object store** | Raw uploaded files | Cheap, durable, large. Re-extraction without re-upload |
| **Chunk store** | Chunk text + metadata, authoritative membership | Single source of truth for what exists; vector and keyword indexes are both derived from it and can be rebuilt from it |
| **Vector store** | Embeddings + ANN index | ANN search; would have to degrade to full scan for lexical matching |
| **Keyword index** | Inverted index for BM25 | Exact-term matching, which vector similarity is bad at |
| **Document registry** | Per-file state, hash, version, ACL | Needed for the admin UI and for state filters in retrieval; too small and too hot for the others |

The chunk store as the authoritative source is the load-bearing choice: it means index drift is recoverable by rebuild rather than a permanent correctness bug.

**The vector store is derived, and nothing writes to it during ingestion.** With `vector_store=chroma`, the Chroma collection is empty on a fresh checkout and stays empty across document uploads until `scripts/sync_chroma_index.py` runs. This is deliberate: ingestion writes SQL, and projecting SQL into an ANN index is a separate, rebuildable step. The consequence to remember is that `VECTOR_STORE=chroma` returns nothing until that script has run, and the index drifts as the corpus changes — so re-run it (idempotent) after corpus changes.

SQL stays authoritative even when Chroma is selected: `ChromaVectorStore` enforces liveness and ACL from SQL at query time, and the sync script only upserts `live` chunks and deletes ids absent from SQL.

At small corpus sizes the brute-force SQLite store is likely both faster and simpler than an ANN index. Chroma earns its cost as the corpus grows toward the 1M-chunk target, not as a default.

### 4.4 Idempotency

FR-4 and FR-8 are the same mechanism viewed from two angles. Content hash on the extracted-and-chunked content, not the file bytes — re-saving a PDF changes the bytes without changing the text, and byte hashing would re-embed it for nothing.

```
hash(chunked content):
  new hash      → ingest, create doc
  existing hash, same doc_id, newer mtime → supersede old version, re-index (FR-8)
  existing hash, different doc_id        → mark duplicate, do not index (FR-4, §4.1)
```

---

## 5. Data Model

```
Document
  doc_id            uuid          primary key
  filename          string
  content_hash      string        hash of chunked content
  version           int           increments on supersede
  state             enum          pending|extracting|chunking|embedding|indexing
                                  |live|failed|superseded|disabled|deleted
  mime_type         string
  byte_size         int
  page_count        int?
  uploaded_at       timestamp
  indexed_at        timestamp?
  error_reason      string?
  acl_tags          string[]      pushed into retrieval filters (FR-16)

Chunk
  chunk_id          uuid          primary key
  doc_id            uuid          fk -> Document
  chunk_index       int
  text              string
  breadcrumb        string        doc title + heading path (FR-11)
  token_count       int
  char_start        int
  char_end          int
  page              int?          citation target (FR-6)
  section_path      string[]

Conversation
  conversation_id   uuid
  created_at        timestamp
  turns             Turn[]

Turn
  turn_id           uuid
  role              enum          user|assistant
  content           string
  citations         chunk_id[]
  query_id          uuid          fk -> QueryLog

QueryLog                                     (FR-28, FR-33)
  query_id          uuid
  trace_id          string        end-to-end request id (NFR-8)
  conversation_id   uuid?
  original_query    string        post-PII-redaction
  rewritten_query   string        logged separately (3.2)
  retrieved_ids     chunk_id[]
  scores            json          per-stage: vector, bm25, fused, reranked
  threshold_applied bool
  abstained         bool
  model             string
  prompt_version    string
  tokens_in/out     int
  ttft_ms           int
  total_ms          int
  citations_stripped int          degradation signal (3.5)
  feedback          enum?         up|down|null (FR-29)
```

`QueryLog` is the substrate for FR-30, the PRD's feedback loop, and the §8 metrics. The fields it carries are exactly the fields needed to answer, after the fact, whether a bad answer was a **content gap** (right document absent) or a **retrieval gap** (document present, not retrieved) — the classification the PRD's improvement loop requires and which is impossible without retained scores.

---

## 6. API Surface

Internal contracts, for the components the architecture defines. Kept minimal: v1 has no multi-tenancy (NG5) and no public API surface beyond the chat endpoint.

### `POST /chat/stream`
Request:
```json
{
  "conversation_id": "uuid | null",
  "message": "string",
  "answer_style": "concise | detailed"
}
```
Response: `text/event-stream`. Events:
```
event: sources     { chunk_id[], breadcrumb, page }   — sent before first token
event: token       { text }                          — sentence-flushed (3.1)
event: citation_warning { stripped_count }           — only when non-zero
event: done        { query_id, abstained, ttft_ms }
event: error       { code, message }                 — user-safe only
```

`sources` precedes the tokens deliberately: the sources panel (FR-20) renders immediately, so the user can see what the system is reading while it reads it. For a refusal this is the whole payload.

Errors: `400` size cap (FR-34), `429` rate limit (FR-31), `500` sanitized. Internal detail — provider errors, stack traces, chunk IDs — never crosses this boundary.

### `POST /admin/documents` (multipart)
`POST /admin/documents/{id}/disable` · `/enable` · `DELETE` (FR-7)
`GET /admin/documents` → list with state and chunk counts (FR-27)

### `POST /feedback`
`{ query_id, rating }` (FR-29)

### Ingestion worker interface
Not HTTP. A queue with at-least-once delivery, so every stage is **idempotent** — a redelivered message must converge to the same state, not double-embed. Enforced by the content-hash path in §4.4.

---

## 7. Cross-Cutting Concerns

### 7.1 Security

| Concern | Treatment | Traces to |
| --- | --- | --- |
| Prompt injection via documents | Retrieved text is data, never instructions. Fixed system prompt, clear delimiters, and escaped rendering of the result. Injection defense is layered, because a single layer is a single point of failure. Note there is deliberately no keyword/regex "output filtering" — that is not a security control; sanitization happens at the render boundary. | FR-32 |
| Unauthorized content leakage | Access filters pushed into both indexes **before** retrieval. Never post-filter the answer — the text was already in context. | FR-16 |
| Malicious uploads | Extraction in a sandboxed process: no network, read-only filesystem, CPU/memory/time caps, no shell interpolation on filenames. | NFR-5 |
| XSS from document content | Escape all document- and model-derived text at render. Never `innerHTML` an answer or a chunk preview. | FR-22 |
| Secret exposure | Secret store only; no secrets in the client bundle or repo. | NFR-5 |
| Transport | HTTPS only, enforced at the gateway. | NFR-5 |
| Cross-origin access | The web UI is a separate origin from the API, so CORS is an explicit allowlist (`cors_allow_origins`), defaulting to localhost:3000 only. A deployment that omits it refuses browser requests rather than serving any origin. Credentials are not allowed, which keeps a future auth change from silently turning a wildcard into a vulnerability. | NFR-5 |
| Authentication | **Not implemented.** Every conversation, feedback, chunk, and admin endpoint is unauthenticated. These must not be exposed publicly until authorization is added; the deployable surface is currently a local/trusted network. | NFR-5 |

### 7.2 Observability

Every request carries a `trace_id` from the gateway through retrieval, generation, validation, and logging (NFR-8). Structured events at each stage boundary.

Dashboards: TTFT and total latency percentiles; refusal rate (tracked as a health signal, not an error — the PRD puts the healthy band at 10–30%); citation strip rate; ingestion throughput and failure rate; provider error and rate-limit rates.

Two alerts worth having on day one:
- **TTFT p95 above 4 s** — degrades before it breaches the 5 s target, giving lead time.
- **Citation strip rate spiking** — indicates model or prompt drift, and it degrades answer quality *silently* from the user's perspective.

### 7.3 Provider abstraction (NFR-10)

Three interfaces, the seam that keeps the system portable:

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str], model: str) -> list[list[float]]: ...

class GenerationProvider(Protocol):
    def stream(self, messages, *, model, max_tokens, **opts) -> Iterator[str]: ...

class RerankerProvider(Protocol):
    def rerank(self, query: str, docs: list[str], top_k: int) -> list[ScoredDoc]: ...
```

Every call site depends on the interface. Model names, dimensions, and pricing are config. **The embedding dimension is pinned in config** — changing embedding models means a full re-embed and index rebuild, and the system should fail loudly on a dimension mismatch rather than return silently wrong similarity scores.

**Current selections** (decided in §13): embeddings are
`sentence-transformers/all-MiniLM-L6-v2` at **384 dimensions** via the HuggingFace
Inference API; generation is **Groq** streaming behind the same interface. The two
must be different vendors — Groq serves no embeddings endpoint — so neither provider
is a swappable detail of the other. Reranking is still the offline lexical stand-in.

The dimension guard fires on *width* only. It cannot detect a corpus holding vectors
from two different models at the same width, which is why a re-embed must be
all-or-nothing: a partial one leaves two vector spaces in one index where every
similarity score is meaningless while still returning plausible-looking neighbours.

### 7.4 Performance budget (NFR-1)

Indicative allocation against the 5 s TTFT target. **These estimates have since been
measured — see `docs/perf_report.md` and `docs/perf_stages.json`.** The table is kept
as the original design intent; the measured column is what the system actually does.

| Stage | Budget | Measured p95 | Notes |
| --- | --- | --- | --- |
| Gateway + guardrails + conversation load | 100 ms | not separately instrumented | |
| Query rewrite | 400 ms | 0.0 ms | Rule-based path resolves every corpus query; the LLM path is never taken |
| Vector + keyword search (parallel) | 150 ms | 19.5 / 19.1 ms | SQLite scan, a floor: pgvector is faster at scale |
| Fusion + rerank | 300 ms | 0.1 / 12.8 ms | |
| LLM time to first token | 3,500 ms | **not measured** | Dominates; the real lever on TTFT. Requires a real provider |
| Citation validation + flush | 50 ms | not separately instrumented | Sentence-buffered, not answer-buffered |
| **Headroom** | ~500 ms | ~1,270 ms | Against a 3,500 ms LLM term |

Retrieval totals 50.6 ms at p95, roughly 20× under its combined budget, so retrieval is
not where TTFT is won or lost. The unmeasured LLM term consumes ~70% of the target and
is the only figure that decides whether NFR-1 holds.

The LLM dominates, which is why TTFT is specified separately from total answer time (NFR-1) and why streaming is non-negotiable. Reranking is the one stage whose budget is genuinely a trade: 300 ms buys a large quality gain, and it is the first thing to cut if TTFT misses. Measurement has not yet forced that trade — rerank costs 12.8 ms — but it remains the designated lever if a provider's first-token latency leaves no headroom.

---

## 8. Deployment

Single deployable unit for the API and a separate worker deployment for ingestion. Splitting them means ingest throughput spikes cannot starve chat of resources.

| Component | Scaling | Notes |
| --- | --- | --- |
| Gateway + Orchestrator | Horizontal | Stateless; conversation state is external |
| Retriever | In-process with the orchestrator | Latency-critical; a network hop buys nothing at this scale |
| Ingestion workers | Queue-driven, autoscale | The only bursty component |
| Vector store / keyword index / chunk store / object store | Managed, or self-hosted equivalent | Availability is a dependency; see risk note |
| Document registry + QueryLog | Managed relational | Transactional state; the analytics load is read-heavy and separable from the transactional path |

**Why the orchestrator holds the retriever in-process:** every additional network hop in the request path is latency we cannot recover elsewhere, and NFR-1 has no slack. The retrieval *code* stays modular and separately testable; only its deployment placement is co-located. Revisit if the corpus grows well past NFR-4.

**Availability.** NFR-2 is 99.5% for the chat endpoint, but retrieval depends on the vector and keyword stores. If a store goes down, the system cannot answer — degrade to a clear "search is temporarily unavailable" rather than an answer from a partial index. A degraded-but-plausible answer is the failure mode this system exists to avoid.

---

## 9. Requirement Traceability

Every PRD requirement mapped to where it is satisfied. Requirements not appearing here are not yet designed.

| Req | Component / section | Req | Component / section |
| --- | --- | --- | --- |
| FR-1 | Ingestion validate (§4) | FR-18 | Citation validator (§3.5) |
| FR-2 | Ingestion validate (§4) | FR-19 | SSE stream (§3.1, §6) |
| FR-3 | Extract, sandboxed (§4, §7.1) | FR-20 | `sources` event (§6) |
| FR-4 | Content hash (§4.4) | FR-21 | `answer_style` param (§3.4) |
| FR-5 | Document state machine (§4.1) | FR-22 | Render escaping (§7.1) |
| FR-6 | Chunk metadata (§5) | FR-23 | Conversation store (§5) |
| FR-7 | Tombstone states (§4.1) | FR-24 | *Not designed — P2* |
| FR-8 | Version supersede (§4.4) | FR-25 | Conversation delete (§6) |
| FR-9 | Chunking (§4.2) | FR-26 | No cross-session store (§5) |
| FR-10 | Chunk metadata (§5) | FR-27 | Admin API (§6) |
| FR-11 | Breadcrumb (§4.2) | FR-28 | QueryLog (§5, §7.2) |
| FR-12 | Hybrid search (§3.3) | FR-29 | Feedback API (§6) |
| FR-13 | Rerank stage (§3.3) | FR-30 | *Not designed — P2; depends on QueryLog* |
| FR-14 | Threshold (§3.3) | FR-31 | Gateway rate limit (§7.1) |
| FR-15 | Dedup + token cap (§3.3) | FR-32 | Injection layering (§7.1) |
| FR-16 | Filter pushdown (§3.3, §7.1) | FR-33 | Redaction on write (§5) |
| FR-17 | Fixed system prompt (§3.4) | FR-34 | Size cap (§3.1) |

| NFR | Section | NFR | Section |
| --- | --- | --- | --- |
| NFR-1 | Perf budget (§7.4) | NFR-6 | Open question 7 — unresolved |
| NFR-2 | Availability (§8) | NFR-7 | Open question 8 — unresolved |
| NFR-3 | Eval gates (§3.3, §3.5) | NFR-8 | Observability (§7.2) |
| NFR-4 | Scale (§3.3, §4.3) | NFR-9 | UI layer — not in scope here |
| NFR-5 | Security (§7.1) | NFR-10 | Provider abstraction (§7.3) |

Three requirements are genuinely undesigned: **FR-24** (turn condensation) and **FR-30** (content-gap report) are both P2, and **NFR-6/NFR-7** cannot be settled until the provider and compliance answers exist. Everything else has a location.

---

## 10. Validation Plan

The architecture makes four claims that are testable, and each should be verified rather than assumed.

1. **Hybrid beats vector-only.** Run the eval set with keyword search disabled. If recall@10 does not drop measurably, the BM25 index is carrying its operational cost for nothing — and if it drops a lot, that is the strongest available evidence for the FR-12 decision.
2. **Reranking earns its 300 ms.** A/B rerank on vs off. This is the largest single latency cost and needs evidence, not intuition.
3. **Chunk strategy is not the dominant factor.** Compare at least two strategies (§4.2 is config-driven for exactly this). If the difference is small, PRD risk row 1 was overweighted and effort belongs elsewhere.
4. **The threshold achieves its band.** Sweep it across the eval set and confirm recall@10 ≥ 0.85 and refusal rate 10–30% *simultaneously*. If no threshold satisfies both, the assumption behind the PRD's target pair is wrong and that is a PRD-level conversation, not a tuning task.

Plus structural checks that need no corpus: state machine transitions are exhaustive and legal; ingestion is idempotent under queue redelivery; a disabled document is unretrievable immediately; a document with zero valid citations produces a refusal.

---

## 11. Build Order

Derived from the PRD's phases, mapped to components.

| Phase | Build | Gate |
| --- | --- | --- |
| 0 | Provider interfaces (§7.3), config, structured logging, trace IDs | Secrets managed; local run documented |
| 1 | Ingestion path (§4), storage layout (§4.3), document registry, admin API | 100 docs indexed; citations resolve to `char_start/char_end` |
| 2 | Retriever (§3.3) + eval harness | Recall@10 ≥ 0.85; §10 checks 1–4 run |
| 3 | Generator, streaming, citation validator (§3.4–3.5) | Groundedness ≥ 0.90; 100% of citations resolvable |
| 4 | Chat UI, conversation store, feedback, QueryLog | End-to-end; refusals and content gaps visible |
| 5 | Guardrails, load test, security review | NFR-1, NFR-5, NFR-9 met |
| 6 | Pilot, threshold re-tuning on real queries | PRD §8 metrics measured |

Phase 2 before Phase 3 is the ordering that matters most. Prompts are cheap to iterate and retrieval is not; tuning generation against an unmeasured retriever produces prompt work that compensates for a retrieval defect it cannot fix.

---

## 12. Assumptions Carried from the PRD

Listed because the architecture rests on them and none are confirmed. The PRD's broken `§13` anchor (§ note, top) should point here.

| ID | Assumption | If wrong |
| --- | --- | --- |
| A-1 | Corpus is text-based PDF/DOCX/MD/HTML, no OCR needed | Extraction rewritten; a vision/OCR path enters ingestion |
| A-2 | Session-scoped conversation is sufficient (FR-26) | A memory store and a retrieval-scoped memory layer enter the query path |
| A-3 | No per-user access filtering in v1 (FR-16 is P1) | ACL propagation through chunking, both indexes, and every filter becomes blocking |
| A-4 | Groundedness ≥ 0.9 is achievable with off-the-shelf models | Fine-tuning enters scope, contradicting NG3 |
| A-5 | A labelled eval set can be built by hand | Every quality gate in the PRD becomes unverifiable and must be renegotiated |
| A-6 | Deployment target permits managed vector/keyword stores | Self-hosting effort in §4.3 and §8 grows substantially |
| A-7 | 1M chunks fits a single-node vector index | Sharding, which changes §3.3 filters and §4.3 membership guarantees |

---

## 13. Decisions Requiring Input

### Settled

| Decision | Choice | Blocks | Notes |
| --- | --- | --- | --- |
| Embedding model and dimension | `sentence-transformers/all-MiniLM-L6-v2`, **384 dimensions** | §7.3, re-embed cost | Answers the *self-hosting* half of PRD open question 8 for embeddings: a hosted open-weights model is not required. The *budget* half is still open. See below. |

**Why MiniLM at 384.** Three properties made it the choice over the alternatives:

- **The dimension is already 384**, so `EMBEDDING_DIM` does not change. The
  alternative 384-dim option (`BAAI/bge-small-en-v1.5`) is equally viable; 768-dim
  models (`all-mpnet-base-v2`, `e5-base-v2`) would have required a wider vector
  column in the initial migration. Holding 384 keeps the pgvector column as built.
- **Served through the HuggingFace Inference API, not local weights.** This avoids a
  ~2.5 GB `torch` + `transformers` install. The tradeoff is a network round-trip on
  every embed, and embedding runs on the *query* path, not only during ingestion —
  so that latency lands inside the NFR-1 5 s budget on every request. Measured
  retrieval is ~50 ms p95 offline (`docs/perf_stages.json`), which is the budget this
  spends out of.
- **Generation is a different vendor, necessarily.** Groq serves the LLM and does
  not offer an embeddings endpoint, so the two halves of the stack cannot share a
  provider. That is a constraint, not a preference, and it is why
  `GenerationProvider` and `EmbeddingProvider` are configured independently.

**Re-embedding is a schema change, not a config tweak.** Every stored vector must be
rebuilt: the current 297 vectors at 384 dimensions were produced by a *hash
function*, not a model, so they are the wrong kind of thing rather than merely the
wrong kind of vector. A width mismatch is caught loudly at query time
(`architecture.md` §7.3), which is why a partial re-embed fails fast instead of
silently degrading recall.

**What this decision does not settle.** PRD open question 8 has two halves. The
self-hosting half is answered *no* for embeddings — a hosted open-weights model is
acceptable, and 384 dimensions is a modest per-request cost. The **budget** half is
untouched: nothing here establishes a spend ceiling, and the answer path makes an
embedding call on *every query*, so volume is a cost driver that has not been
priced. Re-embedding the corpus with MiniLM will also produce the first real
groundedness and recall numbers, which is the input that decision needs.

### Still open

| Decision | Blocks | Depends on |
| --- | --- | --- |
| Vector store and keyword index products | §4.3, §8 | Open question 5 - existing index to reuse? |
| Generation and reranking models | §7.3, §7.4 budget | Open question 8 - budget and self-hosting requirement. Partly answered: generation is Groq, reranking is still the offline lexical stand-in. |
| Deployment topology | §8 | Open question 6 - cloud, on-prem, or VPC |
| Data residency and retention | NFR-6, FR-33 | Open question 7 - compliance constraints |
| Whether FR-24 and FR-30 are in v1 | §9 | P2 priority vs. Phase 4 scope |

The vector store decision is still open, and it is the expensive one. Nothing has
been chosen between pgvector, Chroma, and SQLite beyond the local default; the
checkout runs on SQLite and a derived Chroma projection exists beside it. Settling
this before any deployment avoids building reconciliation logic that a single-node
choice would have made unnecessary.
