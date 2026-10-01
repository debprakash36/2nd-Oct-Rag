# PRD — RAG Chatbot

**Status:** Draft (assumptions-based)
**Source:** `docs/problemstat.txt` was empty; this PRD was derived from the stated intent "build a RAG chatbot". All assumptions are collected in [§13](#13-assumptions--open-questions) and must be confirmed before build starts.

---

## 1. Summary

Build a retrieval-augmented generation (RAG) chatbot that answers user questions about a private document corpus. The system ingests documents, indexes them into a vector store, retrieves relevant passages at query time, and generates grounded answers with citations back to the source documents. It refuses to answer when the corpus does not contain the answer.

**Primary goal:** Users get accurate, source-backed answers without reading the whole corpus manually.

**Non-goal:** Building a general-purpose assistant. The system is scoped to the ingested corpus only.

---

## 2. Goals & Non-Goals

### Goals
- G1. Answer questions grounded in ingested documents, with citations.
- G2. Refuse / abstain when the corpus does not support an answer.
- G3. Support conversational follow-up (multi-turn) that resolves pronouns and references against prior turns.
- G4. New documents can be added without re-engineering; incremental updates only.
- G5. Answer a useful question end-to-end (TTFT for first token) in under ~5 seconds at p95.
- G6. Every user-visible answer is traceable to retrieved source spans for audit/debug.

### Non-Goals
- NG1. Authoring or editing documents.
- NG2. Web search / open-domain answers.
- NG3. Fine-tuning or training models in-house.
- NG4. Native mobile apps in v1 (responsive web only).
- NG5. Multi-tenant billing, org workspaces, or admin analytics in v1.

---

## 3. Target Users & Scenarios

| Persona | Need | Primary scenario |
| --- | --- | --- |
| **End user** (primary) | Fast, trustworthy answers from a large doc set | "What is our refund policy?" with a link to the source clause |
| **Analyst** | Grounded research with citations they can verify | Multi-step question, exports answer + sources |
| **Content owner / Admin** | Keep the corpus current, control what is answerable | Uploads a new policy version, sets it live |
| **Support lead** (secondary) | Reduce ticket volume on repeat questions | Deflect common questions |

### Core user stories
- **US-1:** As a user, I ask a question and get an answer containing citations I can click through to the exact source location.
- **US-2:** As a user, when the answer is not in the documents, I get a clear "not found in the corpus" response rather than a plausible-sounding invention.
- **US-3:** As a user, I can follow up with "what about refunds?" and the system knows I mean refunds *for digital products*.
- **US-4:** As an admin, I upload a new file and can confirm the system has indexed it before announcing it.
- **US-5:** As a support lead, I can review anonymized query logs to see what users cannot get answers for, which drives content gaps.

---

## 4. System Scope

### In scope (v1)
1. Document ingestion from local file upload.
2. Text extraction + cleaning + chunking.
3. Embedding generation and vector index storage.
4. Hybrid retrieval (vector + keyword) with reranking.
5. LLM answer generation with inline citations.
6. Chat UI: streaming, message history, source viewer, thumbs feedback.
7. Answerability thresholding and graceful refusal.
8. Query + answer logging, and a feedback flag.

### Out of scope (v1) → roadmap
- OCR for scanned/image-only PDFs, audio, video.
- Rich document types (spreadsheets, presentations) beyond shallow text extraction.
- Automated web crawling / scheduled connectors.
- Per-user memory beyond the active session.
- Fine-tuning, evaluation harness beyond the metric set in §8.
- Mobile native apps.

---

## 5. Functional Requirements

### 5.1 Ingestion
| ID | Requirement | Priority |
| --- | --- | --- |
| FR-1 | Accept uploads: `.pdf`, `.docx`, `.txt`, `.md`, `.html` (assumption A-1). | P0 |
| FR-2 | Reject unsupported/oversized files (>25 MB) with a clear per-file error. | P0 |
| FR-3 | Extract text, strip boilerplate (headers/footers, repeated nav), normalize whitespace. | P0 |
| FR-4 | Detect near-duplicate documents via content hash; skip re-embedding identical content. | P1 |
| FR-5 | Show upload state per file: `pending → extracting → embedding → live` or `failed`. | P0 |
| FR-6 | Preserve source metadata: filename, path/title, page or section, upload time, doc version. | P0 |
| FR-7 | Support doc-level enable/disable and hard delete (with index cleanup). | P1 |
| FR-8 | Idempotent re-ingest: re-uploading a revised file supersedes the prior version. | P1 |

### 5.2 Chunking
| ID | Requirement | Priority |
| --- | --- | --- |
| FR-9 | Chunk on semantic boundaries (headings, then paragraphs) with ~1000-token windows and ~200-token overlap, tuned per doc type. | P0 |
| FR-10 | Every chunk carries its parent document ID and position, so citations can resolve. | P0 |
| FR-11 | Prepend a breadcrumb (doc title + heading path) to each chunk before embedding. | P1 |

### 5.3 Retrieval
| ID | Requirement | Priority |
| --- | --- | --- |
| FR-12 | Hybrid retrieval: vector similarity **and** keyword/BM25, results fused. | P0 |
| FR-13 | Cross-encoder rerank the fused candidate set down to the final context set. | P1 |
| FR-14 | Apply a relevance threshold; if the top score is below it, do not answer. | P0 |
| FR-15 | Deduplicate near-identical chunks and cap total context tokens. | P0 |
| FR-16 | Apply per-document and per-user access filters before generation, not after. | P1 |

### 5.4 Generation
| ID | Requirement | Priority |
| --- | --- | --- |
| FR-17 | Answer strictly from retrieved context; instruct the model to say it lacks the information otherwise. | P0 |
| FR-18 | Every factual claim carries a bracketed citation to a chunk ID; UI renders them as clickable markers. | P0 |
| FR-19 | Stream tokens to the client as they are produced. | P0 |
| FR-20 | Show the sources panel for every answer, whether or not the answer was a refusal. | P0 |
| FR-21 | Answer length presets: concise (~1 paragraph) and detailed; user-selectable. | P2 |
| FR-22 | Never render raw model output as executable content; escape all user- and document-derived text. | P0 |

### 5.5 Conversation
| ID | Requirement | Priority |
| --- | --- | --- |
| FR-23 | Persist conversation turns for the session; send last N turns as context. | P0 |
| FR-24 | Condense older turns into a running summary once a threshold is exceeded. | P2 |
| FR-25 | "New chat" resets history; user can delete a conversation. | P1 |
| FR-26 | No cross-session memory of the user in v1. | P0 |

### 5.6 Admin & Observability
| ID | Requirement | Priority |
| --- | --- | --- |
| FR-27 | Admin view: document list, status, chunk count, delete/disable. | P1 |
| FR-28 | Log every query with retrieval scores, chosen chunks, model, tokens, latency. | P0 |
| FR-29 | Thumbs up/down on each answer, stored against the query log entry. | P1 |
| FR-30 | Detect and group unanswered/low-confidence queries into a "content gaps" report. | P2 |

### 5.7 Guardrails
| ID | Requirement | Priority |
| --- | --- | --- |
| FR-31 | Rate-limit per user/IP; return a clear message when exceeded. | P1 |
| FR-32 | Strip or neutralize prompt-injection attempts embedded in documents; retrieval content is treated as untrusted data, never instructions. | P0 |
| FR-33 | PII redaction option on logged queries before storage. | P2 |
| FR-34 | Hard cap on input size per turn; reject with a user-friendly message. | P0 |

---

## 6. Non-Functional Requirements

| ID | Category | Requirement |
| --- | --- | --- |
| NFR-1 | Latency | p95 time-to-first-token ≤ 5 s; p95 full answer ≤ 20 s. |
| NFR-2 | Availability | 99.5% monthly for the chat endpoint. |
| NFR-3 | Quality | Groundedness ≥ 0.9 on the eval set; citation correctness ≥ 0.95 (assumption A-4). |
| NFR-4 | Scale | 10k documents / 1M chunks initial; 50 concurrent users. |
| NFR-5 | Security | Secrets in a secret store, never in code or client bundle; HTTPS only; uploads sandboxed during extraction. |
| NFR-6 | Privacy | Corpus never used to train third-party models; zero-retention terms with the model provider where available. |
| NFR-7 | Cost | Target ≤ $0.01 per 1,000 answered questions at expected mix. |
| NFR-8 | Observability | Structured logs with a request/trace ID end-to-end; error rate and refusal rate dashboards. |
| NFR-9 | Accessibility | Keyboard navigable, screen-reader labels on the source panel, WCAG 2.1 AA color contrast. |
| NFR-10 | Portability | Model and embedding providers swappable behind an interface; no vendor lock-in in the retrieval layer. |

---

## 7. Architecture

```
Client (Web UI)
   |  POST /chat/stream  (SSE)
   v
API / Orchestrator
   |-- Conversation state (turns, summary)
   |-- Guardrails  (rate limit, size cap, injection filter)
   v
Retriever
   |-- Query rewrite / expansion
   |-- Vector search  --->  Vector store (chunks + embeddings)
   |-- Keyword search  -->  BM25 index
   |-- Fusion (RRF) --> Reranker (cross-encoder) --> Top-K
   v
Generator (LLM, streaming, citations constrained to chunk IDs)
   v
Post-process (citation validation, refusal check) --> Stream to client --> Log

Ingestion pipeline (separate, async job)
   Upload --> Extract --> Clean --> Chunk --> Embed --> Index --> Mark live
```

**Key design decisions**
- **Retrieval and generation are separate services** so the retrieval layer can be re-tuned and evaluated without touching prompts.
- **Hybrid retrieval by default.** Pure vector search fails on exact identifiers, error codes, names, and dates — a real and common failure mode for document Q&A. Keyword search is not optional.
- **Citations are validated after generation.** Any citation token the model emits that is not in the retrieved set is stripped; the answer is not shown with a fabricated reference.
- **Reranking is a quality lever, not a scale problem.** Candidate fetch is wide (20–50), final context is narrow (4–8).

---

## 8. Success Metrics

### Product
| Metric | Target |
| --- | --- |
| Answered-helpfully rate (thumbs up) | ≥ 70% |
| Refusal rate (correctly abstained) | 10–30% — a healthy signal, not a failure |
| Unanswered-question rate | ≤ 15% of queries |
| Time to first token | ≤ 5 s p95 |
| Weekly active users | Set after pilot |

### Engineering
| Metric | Target |
| --- | --- |
| Retrieval recall@10 on eval set | ≥ 0.85 |
| Groundedness (no unsupported claims) | ≥ 0.90 |
| Citation correctness | ≥ 0.95 |
| Ingestion throughput | 1,000 pages/hour |
| API error rate | < 1% |

**Evaluation approach:** a hand-labelled eval set (assumption A-5) of 150–300 questions with expected source documents, run as a repeatable script on every retrieval/prompt/model change. Qualitative spot checks on top.

---

## 9. Delivery Plan

| Phase | Scope | Exit criteria |
| --- | --- | --- |
| **0. Foundations** | Repo, env config, provider abstractions, logging | Secrets managed; local run documented |
| **1. Ingestion** | FR-1…FR-11, admin view | 100 sample docs indexed correctly; citations resolve |
| **2. Retrieval** | FR-12…FR-16, eval harness | Recall@10 ≥ 0.85 on eval set |
| **3. Generation** | FR-17…FR-22, streaming, refusal | Groundedness ≥ 0.90; citations 100% resolvable |
| **4. Product surface** | Chat UI, history, feedback, FR-25…FR-30 | End-to-end happy path; content-gap report |
| **5. Hardening** | Guardrails, load test, security review, accessibility | NFR-1, NFR-5, NFR-9 met |
| **6. Pilot** | Small user group, feedback loop, tuning | Metrics measured against §8 |

**Key risks and mitigations**

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Bad chunking destroys answer quality | High | Test chunk strategies against the eval set before committing; make it config-driven |
| Model confabulates beyond the corpus | High | Strict prompting + post-generation citation validation + abstention threshold; treat groundedness as a release gate |
| Context window overflow on long docs | Medium | Hierarchical summarization for long documents; cap and trim retrieved context |
| Embedding cost / index rebuild time | Medium | Batch embedding, content-hash dedupe, incremental indexing only |
| Untrusted content in documents (injection) | Medium | Treat retrieved text as data, never instructions; fixed system prompt; output filtering |
| No eval set → flying blind | High | Build the eval set in Phase 2, before prompt tuning, and gate every change on it |
| Corpus growth degrades precision | Medium | Access filters, per-collection indexes, reranking; monitor retrieval score distribution |

---

## 10. Open Questions

1. What is the actual domain and corpus? (Determines chunking, file types, whether OCR is needed.)
2. Who are the real users — internal employees, or external customers?
3. Is the corpus multi-tenant, and does access control need to be per-user?
4. Is conversational memory across sessions required, or session-only is acceptable?
5. Is there an existing embedding/vector index to reuse, or do we start from zero?
6. What is the deployment target — cloud, on-prem, or VPC?
7. Are there compliance constraints (data residency, retention, HIPAA/GDPR)?
8. What is the budget ceiling for inference, and is a self-hosted open-weights model required?
9. Is there a branded voice/tone requirement, and a glossary of domain terms the model must use?
10. What is the definition of "good enough" for the pilot — the §8 target, or a different bar set by stakeholders?

---

## 11. Data & Feedback

- **Feedback signals:** thumbs up/down, citation clicks, follow-up rate on an answer, abandonment mid-answer, explicit "this didn't help" on a refusal.
- **Improvement loop:** weekly review of the lowest-rated and most-refused queries → either a content gap (missing document) or a retrieval gap (document exists, not retrieved). These two require different fixes, so classify before acting.
- **What we do *not* collect:** document contents beyond the corpus itself, user identity in analytics, conversation content without consent.

---

## Appendix A — Glossary

| Term | Meaning |
| --- | --- |
| **Chunk** | A retrievable unit of text, roughly 1000 tokens with overlap. |
| **Embedding** | A vector representation of a chunk used for similarity search. |
| **Hybrid retrieval** | Combining semantic (vector) and lexical (keyword) search. |
| **Reranking** | A second, more expensive model scoring candidates for final selection. |
| **Groundedness** | Whether every claim in the answer is supported by the retrieved context. |
| **Abstention** | The system correctly declining to answer. |
