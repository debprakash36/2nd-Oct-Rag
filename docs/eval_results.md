# Retrieval and Generation Eval Results

> ## ⚠️ Every number in this file is a placeholder, not a gate
>
> The headline `recall@10 = 0.892` was measured with **`FakeEmbeddingProvider`** —
> a SHA-256 hash bucket, not a sentence encoder. It is **not** a valid exit gate and
> must not be cited as one.
>
> **What the same corpus measures with real embeddings:** `0.8378`, which does *not*
> clear the 0.85 target. The first real-embedding baseline is 0.8378.
>
> **The eval set is lexically biased.** Its templates quote document titles and
> scopes verbatim, so a bag-of-words hashing vectoriser scores well on it and a
> genuine semantic model is not rewarded for the thing it is actually good at. This
> benchmark cannot rank embedding models in either direction.
>
> **The real gate is to be set from pilot traffic.** It is deliberately left unset
> rather than replaced with a fresh number from the same biased set.

Evidence for the Phase 2 exit gate (`implementation.md` §4), the four claims in
`architecture.md` §10, and the Phase 3 exit gate (`implementation.md` §5). Method
and caveats live in `docs/eval_guide.md`; this file holds the measurements.

**Environment:** local SQLite store, offline providers (`FakeEmbeddingProvider`,
`LexicalRerankerProvider`). No PostgreSQL, no network model access. Every caveat
in `eval_guide.md` §6 applies to every number below.

**Corpus:** 113 live documents, 297 chunks, `strategy=heading`, `target=1000`,
`overlap=200`, `min=50`.

**Eval set:** 200 questions — `direct` 91 (45.5%), `abstain` 52 (26.0%),
`identifier` 30 (15.0%), `multiturn` 27 (13.5%). All §2.1 minimums pass.

**Date:** 2026-10-01. `PROMPT_VERSION` not applicable (Phase 2 is pre-generation).

---

## 1. Exit gate

**Recall@10 ≥ 0.85: NOT ESTABLISHED.** The `0.892` quoted throughout this file was
produced by `FakeEmbeddingProvider` and is withdrawn as a gate result. On real
embeddings the same 113-document corpus scores **0.8378**, below the 0.85 target.
The gate is **to be set from pilot traffic** — no substitute number is offered here,
because any number drawn from this lexically biased set would repeat the error.

All figures below remain useful as *offline-pipeline* measurements — they exercise
the hybrid stages, the reranker and the abstain path. They are not evidence about
retrieval quality.

```
=== full (hybrid + rerank, threshold 0.10) ===
questions        200
correct          174 (87.0%)
recall@1         0.514
recall@5         0.892
recall@10        0.892  (fake embeddings; target >= 0.85 NOT ESTABLISHED)
MRR@10           0.703
recall@context   0.892
refusal rate     21.0%  (healthy band 10%-30%)
latency          mean 65.3 ms, p95 77.9 ms
by category:
  abstain      80.8%
  direct       93.4%
  identifier   93.3%
  multiturn    70.4%
failure classification: 16 retrieval gap(s) among failures
```

Reading this:

* `direct` and `identifier` both land at ~93%. Semantic-style questions and
  exact-identifier questions are found about equally well, which is what a
  hybrid is supposed to deliver.
* `multiturn` at 70.4% is the weakest answerable class. Turn 2 is dropped to
  "How long does that period last?" and resolution has to recover the scope from
  history. This is the class to watch when Phase 3 exercises rewrite with a real
  model.
* `abstain` at 80.8% means roughly one unanswerable question in five is answered
  anyway. Those are the 10 `answered but unanswerable` failures and they are the
  FR-14 failure mode; a higher threshold trades them against recall, which is
  what §4 sweeps.
* `recall@10` equals `recall@context`, so on this corpus the context budget never
  evicts the evidence: nothing relevant is ranked 9th or 10th while being pushed
  out of the top 8.

### Superseded measurement (kept deliberately)

Before the eval set and metric defects described in `eval_guide.md` §2 were
fixed, the same harness reported **recall@10 = 0.353, refusal 0.0%, and no
threshold satisfying both targets.** That result was reported as a PRD-level
finding. It was not one: it was a measurement defect, and recording it here is
the record §9.3 asks for.

The diagnostic that separated the two:

| Measurement | Value |
| --- | --- |
| Labelled **document** in top-8 (drove reported recall) | 30.7% |
| Labelled **topic** in top-8 | 85.0% |

The retriever found the right subject matter 85% of the time — at the gate
target — while the labels demanded one specific document out of ~12
indistinguishable same-title siblings. Three defects caused this:

1. Questions named only the title, so they had ~12 valid answers but one label.
2. `make_corpus.py` rendered `days + p`, so a single document stated 14 days in
   §1 and 18 in §5 — "the stated limit" had no answer even within one document.
3. `recall@10` was measured over a list truncated to `top_k = 8`, capping it at
   0.80 by construction.

Lesson, worth keeping: a failing gate is only evidence about *the system* once the
*instrument* has been shown to work. The 0.353 was measuring the eval set.

---

## 2. §10 check 1 — hybrid beats vector-only

Keyword stage disabled, threshold 0.0 so the comparison is of rankings.

| Configuration | recall@10 | MRR@10 | identifier recall |
| --- | --- | --- | --- |
| Hybrid (vector + keyword) | **0.892** | 0.703 | **93.3%** |
| Vector only | 0.642 | 0.497 | 6.7% |

*Fake embeddings.* The 0.148 gap between hybrid and vector-only is the honest part of
this table: the keyword stage is doing most of the work, which is consistent with the
lexical bias of the eval set and inconsistent with these being semantic results.

**Delta = +0.250 recall@10.**

The identifier column is the decisive evidence. Exact-identifier lookup collapses
from 93.3% to 6.7% without the keyword index — a 14× degradation on the class
FR-12 exists to serve. `architecture.md` §10 check 1 says a large drop is the
strongest available evidence for the FR-12 decision; this is that.

---

## 3. §10 check 2 — reranking earns its latency

Both legs threshold 0.0.

| Configuration | recall@10 | MRR@10 | mean latency |
| --- | --- | --- | --- |
| Rerank on | **0.892** | 0.703 | 64.6 ms |
| Rerank off | 0.662 | 0.454 | 54.7 ms |

**Delta = +0.230 recall@10 for +9.9 ms.**

**Caveat, and it is not small:** the reranker measured here is
`LexicalRerankerProvider`, a deterministic overlap scorer, not a cross-encoder.
This establishes that the rerank *stage* is load-bearing in this pipeline — it
recovers signal the hash-based fake embeddings cannot carry. It does **not**
establish that a trained cross-encoder earns its latency. §10 check 2 is open
until it is re-run against a real provider.

---

## 4. §10 check 3 — chunk strategy is not dominant

In-memory comparison; see `eval_guide.md` §5 for what this does and does not
measure. Only the delta between strategies is meaningful.

| Strategy | Chunks | Mean tokens | recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| heading | 297 | 810 | **0.838** | 0.799 |
| paragraph | 285 | 846 | 0.831 | 0.807 |
| fixed | 443 | 554 | 0.818 | 0.790 |

**Best − worst = +0.020 recall@10.**

A 2-point spread across three materially different strategies — including `fixed`,
which ignores document structure entirely — means chunk strategy is not the
dominant factor. PRD risk row 1 was overweighted; effort belongs on the reranker
and on the embedding model, both of which move recall far more (checks 1 and 2).
`heading` remains the default because it is not worse and it produces breadcrumbs
for citation.

---

## 5. §10 check 4 — the threshold achieves its band

**This table is `FakeEmbeddingProvider` output.** Its "yes/yes" columns are not a
real result and the conclusion drawn from them below is withdrawn.

```
 fake embeddings — looking for recall@10 >= 0.85 AND refusal rate in 10%-30%
 threshold  recall@10   refusal   R@10 OK   band OK
      0.00      0.892     0.0%       yes        no
      0.05      0.892    20.5%       yes       yes
      0.10      0.892    21.0%       yes       yes    <- chosen
      0.15      0.838    25.0%        no       yes
      0.20      0.804    35.5%        no        no
      0.25      0.595    53.5%        no        no
      0.30      0.392    69.5%        no        no
      0.35      0.291    77.5%        no        no
      0.40      0.149    88.0%        no        no
      0.45      0.074    94.0%        no        no
      0.50      0.068    94.5%        no        no
```

**With real embeddings, no threshold satisfies both targets.** On the current
124-document `rag.db`, recall@10 peaks at **0.8446** — short of 0.85 — while the
refusal band is only reachable from 0.05 upward. The two constraints never overlap,
which `threshold_history.jsonl` records as `recommended: null`, `joint_count: 0`.

Chosen: 0.10, unchanged. With no threshold satisfying both targets, 0.10 is retained
as the best refusal-band point that is not simply "return nothing" — it is not a
calibrated optimum and does not meet the recall target.

The knee at `0.15 → 0.20` above reflects how the *fake* score distribution is shaped
and should not be read as a property of the retriever.

---

## 6. Phase 3 — generation, citations, TTFT

```
python eval/run_eval.py --stage generation
python scripts/bench_ttft.py
```

```
=== generation ===
questions          200
groundedness       0.912  (target >= 0.90 PASS) [135/148 answerable]
citation resolvable 1.000  (269 emitted, 0 stripped)
strip rate         0.000
refusal rate       32.5%
TTFT               p50 2.47 ms, p95 2.94 ms
by category:
  abstain      19.2%
  direct       100.0%
  identifier   100.0%
  multiturn    14.8%

=== TTFT benchmark (n=100, style=concise) ===
provider           fake (model fake-gen-v1)
TTFT p50           58.3 ms
TTFT p95           71.8 ms
retrieval p95      69.5 ms
budget             p95 <= 5000 ms: PASS
```

**Exit gate:** groundedness **0.912 ≥ 0.90 PASS**; **100%** of emitted citations
resolvable (1.000); TTFT p95 **71.8 ms** against a local provider (budget 5 s).

Interpretation:

* **Citation resolvability is an invariant, not a measurement.** The validator
  strips any marker outside `1..K` before it is shown, so the emitted-marker
  resolvable rate is 1.0 by construction. The **strip rate is 0.000** because the
  offline provider can only ever emit markers for passages it was given; the
  strip-and-refuse mechanism is proven instead by `tests/generation/test_validator.py`
  and `tests/generation/test_injection.py`, which feed forged and out-of-range
  markers to the validator directly.
* **The offline provider bounds what groundedness can show.** It is an extractive
  reader, not a model: it cannot hallucinate, so `groundedness` measures the
  retrieval→prompt→validate→assemble *pipeline*. A real provider is dropped in
  behind `get_generation_provider` without changing the harness.
* **`multiturn` 14.8% is a provider limitation, not a pipeline defect.** The
  extractive provider selects on the original anaphoric question ("how long does
  that last?") and ignores the history messages, so it finds no overlapping terms
  and abstains. The history is present in the prompt for a real model to use.
* **TTFT is measured to the first *validated* token**, i.e. after the sentence
  buffer's first flush, which is the number a user experiences. With the offline
  provider it is pipeline overhead (dominated by retrieval); a network model's
  first-token latency is additive and must be re-measured.

**Defects this end-to-end check found, invisible to unit tests:**

1. **Sentence boundary whitespace was lost on assembly.** Splitting and `_tidy`
   strip each sentence, and the assembler concatenated them with no separator, so
   streamed answers read `...[1]This document...`. Fixed in
   `app/generation/assembly.py` (re-inserts one separating space at each
   boundary); locked by `test_sentences_are_separated_when_reassembled`.
2. **Trailing citation markers were detached from the sentence they support.** A
   marker after a terminator (`"...30 days. [2]"`) became the *next* sentence's
   opening, so validation attributed it to the wrong claim and the support check
   failed whenever two passages carried different values. Fixed in
   `app/generation/sentences.py` (`_TRAILING_MARKER` keeps the marker with the
   sentence it follows). This is what moved `identifier` from 50% to 100%.

---

## 7. Open items

* **Check 2 is provisional.** Re-run against a real cross-encoder.
* **Groundedness/citation numbers are provider-bounded.** Re-run
  `--stage generation` and `bench_ttft.py` against a real model; the offline
  provider only exercises the pipeline.
* **A-5 unconfirmed.** Labels are generator-derived, not human-annotated. A
  sampled human review is the cheapest closure.
* **No PostgreSQL measurement.** All numbers are SQLite. HNSW behaviour,
  pgvector recall, and the SQL-side ACL prefilter are untested.
* **Threshold is corpus-specific.** Re-sweep after any corpus, embedding, or
  reranker change, and update §5.
