# Retrieval Eval Guide

**Purpose:** how the Phase 2 eval set is built, how to run the harness, and how to
read the numbers. `docs/eval_results.md` holds the measurements; this file holds
the method.

**Source of truth:** `architecture.md` §3.3 and §10; `implementation.md` §4.

---

## 1. Why the eval set exists before tuning

`implementation.md` §4.2 and `architecture.md` §10 are both explicit that the
labelled set is the *input* to retrieval work, not an output. An agent asked to
"improve retrieval" without labels optimizes against its own guesses and reports
improvement that is not there. Every number in `eval_results.md` is only as
meaningful as the labels behind it, so the labelling rule is the first thing to
understand and the first thing to distrust if a result looks surprising.

---

## 2. Building the set

```bash
python scripts/make_corpus.py --out ./samples --count 100
python scripts/ingest_corpus.py --dir ./samples --stats
python eval/build_dataset.py --questions 200
```

`build_dataset.py` derives labels from the corpus generator's own ground truth
rather than from a human reading the text, and rather than from running the
retriever. That is deliberate on both counts:

* **Not from the retriever.** A label obtained by running retrieval records the
  retriever's own bugs as ground truth. This is the failure mode §4.2 warns about
  and it is invisible once it happens.
* **From the generator.** The documents are templates, so the topic, the scope,
  the stated figure, and the reference are all known by construction. There is no
  annotator disagreement to adjudicate.

### The labelling rule

A label is the set of `chunk_id`s whose text contains the document's **topic
sentence with `{days}` and `{scope}` substituted** — for example
`customers may request a refund within 14 days of purchase for Digital Goods`.
The substitution matters: carrying the raw template made the search look for the
literal string `{days}`, match nothing, and fall back to the sentence's first
word. Every direct label was then "any chunk mentioning *customers*", which is
why those questions used to score at chance.

Each question also stores its `evidence` phrase, so a label can be re-derived and
audited rather than trusted.

### The four classes

| Class | Share | What it tests | Must not be confusable with |
| --- | --- | --- | --- |
| `direct` | ~50% | A title-and-scope paraphrase. The vector half should carry this. | The topic title alone, which does not identify one document |
| `identifier` | ~12% | `POL-RRR-nnn` or an owner surname. Proves the keyword index exists (FR-12). | A semantic paraphrase |
| `multiturn` | ~12% | Turn 2 is unanswerable alone ("And how long is that window?"). Tests rewrite/anaphora. | A standalone question |
| `abstain` | ~26% | Not in the corpus. The only honest way to calibrate the threshold. | An answerable question with a hard label |

Minimums are enforced by `build_dataset.py` and re-printed on every run:
`abstain >= 20%`, `multiturn >= 10%`, `identifier >= 10%`.

### The question must be uniquely answerable

This is the property that took the longest to get right, and getting it wrong
silently destroys the gate. The corpus contains ~11–12 documents per topic
title. A question like *"Under the Refund Policy, what is the stated limit?"*
therefore has ~12 equally valid answers, while the dataset labels exactly one. No
retriever can score above ~1/12 on such a question, because there is nothing in
the question that selects one document over its siblings.

Every question must name the **scope** as well as the title. The corpus scopes
each document to a subject (`Digital Goods`, `Overnight Delivery`, …), carries
that scope in the title and the body, and the questions name it. Estimated
consequence of getting this wrong: see the "superseded measurements" note in
`eval_results.md`.

---

## 3. Running the harness

```bash
# Full configuration (the default). Uses RETRIEVAL_THRESHOLD.
python eval/run_eval.py

# §10 check 1: does the keyword stage earn its cost?
python eval/run_eval.py --ablate keyword

# §10 check 2: does rerank earn its latency?
python eval/run_eval.py --ablate rerank

# §10 check 4: sweep the threshold for the joint target
python eval/run_eval.py --sweep

# Per-question stage scores and both query forms, for failure attribution
python eval/run_eval.py --dump-scores out/scores.jsonl

# §10 check 3: chunk strategy comparison (separate script; see §5 below)
python eval/compare_chunk_strategies.py --strategies heading paragraph fixed
```

The ablations run at `score_threshold=0.0` on purpose. The threshold is
calibrated against **reranked** scores in `[0, 1]`; a no-rerank run gates on
**fused RRF** scores around `1/60`. Leaving the calibrated threshold in place
compares two different score *scales* and reports 0.0 recall as an artefact of
the arithmetic. Ablations compare rankings; `--sweep` measures the threshold.

---

## 4. Reading the numbers

* **recall@k** is measured over the top `k` of the ranking. The metric window is
  widened to 10 (`RetrievalConfig.for_measurement`) because `top_k` — the *context
  budget* — is 8, and scoring recall@10 against an 8-item list caps it at 0.80
  regardless of retriever quality.
* **recall@context** is the same idea at the delivered `top_k`. The gap between
  `recall@10` and `recall@context` is the fraction of answerable questions whose
  evidence is ranked but does not fit the context window.
* **refusal rate** counts *every* abstention, including refusals of answerable
  questions. Those are errors, but they are refusals the user saw, and the pilot
  gate (§8) measures the rate across all traffic. Counting only the
  expected-abstain questions that abstained reports 0% while a fifth of answerable
  questions are being refused.
* **MRR@10** is the mean of `1/rank` of the first relevant chunk, zero for a miss.
  It distinguishes "found it at rank 1" from "found it at rank 9" where recall
  cannot.
* **correct** is a stricter, per-question pass: abstain where expected, and a
  relevant chunk in the returned window where answerable.

### Failure attribution

`--dump-scores` writes per-candidate scores and ranks for every stage, plus the
original and rewritten query (FR-28). The classification the PRD's improvement
loop needs:

* **Content gap** — no stage returned a relevant chunk, and the keyword search
  matched nothing. The document is missing or the label is wrong.
* **Retrieval gap** — a relevant chunk was returned by some stage but did not
  make the final list. Fusion, rerank, or budget dropped it. A different fix.

Without retained per-stage scores these two are indistinguishable, which is why
the dump exists rather than being debug output.

---

## 5. §10 check 3 and the in-memory approximation

`compare_chunk_strategies.py` does **not** re-ingest the corpus. The reason is
the labels: `relevant_chunk_ids` are `heading`-strategy chunk ids, so
re-ingesting under `fixed` gives every chunk a new id and recall collapses to
zero for a reason unrelated to chunking. Relabelling per strategy is mandatory
either way, so a full re-ingest adds only the vector-store round-trip.

The script therefore chunks the stored cleaned text in memory, relabels from the
dataset's `evidence` phrase, and runs the same ranking pipeline (fake embeddings +
cosine, BM25, RRF, dedupe, lexical rerank). What it measures is how chunk
boundaries change what is retrievable — the question §10 check 3 asks.

**What it does not establish:** `pgvector`, the SQL-side ACL prefilter, or the
persistence path. Those are covered by `run_eval.py` and the retrieval tests. Its
absolute recall (~0.83) runs below `run_eval.py`'s (~0.89) because the in-memory
BM25 does not implement the production bigram/trigram matching that boosts exact
identifiers. Only the *delta between strategies* should be read from it.

---

## 6. Caveats that bound every claim

Stated here rather than only in the results file, because they change what the
numbers mean:

1. **The embedding provider is a hash-based fake** (`FakeEmbeddingProvider`). It
   has no semantics; it rewards token overlap. "Vector search" in these results
   is lexical similarity in a different code path, not semantic similarity.
2. **The reranker is a lexical stand-in** (`LexicalRerankerProvider`), not a
   trained cross-encoder. Check 2's numbers measure the *pipeline*, not what a
   real cross-encoder earns.
3. **The labels are generator-derived, not human-annotated.** This is stronger
   than hand-labelling in one respect — no annotation bias, no disagreement — but
   it is not what `implementation.md` §2.1 means by "hand-labelled". Assumption
   A-5 remains formally unconfirmed; a human review of a sample is the cheapest
   way to close it.
4. **The threshold is a property of the corpus, embeddings, and reranker.** It is
   measured, not fixed. Re-run `--sweep` after any change to those and update
   `eval_results.md`.
5. **The corpus is synthetic.** Recall on templated policies with named scopes is
   not recall on real documents. The *method* transfers; the number does not.

Re-run the gate against a real embedding model and a real cross-encoder before
treating any of this as settled.
