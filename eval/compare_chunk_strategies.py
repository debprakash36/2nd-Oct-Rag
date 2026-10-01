"""§10 check 3: is chunk strategy the dominant factor?

`architecture.md` §10 lists four claims the design rests on, and this script
covers the third: *"Chunk strategy is not the dominant factor. Compare at least
two strategies."* `implementation.md` §4.2 makes chunking config-driven
specifically so this is a config change and not a code change.

## Why this does not re-ingest the corpus

The obvious implementation -- re-ingest under each strategy and re-run
`run_eval.py` -- is wrong for this comparison, and the reason is the labels. The
eval set's `relevant_chunk_ids` are `heading`-strategy chunk ids. Re-ingesting
under `fixed` gives every chunk a new id, so the labels no longer refer to
anything and recall collapses to zero for a reason that has nothing to do with
chunking. Relabelling per strategy is mandatory either way, so the *only* thing a
full re-ingest adds over this script is the vector store round-trip.

This script therefore chunks the stored cleaned text in memory, relabels from the
dataset's stored `evidence` phrase (the same non-circular labelling rule
`build_dataset.py` uses), and runs the same ranking pipeline the retriever runs:
fake embeddings + cosine, BM25, RRF, dedupe, lexical rerank. It measures the
variable under test -- how chunk boundaries change what is retrievable -- without
confounding it with vector-store persistence.

Stated plainly because it bounds the claim: this is an in-memory approximation.
It does not exercise `pgvector`, the SQL-side ACL prefilter, or the persistence
path. Those are covered by `run_eval.py` and the retrieval tests. What it
establishes is whether chunk strategy moves recall, which is the question §10
check 3 asks.

Usage::

    python eval/compare_chunk_strategies.py
    python eval/compare_chunk_strategies.py --strategies heading fixed paragraph
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, DocumentState
from app.db.session import session_scope
from app.ingest.chunk import ChunkConfig, Strategy, chunk_document
from app.ingest.keyword import tokenize
from app.providers.embedding import FakeEmbeddingProvider
from app.providers.rerank import LexicalRerankerProvider
from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.rerank import rerank_candidates
from app.retrieval.types import RetrievalCandidate, Stage

#: Same window `run_eval.py` measures recall over. Kept equal deliberately: a
#: strategy comparison that used a different k from the gate would not be
#: comparable to it.
K = 10
#: Fetch depth per stage, matching `RetrievalConfig.fetch_k`.
FETCH_K = 40


@dataclass
class StrategyScore:
    """One strategy's retrieval quality over the answerable questions."""

    name: str
    chunks: int
    mean_chunk_tokens: float
    answerable: int
    recall_at_1: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    unlabelled: int = 0
    #: chunk_id -> candidate, for this strategy.
    labels_by_question: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class _Doc:
    doc_id: str
    text: str
    title: str


def _load_docs(session: Session) -> list[_Doc]:
    rows = session.execute(
        select(Document.doc_id, Document.cleaned_text, Document.filename).where(
            Document.state == DocumentState.LIVE
        )
    ).all()
    out: list[_Doc] = []
    for doc_id, text, filename in rows:
        if not text:
            continue
        title = text.splitlines()[0].lstrip("# ").strip() if text else filename
        out.append(_Doc(doc_id=doc_id, text=text, title=title))
    return out


def _build_strategy(docs: Sequence[_Doc], strategy: Strategy) -> list[RetrievalCandidate]:
    """Chunk every document under `strategy` into retrievable candidates."""
    config = ChunkConfig(strategy=strategy)
    candidates: list[RetrievalCandidate] = []
    for doc in docs:
        for draft in chunk_document(doc.text, doc.title, config=config):
            candidates.append(
                RetrievalCandidate(
                    chunk_id=f"{doc.doc_id}:{draft.index}",
                    doc_id=doc.doc_id,
                    chunk_index=draft.index,
                    text=draft.text,
                    breadcrumb=draft.breadcrumb,
                    char_start=draft.char_start,
                    char_end=draft.char_end,
                    token_count=draft.token_count,
                )
            )
    return candidates


def _cosine(query: Sequence[float], doc: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(query, doc, strict=True))
    qn = math.sqrt(sum(a * a for a in query))
    dn = math.sqrt(sum(b * b for b in doc))
    if qn == 0 or dn == 0:
        return 0.0
    return dot / (qn * dn)


def _bm25(
    query_terms: Sequence[str],
    candidates: Sequence[RetrievalCandidate],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[RetrievalCandidate]:
    """Okapi BM25 over the strategy's chunks. Mirrors the production stage."""
    tokenized = [tokenize(c.text) for c in candidates]
    lengths = [len(t) for t in tokenized]
    avg = sum(lengths) / len(lengths) if lengths else 1.0
    doc_freq: Counter[str] = Counter()
    for terms in tokenized:
        doc_freq.update(set(terms))
    n = len(candidates)

    scored: list[tuple[float, RetrievalCandidate]] = []
    for candidate, terms in zip(candidates, tokenized, strict=True):
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            norm = 1.0 - b + b * (len(terms) / avg if avg else 1.0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * norm)
        if score > 0:
            candidate.scores[Stage.KEYWORD] = score
            scored.append((score, candidate))
    scored.sort(key=lambda s: (-s[0], s[1].chunk_id))
    out = [c for _, c in scored[:FETCH_K]]
    for rank, c in enumerate(out, start=1):
        c.ranks[Stage.KEYWORD] = rank
    return out


def _retrieve(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    vectors: dict[str, list[float]],
    embeddings: FakeEmbeddingProvider,
    reranker: LexicalRerankerProvider,
) -> list[str]:
    """The production ranking pipeline, in memory, over one strategy's chunks."""
    qvec = embeddings.embed([query], model="fake-embed-v1")[0]

    scored = [(_cosine(qvec, vectors[c.chunk_id]), c) for c in candidates]
    scored = [(s, c) for s, c in scored if s > 0]
    scored.sort(key=lambda s: (-s[0], s[1].chunk_id))
    vector_hits = [c for _, c in scored[:FETCH_K]]
    for rank, c in enumerate(vector_hits, start=1):
        c.scores[Stage.VECTOR] = scored[rank - 1][0]
        c.ranks[Stage.VECTOR] = rank

    keyword_hits = _bm25(tokenize(query), candidates)
    fused = reciprocal_rank_fusion(
        {Stage.VECTOR: vector_hits, Stage.KEYWORD: keyword_hits}
    )
    deduped = deduplicate(fused)
    ranked = rerank_candidates(
        query, deduped[:FETCH_K], reranker, top_k=FETCH_K
    )
    return [c.chunk_id for c in ranked[:K]]


def evaluate_strategy(
    docs: Sequence[_Doc], dataset: Sequence[dict], name: Strategy
) -> StrategyScore:
    candidates = _build_strategy(docs, name)
    embeddings = FakeEmbeddingProvider(dim=get_settings().embedding_dim)
    vectors = {
        c.chunk_id: v
        for c, v in zip(
            candidates,
            embeddings.embed([c.text for c in candidates], model="fake-embed-v1"),
            strict=True,
        )
    }
    reranker = LexicalRerankerProvider()

    score = StrategyScore(
        name=name,
        chunks=len(candidates),
        mean_chunk_tokens=(
            sum(c.token_count for c in candidates) / len(candidates)
            if candidates
            else 0.0
        ),
        answerable=0,
    )
    reciprocal = 0.0
    for record in dataset:
        if record["expect_abstain"]:
            continue
        evidence = record.get("evidence", "")
        needle = evidence.lower()
        relevant = {c.chunk_id for c in candidates if needle and needle in c.text.lower()}
        if not relevant:
            score.unlabelled += 1
            continue
        score.answerable += 1
        retrieved = _retrieve(record["question"], candidates, vectors, embeddings, reranker)
        hits = [i for i, cid in enumerate(retrieved, start=1) if cid in relevant]
        if hits:
            if 1 in hits:
                score.recall_at_1 += 1
            if min(hits) <= K:
                score.recall_at_10 += 1
                reciprocal += 1.0 / min(hits)

    n = score.answerable or 1
    score.recall_at_1 /= n
    score.recall_at_10 /= n
    score.mrr_at_10 = reciprocal / n
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="§10 check 3: chunk strategy comparison")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument(
        "--strategies", nargs="+", default=["heading", "fixed"],
        choices=["heading", "paragraph", "fixed"],
        help="strategies to compare (heading paragraph fixed)",
    )
    args = parser.parse_args()
    strategies: list[Strategy] = list(args.strategies)

    dataset = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not dataset:
        print(f"empty dataset at {args.dataset}", file=sys.stderr)
        return 1

    with session_scope() as session:
        docs = _load_docs(session)

    results: list[StrategyScore] = []
    for strategy in strategies:
        result = evaluate_strategy(docs, dataset, strategy)
        results.append(result)
        print(
            f"{strategy:12s} chunks={result.chunks:4d} "
            f"mean_tokens={result.mean_chunk_tokens:6.0f} "
            f"R@1={result.recall_at_1:.3f} R@10={result.recall_at_10:.3f} "
            f"MRR@10={result.mrr_at_10:.3f} "
            f"(unlabelled={result.unlabelled})"
        )

    if len(results) >= 2:
        best = max(results, key=lambda r: r.recall_at_10)
        worst = min(results, key=lambda r: r.recall_at_10)
        delta = best.recall_at_10 - worst.recall_at_10
        print(
            f"\n§10 check 3: best ({best.name}) - worst ({worst.name}) = "
            f"{delta:+.3f} recall@10"
        )
        print(
            "Interpretation: a small delta means chunk strategy is not the "
            "dominant factor and PRD risk row 1 was overweighted. A large delta "
            "means strategy selection matters and should be re-run at corpus "
            "scale before Phase 3 tunes generation against it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
