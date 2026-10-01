"""Eval harness: measure retrieval against the labelled set (implementation.md 2.2, 2.5).

architecture.md §10 lists four claims the architecture makes that are testable and
should be verified rather than assumed. This harness exists to run all four:

1. **Hybrid beats vector-only.** Run with the keyword stage disabled. If recall@10
   does not drop measurably, BM25 is carrying its operational cost for nothing.
2. **Reranking earns its latency.** A/B rerank on vs off.
3. **Chunk strategy is not the dominant factor.** Compare two strategies.
4. **The threshold achieves its band.** Sweep it and confirm recall@10 ≥ 0.85 and
   refusal rate 10-30% *simultaneously*. architecture.md is explicit that if no
   threshold satisfies both, the target pair is wrong and that is a PRD-level
   conversation, not a tuning task — so `--sweep` reports that outcome as a
   failure rather than reporting the best point it found.

Usage::

    python eval/run_eval.py                       # full config
    python eval/run_eval.py --ablate keyword      # check 1
    python eval/run_eval.py --ablate rerank       # check 2
    python eval/run_eval.py --sweep               # check 4
    python eval/run_eval.py --dump-scores out.jsonl

Every query's per-stage scores are written to the dump so a miss can be
attributed to a stage (FR-28): a question no stage returned is a content gap; a
question the right stage returned but that lost at fusion or rerank is a
retrieval gap. The PRD's improvement loop needs that distinction and it is
impossible without retained scores.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import session_scope
from app.generation.assembly import AnswerAssembler, ChunkKind
from app.generation.prompt import (
    STYLE_PRESETS,
    AnswerStyle,
    ContextPassage,
    build_messages,
    new_nonce,
)
from app.generation.validator import MARKER_RE, validate_sentence
from app.ingest.keyword import tokenize
from app.providers.generation import get_generation_provider
from app.retrieval.retriever import RetrievalConfig, Retriever
from app.retrieval.types import AccessFilter, RetrievalResult

#: Depth of the recall@k metric, and therefore how deep the ranking must be
#: measured. Separate from the target below: `top_k` (the context budget) is
#: deliberately smaller than this, so recall@10 needs a wider window to measure.
RECALL_AT_10 = 10
#: §3.3 / §10 recall target. The phase exit gate.
RECALL_AT_10_TARGET = 0.85
#: §3.3 healthy refusal band. Tracked as a health signal, not an error.
REFUSAL_BAND = (0.10, 0.30)
#: §5 Phase 3 exit gate: grounded answers / answerable questions.
GROUNDEDNESS_TARGET = 0.90
#: A claim sentence counts as supported when at least this fraction of its
#: content tokens appear in the passages it cites. 1.0 would reject a faithful
#: paraphrase; the threshold is deliberately below certainty but far above the
#: noise a hallucination would produce.
SUPPORT_COVERAGE = 0.80


@dataclass
class QuestionOutcome:
    """Per-question result, retained so failures can be attributed."""

    qid: str
    category: str
    question: str
    expect_abstain: bool
    abstained: bool
    #: True when the pipeline refused where it should not have, or answered where
    #: it should have refused.
    correct: bool
    retrieved: list[str] = field(default_factory=list)
    relevant_hit_ranks: list[int] = field(default_factory=list)
    stages_run: list[str] = field(default_factory=list)
    timings_ms: float = 0.0
    rewrite_applied: bool = False
    #: FR-28: the query as asked and the query actually searched. Kept per
    #: question so a bad rewrite is separable from a bad retrieval in the dump.
    original_query: str = ""
    rewritten_query: str = ""
    #: Per-candidate scores and ranks at each stage. This is what makes a miss
    #: attributable: no stage returned the relevant chunk is a retrieval or
    #: content gap, whereas a stage returned it and a later stage dropped it is
    #: a fusion/rerank/budget defect. Both used to be lost.
    candidates: list[dict] = field(default_factory=list)
    #: How many candidates the context budget would actually deliver, i.e.
    #: `top_k` in the production config. Reported next to recall@10 because the
    #: measured window is wider than the delivered one.
    context_size: int = 0


@dataclass
class EvalReport:
    """Aggregate metrics plus the per-question outcomes behind them."""

    label: str
    total: int = 0
    correct: int = 0
    abstain_expected: int = 0
    abstain_actual: int = 0
    answerable: int = 0
    answered: int = 0
    #: Every abstention, on any question. This is what the refusal rate is
    #: computed from; `abstain_actual` counts only the correct ones.
    abstained_total: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    refusal_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    #: Answerable questions where a relevant chunk landed inside the *delivered*
    #: context (`top_k`), not just inside the wider measurement window. Recall@10
    #: can exceed this; the gap between the two is the fraction of questions the
    #: generator would have to answer without its evidence.
    recall_at_context: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)
    outcomes: list[QuestionOutcome] = field(default_factory=list)

    def gate_passed(self) -> bool:
        """§11 phase-2 gate: recall@10 ≥ 0.85.

        Refusal rate is reported but deliberately not part of the pass condition.
        §3.3 requires recall and refusal band to be satisfied *simultaneously*,
        which is a property of a calibrated threshold, not of a configuration —
        `--sweep` is where that is checked.
        """
        return self.recall_at_10 >= RECALL_AT_10_TARGET


@dataclass
class GenerationOutcome:
    """Per-question generation result (Phase 3, §5)."""

    qid: str
    category: str
    question: str
    expect_abstain: bool
    abstained: bool
    grounded: bool
    emitted_markers: int = 0
    stripped_markers: int = 0
    ttft_ms: float | None = None
    total_ms: float = 0.0
    answer: str = ""


@dataclass
class GenerationReport:
    """Phase 3 metrics: groundedness, citation correctness, strip rate, TTFT."""

    label: str
    total: int = 0
    answerable: int = 0
    emitted: int = 0
    grounded: int = 0
    abstained_total: int = 0
    emitted_markers: int = 0
    stripped_markers: int = 0
    refusal_rate: float = 0.0
    #: Grounded answers divided by answerable questions. This is the §5 exit
    #: gate: a refusal on an answerable question is a failure to answer, so it
    #: counts against groundedness rather than being excluded from it.
    groundedness: float = 0.0
    #: Fraction of all markers the model emitted that resolved to a retrieved
    #: passage. By construction this is 1.0 — the validator strips invalid
    #: markers before they are shown — which is the point: 100% resolvable is an
    #: invariant of the pipeline, not a property that must be hoped for.
    citation_resolvable_rate: float = 1.0
    #: Stripped markers as a fraction of all markers seen. The earliest signal of
    #: model or prompt drift (architecture.md 7.2).
    strip_rate: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)
    outcomes: list[GenerationOutcome] = field(default_factory=list)

    def gate_passed(self) -> bool:
        return (
            self.groundedness >= GROUNDEDNESS_TARGET
            and self.citation_resolvable_rate >= 1.0
        )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[idx]


def _sentence_supported(sentence: str, cited: Sequence[str]) -> bool:
    """Whether a sentence's content is covered by the passages it cites.

    This is a groundedness proxy a real model can fail: a hallucinated detail
    introduces content tokens that are absent from every cited passage, dropping
    coverage below `SUPPORT_COVERAGE`. For the offline extractive provider every
    sentence is copied from its passage, so it cannot fail — which is why the
    offline groundedness number measures the pipeline, not the model.
    """
    # Markers are citations, not content: tokenising "[1]" would add the digit
    # "1" as an uncovered token and make a short, perfectly grounded sentence
    # look unsupported.
    tokens = tokenize(MARKER_RE.sub("", sentence))
    if not tokens:
        return True
    covered = set()
    for passage_text in cited:
        covered.update(tokenize(passage_text))
    hits = sum(1 for token in tokens if token in covered)
    return hits / len(tokens) >= SUPPORT_COVERAGE


def evaluate_generation(
    session: Session,
    dataset: Sequence[dict],
    *,
    config: RetrievalConfig,
    settings: Settings,
    label: str = "generation",
    style: AnswerStyle = AnswerStyle.CONCISE,
    access: AccessFilter | None = None,
) -> GenerationReport:
    """Run retrieval then generation, scoring groundedness and citations (§5.6).

    The provider is whatever `get_generation_provider` selects. With the offline
    provider the pipeline is exercised end to end but the numbers describe the
    pipeline, not a language model; a real provider is dropped in behind the same
    factory without changing this harness.
    """
    report = GenerationReport(label=label)
    provider = get_generation_provider(settings)
    ttfts: list[float] = []
    per_category_total: dict[str, int] = {}
    per_category_grounded: dict[str, int] = {}
    max_tokens = STYLE_PRESETS[style].max_tokens

    for record in dataset:
        qid = record["qid"]
        category = record["category"]
        question = record["question"]
        expect_abstain = bool(record["expect_abstain"])
        history = record.get("history", [])

        retriever = Retriever(session, config=config, settings=settings)
        result = retriever.retrieve(
            question, history=history, access=access, config=config
        )
        report.total += 1

        if result.abstained or not result.candidates:
            report.abstained_total += 1
            report.outcomes.append(
                GenerationOutcome(
                    qid=qid,
                    category=category,
                    question=question,
                    expect_abstain=expect_abstain,
                    abstained=True,
                    grounded=False,
                )
            )
            per_category_total[category] = per_category_total.get(category, 0) + 1
            continue

        candidates = result.candidates
        passages = [
            ContextPassage(text=c.text, breadcrumb=c.breadcrumb, page=c.page)
            for c in candidates
        ]
        # The dataset stores prior turns as plain question strings; the prompt
        # wants chat messages. The retriever has already used this history to
        # resolve anaphora, so it is attached here for the generator's benefit
        # rather than for retrieval's.
        history_messages = [{"role": "user", "content": turn} for turn in history]
        messages = build_messages(
            question, passages, style, nonce=new_nonce(), history=history_messages
        )
        assembler = AnswerAssembler(passage_count=len(passages))

        start = time.perf_counter()
        first_token_ms: float | None = None
        parts: list[str] = []
        for chunk in assembler.run(
            provider.stream(
                messages,
                model=settings.generation_model,
                max_tokens=max_tokens,
                temperature=settings.generation_temperature,
            )
        ):
            if chunk.kind is ChunkKind.TOKEN:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                parts.append(chunk.text)
            elif chunk.kind is ChunkKind.REFUSAL:
                parts.append(chunk.text)
        total_ms = (time.perf_counter() - start) * 1000

        answer = "".join(parts)
        grounded = False
        if not assembler.abstained:
            report.emitted += 1
            if first_token_ms is not None:
                ttfts.append(first_token_ms)
            grounded = assembler.emitted_markers > 0
            if grounded:
                # Re-validate the finished text sentence by sentence so the
                # support check uses the same marker semantics the stream did.
                for sentence in _sentences(answer):
                    validated = validate_sentence(sentence, len(passages))
                    if not validated.markers:
                        continue
                    cited = [passages[n - 1].text for n in validated.markers]
                    if not _sentence_supported(validated.text, cited):
                        grounded = False
                        break
        else:
            report.abstained_total += 1

        report.emitted_markers += assembler.emitted_markers
        report.stripped_markers += assembler.stripped_count
        if grounded:
            report.grounded += 1
            per_category_grounded[category] = per_category_grounded.get(category, 0) + 1

        report.outcomes.append(
            GenerationOutcome(
                qid=qid,
                category=category,
                question=question,
                expect_abstain=expect_abstain,
                abstained=assembler.abstained,
                grounded=grounded,
                emitted_markers=assembler.emitted_markers,
                stripped_markers=assembler.stripped_count,
                ttft_ms=first_token_ms,
                total_ms=total_ms,
                answer=answer,
            )
        )
        if not expect_abstain:
            report.answerable += 1
        per_category_total[category] = per_category_total.get(category, 0) + 1

    answerable = report.answerable or 1
    total = report.total or 1
    markers = report.emitted_markers + report.stripped_markers
    report.groundedness = report.grounded / answerable
    report.refusal_rate = report.abstained_total / total
    report.citation_resolvable_rate = report.emitted_markers / markers if markers else 1.0
    report.strip_rate = report.stripped_markers / markers if markers else 0.0
    report.ttft_p50_ms = _percentile(ttfts, 0.50)
    report.ttft_p95_ms = _percentile(ttfts, 0.95)
    report.by_category = {
        cat: per_category_grounded.get(cat, 0) / n
        for cat, n in sorted(per_category_total.items())
    }
    return report


def _sentences(text: str) -> list[str]:
    """Sentence split for the groundedness re-check."""
    from app.generation.sentences import split_sentences

    return split_sentences(text)


def print_generation_report(report: GenerationReport, *, show_failures: int = 5) -> None:
    print(f"\n=== {report.label} ===")
    print(f"questions          {report.total}")
    print(
        f"groundedness       {report.groundedness:.3f}  "
        f"(target >= {GROUNDEDNESS_TARGET:.2f} "
        f"{'PASS' if report.groundedness >= GROUNDEDNESS_TARGET else 'FAIL'}) "
        f"[{report.grounded}/{report.answerable} answerable]"
    )
    print(
        f"citation resolvable {report.citation_resolvable_rate:.3f}  "
        f"({report.emitted_markers} emitted, {report.stripped_markers} stripped)"
    )
    print(f"strip rate         {report.strip_rate:.3f}")
    print(f"refusal rate       {report.refusal_rate:.1%}")
    print(
        f"TTFT               p50 {report.ttft_p50_ms:.2f} ms, "
        f"p95 {report.ttft_p95_ms:.2f} ms (offline provider; bench_ttft.py is "
        f"the NFR-1 measurement)"
    )
    if report.by_category:
        print("by category:")
        for cat, acc in report.by_category.items():
            print(f"  {cat:12s} {acc:.1%}")


def load_dataset(path: str | Path) -> list[dict]:
    """Read the JSONL dataset written by `eval/build_dataset.py`."""
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _ranks_of_hits(retrieved: Sequence[str], relevant: Sequence[str]) -> list[int]:
    """1-based positions at which relevant chunks appear in `retrieved`."""
    relevant_set = set(relevant)
    return [i for i, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant_set]


def evaluate(
    session: Session,
    dataset: Sequence[dict],
    *,
    config: RetrievalConfig,
    settings: Settings,
    label: str = "full",
    access: AccessFilter | None = None,
) -> EvalReport:
    """Run the dataset through the retriever and aggregate metrics.

    A fresh `Retriever` per query is deliberate. The retriever holds no mutable
    state between calls, and reusing one across a 200-question run would hide a
    state leak that only shows up under repeated use — which is precisely the
    condition the eval set exists to find.
    """
    report = EvalReport(label=label)
    latencies: list[float] = []
    per_category_total: dict[str, int] = {}
    per_category_hit: dict[str, int] = {}
    reciprocal_sum = 0.0
    context_hits = 0

    # Widen the returned list so recall@10 measures a ranking rather than a
    # truncation. See `RetrievalConfig.for_measurement`.
    context_size = config.top_k
    measured_config = config.for_measurement(RECALL_AT_10)

    for record in dataset:
        qid = record["qid"]
        category = record["category"]
        question = record["question"]
        expect_abstain = bool(record["expect_abstain"])
        relevant = record.get("relevant_chunk_ids", [])
        history = record.get("history", [])

        retriever = Retriever(session, config=measured_config, settings=settings)

        start = time.perf_counter()
        result: RetrievalResult = retriever.retrieve(
            question, history=history, access=access, config=measured_config
        )
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        retrieved = result.chunk_ids
        hits = _ranks_of_hits(retrieved, relevant)
        report.total += 1

        if expect_abstain:
            report.abstain_expected += 1
            if result.abstained:
                report.abstain_actual += 1
                correct = True
            else:
                # Answering an unanswerable question is the failure FR-14 exists
                # to prevent, so it counts as incorrect rather than as a free hit.
                correct = False
        else:
            report.answerable += 1
            if not result.abstained:
                report.answered += 1
            correct = not result.abstained and bool(hits)

        if result.abstained:
            # Every abstention counts toward the refusal rate, including one on a
            # question that *was* answerable. Those are errors, but they are
            # still refusals the user saw, and §8 measures the rate from
            # `QueryLog.abstained` across all traffic. Counting only the
            # expected-abstain questions that abstained reports 0% while the run
            # is visibly refusing a fifth of its answerable questions.
            report.abstained_total += 1

        if correct:
            report.correct += 1

        per_category_total[category] = per_category_total.get(category, 0) + 1
        if correct:
            per_category_hit[category] = per_category_hit.get(category, 0) + 1

        if hits:
            if 1 in hits:
                report.recall_at_1 += 1
            if any(r <= 5 for r in hits):
                report.recall_at_5 += 1
            if any(r <= RECALL_AT_10 for r in hits):
                report.recall_at_10 += 1
            if any(r <= context_size for r in hits):
                context_hits += 1
            best = min(hits)
            if best <= RECALL_AT_10:
                reciprocal_sum += 1.0 / best

        report.outcomes.append(
            QuestionOutcome(
                qid=qid,
                category=category,
                question=question,
                expect_abstain=expect_abstain,
                abstained=result.abstained,
                correct=correct,
                retrieved=retrieved[:RECALL_AT_10],
                relevant_hit_ranks=hits,
                stages_run=[s.value for s in result.trace.stages_run],
                timings_ms=latency_ms,
                rewrite_applied=result.rewrite_applied,
                original_query=result.original_query,
                rewritten_query=result.rewritten_query,
                candidates=[c.to_dict() for c in result.candidates],
                context_size=context_size,
            )
        )

    answerable = report.answerable or 1
    report.recall_at_1 /= answerable
    report.recall_at_5 /= answerable
    report.recall_at_10 /= answerable
    report.mrr_at_10 = reciprocal_sum / answerable
    report.recall_at_context = context_hits / answerable
    total = report.total or 1
    report.refusal_rate = report.abstained_total / total
    report.mean_latency_ms = statistics.fmean(latencies) if latencies else 0.0
    if latencies:
        ordered = sorted(latencies)
        idx = min(int(len(ordered) * 0.95), len(ordered) - 1)
        report.p95_latency_ms = ordered[idx]
    report.by_category = {
        cat: per_category_hit.get(cat, 0) / n
        for cat, n in sorted(per_category_total.items())
    }
    return report


def ranking_only(config: RetrievalConfig) -> RetrievalConfig:
    """A copy with the relevance threshold removed, for ablation runs.

    Ablations compare **rankings**. Leaving the calibrated threshold in place
    compares two different score *scales* instead: the threshold is calibrated
    against reranked scores in [0, 1], while a no-rerank run gates on RRF scores
    around 1/60, so the same cutoff abstains on every question and the ablation
    reports 0.0 recall as a property of the arithmetic rather than of ranking.
    Check 4 is where the threshold is measured, on the full configuration only.
    """
    import dataclasses

    return dataclasses.replace(config, score_threshold=0.0)


def ablation_configs(base: RetrievalConfig) -> dict[str, RetrievalConfig]:
    """The three configurations §10 checks 1 and 2 need."""
    import dataclasses

    base = ranking_only(base)
    return {
        "full": base,
        # Check 1: hybrid vs vector-only.
        "vector_only": dataclasses.replace(base, keyword_enabled=False),
        # Check 2: rerank off.
        "no_rerank": dataclasses.replace(base, rerank_enabled=False),
        # Check 1 inverted: keyword-only, to see the other half.
        "keyword_only": dataclasses.replace(base, vector_enabled=False),
    }


def sweep_threshold(
    session: Session,
    dataset: Sequence[dict],
    *,
    base_config: RetrievalConfig,
    settings: Settings,
    thresholds: Sequence[float] | None = None,
) -> list[EvalReport]:
    """§10 check 4: sweep the threshold and report every point.

    Every point is returned, including the best, because architecture.md is
    explicit that if no threshold hits recall ≥ 0.85 and a 10-30% refusal rate at
    the same time, the target pair is the problem. Surfacing only the best point
    would hide exactly the finding §10 asks for.
    """
    import dataclasses

    if thresholds is None:
        thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    reports: list[EvalReport] = []
    for threshold in thresholds:
        cfg = dataclasses.replace(base_config, score_threshold=threshold)
        report = evaluate(
            session, dataset, config=cfg, settings=settings, label=f"threshold={threshold:.2f}"
        )
        reports.append(report)
    return reports


def print_report(report: EvalReport, *, show_failures: int = 5) -> None:
    """Human-readable summary."""
    print(f"\n=== {report.label} ===")
    print(f"questions        {report.total}")
    print(f"correct          {report.correct} ({report.correct / max(report.total, 1):.1%})")
    print(
        f"recall@1         {report.recall_at_1:.3f}\n"
        f"recall@5         {report.recall_at_5:.3f}\n"
        f"recall@10        {report.recall_at_10:.3f}  "
        f"(target >= {RECALL_AT_10_TARGET:.2f} "
        f"{'PASS' if report.gate_passed() else 'FAIL'})"
    )
    print(f"MRR@10           {report.mrr_at_10:.3f}")
    print(
        f"recall@context   {report.recall_at_context:.3f}  "
        f"(answerable with evidence inside the delivered top-k)"
    )
    print(
        f"refusal rate     {report.refusal_rate:.1%}  "
        f"(healthy band {REFUSAL_BAND[0]:.0%}-{REFUSAL_BAND[1]:.0%})"
    )
    print(
        f"latency          mean {report.mean_latency_ms:.1f} ms, "
        f"p95 {report.p95_latency_ms:.1f} ms"
    )
    if report.by_category:
        print("by category:")
        for cat, acc in report.by_category.items():
            print(f"  {cat:12s} {acc:.1%}")

    failures = [o for o in report.outcomes if not o.correct]
    if failures:
        print(f"\nfailures ({len(failures)}), first {show_failures}:")
        for outcome in failures[:show_failures]:
            reason = (
                "refused but answerable"
                if not outcome.expect_abstain
                else "answered but unanswerable"
            )
            print(
                f"  {outcome.qid} [{outcome.category}] {reason}: "
                f"{outcome.question[:64]!r} retrieved={len(outcome.retrieved)}"
            )


def _classify_failure(report: EvalReport) -> str:
    """Content gap or retrieval gap, per the PRD's improvement loop.

    A question where no stage returned any relevant chunk, and the keyword search
    matched nothing at all, is a labelling or corpus problem. Where a relevant
    chunk was returned by some stage but never made the final list, it is a
    retrieval problem — fusion, rerank, threshold, or budget.
    """
    retrieval_gaps = 0
    for outcome in report.outcomes:
        if outcome.correct or outcome.expect_abstain:
            continue
        if outcome.retrieved:
            retrieval_gaps += 1
    return (
        f"{retrieval_gaps} retrieval gap(s) among failures"
        if retrieval_gaps
        else "all failures returned nothing (content gap or label error)"
    )


def run_check_1(
    session: Session, dataset: Sequence[dict], base: RetrievalConfig, settings: Settings
) -> tuple[EvalReport, EvalReport]:
    """§10 check 1: does the keyword stage earn its cost?"""
    configs = ablation_configs(base)
    full = evaluate(
        session, dataset, config=configs["full"], settings=settings, label="full (hybrid)"
    )
    vector_only = evaluate(
        session,
        dataset,
        config=configs["vector_only"],
        settings=settings,
        label="ablation: vector only",
    )
    return full, vector_only


def run_check_2(
    session: Session, dataset: Sequence[dict], base: RetrievalConfig, settings: Settings
) -> tuple[EvalReport, EvalReport]:
    """§10 check 2: does rerank earn its latency?

    Both legs run at threshold 0.0 so the comparison is of rankings, not of two
    different score scales. See `ranking_only`.
    """
    import dataclasses

    measured = ranking_only(base)
    full = evaluate(
        session, dataset, config=measured, settings=settings, label="full (rerank on)"
    )
    no_rerank = evaluate(
        session,
        dataset,
        config=dataclasses.replace(measured, rerank_enabled=False),
        settings=settings,
        label="ablation: rerank off",
    )
    return full, no_rerank


def dump_scores(report: EvalReport, path: str | Path) -> None:
    """Write per-question outcomes as JSONL for later analysis (FR-28)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for outcome in report.outcomes:
            fh.write(json.dumps(asdict(outcome), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval harness")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--questions", type=int, default=0, help="limit dataset size")
    parser.add_argument(
        "--stage",
        choices=["retrieval", "generation", "all"],
        default="retrieval",
        help="retrieval metrics (default), Phase 3 generation metrics, or both",
    )
    parser.add_argument("--ablate", choices=["keyword", "rerank"], default=None)
    parser.add_argument("--sweep", action="store_true", help="§10 check 4 threshold sweep")
    parser.add_argument("--all-checks", action="store_true", help="run §10 checks 1 and 2")
    parser.add_argument("--dump-scores", default=None, help="write per-question scores here")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    settings = get_settings()
    dataset = load_dataset(args.dataset)
    if args.questions:
        dataset = dataset[: args.questions]
    if not dataset:
        print(
            f"empty dataset at {args.dataset}. Build it first:\n"
            "  python eval/build_dataset.py --questions 200",
            file=sys.stderr,
        )
        return 1

    base = RetrievalConfig.from_settings(settings)
    exit_code = 0

    with session_scope() as session:
        if args.stage == "generation":
            greport = evaluate_generation(
                session, dataset, config=base, settings=settings, label=args.tag or "generation"
            )
            print_generation_report(greport)
            if not greport.gate_passed():
                exit_code = 1

        elif args.stage == "all":
            report = evaluate(
                session, dataset, config=base, settings=settings, label=args.tag or "full"
            )
            print_report(report)
            if not report.gate_passed():
                exit_code = 1
            greport = evaluate_generation(
                session, dataset, config=base, settings=settings, label="generation"
            )
            print_generation_report(greport)
            if not greport.gate_passed():
                exit_code = 1

        elif args.sweep:
            print("=== §10 check 4: threshold sweep ===")
            print(
                f"Looking for a point with recall@10 >= {RECALL_AT_10_TARGET:.2f} AND "
                f"refusal rate in {REFUSAL_BAND[0]:.0%}-{REFUSAL_BAND[1]:.0%}"
            )
            reports = sweep_threshold(session, dataset, base_config=base, settings=settings)
            print(
                f"{'threshold':>10} {'recall@10':>10} {'refusal':>9} "
                f"{'R@10 OK':>8} {'band OK':>8}"
            )
            joint = []
            for r in reports:
                threshold = r.label.split("=")[-1]
                recall_ok = r.recall_at_10 >= RECALL_AT_10_TARGET
                band_ok = REFUSAL_BAND[0] <= r.refusal_rate <= REFUSAL_BAND[1]
                print(
                    f"{threshold:>10} {r.recall_at_10:>10.3f} {r.refusal_rate:>8.1%} "
                    f"{'yes' if recall_ok else 'no':>8} {'yes' if band_ok else 'no':>8}"
                )
                if recall_ok and band_ok:
                    joint.append(r)
            if joint:
                best = max(joint, key=lambda r: r.recall_at_10)
                print(
                    f"\n{len(joint)} threshold(s) satisfy both targets. "
                    f"Best: {best.label} recall@10={best.recall_at_10:.3f} "
                    f"refusal={best.refusal_rate:.1%}"
                )
            else:
                print(
                    "\nNO threshold satisfies both targets simultaneously.\n"
                    "Per architecture.md 3.3 this is a PRD-level finding, not a tuning "
                    "task: the recall >= 0.85 and 10-30% refusal targets are "
                    "incompatible under the current corpus and configuration. Report "
                    "this rather than shipping the best single point."
                )
                exit_code = 1

        elif args.all_checks:
            full, vector_only = run_check_1(session, dataset, base, settings)
            print_report(full)
            print_report(vector_only)
            delta = full.recall_at_10 - vector_only.recall_at_10
            print(f"\n§10 check 1: hybrid vs vector-only delta = {delta:+.3f}")
            print(
                "Interpretation: a small positive delta means the keyword stage is "
                "not earning its cost; a large one is the evidence for FR-12."
            )

            full2, no_rerank = run_check_2(session, dataset, base, settings)
            print_report(full2)
            print_report(no_rerank)
            delta2 = full2.recall_at_10 - no_rerank.recall_at_10
            lat = full2.mean_latency_ms - no_rerank.mean_latency_ms
            print(f"\n§10 check 2: rerank delta = {delta2:+.3f} recall@10 for {lat:+.1f} ms")
            print(
                "Caveat: this environment's reranker is a lexical stand-in, not a "
                "cross-encoder. The measurement is real; what it establishes is the "
                "pipeline's behaviour, not that a real cross-encoder earns 300 ms. "
                "Re-run against a real provider before treating check 2 as settled."
            )

        else:
            config = base
            label = args.tag or "full"
            if args.ablate == "keyword":
                import dataclasses

                config = dataclasses.replace(
                    ranking_only(base), keyword_enabled=False
                )
                label = "ablation: vector only"
            elif args.ablate == "rerank":
                import dataclasses

                config = dataclasses.replace(
                    ranking_only(base), rerank_enabled=False
                )
                label = "ablation: rerank off"

            report = evaluate(session, dataset, config=config, settings=settings, label=label)
            print_report(report)
            print(f"\nfailure classification: {_classify_failure(report)}")
            if args.dump_scores:
                dump_scores(report, args.dump_scores)
                print(f"per-question scores written to {args.dump_scores}")
            if not report.gate_passed():
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())