"""Shared types for the retrieval pipeline.

These are the vocabulary the store implementations, fusion, rerank, and the eval
harness all speak. They exist as one module so a change to a candidate's shape
breaks every consumer at import time rather than silently at the first
production query.

The per-stage score fields on `RetrievalCandidate` are not decoration. FR-28 and
the PRD's improvement loop require distinguishing a **content gap** (the right
document is not in the corpus) from a **retrieval gap** (the document is there
but was not retrieved). That classification is only possible if the score the
document earned at each stage is retained, so these fields are part of the
contract rather than debug output.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Stage(enum.StrEnum):
    """A retrieval stage, for per-stage score attribution.

    Recorded per candidate so a failure can be attributed to a stage rather than
    to "retrieval". A chunk that no keyword search returned but a vector search
    did is a different bug from one neither returned.
    """

    VECTOR = "vector"
    KEYWORD = "keyword"
    FUSED = "fused"
    RERANKED = "reranked"


@dataclass(frozen=True)
class AccessFilter:
    """Filter pushed *into* both stores (FR-16).

    Pushing the filter down rather than post-filtering is a correctness
    requirement, not an optimisation: architecture.md §3.4 states that text read
    into the model's context can leak by paraphrase, so a document the caller may
    not see must never be returned by the store in the first place.

    `allowed_tags` is the caller's grants. A document is visible when it carries
    at least one tag in the intersection with `document_acl_tags`, or when the
    caller's tag set is empty (unrestricted access). Empty-means-unrestricted is
    deliberate: a filter that denied everything by default would fail closed on a
    misconfigured caller and look like a corpus outage.
    """

    allowed_tags: frozenset[str] = frozenset()

    def permits(self, document_acl_tags: list[str] | None) -> bool:
        """Whether a document with these tags may be returned."""
        if not self.allowed_tags:
            return True
        return bool(self.allowed_tags.intersection(document_acl_tags or []))

    def describe(self) -> str:
        """Short description for logs and eval output. Never log the tag values."""
        if not self.allowed_tags:
            return "unrestricted"
        return f"tags={len(self.allowed_tags)}"


UNRESTRICTED = AccessFilter()


@dataclass
class RetrievalCandidate:
    """One chunk as it moves through the pipeline.

    Mutated rather than replaced at each stage so per-stage scores accumulate on a
    single object. Replacing would mean copying forward the fields each stage
    produced, which is where scores get silently dropped.
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    breadcrumb: str = ""
    page: int | None = None
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    acl_tags: list[str] = field(default_factory=list)

    #: Populated as the candidate advances. Absent means the stage did not
    #: return this candidate — a meaningful distinction from a score of zero,
    #: which means the stage returned it and judged it worthless.
    scores: dict[Stage, float] = field(default_factory=dict)
    #: Rank (1-based) within each stage that returned this candidate. RRF
    #: consumes ranks, never raw scores.
    ranks: dict[Stage, int] = field(default_factory=dict)

    def best_score(self) -> float:
        """Highest score across stages.

        Reranked when present (it is the last and most specific judgement), then
        fused, then whatever single stage returned it. Used for thresholding and
        for eviction ordering.
        """
        for stage in (Stage.RERANKED, Stage.FUSED, Stage.VECTOR, Stage.KEYWORD):
            if stage in self.scores:
                return self.scores[stage]
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form for eval score dumps (FR-28)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "breadcrumb": self.breadcrumb,
            "scores": {stage.value: value for stage, value in self.scores.items()},
            "ranks": {stage.value: value for stage, value in self.ranks.items()},
        }


@dataclass
class StageTrace:
    """What each stage contributed, for per-stage eval output.

    §4.1 task 2.9 asks for a per-stage score dump. Keeping the trace on the
    result means the harness does not have to instrument each stage separately,
    and an ablation can report what was removed rather than only the delta.
    """

    vector_returned: int = 0
    keyword_returned: int = 0
    fused: int = 0
    reranked: int = 0
    after_dedupe: int = 0
    after_threshold: int = 0
    after_budget: int = 0
    stages_run: list[Stage] = field(default_factory=list)

    def record(self, stage: Stage, count: int) -> None:
        self.stages_run.append(stage)
        match stage:
            case Stage.VECTOR:
                self.vector_returned = count
            case Stage.KEYWORD:
                self.keyword_returned = count
            case Stage.FUSED:
                self.fused = count
            case Stage.RERANKED:
                self.reranked = count


@dataclass
class RetrievalResult:
    """Everything the orchestrator and the eval harness need from one query."""

    candidates: list[RetrievalCandidate] = field(default_factory=list)
    #: The query as the user asked it (FR-28).
    original_query: str = ""
    #: The query after anaphora resolution. Empty when no rewrite applied.
    #: Logged separately from the original: a bad rewrite is otherwise
    #: indistinguishable from a retrieval failure in production.
    rewritten_query: str = ""
    #: Set when rewrite ran but could not produce a standalone query.
    rewrite_applied: bool = False
    #: True when the top score fell below the threshold, or the fetch was empty.
    abstained: bool = False
    #: Candidates that were ranked but did not clear the threshold. Kept separate
    #: from `candidates` so the eval harness's recall numbers are unaffected, but
    #: available so a refusal can still render the sources panel (FR-20): the user
    #: needs to see what the system looked at even when it declined to answer.
    #: These are never fed to the model.
    abstain_candidates: list[RetrievalCandidate] = field(default_factory=list)
    #: The threshold actually applied, for the eval record.
    threshold_applied: float | None = None
    #: Why the pipeline abstained. Empty string when it did not.
    abstain_reason: str = ""
    trace: StageTrace = field(default_factory=StageTrace)
    #: Per-stage timing in milliseconds. §7.4 budgets these; Phase 2 measures
    #: them so Phase 5 optimises against reality.
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        """The query actually searched with."""
        return self.rewritten_query or self.original_query

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.candidates]