"""Streaming assembly: token stream in, validated events out (FR-18, FR-19).

This is the concrete resolution of the streaming tension in architecture.md 3.1:
FR-19 wants tokens out fast, FR-18 wants citations validated before the user sees
them, and NFR-1 wants a low TTFT. The design buffers **sentence-wise**, not
answer-wise, so the first clean sentence flushes immediately.

One refinement over the literal pseudocode in implementation.md 5.2, and the
reason for it: that pseudocode flushes a sentence with *no* markers immediately
as "transitional text", but also requires that an answer with zero valid
citations become a refusal. Those two cannot both hold — once transitional text
has been streamed it cannot be retracted. This implementation therefore holds
leading ungrounded sentences until the first valid citation arrives; at that
point it releases them and streams freely. If no valid citation ever arrives, the
held text is discarded and a refusal is emitted instead. The common case (the
first sentence carries a citation) is unaffected: the hold lasts a fraction of a
sentence and TTFT is preserved. The hold is bounded by `max_held_chars` so a
runaway ungrounded answer cannot buffer without limit.

**Escalation condition** (implementation.md 5.3 requires this be stated, not
decided by accident): if the eval set measures a non-trivial rate of answers
where an *early* claim is invalidated by a *later* sentence — which sentence-level
validation cannot catch — escalate to whole-answer buffering and accept the TTFT
cost. Until that rate is measured, whole-answer buffering is not justified.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from app.generation.refusal import RefusalReason, refusal_text
from app.generation.sentences import SentenceBuffer
from app.generation.validator import CitationValidator

#: Cap on ungrounded leading text held before the first citation. Chosen well
#: above a normal introduction so the hold is invisible in practice, and finite
#: so an ungrounded answer cannot consume unbounded memory.
DEFAULT_MAX_HELD_CHARS = 4000


class ChunkKind(enum.StrEnum):
    """What a validated stream chunk represents."""

    TOKEN = "token"
    CITATION_WARNING = "citation_warning"
    REFUSAL = "refusal"


@dataclass(frozen=True)
class StreamChunk:
    """One unit the SSE layer turns into an event."""

    kind: ChunkKind
    text: str = ""
    stripped: int = 0


class AnswerAssembler:
    """Validates a token stream sentence by sentence and yields safe chunks.

    State is kept on the instance so the caller can read `abstained` and
    `stripped_count` once iteration finishes, and write them to the QueryLog.
    """

    def __init__(
        self, passage_count: int, *, max_held_chars: int = DEFAULT_MAX_HELD_CHARS
    ) -> None:
        self._validator = CitationValidator(passage_count)
        self._max_held = max_held_chars
        self.abstained = False

    @property
    def grounded(self) -> bool:
        return self._validator.is_grounded

    @property
    def stripped_count(self) -> int:
        return self._validator.stripped_count

    @property
    def emitted_markers(self) -> int:
        return self._validator.emitted_markers

    def run(self, chunks: Iterable[str]) -> Iterator[StreamChunk]:
        """Consume token deltas and yield validated chunks."""
        buffer = SentenceBuffer()
        held: list[str] = []
        held_chars = 0
        # Sentence splitting and `_tidy` strip boundary whitespace from every
        # sentence, so concatenating the emitted sentences would run them
        # together ("...[1]This document..."). Re-insert a single separating
        # space at each boundary. The flag tracks whether a token has already
        # been emitted anywhere in this answer.
        emitted: list[bool] = [False]

        def token(text: str) -> Iterator[StreamChunk]:
            if not text:
                return
            if emitted[0]:
                text = " " + text
            emitted[0] = True
            yield StreamChunk(ChunkKind.TOKEN, text=text)

        def drain(sentences: Iterable[str]) -> Iterator[StreamChunk]:
            nonlocal held, held_chars
            for sentence in sentences:
                was_grounded = self._validator.is_grounded
                result = self._validator.accept(sentence)
                if result.markers and not was_grounded:
                    # First real citation: the answer is grounded, so the held
                    # introduction is safe to release.
                    for text in held:
                        yield from token(text)
                    held, held_chars = [], 0
                if self._validator.is_grounded:
                    yield from token(result.text)
                elif result.text:
                    held.append(result.text)
                    held_chars += len(result.text)
                    if held_chars > self._max_held:
                        # Still ungrounded after a large amount of text. Stop
                        # holding to bound memory; if no citation ever arrives
                        # this becomes a refusal and the text is discarded.
                        held, held_chars = [], 0

        def sentences() -> Iterator[str]:
            for delta in chunks:
                yield from buffer.feed(delta)
            yield from buffer.flush()

        yield from drain(sentences())

        if self._validator.is_grounded:
            # Normally empty: grounded text streams immediately. Retained so a
            # future change to the release rule cannot silently drop content.
            for text in held:
                yield from token(text)
            if self._validator.stripped_count:
                yield StreamChunk(
                    ChunkKind.CITATION_WARNING,
                    stripped=self._validator.stripped_count,
                )
            return

        # Zero valid citations anywhere: ungrounded. Discard everything held and
        # refuse. See the module docstring for why this is the deliberate
        # deviation from the PRD's literal "strip" wording.
        self.abstained = True
        yield StreamChunk(
            ChunkKind.REFUSAL, text=refusal_text(RefusalReason.UNGROUNDED_ANSWER)
        )
