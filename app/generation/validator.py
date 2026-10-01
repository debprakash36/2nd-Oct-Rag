"""Citation validation: closed ID set, strip invalid, zero-valid becomes refusal.

This is the mechanism that makes FR-18 and G6 true rather than aspirational
(architecture.md 3.5). Two rules make it hold:

1. **The retrieved set is closed.** Markers are integers 1..K indexing the
   retrieved passages. A marker outside that range is fabricated by definition,
   so it is removed rather than displayed. Internal `chunk_id` values never reach
   the model or the answer text (implementation.md 5.2).

2. **An answer with zero valid citations becomes a refusal.** This is a
   deliberate deviation from the PRD's literal "strip" wording, documented in
   architecture.md 3.5 and required by the PRD's own NFR-3: an answer with no
   resolvable citation is a hallucination wearing a citation costume, and
   displaying it fails groundedness. The deviation is implemented here and
   commented at the code site so it is not "simplified" away.

Stripped markers are *logged*, never silently dropped: a rising strip rate is the
earliest signal of model or prompt drift (architecture.md 7.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Inline citation marker: `[3]`. Only non-negative integers are markers.
MARKER_RE = re.compile(r"\[(\d+)\]")


def extract_markers(text: str) -> list[int]:
    """All marker integers in `text`, in order of appearance."""
    return [int(m) for m in MARKER_RE.findall(text)]


@dataclass(frozen=True)
class ValidatedSentence:
    """One sentence after citation validation."""

    #: Sentence text with fabricated markers removed.
    text: str
    #: Markers that resolved to a retrieved passage.
    markers: tuple[int, ...]
    #: Markers that did not resolve and were stripped.
    stripped: tuple[int, ...]

    @property
    def is_grounded(self) -> bool:
        return bool(self.markers)


def _tidy(text: str) -> str:
    """Remove the whitespace left behind when a marker is deleted."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def validate_sentence(text: str, passage_count: int) -> ValidatedSentence:
    """Validate one sentence against a retrieved set of `passage_count` passages."""
    markers = extract_markers(text)
    valid = tuple(n for n in markers if 1 <= n <= passage_count)
    stripped = tuple(n for n in markers if not 1 <= n <= passage_count)

    cleaned = MARKER_RE.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= passage_count else "",
        text,
    )
    return ValidatedSentence(text=_tidy(cleaned), markers=valid, stripped=stripped)


class CitationValidator:
    """Accumulates validation state across a whole answer.

    The end-of-answer decision — grounded vs. ungrounded — cannot be made from a
    single sentence: a fabricated marker in sentence two does not invalidate a
    real citation in sentence one, but *no* real citation anywhere does mean the
    answer never grounded. So the validator keeps the running state and answers
    `is_grounded` only once the stream is complete.
    """

    def __init__(self, passage_count: int) -> None:
        self.passage_count = passage_count
        self._grounded = False
        self.emitted_markers = 0
        self.stripped_markers: list[int] = []

    def accept(self, sentence: str) -> ValidatedSentence:
        """Validate `sentence`, fold its result into the running state, return it."""
        result = validate_sentence(sentence, self.passage_count)
        if result.markers:
            self._grounded = True
            self.emitted_markers += len(result.markers)
        self.stripped_markers.extend(result.stripped)
        return result

    @property
    def is_grounded(self) -> bool:
        """Whether at least one valid citation has been seen."""
        return self._grounded

    @property
    def stripped_count(self) -> int:
        return len(self.stripped_markers)
