"""The refusal path and the sources payload (G2, US-2, FR-20).

A refusal is a first-class outcome, not an error: the PRD's whole premise is that
"no answer" is better than a confident wrong one. Three things are refused, and
they are different failures, so they are distinguished in the log rather than
collapsed into one message:

* `NO_EVIDENCE` — retrieval returned nothing above the threshold, or returned an
  empty set. Per architecture.md 3.3 the pipeline short-circuits here with **no
  LLM call**: there is nothing to ground an answer in, so spending tokens to say
  so would be waste (and an invitation to invent).
* `UNGROUNDED_ANSWER` — the model produced text but no sentence carried a valid
  citation. The validator converts this to a refusal (see `validator.py`).
* `STORE_UNAVAILABLE` — a retrieval store is down. Phase 5 owns the health path;
  the reason exists here so the refusal vocabulary is complete.

`build_refusal_messages` provides the refusal's *own* prompt (implementation.md
5.1 task 3.6) for deployments that want the model to phrase the refusal. The
default path uses the fixed text: the threshold refusal must not call the model
at all, and the ungrounded refusal has already had its model call. The function
is used by the refusal tests and is the seam if that default changes.

The sources payload intentionally includes `chunk_id`. Architecture.md 6 lists
`chunk_id[]` in the `sources` event because the UI needs it to open the exact
passage (FR-18, NFR-9). The "never reaches the response" rule in
implementation.md 5.2 applies to the *model prompt and the generated text*; the
server-side marker-to-chunk mapping is the one place the real id is exposed, and
it is what makes a citation clickable rather than decorative.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.generation.prompt import AnswerStyle


class RefusalReason(enum.StrEnum):
    """Why an answer was withheld."""

    NO_EVIDENCE = "no_evidence"
    UNGROUNDED_ANSWER = "ungrounded_answer"
    STORE_UNAVAILABLE = "store_unavailable"


_REFUSAL_TEXT: dict[RefusalReason, str] = {
    RefusalReason.NO_EVIDENCE: (
        "I couldn't find anything in the documents I have access to that answers "
        "that. Try rephrasing, or check the sources below."
    ),
    RefusalReason.UNGROUNDED_ANSWER: (
        "I found documents that look related, but I couldn't ground an answer in "
        "them, so I'm not going to guess. The closest sources are below."
    ),
    RefusalReason.STORE_UNAVAILABLE: (
        "Search is temporarily unavailable, so I can't answer right now. Please "
        "try again shortly."
    ),
}

_REFUSAL_SYSTEM_PROMPT = """\
You are a document question-answering assistant for a private document corpus.
The retrieved documents do not contain a trustworthy answer to the user's
question. Reply with a single short sentence saying you could not find the answer
in the available documents, and do not speculate. Do not invent citations.
"""


def refusal_text(reason: RefusalReason) -> str:
    """The fixed refusal message for `reason`."""
    return _REFUSAL_TEXT[reason]


def build_refusal_messages(
    question: str, reason: RefusalReason, style: AnswerStyle = AnswerStyle.CONCISE
) -> list[dict[str, str]]:
    """Messages for a model-phrased refusal (implementation.md 5.1 task 3.6).

    The answer-prompt version is not attached to this prompt: a refusal carries no
    factual claim, so attributing it to the answer prompt version would corrupt
    the signal `QueryLog.prompt_version` exists to provide.
    """
    # `style` is accepted for call-site symmetry; a refusal is one sentence
    # regardless of the answer length preset.
    _ = style
    return [
        {"role": "system", "content": _REFUSAL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nReason: {reason.value}"},
    ]


@dataclass(frozen=True)
class Source:
    """One entry in the `sources` SSE event (architecture.md 6)."""

    index: int
    chunk_id: str
    breadcrumb: str
    page: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "breadcrumb": self.breadcrumb,
            "page": self.page,
        }


class SourceCandidate(Protocol):
    """The shape `sources_payload` needs: `RetrievalCandidate` structurally.

    A protocol rather than an import keeps this module free of a dependency on
    the retrieval package, which matters because the retrieval package imports
    back into generation-adjacent types and a cycle would be easy to introduce.
    """

    chunk_id: str
    breadcrumb: str
    page: int | None


def sources_payload(candidates: Sequence[SourceCandidate]) -> list[dict[str, object]]:
    """Build the sources event from retrieval candidates (FR-20)."""
    payload: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        payload.append(
            Source(
                index=index,
                chunk_id=str(candidate.chunk_id),
                breadcrumb=str(candidate.breadcrumb or ""),
                page=candidate.page,
            ).to_dict()
        )
    return payload


__all__ = [
    "RefusalReason",
    "Source",
    "build_refusal_messages",
    "refusal_text",
    "sources_payload",
]
