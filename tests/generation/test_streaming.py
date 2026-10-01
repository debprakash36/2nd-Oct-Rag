"""Sentence-buffered streaming (FR-19, NFR-1).

Contains the named test `test_first_token_before_completion`, which asserts the
property the design exists for: the first sentence flushes while the generator is
still producing, so sentence-level validation did not regress into answer-level
buffering (implementation.md 5.3).
"""

from __future__ import annotations

from collections.abc import Iterator

from app.generation.assembly import AnswerAssembler, ChunkKind
from app.generation.sentences import SentenceBuffer, split_sentences


def test_split_respects_decimals_and_abbreviations():
    assert split_sentences("Refunds take 14.5 days. Shipping takes 5 days.") == [
        "Refunds take 14.5 days.",
        "Shipping takes 5 days.",
    ]
    assert split_sentences("See e.g. the policy. It applies.") == [
        "See e.g. the policy.",
        "It applies.",
    ]


def test_split_keeps_unfinished_remainder():
    assert split_sentences("A finished one. An unfinished") == [
        "A finished one.",
        "An unfinished",
    ]


def test_buffer_emits_only_complete_sentences():
    buffer = SentenceBuffer()
    assert buffer.feed("Refunds take 14 da") == []
    assert buffer.feed("ys. Shipping is") == ["Refunds take 14 days."]
    assert buffer.flush() == ["Shipping is"]


def test_first_token_before_completion():
    """First flush arrives while the generator is still producing."""
    finished = {"done": False}

    def tokens() -> Iterator[str]:
        yield "The refund window is 14 days [1]."
        yield " Shipping takes 5 days [1]."
        finished["done"] = True

    assembler = AnswerAssembler(passage_count=1)
    stream = assembler.run(tokens())

    first = next(stream)
    assert first.kind is ChunkKind.TOKEN
    assert "[1]" in first.text
    # The second delta has not been requested yet, so the provider is still
    # producing: this is the TTFT property, not merely "it works eventually".
    assert finished["done"] is False


def test_sentences_are_separated_when_reassembled():
    """Sentence-wise flushing must not jam sentences together.

    The splitter and validator strip boundary whitespace, so without a
    re-inserted separator the streamed answer would read "...[1]This document".
    """
    assembler = AnswerAssembler(passage_count=1)

    def tokens() -> Iterator[str]:
        yield "First claim [1]."
        yield " Second claim [1]."

    answer = "".join(
        c.text for c in assembler.run(tokens()) if c.kind is ChunkKind.TOKEN
    )
    assert answer == "First claim [1]. Second claim [1]."


def test_citation_warning_emitted_when_markers_stripped():
    assembler = AnswerAssembler(passage_count=1)

    def tokens() -> Iterator[str]:
        yield "Real [1]. Fabricated [5]."

    chunks = list(assembler.run(tokens()))
    warning = [c for c in chunks if c.kind is ChunkKind.CITATION_WARNING]
    assert len(warning) == 1
    assert warning[0].stripped == 1
