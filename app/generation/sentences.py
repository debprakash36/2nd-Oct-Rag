"""Sentence splitting and a streaming sentence buffer (FR-18).

Citation validation is per sentence, not per token, because a citation marker
applies to a claim and a claim is a sentence (architecture.md 3.1). The stream
layer therefore has to be able to turn a token stream into a sentence stream
without waiting for the whole answer, which is what `SentenceBuffer` does.

The splitter is deliberately conservative about `"."`. Splitting a decimal
("14.5 days"), an abbreviation ("e.g."), or an initial ("J. Smith") would put a
citation boundary in the middle of a claim, and the validator would then judge
half a sentence. A missed split is harmless — it only means one more sentence is
held before flushing — so every ambiguous case favours *not* splitting.
"""

from __future__ import annotations

import re

#: Tokens whose trailing period is part of the token, not a sentence end.
_ABBREVIATIONS = frozenset(
    {
        "e.g", "i.e", "etc", "vs", "al", "cf", "approx", "no", "fig", "eq",
        "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "corp", "dept",
        "est", "min", "max", "vol", "ch", "sec", "art", "para", "pp",
    }
)

_TERMINATORS = ".!?"
#: A citation marker that trails a sentence terminator ("...30 days. [1]" or
#: "...30 days.[1]"). The marker grounds the sentence it follows, so it must be
#: kept with that sentence rather than becoming the next sentence's opening.
_TRAILING_MARKER = re.compile(r"\s*\[\d+\]")
#: Characters that may close a sentence after its terminator, e.g. `."` or `.)`.
_CLOSERS = "\"')]}\u201d\u2019"
#: A sentence may start after one of these. A lowercase start is treated as a
#: continuation, which avoids splitting "U.S. policy" or a mid-sentence "e.g.".
_UPPER_START = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"'(["


def _is_abbreviation(text: str, dot_index: int) -> bool:
    """Whether the `.` at `dot_index` belongs to an abbreviation or a number."""
    prev = text[dot_index - 1] if dot_index > 0 else ""
    nxt = text[dot_index + 1] if dot_index + 1 < len(text) else ""
    if prev.isdigit() and nxt.isdigit():
        return True  # decimal point
    start = dot_index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    token = text[start:dot_index].lower()
    if token in _ABBREVIATIONS:
        return True
    # A single letter is an initial ("J. Smith"), not a sentence end.
    return len(token) == 1 and token.isalpha()


def _split_complete(text: str) -> tuple[list[str], str]:
    """Split `text` into complete sentences plus an unfinished remainder."""
    sentences: list[str] = []
    start = 0
    i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char == "\n" and i + 1 < length and text[i + 1] == "\n":
            # A blank line ends a block even without a terminator (lists).
            candidate = text[start : i + 1]
            if candidate.strip():
                sentences.append(candidate.strip())
            start = i + 2
            i += 2
            continue
        if char in _TERMINATORS:
            if char == "." and _is_abbreviation(text, i):
                i += 1
                continue
            # Consume repeated terminators and closing punctuation.
            j = i + 1
            while j < length and (text[j] in _TERMINATORS or text[j] in _CLOSERS):
                j += 1
            # Keep any citation markers that trail the terminator with this
            # sentence: "[1]" after the full stop cites this claim, not the next.
            end = j
            while True:
                marker = _TRAILING_MARKER.match(text, end)
                if marker is None:
                    break
                end = marker.end()
            # The boundary is real only if what follows can start a sentence.
            k = end
            while k < length and text[k] in " \t":
                k += 1
            at_break = k >= length or text[k] == "\n" or text[k] in _UPPER_START
            if at_break:
                candidate = text[start:end]
                if candidate.strip():
                    sentences.append(candidate.strip())
                start = end
                i = end
                continue
        i += 1
    remainder = text[start:]
    return sentences, remainder


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences, keeping the trailing fragment if unfinished."""
    complete, remainder = _split_complete(text)
    if remainder.strip():
        complete.append(remainder.strip())
    return complete


class SentenceBuffer:
    """Accumulates streamed token deltas and yields completed sentences.

    A sentence is emitted as soon as its terminator arrives, so buffering is
    bounded by one sentence rather than by the whole answer. That is the property
    that keeps TTFT low while still validating per sentence
    (architecture.md 3.1, implementation.md 5.3).
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        """Add a token delta; return any sentences that are now complete."""
        self._buffer += delta
        complete, self._buffer = _split_complete(self._buffer)
        return complete

    def flush(self) -> list[str]:
        """Return the unfinished remainder, if any, and reset."""
        remainder = self._buffer
        self._buffer = ""
        return [remainder.strip()] if remainder.strip() else []
