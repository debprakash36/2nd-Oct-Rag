"""Config-driven chunking (FR-9, FR-10).

All sizing is configuration, not code, because Phase 2 has to compare strategies
against the eval set without a code change per experiment (architecture.md 4.2).
`ChunkConfig` is a Pydantic model so an invalid value fails at load rather than
producing pathological chunks.

The offset invariant
--------------------
Every chunk carries `char_start`/`char_end` into the document's canonical
cleaned text, and the invariant is:

    cleaned_text[chunk.char_start:chunk.char_end] == chunk.text

This is what makes a citation resolve to an exact passage instead of a whole
file (FR-10). It is asserted directly in tests/ingest/test_offsets.py.

The two ways to break it are (a) computing offsets against raw text and then
cleaning the chunk, and (b) normalizing or searching for the text after slicing.
Both are structurally prevented here: offsets are tracked as positions as the
document is walked, and `ChunkDraft.text` is produced by exactly one expression
— the canonical slice. Text is never re-derived from a search, because the same
sentence can appear twice in a policy document and a search would silently point
the citation at the wrong copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.core.errors import ChunkingError
from app.core.logging import get_logger
from app.ingest.breadcrumb import Heading, build_breadcrumb, detect_headings, section_path_at

log = get_logger("app.ingest.chunk")

Strategy = Literal["heading", "paragraph", "fixed"]

_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_WHITESPACE = re.compile(r"\s+")
#: Characters per token for a stable English average. Only used to convert the
#: token target into a character window in the `fixed` baseline strategy.
CHARS_PER_TOKEN = 4


class ChunkConfig(BaseModel):
    """Chunk sizing. Every field is overridable per corpus (FR-9)."""

    target_tokens: int = Field(default=1000, gt=0)
    overlap_tokens: int = Field(default=200, ge=0)
    strategy: Strategy = "heading"
    min_chunk_tokens: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkConfig:
        """Reject an overlap that would prevent the window from advancing.

        Overlap >= target means each window already contains the next one, so the
        chunker cannot make progress. Failing at construction is clearer than
        hanging during ingest.
        """
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be less than "
                f"target_tokens ({self.target_tokens})"
            )
        return self


def count_tokens(text: str) -> int:
    """Approximate token count.

    A word heuristic, not a model-specific tokenizer. Exactness matters for
    provider billing and hard context limits; for deciding whether a window is
    full, a consistent approximation is sufficient and keeps a tokenizer
    dependency out of the ingest hot path. Phase 3 swaps in the real tokenizer
    for the request-time context budget, where being wrong has a cost.
    """
    return len(_WHITESPACE.findall(text))


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before it is assigned a `chunk_id` and written to the store."""

    index: int
    text: str
    char_start: int
    char_end: int
    breadcrumb: str
    section_path: list[str]
    token_count: int
    page: int | None = None


@dataclass(frozen=True)
class _Unit:
    """A structural unit of the document: a heading section or a paragraph.

    Offsets are the unit's extent in the canonical text, including the newline
    run that separates it from the next unit. Contiguity of the emitted windows
    depends on those boundaries lining up exactly.
    """

    start: int
    end: int
    heading: Heading | None


def chunk_document(
    cleaned_text: str,
    doc_title: str,
    *,
    headings: list[Heading] | None = None,
    config: ChunkConfig | None = None,
) -> list[ChunkDraft]:
    """Split canonical cleaned text into chunks with exact offsets.

    `headings` may be supplied by the caller; when omitted they are detected from
    `cleaned_text` itself. Detection is the default specifically so an outline
    measured against a different rendering of the text cannot be paired with this
    one — that mismatch is the offset bug in its most likely form.
    """
    cfg = config or ChunkConfig()
    if not cleaned_text.strip():
        raise ChunkingError("cannot chunk empty text")

    outline = detect_headings(cleaned_text) if headings is None else headings
    units = _segment(cleaned_text, outline, cfg)
    drafts = _assemble(cleaned_text, units, cfg)
    return _attach_metadata(drafts, outline, doc_title, cfg)


def _segment(text: str, headings: list[Heading], cfg: ChunkConfig) -> list[_Unit]:
    """Break the document into units, recording each unit's exact offsets."""
    if cfg.strategy == "fixed":
        return _segment_fixed(text, cfg)
    if cfg.strategy == "paragraph":
        return _segment_paragraph(text)
    return _segment_by_heading(text, headings)


def _segment_by_heading(text: str, headings: list[Heading]) -> list[_Unit]:
    """One unit per heading section: the heading line plus the body under it.

    A document with no detectable headings falls back to paragraph
    segmentation, so a heading-free document still chunks sensibly rather than
    collapsing to a single oversized chunk.
    """
    if not headings:
        return _segment_paragraph(text)

    starts = sorted({0} | {h.char_start for h in headings})
    units: list[_Unit] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        if end > start and text[start:end].strip():
            heading = next((h for h in headings if h.char_start == start), None)
            units.append(_Unit(start=start, end=end, heading=heading))
    return units


def _segment_paragraph(text: str) -> list[_Unit]:
    """One unit per blank-line-delimited paragraph.

    Units span the *paragraphs*, not the separators between them. The separators
    are located in order and each unit runs from the end of the previous
    separator to the start of the next, which keeps `text[start:end]` equal to
    the paragraph itself and so preserves the offset invariant. Boundaries are
    trimmed to the paragraph's own text so the offsets point at content rather
    than at leading blank lines.
    """
    units: list[_Unit] = []
    cursor = 0
    for match in _PARAGRAPH.finditer(text):
        _append_unit(units, text, cursor, match.start())
        cursor = match.end()
    _append_unit(units, text, cursor, len(text))

    if not units and text.strip():
        units.append(_Unit(start=0, end=len(text), heading=None))
    return units


def _append_unit(units: list[_Unit], text: str, start: int, end: int) -> None:
    """Append the span `[start, end)` trimmed to its own non-whitespace text."""
    lead = len(text[start:end]) - len(text[start:end].lstrip())
    trimmed_start = start + lead
    trimmed_end = end - (len(text[start:end]) - len(text[start:end].rstrip()))
    if trimmed_end > trimmed_start:
        units.append(_Unit(start=trimmed_start, end=trimmed_end, heading=None))


def _segment_fixed(text: str, cfg: ChunkConfig) -> list[_Unit]:
    """Baseline strategy: fixed character windows, ignoring structure.

    Exists so Phase 2 can measure what the semantic strategies actually buy
    (architecture.md 10, check 3). It keeps the same offset discipline as the
    others — only the boundaries differ, which is what makes the comparison a
    fair one.
    """
    window = cfg.target_tokens * CHARS_PER_TOKEN
    step = max(1, window - cfg.overlap_tokens * CHARS_PER_TOKEN)
    return [
        _Unit(start=s, end=min(s + window, len(text)), heading=None)
        for s in range(0, len(text), step)
        if text[s : s + window].strip()
    ]


def _assemble(text: str, units: list[_Unit], cfg: ChunkConfig) -> list[ChunkDraft]:
    """Group units into windows, applying overlap and the min-size filter.

    A unit that is itself larger than the target is emitted on its own rather
    than split, so a sentence is never cut in half (FR-9). That can produce a
    chunk above `target_tokens`; this is the documented trade for never breaking
    a sentence across chunks.
    """
    windows: list[tuple[int, int]] = []
    window_start: int | None = None
    window_end = 0
    window_tokens = 0

    def close_window() -> None:
        nonlocal window_start, window_end, window_tokens
        if window_start is not None and window_end > window_start:
            windows.append((window_start, window_end))
        window_start, window_end, window_tokens = None, 0, 0

    for unit in units:
        unit_text = text[unit.start : unit.end]
        unit_tokens = count_tokens(unit_text)

        if window_start is None:
            window_start, window_end, window_tokens = unit.start, unit.end, unit_tokens
            continue

        if unit_tokens > cfg.target_tokens:
            # Oversized unit: flush what we have, then emit it whole.
            close_window()
            windows.append((unit.start, unit.end))
            continue

        if window_tokens + unit_tokens > cfg.target_tokens:
            closed_start, closed_end = window_start, window_end
            close_window()
            # Overlap: rewind into the tail of the window just closed so a fact
            # spanning a unit boundary is retrievable from at least one whole
            # chunk. The rewind is clamped to the *closed window's* start — not to
            # the incoming unit's start, which is always past the closed window
            # and would clamp the overlap to zero, silently disabling it.
            rewind = _overlap_chars(max(closed_end - closed_start, 0), cfg)
            new_start = max(closed_start, closed_end - rewind)
            window_start = new_start
            window_end = unit.end
            # Rewound text counts toward the window, or the window would
            # immediately exceed the target again on the next unit.
            window_tokens = count_tokens(text[new_start:closed_end]) + unit_tokens
            continue

        window_end, window_tokens = unit.end, window_tokens + unit_tokens

    close_window()
    return _finalize(text, windows, cfg)


def _overlap_chars(window_chars: int, cfg: ChunkConfig) -> int:
    """Character length of the overlap region, derived from the token target."""
    if cfg.overlap_tokens == 0 or window_chars <= 0:
        return 0
    return int(window_chars * (cfg.overlap_tokens / cfg.target_tokens))


def _finalize(
    text: str, windows: list[tuple[int, int]], cfg: ChunkConfig
) -> list[ChunkDraft]:
    """Slice each window, drop undersized ones, and renumber.

    `cleaned_text[char_start:char_end]` is the one and only place chunk text is
    produced. Everything downstream reads the stored `text`; nothing re-derives
    it, which is what keeps offsets and text in agreement.
    """
    kept: list[tuple[int, int]] = []
    for start, end in windows:
        if count_tokens(text[start:end]) >= cfg.min_chunk_tokens:
            kept.append((start, end))
        else:
            log.debug("dropped undersized window", extra={"start": start, "end": end})

    dropped = len(windows) - len(kept)
    if dropped:
        # A large drop here usually means min_chunk_tokens is too high for the
        # corpus rather than that the document is mostly noise
        # (implementation.md 3.5). Surfaced so it is noticed.
        log.info("dropped undersized chunks", extra={"dropped": dropped, "kept": len(kept)})

    return [
        ChunkDraft(
            index=i,
            text=text[start:end],
            char_start=start,
            char_end=end,
            breadcrumb="",
            section_path=[],
            token_count=count_tokens(text[start:end]),
        )
        for i, (start, end) in enumerate(kept)
    ]


def _attach_metadata(
    drafts: list[ChunkDraft], headings: list[Heading], doc_title: str, cfg: ChunkConfig
) -> list[ChunkDraft]:
    """Attach section path and breadcrumb to each chunk."""
    enriched = [
        ChunkDraft(
            index=draft.index,
            text=draft.text,
            char_start=draft.char_start,
            char_end=draft.char_end,
            breadcrumb=build_breadcrumb(
                doc_title, section_path_at(headings, draft.char_start)
            ),
            section_path=section_path_at(headings, draft.char_start),
            token_count=draft.token_count,
            page=draft.page,
        )
        for draft in drafts
    ]
    if not enriched:
        raise ChunkingError(
            f"chunking produced no chunks; min_chunk_tokens={cfg.min_chunk_tokens} "
            "is likely too high for this document"
        )
    log.info(
        "chunked document",
        extra={
            "chunks": len(enriched),
            "strategy": cfg.strategy,
            "target_tokens": cfg.target_tokens,
            "overlap_tokens": cfg.overlap_tokens,
            "min_chunk_tokens": cfg.min_chunk_tokens,
            "headings": len(headings),
        },
    )
    return enriched
