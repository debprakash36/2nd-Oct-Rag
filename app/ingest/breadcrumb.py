"""Heading detection and breadcrumb construction (FR-11).

Headings are detected from the canonical cleaned text using the markdown
convention rather than from each parser's own document outline. Parser-specific
outlines would mean per-format heading quality, and the conventions below cover
the v1 formats well enough that the consistency is worth more.

The breadcrumb restores a chunk's subject at embedding time. A chunk reading
"must be filed within 30 days" is near-useless in isolation; prefixed with
"Refund Policy > Digital Products", it is retrievable on its own. That is the
whole point of FR-11, and it is why the breadcrumb goes on the *embedding input*
only — storing it in `Chunk.text` would push document titles into search results
and into every citation the user sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logging import get_logger

log = get_logger("app.ingest.breadcrumb")

_MD_ATX = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*#*[ \t]*$")
_SETEXT_H1 = re.compile(r"^=+[ \t]*$")
_SETEXT_H2 = re.compile(r"^-{2,}[ \t]*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?[ \t]+(\S.*)$")
_ALL_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9 ,'&/()\-]{2,}$")

#: Beyond this a "heading" is a sentence, and treating it as a heading produces a
#: breadcrumb that is worse than no breadcrumb.
MAX_HEADING_CHARS = 120
#: Headings must be reasonably short to be one. Short lines that are lowercase
#: and unpunctuated are usually list items or table rows, not headings.
MIN_HEADING_CHARS = 2


@dataclass(frozen=True)
class Heading:
    """A detected heading and its offset in the cleaned text."""

    level: int
    text: str
    char_start: int


def looks_like_heading(line: str) -> tuple[int, str] | None:
    """Classify one line. Returns (level, text) or None.

    Heuristics, in confidence order:
      1. Markdown ATX — the strongest signal, so it wins outright.
      2. Setext underlines — level comes from the underline character.
      3. Numbered section — `2.1 Scope` is a heading far more often than prose.
      4. Short all-caps line — a common title and slide style.

    Deliberately conservative: a false positive adds a spurious breadcrumb
    segment, which is a small cost, while a false negative loses the section
    context entirely, which hurts retrieval.
    """
    stripped = line.strip()
    if not (MIN_HEADING_CHARS <= len(stripped) <= MAX_HEADING_CHARS):
        return None

    if match := _MD_ATX.match(stripped):
        return len(match.group(1)), match.group(2).strip()

    numbered = _NUMBERED.match(stripped)
    if numbered and not stripped.endswith((".", "!", "?")):
        return numbered.group(1).count(".") + 1, numbered.group(2).strip()

    if _ALL_CAPS.match(stripped) and len(stripped.split()) <= 10:
        return 1, stripped

    return None


def detect_headings(text: str) -> list[Heading]:
    """Detect headings in canonical cleaned text, with offsets.

    Setext underlines are resolved here rather than in `looks_like_heading`
    because they require the following line, which a line-level classifier
    cannot see.
    """
    lines = text.split("\n")
    # Offset of the start of each line, so a heading's char_start is exact.
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1  # +1 for the '\n' join

    headings: list[Heading] = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Setext: the underline determines the level, the line above is the text.
        if stripped and i > 0:
            if _SETEXT_H1.match(stripped):
                prev = lines[i - 1].strip()
                if MIN_HEADING_CHARS <= len(prev) <= MAX_HEADING_CHARS:
                    headings.append(Heading(1, prev, starts[i - 1]))
                    continue
            elif _SETEXT_H2.match(stripped):
                prev = lines[i - 1].strip()
                if MIN_HEADING_CHARS <= len(prev) <= MAX_HEADING_CHARS:
                    headings.append(Heading(2, prev, starts[i - 1]))
                    continue

        if not stripped or _SETEXT_H1.match(stripped) or _SETSET_H2_GUARD(stripped):
            continue

        if detected := looks_like_heading(line):
            level, heading_text = detected
            headings.append(Heading(level, heading_text, starts[i]))

    # A heading must not also be a chunk's entire content, and duplicates at the
    # same offset (ATX plus setext on adjacent lines) add nothing to a breadcrumb.
    deduped: list[Heading] = []
    for heading in headings:
        if deduped and deduped[-1].text == heading.text:
            continue
        deduped.append(heading)

    log.debug("detected headings", extra={"count": len(deduped)})
    return deduped


def _SETSET_H2_GUARD(stripped: str) -> bool:
    """True for setext H2 underlines, which are never headings themselves."""
    return bool(_SETEXT_H2.match(stripped))


def build_breadcrumb(doc_title: str, section_path: list[str]) -> str:
    """Compose the breadcrumb prepended to the embedding input.

    Depth is capped at 4 segments. A deep hierarchy produces a breadcrumb that
    dominates the chunk's own text in the embedding, pushing the actual content
    out of the vector's attention.
    """
    parts = [doc_title.strip()]
    for segment in section_path[:3]:
        cleaned = segment.strip()
        if cleaned and cleaned != parts[-1]:
            parts.append(cleaned)
    return " > ".join(p for p in parts if p)


def section_path_at(headings: list[Heading], char_start: int) -> list[str]:
    """Return the heading ancestry covering a given offset.

    The path is the stack of headings open at `char_start`, by level: a level-1
    heading resets the stack, a level-2 heading appends under it, and so on.
    """
    stack: list[tuple[int, str]] = []
    for heading in headings:
        if heading.char_start > char_start:
            break
        while stack and stack[-1][0] >= heading.level:
            stack.pop()
        stack.append((heading.level, heading.text))
    return [text for _, text in stack]
