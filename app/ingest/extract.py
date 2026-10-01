"""Text extraction and cleaning (FR-3).

Two responsibilities, deliberately separate:

1. `extract_*` parse a file into raw text.
2. `clean_text` produces the **canonical cleaned text** — the single string every
   chunk offset is measured against.

Separating them is what makes offsets trustworthy. The common bug (called out in
implementation.md 3.2) is computing offsets against one rendering of the text
while slicing another. Here there is exactly one canonical output, it is
persisted on the document, and every offset derives from it. The invariant
`document.cleaned_text[chunk.char_start:chunk.char_end] == chunk.text` is
asserted in tests/ingest/test_offsets.py.
"""

from __future__ import annotations

import io
import re
import unicodedata
from collections.abc import Callable

from app.core.errors import ExtractionError
from app.ingest.sandbox import ExtractionResult

# Page-number-only lines and bare rules: the residue of header/footer stripping
# that survives and would otherwise become its own tiny chunk.
_PAGE_NOISE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)
_BLANK_RUN = re.compile(r"\n{3,}")
_HORIZONTAL_RULE = re.compile(r"^\s*(?:[-*_=]\s*){3,}$")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(raw: str) -> str:
    """Produce the canonical cleaned text.

    Normalization is lossy in only one direction: it never reorders or drops
    sentences, because offsets must stay contiguous. It strips the noise that
    would otherwise become its own chunk and make a citation look wrong.
    """
    # NFC so visually identical text produces identical strings — otherwise a
    # composed/decomposed mismatch silently breaks offset resolution.
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = _TRAILING_WS.sub("", text)

    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _PAGE_NOISE.match(stripped) or _HORIZONTAL_RULE.match(stripped):
            # Drop the line entirely rather than blanking it, so the min-token
            # filter cannot rescue it into a chunk.
            continue
        kept.append(line)
    return _BLANK_RUN.sub("\n\n", "\n".join(kept)).strip()


def _pdf(data: bytes) -> ExtractionResult:
    from pypdf import PdfReader

    try:
        # A stream, not raw bytes: pypdf needs a file-like object and a temp
        # write would put untrusted bytes on the filesystem.
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # An encrypted PDF with an empty user password still opens, so try
            # before declaring it unreadable.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ExtractionError(f"encrypted PDF: {type(exc).__name__}") from exc
        pages = [page.extract_text() or "" for page in reader.pages]
        return ExtractionResult(text="\n\n".join(pages), page_count=len(reader.pages))
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"pdf parse failed: {type(exc).__name__}: {exc}") from exc


def _docx(data: bytes) -> ExtractionResult:
    import docx

    try:
        # python-docx takes a path or a stream; a stream keeps untrusted bytes
        # off the filesystem (NFR-5).
        document = docx.Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return ExtractionResult(text="\n\n".join(p for p in parts if p), page_count=None)
    except Exception as exc:
        raise ExtractionError(f"docx parse failed: {type(exc).__name__}: {exc}") from exc


def _html(stream: bytes) -> ExtractionResult:
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(stream, "html.parser")
        # Structural chrome is boilerplate, and FR-3 requires it stripped.
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()
        return ExtractionResult(text=soup.get_text(separator="\n"), page_count=None)
    except Exception as exc:
        raise ExtractionError(f"html parse failed: {type(exc).__name__}: {exc}") from exc


def _plain(stream: bytes) -> ExtractionResult:
    return ExtractionResult(text=stream.decode("utf-8", errors="replace"), page_count=None)


#: Extension to extractor. Extension rather than MIME because clients commonly
#: send a generic or absent content type for `.md` and `.txt`.
_EXTRACTORS: dict[str, Callable[[bytes], ExtractionResult]] = {
    ".pdf": _pdf,
    ".docx": _docx,
    ".html": _html,
    ".htm": _html,
    ".txt": _plain,
    ".md": _plain,
    ".markdown": _plain,
}

SUPPORTED_EXTENSIONS = frozenset(_EXTRACTORS)


def supported_extension(filename: str) -> str | None:
    """Return the lowercase extension if supported, else None."""
    dot = filename.rfind(".")
    if dot == -1:
        return None
    ext = filename[dot:].lower()
    return ext if ext in _EXTRACTORS else None


def extract(data: bytes, filename: str) -> ExtractionResult:
    """Extract raw text from one file. Call inside the sandbox."""
    ext = supported_extension(filename)
    if ext is None:
        raise ExtractionError(f"unsupported extension for {filename!r}")

    result = _EXTRACTORS[ext](data)
    if not result.text.strip():
        # Named explicitly because the most common cause is a scanned document,
        # and the actionable message is that OCR is out of scope for v1.
        raise ExtractionError(
            f"no text extracted from {filename!r}; likely scanned or image-only "
            "(OCR is out of scope for v1)"
        )

    cleaned = clean_text(result.text)
    if not cleaned:
        raise ExtractionError(f"text was entirely noise after cleaning: {filename!r}")
    return ExtractionResult(text=cleaned, page_count=result.page_count)
