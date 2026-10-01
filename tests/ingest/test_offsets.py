"""Offset round-trip: the invariant that makes citations exact (FR-10).

`document.cleaned_text[chunk.char_start:chunk.char_end] == chunk.text`

If this breaks, FR-18 degrades from "links to the exact passage" to "links to a
file" — the specific failure the architecture calls out. These tests assert the
invariant directly rather than trusting that the chunker is correct.
"""

from __future__ import annotations

from app.core.config import Settings
from app.db.models import Document, DocumentState
from app.ingest.chunk import ChunkConfig, chunk_document, count_tokens
from app.ingest.extract import clean_text, extract
from app.ingest.worker import create_document, ingest_document

#: A phrase on a single line. The sample document hard-wraps its body text, so a
#: phrase spanning a line break would not exist in the cleaned text.
PHRASE = "may request a refund within 30 days"


def _ingest(session, object_store, provider, settings: Settings, data: bytes, filename: str):
    doc = create_document(session, filename=filename, data=data, object_store=object_store)
    session.commit()
    ingest_document(
        session, doc.doc_id, object_store=object_store, provider=provider, settings=settings
    )
    session.commit()
    return session.get(Document, doc.doc_id)


def test_offsets_round_trip(session, object_store, provider, settings_env, sample_md):
    """Every chunk's offsets reconstruct its text exactly."""
    doc = _ingest(session, object_store, provider, settings_env, sample_md, "policy.md")

    assert doc.state == DocumentState.LIVE
    assert doc.cleaned_text, "canonical cleaned text must be persisted"

    chunks = sorted(doc.chunks, key=lambda c: c.chunk_index)
    assert chunks, "expected at least one chunk"

    for chunk in chunks:
        reconstructed = doc.cleaned_text[chunk.char_start : chunk.char_end]
        assert reconstructed == chunk.text, (
            f"offset round-trip failed for chunk {chunk.chunk_index}: "
            f"offsets [{chunk.char_start}:{chunk.char_end}] do not reconstruct the text"
        )
        assert chunk.char_start < chunk.char_end, "chunk offsets must be non-empty"


def test_offsets_survive_chunking_strategies(sample_md):
    """The invariant holds for every strategy, not just the default.

    A strategy that reconstructs differently is exactly the kind of defect that
    only shows up after Phase 2 switches strategies for an experiment.
    """
    cleaned = extract(sample_md, "policy.md").text

    for strategy in ("heading", "paragraph", "fixed"):
        config = ChunkConfig(strategy=strategy, target_tokens=120, overlap_tokens=20,
                             min_chunk_tokens=5)
        drafts = chunk_document(cleaned, "policy", config=config)
        assert drafts, f"{strategy} produced no chunks"
        for draft in drafts:
            assert cleaned[draft.char_start : draft.char_end] == draft.text, (
                f"offset round-trip failed under strategy={strategy}"
            )


def test_known_phrase_offsets_locate_the_phrase(sample_md):
    """A specific phrase's offsets land on that phrase, not a copy of it.

    Policies repeat sentences, so this is the check that would catch an
    offset derived by searching for text rather than by tracking position.
    """
    cleaned = extract(sample_md, "policy.md").text
    expected = cleaned.index(PHRASE)
    config = ChunkConfig(target_tokens=200, overlap_tokens=20, min_chunk_tokens=5)
    drafts = chunk_document(cleaned, "policy", config=config)

    containing = [d for d in drafts if PHRASE in d.text]
    assert containing, f"expected a chunk containing the phrase {PHRASE!r}"
    for draft in containing:
        assert draft.char_start <= expected < draft.char_end, (
            "chunk containing the phrase does not cover the phrase's first occurrence"
        )


def test_chunk_text_has_no_breadcrumb(session, object_store, provider, settings_env, sample_md):
    """Breadcrumb is on the embedding input, never in the stored text.

    Storing it would put document titles into search results and into every
    citation the user reads.
    """
    doc = _ingest(session, object_store, provider, settings_env, sample_md, "policy.md")
    for chunk in doc.chunks:
        assert "Refund Policy >" not in chunk.text
        assert chunk.breadcrumb, "breadcrumb should still be recorded on the row"


def test_clean_text_is_idempotent():
    """Cleaning twice changes nothing.

    If it did, offsets computed against a re-cleaned document would not match
    the persisted text.
    """
    once = clean_text("Line one   \n\n\n\nLine two\n\n\n12\n\n----\n\nLine three")
    assert clean_text(once) == once


def test_token_count_positive(sample_md):
    cleaned = extract(sample_md, "policy.md").text
    drafts = chunk_document(cleaned, "policy",
                            config=ChunkConfig(target_tokens=100, overlap_tokens=10,
                                               min_chunk_tokens=5))
    for draft in drafts:
        assert draft.token_count > 0
        assert count_tokens(draft.text) == draft.token_count
