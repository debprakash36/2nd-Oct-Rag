"""Chunking strategy tests.

Both strategies here were previously non-functional in a way that no offset or
invariant test could catch: paragraph segmentation collapsed to a single chunk
and overlap was clamped to zero. These tests assert the behaviour directly.
"""

from __future__ import annotations

import itertools

from app.ingest.chunk import ChunkConfig, chunk_document


def _paragraphs(n: int, words: int = 40) -> str:
    body = " ".join(f"word{i}" for i in range(words))
    return "\n\n".join(f"Paragraph {p} {body}" for p in range(n))


def _paragraph_chunks(text: str, **kwargs):
    kwargs.setdefault("overlap_tokens", 20)
    config = ChunkConfig(
        strategy="paragraph", min_chunk_tokens=kwargs.pop("min_chunk_tokens", 10), **kwargs
    )
    return chunk_document(text, "Doc", config=config)


def test_paragraph_strategy_actually_segments():
    """Paragraph strategy must produce more than one unit-sized chunk.

    The separator search previously built units out of the blank-line separators
    themselves, so every candidate was whitespace, every candidate was filtered
    out, and the fallback collapsed the whole document into one chunk. A strategy
    that does not segment is indistinguishable from a broken one when only
    invariants are checked.
    """
    drafts = _paragraph_chunks(_paragraphs(12), target_tokens=100)
    assert len(drafts) > 1, "paragraph strategy must split a multi-paragraph document"


def test_paragraph_units_span_paragraphs_not_separators():
    """Each chunk's span must contain real text, not just blank lines."""
    drafts = _paragraph_chunks(_paragraphs(8), target_tokens=100)
    assert drafts
    for draft in drafts:
        assert draft.text.strip(), "a chunk of pure whitespace is not content"


def test_overlap_is_actually_applied():
    """Consecutive chunks must share text when overlap is configured.

    `overlap_tokens=0` vs a positive value must produce different documents. With
    the rewind clamped to the incoming unit's start — which is always past the
    closed window — overlap was always zero and the setting did nothing.
    """
    text = _paragraphs(20)

    without = _paragraph_chunks(text, target_tokens=120, overlap_tokens=0)
    with_overlap = _paragraph_chunks(text, target_tokens=120, overlap_tokens=40)

    assert len(without) > 1
    assert len(with_overlap) > 1

    # The first window starts at 0 either way — overlap only affects windows after
    # the first, so the comparison must be on the second chunk's start offset.
    assert with_overlap[0].char_start == without[0].char_start == 0
    assert with_overlap[1].char_start < without[1].char_start, (
        "a positive overlap must start the next chunk earlier, not later"
    )
    assert any(
        b.char_start < a.char_end for a, b in itertools.pairwise(with_overlap)
    ), "consecutive overlapping chunks must share a region of the source text"


def test_zero_overlap_chunks_do_not_share_a_region():
    """The control: with overlap disabled, consecutive windows must not overlap."""
    drafts = _paragraph_chunks(_paragraphs(20), target_tokens=120, overlap_tokens=0)
    assert len(drafts) > 1
    for a, b in itertools.pairwise(drafts):
        assert b.char_start >= a.char_end, (
            "with overlap_tokens=0 each chunk must start where the previous ended"
        )


def test_fixed_strategy_respects_window():
    """The baseline strategy slices on a fixed character window."""
    text = "word " * 4000
    drafts = chunk_document(
        text,
        "Doc",
        config=ChunkConfig(
            strategy="fixed", target_tokens=100, overlap_tokens=0, min_chunk_tokens=10
        ),
    )
    assert len(drafts) > 1
    assert all(d.token_count <= 100 for d in drafts)


def test_empty_text_is_rejected():
    from app.core.errors import ChunkingError

    try:
        chunk_document("   \n\n  ", "Doc")
    except ChunkingError:
        return
    raise AssertionError("empty text must raise rather than produce zero chunks")


def test_oversized_paragraph_is_emitted_whole_not_split():
    """A unit larger than the target is kept whole so sentences are not cut (FR-9)."""
    huge = " ".join(f"w{i}" for i in range(2000))
    drafts = _paragraph_chunks(
        f"Short intro.\n\n{huge}\n\nShort outro.",
        target_tokens=100,
        min_chunk_tokens=10,
    )
    assert any(d.token_count > 100 for d in drafts), (
        "an oversized paragraph must stay whole; the target is a soft limit"
    )
    assert all(d.text.strip() for d in drafts)


def test_chunk_indices_are_sequential_from_zero():
    """Indices must be dense and ordered; a gap would break citation paging."""
    drafts = _paragraph_chunks(_paragraphs(15), target_tokens=100)
    assert [d.index for d in drafts] == list(range(len(drafts)))


def test_offsets_reconstruct_text_for_every_strategy():
    """The citation invariant, checked across all strategies rather than one."""
    text = _paragraphs(10)
    for strategy in ("heading", "paragraph", "fixed"):
        drafts = chunk_document(
            text,
            "Doc",
            config=ChunkConfig(
                strategy=strategy, target_tokens=80, overlap_tokens=10, min_chunk_tokens=5
            ),
        )
        assert drafts, f"{strategy} produced no chunks"
        for draft in drafts:
            assert text[draft.char_start : draft.char_end] == draft.text, (
                f"{strategy} produced a chunk whose text does not match its offsets"
            )


def _as_config(**kwargs) -> ChunkConfig:
    return ChunkConfig(**kwargs)