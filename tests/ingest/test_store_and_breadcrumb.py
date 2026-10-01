"""Object store containment and breadcrumb construction."""

from __future__ import annotations

import pytest

from app.ingest.breadcrumb import (
    build_breadcrumb,
    detect_headings,
    looks_like_heading,
    section_path_at,
)
from app.ingest.objectstore import LocalObjectStore


def test_objectstore_roundtrip(object_store: LocalObjectStore):
    key = object_store.new_key("abc123", "policy.md")
    object_store.put(key, b"hello")
    assert object_store.get(key) == b"hello"
    assert object_store.exists(key)


def test_objectstore_rejects_traversal(object_store: LocalObjectStore):
    """Path traversal must not escape the store root (NFR-5).

    Without the containment check, a crafted key reads or writes anywhere the
    process can reach.
    """
    with pytest.raises(ValueError, match="escapes object store root"):
        object_store.put("../../escape.txt", b"nope")
    with pytest.raises(ValueError, match="escapes object store root"):
        object_store.get("../../../etc/passwd")


def test_objectstore_key_is_not_user_controlled(object_store: LocalObjectStore):
    """Keys are generated server-side so a filename cannot influence the path."""
    key = object_store.new_key("doc1", "../../evil.txt")
    assert ".." not in key
    assert key.startswith("do/")


def test_objectstore_delete_is_idempotent(object_store: LocalObjectStore):
    key = object_store.new_key("abc", "a.txt")
    object_store.put(key, b"x")
    object_store.delete(key)
    object_store.delete(key)  # must not raise


def test_objectstore_reupload_does_not_clobber(object_store: LocalObjectStore):
    """Re-uploading the same filename keeps the previous object.

    Overwriting before the new version is indexed would leave a document whose
    stored bytes no longer match its indexed content.
    """
    first = object_store.new_key("doc1", "policy.md")
    second = object_store.new_key("doc1", "policy.md")
    assert first != second


def test_markdown_headings_detected():
    text = "# Title\n\nBody text.\n\n## Section\n\nMore body.\n\n### Subsection\n\nText."
    headings = detect_headings(text)
    levels = [(h.level, h.text) for h in headings]
    assert (1, "Title") in levels
    assert (2, "Section") in levels
    assert (3, "Subsection") in levels


def test_setext_headings_detected():
    text = "Document Title\n=============\n\nBody.\n\nSub Section\n-----------\n\nMore."
    headings = detect_headings(text)
    texts = [h.text for h in headings]
    assert "Document Title" in texts
    assert "Sub Section" in texts


def test_numbered_headings_detected():
    detected = looks_like_heading("2.1 Scope of the policy")
    assert detected is not None
    assert detected[0] == 2


def test_sentence_is_not_a_heading():
    """A false positive adds a spurious breadcrumb segment, which is a cost."""
    assert looks_like_heading("The customer must file a claim within thirty days.") is None


def test_long_all_caps_is_not_a_heading():
    assert (
        looks_like_heading(
            "THIS SENTENCE IS WRITTEN ENTIRELY IN CAPITAL LETTERS AND KEEPS GOING ON"
        )
        is None
    )


def test_heading_offsets_are_exact():
    text = "Intro paragraph.\n\n## Refund Window\n\nCustomers may refund."
    headings = detect_headings(text)
    refund = next(h for h in headings if h.text == "Refund Window")
    assert text[refund.char_start : refund.char_start + len("## Refund Window")] == (
        "## Refund Window"
    )


def test_section_path_nests_by_level():
    text = "# Top\n\na\n\n## Middle\n\nb\n\n### Deep\n\nc\n\n## Other\n\nd"
    headings = detect_headings(text)
    deep_start = next(h.char_start for h in headings if h.text == "Deep")
    path = section_path_at(headings, deep_start + 10)
    assert path == ["Top", "Middle", "Deep"]

    other_start = next(h.char_start for h in headings if h.text == "Other")
    assert section_path_at(headings, other_start + 10) == ["Top", "Other"], (
        "a new level-2 heading must reset the stack, not append to the old one"
    )


def test_breadcrumb_joins_title_and_path():
    assert build_breadcrumb("Refund Policy", ["Digital Products", "30 Days"]) == (
        "Refund Policy > Digital Products > 30 Days"
    )


def test_breadcrumb_caps_depth():
    """A deep hierarchy would dominate the chunk's own text in the embedding."""
    deep = [f"Section {i}" for i in range(10)]
    breadcrumb = build_breadcrumb("Title", deep)
    assert len(breadcrumb.split(" > ")) <= 4


def test_breadcrumb_handles_empty_parts():
    assert build_breadcrumb("Title", []) == "Title"
    assert build_breadcrumb("Title", ["", "  "]) == "Title"
