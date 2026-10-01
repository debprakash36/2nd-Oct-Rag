"""Sandbox and extraction behaviour (FR-3, NFR-5)."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import ExtractionError, SandboxError, ValidationError
from app.ingest.extract import clean_text, extract, supported_extension
from app.ingest.sandbox import run_sandboxed
from app.ingest.validate import _safe_filename, guess_mime, validate_upload


def test_extract_markdown(sample_md):
    result = extract(sample_md, "policy.md")
    assert "Refund Policy" in result.text
    assert result.text.strip() == result.text, "cleaned text must not have leading/trailing space"


def test_extract_html_strips_boilerplate():
    html = b"""<html><head><style>.a{color:red}</style></head>
    <body><nav>Home About Contact</nav><h1>Policy</h1>
    <p>Refunds are available within 30 days of purchase.</p>
    <footer>Copyright 2026</footer></body></html>"""
    result = extract(html, "page.html")
    assert "Policy" in result.text
    assert "Refunds are available" in result.text
    # FR-3 requires boilerplate stripped.
    assert "color:red" not in result.text
    assert "Copyright" not in result.text


def test_empty_file_raises():
    with pytest.raises(ExtractionError, match="no text extracted"):
        extract(b"", "empty.txt")


def test_scanned_pdf_message_mentions_ocr(sample_md):
    """A scanned document is the common cause, so the message must name it."""
    with pytest.raises(ExtractionError, match="OCR is out of scope"):
        extract(b"   \n\n  ", "scanned.txt")


def test_unsupported_extension_raises():
    with pytest.raises(ExtractionError, match="unsupported extension"):
        extract(b"data", "archive.zip")


def test_supported_extension_is_case_insensitive():
    assert supported_extension("POLICY.PDF") == ".pdf"
    assert supported_extension("Policy.MD") == ".md"


def test_clean_text_removes_page_numbers():
    """Page numbers would otherwise become their own undersized chunks."""
    cleaned = clean_text("Real content here.\n\n12\n\nMore real content.\n\nPage 3 of 10\n")
    assert "12" not in cleaned
    assert "Page 3 of 10" not in cleaned
    assert "Real content here." in cleaned


def test_clean_text_normalizes_line_endings():
    assert clean_text("a\r\nb\rc") == clean_text("a\nb\nc")


def test_clean_text_collapses_blank_runs():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_sandbox_enforces_timeout():
    """A hanging extractor must be killed, not awaited forever."""
    with pytest.raises(SandboxError, match="timeout"):
        run_sandboxed("sleep", timeout_seconds=1, max_output_chars=1000, seconds=30)


def test_sandbox_rejects_wrong_return_type():
    """A child returning the wrong shape is an error, not a silent empty result."""
    with pytest.raises(ExtractionError, match="expected ExtractionResult"):
        run_sandboxed("wrong_type", timeout_seconds=10, max_output_chars=1000)


def test_sandbox_caps_output():
    """The cap must bound memory from a decompression bomb (NFR-5)."""
    result = run_sandboxed(
        "echo", data=b"x" * 10_000, timeout_seconds=10, max_output_chars=100
    )
    assert len(result.text) <= 100


def test_sandbox_preserves_structure_across_process_boundary():
    """`page_count` is needed for citation display (FR-6) and must survive the
    process boundary rather than requiring a second extraction pass."""
    cleaned = extract(sample_bytes(), "policy.md")
    result = run_sandboxed("extract", data=sample_bytes(), filename="policy.md",
                           timeout_seconds=20, max_output_chars=100_000)
    assert result.text == cleaned.text, "sandboxed extraction must match direct extraction"
    assert result.page_count == cleaned.page_count


def test_sandbox_propagates_child_exception():
    with pytest.raises(ExtractionError, match="deliberate sandbox failure"):
        run_sandboxed("boom", timeout_seconds=10, max_output_chars=1000)


def test_sandbox_rejects_unknown_operation():
    """The operation registry is the whole capability surface."""
    with pytest.raises(ValueError, match="unknown sandbox operation"):
        run_sandboxed("rm_rf", timeout_seconds=1)


def test_sandbox_capabilities_reported_not_assumed():
    """NFR-5 is only partly satisfiable on Windows; the gap must be visible."""
    from app.ingest.sandbox import sandbox_capabilities

    caps = sandbox_capabilities()
    assert caps["wall_clock_timeout"] is True
    assert caps["output_cap"] is True
    # Recorded, not hidden: on Windows this is False, and a production deploy
    # must run the worker in a container to close the gap.
    assert "posix_resource_limits" in caps
    assert "network_isolation" in caps


def sample_bytes() -> bytes:
    """A markdown document used by the sandbox round-trip tests."""
    return b"# Title\n\nSome content that should survive the process boundary.\n"


def test_safe_filename_strips_path_and_control_chars():
    """A stored filename must not carry a path or a newline.

    Newlines would allow log injection (FR-3/FR-22); separators would produce
    confusing exports. The name is never used to build a filesystem path — object
    keys are generated server-side (NFR-5).
    """
    sanitized = _safe_filename("../../etc/passwd")
    assert "/" not in sanitized
    assert not sanitized.startswith("."), "a leading dot could hide the file"
    assert "\\" not in _safe_filename("..\\..\\windows\\system32")
    assert "\n" not in _safe_filename("bad\nname.txt")
    assert "\x00" not in _safe_filename("null\x00.txt")
    assert len(_safe_filename("a" * 500)) == 255


def test_safe_filename_never_empty():
    assert _safe_filename("...") == "document"
    assert _safe_filename("") == "document"


def test_guess_mime_ignores_client_claim():
    """Validation is by extension, not declared type (FR-1).

    A client controls the content type it sends, so trusting it would let any
    file be uploaded as a PDF.
    """
    assert guess_mime("a.pdf") == "application/pdf"
    assert guess_mime("a.exe") == "application/octet-stream"


class _FakeUpload:
    """Minimal UploadFile stand-in for validation tests."""

    def __init__(self, filename: str, size: int) -> None:
        self.filename = filename
        self.size = size


def test_validate_rejects_unsupported_type(settings_env: Settings):
    upload = _FakeUpload("archive.zip", 100)
    with pytest.raises(ValidationError) as exc:
        validate_upload(upload, settings_env)
    assert "not a supported file type" in exc.value.user_message


def test_validate_rejects_oversize(settings_env: Settings):
    upload = _FakeUpload("big.pdf", settings_env.max_upload_bytes + 1)
    with pytest.raises(ValidationError) as exc:
        validate_upload(upload, settings_env)
    assert "too large" in exc.value.user_message


def test_validate_rejects_empty(settings_env: Settings):
    upload = _FakeUpload("empty.txt", 0)
    with pytest.raises(ValidationError) as exc:
        validate_upload(upload, settings_env)
    assert "is empty" in exc.value.user_message


def test_validate_accepts_supported(settings_env: Settings):
    name, extension = validate_upload(_FakeUpload("policy.md", 500), settings_env)
    assert (name, extension) == ("policy.md", ".md")


def test_validation_error_message_does_not_leak_paths(settings_env: Settings):
    """The user message must name the file, not the server's filesystem path."""
    upload = _FakeUpload("a" * 300 + ".zip", 100)
    with pytest.raises(ValidationError) as exc:
        validate_upload(upload, settings_env)
    assert "Traceback" not in exc.value.user_message
    assert "/" not in exc.value.user_message
