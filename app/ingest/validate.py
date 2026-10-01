"""Upload validation (FR-1, FR-2).

Validation is by extension, not by declared MIME type. A client controls the
content type it sends, so trusting it would let any file be uploaded as a PDF.
The extension determines the parser, which is what makes the choice security
relevant rather than cosmetic.

Size is checked before the file is read into memory wherever the framework allows
it. `UploadFile` streams to a spooled temp file, so reading `size` first avoids
materializing a 25 MB file only to reject it.
"""

from __future__ import annotations

import re

from fastapi import UploadFile

from app.core.config import Settings
from app.core.errors import ValidationError
from app.ingest.extract import SUPPORTED_EXTENSIONS, supported_extension

#: Conservative ceiling on the stored filename. Long enough for any real name,
#: short enough to keep the DB column bounded.
MAX_FILENAME_CHARS = 255
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def validate_upload(upload: UploadFile, settings: Settings) -> tuple[str, str]:
    """Validate one uploaded file. Returns (safe_filename, extension).

    Raises `ValidationError` with a message safe to show the user. Per-file
    validation rather than rejecting the whole batch, because one bad file in a
    50-file upload should not discard the other 49 (FR-2).
    """
    raw_name = upload.filename or ""
    filename = _safe_filename(raw_name)

    extension = supported_extension(filename)
    if extension is None:
        raise ValidationError(
            f"unsupported file type for {raw_name!r}",
            user_message=(
                f"'{filename}' is not a supported file type. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    size = upload.size or 0
    if size == 0:
        raise ValidationError(
            f"empty upload: {raw_name!r}", user_message=f"'{filename}' is empty."
        )
    if size > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise ValidationError(
            f"{filename} is {size} bytes, over the {settings.max_upload_bytes} limit",
            user_message=f"'{filename}' is too large. Maximum size is {limit_mb} MB.",
        )

    return filename, extension


def guess_mime(filename: str) -> str:
    """Best-effort MIME type from the extension.

    Used only as a record on the document. Validation never trusts it — see the
    module docstring.
    """
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".html": "text/html",
        ".htm": "text/html",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
    }.get(supported_extension(filename) or "", "application/octet-stream")


def _safe_filename(raw: str) -> str:
    """Reduce an uploaded filename to a safe display name.

    Control characters are stripped, path separators are removed, and the length
    is capped. This name is stored and shown in the UI, so it must not be able to
    carry a newline (log injection) or a path (confusing later exports).

    It is never used to build a filesystem path — object keys are generated
    server-side in `objectstore.new_key`.
    """
    name = _CONTROL_CHARS.sub("", raw).replace("/", "_").replace("\\", "_").strip()
    # Strip leading dots so the name cannot become a hidden file or a relative
    # path fragment. The first surviving character anchors the name, which is
    # why "..." becomes "document" rather than an empty string.
    name = name.lstrip(".") or "document"
    return name[:MAX_FILENAME_CHARS]
