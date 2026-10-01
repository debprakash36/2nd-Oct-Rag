"""Local filesystem object store.

Stand-in for S3 so the pipeline runs without cloud credentials. The interface
deliberately matches what an S3 client looks like (`put`/`get`/`delete`), so
swapping in boto3 is a change to this file only (architecture.md 4.3).

Object keys are generated server-side, never derived from the uploaded filename.
A user-supplied name containing `../` would otherwise escape the store root.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger("app.ingest.objectstore")


class LocalObjectStore:
    """Filesystem-backed object storage."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LocalObjectStore:
        return cls((settings or get_settings()).object_store_dir)

    def put(self, key: str, data: bytes) -> str:
        """Store bytes at `key` and return the key."""
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        """Read bytes back. Raises FileNotFoundError if absent."""
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def delete(self, key: str) -> None:
        """Remove an object. Missing keys are not an error."""
        path = self._resolve(key)
        if path.is_file():
            path.unlink()
            log.info("deleted object", extra={"key": key})

    def purge_prefix(self, prefix: str) -> int:
        """Remove every object under a prefix. Returns the count."""
        root = self._resolve(prefix)
        if not root.is_dir():
            return 0
        count = sum(1 for p in root.rglob("*") if p.is_file())
        shutil.rmtree(root, ignore_errors=True)
        return count

    def new_key(self, doc_id: str, filename: str) -> str:
        """Generate a safe storage key for a document.

        The key includes the doc_id and a random component, so re-uploading the
        same filename does not overwrite the previous object before the new
        version has been indexed.
        """
        suffix = Path(filename).suffix.lower()[:16]
        return f"{doc_id[:2]}/{doc_id}_{uuid.uuid4().hex[:8]}{suffix}"

    def _resolve(self, key: str) -> Path:
        """Resolve a key inside the store root, rejecting traversal.

        `resolve()` followed by a containment check is the reliable form. The
        check is what matters: without it, a key like `../../secrets` resolves
        outside the root and the containment test is the only thing standing
        between an upload and arbitrary file read.
        """
        candidate = (self._root / key).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"key escapes object store root: {key!r}")
        return candidate
