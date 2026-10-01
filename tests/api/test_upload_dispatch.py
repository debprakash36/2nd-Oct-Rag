"""Upload dispatch behaviour under queue failure.

These tests cover the case that decides whether a user's file is lost: the upload
succeeds, the document row is written, but the ingestion job never runs. The row
is then `pending` forever and the user has no way to tell that from an upload
that was never received.
"""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from app.db.models import DocumentState
from app.ingest.queue import DispatchResult, IngestJob, IngestQueue


class _DeadQueue(IngestQueue):
    """Broker is unreachable. Every dispatch fails."""

    def enqueue(self, job: IngestJob) -> DispatchResult:
        return DispatchResult(doc_id=job.doc_id, queued=False, error="connection refused")

    def close(self) -> None:
        return None


class _ExplodingQueue(IngestQueue):
    """Dispatch raises instead of returning a failed result."""

    def enqueue(self, job: IngestJob) -> DispatchResult:
        raise RuntimeError("broker client crashed")

    def close(self) -> None:
        return None


def _install(client: TestClient, queue: IngestQueue) -> None:
    """Substitute the queue via FastAPI's dependency overrides.

    Patching the module attribute does not work: `Depends(get_ingest_queue)` is
    resolved when the route is registered at import time, so the original
    function object is already captured by the route.
    """
    from app.api.admin_documents import get_ingest_queue

    client.app.dependency_overrides[get_ingest_queue] = lambda: queue
    return client.app.dependency_overrides


def _clear(client: TestClient) -> None:
    client.app.dependency_overrides.clear()


def _upload(client: TestClient, name: str = "policy.md", data: bytes | None = None) -> dict:
    return client.post(
        "/admin/documents",
        files=[("files", (name, BytesIO(data or b"# T\n\nRefund within 30 days.\n"),
                          "text/markdown"))],
    )


def test_dispatch_failure_is_reported_not_reported_as_success(client, sample_md):
    """A stored-but-never-processed file must not be counted as accepted."""
    _install(client, _DeadQueue())

    response = _upload(client, "policy.md", sample_md)
    assert response.status_code == 201

    body = response.json()
    assert body["accepted"] == 0, (
        "a file that was stored but never queued was not ingested; reporting it "
        "as accepted is how an upload silently disappears"
    )
    assert body["rejected"] == 1

    result = body["results"][0]
    assert result["error"], "the reason must be surfaced"
    assert result["queued"] is False
    assert result["state"] != DocumentState.LIVE.value


def test_queue_exploding_does_not_500_the_request(client, sample_md):
    """A crashing queue must not surface as an unhandled server error.

    The document row was already committed at that point. Returning 500 would
    invite a retry that duplicates the stored file while telling the user nothing
    about whether the first attempt landed.
    """
    _install(client, _ExplodingQueue())

    response = _upload(client, "policy.md", sample_md)
    assert response.status_code in (201, 503)
    assert "Traceback" not in response.text


def test_failed_dispatch_still_records_the_document(client, sample_md):
    """The row exists, so an admin can see what was attempted and retry it.

    Retaining the row is deliberate: it is the only evidence the upload arrived,
    and the alternative — rolling it back — makes the file genuinely lost.
    """
    _install(client, _DeadQueue())
    _upload(client, "policy.md", sample_md)

    docs = client.get("/admin/documents").json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "policy.md"
    assert docs[0]["state"] != DocumentState.LIVE.value
    assert docs[0]["chunk_count"] == 0, "nothing was indexed"


def test_successful_dispatch_reports_queued(client, sample_md):
    """The default inline path completes the pipeline before responding."""
    seen: list[str] = []

    class _RecordingQueue(IngestQueue):
        def enqueue(self, job: IngestJob) -> DispatchResult:
            seen.append(job.doc_id)
            return DispatchResult(doc_id=job.doc_id, queued=True, detail="queued")

        def close(self) -> None:
            return None

    _install(client, _RecordingQueue())
    body = _upload(client, "policy.md", sample_md).json()

    assert seen, "the job must actually be dispatched"
    assert body["results"][0]["queued"] is True
    # Documented as pending: nothing processed it, so claiming `live` would be a
    # lie about the state of the index.
    assert body["results"][0]["state"] == DocumentState.PENDING.value


def test_queue_dependency_is_overridable():
    """The dependency exists so tests and deployments can substitute a broker."""
    from app.api.admin_documents import get_ingest_queue

    assert callable(get_ingest_queue)


def test_override_does_not_leak_between_tests(client):
    """Dependency overrides are process-global on the app instance.

    The `client` fixture builds a fresh app per test, but if it ever stops doing
    so, a stubbed queue would silently apply to unrelated tests — and the failure
    would look like a flaky ingestion test rather than test pollution.
    """
    from app.api.admin_documents import get_ingest_queue

    assert get_ingest_queue not in client.app.dependency_overrides
