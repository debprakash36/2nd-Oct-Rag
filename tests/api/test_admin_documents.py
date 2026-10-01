"""Admin API surface (FR-1, FR-2, FR-7, FR-27)."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from app.db.models import DocumentState


def _upload(client: TestClient, name: str, data: bytes) -> dict:
    return client.post(
        "/admin/documents",
        files=[("files", (name, BytesIO(data), "text/markdown"))],
    )


def test_health_reports_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["checks"]["database"] == "ok"


def test_upload_ingests_document(client: TestClient, sample_md: bytes):
    response = _upload(client, "policy.md", sample_md)
    assert response.status_code == 201

    body = response.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 0

    result = body["results"][0]
    assert result["state"] == DocumentState.LIVE.value
    assert result["chunks"] > 0
    assert result["doc_id"]


def test_list_shows_state_and_chunk_count(client: TestClient, sample_md: bytes):
    _upload(client, "policy.md", sample_md)

    response = client.get("/admin/documents")
    assert response.status_code == 200

    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["state"] == DocumentState.LIVE.value
    assert docs[0]["chunk_count"] > 0
    assert docs[0]["filename"] == "policy.md"


def test_batch_isolates_bad_files(client: TestClient, sample_md: bytes):
    """One bad file must not discard the rest of the batch (FR-2)."""
    response = client.post(
        "/admin/documents",
        files=[
            ("files", ("good.md", BytesIO(sample_md), "text/markdown")),
            ("files", ("bad.zip", BytesIO(b"PK\x03\x04"), "application/zip")),
            ("files", ("empty.txt", BytesIO(b""), "text/plain")),
        ],
    )
    assert response.status_code == 201

    body = response.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 2

    by_name = {r["filename"]: r for r in body["results"]}
    assert by_name["good.md"]["state"] == DocumentState.LIVE.value
    assert "not a supported file type" in by_name["bad.zip"]["error"]
    assert "is empty" in by_name["empty.txt"]["error"]


def test_oversize_rejected(client: TestClient, settings_env, monkeypatch):
    # The endpoint reads settings through the injected dependency, so the test
    # must lower the limit on the same instance the app is wired to.
    monkeypatch.setattr(settings_env, "max_upload_bytes", 100, raising=False)

    response = _upload(client, "big.md", b"x" * 500)
    assert response.status_code == 201
    assert response.json()["rejected"] == 1
    assert "too large" in response.json()["results"][0]["error"]


def test_duplicate_is_flagged_not_dropped(client: TestClient, sample_md: bytes):
    """FR-4: recorded so an admin can decide, never silently dropped.

    The upload is acknowledged -- not rejected, not reported as an error -- and the
    `duplicate_of` pointer plus the listing entry are what surface it. But it is not
    indexed: it reports state `duplicate` rather than `live`, so it is excluded from
    retrieval while remaining visible and promotable.
    """
    _upload(client, "policy.md", sample_md)
    response = _upload(client, "policy-copy.md", sample_md)

    body = response.json()
    assert body["rejected"] == 0, "a duplicate is not a failure; it is a flag"
    result = body["results"][0]
    assert result["duplicate_of"], "duplicate must be reported, not hidden"
    assert result["state"] == DocumentState.DUPLICATE.value

    # The listing exposes it too, so an admin sees it without opening each file.
    listed = {d["filename"]: d for d in client.get("/admin/documents").json()}
    assert listed["policy-copy.md"]["duplicate_of"] == listed["policy.md"]["doc_id"]


def test_disable_and_enable_roundtrip(client: TestClient, sample_md: bytes):
    doc_id = _upload(client, "policy.md", sample_md).json()["results"][0]["doc_id"]

    disabled = client.post(f"/admin/documents/{doc_id}/disable")
    assert disabled.status_code == 200
    assert disabled.json()["state"] == DocumentState.DISABLED.value
    assert disabled.json()["chunk_count"] > 0, "disable keeps chunks (tombstone, not delete)"

    enabled = client.post(f"/admin/documents/{doc_id}/enable")
    assert enabled.json()["state"] == DocumentState.LIVE.value


def test_delete_returns_204_and_hides_document(client: TestClient, sample_md: bytes):
    doc_id = _upload(client, "policy.md", sample_md).json()["results"][0]["doc_id"]

    assert client.delete(f"/admin/documents/{doc_id}").status_code == 204
    # 404, not 400: the resource is gone. Phase 4 moved `NotFoundError.status_code`
    # off the blanket 400 so a client can tell "deleted" from "your request was
    # malformed" — the difference between stopping and retrying forever.
    assert client.get(f"/admin/documents/{doc_id}").status_code == 404
    assert all(d["doc_id"] != doc_id for d in client.get("/admin/documents").json())


def test_get_single_document(client: TestClient, sample_md: bytes):
    doc_id = _upload(client, "policy.md", sample_md).json()["results"][0]["doc_id"]
    response = client.get(f"/admin/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["doc_id"] == doc_id


def test_unknown_document_returns_safe_error(client: TestClient):
    """NFR-5: no stack trace, no internal detail."""
    response = client.get("/admin/documents/does-not-exist")
    assert response.status_code == 404
    assert "Traceback" not in response.text
    # The id is echoed, so this asserts the *message* is the safe one rather than a
    # provider error or a path. The id is client-supplied, not internal.
    assert response.json()["detail"] == "document does-not-exist not found."


def test_trace_id_header_returned(client: TestClient):
    """NFR-8: a request is reconstructable from its ID alone."""
    response = client.get("/health")
    assert response.headers.get("X-Trace-Id")


def test_inbound_trace_id_is_honoured(client: TestClient):
    response = client.get("/health", headers={"X-Trace-Id": "abc123"})
    assert response.headers["X-Trace-Id"] == "abc123"
