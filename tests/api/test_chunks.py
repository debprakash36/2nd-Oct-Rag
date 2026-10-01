"""Exact-passage endpoint tests (FR-18, FR-20, NFR-9).

This endpoint exists to satisfy "click a citation, see the exact passage". The tests
that matter are the ones about a passage outliving its document, because that is the
case where a citation the user saved last week stops resolving.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


@pytest.fixture
def chunk_id(live_doc) -> str:
    return live_doc.chunks[0].chunk_id


class TestReadPassage:
    def test_returns_exact_text(self, client: TestClient, chunk_id: str, live_doc):
        response = client.get(f"/chunks/{chunk_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == live_doc.chunks[0].text
        assert body["chunk_id"] == chunk_id

    def test_includes_source_location(self, client: TestClient, chunk_id: str, live_doc):
        body = client.get(f"/chunks/{chunk_id}").json()
        assert body["filename"] == "policy.md"
        assert body["document_id"] == live_doc.doc_id
        assert "breadcrumb" in body
        assert "page" in body

    def test_text_is_not_html_escaped_server_side(
        self, client: TestClient, chunk_id: str, live_doc
    ):
        """Escaping is a render-boundary job (FR-22), so the API returns raw text.

        Escaping here would hand the client `&amp;` and force it to unescape before
        displaying, and a client that forgot would render the escape literally.
        """
        text = client.get(f"/chunks/{chunk_id}").json()["text"]
        assert "<" not in text or "passage" in text.lower()

    def test_unknown_chunk_returns_404(self, client: TestClient):
        response = client.get("/chunks/does-not-exist")
        assert response.status_code == 404
        assert "Traceback" not in response.text

    def test_missing_and_dead_chunk_are_indistinguishable(self, client: TestClient):
        """404 for both, so the endpoint is not a chunk-id oracle.

        A client is unauthenticated in v1, so distinguishing them would hand out free
        reconnaissance for no user benefit.
        """
        missing = client.get("/chunks/does-not-exist")
        assert missing.status_code == 404
        assert "Traceback" not in missing.text


class TestDocumentState:
    def test_disabled_document_hides_passage(
        self, client: TestClient, chunk_id: str, session: Session
    ):
        """A citation must not leak text from a document the corpus no longer serves."""
        from app.ingest.tombstone import disable

        disable(session, live_doc_id(chunk_id, session))
        session.commit()

        response = client.get(f"/chunks/{chunk_id}")
        assert response.status_code == 404

    def test_deleted_document_hides_passage(
        self, client: TestClient, chunk_id: str, session: Session
    ):
        from app.ingest.tombstone import mark_deleted

        mark_deleted(session, live_doc_id(chunk_id, session))
        session.commit()

        assert client.get(f"/chunks/{chunk_id}").status_code == 404

    def test_re_enabled_document_restores_passage(
        self, client: TestClient, chunk_id: str, session: Session
    ):
        """State gating is not destructive: the passage comes back."""
        from app.ingest.tombstone import disable, enable

        doc_id = live_doc_id(chunk_id, session)
        disable(session, doc_id)
        session.commit()
        assert client.get(f"/chunks/{chunk_id}").status_code == 404

        enable(session, doc_id)
        session.commit()
        assert client.get(f"/chunks/{chunk_id}").status_code == 200


def live_doc_id(chunk_id: str, session: Session) -> str:
    from app.db.models import Chunk

    return session.get(Chunk, chunk_id).doc_id


class TestPayloadShape:
    def test_response_is_json_not_html(self, client: TestClient, chunk_id: str):
        """The passage is untrusted corpus text; it must never be served as markup."""
        response = client.get(f"/chunks/{chunk_id}")
        assert response.headers["content-type"].startswith("application/json")

    def test_citation_markers_absent_from_passage(self, client: TestClient, chunk_id: str):
        """FR-18/FR-17: the stored passage is the source text, not the answer.

        If a marker survived ingestion, the sources panel would render the model's
        numbering as part of the document, which reads as authoritative.
        """
        text = client.get(f"/chunks/{chunk_id}").json()["text"]
        assert "[1]" not in text
        assert "passage n=" not in text