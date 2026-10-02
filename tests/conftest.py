"""Shared pytest fixtures.

Every test runs against a fresh temp SQLite file with a pinned embedding
dimension. The dimension is pinned rather than discovered because the idempotency
tests depend on embeddings being deterministic across the run, and a random
dimension would make a re-embed produce different vectors and look like drift.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.core.config import Settings

os.environ.setdefault("ENVIRONMENT", "test")
# The developer's `.env` may set API_TOKEN. Tests construct an open API unless a
# test sets the token itself, so the env var has to win over the file.
os.environ["API_TOKEN"] = ""


@pytest.fixture
def settings_env(tmp_path: Path) -> Iterator[Settings]:
    """Settings pointed at a per-test temp database and object store."""
    from app.core.logging import configure_logging

    configure_logging("INFO")
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        object_store_dir=str(tmp_path / "objects"),
        embedding_provider="fake",
        embedding_model="fake-embed-v1",
        embedding_dim=64,
    )
    yield settings
    from app.db.session import reset_engine

    reset_engine()


@pytest.fixture
def engine(settings_env: Settings) -> Iterator[Engine]:
    """A fresh engine with the schema created.

    Also registers the session factory against this engine. Anything reaching for
    `get_session_factory()` with no argument -- `tests/api/test_conversations.py`
    does -- reuses whatever is cached here, so it lands in the temp database rather
    than the configured one. The fixture that establishes the test database has to
    establish the session factory too, or the two disagree and the test writes
    somewhere it did not choose.
    """
    from app.db.session import create_all, get_engine, get_session_factory

    eng = get_engine(settings_env)
    create_all(eng)
    get_session_factory(settings_env)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session bound to the test engine."""
    from app.db.session import get_session_factory

    factory = get_session_factory()
    with factory() as s:
        yield s


@pytest.fixture
def object_store(settings_env: Settings):
    from app.ingest.objectstore import LocalObjectStore

    return LocalObjectStore.from_settings(settings_env)


@pytest.fixture
def provider(settings_env: Settings):
    from app.providers.embedding import get_embedding_provider

    return get_embedding_provider(settings_env)


@pytest.fixture
def client(engine: Engine, settings_env: Settings) -> Iterator[TestClient]:
    """TestClient with the app's settings wired to the per-test environment.

    Overriding the `get_settings` dependency (not the module-level singleton) is
    what makes the query path use the same embedding dimension the test database
    was written with. Without it, the chat endpoint would embed a query at the
    production default dimension against 64-wide stored vectors and fail, while
    the ingestion tests pass because they inject their provider directly.
    """
    from app.core.config import get_settings
    from app.main import create_app

    app = create_app()
    app.state.settings = settings_env
    app.dependency_overrides[get_settings] = lambda: settings_env
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def live_doc(session, object_store, provider, settings_env, sample_md):
    """A document ingested to `live`. Most index tests need a populated one."""
    from app.ingest.worker import create_document, ingest_document

    doc = create_document(
        session, filename="policy.md", data=sample_md, object_store=object_store
    )
    session.commit()
    ingest_document(
        session, doc.doc_id, object_store=object_store, provider=provider, settings=settings_env
    )
    session.commit()
    return doc


@pytest.fixture
def long_md() -> bytes:
    """A document long enough to produce several chunks.

    Retrieval tests need more than one chunk to say anything about ranking: with a
    single chunk every query trivially retrieves it, so recall is 1.0 and no
    ordering or discrimination claim can be tested.
    """
    sections = []
    topics = [
        ("Refunds", "Customers may request a refund within 30 days of purchase."),
        ("Shipping", "Standard shipping arrives within 5 business days."),
        ("Warranty", "Products are covered for 24 months from delivery."),
        ("Privacy", "Personal data is retained for 90 days after closure."),
        ("Billing", "Invoices are due within 14 days of issue."),
    ]
    for heading, sentence in topics:
        body = [sentence, sentence, sentence]
        sections.append(f"## {heading}\n\n" + "\n\n".join(body))
    # The heading strategy groups by heading and accumulates toward a
    # ~1000-token target, so this needs to clear 1000 tokens comfortably rather
    # than sit just above the boundary.
    filler = "\n\n".join(
        f"Additional context paragraph {n} for section {n % len(topics) + 1} "
        f"covering operational detail, exceptions, and escalation paths."
        for n in range(120)
    )
    return ("# Long Policy Document\n\n" + "\n\n".join(sections) + "\n\n" + filler).encode()


@pytest.fixture
def multi_chunk_doc(session, object_store, provider, settings_env, long_md):
    """A live document guaranteed to span several chunks."""
    from app.ingest.worker import create_document, ingest_document

    doc = create_document(
        session, filename="long.md", data=long_md, object_store=object_store
    )
    session.commit()
    ingest_document(
        session, doc.doc_id, object_store=object_store, provider=provider, settings=settings_env
    )
    session.commit()
    assert len(doc.chunks) >= 2, "fixture must span multiple chunks for retrieval tests"
    return doc


@pytest.fixture
def second_live_doc(session, object_store, provider, settings_env, sample_md):
    """A second, distinct live document, for tests that need two."""
    from app.ingest.worker import create_document, ingest_document

    other = sample_md + b"\n\n## Escalation\n\nEscalate to a supervisor after 24 hours.\n"
    doc = create_document(
        session, filename="other.md", data=other, object_store=object_store
    )
    session.commit()
    ingest_document(
        session, doc.doc_id, object_store=object_store, provider=provider, settings=settings_env
    )
    session.commit()
    return doc


@pytest.fixture
def sample_md() -> bytes:
    """A small markdown document with headings and enough text to chunk."""
    return b"""# Refund Policy

## Digital Products

Customers who purchased a digital product may request a refund within 30 days
of the initial purchase date. The product must not have been downloaded more
than once. Refund requests are processed within 5 business days of approval.

Products marked as non-refundable at the point of sale are excluded from this
policy. Gift cards cannot be refunded except where required by law.

## Physical Goods

Physical goods may be returned within 14 days of delivery in their original
packaging. The customer pays return shipping unless the item was faulty. Faulty
items are refunded in full including original delivery cost.

## Chargebacks

If a customer disputes a charge with their bank, we respond to the dispute within
10 business days. Chargebacks are more expensive than refunds for us, so we
always attempt to resolve directly with the customer first.

# Shipping

Standard shipping takes 3 to 5 business days. Express shipping is available for
an additional fee and arrives within 2 business days. We ship to most countries
but not to all territories.
"""
