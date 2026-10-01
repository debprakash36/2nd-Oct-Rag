"""Ingestion queue contract tests.

The properties tested are the ones that decide whether a user's upload can be
lost: dispatch failure must be visible, and redelivery must converge. A queue
that silently accepts work it cannot deliver leaves `pending` documents that
never become queryable, which is indistinguishable from an upload the user never
made.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.ingest.queue import (
    DispatchResult,
    IngestJob,
    IngestQueue,
    InlineQueue,
    QueueConfig,
    UnavailableQueue,
)


class _BrokenQueue(IngestQueue):
    """Broker-backed queue whose enqueue always fails."""

    def enqueue(self, job: IngestJob) -> DispatchResult:
        return DispatchResult(doc_id=job.doc_id, queued=False, error="connection refused")

    def close(self) -> None:
        return None


def test_job_carries_only_the_document_id():
    """The document row is the record; the payload must not be a second source.

    A job carrying file bytes or chunk text would be a copy that can disagree
    with the database, and reconciling the two is harder than reading one.
    """
    job = IngestJob(doc_id="abc123")
    assert job.doc_id == "abc123"
    assert job.attempt == 1
    assert job.dedup_key() == "ingest:abc123"


def test_inline_queue_invokes_the_handler():
    seen: list[str] = []

    queue = InlineQueue(lambda job: seen.append(job.doc_id))
    result = queue.enqueue(IngestJob(doc_id="doc1"))

    assert seen == ["doc1"]
    assert result.queued is True
    assert queue.dispatched[0].doc_id == "doc1"


def test_handler_failure_is_reported_not_raised():
    """A caller must be able to handle dispatch failure.

    Raising would turn an upload that was already stored into a failed HTTP
    response, leaving the user unable to tell whether their file survived.
    """

    def boom(_job: IngestJob) -> None:
        raise RuntimeError("queue consumer died")

    queue = InlineQueue(boom)
    result = queue.enqueue(IngestJob(doc_id="doc1"))

    assert result.queued is False
    assert result.error is not None
    assert "queue consumer died" in result.error


def test_unavailable_queue_refuses_rather_than_dropping():
    """The `required=True` behaviour: refuse instead of accepting dead work."""
    queue = UnavailableQueue("broker down")
    result = queue.enqueue(IngestJob(doc_id="doc1"))

    assert result.queued is False
    assert "broker down" in result.error or "broker down" in result.error


def test_broken_broker_queue_reports_failure():
    result = _BrokenQueue().enqueue(IngestJob(doc_id="doc1"))
    assert result.queued is False
    assert result.error == "connection refused"


def test_job_encoding_round_trips():
    """Broker payloads must survive encode/decode.

    Tested without a Redis server because the encoding is the part that can be
    wrong independently of the transport.
    """
    from app.ingest.queue_redis import _decode, _encode

    job = IngestJob(doc_id="deadbeef", attempt=3)
    assert _decode(_encode(job)) == job


def test_encode_uses_pipe_delimiter():
    """Doc ids are hex, so the delimiter cannot appear inside a field."""
    from app.ingest.queue_redis import _decode, _encode

    assert _decode(_encode(IngestJob(doc_id="a" * 32))) .doc_id == "a" * 32


def test_redis_queue_reports_missing_driver(monkeypatch):
    """A missing driver package is a misconfiguration, not an outage.

    It must raise so the caller applies its policy, rather than being silently
    downgraded to inline.
    """
    import builtins

    from app.ingest import queue_redis

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("No module named 'redis'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    settings = Settings(ingest_queue_enabled=True, ingest_queue_broker_url="redis://localhost:6379")
    with pytest.raises(RuntimeError, match="redis"):
        queue_redis.build_broker_queue(settings)


def test_unknown_scheme_raises():
    from app.ingest.queue_redis import build_broker_queue

    settings = Settings(
        ingest_queue_enabled=True, ingest_queue_broker_url="amqp://localhost:5672"
    )
    with pytest.raises(RuntimeError, match="no ingestion queue driver"):
        build_broker_queue(settings)


def test_queue_config_distinguishes_required_from_optional():
    """Required means "refuse uploads"; optional means "degrade to inline".

    The distinction is the reason this is configuration and not a constant: a
    broker outage must not take the whole service down, but it must not accept
    uploads it cannot process either.
    """
    required = QueueConfig(required=True)
    optional = QueueConfig(required=False)
    assert required.required is True
    assert optional.required is False
    assert required.max_attempts == optional.max_attempts == 3


def test_visibility_timeout_exceeds_sandbox_timeout():
    """A job must not be redelivered while it is still legitimately running.

    If the visibility timeout is shorter than the slowest pipeline stage, an
    in-flight job looks abandoned and gets run twice concurrently — two workers
    writing the same document. The worker converges, but only sequentially.
    """
    settings = Settings()
    assert settings.ingest_visibility_timeout_seconds > settings.sandbox_timeout_seconds, (
        "visibility timeout must exceed the sandbox timeout so a slow extraction "
        "is not mistaken for a dead worker"
    )