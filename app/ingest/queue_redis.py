"""Broker-backed ingestion queue.

Kept separate from `app/ingest/queue.py` so the interface and the at-least-once
contract stay readable without a broker import, and so the broker driver can be
swapped without touching either the API or the worker.

`build_broker_queue` resolves the configured driver. It raises on connection
failure rather than returning a degraded queue, because the caller's decision
between refusing uploads and degrading to inline depends on the *reason* — a
missing driver package is a misconfiguration and should not be silently
downgraded, while an unreachable broker is the outage case that policy covers.

Redis is the reference driver: a `BLPOP`-based queue gives at-least-once delivery
with no broker-side dedup, which is safe here because the worker converges on
redelivery (see `app/ingest/worker.py`).
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.ingest.queue import DispatchResult, IngestJob, IngestQueue

log = get_logger("app.ingest.queue.redis")

QUEUE_KEY = "rag:ingest:jobs"
RESULT_TTL_SECONDS = 3600


def build_broker_queue(settings: Settings) -> IngestQueue:
    """Return the configured broker queue, or raise.

    Raised errors are the caller's cue to apply policy: refuse uploads
    (`ingest_queue_required`) or degrade to inline.
    """
    url = settings.ingest_queue_broker_url
    if not url:
        raise RuntimeError("ingest_queue_broker_url is empty but a broker queue was requested")

    if url.startswith(("redis://", "rediss://")):
        return RedisQueue(settings)
    raise RuntimeError(f"no ingestion queue driver registered for {url.split('://', 1)[0]}://")


class RedisQueue(IngestQueue):
    """At-least-once queue backed by a Redis list.

    A list is used rather than a stream or pub/sub for one reason: pub/sub drops
    messages when no consumer is connected, which would lose uploads silently.
    `BLPOP` leaves the message in the list until a worker takes it, so a worker
    outage delays ingestion rather than discarding it.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "the redis package is required for a redis:// ingestion queue"
            ) from exc

        self._settings = settings
        self._client: Any = redis.Redis.from_url(
            settings.ingest_queue_broker_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            # A broker that cannot be reached must fail fast so the API can apply
            # its degraded/required policy, rather than hanging the upload.
            retry_on_timeout=False,
        )
        # Fail at construction, not on first upload: a connection check here means
        # the outage surfaces at startup instead of as a per-file error later.
        self._client.ping()

    def enqueue(self, job: IngestJob) -> DispatchResult:
        try:
            self._client.lpush(QUEUE_KEY, _encode(job))
        except Exception as exc:
            return DispatchResult(
                doc_id=job.doc_id, queued=False, error=f"{type(exc).__name__}: {exc}"
            )
        return DispatchResult(doc_id=job.doc_id, queued=True, detail="queued")

    def dequeue(self, timeout_seconds: int = 5) -> IngestJob | None:
        """Pop the next job, waiting up to `timeout_seconds`. Test/worker helper."""
        item = self._client.blpop(QUEUE_KEY, timeout=timeout_seconds)
        if item is None:
            return None
        return _decode(item[1])

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            log.warning("failed to close redis client cleanly")


def _encode(job: IngestJob) -> str:
    # Pipe-delimited rather than JSON: the two fields are scalars with no
    # escaping concerns, and a broker-side `LRANGE` during an incident then
    # reads without a decoder.
    return f"{job.doc_id}|{job.attempt}"


def _decode(raw: str) -> IngestJob:
    doc_id, _, attempt = raw.partition("|")
    return IngestJob(doc_id=doc_id, attempt=int(attempt) if attempt else 1)