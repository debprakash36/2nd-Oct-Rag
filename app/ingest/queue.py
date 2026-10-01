"""Ingestion job dispatch (FR-5, G4).

The queue is behind an interface rather than a library, because the two properties
that matter are properties of the *protocol*, not of any one broker:

* Delivery is at least once. The worker is required to converge on redelivery
  (see `ingest_document`), so a duplicate delivery costs a no-op rather than
  correctness.
* Dispatch must not be able to make an upload vanish. If the broker is
  unavailable, an upload is stored and marked `failed` with the reason. Silently
  dropping the message would leave a `pending` document that never becomes
  queryable, which looks identical to a document the user forgot to upload.

`InlineQueue` runs the job in the request. That is the local/test default, and it
is a real implementation of the contract, not a stub: it enforces the same
exception and return semantics a broker-backed queue must.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

log = get_logger("app.ingest.queue")


@dataclass(frozen=True)
class IngestJob:
    """A unit of queued work.

    Only the document id is carried. The job does not hold the file bytes or the
    chunks: the document row is the record, and the worker reads from the object
    store. A payload that can go stale relative to the database is a second
    source of truth, and reconciling two is harder than looking one up.
    """

    doc_id: str
    attempt: int = 1

    #: Used by brokers for dedup. `ingest_document` already converges, so this is
    #: an optimisation that saves a redundant extraction, not a correctness
    #: mechanism — correctness must not depend on the broker honouring it.
    def dedup_key(self) -> str:
        return f"ingest:{self.doc_id}"


@dataclass
class DispatchResult:
    """Outcome of handing a job to the queue."""

    doc_id: str
    queued: bool
    detail: str = ""
    error: str | None = None


class IngestQueue(ABC):
    """Interface the broker-backed implementation must satisfy."""

    @abstractmethod
    def enqueue(self, job: IngestJob) -> DispatchResult:
        """Hand a job to the queue. Must not raise for broker unavailability."""

    @abstractmethod
    def close(self) -> None:
        """Release broker connections."""


class InlineQueue(IngestQueue):
    """Runs the job immediately, in-process.

    Used by tests and local single-process runs. The handler is invoked
    synchronously and exceptions propagate to the caller, which is the same
    information a broker-backed worker gets from its own error handling — the
    point being that callers must handle dispatch failure either way, so testing
    against the inline queue exercises the real contract.
    """

    def __init__(self, handler: Callable[[IngestJob], Any]) -> None:
        self._handler = handler
        self._jobs: list[IngestJob] = []

    @property
    def dispatched(self) -> list[IngestJob]:
        """Jobs handed to this queue. Tests assert on this."""
        return list(self._jobs)

    def enqueue(self, job: IngestJob) -> DispatchResult:
        self._jobs.append(job)
        try:
            self._handler(job)
        except Exception as exc:
            # Reported, not raised. An upload API that failed the request because
            # a background job could not start would tell the user nothing about
            # whether their file was stored; the document row is the record.
            log.error(
                "inline ingest failed",
                extra={"doc_id": job.doc_id, "reason": f"{type(exc).__name__}: {exc}"},
            )
            return DispatchResult(
                doc_id=job.doc_id,
                queued=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return DispatchResult(doc_id=job.doc_id, queued=True, detail="processed inline")

    def close(self) -> None:
        return None


@dataclass
class QueueConfig:
    """Broker connection settings.

    `required` is the difference between a queue and a database: a broker outage
    must not stop the service from serving queries, but it must stop the service
    from accepting uploads it cannot process. Ingestion is degraded, not broken.
    """

    broker_url: str = ""
    enabled: bool = False
    #: Reject uploads instead of accepting them into an undeliverable queue.
    required: bool = True
    #: Deliveries attempted before a job is considered failed.
    max_attempts: int = 3
    visibility_timeout_seconds: int = 300
    extras: dict[str, str] = field(default_factory=dict)


class UnavailableQueue(IngestQueue):
    """Stand-in used when a broker is configured but unreachable.

    Refuses uploads rather than accepting work that cannot be delivered. This is
    the `required=True` behaviour; it is a distinct class rather than a branch so
    the degraded path is directly testable.
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def enqueue(self, job: IngestJob) -> DispatchResult:
        return DispatchResult(
            doc_id=job.doc_id,
            queued=False,
            error=f"ingestion queue unavailable: {self._reason}",
        )

    def close(self) -> None:
        return None