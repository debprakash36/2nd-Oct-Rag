"""Typed errors with separate internal and user-safe messages.

NFR-5: internal detail — provider errors, stack traces, file paths, chunk IDs —
must never cross the API boundary. Every error carries both, and only the
`user_message` is ever serialized to a client.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class. `user_message` is safe to return to a client.

    `status_code` is part of the error rather than something the exception handler
    infers from the type, so an error raised deep in the store reaches the client as
    the right status without the handler needing to know about every subclass. It
    defaults to 400 because most of these are input problems; 404 and 429 are set
    explicitly by the subclasses that mean them.
    """

    status_code: int = 400

    def __init__(
        self,
        detail: str,
        *,
        user_message: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.user_message = user_message or "Something went wrong. Please try again."
        if status_code is not None:
            self.status_code = status_code

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.detail


class ValidationError(AppError):
    """Input rejected before any work began (FR-2, FR-34)."""

    def __init__(self, detail: str, *, user_message: str) -> None:
        super().__init__(detail, user_message=user_message)


class ExtractionError(AppError):
    """Text extraction failed or produced nothing usable (FR-3)."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail, user_message="This file could not be read. It may be "
                                             "corrupt, password-protected, or scanned.")


class SandboxError(ExtractionError):
    """Extraction exceeded a resource limit or violated a sandbox rule (NFR-5)."""


class ChunkingError(AppError):
    """Chunking produced no chunks (FR-9)."""


class EmbeddingError(AppError):
    """The embedding provider failed or returned the wrong shape."""


class GenerationError(AppError):
    """The generation provider failed, or could not be reached.

    Distinct from `EmbeddingError` because the two are surfaced differently: an
    embedding failure is almost always an ingestion-time or configuration fault,
    while a generation failure lands on a live request where the answer has already
    started streaming. A single error type would force the query path to report a
    misconfiguration as though it were a transient outage.

    503 rather than the 400 default: the request was well formed and the failure is
    an unreachable third party, which is exactly what 503 is for. A client that
    retries on 503 and does not retry on 400 gets the right behaviour for free.

    The `detail` carries the model name and HTTP status for the operator; the
    `user_message` stays generic, so nothing about the provider crosses the API
    boundary (NFR-5).
    """

    status_code: int = 503


class StateTransitionError(AppError):
    """An illegal document state transition was attempted.

    Raised rather than logged and ignored: a silent no-op here would leave a
    document stuck in a transient state with no way to tell whether the work
    happened.
    """

    def __init__(self, doc_id: str, source: str, target: str) -> None:
        super().__init__(
            f"illegal transition for {doc_id}: {source} -> {target}",
            user_message="This document is not in a state that allows that operation.",
        )
        self.doc_id = doc_id
        self.source = source
        self.target = target


class NotFoundError(AppError):
    """Requested resource does not exist."""

    #: 404, not the 400 the base class defaults to. A client polling a conversation
    #: it just deleted has to be able to distinguish "gone" from "your request was
    #: malformed", or it retries forever against a resource that will never return.
    status_code = 404

    def __init__(self, what: str) -> None:
        super().__init__(f"{what} not found", user_message=f"{what} not found.")


class StoreUnavailableError(AppError):
    """A retrieval dependency is unreachable (NFR-2, architecture.md §8).

    The distinction this type exists to protect is the one that matters most in this
    system: a *store being down* must not look like *the corpus having nothing to
    say*. If the vector or keyword index is unreachable, retrieval cannot honestly
    report "no relevant passages found" — that is a refusal, and a refusal is a claim
    about the corpus. Answering from whichever half of the index still responded, or
    abstaining as though nothing matched, produces a confidently wrong result from a
    partially-available system. A degraded-but-plausible answer is the failure mode
    this project exists to prevent.

    So this is a 503 with a message that says the search is unavailable, distinct from
    a 200 refusal that says the corpus has no answer. The user must be able to tell
    "come back later" from "we do not have that".
    """

    status_code = 503

    def __init__(self, component: str) -> None:
        super().__init__(
            f"{component} unavailable",
            user_message=(
                "Search is temporarily unavailable, so this question cannot be "
                "answered reliably right now. Please try again shortly."
            ),
        )
        self.component = component


class DuplicateDocumentError(AppError):
    """Content matches an existing document (FR-4)."""

    def __init__(self, existing_doc_id: str, filename: str) -> None:
        super().__init__(
            f"duplicate of {existing_doc_id} ({filename})",
            user_message=f"This content already exists in the corpus as '{filename}'.",
        )
        self.existing_doc_id = existing_doc_id
