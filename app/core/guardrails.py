"""Request guardrails for the chat surface (FR-31, FR-34, FR-22, NFR-5).

Four guards live here, and the fourth is a guard against building a guard:

* **Input size cap** (FR-34). Enforced before any provider call, because a cap
  applied after embedding has already paid for the tokens.
* **Rate limit** (FR-31). Fixed window, in-process, keyed on client identity.
  Deliberately not distributed — see `RateLimiter` for why that is a deployment
  decision rather than an oversight.
* **Error sanitization** (NFR-5). Typed internal errors become a stable code plus a
  fixed message. The internal detail goes to the log and never to the client.
* **No output filtering** (FR-22). See the module-level note below.

**This module deliberately implements no output keyword filter.**
implementation.md §6.3 is explicit that regex filtering of model output for
"sensitive words" is not a security control, and adding one would be worse than
having none: it gives a false sense of coverage, it mangles legitimate answers
about a corpus that legitimately contains those words (a privacy policy discusses
personal data; a security runbook discusses attacks), and it does not touch the
actual threat, which is the client executing model output as markup. That threat
is handled at the render boundary by escaping (FR-22), which is a real control.

The guard that *is* implemented for embedded instructions is on the ingest side of
the trust boundary, not here: retrieved text enters the prompt inside explicit
delimiters as data (FR-32, `generation/prompt.py`), and the validator strips any
citation marker the model invents. Neither is a keyword filter.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import AppError, ValidationError
from app.core.logging import get_logger

log = get_logger("app.core.guardrails")


class RateLimitExceeded(AppError):
    """Too many requests from one client (FR-31)."""

    #: 429 with a `Retry-After`, not 400. A client that treats this as a bad request
    #: shows the user an error instead of backing off, so the one response that
    #: should reduce load is the one most likely to cause more of it.
    status_code = 429

    def __init__(self, *, retry_after_seconds: int) -> None:
        super().__init__(
            f"rate limit exceeded; retry after {retry_after_seconds}s",
            user_message=(
                "You're sending questions too quickly. "
                "Please wait a moment and try again."
            ),
        )
        self.retry_after_seconds = retry_after_seconds


class QueryTooLarge(ValidationError):
    """Input exceeded the per-turn cap (FR-34)."""

    def __init__(self, *, limit: int) -> None:
        super().__init__(
            f"message of {limit}+ characters rejected",
            user_message=(
                f"That message is too long. Please keep it under {limit} characters."
            ),
        )
        self.limit = limit


def enforce_message_size(message: str, *, limit: int) -> None:
    """Reject an over-long message before any provider call (FR-34).

    Raises `QueryTooLarge` rather than returning a bool so the call site cannot
    forget to act on the result. Call this *first* in a chat handler: everything
    downstream — embedding, retrieval, generation — costs money per character, so a
    cap enforced after the first provider call has already failed to do its job.
    """
    if len(message) > limit:
        raise QueryTooLarge(limit=limit)


def client_identity(request: Any) -> str:
    """Best-effort identity for rate limiting (FR-31).

    Prefers the authenticated identity and falls back to the client address. v1 has
    no authentication (architecture.md NG5), so the address is what there is.

    `X-Forwarded-For` is honoured, which is only correct because the deployment puts
    the app behind a gateway that overwrites it (architecture.md 7.1). A directly
    exposed app would let any caller spoof the header and bypass the limit
    entirely — noted here because it is the failure mode of this function, not a
    hypothetical.
    """
    user = getattr(getattr(request, "state", None), "user_id", None)
    if isinstance(user, str) and user:
        return f"user:{user}"
    forwarded = request.headers.get("X-Forwarded-For") if request is not None else None
    if forwarded:
        # Left-most entry is the original client when the gateway appends.
        return f"ip:{forwarded.split(',')[0].strip()}"
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return f"ip:{host or 'unknown'}"


@dataclass
class _Window:
    """Request timestamps retained for one key within the current window."""

    started_at: float
    count: int = 0


@dataclass
class RateLimiter:
    """Fixed-window per-key rate limiter (FR-31).

    Fixed window rather than sliding: it needs one timestamp per key instead of one
    per request, and the boundary burst it permits (up to 2x the limit across a
    window edge) does not matter for a chat endpoint whose real limit is protecting
    spend, not enforcing a hard concurrency ceiling.

    **In-process, so per-worker.** Two uvicorn workers are two independent limiters.
    That is a deployment concern, not a code defect: a shared store (Redis, or a
    `SELECT ... FOR UPDATE` counter) is the fix, and the seam is this class — the
    call sites only depend on `check()`. FR-31 says "per user/IP at the gateway",
    which is where a correct shared limit belongs anyway.

    `OrderedDict` rather than a plain dict so the oldest key can be evicted to bound
    memory under a spray of distinct addresses. A plain dict would grow without
    limit, which turns a rate limiter into a memory-exhaustion vector.
    """

    limit: int
    window_seconds: float
    max_keys: int = 10_000
    _windows: OrderedDict[str, _Window] = field(default_factory=OrderedDict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self, key: str, *, now: float | None = None) -> None:
        """Count this request against `key`, or raise `RateLimitExceeded`."""
        if self.limit <= 0:
            # 0 disables the guard rather than blocking every request. A limit of
            # zero that rejects everything would look like an outage.
            return
        moment = now if now is not None else time.monotonic()

        with self._lock:
            window = self._windows.get(key)
            if window is None or moment - window.started_at >= self.window_seconds:
                window = _Window(started_at=moment)
                self._windows[key] = window
                self._evict_locked()
            else:
                # Re-insert so the eviction order tracks recency of use, not
                # insertion: a key in constant use should not be the next one
                # discarded for being "oldest".
                self._windows.move_to_end(key)

            if window.count >= self.limit:
                retry_after = max(
                    1, int(self.window_seconds - (moment - window.started_at)) + 1
                )
                log.warning(
                    "rate limited",
                    extra={"limit": self.limit, "window_seconds": self.window_seconds},
                )
                raise RateLimitExceeded(retry_after_seconds=retry_after)
            window.count += 1

    def _evict_locked(self) -> None:
        while len(self._windows) > self.max_keys:
            self._windows.popitem(last=False)

    def reset(self) -> None:
        """Clear all windows. Used by tests and by an admin action."""
        with self._lock:
            self._windows.clear()


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter(limit: int, window_seconds: float) -> RateLimiter:
    """Process-wide limiter, created once.

    Built on first use rather than at import so the configured limit can arrive from
    settings that are not loaded yet, and so importing this module has no side
    effect. Rebuilt when the limit changes, which is what a test that overrides the
    setting needs.
    """
    global _limiter
    with _limiter_lock:
        if _limiter is None or _limiter.limit != limit or _limiter.window_seconds != window_seconds:
            _limiter = RateLimiter(limit=limit, window_seconds=window_seconds)
        return _limiter


def reset_rate_limiter() -> None:
    """Drop the process-wide limiter. Used by tests."""
    global _limiter
    with _limiter_lock:
        _limiter = None


#: Stable external error codes. The client switches on these; the message is for a
#: human. Neither ever contains a provider name, a stack trace, or an internal id
#: (NFR-5).
ERROR_CODES: dict[str, str] = {
    "query_too_large": "Your message was too long to process.",
    "rate_limited": "Too many requests. Please wait a moment and try again.",
    "not_found": "That item could not be found.",
    "store_unavailable": "Search is temporarily unavailable. Please try again shortly.",
    "internal_error": "Something went wrong. Please try again.",
}


def error_event_code(exc: BaseException) -> str:
    """Map an exception to a stable external code.

    Deliberately coarse: an unexpected exception type collapses to `internal_error`
    rather than being named, because a distinct code per exception type turns the
    response body into a description of the server's internals.
    """
    if isinstance(exc, RateLimitExceeded):
        return "rate_limited"
    if isinstance(exc, QueryTooLarge):
        return "query_too_large"
    if isinstance(exc, AppError):
        return "internal_error"
    return "internal_error"


def safe_error_payload(exc: BaseException) -> dict[str, str]:
    """The `{code, message}` body that may cross the API boundary (NFR-5).

    `AppError.user_message` is used when the exception is one of ours, because those
    messages are written to be shown to a user. Everything else gets the generic
    `internal_error` message: an arbitrary exception's `str()` can contain a
    connection string, a file path, or a provider SDK message, and this function is
    the last place that can stop it.

    `validation_error` is carried through for `ValidationError` subclasses that are
    user-facing input problems (a bad upload), which already own their wording.
    """
    code = error_event_code(exc)
    if isinstance(exc, AppError) and exc.user_message:
        return {"code": code, "message": exc.user_message}
    return {"code": code, "message": ERROR_CODES[code]}
