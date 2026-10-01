"""Guardrail tests (FR-31, FR-34, FR-22, NFR-5).

The rate limiter is tested directly rather than through HTTP because the interesting
cases are time-dependent — window rollover, eviction — and driving those through a
request would mean sleeping.
"""

from __future__ import annotations

import pytest

from app.core.errors import AppError
from app.core.guardrails import (
    ERROR_CODES,
    QueryTooLarge,
    RateLimiter,
    RateLimitExceeded,
    client_identity,
    enforce_message_size,
    error_event_code,
    get_rate_limiter,
    reset_rate_limiter,
    safe_error_payload,
)


class _FakeClient:
    host = "203.0.113.9"


class _FakeState:
    """Stand-in for `request.state`. `user_id` is set only in the auth test."""


class _FakeRequest:
    def __init__(self, headers=None, client=None):
        self.headers = headers or {}
        self.client = client if client is not None else _FakeClient()
        self.state = _FakeState()


class TestSizeCap:
    def test_accepts_at_limit(self):
        enforce_message_size("a" * 100, limit=100)  # must not raise

    def test_rejects_over_limit(self):
        with pytest.raises(QueryTooLarge):
            enforce_message_size("a" * 101, limit=100)

    def test_error_is_typed_and_user_safe(self):
        with pytest.raises(QueryTooLarge) as exc:
            enforce_message_size("a" * 5000, limit=100)
        err = exc.value
        assert err.status_code == 400
        # The internal detail names the limit, not the payload. Logging the caller's
        # full message on a rejection would put their content in the log for a
        # request that was refused precisely to avoid processing it.
        assert "5000" not in err.detail
        assert "100" in err.detail
        # The user message names the limit they can act on, and nothing else.
        assert "100" in err.user_message
        assert "5000" not in err.user_message

    def test_empty_message_rejected_by_schema_not_here(self):
        """Emptiness is a pydantic concern; the cap only bounds length."""
        enforce_message_size("", limit=10)


class TestRateLimiter:
    def test_allows_up_to_limit(self):
        limiter = RateLimiter(limit=3, window_seconds=60)
        for _ in range(3):
            limiter.check("k", now=0.0)

    def test_blocks_beyond_limit(self):
        limiter = RateLimiter(limit=2, window_seconds=60)
        limiter.check("k", now=0.0)
        limiter.check("k", now=0.0)
        with pytest.raises(RateLimitExceeded) as exc:
            limiter.check("k", now=0.0)
        assert exc.value.status_code == 429
        assert exc.value.retry_after_seconds > 0

    def test_keys_are_independent(self):
        limiter = RateLimiter(limit=1, window_seconds=60)
        limiter.check("a", now=0.0)
        limiter.check("b", now=0.0)
        with pytest.raises(RateLimitExceeded):
            limiter.check("a", now=0.0)

    def test_window_rolls_over(self):
        limiter = RateLimiter(limit=1, window_seconds=10)
        limiter.check("k", now=0.0)
        with pytest.raises(RateLimitExceeded):
            limiter.check("k", now=5.0)
        limiter.check("k", now=11.0)

    def test_zero_limit_disables(self):
        """A limit of 0 must degrade to 'no limit', not block everything."""
        limiter = RateLimiter(limit=0, window_seconds=60)
        for _ in range(100):
            limiter.check("k", now=0.0)

    def test_eviction_bounds_memory(self):
        """A plain dict would grow forever under a spray of distinct addresses."""
        limiter = RateLimiter(limit=5, window_seconds=60, max_keys=10)
        for i in range(100):
            limiter.check(f"key-{i}", now=0.0)
        assert len(limiter._windows) == 10

    def test_eviction_keeps_recently_used_keys(self):
        """Eviction is LRU on *use*, not on insertion.

        `move_to_end` is what makes this work: `hot` is re-touched on every call, so
        it stays at the young end and the cold keys are the ones discarded. Without
        the re-insert, a long-lived client that checked in early would be the first
        key thrown away for a single request having arrived later.
        """
        limiter = RateLimiter(limit=5, window_seconds=60, max_keys=3)
        limiter.check("hot", now=0.0)
        for i in range(4):
            limiter.check(f"filler-{i}", now=0.0)
            # Re-touching `hot` between fillers is what makes it recent.
            limiter.check("hot", now=0.0)
        assert "hot" in limiter._windows

    def test_eviction_discards_least_recently_used(self):
        limiter = RateLimiter(limit=5, window_seconds=60, max_keys=3)
        for i in range(4):
            limiter.check(f"old-{i}", now=0.0)
        for i in range(4):
            limiter.check(f"new-{i}", now=0.0)
        assert "old-0" not in limiter._windows

    def test_reset_clears(self):
        limiter = RateLimiter(limit=1, window_seconds=60)
        limiter.check("k", now=0.0)
        limiter.reset()
        limiter.check("k", now=0.0)

    def test_retry_after_shrinks_as_window_elapses(self):
        limiter = RateLimiter(limit=1, window_seconds=60)
        limiter.check("k", now=0.0)
        with pytest.raises(RateLimitExceeded) as early:
            limiter.check("k", now=1.0)
        with pytest.raises(RateLimitExceeded) as late:
            limiter.check("k", now=55.0)
        assert late.value.retry_after_seconds < early.value.retry_after_seconds


class TestProcessLimiter:
    def setup_method(self):
        reset_rate_limiter()

    def teardown_method(self):
        reset_rate_limiter()

    def test_same_settings_return_same_instance(self):
        a = get_rate_limiter(10, 60.0)
        b = get_rate_limiter(10, 60.0)
        assert a is b

    def test_changed_settings_rebuild(self):
        """A test overriding the setting must not inherit the previous limiter."""
        assert get_rate_limiter(10, 60.0) is not get_rate_limiter(20, 60.0)


class TestClientIdentity:
    def test_uses_forwarded_ip(self):
        req = _FakeRequest(headers={"X-Forwarded-For": "198.51.100.7, 10.0.0.1"})
        assert client_identity(req) == "ip:198.51.100.7"

    def test_falls_back_to_peer_address(self):
        assert client_identity(_FakeRequest()) == "ip:203.0.113.9"

    def test_missing_client_is_stable(self):
        """Must not raise and must not collapse everything into one bucket."""
        req = _FakeRequest()
        req.client = None
        assert client_identity(req) == "ip:unknown"

    def test_prefers_authenticated_user(self):
        req = _FakeRequest()
        req.state.user_id = "user-7"
        assert client_identity(req) == "user:user-7"


class TestErrorSanitization:
    def test_rate_limit_code_and_status(self):
        exc = RateLimitExceeded(retry_after_seconds=30)
        assert error_event_code(exc) == "rate_limited"
        assert exc.status_code == 429

    def test_oversize_code(self):
        assert error_event_code(QueryTooLarge(limit=10)) == "query_too_large"

    def test_arbitrary_exception_collapses_to_internal(self):
        """A code per exception type would describe the server's internals."""
        exc = RuntimeError("postgres://user:hunter2@db:5432 refused")
        assert error_event_code(exc) == "internal_error"
        payload = safe_error_payload(exc)
        assert payload["code"] == "internal_error"
        assert "hunter2" not in payload["message"]
        assert payload["message"] == ERROR_CODES["internal_error"]

    def test_payload_is_never_empty(self):
        for exc in (
            RuntimeError("boom"),
            ValueError("bad"),
            KeyError("missing"),
            RateLimitExceeded(retry_after_seconds=5),
        ):
            payload = safe_error_payload(exc)
            assert set(payload) == {"code", "message"}
            assert payload["message"]

    def test_our_user_message_is_used(self):
        exc = QueryTooLarge(limit=10)
        assert "10" in safe_error_payload(exc)["message"]

    def test_every_code_has_a_message(self):
        """A code with no message would reach the client as an empty string."""
        assert set(ERROR_CODES) == {
            "query_too_large",
            "rate_limited",
            "not_found",
            "store_unavailable",
            "internal_error",
        }
        assert all(ERROR_CODES.values())

    def test_unknown_app_error_is_still_safe(self):
        class Weird(AppError):
            pass

        exc = Weird("sqlite:///private/secret.db failed")
        payload = safe_error_payload(exc)
        assert "secret.db" not in payload["message"]