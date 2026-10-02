"""Shared-secret gate for every route except health and login.

Empty `API_TOKEN` leaves the API open so the test suite and a fresh checkout
keep working. A configured token is required as `Authorization: Bearer …`.
The comparison is constant-time, and a mismatch is a 401 with a user-safe
message — the token itself is never logged.
"""

from __future__ import annotations

import hmac

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import Settings, get_settings

_OPEN_PATHS = frozenset({"/health", "/auth/status", "/auth/login"})


def token_matches(presented: str, expected: str) -> bool:
    """Constant-time equality. Different lengths are a mismatch, not an error."""
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def settings_for(scope: Scope) -> Settings:
    """Settings bound to this app, falling back to the process singleton."""
    app = scope.get("app")
    bound = getattr(getattr(app, "state", None), "settings", None)
    if isinstance(bound, Settings):
        return bound
    return get_settings()


class AuthMiddleware:
    """Reject requests that do not present the configured bearer token."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = settings_for(scope)
        path = scope.get("path", "")
        if not settings.api_token or scope.get("method") == "OPTIONS" or path in _OPEN_PATHS:
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        header = request.headers.get("authorization", "")
        prefix = "Bearer "
        presented = header[len(prefix) :] if header.startswith(prefix) else ""
        if presented and token_matches(presented, settings.api_token):
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            status_code=401,
            content={"detail": "Sign in required."},
        )
        await response(scope, receive, send)
