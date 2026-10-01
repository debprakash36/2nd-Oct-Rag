"""Request middleware: trace IDs and access logging (NFR-8)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, set_trace_id

log = get_logger("app.access")


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Bind a trace ID to the request and emit one access log line.

    An inbound `X-Trace-Id` is honoured so a trace can be stitched across
    services; otherwise a new one is generated.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex
        set_trace_id(trace_id)
        request.state.trace_id = trace_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            log.exception(
                "request failed",
                extra={"path": request.url.path, "method": request.method,
                       "elapsed_ms": elapsed_ms},
            )
            raise
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Trace-Id"] = trace_id
        log.info(
            "request",
            extra={"path": request.url.path, "method": request.method,
                   "status": response.status_code, "elapsed_ms": elapsed_ms},
        )
        return response
