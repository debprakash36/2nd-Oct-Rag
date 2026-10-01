"""Structured logging with request-scoped trace IDs.

`trace_id` is bound to a context variable so any log call anywhere in a request
emits the same ID without being passed the request. This is what makes a single
user-visible answer reconstructable from logs alone (NFR-8).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_configured = False


def set_trace_id(value: str) -> None:
    """Bind a trace ID to the current context."""
    _trace_id.set(value)


def get_trace_id() -> str | None:
    """Return the current context's trace ID, if any."""
    return _trace_id.get()


class TraceIdFilter(logging.Filter):
    """Inject the context's `trace_id` onto every record.

    Added to the root handler rather than passed per-call so that log statements
    in deep library code are correlatable without touching them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
        }
        # Structured extras passed via logger.info(..., extra={...}) land here.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName", "trace_id",
}


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON handler on the root logger. Idempotent."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(TraceIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
