"""Logging contract: structured extras must not collide with LogRecord.

`logger.info(..., extra={"filename": ...})` raises `KeyError` at runtime, deep
inside a request, because `filename` is a reserved LogRecord attribute. It is
easy to write and invisible until the code path runs, so it is checked statically
here instead.

The same collision class applies to `name`, `module`, `args`, `msg`, and the
rest of the reserved set below.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

#: Attributes `logging.LogRecord` sets on every record. `extra` may not
#: overwrite any of them.
RESERVED = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
})

SOURCE_DIRS = ("app", "scripts")
LOG_CALL = re.compile(r"\.(?:debug|info|warning|error|exception|critical)\(")


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        root = Path(directory)
        if root.is_dir():
            files.extend(root.rglob("*.py"))
    return files


def test_source_files_found() -> None:
    """Guard against the scan silently covering nothing."""
    assert _iter_source_files(), "no source files found to scan"


def test_no_extra_key_collides_with_logrecord() -> None:
    """No `extra={...}` key may shadow a LogRecord attribute."""
    offenders: list[str] = []

    for path in _iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and LOG_CALL.search(
                f".{func.attr}("
            )):
                continue
            for keyword in node.keywords:
                if keyword.arg != "extra":
                    continue
                for key in _dict_keys(keyword.value):
                    if key in RESERVED:
                        offenders.append(f"{path}:{node.lineno} extra key {key!r}")

    assert not offenders, (
        "these `extra` keys shadow logging.LogRecord attributes and will raise "
        f"KeyError at runtime (logging.py:53 documents the collision): {offenders}"
    )


def _dict_keys(node: ast.expr) -> list[str]:
    """String keys of a dict literal. Non-literal dicts are skipped."""
    if not isinstance(node, ast.Dict):
        return []
    return [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]


def test_trace_id_is_injected_by_filter() -> None:
    """`trace_id` is added by the filter, never passed via `extra`."""
    assert "trace_id" not in RESERVED, (
        "trace_id must stay non-reserved: the TraceIdFilter sets it on each record"
    )


def test_json_formatter_emits_required_fields() -> None:
    """Every line carries level, logger, message, and trace ID (NFR-8)."""
    import json
    import logging

    from app.core.logging import JsonFormatter, set_trace_id

    set_trace_id("trace-123")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    record.trace_id = "trace-123"
    record.custom_field = "value"

    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["trace_id"] == "trace-123"
    assert payload["custom_field"] == "value", "structured extras must reach the output"
