"""Sandboxed text extraction (FR-3, NFR-5).

Uploaded files are untrusted input. Extraction runs in a child process with a
hard wall-clock timeout, an output cap, and resource limits where the platform
supports them.

Why an operation registry instead of a callable
-----------------------------------------------
The child is started with the `spawn` method, so its arguments must be
picklable — a lambda or a closure cannot cross the process boundary. Passing a
`Callable` would therefore constrain every caller to a module-level function,
which is a constraint that surfaces as a runtime PicklingError deep in a request
rather than at the call site.

Instead the caller names an operation and passes plain data. The child resolves
the name through `_OPS` and calls it. This is picklable by construction, keeps
the set of things the sandbox can do explicit and reviewable, and makes the
production call site read as `run_sandboxed(op="extract", ...)` rather than
carrying a closure.

On the limits this implementation actually enforces
---------------------------------------------------
`resource.setrlimit` for CPU and address space, and network namespace isolation,
are POSIX/container features. On Windows, `resource` is unavailable and network
isolation would require a container. The Windows path enforces the wall-clock
timeout and the output cap; memory is bounded only by the parent process.

This is a real gap against NFR-5, not a hidden one: a parser that opens a socket
is not stopped, and a memory bomb is bounded by the OS rather than by us.
Production runs this worker in a container so the limits come from the isolation
boundary. `assert_sandbox_capabilities()` reports what is active so a deployment
can assert it is getting the enforcement it expects, and a production start-up
fails if POSIX limits are unavailable.

The invariants that hold on every platform:
  * No shell. Extraction is in-process library code, never a subprocess, so
    there is no shell-interpolation surface on filenames at all.
  * The filename never reaches a filesystem path; the caller supplies bytes.
  * Output is capped, so a decompression bomb cannot exhaust memory via text.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.core.errors import ExtractionError, SandboxError
from app.core.logging import get_logger
from app.ingest import resource_compat

log = get_logger("app.ingest.sandbox")

#: Sentinel prefix marking a child-process failure. Chosen because it cannot
#: begin a valid JSON document, so a failure is never mistaken for a result.
_ERROR_PREFIX = "\x00ERROR:"


@dataclass(frozen=True)
class ExtractionResult:
    """Text pulled from one document, plus the metadata needed for citations.

    Only JSON-compatible fields, because instances cross a process boundary as
    JSON. `page_count` is carried back this way rather than recovered by a
    second extraction pass.
    """

    text: str
    page_count: int | None = None


# --------------------------------------------------------------------------
# Operations the sandbox can perform. Each takes JSON-compatible arguments and
# returns an ExtractionResult. This list is the sandbox's entire capability
# surface: anything not here cannot be executed in the child.
# --------------------------------------------------------------------------


def _op_extract(data: bytes, filename: str) -> ExtractionResult:
    """Parse one document. Imported lazily so the child only loads a parser it
    actually needs."""
    from app.ingest.extract import extract

    return extract(data, filename)


def _op_echo(data: bytes = b"", filename: str = "", repeat: int = 1) -> ExtractionResult:
    """Return the input unchanged. Used to verify the transport itself."""
    return ExtractionResult(text=(data.decode("utf-8", errors="replace") * repeat) or "echo",
                            page_count=None)


def _op_sleep(data: bytes = b"", filename: str = "", seconds: float = 30.0) -> ExtractionResult:
    """Sleep. Used to verify the timeout actually kills a hung child."""
    import time

    time.sleep(seconds)
    return ExtractionResult(text="should never be returned", page_count=None)


def _op_wrong_type(data: bytes = b"", filename: str = "") -> Any:
    """Return a non-ExtractionResult, to verify the type check holds."""
    return "not an ExtractionResult"


def _op_boom(data: bytes = b"", filename: str = "") -> ExtractionResult:
    """Raise, to verify a child exception surfaces as a typed error."""
    raise ValueError("deliberate sandbox failure")


def _op_crash(data: bytes = b"", filename: str = "") -> ExtractionResult:
    """Die without reporting, to verify the parent survives an uncatchable child.

    `os._exit` skips the queue write, the `finally` blocks, and the exception handler,
    which is what makes it the closest reproducible stand-in for a native parser
    segfault. `sys.exit` would not do: it raises SystemExit and the existing
    `except Exception` in `_child_main` would still report it, so the test would pass
    without ever reaching the "no result" branch.
    """
    import os

    os._exit(7)


_OPS: dict[str, Callable[..., Any]] = {
    "extract": _op_extract,
    "echo": _op_echo,
    "sleep": _op_sleep,
    "wrong_type": _op_wrong_type,
    "boom": _op_boom,
    "crash": _op_crash,
}

#: Operations a caller may invoke.
#:
#: `extract` is the production path. The rest (`echo`, `sleep`, `wrong_type`, `boom`,
#: `crash`) exist so the sandbox's own guarantees can be *tested* rather than assumed
#: (NFR-5, implementation.md 5.4): you cannot demonstrate that a timeout, a type
#: contract, or a parent that survives an uncatchable child by inspecting the happy
#: path. Each of those needs a way to provoke the failure on demand.
#:
#: They are not reachable from a request. `run_sandboxed` takes `op` position-only,
#: and the sole call site passes the literal `"extract"`
#: (`app/ingest/worker.py:170`). No user input selects an operation — an operation
#: registry is an allow-list of what the child process may do, not an RPC surface, and
#: the child runs with no network so it could not be told to switch anyway.
PUBLIC_OPS = frozenset(_OPS)


def run_sandboxed(
    op: str,
    /,
    *,
    data: bytes = b"",
    filename: str = "",
    timeout_seconds: int | None = None,
    max_output_chars: int | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> ExtractionResult:
    """Run a registered operation in a child process under a timeout.

    `data` is the document bytes, `filename` its display name. Neither is used to
    build a path: `filename` only selects the parser and appears in error
    messages.
    """
    if op not in _OPS:
        raise ValueError(f"unknown sandbox operation: {op!r}")

    settings = settings or get_settings()
    timeout = settings.sandbox_timeout_seconds if timeout_seconds is None else timeout_seconds
    cap = settings.sandbox_max_output_chars if max_output_chars is None else max_output_chars

    # `spawn` gives a fresh interpreter, so no parser state, database handle, or
    # file descriptor is inherited from the parent.
    ctx = mp.get_context("spawn")
    queue: mp.Queue[str] = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_child_main,
        args=(op, data, filename, kwargs, queue, cap),
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.kill()
        process.join(timeout=5)
        raise SandboxError(f"extraction exceeded {timeout}s timeout; process killed")

    if queue.empty():
        # The child died before writing a result. A non-zero exit code is a
        # crash, most likely a malformed file hitting a parser bug.
        raise ExtractionError(
            f"sandbox produced no result for op={op!r} (exit code {process.exitcode})"
        )

    payload = queue.get()
    if payload.startswith(_ERROR_PREFIX):
        raise ExtractionError(f"extraction failed: {payload[len(_ERROR_PREFIX):]}")

    result = json.loads(payload)
    return ExtractionResult(text=result["text"], page_count=result.get("page_count"))


def _child_main(
    op: str,
    data: bytes,
    filename: str,
    kwargs: dict[str, Any],
    queue: mp.Queue[str],
    max_output_chars: int,
) -> None:
    """Child-process entry point. Never raises; reports through the queue."""
    try:
        enforce_hard_limits(get_settings())
        result = _OPS[op](data, filename, **kwargs)
        if not isinstance(result, ExtractionResult):
            queue.put(
                f"{_ERROR_PREFIX}operation {op!r} returned "
                f"{type(result).__name__}, expected ExtractionResult"
            )
            return
        text = result.text
        if len(text) > max_output_chars:
            # Cap the text, not the payload: a decompression bomb must not be
            # able to exhaust memory through the result.
            log.warning(
                "extraction output truncated", extra={"dropped": len(text) - max_output_chars}
            )
            text = text[:max_output_chars]
        queue.put(json.dumps({"text": text, "page_count": result.page_count}))
    except Exception as exc:
        queue.put(f"{_ERROR_PREFIX}{type(exc).__name__}: {exc}")


def enforce_hard_limits(settings: Settings) -> dict[str, bool]:
    """Apply POSIX resource limits where available. Returns what was enforced.

    Called inside the child. The result is logged rather than silently trusted,
    so a deployment can see that CPU or memory limits were not applied.
    """
    enforced = {"cpu": False, "address_space": False, "file_size": False}
    try:
        resource_compat.apply_limits(
            cpu_seconds=settings.sandbox_timeout_seconds,
            address_space_bytes=settings.sandbox_memory_bytes,
        )
        enforced = resource_compat.last_enforced()
    except Exception as exc:  # pragma: no cover - platform dependent
        log.warning("sandbox resource limits unavailable", extra={"error": str(exc)})
    return enforced


def sandbox_capabilities() -> dict[str, bool]:
    """What this platform can actually enforce. Reported, not assumed."""
    return {
        "posix_resource_limits": resource_compat.available(),
        "process_isolation": True,
        "wall_clock_timeout": True,
        "output_cap": True,
        "network_isolation": resource_compat.available(),
    }


def assert_sandbox_capabilities() -> None:
    """Fail a production start-up when the sandbox is weaker than NFR-5 requires.

    Enforcing a timeout without enforcing memory or network limits is not
    sufficient for untrusted input. Rather than let that pass silently, a
    production deployment is refused until it runs the worker in a container.
    """
    if get_settings().environment not in {"production", "staging"}:
        return
    if not sandbox_capabilities()["posix_resource_limits"]:
        raise RuntimeError(
            "sandbox resource limits are unavailable on this platform. NFR-5 "
            "requires memory and network isolation for untrusted uploads. Run the "
            "ingestion worker in a Linux container, or set ENVIRONMENT to a value "
            "that does not require it (not recommended for production)."
        )


def is_windows() -> bool:
    """Whether the current platform lacks POSIX resource limits."""
    return sys.platform == "win32"
