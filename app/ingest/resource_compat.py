"""Thin portability shim over `resource` (POSIX only).

Exists so `sandbox.py` can call into limits without guarding every call site
with a platform check, and so `last_enforced()` can report honestly which
limits actually took effect.
"""

from __future__ import annotations

import sys
from types import ModuleType

_enforced: dict[str, bool] = {"cpu": False, "address_space": False, "file_size": False}
_module: ModuleType | None = None

if sys.platform != "win32":
    try:
        import resource as _resource

        _module = _resource
    except ImportError:  # pragma: no cover - unusual POSIX build
        _module = None


def available() -> bool:
    """Whether POSIX resource limits can be applied on this platform."""
    return _module is not None


def apply_limits(*, cpu_seconds: int, address_space_bytes: int) -> None:
    """Apply CPU, address space, and file size limits. Best effort.

    Each limit is applied independently so one unsupported limit does not
    prevent the others. A failure is recorded, not raised: a missing limit is a
    degraded sandbox, not a reason to fail an otherwise valid document. The
    caller's log line makes the degradation visible.
    """
    if _module is None:
        return
    r = _module
    for name, limit, key in (
        (r.RLIMIT_CPU, cpu_seconds, "cpu"),
        (r.RLIMIT_AS, address_space_bytes, "address_space"),
        (r.RLIMIT_FSIZE, address_space_bytes, "file_size"),
    ):
        try:
            _soft, hard = r.getrlimit(name)
            # Never raise an existing hard limit: setrlimit fails for an
            # unprivileged process attempting to exceed it.
            target = limit if hard == r.RLIM_INFINITY else min(limit, hard)
            r.setrlimit(name, (target, hard))
            _enforced[key] = True
        except (ValueError, OSError):
            _enforced[key] = False


def last_enforced() -> dict[str, bool]:
    """Report which limits were applied by the most recent `apply_limits`."""
    return dict(_enforced)
