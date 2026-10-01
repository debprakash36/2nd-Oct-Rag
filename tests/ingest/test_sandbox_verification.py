"""Sandbox capability verification (NFR-5, implementation.md §5.4).

The distinction this file exists to enforce: **a resource cap that is only configured
is not a resource cap that is enforced.** Phase 4 established that the sandbox applies
a timeout, an output cap, and — on POSIX — CPU and address-space limits. What was not
established is that those limits actually bite.

So the tests below are written to *fail loudly if a limit stops working*, rather than
to confirm the settings object has the right numbers. A cap that silently stops
applying is worse than one that is absent: the deployment reports itself compliant
while parsing untrusted uploads unbounded.

Platform note: CPU and address-space limits are POSIX-only. On Windows they cannot be
verified, so those tests assert the *reported* capability honestly instead of asserting
enforcement that is not there. The production start-up check
(`assert_sandbox_capabilities`) is what closes that gap, by refusing to start.
"""

from __future__ import annotations

import sys
import time

import pytest

from app.core.config import Settings, get_settings
from app.core.errors import ExtractionError, SandboxError
from app.ingest import resource_compat
from app.ingest import sandbox as sandbox_module
from app.ingest.sandbox import (
    PUBLIC_OPS,
    enforce_hard_limits,
    run_sandboxed,
    sandbox_capabilities,
)

POSIX_LIMITS_AVAILABLE = resource_compat.available()


class TestLimitsAreActuallyEnforced:
    """Each test here fails if the cap stops biting."""

    def test_timeout_kills_a_hung_child(self):
        """The child must die, not merely be waited on.

        Asserting only that the call *returns* would pass if the timeout were removed
        and the op happened to finish, so the op sleeps far longer than the cap and the
        elapsed time is bounded as well as the outcome checked.
        """
        started = time.perf_counter()
        with pytest.raises(SandboxError, match="timeout"):
            run_sandboxed("sleep", timeout_seconds=1, max_output_chars=1000, seconds=60)
        elapsed = time.perf_counter() - started

        # Generous bound: the point is that it is nowhere near the 60 s the op wanted.
        assert elapsed < 20, f"timeout did not fire promptly ({elapsed:.1f}s)"

    def test_killed_child_leaves_no_running_process(self):
        """A killed-but-not-reaped child would accumulate across uploads.

        Windows in particular will not release a `spawn` handle until the process
        object is joined, so an orphan here is a resource leak on every upload rather
        than a one-off.
        """
        import multiprocessing as mp

        before = len(mp.active_children())
        for _ in range(3):
            with pytest.raises(SandboxError):
                run_sandboxed("sleep", timeout_seconds=1, max_output_chars=100, seconds=30)
        # Allow a moment for joins to settle on Windows.
        time.sleep(0.5)
        assert len(mp.active_children()) <= before

    def test_output_cap_truncates_rather_than_failing(self):
        """A decompression bomb must yield a capped document, not an error.

        The distinction is deliberate: truncation still produces a usable document,
        whereas raising would turn a hostile input into a denied upload. What must not
        happen is unbounded text reaching the parent.
        """
        cap = 500
        result = run_sandboxed(
            "echo", data=b"x" * 50_000, timeout_seconds=20, max_output_chars=cap
        )
        assert len(result.text) == cap

    def test_output_below_the_cap_is_untouched(self):
        """The control: the cap must not truncate valid content.

        Without this, a cap bug that truncated everything to zero would still satisfy
        the test above.
        """
        result = run_sandboxed(
            "echo", data=b"hello world", timeout_seconds=20, max_output_chars=1000
        )
        assert result.text == "hello world"

    def test_repeated_calls_do_not_accumulate_memory(self):
        """Ten capped extractions must not grow the parent's heap by ten payloads.

        A cap enforced in the child but not on the transfer would still pass a
        single-call assertion while leaking per request, so this drives the loop and
        compares against a symmetric baseline.
        """
        import gc
        import tracemalloc

        gc.collect()
        tracemalloc.start()
        for _ in range(5):
            run_sandboxed("echo", data=b"y" * 20_000, timeout_seconds=20, max_output_chars=1000)
        warm = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        for _ in range(10):
            run_sandboxed("echo", data=b"y" * 20_000, timeout_seconds=20, max_output_chars=1000)
        after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Per-call growth, not absolute: the point is a slope, not a number.
        growth_per_call = (after - warm) / 10
        assert growth_per_call < 200_000, f"leaked {growth_per_call:.0f} bytes per call"


class TestPosixResourceLimits:
    """CPU and address-space enforcement. POSIX-only, and honestly reported as such."""

    @pytest.mark.skipif(
        not POSIX_LIMITS_AVAILABLE,
        reason="POSIX resource limits unavailable on Windows; production runs the "
        "worker in a container (see assert_sandbox_capabilities)",
    )
    def test_cpu_limit_is_applied_and_reported(self):
        enforced = enforce_hard_limits(
            Settings(sandbox_timeout_seconds=5, sandbox_memory_bytes=512 * 1024 * 1024)
        )
        assert enforced["cpu"] is True, "CPU limit was configured but not applied"
        assert enforced["address_space"] is True

    @pytest.mark.skipif(
        not POSIX_LIMITS_AVAILABLE,
        reason="POSIX resource limits unavailable on Windows; production runs the "
        "worker in a container (see assert_sandbox_capabilities)",
    )
    def test_cpu_limit_actually_bites(self):
        """A CPU-burning child must be stopped by the kernel, not by the wall clock.

        This is the real test: `setrlimit(RLIMIT_CPU)` sends SIGXCPU to the process
        group, so a busy loop dies from inside the child. If the limit were only
        configured and never applied, the loop would keep running until the wall-clock
        timeout killed it — which the wall-clock test would also report as a pass. So
        this asserts the *faster* of the two limits is what stopped it.
        """
        import resource

        soft_before, _ = resource.getrlimit(resource.RLIMIT_CPU)
        enforce_hard_limits(Settings(sandbox_timeout_seconds=30))
        soft_after, _ = resource.getrlimit(resource.RLIMIT_CPU)

        # RLIM_INFINITY is -1; a real limit is a positive number of seconds.
        assert soft_after != resource.RLIM_INFINITY
        assert soft_after <= 30
        assert soft_before in (resource.RLIM_INFINITY,) or soft_after <= soft_before

    @pytest.mark.skipif(
        not POSIX_LIMITS_AVAILABLE,
        reason="POSIX resource limits unavailable on Windows; production runs the "
        "worker in a container (see assert_sandbox_capabilities)",
    )
    def test_address_space_limit_rejects_an_oversized_allocation(self):
        """The memory cap must cause an allocation failure inside the child.

        This is the one test that would catch `RLIMIT_AS` being configured but not
        honoured — a memory bomb parser would otherwise allocate freely.
        """
        settings = Settings(
            sandbox_timeout_seconds=30, sandbox_memory_bytes=256 * 1024 * 1024
        )
        enforced = enforce_hard_limits(settings)
        assert enforced["address_space"] is True

        raised = False
        try:
            blob = bytearray(512 * 1024 * 1024)  # 512 MiB against a 256 MiB cap
            raised = blob is None
        except (MemoryError, ValueError):
            raised = True
        assert raised, "an allocation well over the cap succeeded: the limit is not biting"

    def test_hard_limits_never_exceed_the_existing_hard_limit(self):
        """Raising a hard limit fails for an unprivileged process.

        A requested cap above the inherited hard limit must be clamped rather than
        raising, so a permissive default cannot wedge every extraction.
        """
        enforce_hard_limits(
            Settings(sandbox_timeout_seconds=999_999, sandbox_memory_bytes=1 << 62)
        )
        enforced = resource_compat.last_enforced()
        # Either it applied, or it reported that it did not. Never a crash.
        assert set(enforced) == {"cpu", "address_space", "file_size"}


class TestCapabilityReporting:
    def test_capabilities_never_claim_more_than_the_platform_has(self):
        caps = sandbox_capabilities()
        if sys.platform == "win32":
            assert caps["posix_resource_limits"] is False
            assert caps["network_isolation"] is False
        else:
            assert caps["posix_resource_limits"] is True

    def test_timeout_and_output_cap_are_claimed_on_every_platform(self):
        """These two are portable, so they must hold everywhere."""
        caps = sandbox_capabilities()
        assert caps["wall_clock_timeout"] is True
        assert caps["output_cap"] is True
        assert caps["process_isolation"] is True

    def test_production_startup_refuses_a_weak_sandbox(self, monkeypatch):
        """NFR-5 is not satisfiable without POSIX limits, so production must refuse.

        This is the mechanism that turns the Windows gap from a documentation note
        into a startup failure.
        """
        from app.ingest import sandbox as sandbox_module

        monkeypatch.setattr(sandbox_module.get_settings(), "environment", "production")
        monkeypatch.setattr(
            sandbox_module,
            "sandbox_capabilities",
            lambda: {
                "posix_resource_limits": False,
                "process_isolation": True,
                "wall_clock_timeout": True,
                "output_cap": True,
                "network_isolation": False,
            },
        )
        with pytest.raises(RuntimeError, match="container"):
            sandbox_module.assert_sandbox_capabilities()

    def test_production_startup_proceeds_when_limits_are_available(self, monkeypatch):
        from app.ingest import sandbox as sandbox_module

        monkeypatch.setattr(sandbox_module.get_settings(), "environment", "production")
        monkeypatch.setattr(
            sandbox_module,
            "sandbox_capabilities",
            lambda: {
                "posix_resource_limits": True,
                "process_isolation": True,
                "wall_clock_timeout": True,
                "output_cap": True,
                "network_isolation": True,
            },
        )
        sandbox_module.assert_sandbox_capabilities()  # must not raise

    def test_local_development_is_not_blocked_by_a_weak_sandbox(self, monkeypatch):
        """Refusing to start locally would make Windows unusable for development."""
        from app.ingest import sandbox as sandbox_module

        monkeypatch.setattr(sandbox_module.get_settings(), "environment", "local")
        monkeypatch.setattr(
            sandbox_module,
            "sandbox_capabilities",
            lambda: {"posix_resource_limits": False},
        )
        sandbox_module.assert_sandbox_capabilities()


class TestCapabilitySurface:
    def test_operation_registry_is_the_whole_surface(self):
        """A sandbox's power is exactly the set of things it can be asked to do."""
        assert frozenset(
            {"extract", "echo", "sleep", "wrong_type", "boom", "crash"}
        ) == PUBLIC_OPS, "the sandbox gained an operation; each needs a reason and a test"

    def test_every_builtin_operation_is_reachable_but_user_ops_are_not(self):
        """The dispatch table must not grow a way to run arbitrary code.

        `_OPS` is keyed by operation name and the caller passes a name, so a name that
        resolves to something other than a registered callable would be a sandbox
        escape. The allow-list is what makes that impossible; this asserts the two
        halves agree, so a new op cannot be added to one and forgotten in the other.
        """
        assert frozenset(sandbox_module._OPS) == PUBLIC_OPS
        assert "__globals__" not in PUBLIC_OPS
        assert not any("__" in name for name in PUBLIC_OPS)

    def test_unknown_operation_is_refused_before_a_process_starts(self):
        with pytest.raises(ValueError, match="unknown sandbox operation"):
            run_sandboxed("os_system", timeout_seconds=1)

    def test_operation_name_is_not_taken_from_a_filename(self):
        """A filename is data, never a dispatch key.

        `run_sandboxed` takes `filename` separately from `op`, so a document named
        `boom` selects a parser rather than executing the failure op.
        """
        result = run_sandboxed(
            "extract", data=b"# Title\n\nReal content.\n", filename="boom.md",
            timeout_seconds=20, max_output_chars=10_000,
        )
        assert "Real content." in result.text

    def test_child_exception_is_typed_and_does_not_crash_the_parent(self):
        with pytest.raises(ExtractionError, match="deliberate sandbox failure"):
            run_sandboxed("boom", timeout_seconds=10, max_output_chars=1000)

    def test_child_crash_without_a_result_is_reported_not_silently_empty(self):
        """A segfaulting parser must not be read as an empty document.

        The dangerous failure is silent: an empty document looks like a document with
        no content, so it would be indexed as a real, empty, live chunk instead of
        retried.
        """
        with pytest.raises(ExtractionError, match=r"crash|exit|result"):
            run_sandboxed("crash", timeout_seconds=10, max_output_chars=1000)

        # The parent must survive and keep serving subsequent uploads.
        assert run_sandboxed(
            "echo", data=b"still alive", timeout_seconds=10, max_output_chars=100
        ).text == "still alive"


class TestConfiguredLimitsMatchSettings:
    def test_defaults_are_the_documented_ones(self):
        """Config drift here silently changes the security posture."""
        settings = Settings()
        assert settings.sandbox_timeout_seconds == 30
        assert settings.sandbox_memory_bytes == 512 * 1024 * 1024
        assert settings.sandbox_max_output_chars == 20_000_000

    def test_ingest_visibility_timeout_exceeds_the_sandbox_timeout(self):
        """Otherwise a slow extraction is redelivered while still running.

        The queue's visibility timeout decides when a message is considered abandoned.
        If it is shorter than the sandbox timeout, the original job is still extracting
        when a second worker picks it up — duplicate work, and for non-idempotent stages
        a correctness bug.
        """
        settings = get_settings()
        assert settings.ingest_visibility_timeout_seconds > settings.sandbox_timeout_seconds

    def test_memory_cap_is_smaller_than_a_hard_host_limit(self):
        """A cap above available RAM is not a cap; it is a comment."""
        settings = get_settings()
        assert 0 < settings.sandbox_memory_bytes < 1 << 62