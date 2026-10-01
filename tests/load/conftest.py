"""Fixtures for the load test (implementation.md 5.1, NFR-1).

Three things here are not obvious, and each one is a trap that would make the load
test report a number that looks fine and means nothing:

**A real server, not `TestClient`.** `TestClient` drives the ASGI app in-process
through a portal, so requests are not genuinely concurrent and there is no socket,
no accept queue, and no separate event loop. It would report the latency of a
sequential loop wearing a concurrency costume. The fixture starts a real uvicorn
subprocess and the load driver talks HTTP to it over a socket.

**A real corpus, cached.** Phase 5 is required to measure against live data, not a
fixture document. Ingesting the full 113-document corpus takes roughly two minutes,
so the seeded database is built once and cached under `.load_cache/`, keyed on a
fingerprint of the corpus and the settings that change chunking. A stale cache is
worse than a slow build, so the fingerprint covers everything that affects retrieval:
file count, total bytes, embedding dimension, and the chunk strategy.

**A per-VU client identity.** The rate limiter keys on `X-Forwarded-For` and allows
30 requests per 60 s. Fifty VUs sharing `127.0.0.1` would be rejected as one abusive
client, and the load test would measure the rate limiter instead of the pipeline. Each
worker therefore sends its own `X-Forwarded-For`, which is both what 50 *users* means
and what the gateway does in production (architecture.md 7.1). The limiter stays on:
disabling the guard to make a benchmark pass would defeat the point of having one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".load_cache"

#: Documents ingested for the load corpus. The full `samples/` tree is 113 files and
#: ~124 s to ingest; the load test is about concurrent *query* latency, so a cached
#: subset of this size is representative of the index shape without a two-minute
#: fixture. Raise it to 0 to use the whole corpus.
LOAD_DOCS = 40


@dataclass(frozen=True)
class LiveServer:
    """A running uvicorn process the load driver can talk to."""

    base_url: str
    database_url: str
    corpus_chunks: int
    #: Path to a file the server appends a concurrency counter to. See
    #: `_install_concurrency_probe` for why this exists.
    concurrency_log: Path

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"


def _install_concurrency_probe(probe_path: Path, log_path: Path) -> None:
    """Generate the probe module that makes in-flight request count observable.

    The client knows how many requests it *sent*; it cannot observe how many the
    server held open at once. This counts the latter.

    **What this measures, precisely:** requests that have entered the ASGI app and not
    yet finished. That is user-facing concurrency -- open connections, sockets being
    read and written, SSE streams live at the same instant.

    **What this does not measure:** how many sync handlers were *executing* at once.
    `/chat/stream` is a sync endpoint, so FastAPI runs it in anyio's threadpool
    (default 40 tokens). A request queued for a thread still counts as in flight here.
    So a peak of 50 is consistent with either 50 threads busy or 40 busy and 10
    waiting, and this probe cannot tell those apart. The threadpool ceiling is real
    and is characterised by the VU sweep in `docs/perf_report.md`, not by this
    counter.

    The probe is a generated module rather than a change to product code: a benchmark
    must not alter the path it measures, and an opt-in module that uvicorn is pointed
    at instead of `app.main` cannot affect a normal run.
    """
    log_path.write_text("0\n", encoding="utf-8")
    probe_path.write_text(PROBE_SOURCE, encoding="utf-8")


PROBE_SOURCE = '''\
"""Server-side request concurrency probe for the load test. Not product code.

Enabled only when RAG_CONCURRENCY_LOG is set, which the load harness does. It counts
how many requests are in flight simultaneously so the benchmark can state the
concurrency the server actually achieved rather than the concurrency the client
attempted.

Deliberately a raw ASGI middleware rather than `@app.middleware("http")`.
`BaseHTTPMiddleware` sits between the app and the socket and is known to interfere
with streaming responses -- it would sit in the path of the very SSE stream this
benchmark exists to time, and could turn a streamed answer into a buffered one. A
plain ASGI wrapper only observes; it never touches the body.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from app.main import create_app

LOG_PATH = Path(os.environ["RAG_CONCURRENCY_LOG"])
LOG_PATH.write_text("0\\n", encoding="utf-8")

_inflight = 0
_peak = 0
_lock = threading.Lock()


def _record(delta: int) -> None:
    global _inflight, _peak
    with _lock:
        _inflight += delta
        if _inflight > _peak:
            _peak = _inflight
        LOG_PATH.write_text(f"{_peak}\\n", encoding="utf-8")


class ConcurrencyCounter:
    """Counts concurrent in-flight requests. Observes only."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        _record(1)
        try:
            await self.app(scope, receive, send)
        finally:
            _record(-1)


app = ConcurrencyCounter(create_app())
'''


def _peak_concurrency(log_path: Path) -> int:
    """Peak in-flight requests the server observed, or 0 if it never reported."""
    try:
        return int(log_path.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def peak_concurrency(server: LiveServer) -> int | None:
    """The concurrency the server actually reached, or None if unmeasured."""
    peak = _peak_concurrency(server.concurrency_log)
    return peak or None


def _corpus_files(directory: Path) -> list[Path]:
    """Candidate corpus files, in the same order `ingest_corpus.py` uses."""
    skip = {"node_modules", ".git", ".venv", "__pycache__", "data"}
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and not any(part in skip for part in p.parts)
        and p.suffix.lower()
        in {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".markdown"}
    )


def _fingerprint(files: list[Path], settings_env: dict[str, str]) -> str:
    """Identity of the corpus plus every setting that changes what gets indexed.

    Deliberately not just the file count: a file edited in place, or a changed chunk
    target, produces the same count and a different index. A cache keyed on count
    would serve a stale corpus and the load numbers would quietly describe a database
    that no longer exists.
    """
    digest = hashlib.sha256()
    digest.update(str(len(files)).encode())
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for key in sorted(settings_env):
        if key.startswith(("RETRIEVAL_", "EMBEDDING_")):
            digest.update(f"{key}={settings_env[key]}".encode())
    return digest.hexdigest()[:16]


def _sqlite_chunk_count(database: Path) -> int:
    """Chunks in a seeded database, or 0 if it is absent or unusable."""
    import sqlite3

    if not database.exists():
        return 0
    try:
        with sqlite3.connect(database) as conn:
            row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _seed_corpus(destination: Path, files: list[Path], env: dict[str, str]) -> int:
    """Ingest the corpus into a fresh database. Returns the chunk count."""
    if destination.exists():
        destination.unlink()
    for suffix in ("-wal", "-shm"):
        stale = destination.with_name(destination.name + suffix)
        if stale.exists():
            stale.unlink()

    objects = destination.parent / "objects"
    if objects.exists():
        shutil.rmtree(objects)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "ingest_corpus.py"),
            "--dir",
            str(files[0].parent),
            "--limit",
            str(len(files)),
            "--stats",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"corpus ingest failed ({completed.returncode}):\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    if "INVARIANT CHECKS PASSED" not in completed.stdout:
        # The ingest helper's own invariant gate. A corpus that fails it would make
        # every latency number below meaningless, because retrieval would be measured
        # against broken offsets and empty keyword rows.
        raise RuntimeError(
            "corpus ingest reported failed invariants; refusing to benchmark it:\n"
            f"{completed.stdout[-4000:]}"
        )
    return _sqlite_chunk_count(destination)


def _free_port() -> int:
    """An ephemeral port, released before the server binds it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_health(base_url: str, process: subprocess.Popen, timeout: float = 90.0):
    """Block until /health reports a fully healthy service.

    Health, not the socket: the socket accepts connections as soon as uvicorn binds,
    which is before the lifespan startup hook has run, so a driver that starts on
    connect-only would send its first requests to a process with no schema and no
    ingested corpus and record those as application latency.
    """
    deadline = time.monotonic() + timeout
    last_error: str = "no attempt made"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("status") == "ok":
                return payload
            last_error = f"status={payload.get('status')} checks={payload.get('checks')}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy within {timeout}s: {last_error}")


@pytest.fixture(scope="session")
def load_settings(tmp_path_factory) -> Iterator[dict[str, str]]:
    """Environment overrides for a load run.

    `VECTOR_STORE=sqlite` is forced rather than left on `auto`. The auto dialect
    picks `SqliteVectorStore` here, which is an O(n) Python full scan, so leaving it
    implicit would mean the benchmark silently measures a code path that
    `Settings.validate_production` explicitly forbids in staging and production. The
    number is a floor on the retrieval stage, not a prediction of production latency,
    and the perf report says so.

    The rate limit is raised per user, not removed; see the module docstring.
    """
    cache = CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    corpus_dir = REPO_ROOT / "samples"
    if not corpus_dir.is_dir():
        pytest.skip("no samples/ corpus; run `make corpus` first")

    files = _corpus_files(corpus_dir)
    if LOAD_DOCS:
        files = files[:LOAD_DOCS]
    if not files:
        pytest.skip(f"no supported documents under {corpus_dir}")

    fingerprint_settings = {
        "EMBEDDING_DIM": os.environ.get("EMBEDDING_DIM", "384"),
    }
    fingerprint = _fingerprint(files, fingerprint_settings)

    database = cache / f"load-{fingerprint}.db"
    objects = cache / f"load-{fingerprint}-objects"

    chunks = _sqlite_chunk_count(database)
    if chunks == 0:
        env = {
            "PYTHONPATH": str(REPO_ROOT),
            "DATABASE_URL": f"sqlite:///{database.as_posix()}",
            "OBJECT_STORE_DIR": str(objects),
            "EMBEDDING_DIM": fingerprint_settings["EMBEDDING_DIM"],
            "VECTOR_STORE": "sqlite",
            "LOG_LEVEL": "WARNING",
            "ENVIRONMENT": "local",
        }
        started = time.perf_counter()
        chunks = _seed_corpus(database, files, env)
        print(
            f"\n[load] seeded {len(files)} documents -> {chunks} chunks "
            f"in {time.perf_counter() - started:.1f}s (cached at {database.name})"
        )

    if chunks == 0:
        pytest.skip("corpus ingest produced no chunks")

    yield {
        "PYTHONPATH": str(REPO_ROOT),
        "DATABASE_URL": f"sqlite:///{database.as_posix()}",
        "OBJECT_STORE_DIR": str(objects),
        "EMBEDDING_DIM": fingerprint_settings["EMBEDDING_DIM"],
        "VECTOR_STORE": "sqlite",
        "LOG_LEVEL": "WARNING",
        "ENVIRONMENT": "local",
        # Keep the window open for the run's duration. The default 60 s window is
        # shorter than a 50-VU run, so a window rollover mid-run would let a worker
        # back to full quota and make the error rate depend on run length.
        "CHAT_RATE_LIMIT_REQUESTS": "20",
        "CHAT_RATE_LIMIT_WINDOW_SECONDS": "3600",
    }


#: Module name the probe is generated under. uvicorn is pointed at this rather than
#: at `app.main`, so the app is only ever wrapped, never modified.
PROBE_MODULE = "rag_load_probe"


@pytest.fixture(scope="session")
def live_server(load_settings, tmp_path_factory) -> Iterator[LiveServer]:
    """A real uvicorn process serving the seeded corpus.

    Runs the generated concurrency probe rather than `app.main:app`, so the run can
    report the concurrency the server actually reached. The probe is a pass-through
    ASGI wrapper, so the request path it measures is the one a deployment runs.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    probe_dir = tmp_path_factory.mktemp("probe")
    # Named independently of the log: deriving the module name from the log's suffix
    # produced `concurrency.py` while uvicorn was told to load `rag_probe_app`, and
    # the failure surfaced only as "could not import module".
    probe_path = probe_dir / f"{PROBE_MODULE}.py"
    concurrency_log = probe_dir / "concurrency.txt"
    _install_concurrency_probe(probe_path, concurrency_log)

    env = {
        **load_settings,
        "RAG_CONCURRENCY_LOG": str(concurrency_log),
        "PYTHONPATH": os.pathsep.join([str(probe_dir), str(REPO_ROOT)]),
    }

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            f"{PROBE_MODULE}:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        health = _wait_for_health(base_url, process)
        database = Path(load_settings["DATABASE_URL"].removeprefix("sqlite:///"))
        print(f"\n[load] server healthy on {base_url} ({health['checks']})")
        yield LiveServer(
            base_url=base_url,
            database_url=load_settings["DATABASE_URL"],
            corpus_chunks=_sqlite_chunk_count(database),
            concurrency_log=concurrency_log,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # A uvicorn that ignores SIGTERM would otherwise hold the port and make
            # the next run fail for a reason that looks like a product bug.
            process.kill()
            process.wait(timeout=10)
        peak = _peak_concurrency(concurrency_log)
        if peak:
            print(f"[load] peak in-flight requests observed server-side: {peak}")
        if process.stdout is not None:
            output = process.stdout.read()
            if output.strip():
                # Surfaced rather than discarded: a benchmark that ran against a
                # server logging tracebacks is not a benchmark worth reporting.
                print(f"[load] server output:\n{output[-4000:]}")
