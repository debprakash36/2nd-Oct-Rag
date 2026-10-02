"""Re-embed every chunk with a real model, resumably, on a staging copy.

**The problem.** All 308 chunks in `rag.db` carry `embedding_model =
'fake-embed-v1'`. Those vectors are a SHA-256 hash bucket, not a semantic
embedding: similarity between them is a hash collision rate, not meaning. It
looked like a working system because counts matched, `/health` said `ok`, and
queries returned candidates -- but nothing was being retrieved by meaning, and
the 11 rows at 64 dimensions raised `ValueError: embedding dimension mismatch`
on top of that. `docs/known_issues.md` item 1 made the second problem visible;
this fixes both by replacing the vectors.

**Why a script and not a loop.** Because it has to be resumable. Re-embedding
308 chunks is ~30 batched network calls, and a run that dies at chunk 200 must
not leave the database half-converted -- a mixed corpus is worse than the
uniformly wrong one it replaced, because every subsequent query returns
plausible-looking results drawn from two different vector spaces. So progress is
committed per batch and written into the target rows themselves, and a re-run
resumes from whatever is already correct. There is no separate progress file to
drift out of sync with the data: `embedding_model` on each row *is* the record
of what has been done.

**Safety.** Writes go to a staging copy, never to the live database. The caller
is expected to have made a backup and to verify the result before swapping;
this script deliberately does not touch `rag.db`, and `--apply` refers to the
*staging* file, not to promoting it. Promoting is a separate, manual,
human decision.

Usage:
    python scripts/reembed_chunks.py --dry-run
    python scripts/reembed_chunks.py --db rag_reembed.db --apply
    python scripts/reembed_chunks.py --db rag_reembed.db --apply --resume
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Written onto every chunk this script converts. Also the resume marker: a row
#: whose `embedding_model` already equals this and whose width already equals
#: `dim` is done, whatever the contents of the other columns.
TARGET_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: all-MiniLM-L6-v2's native width. Not configurable: a corpus indexed at one
#: width cannot be queried at another, so allowing an arbitrary value here would
#: just produce a database that fails at query time instead of at run time.
TARGET_DIM = 384

#: Sent per request. The hosted API accepts a list, and a batch is one billable
#: request, so this trades latency against the chance of losing a whole batch to
#: a timeout. 16 keeps the worst-case work per call small.
DEFAULT_BATCH = 16

#: Retries for a transient failure (429, 5xx, connection reset). 429 is the one
#: that matters in practice on a free tier: the API rate-limits per minute, and
#: 308 chunks at 16 per batch is enough to trip it.
DEFAULT_RETRIES = 5

#: Base delay for exponential backoff, in seconds. Jittered per attempt so that
#: a retry storm from several runs does not resynchronise into a second storm.
BASE_BACKOFF = 2.0


@dataclass
class ChunkRow:
    """A chunk awaiting conversion."""

    chunk_id: str
    doc_id: str
    text: str
    token_count: int


@dataclass
class RunStats:
    """What a run did, for the summary at the end."""

    already_done: int = 0
    converted: int = 0
    batches: int = 0
    retries: int = 0
    failures: list[str] = field(default_factory=list)


def open_db(path: str) -> sqlite3.Connection:
    """Open the staging database for writing.

    `isolation_level=None` puts the driver in autocommit, so each `commit()` is
    an actual durable boundary. The default behaviour defers to an implicit
    transaction opened on the first write, which would defeat the per-batch
    commit that makes this resumable.
    """
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def survey(connection: sqlite3.Connection) -> dict[str, int]:
    """Describe the corpus: what model, what width, how many.

    Width is read with `json_array_length` rather than `length()` so the numbers
    are the real vector widths. Counting characters would work here only because
    widths correlate with serialised size.
    """
    rows = connection.execute(
        "SELECT c.embedding_model, "
        "       json_array_length(c.embedding) AS width, "
        "       COUNT(*) AS n "
        "FROM chunks c "
        "GROUP BY c.embedding_model, width "
        "ORDER BY n DESC"
    ).fetchall()
    return {
        f"{model or '<null>'} @ {width}": n for model, width, n in rows
    }


def pending(connection: sqlite3.Connection, *, resume: bool) -> list[ChunkRow]:
    """Chunks still needing conversion, oldest first.

    With `resume`, rows already at the target model *and* width are skipped. The
    width check is not redundant: a row can carry the target model name from an
    interrupted or partial earlier run and still hold a truncated or partially
    written vector, and trusting the label alone would skip it forever.
    """
    if not resume:
        rows = connection.execute(
            "SELECT chunk_id, doc_id, text, token_count FROM chunks "
            "ORDER BY doc_id, chunk_index"
        ).fetchall()
    else:
        # A row is done only if it is at BOTH the target model and the target
        # width, so "pending" is the negation of that pair -- expressed as a
        # disjunction. Writing it as `embedding_model = ?` here would be the
        # obvious-looking and wrong inversion: it selects rows already at the
        # target model, so a resumed run would skip the entire `fake-embed-v1`
        # remainder and report a fully converted database that was never
        # converted. The width disjunct alone cannot fix that, because the 295
        # wrong-model rows are all already 384-wide.
        #
        # `IS NULL` is kept as an explicit disjunct because `!=` against NULL
        # yields NULL, not TRUE, and a NULL-labelled row would otherwise drop out.
        rows = connection.execute(
            "SELECT chunk_id, doc_id, text, token_count FROM chunks "
            "WHERE embedding_model IS NULL "
            "   OR embedding_model != ? "
            "   OR json_array_length(embedding) IS NULL "
            "   OR json_array_length(embedding) != ? "
            "ORDER BY doc_id, chunk_index",
            (TARGET_MODEL, TARGET_DIM),
        ).fetchall()
    return [
        ChunkRow(
            chunk_id=str(chunk_id),
            doc_id=str(doc_id),
            text=str(text or ""),
            token_count=int(token_count or 0),
        )
        for chunk_id, doc_id, text, token_count in rows
    ]


def build_provider() -> object:
    """Construct the hosted embedding provider from the local `.env`.

    Reads settings rather than taking the token as an argument, so the value
    never appears in a shell history or a process listing. Fails loudly if
    `EMBEDDING_PROVIDER` is not `huggingface`: silently falling back to the fake
    provider here would rewrite 308 vectors with a hash function and report
    success, which is the exact failure this script exists to undo.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from app.core.config import get_settings
    from app.providers.embedding import HuggingFaceEmbeddingProvider

    settings = get_settings()
    if settings.embedding_provider != "huggingface":
        raise SystemExit(
            f"refusing to run: EMBEDDING_PROVIDER is "
            f"{settings.embedding_provider!r}, not 'huggingface'. This script "
            f"writes real embeddings; run it with the hosted provider set."
        )
    if not settings.hf_token:
        raise SystemExit("refusing to run: HF_TOKEN is empty.")

    provider = HuggingFaceEmbeddingProvider(
        token=settings.hf_token,
        base_url=settings.hf_inference_url,
        model=TARGET_MODEL,
        dim=TARGET_DIM,
        timeout=settings.hf_timeout_seconds,
        batch_size=DEFAULT_BATCH,
    )
    return provider


def embed_batch(
    provider: object,
    texts: list[str],
    *,
    retries: int,
    stats: RunStats,
) -> list[list[float]] | None:
    """Embed one batch, retrying transient failures with jittered backoff.

    Returns None if every attempt failed. Retries on 429 and 5xx via the
    provider's own exception, and never on a 4xx that is not a rate limit: a bad
    token or a wrong model name returns the same error forever, so retrying it
    just delays the real diagnosis. The provider's `EmbeddingError` message
    deliberately carries only the status, so retry classification is on the
    status, not on string matching.
    """
    from app.core.errors import EmbeddingError

    delay = BASE_BACKOFF
    for attempt in range(retries + 1):
        try:
            vectors = provider.embed(texts, model=TARGET_MODEL)  # type: ignore[attr-defined]
            if len(vectors) != len(texts):
                raise EmbeddingError(
                    f"provider returned {len(vectors)} vectors for {len(texts)} inputs"
                )
            for vector in vectors:
                if len(vector) != TARGET_DIM:
                    raise EmbeddingError(
                        f"provider returned {len(vector)} dimensions, expected "
                        f"{TARGET_DIM}"
                    )
            return vectors
        except EmbeddingError as exc:
            message = str(exc)
            retriable = "429" in message or any(
                code in message for code in ("500", "502", "503", "504")
            ) or "ConnectError" in message or "TimeoutException" in message
            if not retriable or attempt == retries:
                if retriable:
                    stats.failures.append(f"gave up after {retries} retries: {message}")
                else:
                    stats.failures.append(f"not retriable: {message}")
                return None
            stats.retries += 1
            sleep_for = delay * (1.0 + random.random() * 0.3)
            print(
                f"    transient ({message[:60]}); "
                f"retry {attempt + 1}/{retries} in {sleep_for:.1f}s",
                flush=True,
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, 60.0)
    return None


def write_batch(
    connection: sqlite3.Connection,
    rows: list[ChunkRow],
    vectors: list[list[float]],
) -> None:
    """Persist one converted batch and its model label in a single transaction.

    `embedding_model` is written in the same statement as the vector so a row can
    never claim to be at the target model while holding something else. Without
    that, a crash between two writes would leave a row that resume trusts and
    retrieval cannot use.
    """
    payload = [
        (json.dumps(vector), TARGET_MODEL, row.chunk_id)
        for row, vector in zip(rows, vectors, strict=True)
    ]
    connection.execute("BEGIN")
    try:
        connection.executemany(
            "UPDATE chunks SET embedding = ?, embedding_model = ? WHERE chunk_id = ?",
            payload,
        )
        connection.execute("COMMIT")
    except sqlite3.Error:
        connection.execute("ROLLBACK")
        raise


def verify(connection: sqlite3.Connection) -> tuple[bool, str]:
    """Confirm every chunk is at the target model and width.

    This is the check that makes the result trustworthy, and it is deliberately
    strict: a partial conversion must not be able to pass as success.
    """
    total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    good = connection.execute(
        "SELECT COUNT(*) FROM chunks "
        "WHERE embedding_model = ? AND json_array_length(embedding) = ?",
        (TARGET_MODEL, TARGET_DIM),
    ).fetchone()[0]
    other = connection.execute(
        "SELECT c.embedding_model, json_array_length(c.embedding) AS width, COUNT(*) "
        "FROM chunks c GROUP BY 1, 2 ORDER BY 3 DESC"
    ).fetchall()
    detail = "; ".join(
        f"{model or '<null>'} @ {width}: {n}" for model, width, n in other
    )
    if good == total:
        return True, f"{good}/{total} chunks at {TARGET_MODEL} @ {TARGET_DIM}"
    return False, f"{good}/{total} converted -- {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default=str(REPO_ROOT / "rag_reembed.db"),
        help="database to write. Never rag.db; stage first, then swap by hand.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually embed and write (default is survey-only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="survey and report only; same as omitting --apply",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip rows already at the target model and width",
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="convert at most N chunks (0 = all). For smoke-testing the API path.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if db_path.resolve() == (REPO_ROOT / "rag.db").resolve():
        print(
            "refusing to write rag.db directly. Re-embed a staging copy "
            "(rag_reembed.db) and swap it in yourself after verifying.",
            file=sys.stderr,
        )
        return 2
    if not db_path.exists():
        print(f"no such database: {db_path}", file=sys.stderr)
        return 2

    connection = open_db(str(db_path))
    try:
        print(f"database: {db_path}")
        print(f"model:    {TARGET_MODEL} @ {TARGET_DIM}\n")
        print("current state:")
        for label, count in survey(connection).items():
            print(f"  {label}: {count}")

        todo = pending(connection, resume=args.resume)
        total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        already = total - len(todo)

        if args.limit > 0:
            todo = todo[: args.limit]

        print(f"\n{len(todo)} chunk(s) to convert; {already} already at target")
        if args.dry_run or not args.apply:
            if todo:
                print(f"  first: {todo[0].chunk_id} ({todo[0].token_count} tokens)")
                print(f"  last:  {todo[-1].chunk_id} ({todo[-1].token_count} tokens)")
                longest = max(todo, key=lambda r: len(r.text))
                print(
                    f"  longest text: {len(longest.text)} chars ({longest.chunk_id})"
                )
            print("\ndry run. No network calls made, nothing written.")
            return 0

        if not todo:
            ok, detail = verify(connection)
            print(f"\n{detail}")
            print("nothing to do; database already converted." if ok else "INCOMPLETE")
            return 0 if ok else 1

        provider = build_provider()
        stats = RunStats(already_done=already)

        print(f"\nconverting in batches of {args.batch}...", flush=True)
        for start in range(0, len(todo), args.batch):
            batch = todo[start : start + args.batch]
            vectors = embed_batch(
                provider, [row.text for row in batch],
                retries=args.retries, stats=stats,
            )
            if vectors is None:
                # Leave the remaining rows untouched and stop. They are still
                # flagged as pending, so a re-run with --resume picks up from
                # exactly here.
                print(
                    f"\nstopped at {stats.converted}/{len(todo)}: batch failed and "
                    f"was not written. Re-run with --resume to continue.",
                    file=sys.stderr,
                )
                for failure in stats.failures[-3:]:
                    print(f"  {failure}", file=sys.stderr)
                break
            write_batch(connection, batch, vectors)
            stats.converted += len(batch)
            stats.batches += 1
            done = stats.converted
            print(
                f"  [{done:>4}/{len(todo)}] {batch[0].chunk_id}..{batch[-1].chunk_id}",
                flush=True,
            )

        print(
            f"\nconverted {stats.converted} in {stats.batches} batch(es)"
            f"{f', {stats.retries} retries' if stats.retries else ''}"
        )
        ok, detail = verify(connection)
        print(f"verify: {detail}")
        if not ok:
            print(
                "\nINCOMPLETE -- do not swap this file in. Re-run with --resume.",
                file=sys.stderr,
            )
            return 1
        print("all chunks converted and verified in the staging database.")
        print("rag.db is unchanged. Swap it in yourself once you have checked it.")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())