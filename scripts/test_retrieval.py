"""Run the retrieval eval. One command, no environment setup.

Every other entry point into the eval harness needs five environment variables set
correctly first, and a mistyped `DATABASE_URL` fails with a stack trace that looks
like a code bug rather than a configuration mistake. This wraps that.

    .venv\\Scripts\\python.exe scripts\\test_retrieval.py
    .venv\\Scripts\\python.exe scripts\\test_retrieval.py --questions 20
    .venv\\Scripts\\python.exe scripts\\test_retrieval.py --sweep
    .venv\\Scripts\\python.exe scripts\\test_retrieval.py --db rag.db

Any argument is passed through to `eval/run_eval.py`, so this stays useful as the
harness grows new flags.

**Corpus selection is checked, not assumed.** `rag.db` currently holds 11 chunks
with 64-dimension embeddings alongside 297 at 384, and one wrong-width row aborts
every query. Rather than letting that surface as
`ValueError: embedding dimension mismatch`, this counts the rows first and says
which database is usable and why.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: A 384-wide vector serialises to far more JSON than this; 64-wide is far less.
#: Used to classify rows without parsing every vector.
WIDE_ENOUGH_CHARS = 3000

#: Preferred first, then the working database, then the default.
CANDIDATES = ("rag_eval.db", "rag.db")


def count_mismatched(db: Path) -> int:
    """Chunks whose stored vector width is too narrow to be scored against."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return int(
            connection.execute(
                "SELECT count(*) FROM chunks "
                "WHERE embedding IS NOT NULL AND length(embedding) < ?",
                (WIDE_ENOUGH_CHARS,),
            ).fetchone()[0]
        )
    except sqlite3.Error:
        return 0
    finally:
        connection.close()


def choose_db(explicit: str | None) -> tuple[Path, str]:
    """Pick a corpus, and explain the choice.

    Returns the database and a note. Raises with actionable guidance if the chosen
    corpus is unusable, because "eval reports recall 0.0" and "eval cannot run" are
    very different problems and should not look alike.

    An explicit `--db` is still checked. Honouring the flag without checking it would
    hand the user a `ValueError` from deep inside the vector store, which reads as a
    code bug; the flag should select a corpus, not bypass the diagnosis.
    """
    if explicit:
        path = REPO_ROOT / explicit
        if not path.exists():
            raise SystemExit(f"no such database: {path}")
        bad = count_mismatched(path)
        if bad:
            raise SystemExit(
                f"{path.name} has {bad} chunk(s) with 64-dim embeddings.\n"
                "One wrong-width row aborts every query, so this corpus cannot be\n"
                "scored at all. Point --db at a clean copy, or repair it:\n"
                "  python scripts/purge_bad_dimension_chunks.py   # dry run\n"
                "  HF_TOKEN=... in .env, then re-embed the corpus"
            )
        return path, "chosen explicitly, clean"

    for name in CANDIDATES:
        path = REPO_ROOT / name
        if not path.exists():
            continue
        if count_mismatched(path) == 0:
            note = "no short-embedding rows" if name != "rag.db" else "clean"
            return path, note

    raise SystemExit(
        "no usable corpus found.\n"
        f"  Looked for: {', '.join(CANDIDATES)}\n"
        "  rag.db holds 11 chunks with 64-dim embeddings; one wrong-width row\n"
        "  aborts every query, so it cannot be scored.\n\n"
        "To build a clean copy:\n"
        "  python scripts/purge_bad_dimension_chunks.py --db rag.db   # dry run first\n\n"
        "To fix it properly, set HF_TOKEN in .env, set EMBEDDING_PROVIDER=huggingface,\n"
        "and re-embed the whole corpus. The stored vectors came from a hash function,\n"
        "so none of them are usable with a real model -- see docs/architecture.md 13."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", help="database to evaluate against (default: auto)")
    args, passthrough = parser.parse_known_args()

    db, note = choose_db(args.db)

    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "DATABASE_URL": f"sqlite:///{db.name}",
        "OBJECT_STORE_DIR": "./data/objects",
        # Pinned rather than discovered: a discovery here would silently mask a
        # corpus built at a different width, which is the exact failure this
        # script exists to report clearly.
        "EMBEDDING_DIM": "384",
        "VECTOR_STORE": "sqlite",
        "ENVIRONMENT": "local",
        "LOG_LEVEL": "ERROR",
    }

    print(f"corpus: {db.name} ({note})")
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        docs = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
    finally:
        connection.close()
    print(f"  docs: {docs}  chunks: {chunks}")
    print(flush=True)  # before handing the terminal to the child process

    return subprocess.call(
        [sys.executable, "eval/run_eval.py", "--stage", "retrieval", *passthrough],
        cwd=REPO_ROOT,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
