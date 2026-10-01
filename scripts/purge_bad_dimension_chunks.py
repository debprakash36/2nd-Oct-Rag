"""Dry-run: report 64-dim chunks and confirm a clean LIVE copy of each exists.

Deleting a row from a real corpus is not something to do on a hunch, so this only
reports. Nothing is written unless --apply is passed.

The check that matters is not "the row is corrupt" -- it obviously is -- but
"a healthy copy of the same content still exists". If it does not, deleting the
bad row destroys the only copy of that document and the corpus silently loses
content. That is the failure this guards against.

Usage:
    python scripts/purge_bad_dimension_chunks.py            # report
    python scripts/purge_bad_dimension_chunks.py --apply    # delete
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Bounded by JSON length, not parsed: a 64-wide vector serialises to well under
#: 3000 characters and a 384-wide one to well over, so this separates them without
#: paying to parse 300 vectors.
SHORT_EMBEDDING_MAX_CHARS = 3000


def find_bad_chunks(connection: sqlite3.Connection) -> list[dict[str, object]]:
    """Rows whose stored embedding width differs from the configured width."""
    rows = connection.execute(
        "SELECT c.chunk_id, c.doc_id, d.filename, d.state, d.content_hash, "
        "       length(c.embedding) AS embed_len, d.duplicate_of "
        "FROM chunks c "
        "JOIN documents d ON d.doc_id = c.doc_id "
        "WHERE c.embedding IS NOT NULL "
        f"  AND length(c.embedding) < {SHORT_EMBEDDING_MAX_CHARS} "
        "ORDER BY d.uploaded_at"
    ).fetchall()
    keys = (
        "chunk_id",
        "doc_id",
        "filename",
        "state",
        "content_hash",
        "embed_len",
        "duplicate_of",
    )
    return [dict(zip(keys, row, strict=True)) for row in rows]


def find_clean_copies(
    connection: sqlite3.Connection, content_hash: str, exclude_doc_id: str
) -> list[dict[str, object]]:
    """Healthy, retrievable documents carrying the same content."""
    rows = connection.execute(
        "SELECT d.doc_id, d.filename, d.state, d.content_hash, count(c.chunk_id) AS chunks, "
        "       min(length(c.embedding)) AS min_len, max(length(c.embedding)) AS max_len "
        "FROM documents d "
        "JOIN chunks c ON c.doc_id = d.doc_id "
        "WHERE d.content_hash = ? AND d.doc_id != ? "
        "  AND c.embedding IS NOT NULL "
        f"  AND length(c.embedding) >= {SHORT_EMBEDDING_MAX_CHARS} "
        "GROUP BY d.doc_id, d.filename, d.state, d.content_hash "
        "ORDER BY d.uploaded_at",
        (content_hash, exclude_doc_id),
    ).fetchall()
    keys = ("doc_id", "filename", "state", "content_hash", "chunks", "min_len", "max_len")
    return [dict(zip(keys, row, strict=True)) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "rag.db"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete (default is report-only)",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    connection.execute("PRAGMA foreign_keys=OFF")

    bad = find_bad_chunks(connection)
    if not bad:
        print("no short-embedding chunks found; nothing to do")
        return 0

    print(f"short-embedding chunks: {len(bad)}\n")

    # Which documents would lose their only copy?
    at_risk: list[dict[str, object]] = []
    for row in bad:
        content_hash = str(row["content_hash"])
        doc_id = str(row["doc_id"])
        clean = find_clean_copies(connection, content_hash, doc_id)
        shared = clean[0]["content_hash"] if clean else None
        healthy = [c for c in clean if str(c["state"]) == "LIVE"]
        print(
            f"  chunk {row['chunk_id']}  doc {row['doc_id']}  "
            f"{row['filename']}  state={row['state']}  "
            f"embed_len={row['embed_len']}  hash={content_hash[:12]}"
        )
        if healthy:
            print(
                f"      -> clean LIVE copy present: doc {healthy[0]['doc_id']} "
                f"({healthy[0]['chunks']} chunk(s), "
                f"len {healthy[0]['min_len']}-{healthy[0]['max_len']})"
            )
        else:
            print("      -> NO clean LIVE copy with this hash. DO NOT DELETE.")
            at_risk.append(row)
        del shared

    print()
    if at_risk:
        print(f"REFUSING: {len(at_risk)} row(s) have no healthy copy. Re-embed instead.")
        return 1

    if not args.apply:
        print("dry run only. Re-run with --apply to delete.")
        return 0

    doc_ids = sorted({str(r["doc_id"]) for r in bad})
    chunk_ids = [str(r["chunk_id"]) for r in bad]
    placeholders = ",".join("?" * len(chunk_ids))
    connection.execute(f"DELETE FROM chunk_terms WHERE chunk_id IN ({placeholders})", chunk_ids)
    deleted_terms = connection.total_changes
    connection.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
    deleted_chunks = connection.total_changes
    placeholders = ",".join("?" * len(doc_ids))
    connection.execute(f"DELETE FROM documents WHERE doc_id IN ({placeholders})", doc_ids)
    deleted_docs = connection.total_changes
    connection.commit()

    print(
        f"deleted {deleted_chunks} chunk(s), {deleted_terms} chunk_term(s), "
        f"{deleted_docs} document(s)"
    )
    remaining = connection.execute(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        "WHERE c.embedding IS NOT NULL "
        f"AND length(c.embedding) < {SHORT_EMBEDDING_MAX_CHARS}"
    ).fetchone()[0]
    print(f"short-embedding chunks remaining: {remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
