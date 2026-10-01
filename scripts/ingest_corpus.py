# Corpus ingest helper (Phase 1 verification).
#
# Ingests a directory of documents and reports per-state counts. The stats block
# is the Phase 1 exit gate (implementation.md 3.5): documents=N live=N failed=0,
# with a chunk count consistent with the configured strategy.
#
# A silent drop in chunk count usually means the min-token filter is too
# aggressive, not that the documents are mostly noise. Check that before moving
# on.

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger
from app.db.models import Chunk, ChunkTerm, Document, DocumentState
from app.db.session import create_all, get_engine, session_scope
from app.ingest.chunk import ChunkConfig
from app.ingest.objectstore import LocalObjectStore
from app.ingest.worker import create_document, ingest_document
from app.providers.embedding import get_embedding_provider

log = get_logger("scripts.ingest")


def discover(directory: Path) -> list[Path]:
    """All candidate files under a directory, sorted for determinism."""
    skip = {"node_modules", ".git", ".venv", "__pycache__", "data"}
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file()
        and not any(part in skip for part in p.parts)
        and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".markdown"}
    ]
    return sorted(files)


def verify_invariants(session, total_chunks: int, with_vectors: int) -> list[str]:
    """Post-ingest checks. Returns a list of failures; empty means clean.

    These are the properties a stats block cannot show. A run can report
    `failed=0` and still have served no citations: chunks whose offsets do not
    reconstruct their text, a keyword index empty for live documents, or vectors
    of the wrong width. Each is a silent failure — the corpus looks healthy while
    retrieval returns nothing useful.
    """
    problems: list[str] = []

    # 1. Every chunk's stored text matches the slice of canonical text its offsets
    #    point at. This is the citation invariant.
    #
    #    `substr` is 1-indexed in SQLite, hence `char_start + 1`, while the
    #    offsets stored on the chunk are 0-indexed.
    mismatch_predicate = (
        func.substr(
            Document.cleaned_text,
            Chunk.char_start + 1,
            Chunk.char_end - Chunk.char_start,
        )
        != Chunk.text
    )
    mismatched = session.execute(
        select(func.count())
        .select_from(Chunk)
        .join(Document, Document.doc_id == Chunk.doc_id)
        .where(
            Document.state == DocumentState.LIVE,
            Document.cleaned_text.is_not(None),
            mismatch_predicate,
        )
    ).scalar_one()
    if mismatched:
        problems.append(
            f"{mismatched} live chunks have text that does not match their offsets"
        )

    # 2. Keyword rows exist for live documents. An empty keyword index means
    #    keyword search returns nothing while the corpus looks fine.
    live_without_terms = session.execute(
        select(func.count())
        .select_from(Document)
        .where(
            Document.state == DocumentState.LIVE,
            ~Document.chunks.any(),
        )
    ).scalar_one()
    if live_without_terms:
        problems.append(f"{live_without_terms} live documents have no chunks")

    term_states = {
        state.value
        for state in session.execute(select(ChunkTerm.state).distinct()).scalars()
    }
    if term_states and term_states != {DocumentState.LIVE.value}:
        problems.append(
            f"keyword rows recorded states {sorted(term_states)}; expected only "
            f"'{DocumentState.LIVE.value}' — the denormalised state has drifted"
        )

    # 3. Every live chunk carries an embedding of the configured width.
    if total_chunks != with_vectors:
        problems.append(
            f"{total_chunks - with_vectors} chunks have no embedding; retrieval "
            "would silently skip them"
        )

    # 4. No chunk points outside its document's cleaned text.
    out_of_range = session.execute(
        select(func.count())
        .select_from(Chunk)
        .join(Document, Document.doc_id == Chunk.doc_id)
        .where(
            Document.cleaned_text.is_not(None),
            func.length(Document.cleaned_text).is_not(None),
            Chunk.char_end > func.length(Document.cleaned_text),
        )
    ).scalar_one()
    if out_of_range:
        problems.append(f"{out_of_range} chunks extend past the end of their document")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a document corpus")
    parser.add_argument("--dir", default="./samples", help="directory to ingest")
    parser.add_argument("--stats", action="store_true", help="print per-state statistics")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of files")
    parser.add_argument("--target-tokens", type=int, default=1000)
    parser.add_argument("--overlap-tokens", type=int, default=200)
    parser.add_argument("--min-chunk-tokens", type=int, default=50)
    parser.add_argument(
        "--strategy", default="heading", choices=["heading", "paragraph", "fixed"]
    )
    args = parser.parse_args(argv)

    configure_logging("WARNING")
    settings = get_settings()
    create_all(get_engine(settings))

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2

    files = discover(directory)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"error: no supported documents found under {directory}", file=sys.stderr)
        return 2

    chunk_config = ChunkConfig(
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        strategy=args.strategy,  # type: ignore[arg-type]
    )
    store = LocalObjectStore.from_settings(settings)
    provider = get_embedding_provider(settings)

    succeeded = failed = 0
    with session_scope() as session:
        for path in files:
            data = path.read_bytes()
            try:
                doc = create_document(
                    session, filename=path.name, data=data, object_store=store
                )
                session.commit()
                outcome = ingest_document(
                    session,
                    doc.doc_id,
                    object_store=store,
                    provider=provider,
                    settings=settings,
                    chunk_config=chunk_config,
                )
                session.commit()
                succeeded += 1
                log.info(
                    "ingested",
                    extra={
                        "file": path.name,
                        "chunks": outcome.chunks,
                        "state": outcome.state.value,
                    },
                )
            except AppError as exc:
                session.commit()
                failed += 1
                print(f"  failed: {path.name}: {exc.user_message}", file=sys.stderr)
            except Exception as exc:
                session.commit()
                failed += 1
                print(f"  failed: {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.stats:
        with session_scope() as session:
            states = Counter(
                {
                    state.value: count
                    for state, count in session.execute(
                        select(Document.state, func.count()).group_by(Document.state)
                    ).all()
                }
            )
            total_chunks = session.execute(
                select(func.count()).select_from(Chunk)
            ).scalar_one()
            documents = session.execute(
                select(func.count())
                .select_from(Document)
                .where(Document.state != DocumentState.DELETED)
            ).scalar_one()
            with_vectors = session.execute(
                select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
            ).scalar_one()

            print()
            print("=" * 62)
            print("INGEST STATS")
            print("=" * 62)
            print(
                f"documents={documents} live={states.get('live', 0)} "
                f"failed={states.get('failed', 0)}"
            )
            print(f"chunks={total_chunks} with_vectors={with_vectors}")
            print(
                f"strategy={chunk_config.strategy} target={chunk_config.target_tokens} "
                f"overlap={chunk_config.overlap_tokens} min={chunk_config.min_chunk_tokens}"
            )
            if states:
                for state, count in sorted(states.items()):
                    print(f"  {state:<12} {count}")
            if total_chunks:
                print(f"avg_chunks_per_doc={total_chunks / max(documents, 1):.1f}")
            print("=" * 62)

            if states.get("failed", 0):
                print("\nfailed documents:")
                for _doc_id, filename, reason in session.execute(
                    select(Document.doc_id, Document.filename, Document.error_reason).where(
                        Document.state == DocumentState.FAILED
                    )
                ).all():
                    print(f"  {filename}: {reason}")

            problems = verify_invariants(session, total_chunks, with_vectors)
            print()
            if problems:
                print("INVARIANT CHECKS FAILED")
                for problem in problems:
                    print(f"  - {problem}")
                return 1
            print("INVARIANT CHECKS PASSED")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())