# Dump every chunk and its stored embedding to a human-readable text file.
#
# Inspection aid, not part of the pipeline. Chunks are read from the authoritative
# chunk store (`Chunk`), joined to `Document` for the filename and state, and
# written in document order. The embedding column holds the full vector that was
# written at ingest time (JSON on SQLite, `vector(N)` on Postgres), so what is
# shown is exactly what the vector search compares against.
#
# A 384-dimension vector printed in full is long. `--preview N` shows only the
# first N components per chunk, which keeps the file skimmable when the goal is
# to see the shape rather than every float.

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import session_scope


def _fmt_vector(values: list[float], preview: int) -> str:
    if not values:
        return "(none)"
    shown = values if preview <= 0 else values[:preview]
    body = ", ".join(f"{v:.6g}" for v in shown)
    if preview > 0 and len(values) > preview:
        body += f", ... ({len(values) - preview} more)"
    return f"[{body}]"


def _norm(values: list[float]) -> float:
    return sum(v * v for v in values) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump chunks and embeddings to text")
    parser.add_argument("--out", default="data/chunks_and_embeddings.txt")
    parser.add_argument("--preview", type=int, default=0, help="components per vector; 0 = all")
    parser.add_argument("--limit", type=int, default=0, help="max chunks; 0 = all")
    parser.add_argument("--doc", default=None, help="only chunks whose filename contains this")
    args = parser.parse_args()

    settings = get_settings()
    stmt = (
        select(Chunk, Document)
        .join(Document, Chunk.doc_id == Document.doc_id)
        .order_by(Document.filename, Chunk.chunk_index)
    )
    if args.doc:
        stmt = stmt.where(Document.filename.contains(args.doc))
    if args.limit:
        stmt = stmt.limit(args.limit)

    with session_scope() as session:
        rows = list(session.execute(stmt).all())
        doc_count = session.execute(select(func.count(Document.doc_id))).scalar_one()
        chunk_count = session.execute(select(func.count(Chunk.chunk_id))).scalar_one()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("Chunks and embeddings dump")
    lines.append("=" * 78)
    lines.append(f"generated_at    : {dt.datetime.now(dt.UTC).isoformat()}")
    lines.append(f"database_url    : {settings.database_url}")
    lines.append(f"embedding_model : {settings.embedding_model}")
    lines.append(f"embedding_dim   : {settings.embedding_dim}")
    lines.append(f"documents       : {doc_count}")
    lines.append(f"chunks          : {chunk_count}")
    lines.append(f"chunks in file  : {len(rows)}")
    if args.preview:
        lines.append(f"vector preview  : first {args.preview} components")
    lines.append("")

    for chunk, document in rows:
        embedding = list(chunk.embedding) if chunk.embedding else []
        lines.append("=" * 78)
        lines.append(f"chunk_id        : {chunk.chunk_id}")
        lines.append(f"doc_id          : {chunk.doc_id}")
        lines.append(f"filename        : {document.filename}")
        lines.append(f"state           : {document.state.value}")
        lines.append(f"mime_type       : {document.mime_type}")
        lines.append(f"chunk_index     : {chunk.chunk_index}")
        lines.append(f"page            : {chunk.page}")
        lines.append(f"breadcrumb      : {chunk.breadcrumb}")
        lines.append(f"section_path    : {' > '.join(chunk.section_path)}")
        lines.append(f"token_count     : {chunk.token_count}")
        lines.append(f"char_range      : {chunk.char_start}..{chunk.char_end}")
        lines.append(f"embedding_model : {chunk.embedding_model}")
        lines.append(f"embedding_dim   : {len(embedding)}")
        if embedding:
            lines.append(f"embedding_norm  : {_norm(embedding):.6f}")
        lines.append("text:")
        lines.append(chunk.text)
        lines.append("embedding:")
        lines.append(_fmt_vector(embedding, args.preview))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} chunk(s) to {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
