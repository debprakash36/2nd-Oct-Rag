"""Do the checked-in artifacts agree? Opt-in; run with `pytest -m indexcheck`.

Deselected by default because its answer depends on the working tree, not on the
code: it reads the committed `rag.db` and the `data/chroma` projection sitting
beside it. A developer who has re-ingested the corpus will get a different result
from a developer who has not, and neither of them is running a code change.

That is also precisely why it is worth having. Before `measure_divergence` existed,
nothing in the system could state this fact. The backend was inferred, the
population on disk was invisible to every check, and a derived index that was 7
chunks behind the authoritative table looked exactly like a healthy one.

**This test fails on a tree where the index has not been re-projected.** That is the
intended behaviour, not a broken test: it is reporting that
`scripts/sync_chroma_index.py` has not been run since the last ingest. Re-project
(`make chroma-sync`) or accept the lag deliberately.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO_ROOT / "rag.db"
CHROMA_DIR = REPO_ROOT / "data" / "chroma"

pytestmark = pytest.mark.indexcheck


def _sql_live_chunks(db: Path) -> set[str]:
    """Live chunk ids in the authoritative table, mirroring `live_chunk_count`."""
    connection = sqlite3.connect(db)
    try:
        rows = connection.execute(
            "SELECT c.chunk_id FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE d.state = 'LIVE' AND c.embedding IS NOT NULL"
        ).fetchall()
    finally:
        connection.close()
    return {row[0] for row in rows}


def _chroma_ids(chroma_dir: Path) -> set[str]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name="chunks", metadata={"hnsw:space": "cosine"}
    )
    return set(collection.get(include=[])["ids"])


def test_sql_is_a_superset_of_the_derived_index():
    """Chroma must never claim a vector SQL does not have.

    The reverse direction is the dangerous one. A stale id in the derived index
    occupies an ANN result slot forever and drags recall down, and `sync_chroma_index`
    only removes ids it saw in this run -- so an id deleted from SQL since the last
    sync survives until the next full re-projection.
    """
    if not DB_PATH.exists():
        pytest.skip(f"no {DB_PATH.name} in the working tree")

    sql_ids = _sql_live_chunks(DB_PATH)
    if not CHROMA_DIR.exists():
        pytest.skip("no data/chroma; nothing has been projected")

    chroma_ids = _chroma_ids(CHROMA_DIR)

    orphans = chroma_ids - sql_ids
    assert not orphans, (
        f"{len(orphans)} vector(s) in data/chroma have no live row in {DB_PATH.name}. "
        f"They will keep consuming ANN result slots. Re-project: make chroma-sync"
    )


def test_the_derived_index_is_current():
    """Report the gap between the two stores, and fail when there is one.

    Expected to fail on a tree where documents were ingested after the last
    `sync_chroma_index` run -- that lag is real and is exactly what this test exists
    to make visible rather than absorb.
    """
    if not DB_PATH.exists():
        pytest.skip(f"no {DB_PATH.name} in the working tree")
    if not CHROMA_DIR.exists():
        pytest.skip("no data/chroma; nothing has been projected")

    sql_ids = _sql_live_chunks(DB_PATH)
    chroma_ids = _chroma_ids(CHROMA_DIR)

    missing = sql_ids - chroma_ids
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise AssertionError(
            f"derived index is behind the authoritative chunks: "
            f"{len(chroma_ids)}/{len(sql_ids)} live chunks, {len(missing)} missing.\n"
            f"Ingestion writes SQL and never the index, so a corpus change always\n"
            f"leaves the projection behind until it is re-run. Re-project with:\n"
            f"    make chroma-sync\n"
            f"Missing (first 5 of {len(missing)}): {sample}"
        )
