"""Retrieval package (architecture.md §3.3).

Phase 2 deliverable. Entry point is `Retriever`, which composes:

    rewrite -> [vector ‖ keyword] -> RRF fusion -> dedupe -> rerank -> threshold -> budget

The store implementations behind the two search halves are selected by
`build_vector_store` and `build_keyword_index`; nothing outside this package
should import a concrete store.

`build_vector_store` returns one of three implementations -- `PgVectorStore`,
`ChromaVectorStore`, or `SqliteVectorStore` -- chosen by the `vector_store`
setting (default `auto`, the original dialect-derived behaviour). The Chroma
backend is populated by `sync_chroma_index` in this module (also exposed as
`scripts/sync_chroma_index.py`); ingestion does not write to it.
"""
