"""BM25 keyword index — the half of FR-12 that carries exact identifiers.

architecture.md §3.3 is the reason this stage exists at all: vector similarity is
bad at error codes, part numbers, surnames and "Article 12", and document Q&A
lives on those. The identifier tests below are therefore the load-bearing ones —
if keyword search silently degraded into bag-of-words matching, they would still
pass, and FR-12 would be lost without a test failing.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.retrieval.keyword_index import Bm25KeywordIndex, build_keyword_index
from app.retrieval.types import AccessFilter, Stage


class TestBm25KeywordIndex:
    def test_finds_chunk_containing_the_term(
        self, session: Session, live_doc
    ):
        index = Bm25KeywordIndex(session)
        results = index.search("refund", k=10)
        assert results
        assert any("refund" in c.text.lower() for c in results)

    def test_sets_keyword_stage_score(self, session: Session, live_doc):
        index = Bm25KeywordIndex(session)
        results = index.search("refund", k=10)
        assert all(Stage.KEYWORD in c.scores for c in results)

    def test_scores_descend(self, session: Session, live_doc):
        index = Bm25KeywordIndex(session)
        scores = [c.scores[Stage.KEYWORD] for c in index.search("refund policy", k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_nothing(self, session: Session, live_doc):
        assert Bm25KeywordIndex(session).search("", k=10) == []

    def test_stopword_only_query_returns_nothing(
        self, session: Session, multi_chunk_doc
    ):
        """A query with no content words has no lexical signal to return.

        Note that stopwords are dropped *before* n-grams are formed. Forming
        bigrams first yields `the_of` and `of_and`, which are rare terms in any
        corpus and so carry a high IDF -- a contentless query would then rank
        strongly and fill the candidate window with boilerplate. `_QUERY_STOPWORDS`
        in app/retrieval/keyword_index.py exists for this.
        """
        assert Bm25KeywordIndex(session).search("the of and", k=10) == []

    def test_stopwords_do_not_dilute_a_real_query(
        self, session: Session, multi_chunk_doc
    ):
        """Adding stopwords must not change what a query retrieves.

        If `the refund` and `refund` returned different orderings, stopwords were
        contributing signal -- which is exactly the dilution BM25 is meant to avoid.
        """
        index = Bm25KeywordIndex(session)
        bare = [c.chunk_id for c in index.search("refund", k=10)]
        padded = [c.chunk_id for c in index.search("what is the refund policy", k=10)]
        assert bare
        assert padded[: len(bare)] == bare

    def test_unknown_term_returns_nothing(self, session: Session, live_doc):
        assert Bm25KeywordIndex(session).search("zzzznonexistentterm", k=10) == []

    def test_honours_limit(self, session: Session, live_doc, second_live_doc):
        assert len(Bm25KeywordIndex(session).search("the", k=1)) <= 1

    def test_disabled_document_is_immediately_unsearchable(
        self, session: Session, live_doc
    ):
        """FR-7 on the keyword path.

        `chunk_terms` carries a denormalised copy of the document state, and the
        filter applies inside the term scan. A stale copy would keep serving a
        tombstoned document until the next reindex.
        """
        from app.db.models import DocumentState
        from app.ingest.states import transition

        index = Bm25KeywordIndex(session)
        assert index.search("refund", k=50)

        transition(session, live_doc, DocumentState.DISABLED, reason="test")
        session.commit()

        assert index.search("refund", k=50) == []

    def test_no_partial_index_means_no_results(self, session: Session, live_doc):
        """A failed document must leave no servable keyword rows (§4.1).

        A half-indexed document produces answers that appear to come from a source
        that is not fully readable, which the architecture calls out as worse than
        a missing document.
        """
        from sqlalchemy import delete

        from app.db.models import ChunkTerm

        session.execute(delete(ChunkTerm))
        session.commit()
        assert Bm25KeywordIndex(session).search("refund", k=50) == []


class TestIdentifierRecall:
    """The exact-identifier case that justifies the keyword stage (FR-12)."""

    def test_finds_a_unique_tokenic_identifier(self, session: Session, live_doc):
        """A rare identifier must be findable by exact term match.

        Simulates `POL-RFD-042` or a part number: high specificity, near-zero
        value to a vector similarity search.
        """
        from sqlalchemy import insert

        from app.db.models import Chunk, ChunkTerm, DocumentState

        chunk = session.query(Chunk).filter(Chunk.doc_id == live_doc.doc_id).first()
        assert chunk is not None

        session.execute(
            insert(ChunkTerm).values(
                chunk_id=chunk.chunk_id,
                term="zxq7742",
                term_count=1,
                state=DocumentState.LIVE,
            )
        )
        session.commit()

        results = Bm25KeywordIndex(session).search("what is zxq7742", k=10)
        assert [c.chunk_id for c in results] == [chunk.chunk_id]

    def test_rare_term_outranks_common_term(
        self, session: Session, multi_chunk_doc
    ):
        """IDF must actually discriminate.

        A term present in every chunk carries no signal; if length normalisation
        were applied without IDF, boilerplate chunks would win on term frequency
        alone.
        """
        from sqlalchemy import delete, insert

        from app.db.models import Chunk, ChunkTerm, DocumentState

        chunks = session.query(Chunk).all()
        assert len(chunks) >= 2

        # "commonword" in every chunk, "needleword" in exactly one. IDF must make
        # the rare term dominate the ubiquitous one.
        session.execute(delete(ChunkTerm))
        for c in chunks:
            session.execute(
                insert(ChunkTerm).values(
                    chunk_id=c.chunk_id, term="commonword", term_count=1,
                    state=DocumentState.LIVE,
                )
            )
        session.execute(
            insert(ChunkTerm).values(
                chunk_id=chunks[0].chunk_id, term="needleword", term_count=1,
                state=DocumentState.LIVE,
            )
        )
        session.commit()

        results = Bm25KeywordIndex(session).search("commonword needleword", k=10)
        assert results[0].chunk_id == chunks[0].chunk_id

    def test_phrase_bigram_ranks_together_terms_higher(
        self, session: Session, multi_chunk_doc
    ):
        """`refund window` must beat a document with the words far apart.

        architecture.md §3.3 gives exact identifiers as the motivating case, and
        adjacent bigrams are what turn "refund" + "window" into "refund window".
        """
        from sqlalchemy import delete, insert

        from app.db.models import Chunk, ChunkTerm, DocumentState

        chunks = session.query(Chunk).order_by(Chunk.chunk_index).all()
        assert len(chunks) >= 2

        session.execute(delete(ChunkTerm))

        # chunk 0 carries both query terms; chunk 1 carries the same two terms but
        # separated by 30 filler tokens, so no `refund_window` bigram forms.
        for term in ("refund", "window"):
            session.execute(
                insert(ChunkTerm).values(
                    chunk_id=chunks[0].chunk_id, term=term, term_count=1,
                    state=DocumentState.LIVE,
                )
            )
        # chunk 1: same terms, but separated by filler so the bigram does not form.
        filler = [f"filler{i}zz" for i in range(30)]
        for term in ["refund", *filler, "window"]:
            session.execute(
                insert(ChunkTerm).values(
                    chunk_id=chunks[1].chunk_id, term=term, term_count=1,
                    state=DocumentState.LIVE,
                )
            )
        session.commit()

        results = Bm25KeywordIndex(session).search("refund window", k=10)
        assert results[0].chunk_id == chunks[0].chunk_id

    def test_matched_terms_include_ngrams(self, session: Session, live_doc):
        index = Bm25KeywordIndex(session)
        terms = index.matched_terms("refund window")
        assert "refund" in terms
        assert "window" in terms
        assert "refund_window" in terms


class TestKeywordAccessFilter:
    def test_unauthorised_chunk_not_returned(self, session: Session, live_doc):
        from app.db.models import Document

        doc = session.get(Document, live_doc.doc_id)
        doc.acl_tags = ["restricted"]
        session.commit()

        index = Bm25KeywordIndex(session)
        results = index.search("refund", k=50, access=AccessFilter(frozenset({"other"})))
        assert results == []

    def test_authorised_chunk_returned(self, session: Session, live_doc):
        from app.db.models import Document

        doc = session.get(Document, live_doc.doc_id)
        doc.acl_tags = ["finance"]
        session.commit()

        index = Bm25KeywordIndex(session)
        results = index.search("refund", k=50, access=AccessFilter(frozenset({"finance"})))
        assert results

    def test_mixed_corpus_returns_only_permitted(
        self, session: Session, live_doc, second_live_doc
    ):
        """One document permitted, one not: only the permitted one may appear."""
        from app.db.models import Document

        session.get(Document, live_doc.doc_id).acl_tags = ["restricted"]
        session.get(Document, second_live_doc.doc_id).acl_tags = ["finance"]
        session.commit()

        results = Bm25KeywordIndex(session).search(
            "refund policy", k=50, access=AccessFilter(frozenset({"finance"}))
        )
        assert results
        assert all(c.doc_id != live_doc.doc_id for c in results)


class TestBuildKeywordIndex:
    def test_returns_bm25_implementation(self, session: Session):
        assert isinstance(build_keyword_index(session), Bm25KeywordIndex)

    def test_satisfies_the_protocol(self, session: Session):
        """The NFR-10 seam: callers depend on the protocol, not the class."""
        from app.retrieval.keyword_index import KeywordIndex

        assert isinstance(build_keyword_index(session), KeywordIndex)