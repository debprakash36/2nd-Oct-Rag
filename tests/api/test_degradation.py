"""Store-down degradation (NFR-2, architecture.md §8).

The correctness requirement this file exists to protect: **a store being down must not
look like the corpus having nothing to say.** A degraded-but-plausible answer is the
failure mode the whole project is built to avoid, so the distinction gets its own
tests rather than being inferred from the retriever's behaviour.

Three outcomes that must stay distinguishable:

    store reachable, no matches  ->  200, refusal      ("we do not have that")
    store unreachable            ->  503, unavailable  ("come back later")
    one of two stages down       ->  503, unavailable  (never answer from half an index)
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.errors import StoreUnavailableError
from app.db.session import session_scope
from app.retrieval.retriever import Retriever


def _broken_stage(monkeypatch, stage: str) -> None:
    """Make one search stage fail the way an unreachable store would.

    Always through `monkeypatch`: a bare `setattr` on the class outlives the test and
    silently poisons every later one, which reads as a mysterious unrelated failure.
    """
    monkeypatch.setattr(
        Retriever,
        f"_{stage}_candidates",
        lambda self, q, a, s: (_ for _ in ()).throw(
            OperationalError("SELECT 1", {}, Exception("connection refused"))
        ),
    )


@pytest.fixture
def no_limiter() -> Iterator[None]:
    """Keep the process-wide rate limiter out of unrelated tests.

    FR-31's limit is per client and the test client has one address, so a suite making
    more chat calls than the limit would start failing with 429 for no related reason.
    """
    from app.core.guardrails import reset_rate_limiter

    reset_rate_limiter()
    yield
    reset_rate_limiter()


class TestStoreDownIsNotARefusal:
    def test_unreachable_store_raises_unavailable_not_empty_result(
        self, live_doc, session: Session, settings_env, monkeypatch
    ):
        """The store failing is not the corpus being empty.

        Simulated at the store's `search`, which is where an unreachable Postgres
        actually surfaces, rather than by wrapping the session: the retriever opens its
        own short-lived sessions per stage, so a wrapper on the injected session would
        never be reached.
        """
        from app.retrieval import vector_store as vector_store_module

        def unavailable(*_args, **_kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

        monkeypatch.setattr(vector_store_module.SqliteVectorStore, "search", unavailable)
        retriever = Retriever(session, settings=settings_env)
        with pytest.raises(StoreUnavailableError) as exc:
            retriever.retrieve("refund window digital product")
        assert exc.value.status_code == 503

    def test_error_names_the_component_but_hides_it_from_the_user(self):
        exc = StoreUnavailableError("vector")
        # The component name is safe to log...
        assert exc.component == "vector"
        # ...but the user must not learn which infrastructure failed (NFR-5).
        assert "vector" not in exc.user_message
        assert "temporarily unavailable" in exc.user_message

    def test_one_down_stage_fails_the_whole_query(
        self, live_doc, session: Session, settings_env, monkeypatch
    ):
        """Half an index must never produce an answer.

        If the vector store is down but the keyword index answers, that answer would
        look like a successful retrieval from a corpus that merely happens to match
        lexically. It is not: an unknown number of passages were never considered, so
        relevance cannot be claimed. Both stages are required, not either.
        """
        _broken_stage(monkeypatch, "vector")
        retriever = Retriever(session, settings=settings_env)
        with pytest.raises(StoreUnavailableError):
            retriever.retrieve("refund window digital product")

    def test_the_other_stage_down_also_fails(
        self, live_doc, session: Session, settings_env, monkeypatch
    ):
        _broken_stage(monkeypatch, "keyword")
        retriever = Retriever(session, settings=settings_env)
        with pytest.raises(StoreUnavailableError):
            retriever.retrieve("refund window digital product")


class TestChatEndpointDegradation:
    def test_unavailable_store_returns_503_with_a_safe_message(
        self, client: TestClient, live_doc, monkeypatch, no_limiter
    ):
        """A 503 saying "unavailable", distinct from a 200 refusal."""
        monkeypatch.setattr(
            Retriever,
            "retrieve",
            lambda self, q, **kw: (_ for _ in ()).throw(StoreUnavailableError("vector")),
        )

        response = client.post("/chat/stream", json={"message": "refund window"})

        assert response.status_code == 503
        assert "temporarily unavailable" in response.text
        # No infrastructure detail crosses the boundary (NFR-5).
        assert "connection refused" not in response.text
        assert "Traceback" not in response.text

    def test_a_genuine_refusal_is_still_a_200(
        self, client: TestClient, live_doc, settings_env, monkeypatch, no_limiter
    ):
        """The control case: abstention must stay a refusal.

        Without this, a "fix" that turned every abstention into a 503 would pass the
        tests above while breaking the product.
        """
        monkeypatch.setattr(settings_env, "retrieval_threshold", 1.1, raising=False)

        response = client.post("/chat/stream", json={"message": "unrelated question"})

        assert response.status_code == 200
        assert "event: done" in response.text
        assert "temporarily unavailable" not in response.text

    def test_unavailable_retrieval_is_not_recorded_as_a_refusal(
        self, client: TestClient, live_doc, monkeypatch, no_limiter
    ):
        """A failed retrieval must not land in QueryLog as an abstention.

        If it did, the refusal rate (PRD §8, healthy band 10-30%) would silently absorb
        every outage: during an incident the dashboard would show refusals climbing
        rather than errors climbing, pointing the investigation at the wrong thing.
        """
        monkeypatch.setattr(
            Retriever,
            "retrieve",
            lambda self, q, **kw: (_ for _ in ()).throw(StoreUnavailableError("vector")),
        )

        client.post("/chat/stream", json={"message": "refund window during outage"})

        with session_scope() as s:
            recorded = s.execute(
                text("SELECT count(*) FROM query_logs WHERE original_query = :q"),
                {"q": "refund window during outage"},
            ).scalar_one()
        assert recorded == 0

    def test_conversation_is_left_consistent_after_an_outage(
        self, client: TestClient, live_doc, monkeypatch, no_limiter
    ):
        """The question is still recorded; no half-written answer is.

        The user turn is written before retrieval precisely so an outage leaves the
        question visible. Asserting the assistant turn is absent guards against an
        empty placeholder being written for an answer that was never generated.
        """
        from app.db.models import Turn, TurnRole

        monkeypatch.setattr(
            Retriever,
            "retrieve",
            lambda self, q, **kw: (_ for _ in ()).throw(StoreUnavailableError("vector")),
        )
        cid = client.post("/conversations").json()["conversation_id"]

        client.post(
            "/chat/stream",
            json={"message": "refund window", "conversation_id": cid},
        )

        detail = client.get(f"/conversations/{cid}").json()
        assert [t["role"] for t in detail["turns"]] == ["user"]
        assert detail["turns"][0]["content"] == "refund window"
        with session_scope() as s:
            assert (
                s.query(Turn).filter_by(conversation_id=cid, role=TurnRole.ASSISTANT).count()
                == 0
            )


class TestHealthReportsStores:
    def test_health_reports_both_retrieval_stores(self, client: TestClient, live_doc):
        body = client.get("/health").json()
        assert body["checks"]["vector_store"] == "ok"
        assert body["checks"]["keyword_index"] == "ok"

    def test_health_is_503_when_a_retrieval_store_is_unreadable(
        self, client: TestClient, live_doc, monkeypatch
    ):
        """A bare `SELECT 1` can succeed while the chunk tables do not.

        This is the failure the store checks exist for: the database answers liveness,
        so a health check that only pings SQL reports a service that cannot answer a
        single question.
        """
        import app.api.health as health_module

        real = health_module._check_retrieval_stores

        def partial(session, settings):
            checks, _usable = real(session, settings)
            checks["vector_store"] = "error: OperationalError"
            return checks, False

        monkeypatch.setattr(health_module, "_check_retrieval_stores", partial)

        response = client.get("/health")

        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_health_error_does_not_leak_the_dsn(
        self, client: TestClient, live_doc, monkeypatch
    ):
        """A connection failure's message can contain the connection string."""
        import app.api.health as health_module

        monkeypatch.setattr(
            health_module,
            "_check_retrieval_stores",
            lambda session, settings: ({"vector_store": "error: OperationalError"}, False),
        )

        body = client.get("/health").text

        assert "postgresql://" not in body
        assert "password" not in body.lower()

    def test_health_still_reports_environment_and_dim(self, client: TestClient, live_doc):
        body = client.get("/health").json()
        assert body["checks"]["environment"] == "test"
        assert body["checks"]["embedding_dim"] > 0