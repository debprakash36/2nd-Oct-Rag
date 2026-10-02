"""Launch checker: local sqlite is allowed; a public sqlite config is not."""

from __future__ import annotations

from app.core.config import Settings
from app.core.launch import assess, overall_ok


def test_local_sqlite_with_a_token_is_a_demo_not_a_failure() -> None:
    settings = Settings(
        environment="local",
        database_url="sqlite:///./rag.db",
        vector_store="sqlite",
        embedding_provider="fake",
        generation_provider="fake",
        api_token="x",
        _env_file=None,
    )
    checks = assess(settings, session=None)
    assert overall_ok(checks, allow_local=True)
    assert all(c.state != "fail" for c in checks)
    blocked = next(c for c in checks if c.name == "Phase 6 traffic gate")
    assert blocked.state == "blocked"


def test_production_sqlite_fails() -> None:
    settings = Settings(
        environment="production",
        database_url="sqlite:///./rag.db",
        vector_store="sqlite",
        embedding_provider="huggingface",
        generation_provider="groq",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        api_token="x",
        _env_file=None,
    )
    checks = assess(settings, session=None)
    assert any(c.state == "fail" and "Postgres" in c.detail for c in checks)
    assert not overall_ok(checks, allow_local=False)


def test_production_localhost_cors_fails() -> None:
    settings = Settings(
        environment="production",
        database_url="postgresql+psycopg://rag:rag@localhost:5432/rag",
        vector_store="pgvector",
        embedding_provider="huggingface",
        generation_provider="groq",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        api_token="x",
        cors_allow_origins="http://localhost:3000",
        _env_file=None,
    )
    checks = assess(settings, session=None)
    cors = next(c for c in checks if c.name == "CORS")
    assert cors.state == "fail"
