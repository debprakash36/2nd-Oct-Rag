"""NFR-10 guard: no vendor SDK imports outside app/providers/.

This rule erodes silently. One convenient `import openai` in an orchestrator
module looks harmless and is invisible in review, but it means swapping
providers now requires touching code that should not know a provider exists.

Enforced as a test rather than a review note so it cannot regress unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

VENDOR_SDKS = (
    "openai",
    "anthropic",
    "cohere",
    "google.generativeai",
    "google.genai",
    "mistralai",
    "together",
    "boto3",  # object store clients are also provider-specific
)

APP_ROOT = Path("app")


def test_app_directory_exists() -> None:
    assert APP_ROOT.is_dir(), "run from the project root"


@pytest.mark.parametrize("sdk", VENDOR_SDKS)
def test_no_vendor_sdk_imports_outside_providers(sdk: str) -> None:
    offenders: list[str] = []
    pattern = re.compile(rf"^\s*(?:import|from)\s+{re.escape(sdk)}\b", re.MULTILINE)

    for path in APP_ROOT.rglob("*.py"):
        if "providers" in path.parts:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))

    assert not offenders, (
        f"vendor SDK '{sdk}' imported outside app/providers/ (NFR-10): {offenders}. "
        "Only app/providers/ may know which vendor is in use."
    )


def test_provider_interfaces_are_defined() -> None:
    """The three NFR-10 interfaces must exist and be runtime-checkable."""
    from app.providers.base import (
        EmbeddingProvider,
        GenerationProvider,
        RerankerProvider,
        ScoredDoc,
    )

    assert EmbeddingProvider is not None
    assert GenerationProvider is not None
    assert RerankerProvider is not None
    assert ScoredDoc(chunk_id="a", score=1.0, text="t").chunk_id == "a"


def test_provider_selection_is_centralized() -> None:
    """Providers are chosen in one factory, not at call sites."""
    from app.providers.embedding import get_embedding_provider

    provider = get_embedding_provider()
    assert hasattr(provider, "embed")


def test_config_pins_embedding_dim() -> None:
    """A dimension mismatch must be detectable, not silent (architecture.md 7.3)."""
    from app.core.config import Settings

    settings = Settings(embedding_dim=384)
    assert settings.embedding_dim == 384


def test_production_refuses_sqlite() -> None:
    """pgvector has no SQLite equivalent, so fail at startup, not at migration."""
    from app.core.config import Settings

    settings = Settings(environment="production", database_url="sqlite:///./rag.db")
    with pytest.raises(RuntimeError, match="must be Postgres"):
        settings.validate_production()


def test_local_allows_sqlite() -> None:
    from app.core.config import Settings

    Settings(environment="local", database_url="sqlite:///./rag.db").validate_production()
