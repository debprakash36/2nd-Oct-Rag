"""Provider interfaces — the NFR-10 portability seam.

These protocols are the only place vendor SDKs are permitted to appear
(`app/providers/`, enforced by tests/test_no_vendor_leak.py). Every consumer
depends on the interface, so swapping embedding or generation vendors is a
config change plus one new implementation, not a refactor.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ScoredDoc:
    """A candidate passage with a relevance score.

    `chunk_id` is an internal identifier and must never reach a model prompt or
    an API response (architecture.md 3.5).
    """

    chunk_id: str
    score: float
    text: str


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors."""

    def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        """Batch-embed `texts`, preserving input order in the output.

        Order preservation is a contract, not an implementation detail: the
        caller zips results back onto chunk rows positionally, and a reordered
        response would silently mislabel every vector in the index.
        """
        ...


@runtime_checkable
class GenerationProvider(Protocol):
    """Streams a chat completion."""

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield text deltas as they are produced.

        Must be lazy. A provider that buffers the full response before yielding
        the first delta breaks the TTFT budget (NFR-1), which is why the
        streaming requirement is a documented contract rather than a preference.
        """
        ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Scores query/document pairs more accurately than bi-encoder similarity."""

    def rerank(self, query: str, docs: Sequence[str], *, top_k: int) -> list[ScoredDoc]:
        """Return the top_k docs, highest score first."""
        ...
