"""Provider implementations and the factory that selects them.

`get_embedding_provider()` is the single place a concrete provider is chosen.
Keeping the choice in one function is what makes NFR-10 real: adding a vendor
means adding a branch here, and nothing else in the codebase changes.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections.abc import Callable, Sequence
from functools import lru_cache

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import EmbeddingError
from app.core.logging import get_logger
from app.providers.base import EmbeddingProvider, RerankerProvider, ScoredDoc

log = get_logger("app.providers")

#: Total attempts per batch, including the first. Bounded because this sits inside
#: a user-facing request: four attempts at a 0.5/1/2 s backoff is ~3.5 s of added
#: latency in the worst case, which the 5 s TTFT budget (NFR-1) can absorb, while
#: an unbounded retry would turn a provider outage into a hung request.
DEFAULT_MAX_ATTEMPTS = 4

#: First backoff step, doubled each attempt and capped. The cap matters because
#: the exponent is applied per attempt: without it a larger attempt count walks
#: into a multi-minute stall.
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0

#: Statuses worth retrying. 429 is the hosted API's rate limit and the reason this
#: exists at all -- a sweep is ~1400 sequential calls and trips it. 5xx is a
#: provider-side fault that a retry usually clears.
#:
#: Every other 4xx is deliberately excluded. A 401 or 403 is a bad token and a 404
#: a bad model name: both return identically on every retry, so retrying them only
#: delays the real diagnosis by three backoffs. Failing fast keeps the message --
#: "check the model name and HF_TOKEN" -- the first thing an operator sees.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class FakeEmbeddingProvider:
    """Deterministic hash-based embeddings for tests and local runs.

    Vectors carry a small amount of real signal: tokens are hashed into buckets
    and the vector is L2-normalized, so texts sharing tokens score higher than
    texts that do not. That is enough for retrieval tests to be meaningful
    without a network call, and it is fully deterministic across runs so
    idempotency tests are not flaky.

    It is not semantic. Phase 2 replaces it with a real provider before any
    quality claim is made.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        vectors = [self._embed_one(text) for text in texts]
        log.debug("embedded batch", extra={"count": len(vectors), "model": model, "dim": self._dim})
        return vectors

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            # Two buckets per token so co-occurrence is representable.
            for offset in (0, 8):
                idx = int.from_bytes(digest[offset : offset + 4], "big") % self._dim
                sign = 1.0 if digest[offset + 4] % 2 == 0 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Empty or stopword-only text. A zero vector has no direction and
            # breaks cosine similarity, so fall back to a fixed unit vector.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]


class HuggingFaceEmbeddingProvider:
    """Embeddings from the HuggingFace Inference API feature-extraction pipeline.

    The hosted API rather than local `sentence-transformers`. That choice trades a
    network round-trip on every embed for avoiding a ~2.5 GB torch install, which
    is a reasonable deal for a 113-document corpus but is a real cost: embedding
    runs on every query, so this latency lands inside the 5 s TTFT budget (NFR-1)
    on the request path as well as during ingestion.

    `huggingface_hub` is not a dependency. The endpoint is a single HTTP POST and
    the response is JSON, so `httpx` is the whole client. Adding an SDK here would
    have bought nothing but a large install and another thing to keep current.

    **Vectors are L2-normalized before they are returned.** The stores assume
    normalized input -- `PgVectorStore` documents that embeddings are L2-normalized
    and uses cosine distance to match -- and the API's `normalize` option is
    honoured server-side but re-normalized here anyway. Two reasons: the flag is a
    server default rather than a contract, and a corpus indexed with unnormalized
    vectors silently changes every similarity score if it is ever dropped.
    """

    def __init__(
        self,
        *,
        token: str,
        base_url: str,
        model: str,
        dim: int,
        timeout: float,
        batch_size: int,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._token = token
        # The route is `/{models}/{model}/pipeline/feature-extraction` -- note the
        # `/models` segment, which is easy to drop and produces a 400 that looks
        # like a bad model name or a bad token rather than a malformed URL. Verified
        # against the live API: with `/models` omitted this 400s, and with it the same
        # request returns 200 and a 384-dimension vector.
        #
        # `base_url` stays a plain host so callers cannot silently reintroduce the
        # bug by passing an endpoint that already names a model.
        self._url = (
            f"{base_url.rstrip('/')}/models/{model}/pipeline/feature-extraction"
        )
        self._model = model
        self._dim = dim
        self._timeout = timeout
        self._batch_size = batch_size
        self._max_attempts = max(1, max_attempts)
        # Injected rather than called directly so tests can assert on the backoff
        # schedule without actually sleeping through it.
        self._sleep = sleep
        self._jitter = jitter
        # Module-level `httpx` so tests can swap this single seam rather than
        # patching the library globally. The default is the real `httpx.post`; a
        # test passes a stub. Injected rather than monkeypatched because the
        # alternative reaches into the library from outside and is easy to leave
        # patched for the rest of the session.
        self._post = httpx.post

    def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(self._embed_batch(batch))
        log.debug(
            "embedded batch",
            extra={"count": len(vectors), "model": model or self._model, "dim": self._dim},
        )
        return vectors

    def _backoff(self, attempt: int) -> float:
        """Seconds to wait before `attempt`+1, jittered.

        Equal jitter: half the ceiling is fixed and the other half is random, so
        the delay never collapses toward zero (which would defeat the backoff) and
        never exceeds the ceiling. Uncorrelated retries matter because the failure
        mode here is a provider-wide rate limit -- several callers backing off in
        lockstep arrive together and trip it again.
        """
        ceiling = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
        return ceiling * (0.5 + 0.5 * self._jitter())

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Post one batch, retrying transient failures.

        Retryable: connection and timeout errors (`httpx.TransportError` covers
        ConnectError, ReadTimeout, ConnectTimeout, ReadError and friends) plus the
        statuses in `RETRYABLE_STATUS_CODES`. Everything else -- a 401, a 404, a
        malformed body -- raises on the first attempt, because repeating a
        deterministic failure only delays the diagnosis.
        """
        last_error = ""
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._post(
                    self._url,
                    json={"inputs": batch, "options": {"normalize": True}},
                    headers=self._headers(),
                    timeout=self._timeout,
                )
            except httpx.TransportError as exc:
                # Type only: the exception message can carry the URL, and a request
                # header echoed into a log is how a token ends up somewhere it should
                # not (NFR-5).
                last_error = f"huggingface request failed: {type(exc).__name__}"
                retryable = True
            else:
                if response.status_code == 200:
                    return self._parse(response.json(), len(batch))
                # The body can echo the request, so the status alone is reported.
                last_error = (
                    f"huggingface returned HTTP {response.status_code} for "
                    f"{self._model}; check the model name and HF_TOKEN"
                )
                retryable = response.status_code in RETRYABLE_STATUS_CODES

            if not retryable:
                raise EmbeddingError(last_error)
            if attempt == self._max_attempts:
                noun = "attempt" if attempt == 1 else "attempts"
                raise EmbeddingError(f"{last_error} (after {attempt} {noun})")

            delay = self._backoff(attempt)
            # No URL, no body, no headers: NFR-5.
            log.warning(
                "huggingface embedding attempt failed; retrying",
                extra={
                    "attempt": attempt,
                    "max_attempts": self._max_attempts,
                    "delay_seconds": round(delay, 3),
                    "reason": last_error.split(";")[0],
                },
            )
            self._sleep(delay)

        # Unreachable: the loop either returns or raises. Present so the signature
        # is total rather than relying on control flow a reader has to verify.
        raise EmbeddingError(last_error or "huggingface embedding failed")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _parse(self, payload: object, expected: int) -> list[list[float]]:
        """Turn the response into vectors, or fail with a specific reason.

        The shape of this response is not stable across models: a single input
        returns a bare vector, several inputs return a list of vectors, and a
        token-classification-style model can add a nesting level. Getting it wrong
        would zip misaligned vectors onto chunk rows positionally, so every branch
        is checked rather than assumed.
        """
        rows: object = payload
        # Unwrap a single-input response, which comes back as one flat vector.
        if isinstance(rows, list) and rows and isinstance(rows[0], (int, float)):
            rows = [rows]
        if not isinstance(rows, list) or len(rows) != expected:
            got = len(rows) if isinstance(rows, list) else type(rows).__name__
            raise EmbeddingError(
                f"huggingface returned {got} embedding(s) for {expected} input(s); "
                f"a reordered or reshaped response would mislabel every vector"
            )

        vectors: list[list[float]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, list) or not all(
                isinstance(v, (int, float)) for v in row
            ):
                raise EmbeddingError(
                    f"huggingface embedding {index} is not a numeric vector"
                )
            vectors.append(self._normalize([float(v) for v in row]))
        return vectors

    def _normalize(self, vector: list[float]) -> list[float]:
        if len(vector) != self._dim:
            # architecture.md 7.3 requires this to fail loudly: a wrong-width vector
            # scores as though it meant something, and the symptom surfaces later
            # as a retrieval-quality regression with no error to explain it.
            raise EmbeddingError(
                f"embedding dimension mismatch from {self._model}: got "
                f"{len(vector)}, EMBEDDING_DIM is {self._dim}. The stored index was "
                f"built at a different width and must be rebuilt."
            )
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            raise EmbeddingError(
                f"huggingface returned a zero vector for {self._model}; it has no "
                f"direction and breaks cosine similarity"
            )
        return [v / norm for v in vector]


class NullRerankerProvider:
    """Pass-through reranker used until a real cross-encoder is configured.

    Scores are 0.0, which is below any configured relevance threshold, so a
    pipeline that enables reranking without a provider fails the threshold check
    and abstains. That is the safe direction to fail: no answer rather than an
    unranked one.
    """

    def rerank(self, query: str, docs: Sequence[str], *, top_k: int) -> list[ScoredDoc]:
        return [
            ScoredDoc(chunk_id=str(i), score=0.0, text=doc)
            for i, doc in enumerate(docs[:top_k])
        ]


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


@lru_cache
def _provider_for(provider: str, dim: int, token: str, base_url: str) -> EmbeddingProvider:
    """Cached on the settings that change the instance, not just the provider name.

    Same reasoning as the generation factory: keying on the provider name alone
    would pin the first construction's token and endpoint, so rotating `HF_TOKEN`
    would do nothing until a restart.
    """
    if provider == "fake":
        return FakeEmbeddingProvider(dim)
    if provider == "huggingface":
        s = get_settings()
        return HuggingFaceEmbeddingProvider(
            token=token,
            base_url=base_url,
            model=s.hf_embedding_model,
            dim=dim,
            timeout=s.hf_timeout_seconds,
            batch_size=s.hf_embed_batch_size,
        )
    raise ValueError(f"unknown embedding provider: {provider!r}")


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider (NFR-10 seam)."""
    s = settings or get_settings()
    return _provider_for(s.embedding_provider, s.embedding_dim, s.hf_token, s.hf_inference_url)


def get_reranker_provider(settings: Settings | None = None) -> RerankerProvider:
    """Return the configured reranker provider.

    Phase 2 replaces this with a real cross-encoder; the ablation harness
    (`eval/run_eval.py --ablate rerank`) depends on it being swappable.
    """
    return NullRerankerProvider()
