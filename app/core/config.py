"""Application configuration.

All settings come from the environment. Secrets are required with no default so a
misconfigured deployment fails at startup rather than silently degrading.

`EMBEDDING_DIM` is pinned rather than discovered. Changing the embedding model
changes the dimension, which invalidates every stored vector; the system must fail
loudly on mismatch instead of returning silently wrong similarity scores
(architecture.md 7.3).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings, loaded once and validated at startup."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Persistence -------------------------------------------------------
    # Postgres is the target (implementation.md 1). SQLite is accepted so the
    # test suite and local runs work without a database server; production
    # deployments must use Postgres because the pgvector column type is
    # Postgres-specific.
    database_url: str = "sqlite:///./rag.db"

    # --- Embeddings --------------------------------------------------------
    # Pinned. See module docstring.
    #
    # Three providers, and the default is still the offline one. That default is
    # deliberate: `huggingface` bills per request and reaches the network on every
    # embed, so a checkout that has not been configured should neither spend money
    # nor silently depend on a third party being up. Tests and CI stay on `fake`,
    # which is deterministic and needs no key.
    embedding_provider: Literal["fake", "huggingface"] = "fake"
    embedding_model: str = "fake-embed-v1"
    embedding_dim: int = Field(default=384, gt=0)

    # --- HuggingFace Inference API (embeddings) ----------------------------
    # `huggingface_hub` is deliberately not a dependency. The feature-extraction
    # pipeline is a single HTTP POST, so `httpx` -- already required by the test
    # harness -- is the whole client, and the hosted path costs no extra install.
    hf_token: str = ""
    # `router.huggingface.co/hf-inference` is the current hosted endpoint. The older
    # `api-inference.huggingface.co` no longer resolves -- a request to it fails to
    # connect, which surfaces as an opaque ConnectError rather than a useful status.
    # The provider appends `/models/{model}/pipeline/feature-extraction`.
    hf_inference_url: str = "https://router.huggingface.co/hf-inference"
    # `sentence-transformers/all-MiniLM-L6-v2` is 384-dimensional, so switching to
    # it leaves `embedding_dim` unchanged. The stored vectors still all have to be
    # rebuilt: they were produced by a hash function, not a model.
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Per-request ceiling. Embedding runs on every query, so a hung upstream must
    #: fail fast rather than sit inside the 5 s TTFT budget (NFR-1).
    hf_timeout_seconds: float = Field(default=30.0, gt=0)
    #: Inputs per request. The feature-extraction pipeline accepts a list and
    #: returns one vector per input, but large batches time out more often and make
    #: a partial failure harder to attribute, so the default is deliberately small.
    hf_embed_batch_size: int = Field(default=16, ge=1, le=256)

    # --- Ingestion limits (FR-2) ------------------------------------------
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    allowed_mime_prefixes: tuple[str, ...] = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
        "text/html",
    )

    # --- Sandbox limits (NFR-5) --------------------------------------------
    # Enforced by app/ingest/sandbox.py. Without Docker these bound process
    # time only; see sandbox.py for the degradation note.
    sandbox_timeout_seconds: int = Field(default=30, gt=0)
    sandbox_memory_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    sandbox_max_output_chars: int = Field(default=20_000_000, gt=0)

    # --- Object store ------------------------------------------------------
    object_store_dir: str = "./data/objects"

    # --- Ingestion queue ---------------------------------------------------
    # Off by default so a local run and the test suite need no broker, and the
    # pipeline completes inline (the document is `live` by the time the upload
    # response returns). Enable with a broker URL in staging/production.
    #
    # `required` distinguishes a queue from a database: a broker outage must not
    # stop the service answering queries, but when ingestion cannot run it must
    # stop accepting uploads rather than leaving `pending` rows that never become
    # queryable — those are indistinguishable from lost files to the user.
    ingest_queue_enabled: bool = False
    ingest_queue_broker_url: str = ""
    ingest_queue_required: bool = True
    ingest_max_attempts: int = Field(default=3, gt=0)
    ingest_visibility_timeout_seconds: int = Field(default=300, gt=0)

    # --- Retrieval (FR-12..FR-15) ------------------------------------------
    # Every number architecture.md 3.3 or 4.1 declares tunable is here rather
    # than a literal in the pipeline, because 3.3 states the relevance threshold
    # is a *measured* parameter: the eval harness sweeps it to hit recall@10 and a
    # 10-30% refusal rate simultaneously. A threshold in code cannot be swept.
    #
    # Defaults are the architecture's stated starting points (3.3: wide fetch
    # 20-50, narrow context 4-8; 7.4 budgets the stages). They are starting
    # points, not tuned values -- see docs/eval_results.md.
    retrieval_fetch_k: int = Field(default=40, gt=0)
    retrieval_rerank_k: int = Field(default=40, gt=0)
    retrieval_top_k: int = Field(default=8, gt=0)
    # Measured, not guessed: `eval/run_eval.py --sweep` selects the value that
    # holds recall@10 >= 0.85 and a 10-30% refusal rate simultaneously.
    #
    # **No threshold currently satisfies both.** On real embeddings
    # (sentence-transformers/all-MiniLM-L6-v2, 200 questions) recall@10 peaks at
    # 0.8446 at threshold 0.00-0.05, against the 0.85 target -- short by 0.0054,
    # about one question -- while refusal only enters the 10-30% band from 0.05
    # upward. At 0.10 the figures are recall@10 0.8378, refusal 21.5%; at 0.15
    # recall falls further to 0.8311 and the band is still met; refusal leaves the
    # band at 0.20 (34.5%). `docs/eval/threshold_history.jsonl` records
    # `recommended: null` and `joint_count: 0` for that sweep.
    #
    # The value stays at 0.10 because it is the best refusal-band point that is
    # not simply "return nothing", not because it satisfies both targets --
    # it does not, and claiming otherwise was the bug in this comment before.
    #
    # An earlier revision of this comment justified 0.10 with "recall@10 0.892,
    # refusal 21.0%". Those numbers came from the fake-embed-v1 provider, a
    # SHA-256 hash bucket rather than a sentence encoder, and were not a
    # measurement of retrieval quality. Re-measure after any change to the
    # corpus, the reranker, or the embeddings -- it is a property of those, not a
    # constant.
    retrieval_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    retrieval_max_context_tokens: int = Field(default=4000, gt=0)
    # Prior *user* questions passed to retrieval for anaphora resolution. Separate
    # from `chat_history_turns`, which bounds the generation prompt: a follow-up
    # can depend on a question the model no longer sees verbatim. A hard cut of
    # the last 10 user turns; 0 sends the current question alone.
    retrieval_memory_turns: int = Field(default=10, ge=0)
    # Which vector store backend serves the vector half of hybrid retrieval.
    #
    # There is deliberately no "auto". It inferred the backend from the database
    # dialect, which let a single working tree disagree with itself unreported: this
    # setting, the populated ANN index under `data/chroma`, and the authoritative
    # chunks in SQL could each look "correct", and retrieval silently depended on
    # which one won. Nothing compared the configured backend against the data that
    # actually existed, so a stale index sitting on disk looked exactly like a
    # serving one. See `measure_divergence` in app/retrieval/vector_store.py, which
    # is the check that inference made unnecessary.
    #
    # The default matches the default `database_url` (SQLite). Chroma is a *derived*
    # projection of SQL, not a peer store: ingestion never writes it, so it is stale
    # by construction after any corpus change and SQL is always the more complete of
    # the two. Defaulting to the authoritative store is the safe direction to be
    # wrong in.
    #
    # Named rather than boolean because the three backends differ in cost model, not
    # just in location, and "use a vector db" is not a setting worth guessing. A
    # Postgres deployment must now name `pgvector` explicitly; the dialect mismatch
    # is raised at startup rather than inferred.
    vector_store: Literal["pgvector", "sqlite", "chroma"] = "sqlite"
    # Chroma persistent-client location. Relative paths resolve against the process
    # working directory, matching `object_store_dir` above.
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "chunks"
    retrieval_reserve_for_answer: int = Field(default=500, ge=0)
    retrieval_rerank_enabled: bool = True
    retrieval_keyword_enabled: bool = True
    retrieval_vector_enabled: bool = True
    # Which reranker implementation. "lexical" is the offline stand-in; "null"
    # scores everything zero, which abstains under any positive threshold.
    reranker_provider: Literal["lexical", "null"] = "lexical"
    # Domain synonyms for query expansion (architecture.md 3.2 step 2). Empty in
    # v1 because the glossary depends on open question 9; populated as
    # "term=first,synonym,synonym" pairs.
    retrieval_glossary: str = ""

    # --- Generation (FR-17..FR-22) ----------------------------------------
    # The provider is swappable behind `GenerationProvider` (NFR-10). "fake" is
    # the offline deterministic extractive provider: it streams a grounded,
    # cited answer from the retrieved passages so the query path is testable
    # end to end without a vendor SDK. It is *not* a language model; see
    # app/providers/generation.py for what its output does and does not prove.
    generation_provider: Literal["fake", "groq"] = "fake"
    generation_model: str = "fake-gen-v1"
    generation_max_tokens: int = Field(default=1024, gt=0)
    generation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Groq (generation) -------------------------------------------------
    # Groq exposes an OpenAI-compatible `/chat/completions`, so the streaming
    # client is plain `httpx` over SSE rather than a vendor SDK. Groq does not
    # serve embeddings, which is why embeddings come from HuggingFace above --
    # the two halves of the stack necessarily have different providers.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # A current small instruct model. Config, not code, so a model bump is not a
    # deploy (PRD open question 8 is still open on cost).
    # llama-3.3-70b-versatile was shut down by Groq on 2026-08-16. gpt-oss-120b
    # is the documented replacement.
    groq_generation_model: str = "openai/gpt-oss-120b"
    #: Bounds a stalled stream. The TTFT budget is 5 s (NFR-1) and architecture.md
    #: 7.4 budgets 3.5 s of it for the model, so a connect timeout well above that
    #: would spend the whole budget waiting for a socket.
    groq_timeout_seconds: float = Field(default=60.0, gt=0)
    #: Retries are *off* by default, and the default is the safe answer here rather
    #: than an oversight. The chat path streams to the user as tokens arrive, so a
    #: retry that fires after the first delta would either duplicate text already
    #: read or stall a partially delivered answer. Raising this therefore buys
    #: resilience for a connection that fails *before* the answer starts and
    #: nothing more -- `GroqGenerationProvider` tracks whether a delta has been
    #: yielded and stops retrying the moment one has. It counts retries, not
    #: attempts, so 0 means exactly one attempt.
    #:
    #: Set it above 0 only with that narrow scope in mind; raising it is not a way to
    #: make the streaming path safer. Backoff policy is shared with the embedding
    #: provider in `app/providers/retry.py`.
    groq_max_retries: int = Field(default=0, ge=0, le=5)

    # --- Chat guardrail (FR-34) -------------------------------------------
    # P0 and cheap, so the cap is enforced from the first streaming endpoint
    # rather than added in Phase 4. Counted in characters: the char budget is a
    # backstop against a pathological payload, not a token accounting of the
    # model context (that is retrieval_max_context_tokens).
    chat_max_query_chars: int = Field(default=4000, gt=0)

    # --- Conversation (FR-23, FR-25) ---------------------------------------
    # How many prior turns are sent as context. Defaults to 6, about three
    # exchanges: enough for "and what about shipping?" to resolve, short enough that
    # a long thread does not dominate the prompt.
    #
    # A hard cut, not a summary. FR-24's condensation into a running summary is
    # deferred (implementation.md §9.1), so past this window anaphora resolution
    # simply stops working. That is the intended v1 behaviour: a visible limitation
    # beats a summary that quietly drops the fact the user was asking about.
    chat_history_turns: int = Field(default=6, ge=0)

    # --- Rate limit (FR-31) -------------------------------------------------
    # Per client identity, fixed window, enforced in-process. `0` disables the
    # guard rather than blocking every request, so a misconfigured deployment
    # degrades to "no limit" instead of "total outage".
    #
    # Per-process, not shared: see `RateLimiter` in app/core/guardrails.py. The
    # architecture puts the real limit at the gateway (architecture.md §6).
    chat_rate_limit_requests: int = Field(default=30, ge=0)
    chat_rate_limit_window_seconds: float = Field(default=60.0, gt=0)

    # --- Browser client (FR-23, FR-29) -------------------------------------
    # The web UI is served from a different origin than the API, so the browser
    # needs an explicit allowance. Defaults to the local dev ports only: an
    # unconfigured deployment must fail closed rather than serve any origin.
    #
    # Comma-separated. `*` is accepted but should not be used with credentialed
    # requests; there are none here, but a wildcard silently turns any future
    # cookie-auth addition into a cross-origin vulnerability.
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- Access --------------------------------------------------------------
    # Empty means open, which is what tests and an unconfigured checkout need.
    # A non-empty value requires `Authorization: Bearer <token>` on every route
    # except health and the login probe. The browser never receives this value
    # except by the operator typing it in; it is not a NEXT_PUBLIC variable.
    api_token: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    def validate_production(self) -> None:
        """Startup checks that only make sense for a real deployment."""
        deployed = self.environment in {"production", "staging"}
        if deployed and self.database_url.startswith("sqlite"):
            raise RuntimeError(
                "database_url must be Postgres in staging/production. "
                "The vector column uses pgvector, which SQLite does not provide. "
                "Set DATABASE_URL to a postgres:// DSN."
            )
        if deployed and self.vector_store == "sqlite":
            # Gated on the environment, which this check was missing. The message and
            # the field's own comment both scope it to staging/production -- the
            # setting is documented as "kept for local development and tests" -- but
            # ungated it made that documented use impossible: a local run or the
            # benchmark harness that pinned `VECTOR_STORE=sqlite` crashed at startup
            # with a message describing an environment it was not in.
            #
            # The dialect check above catches an accidental SQLite *database*; this
            # catches an explicit `VECTOR_STORE=sqlite` against a real Postgres.
            # Same reason, different mistake: the SQLite store full-scans in Python,
            # so it would pass the latency budget on a small corpus and fail at
            # NFR-4 scale with nothing in the logs to explain it.
            raise RuntimeError(
                "vector_store must not be 'sqlite' in staging/production. "
                "SqliteVectorStore is an O(n) full scan with no ANN index, kept for "
                "local development and tests. Use 'pgvector', 'chroma', or 'auto'."
            )
        # Same reasoning as the two checks above, applied to the model providers: a
        # `fake` provider answers from a hash function or a canned fixture, so a
        # deployment that forgot to set these starts cleanly, serves 200s, and
        # produces retrieval and generation numbers that look like measurements but
        # are not. That is worse than refusing to boot, because the evidence it
        # manufactures survives into the Phase 6 pilot metrics (implementation.md
        # §8.1a) -- an eval run against fake embeddings is what produced the 0.892
        # recall@10 recorded as the baseline in docs/known_issues.md.
        #
        # Gated on the environment for the same reason as `vector_store`: tests and
        # local runs are *supposed* to use `fake`, they are deterministic and free.
        # Only a real deployment is a fault.
        #
        # `embedding_model` is checked as well as the provider, because the two can
        # disagree. Switching EMBEDDING_PROVIDER to huggingface while leaving the
        # default `fake-embed-v1` would pass a provider-only check and then send that
        # string to the HuggingFace API as a model name.
        if deployed and self.embedding_provider == "fake":
            raise RuntimeError(
                "embedding_provider must not be 'fake' in staging/production. "
                "The fake provider returns deterministic fixtures, not embeddings, "
                "so retrieval quality measured against it is meaningless. "
                "Set EMBEDDING_PROVIDER=huggingface."
            )
        if deployed and self.generation_provider == "fake":
            raise RuntimeError(
                "generation_provider must not be 'fake' in staging/production. "
                "The fake provider returns canned answers and would let a deployment "
                "appear to work while answering nothing it was asked. "
                "Set GENERATION_PROVIDER=groq."
            )
        if deployed and self.embedding_model.startswith("fake"):
            raise RuntimeError(
                f"embedding_model must not be a fake model name in "
                f"staging/production (got {self.embedding_model!r}). That default "
                f"belongs to the offline provider; with EMBEDDING_PROVIDER="
                f"huggingface it would be sent to the API as a model name. Set "
                f"EMBEDDING_MODEL, e.g. sentence-transformers/all-MiniLM-L6-v2."
            )
        if deployed and not self.api_token:
            raise RuntimeError(
                "api_token must be set in staging/production. An empty token "
                "leaves every endpoint open, including document upload and delete. "
                "Set API_TOKEN to a long random value."
            )

    @property
    def glossary(self) -> dict[str, tuple[str, ...]]:
        """Parsed retrieval glossary, for query expansion (architecture.md 3.2).

        Parsed on access rather than in a validator because an unparseable entry
        should be visible as a warning during a retrieval call, not crash settings
        construction at import time in a process that may not retrieve anything.
        Malformed entries are skipped rather than raising: an empty glossary
        degrades query expansion to a no-op, which is recoverable, whereas a
        startup crash over a typo in an optional feature is not.
        """
        out: dict[str, tuple[str, ...]] = {}
        for entry in self.retrieval_glossary.split(","):
            entry = entry.strip()
            if not entry or "=" not in entry:
                continue
            term, _, synonyms = entry.partition("=")
            term = term.strip().lower()
            values = tuple(s.strip() for s in synonyms.split("|") if s.strip())
            if term and values:
                out[term] = values
        return out


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
