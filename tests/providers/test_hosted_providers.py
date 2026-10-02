"""Groq and HuggingFace providers: protocol conformance without a network call (NFR-10).

These providers are the first that reach outside the process, so the tests here are
about the failure modes a live network introduces rather than about the happy path:
a provider that returns the wrong *shape*, buffers instead of streaming, or reports
an upstream status that must not reach the client.

No test makes a network call. `httpx.MockTransport` stands in for both upstreams, so
the suite stays deterministic and costs nothing to run. The cost of that choice is
that it cannot catch a change in the vendors' actual wire format -- see the module
note at the end.
"""

from __future__ import annotations

import json
import math

import httpx
import pytest

from app.core.errors import EmbeddingError, GenerationError
from app.providers.base import EmbeddingProvider, GenerationProvider
from app.providers.embedding import FakeEmbeddingProvider, HuggingFaceEmbeddingProvider
from app.providers.generation import FakeGenerationProvider, GroqGenerationProvider

DIM = 8


def _sse(*deltas: str) -> str:
    frames = []
    for delta in deltas:
        frames.append(
            "data: "
            + json.dumps({"choices": [{"delta": {"content": delta}}]})
        )
    frames.append("data: [DONE]")
    return "\n\n".join(frames) + "\n\n"


class TestProtocolConformance:
    """NFR-10: a provider is anything satisfying the protocol, not a subclass of it."""

    def test_huggingface_satisfies_the_embedding_protocol(self):
        provider = HuggingFaceEmbeddingProvider(
            token="", base_url="https://example.test", model="m", dim=DIM,
            timeout=1.0, batch_size=4,
        )
        assert isinstance(provider, EmbeddingProvider)

    def test_groq_satisfies_the_generation_protocol(self):
        provider = GroqGenerationProvider(
            api_key="k", base_url="https://example.test", timeout=1.0, max_retries=0
        )
        assert isinstance(provider, GenerationProvider)

    def test_the_offline_providers_are_unchanged(self):
        assert isinstance(FakeEmbeddingProvider(DIM), EmbeddingProvider)
        assert isinstance(FakeGenerationProvider(), GenerationProvider)


class TestHuggingFaceEndpointShape:
    """The hosted URL is part of the contract, so it is pinned rather than implied.

    Nothing else in the suite caught this. Every other test here replaces `_post`
    with a mock that returns whatever the handler invents, so the URL the provider
    actually builds was never checked against anything -- and it was wrong. Against
    the live API a request to the old shape returned HTTP 400, which
    `_embed_batch` reports as "check the model name and HF_TOKEN". That message
    sends an operator to debug two things that were both correct while the real
    fault, a missing `/models` path segment, went unmentioned.

    The check is on the path structure rather than the whole URL, so a host change
    does not need this test updated, but a routing change does.
    """

    def test_url_places_the_model_under_the_models_segment(self):
        provider = HuggingFaceEmbeddingProvider(
            token="hf_secret",
            base_url="https://router.huggingface.co/hf-inference",
            model="sentence-transformers/all-MiniLM-L6-v2",
            dim=384,
            timeout=1.0,
            batch_size=4,
        )
        assert provider._url == (
            "https://router.huggingface.co/hf-inference/models/"
            "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        )

    def test_model_appears_once_and_before_the_pipeline_segment(self):
        """The failure mode was the model missing the `/models` prefix entirely."""
        provider = HuggingFaceEmbeddingProvider(
            token="", base_url="https://example.test/base", model="org/my-model",
            dim=DIM, timeout=1.0, batch_size=4,
        )
        assert provider._url.count("org/my-model") == 1
        assert "/models/org/my-model/pipeline/" in provider._url

    def test_trailing_slash_on_the_base_does_not_double_up(self):
        provider = HuggingFaceEmbeddingProvider(
            token="", base_url="https://example.test/base/", model="m",
            dim=DIM, timeout=1.0, batch_size=4,
        )
        assert "//pipeline" not in provider._url.replace("https://", "")

    def test_default_base_url_is_the_live_endpoint(self):
        """`api-inference.huggingface.co` no longer resolves.

        Left in place, every hosted embedding call failed with a ConnectError that
        no amount of debugging the token or the model name would have explained.
        """
        from app.core.config import Settings

        assert Settings().hf_inference_url == "https://router.huggingface.co/hf-inference"


class TestHuggingFaceEmbedding:
    def _provider(self, handler, **kwargs) -> HuggingFaceEmbeddingProvider:
        """A provider wired to a mock transport.

        `HuggingFaceEmbeddingProvider._embed_batch` uses `httpx.post`, so the mock is
        installed by patching that one function rather than a client attribute. The
        transport-based approach was tried first and is misleading here: the
        provider does not hold a client, so there is nothing to swap.
        """
        defaults = dict(
            token="hf_secret", base_url="https://example.test", model="m",
            dim=DIM, timeout=1.0, batch_size=4,
        )
        defaults.update(kwargs)
        provider = HuggingFaceEmbeddingProvider(**defaults)  # type: ignore[arg-type]

        def fake_post(url, *, json, headers, timeout):
            request = httpx.Request("POST", url, json=json, headers=headers)
            return handler(request)

        provider._post = fake_post  # type: ignore[attr-defined]
        return provider

    def test_returns_l2_normalized_vectors_in_input_order(self):
        raw = [[3.0, 4.0] + [0.0] * (DIM - 2), [0.0] * (DIM - 1) + [1.0]]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=raw)

        provider = self._provider(handler)
        vectors = provider.embed(["first", "second"], model="m")

        assert len(vectors) == 2
        for vector in vectors:
            assert len(vector) == DIM
            # Normalized: this is what the cosine-searching stores assume.
            assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-6)
        # Order preserved: first input got the 3-4-0 vector.
        assert math.isclose(vectors[0][0], 0.6)
        assert math.isclose(vectors[0][1], 0.8)

    def test_sends_the_token_and_asks_for_normalization(self):
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=[[1.0] + [0.0] * (DIM - 1)])

        provider = self._provider(handler)
        provider.embed(["x"], model="m")

        assert seen["auth"] == "Bearer hf_secret"
        assert seen["body"]["options"]["normalize"] is True  # type: ignore[index]

    def test_batches_large_inputs(self):
        batch_inputs: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            inputs = json.loads(request.content)["inputs"]
            batch_inputs.append(inputs)
            n = len(inputs)
            # One-hot per input, distinct so ordering is observable within a batch.
            # The index is *within the batch*, which is the contract: the caller zips
            # results onto rows in call order, and `embed` concatenates batches in
            # order, so input k overall lands at output position k overall.
            return httpx.Response(
                200,
                json=[[1.0 if i == j else 0.0 for j in range(DIM)] for i in range(n)],
            )

        provider = self._provider(handler, batch_size=3)
        texts = [f"t{i}" for i in range(7)]
        vectors = provider.embed(texts, model="m")

        assert len(vectors) == 7, "every input must be embedded across batches"
        assert [len(b) for b in batch_inputs] == [3, 3, 1]
        # Inputs were sent in order, split at the batch size.
        assert [t for b in batch_inputs for t in b] == texts
        # Within each batch the returned order matches the sent order.
        for batch, start in zip(batch_inputs, (0, 3, 6), strict=True):
            for offset, _text in enumerate(batch):
                assert vectors[start + offset][offset] == pytest.approx(1.0)

    def test_dimension_mismatch_fails_loudly(self):
        """A wrong-width vector scores as though it meant something (7.3)."""
        provider = self._provider(
            lambda r: httpx.Response(200, json=[[0.0] * (DIM + 1)])
        )
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            provider.embed(["a"], model="m")

    def test_count_mismatch_is_rejected(self):
        """Fewer vectors than inputs would mislabel every chunk positionally."""
        provider = self._provider(
            lambda r: httpx.Response(200, json=[[0.0] * DIM])
        )
        with pytest.raises(EmbeddingError, match="mislabel"):
            provider.embed(["a", "b"], model="m")

    def test_single_input_unwrapping(self):
        """A one-input request can come back as a bare vector, not a list."""
        provider = self._provider(
            lambda r: httpx.Response(200, json=[1.0] + [0.0] * (DIM - 1))
        )
        vectors = provider.embed(["only"], model="m")
        assert len(vectors) == 1 and math.isclose(vectors[0][0], 1.0)

    def test_zero_vector_is_rejected(self):
        """A zero vector has no direction and breaks cosine similarity."""
        provider = self._provider(lambda r: httpx.Response(200, json=[[0.0] * DIM]))
        with pytest.raises(EmbeddingError, match="zero vector"):
            provider.embed(["a"], model="m")

    def test_upstream_error_status_names_the_status_not_the_body(self):
        provider = self._provider(
            lambda r: httpx.Response(503, text="upstream secret detail")
        )
        with pytest.raises(EmbeddingError) as exc:
            provider.embed(["a"], model="m")
        assert "503" in str(exc.value)
        assert "secret detail" not in str(exc.value), "the body must not be echoed"

    def test_transport_failure_is_wrapped(self):
        def boom(*a, **k):
            raise httpx.ConnectError("refused")

        provider = self._provider(lambda r: httpx.Response(200))
        provider._post = boom  # type: ignore[attr-defined]
        with pytest.raises(EmbeddingError, match="ConnectError"):
            provider.embed(["a"], model="m")
    def test_empty_input_short_circuits(self):
        def boom(*a, **k):  # pragma: no cover - must not be called
            raise AssertionError("should not call the API for empty input")

        provider = self._provider(lambda r: httpx.Response(200))
        provider._post = boom  # type: ignore[attr-defined]
        assert provider.embed([], model="m") == []


class TestGroqGeneration:
    def _provider(self, handler=None) -> GroqGenerationProvider:
        provider = GroqGenerationProvider(
            api_key="gsk_secret",
            base_url="https://example.test",
            timeout=1.0,
            max_retries=0,
        )
        if handler is not None:
            seen: dict[str, object] = {}

            def factory(**kwargs):
                return httpx.Client(transport=httpx.MockTransport(handler))

            provider._client_factory = factory  # type: ignore[attr-defined]
            provider._seen = seen  # type: ignore[attr-defined]
        return provider

    def _handler(self, body: str, status: int = 200, seen: dict | None = None):
        def handler(request: httpx.Request) -> httpx.Response:
            if seen is not None:
                seen["auth"] = request.headers.get("Authorization")
                seen["body"] = json.loads(request.content)
            return httpx.Response(status, text=body)

        return handler

    def _run(self, provider, body: str, status: int = 200, seen: dict | None = None) -> list[str]:
        provider._client_factory = lambda **kw: httpx.Client(  # type: ignore[attr-defined]
            transport=httpx.MockTransport(self._handler(body, status, seen))
        )
        return list(
            provider.stream(
                [{"role": "user", "content": "hi"}],
                model="m", max_tokens=16, temperature=0.0,
            )
        )

    def test_yields_each_delta_separately_and_in_order(self):
        provider = self._provider()
        deltas = self._run(provider, _sse("Refunds", " are", " processed."))
        assert deltas == ["Refunds", " are", " processed."], (
            "deltas must be yielded individually and lazily so the sentence buffer "
            "and TTFT measurement work (NFR-1)"
        )

    def test_done_sentinel_ends_the_stream(self):
        provider = self._provider()
        assert self._run(provider, _sse("a", "b")) == ["a", "b"]

    def test_non_json_frame_is_skipped_not_fatal(self):
        """A keep-alive comment should not abort an answer arriving correctly."""
        body = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n: ping\n\ndata: [DONE]\n\n'
        provider = self._provider()
        assert self._run(provider, body) == ["ok"]

    def test_empty_content_frames_are_skipped(self):
        body = (
            'data: {"choices":[{"delta":{}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"text"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        provider = self._provider()
        assert self._run(provider, body) == ["text"]

    def test_error_status_reports_status_without_the_body(self):
        provider = self._provider()
        with pytest.raises(GenerationError) as exc:
            self._run(provider, '{"error":"invalid api key gsk_secret"}', status=401)
        assert "401" in str(exc.value)
        assert "gsk_secret" not in str(exc.value), "the body may echo the key"

    def test_missing_key_fails_at_construction(self):
        """A missing key is a startup fault, not a 503 discovered by a user."""
        with pytest.raises(GenerationError, match="GROQ_API_KEY"):
            GroqGenerationProvider(
                api_key="", base_url="https://example.test", timeout=1.0, max_retries=0
            )

    def test_sends_auth_and_stream_flag(self):
        seen: dict[str, object] = {}
        provider = self._provider()
        self._run(provider, _sse("x"), seen=seen)
        assert seen["auth"] == "Bearer gsk_secret"
        assert seen["body"]["stream"] is True  # type: ignore[index]


class TestStartupRefusesAnUnusableProvider:
    """A missing key must stop the boot, not surface as a 503 on a user's question."""

    def test_groq_with_an_empty_key_is_refused(self):
        from app.core.config import Settings
        from app.main import _assert_providers_constructible

        settings = Settings(generation_provider="groq", groq_api_key="")
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            _assert_providers_constructible(settings)

    def test_groq_with_a_key_is_accepted(self):
        from app.core.config import Settings
        from app.main import _assert_providers_constructible

        settings = Settings(generation_provider="groq", groq_api_key="gsk_x")
        _assert_providers_constructible(settings)  # must not raise

    def test_the_offline_default_is_accepted(self):
        from app.core.config import Settings
        from app.main import _assert_providers_constructible

        _assert_providers_constructible(Settings(generation_provider="fake"))

    def test_the_check_makes_no_network_call(self, monkeypatch):
        """A boot that depends on a third party being up is a restart loop."""
        import httpx

        def boom(*a, **k):  # pragma: no cover - must not be called
            raise AssertionError("startup validation must not reach the network")

        monkeypatch.setattr(httpx, "post", boom)
        monkeypatch.setattr(httpx, "Client", boom)
        from app.core.config import Settings
        from app.main import _assert_providers_constructible

        _assert_providers_constructible(
            Settings(
                generation_provider="groq",
                groq_api_key="gsk_x",
                embedding_provider="huggingface",
                hf_token="hf_x",
            )
        )


class TestFactoryCaching:
    """The factory cache must not pin stale credentials.

    Regression: `_provider_for` was keyed on the provider *name* alone, so a second
    call with a different key returned the first call's instance. Rotating a token
    or switching an endpoint would have done nothing until a process restart, and a
    test selecting a provider twice would silently get the first construction's
    failure.
    """

    def test_changing_the_groq_key_rebuilds_the_provider(self):
        from app.core.config import Settings
        from app.providers.generation import get_generation_provider

        first = get_generation_provider(Settings(generation_provider="groq", groq_api_key="k1"))
        second = get_generation_provider(
            Settings(generation_provider="groq", groq_api_key="k2")
        )
        assert first is not second

    def test_an_empty_key_does_not_poison_a_later_valid_one(self):
        from app.core.config import Settings
        from app.providers.generation import get_generation_provider

        with pytest.raises(GenerationError):
            get_generation_provider(Settings(generation_provider="groq", groq_api_key=""))
        # Same provider name, now with a key: must construct rather than replay the
        # cached failure.
        provider = get_generation_provider(
            Settings(generation_provider="groq", groq_api_key="gsk_ok")
        )
        assert isinstance(provider, GroqGenerationProvider)

    def test_changing_the_hf_token_rebuilds_the_provider(self):
        from app.core.config import Settings
        from app.providers.embedding import get_embedding_provider

        first = get_embedding_provider(
            Settings(embedding_provider="huggingface", hf_token="t1", embedding_dim=8)
        )
        second = get_embedding_provider(
            Settings(embedding_provider="huggingface", hf_token="t2", embedding_dim=8)
        )
        assert first is not second


class TestErrorSurfacing:
    def test_generation_error_is_a_503_with_a_generic_user_message(self):
        """A client may retry a 503; it may not retry a 400. And the model name
        must not cross the API boundary (NFR-5)."""
        from app.core.guardrails import safe_error_payload

        err = GenerationError("groq HTTP 401 for model 'llama-3.3-70b-versatile'")
        assert err.status_code == 503
        payload = safe_error_payload(err)
        assert "llama" not in payload["message"]
        assert "401" not in payload["message"]
