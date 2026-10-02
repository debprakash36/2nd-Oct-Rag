"""Generation providers and the factory that selects them (NFR-10 seam).

`get_generation_provider()` is the single place a concrete generator is chosen,
mirroring `get_embedding_provider` and `get_reranker_provider`. Adding a vendor
means adding a branch here, and nothing else in the codebase changes.

`FakeGenerationProvider` is the offline stand-in. It is **not** a language model
and its output must not be read as evidence about model quality: it is an
extractive reader that selects the retrieved sentences most overlapping the
question and appends the passage's marker. What it does provide, honestly, is a
grounded-and-cited stream:

* it can only ever emit markers for passages that were actually retrieved, so it
  is the mechanism that proves the citation pipeline end to end;
* it streams lazily, one word at a time, so TTFT and the sentence buffer are
  exercised for real (NFR-1);
* it never invents content, so a groundedness score computed against it measures
  the *pipeline*, not the model.

Those three properties are exactly what Phase 3 can verify without model access.
The provider is swapped for a real one behind the same interface before any
claim about answer quality is made.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterator
from functools import lru_cache

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import GenerationError
from app.core.logging import get_logger
from app.generation.prompt import parse_context_block
from app.generation.sentences import split_sentences
from app.ingest.keyword import tokenize
from app.providers.base import GenerationProvider
from app.providers.retry import RETRYABLE_STATUS_CODES, backoff_seconds

log = get_logger("app.providers.generation")

#: Terms that carry no discriminative signal in an extractive selection.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
        "are", "was", "were", "be", "been", "what", "which", "who", "whom",
        "this", "that", "these", "those", "it", "its", "as", "at", "by",
        "with", "from", "does", "do", "did", "how", "when", "where", "why",
        "can", "could", "should", "would", "will", "i", "you", "we", "they",
        "me", "my", "our", "your", "their", "say", "says", "about", "under",
    }
)

#: Fixed text for the no-evidence case. It carries no marker, so the validator
#: converts it to a refusal (implementation.md 5.2) rather than showing an
#: answer that claims to know. Kept here as the offline provider's abstention.
_NO_EVIDENCE = "I could not find this in the available documents."


class FakeGenerationProvider:
    """Deterministic extractive reader that streams a cited answer.

    Selection is by simple term overlap, which is enough to produce a grounded
    answer from the passages the retriever chose. It is intentionally not clever:
    a clever offline stand-in would make the pipeline look better than it is and
    hide the fact that no model was involved.
    """

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield the answer one word at a time, lazily.

        Laziness is a contract (app/providers/base.py): building the whole answer
        string before the first yield would break the TTFT budget this provider
        exists to exercise.
        """
        answer = self._compose(messages)
        for produced, word in enumerate(answer.split(" ")):
            if produced >= max_tokens:
                break
            # Trailing space reconstructs the text; the consumer strips per
            # sentence, so spacing does not affect validation.
            yield word if produced == 0 else f" {word}"

    def _compose(self, messages: list[dict[str, str]]) -> str:
        content = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                break
        block, _, question = content.rpartition("\n\nQuestion: ")
        if not block:
            block, question = content, content
        passages = parse_context_block(block)
        if not passages:
            return _NO_EVIDENCE

        query_terms = {t for t in tokenize(question) if t not in _STOPWORDS}
        if not query_terms:
            # A question with no content terms ("what about that?") gives the
            # extractor nothing to select on. Abstain rather than pick a
            # passage at random and dress it as an answer.
            return _NO_EVIDENCE

        scored: list[tuple[int, int, str]] = []
        for passage in passages:
            overlap = len(query_terms.intersection(tokenize(passage.text)))
            if overlap > 0:
                scored.append((-overlap, passage.number, passage.text))
        if not scored:
            return _NO_EVIDENCE

        scored.sort()
        selected = scored[:2]

        parts: list[str] = []
        for _neg, number, text in selected:
            sentence = _best_sentence(text, query_terms)
            if sentence:
                parts.append(f"{sentence} [{number}]")
        if not parts:
            return _NO_EVIDENCE
        return " ".join(parts)


def _best_sentence(text: str, query_terms: set[str]) -> str:
    """The sentence of `text` with the most query-term overlap."""
    best = ""
    best_overlap = -1
    for sentence in split_sentences(text):
        overlap = len(query_terms.intersection(tokenize(sentence)))
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best.strip()


class GroqGenerationProvider:
    """Streaming chat completions from Groq's OpenAI-compatible endpoint.

    Groq serves `/chat/completions` with the same request and SSE response shape as
    OpenAI, so this is `httpx` over a text/event-stream rather than a vendor SDK.
    The sentence-buffered citation validator upstream consumes text deltas and
    knows nothing about the vendor, so nothing outside this module changes.

    **Laziness is the contract this class exists to honour.** `app/providers/base.py`
    requires the first delta to be yielded as it is produced, because buffering the
    response before yielding would break NFR-1 -- the whole reason streaming is
    mandatory for this system rather than a nice-to-have. The SSE frames are decoded
    and yielded one at a time, never collected into a string first.

    **Mid-stream errors are not retried.** The chat endpoint has already written
    text to the user by the time a later frame fails, so a retry would either
    duplicate what they have read or stall a half-delivered answer. A connection
    that fails *before* any delta is safe to retry, and that is the only case
    `groq_max_retries` covers.

    That constraint is enforced by tracking whether a delta has been yielded. The
    retry decision is made inside the same generator that yields, so "has the user
    seen anything yet?" is answerable rather than guessed: once one delta is out,
    the next transport error propagates immediately. Retrying by wrapping `stream()`
    from outside would not be able to tell the two cases apart.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        max_retries: int,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if not api_key:
            # Raised at construction rather than at the first request: a missing key
            # is a startup fault, and discovering it as a 503 on a user query hides
            # the actual cause behind a symptom.
            raise GenerationError(
                "GROQ_API_KEY is empty but generation_provider is 'groq'. Set it in "
                ".env, or set GENERATION_PROVIDER=fake for the offline provider."
            )
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._timeout = timeout
        # `groq_max_retries` counts *retries*, not attempts, so 0 means one attempt
        # and the default really is "no retry" -- see the config comment.
        self._max_retries = max(0, max_retries)
        # Injected so tests can assert the backoff schedule without sleeping it out.
        self._sleep = sleep
        self._jitter = jitter
        # Injectable transport seam; see the note on the embedding provider's
        # equivalent. The real streaming client is built per call, because a
        # `client.stream` context cannot outlive the generator that yields from it.
        self._client_factory = httpx.Client

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        # gpt-oss spends part of max_tokens on a hidden reasoning channel. A
        # concise budget of 256 can be used up before any answer text is emitted,
        # which the citation check then treats as an ungrounded refusal.
        budget = max_tokens
        extra: dict[str, object] = {}
        if "gpt-oss" in model:
            budget = max_tokens + 1024
            extra["reasoning_effort"] = "low"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": budget,
            "temperature": temperature,
            "stream": True,
            **extra,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        while True:
            delivered = False  # has any delta reached the caller on this attempt?
            try:
                with (
                    self._client_factory(timeout=self._timeout) as client,
                    client.stream("POST", self._url, json=payload, headers=headers) as response,
                ):
                    if response.status_code != 200:
                        # Status only. The body of a provider error can echo the
                        # request, and this message reaches a log and potentially an
                        # operator's terminal (NFR-5).
                        response.read()
                        if (
                            response.status_code not in RETRYABLE_STATUS_CODES
                            or attempt >= self._max_retries
                        ):
                            raise GenerationError(
                                f"groq returned HTTP {response.status_code} for model "
                                f"{model!r}; check GROQ_API_KEY and the model name"
                            )
                        delay = self._backoff(attempt)
                        # No URL, no body, no headers: NFR-5.
                        log.warning(
                            "groq attempt failed before streaming began; retrying",
                            extra={
                                "attempt": attempt + 1,
                                "max_retries": self._max_retries,
                                "delay_seconds": round(delay, 3),
                                "status": response.status_code,
                            },
                        )
                        self._sleep(delay)
                        attempt += 1
                        continue

                    for delta in self._deltas(response):
                        # Set *before* the yield resumes, so a transport error on the
                        # next frame is correctly seen as mid-stream.
                        delivered = True
                        yield delta
                    return
            except httpx.TransportError as exc:
                # Retried only while nothing has been delivered. After the first
                # delta this propagates: the user is already reading text, and a
                # retry would duplicate or stall it.
                if delivered or attempt >= self._max_retries:
                    raise GenerationError(
                        f"groq request failed: {type(exc).__name__}"
                    ) from exc
                delay = self._backoff(attempt)
                log.warning(
                    "groq attempt failed before streaming began; retrying",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": self._max_retries,
                        "delay_seconds": round(delay, 3),
                        "reason": type(exc).__name__,
                    },
                )
                self._sleep(delay)
                attempt += 1
                continue
            except httpx.HTTPError as exc:
                # Not a transport fault (a malformed response, a bad URL). Repeating
                # it returns the same thing, so it is wrapped and raised at once.
                raise GenerationError(
                    f"groq request failed: {type(exc).__name__}"
                ) from exc

    def _backoff(self, attempt: int) -> float:
        """Seconds to wait before the next attempt. Policy in `app.providers.retry`.

        Identical to the embedding provider's, deliberately -- one policy module, so
        the two providers cannot drift apart.

        `attempt` here is 0-based (the count of failures so far), so it is shifted by
        one before reaching the policy. The embedding provider's loop is 1-based and
        passes its counter straight through; without this shift the two providers
        would start their schedules one step apart while still sharing the constants.
        """
        return backoff_seconds(attempt + 1, jitter=self._jitter)

    def _deltas(self, response: object) -> Iterator[str]:
        """Yield the `content` of each SSE `data:` frame, lazily.

        The terminal `data: [DONE]` sentinel carries no content and ends the stream.
        A frame that is not JSON is skipped rather than raised on: a provider that
        emits a keep-alive comment mid-stream should not abort an answer that is
        otherwise arriving correctly.
        """
        for line in response.iter_lines():  # type: ignore[attr-defined]
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    return
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                log.debug("skipping non-JSON SSE frame", extra={"length": len(data)})
                continue
            for choice in event.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    yield content


@lru_cache
def _provider_for(provider: str, api_key: str, base_url: str) -> GenerationProvider:
    """Cached on the settings that change the instance, not just the provider name.

    Keying on the name alone would return a provider built with whatever key was
    current when the cache was first filled, so rotating `GROQ_API_KEY` or switching
    `GROQ_BASE_URL` would have no effect until the process restarted -- and a test
    that selected a provider twice would silently get the first construction's
    failure. The secret is in the key rather than captured separately, so no token
    is stored anywhere else.
    """
    if provider == "fake":
        return FakeGenerationProvider()
    if provider == "groq":
        s = get_settings()
        return GroqGenerationProvider(
            api_key=api_key,
            base_url=base_url,
            timeout=s.groq_timeout_seconds,
            max_retries=s.groq_max_retries,
        )
    raise ValueError(f"unknown generation provider: {provider!r}")


def get_generation_provider(settings: Settings | None = None) -> GenerationProvider:
    """Return the configured generation provider (NFR-10 seam)."""
    s = settings or get_settings()
    return _provider_for(s.generation_provider, s.groq_api_key, s.groq_base_url)
