"""`POST /chat/stream`: the orchestrated query path (FR-19, FR-20, FR-28).

Implements the request flow in architecture.md 3.1: retrieve, then generate, then
validate citations sentence-wise while streaming. The SSE event vocabulary is
architecture.md 6:

    event: sources          { sources: [{index, chunk_id, breadcrumb, page}] }
    event: token            { text }                       sentence-flushed
    event: citation_warning { stripped_count }             only when non-zero
    event: done             { query_id, abstained, ttft_ms }
    event: error            { code, message }              user-safe only

`sources` is emitted before the first token on purpose (FR-20): the panel renders
immediately so the user can see what the system is reading while it reads it, and
for a refusal it is the entire payload.

Retrieval runs in the (sync) endpoint before the response starts. Generation is
streamed, and the QueryLog row is written from inside the generator so a
client that disconnects mid-answer is still recorded — a query that was never
logged is a query that can never be classified in the improvement loop.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError, NotFoundError
from app.core.guardrails import (
    client_identity,
    enforce_message_size,
    get_rate_limiter,
    safe_error_payload,
)
from app.core.logging import get_logger
from app.db.conversation import (
    append_turn,
    get_conversation,
    message_history,
    query_history,
)
from app.db.models import TurnRole
from app.db.query_log import record_query
from app.db.session import get_db, get_session_factory
from app.generation.assembly import AnswerAssembler, ChunkKind
from app.generation.prompt import (
    PROMPT_VERSION,
    STYLE_PRESETS,
    AnswerStyle,
    ContextPassage,
    build_messages,
    new_nonce,
)
from app.generation.refusal import RefusalReason, refusal_text, sources_payload
from app.ingest.keyword import tokenize
from app.providers.base import GenerationProvider
from app.providers.generation import get_generation_provider
from app.retrieval.retriever import Retriever
from app.retrieval.types import RetrievalCandidate

log = get_logger("app.api.chat")

router = APIRouter(tags=["chat"])


@dataclass
class _StreamState:
    """Mutable per-request streaming state.

    Extracted so the `finally` block can persist a partial answer when the client
    disconnects before `done`, without a pile of nonlocal scalars.
    """

    abstained: bool
    ttft_ms: int | None = None
    citations_stripped: int = 0
    recorded: bool = False
    #: Set once a persist attempt has been made, successfully or not.
    #:
    #: `recorded` alone is not enough to gate the `finally` retry: a persist that
    #: *fails* leaves `recorded` False, so the retry in `finally` fires and attempts
    #: the same doomed write a second time, logging a second stack trace and
    #: reporting the cause as an interrupted query when the client was actually
    #: served in full.
    persist_attempted: bool = False
    #: Why the last persist failed, or None if it did not.
    persist_error: str | None = None


class ChatRequest(BaseModel):
    """Request body for `/chat/stream` (architecture.md 6)."""

    message: str = Field(min_length=1)
    conversation_id: str | None = None
    answer_style: Literal["concise", "detailed"] = "concise"


def _safe_persist(
    persist: Callable[[], str],
    log_session: Session,
    state: _StreamState,
    *,
    suppress: bool,
) -> str | None:
    """Run `persist`, degrading to a lost log row rather than a broken stream.

    Writing the QueryLog row is telemetry, but it sits on the request's critical
    path: it runs *after* the last token has been flushed to the client. A failure
    here used to escape the `except AppError` handler entirely, so a locked SQLite
    database produced an unhandled ASGI exception and a stream that ended with no
    `done` and no `error` - the user had already read the whole answer and then had
    the connection drop under it. Under load that is not a rare edge; the 50-VU
    sweep reproduced it repeatedly.

    The caller states the policy rather than this inferring it from stream state,
    because the two call sites are genuinely different:

    - **After the answer was delivered** (`suppress=False`): the answer is real and
      the failure is only in the bookkeeping. Emitting an `error` event now would
      contradict text the user has already read, so the stream completes with a null
      `query_id` and the loss is logged. The one visible consequence is honest:
      FR-29 feedback buttons need a query id, so that turn simply has none.
    - **From `finally`, while an exception is already propagating** (`suppress=True`):
      the failure must be swallowed unconditionally. Raising here would discard the
      original error mid-unwind and replace it with a database error, so the client
      would be told the wrong thing went wrong.

    Returning `None` is a meaningful signal, not a type accident: it means
    "answered, but not recorded".
    """
    try:
        return persist()
    except Exception as exc:
        state.persist_error = f"{type(exc).__name__}: {exc}"
        try:
            log_session.rollback()
        except Exception:  # pragma: no cover - rollback of a dead connection
            log.warning("rollback failed while recovering from persist error")
        log.exception(
            "failed to record query log",
            extra={"suppressed": suppress, "delivered": state.ttft_ms is not None},
        )
        if suppress:
            return None
        if state.ttft_ms is None:
            # Nothing was promised to the client yet, so this is a real service
            # error rather than a truncation; let it surface normally.
            raise
        return None


def _sse(event: str, data: Mapping[str, object]) -> str:
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _passages(candidates: list[RetrievalCandidate]) -> list[ContextPassage]:
    return [
        ContextPassage(text=c.text, breadcrumb=c.breadcrumb, page=c.page)
        for c in candidates
    ]


def _count_tokens(texts: list[str]) -> int:
    """Approximate token count.

    Deliberately the same tokenizer the keyword index uses, so the number is
    consistent across the pipeline. It is an approximation of the model's own
    tokenizer and is logged as such; it is used for cost and context tracking,
    never for a correctness decision.
    """
    return sum(len(tokenize(text)) for text in texts)


def get_generation_provider_dep(
    settings: Settings = Depends(get_settings),
) -> GenerationProvider:
    return get_generation_provider(settings)


@dataclass(frozen=True)
class ConversationHistory:
    """Prior turns in the two shapes the pipeline consumes (FR-23).

    One dataclass rather than a dict: the two fields have different element types,
    so a `dict[str, list]` forces every call site into a cast, and the whole point
    of keeping them separate is that they are not interchangeable.
    """

    #: Interleaved user/assistant turns for `build_messages`.
    messages: list[dict[str, str]] = field(default_factory=list)
    #: User questions only, for the retriever's anaphora resolution. Excludes the
    #: question being asked right now.
    queries: list[str] = field(default_factory=list)

    @property
    def has_history(self) -> bool:
        return bool(self.messages)

    #: Whether the conversation exists and turns may be written to it. False when no
    #: id was supplied, when history is disabled, or when the id is stale.
    persistable: bool = False


def _conversation_history(
    session: Session, payload: ChatRequest, settings: Settings
) -> ConversationHistory:
    """Load prior turns for this request (FR-23), and record the user's new turn.

    Returns both views the pipeline needs: `messages` (interleaved, for
    `build_messages`) and `queries` (user questions only, for the retriever's
    anaphora resolution). See `app/db/conversation.py` for why they differ.

    The user's turn is appended *before* retrieval so that a retrieval failure or a
    disconnected stream still leaves the question visible in the thread. Appending
    after the fact would lose exactly the questions worth debugging.

    A conversation id that no longer exists is treated as a fresh thread rather than
    a 404: the client's view is stale, and the useful response is to answer the
    question. This is also what makes a deleted conversation leave no orphaned turn —
    the turn is simply not written.
    """
    conversation_id = payload.conversation_id
    if conversation_id is None:
        return ConversationHistory()

    try:
        get_conversation(session, conversation_id)
    except NotFoundError:
        log.info(
            "chat on unknown conversation; starting a new thread",
            extra={"conversation_id": conversation_id},
        )
        return ConversationHistory()

    # The conversation exists, so this exchange is persistable regardless of whether
    # history is in play. Tying persistence to the history window would mean turning
    # `chat_history_turns` to 0 also silently stopped recording turns, losing the
    # FR-30 evidence with what is meant to be a context-window setting.
    persistable = True

    # History is read *before* the current question is appended, so the model and the
    # retriever both see strictly prior turns.
    #
    # The order matters and was previously wrong. Appending first and slicing the tail
    # off afterwards looks equivalent but leaks the current question into the
    # history as a prior user turn — which means the prompt carries the same
    # sentence twice, once as context and once as the question, and the retriever
    # resolves every anaphora against the question being asked. Reading first makes
    # the exclusion structural instead of something a future edit can un-break.
    #
    # The two windows are independent. `chat_history_turns` bounds the generation
    # prompt; `retrieval_memory_turns` is the last N user questions the retriever
    # may resolve a follow-up against. Zero on either side disables that view only.
    message_limit = settings.chat_history_turns
    query_limit = settings.retrieval_memory_turns
    prior_messages = (
        message_history(session, conversation_id, limit=message_limit)
        if message_limit > 0
        else []
    )
    prior_queries = (
        query_history(session, conversation_id, limit=query_limit)
        if query_limit > 0
        else []
    )

    # Appended now, still before retrieval, so a retrieval failure or a disconnected
    # stream leaves the question visible in the thread. Those are exactly the
    # questions worth having on record.
    append_turn(
        session,
        conversation_id,
        role=TurnRole.USER,
        content=payload.message,
    )
    # Committed here, not left to the request teardown, for two reasons. The question
    # must be durable even if generation never happens. And the streaming generator
    # writes the assistant turn through its *own* session; leaving this one holding
    # an open write transaction would block that write outright on SQLite, which
    # allows a single writer.
    session.commit()

    return ConversationHistory(
        persistable=persistable,
        messages=prior_messages,
        queries=prior_queries,
    )


@router.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    request: Request,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    provider: GenerationProvider = Depends(get_generation_provider_dep),
) -> StreamingResponse:
    """Retrieve, stream a grounded answer, and persist the query log."""
    started = time.perf_counter()

    # FR-34: reject before any provider call, with a user-safe message. Delegated to
    # the guardrail rather than kept inline so the limit, the message, and the
    # exception type have one definition shared with every future chat entry point.
    enforce_message_size(payload.message, limit=settings.chat_max_query_chars)

    # FR-31: after the size cap (an oversized payload should not consume a rate
    # limit slot — that would let one client deny service to its own retries) and
    # before retrieval.
    get_rate_limiter(
        settings.chat_rate_limit_requests, settings.chat_rate_limit_window_seconds
    ).check(client_identity(request))

    # FR-23: prior turns are resolved *before* retrieval, because retrieval is what
    # anaphora resolution exists to fix. A conversation id that does not exist is
    # not an error: the client may hold a stale id from a deleted thread, and
    # starting a fresh thread beats rejecting the question. Logged rather than
    # surfaced, since there is nothing the user can do about it.
    history = _conversation_history(session, payload, settings)

    style = AnswerStyle(payload.answer_style)
    retriever = Retriever(session, settings=settings)
    # Only user turns go to the retriever; `query_history` exists for exactly this
    # split. See app/db/conversation.py.
    result = retriever.retrieve(
        payload.message,
        history=history.queries or None,
    )

    # On threshold abstention `result.candidates` is empty by design; the
    # pre-threshold ranking is still the honest "here is what I looked at" for the
    # sources panel (FR-20), so prefer it when present.
    shown = result.candidates or result.abstain_candidates
    sources = sources_payload(shown)
    passages = _passages(shown)

    trace_id = getattr(request.state, "trace_id", None)
    context_texts = [p.text for p in passages]

    def event_stream() -> Iterator[str]:
        # `sources` first, always (FR-20), including for a refusal.
        yield _sse("sources", {"sources": sources})

        log_session = get_session_factory(settings)()
        state = _StreamState(abstained=result.abstained)
        answer_parts: list[str] = []

        def persist() -> str:
            """Write the QueryLog row and return its id (FR-28).

The assistant turn is written in the same transaction as the log row,
            which is why `QueryLog.conversation_id` is a plain string rather than a
            foreign key. Only the assistant turn is appended here - the user turn was
            written before retrieval, so an unanswered question still shows up in the
            thread next to a retrieval failure, instead of leaving a gap where the
            user can no longer see what they asked.
            """
            state.persist_attempted = True
            answer_text = "".join(answer_parts)
            entry = record_query(
                log_session,
                trace_id=trace_id,
                conversation_id=payload.conversation_id,
                original_query=payload.message,
                result=result,
                retrieved=shown,
                model=settings.generation_model,
                prompt_version=PROMPT_VERSION,
                tokens_in=_count_tokens(
                    [payload.message, *context_texts]
                    + [t["content"] for t in history.messages]
                ),
                tokens_out=_count_tokens([answer_text]) if answer_text else 0,
                ttft_ms=state.ttft_ms,
                total_ms=int((time.perf_counter() - started) * 1000),
                citations_stripped=state.citations_stripped,
                abstained=state.abstained,
            )
            # The assistant turn is written *after* the log row so it can carry the
            # query id. Written before, `query_id` would have to be backfilled
            # afterwards, and a reloaded thread would show no feedback buttons for
            # any answer (FR-29). Both rows commit together.
            #
            # `persistable`, not `conversation_id is not None`: a stale id must not
            # produce an orphan assistant turn pointing at a conversation that does
            # not exist. The user turn was skipped for the same reason.
            if history.persistable and payload.conversation_id is not None:
                append_turn(
                    log_session,
                    payload.conversation_id,
                    role=TurnRole.ASSISTANT,
                    content=answer_text,
                    query_id=entry.query_id,
                    # The cited chunks, in the order the markers appear in the
                    # answer. `sources` is already ordered by retrieval rank, which
                    # is the order the assembler numbers markers in, so the two
                    # agree; the citation validator guarantees every surviving
                    # marker refers to a real index.
                    citations=[str(s["chunk_id"]) for s in sources],
                    abstained=state.abstained,
                )
            log_session.commit()
            state.recorded = True
            return entry.query_id

        try:
            if result.abstained or not passages:
                # architecture.md 3.3: short-circuit to refusal with no LLM call.
                text = refusal_text(RefusalReason.NO_EVIDENCE)
                state.ttft_ms = int((time.perf_counter() - started) * 1000)
                state.abstained = True
                answer_parts.append(text)
                yield _sse("token", {"text": text})
            else:
                messages = build_messages(
                    payload.message,
                    passages,
                    style,
                    nonce=new_nonce(),
                    # FR-23: both roles, so the model can resolve "how long is
                    # that?" against what it said last turn. Inserted before the
                    # context block by `build_messages`, which keeps the retrieved
                    # passages as the most recent thing the model reads.
                    history=history.messages or None,
                )
                preset = STYLE_PRESETS[style]
                max_tokens = min(preset.max_tokens, settings.generation_max_tokens)
                assembler = AnswerAssembler(len(passages))
                stream = provider.stream(
                    messages,
                    model=settings.generation_model,
                    max_tokens=max_tokens,
                    temperature=settings.generation_temperature,
                )
                for chunk in assembler.run(stream):
                    if chunk.kind is ChunkKind.CITATION_WARNING:
                        yield _sse(
                            "citation_warning", {"stripped_count": chunk.stripped}
                        )
                        continue
                    if state.ttft_ms is None:
                        state.ttft_ms = int((time.perf_counter() - started) * 1000)
                    answer_parts.append(chunk.text)
                    yield _sse("token", {"text": chunk.text})
                state.abstained = assembler.abstained
                state.citations_stripped = assembler.stripped_count

            query_id = _safe_persist(persist, log_session, state, suppress=False)
            yield _sse(
                "done",
                {
                    "query_id": query_id,
                    "abstained": state.abstained,
                    "ttft_ms": state.ttft_ms,
                },
            )
        except AppError as exc:
            # Typed errors carry a user-safe message (NFR-5). The code is derived
            # from the exception type by `safe_error_payload` so the SSE `error`
            # event and the JSON error path cannot drift apart — a client that
            # switches on `code` has one contract, not two.
            log_session.rollback()
            log.warning("chat stream error", extra={"detail": exc.detail})
            payload_out = safe_error_payload(exc)
            yield _sse("error", payload_out)
        finally:
            if not state.persist_attempted:
                # Client disconnected, or the stream failed, before `done`. Record the
                # partial answer so the query is not lost from the improvement loop
                # (FR-30). Suppressed unconditionally: we are already unwinding an
                # exception, and raising here would replace it with a database error.
                _safe_persist(persist, log_session, state, suppress=True)
            log_session.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering so tokens are not held by an intermediary
            # and delivered in one burst (architecture.md 8).
            "X-Accel-Buffering": "no",
        },
    )
