"""Query rewrite: make a multi-turn question standalone (architecture.md §3.2).

§3.2 orders the work deliberately: anaphora resolution first and rule-based,
query expansion second, LLM rewrite last. The reason is latency — §7.4 budgets
400 ms for this stage out of a 5 s TTFT target, and a rule-based path handles the
common case ("what about refunds?") without a network call at all.

Three stages, each skipped when the previous one produced a standalone query:

1. `resolve_anaphora` — replace pronouns and elliptical references with
   antecedents drawn from prior turns. Pure string work, no model call.
2. `expand_query` — add glossary synonyms when a glossary is configured.
3. `standalone_rewrite` — optional LLM rewrite, off by default. When enabled it
   is the only rewrite stage that costs latency.

The original query and the rewritten query are both returned so FR-28 can log
them separately. A bad rewrite is otherwise invisible in production and looks
exactly like a retrieval failure.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Pronouns that usually refer to the previous turn's subject. Kept explicit
#: rather than using a pronoun list: a mis-resolved pronoun produces a confidently
#: wrong standalone query, which is worse than leaving the question unresolved.
_PRONOUNS = {
    "it",
    "its",
    "they",
    "them",
    "their",
    "theirs",
    "this",
    "that",
    "these",
    "those",
}

#: Openers whose subject is the previous turn's topic. "What about X?" is a topic
#: shift rather than a reference, so the antecedent is the topic noun, not a pronoun.
_TOPIC_OPENER_RE = re.compile(
    r"^\s*(?:and\s+)?(?:what\s+about|how\s+about|what'?s|whats)\s+(?P<topic>.+?)\s*\??$",
    re.IGNORECASE,
)

#: Words dropped when reducing a prior question to a bare topic.
_QUESTION_LEAD_RE = re.compile(
    r"^\s*(?:do|does|did|is|are|was|were|can|could|would|should|will|shall|may|might|"
    r"tell\s+me\s+about|explain|describe|what\s+is|what\s+are|what'?s)\s+",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "do", "does", "did", "can", "could", "would", "about", "what", "how", "why",
    "when", "where", "which", "who", "i", "you", "we", "it", "that", "this",
}

#: Leading determiners stripped when reducing a turn to a bare topic phrase.
#: "What is the warranty?" -> "warranty". Without this the antecedent carries a
#: leading article into every rewritten query, which both reads badly and adds a
#: term with no retrieval signal.
_LEADING_DETERMINER_RE = re.compile(r"^(?:the|a|an|my|our|your|their)\s+", re.IGNORECASE)

#: Minimum tokens in a prior turn for it to be worth treating as a topic source.
#: A one-word reply ("yes") contains no antecedent.
_MIN_ANTECEDENT_TOKENS = 2

#: Longest antecedent phrase substituted in. An unbounded antecedent produces
#: runaway queries where each turn prepends the last.
_MAX_ANTECEDENT_TOKENS = 6


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of the rewrite stage."""

    query: str
    applied: bool = False
    stage: str = "none"
    original: str = ""

    @property
    def changed(self) -> bool:
        return self.applied and self.query != self.original


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _content_tokens(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS]


def extract_topic(text: str) -> str:
    """Reduce a turn to the noun phrase it is about.

    Strips a leading interrogative and trailing punctuation so "Do refunds expire?"
    yields "refunds expire". Returning the whole turn instead would prepend a
    question to a question, which reads as nonsense to the retriever and dilutes
    the terms that actually matter.
    """
    stripped = text.strip().rstrip("?.").strip()
    stripped = _QUESTION_LEAD_RE.sub("", stripped).strip()
    stripped = _LEADING_DETERMINER_RE.sub("", stripped).strip()
    return stripped.rstrip("?.").strip()


def resolve_anaphora(
    query: str,
    history: Sequence[str] | None = None,
    *,
    max_antecedent_tokens: int = _MAX_ANTECEDENT_TOKENS,
) -> RewriteResult:
    """Replace references in `query` with antecedents from prior turns.

    Rule-based, per §3.2 step 1, so the common multi-turn case costs no model
    call. Two forms are handled:

    * Pronoun reference — "Can I do that?" resolves to the newest usable prior topic.
    * Elliptical topic shift — "What about refunds?" prepends that topic,
      because the topic word alone is often too generic to retrieve.

    Returns the query unchanged when nothing resolvable is found. Guessing here
    would be worse than abstaining from the rewrite: a wrong standalone query is
    indistinguishable from a retrieval bug when it reaches the eval harness.
    """
    history = [h for h in (history or []) if h and h.strip()]
    if not history:
        return RewriteResult(query=query, original=query)

    # Newest usable turn wins. A short reply ("yes") has no antecedent, so the
    # window is walked backward until a turn long enough to name a topic is found.
    # Stopping at history[-1] would drop the rest of the retrieval memory window.
    antecedent = ""
    for turn in reversed(history):
        candidate = extract_topic(turn)
        if len(_tokens(candidate)) >= _MIN_ANTECEDENT_TOKENS:
            antecedent = candidate
            break
    antecedent_tokens = _tokens(antecedent)
    if len(antecedent_tokens) < _MIN_ANTECEDENT_TOKENS:
        return RewriteResult(query=query, original=query)

    antecedent_phrase = " ".join(antecedent_tokens[:max_antecedent_tokens])

    # Elliptical topic shift: "what about refunds?" keeps its own topic and gains
    # the prior one as context.
    match = _TOPIC_OPENER_RE.match(query)
    if match:
        topic = match.group("topic").strip().rstrip("?.").strip()
        if topic and not _contains_pronoun(topic):
            rewritten = f"{antecedent_phrase} {topic}"
            log.debug("rewrote elliptical topic", extra={"rewritten": rewritten})
            return RewriteResult(
                query=rewritten, applied=True, stage="anaphora", original=query
            )

    query_tokens = _tokens(query)
    if _contains_pronoun(query):
        # Drop the pronoun itself before prepending; leaving "it" in place makes
        # the standalone query read as broken text to a lexical index.
        cleaned = " ".join(t for t in query_tokens if t not in _PRONOUNS)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            cleaned = antecedent_phrase
        elif not cleaned.endswith("?"):
            # Preserve the interrogative shape; a fragment with no question form
            # retrieves differently and reads as a fragment in the logs.
            cleaned = f"{cleaned}?"
        rewritten = f"{antecedent_phrase} {cleaned}"
        log.debug("rewrote pronoun reference", extra={"rewritten": rewritten})
        return RewriteResult(query=rewritten, applied=True, stage="anaphora", original=query)

    return RewriteResult(query=query, original=query)


def _contains_pronoun(text: str) -> bool:
    return any(token in _PRONOUNS for token in _tokens(text))


def expand_query(
    query: str,
    glossary: Mapping[str, Iterable[str]] | None = None,
    *,
    max_expansions: int = 4,
) -> RewriteResult:
    """Add configured synonyms from a glossary (architecture.md §3.2 step 2).

    The glossary is empty in v1 — architecture.md §3.2 marks it as dependent on
    open question 9. The hook exists so adding one is a configuration change, and
    expansion is appended after the original query so a failure to expand never
    degrades the original terms.

    Expansion is capped: unbounded synonym injection pulls in terms that dilute
    the signal and, for a vector index, moves the query away from the passage it
    was closest to.
    """
    if not glossary:
        return RewriteResult(query=query, original=query)

    present = {t for t in _tokens(query)}
    additions: list[str] = []
    for term, synonyms in glossary.items():
        if term.lower() in present:
            additions.extend(synonyms)
    if not additions:
        return RewriteResult(query=query, original=query)

    seen: set[str] = set()
    unique: list[str] = []
    for synonym in additions:
        key = synonym.lower()
        if key not in present and key not in seen:
            seen.add(key)
            unique.append(synonym)
        if len(unique) >= max_expansions:
            break

    if not unique:
        return RewriteResult(query=query, original=query)

    rewritten = f"{query} {' '.join(unique)}"
    log.debug("expanded query", extra={"added": unique})
    return RewriteResult(query=rewritten, applied=True, stage="expansion", original=query)


def standalone_rewrite(
    query: str,
    history: Sequence[str] | None = None,
    *,
    enabled: bool = False,
) -> RewriteResult:
    """Optional LLM rewrite to a self-contained question (§3.2 step 3).

    Disabled by default. When `enabled` is True but no LLM provider is wired up,
    this returns the query unchanged and logs once. That is deliberate: silently
    skipping the requested stage would report a rewrite that never happened.

    Phase 3 wires the generation provider in here. The signature is fixed now so
    the retriever does not change when it does.
    """
    if not enabled:
        return RewriteResult(query=query, original=query)
    log.info(
        "standalone rewrite requested but no rewrite provider is configured; "
        "returning the query unchanged"
    )
    return RewriteResult(query=query, original=query)


def rewrite_query(
    query: str,
    history: Sequence[str] | None = None,
    *,
    glossary: Mapping[str, Iterable[str]] | None = None,
    allow_llm: bool = False,
) -> RewriteResult:
    """Run the three rewrite stages in order, stopping at the first that helps.

    §3.2 specifies this ordering. Each stage is skipped when an earlier one has
    already produced a standalone query — running all three would spend latency
    for a rewrite the first stage already made unnecessary.
    """
    result = resolve_anaphora(query, history)
    if not result.changed:
        result = expand_query(result.query, glossary)
    if not result.changed:
        result = standalone_rewrite(result.query, history, enabled=allow_llm)
    return result