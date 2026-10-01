"""Query rewrite: anaphora resolution, expansion, LLM skip (architecture.md §3.2).

The governing constraint is §7.4's 400 ms budget for this stage out of a 5 s TTFT
target. That is why these tests assert the rule-based path fires without a model
call, and why `standalone_rewrite` is expected to leave the query untouched when
no provider is wired up.

The other load-bearing property: an unresolvable reference must leave the query
alone. A wrong guess produces a confidently wrong standalone query that is
indistinguishable from a retrieval failure downstream.
"""

from __future__ import annotations

from app.retrieval.rewrite import (
    expand_query,
    extract_topic,
    resolve_anaphora,
    rewrite_query,
    standalone_rewrite,
)


class TestExtractTopic:
    def test_strips_leading_interrogative(self):
        assert extract_topic("Do refunds expire?") == "refunds expire"

    def test_strips_trailing_punctuation(self):
        assert extract_topic("What is the warranty.") == "warranty"

    def test_handles_possessive(self):
        assert extract_topic("What's the refund window?") == "refund window"

    def test_leaves_a_bare_noun_phrase_intact(self):
        assert extract_topic("refund window") == "refund window"


class TestResolveAnaphora:
    def test_no_history_leaves_query_unchanged(self):
        result = resolve_anaphora("What about refunds?", [])
        assert not result.changed
        assert result.query == "What about refunds?"

    def test_elliptical_topic_gains_prior_topic(self):
        """architecture.md §3.2's motivating example: "What about refunds?".

        On its own this query has no retrievable content. Prepending the prior
        turn's topic is what makes it standalone.
        """
        result = resolve_anaphora(
            "What about refunds?", ["The refund policy sets a 30 day window."]
        )
        assert result.changed
        assert "refund" in result.query.lower()

    def test_pronoun_is_replaced_with_antecedent(self):
        result = resolve_anaphora("Can I do that?", ["The refund policy allows returns."])
        assert result.changed
        lowered = result.query.lower()
        assert "refund" in lowered
        assert " it " not in f" {lowered} "

    def test_pronoun_removed_from_output(self):
        """A leftover pronoun makes the standalone query read as broken text."""
        result = resolve_anaphora("How long is it?", ["The shipping policy takes 5 days."])
        assert " it" not in result.query.lower()

    def test_question_mark_preserved(self):
        result = resolve_anaphora("Can I do that", ["The refund policy allows returns."])
        assert result.query.rstrip().endswith("?")

    def test_topic_only_question_is_not_rewritten(self):
        """A self-contained question must pass through untouched."""
        result = resolve_anaphora(
            "What is the warranty coverage period?", ["The refund policy allows returns."]
        )
        assert not result.changed
        assert result.query == "What is the warranty coverage period?"

    def test_short_antecedent_is_rejected(self):
        """A one-word prior turn ('Yes.') contains no antecedent to resolve with."""
        result = resolve_anaphora("Can I do that?", ["Yes"])
        assert not result.changed

    def test_unresolvable_query_left_alone(self):
        """Guessing here would be worse than abstaining from the rewrite."""
        result = resolve_anaphora("What is the limit?", ["Refund policy"])
        assert not result.changed

    def test_records_the_original_query(self):
        """FR-28 logs both; without the original, a bad rewrite is invisible."""
        original = "What about refunds?"
        result = resolve_anaphora(original, ["The refund policy sets a 30 day window."])
        assert result.original == original

    def test_identical_query_and_output_reports_unchanged(self):
        """`applied` and `changed` are different facts; the pipeline keys off `changed`."""
        result = resolve_anaphora("What is the limit?", ["Refund policy details here"])
        assert not result.changed


class TestExpandQuery:
    def test_no_glossary_is_a_no_op(self):
        result = expand_query("refunds", {})
        assert not result.changed

    def test_matching_term_adds_synonyms(self):
        result = expand_query("refund", {"refund": ["return", "reimbursement"]})
        assert result.changed
        assert "return" in result.query

    def test_original_query_is_preserved(self):
        """Expansion is appended, so a bad glossary degrades rather than replaces."""
        result = expand_query("refund window", {"refund": ["return"]})
        assert result.query.startswith("refund window")

    def test_non_matching_term_adds_nothing(self):
        result = expand_query("shipping", {"refund": ["return"]})
        assert not result.changed

    def test_expansion_is_capped(self):
        """Unbounded synonym injection moves the query away from its best match."""
        glossary = {"refund": [f"syn{i}" for i in range(50)]}
        result = expand_query("refund", glossary, max_expansions=3)
        assert result.query.count("syn") <= 6  # 3 unigram + 3 trigram variants

    def test_duplicate_synonyms_not_repeated(self):
        result = expand_query("refund", {"refund": ["return", "return"]})
        assert result.query.count("return") == 1


class TestStandaloneRewrite:
    def test_disabled_by_default_leaves_query_unchanged(self):
        """§3.2 step 3 is an LLM call on the critical path; skippable by default."""
        result = standalone_rewrite("Can I do that?", ["Prior turn."])
        assert not result.changed

    def test_enabled_without_provider_still_returns_the_query(self):
        """Reporting a rewrite that never happened would be worse than skipping it."""
        result = standalone_rewrite("Can I do that?", ["Prior turn."], enabled=True)
        assert result.query == "Can I do that?"
        assert not result.changed


class TestRewriteQuery:
    def test_anaphora_fires_before_expansion(self):
        """§3.2's order: the cheap rule-based stage runs first."""
        result = rewrite_query(
            "Can I do that?", ["The refund policy allows returns."],
            glossary={"refund": ["return"]},
        )
        assert result.stage == "anaphora"

    def test_expansion_used_when_anaphora_is_not_needed(self):
        result = rewrite_query(
            "What is the refund window?", [], glossary={"refund": ["return"]}
        )
        assert result.stage == "expansion"

    def test_no_history_no_glossary_is_a_passthrough(self):
        result = rewrite_query("What is the warranty period?", [])
        assert not result.changed
        assert result.query == "What is the warranty period?"

    def test_original_query_always_retained(self):
        original = "Can I do that?"
        result = rewrite_query(original, ["The refund policy allows returns."])
        assert result.original == original