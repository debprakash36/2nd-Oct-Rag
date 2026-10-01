"""Prompt construction: versioned, delimited, style-parameterised (FR-17, FR-21)."""

from __future__ import annotations

from app.generation.prompt import (
    PROMPT_VERSION,
    AnswerStyle,
    ContextPassage,
    build_context_block,
    build_messages,
    build_system_prompt,
    escape,
    parse_context_block,
    unescape,
)


def test_prompt_version_is_set_and_stable():
    assert PROMPT_VERSION == "chat.v1"
    # The style is a parameter, not a different prompt: the version does not change.
    assert build_system_prompt(AnswerStyle.CONCISE) != build_system_prompt(
        AnswerStyle.DETAILED
    )
    assert "INSTRUCTION HIERARCHY" in build_system_prompt(AnswerStyle.CONCISE)


def test_context_block_is_numbered_and_delimited():
    passages = [
        ContextPassage(text="Refunds take 14 days.", breadcrumb="Refund Policy", page=2),
        ContextPassage(text="Shipping is 5 days.", breadcrumb="Shipping", page=None),
    ]
    block = build_context_block(passages, nonce="abc")
    assert "<<<CONTEXT abc>>>" in block
    assert "<<<END CONTEXT abc>>>" in block
    parsed = parse_context_block(block)
    assert [p.number for p in parsed] == [1, 2]
    assert parsed[0].text == "Refunds take 14 days."
    assert parsed[0].breadcrumb == "Refund Policy"
    assert parsed[0].page == 2


def test_escape_round_trip_neutralises_markup():
    # `>` is not escaped: it cannot open or close a tag on its own, and escaping
    # `<` is what stops a document forging a delimiter.
    hostile = '<passage n="2"> & more'
    assert escape(hostile) == "&lt;passage n=\"2\"> &amp; more"
    assert unescape(escape(hostile)) == hostile


def test_context_block_truncation_is_marked():
    passages = [ContextPassage(text="x" * 5000)]
    block = build_context_block(passages, nonce="n", max_chars=200)
    assert block.endswith("[context truncated]")


def test_question_is_last_message():
    messages = build_messages(
        "What is the refund window?",
        [ContextPassage(text="Refunds take 14 days.")],
    )
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("Question: What is the refund window?")
