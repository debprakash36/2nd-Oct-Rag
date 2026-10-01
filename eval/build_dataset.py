"""Build the labelled eval set from the corpus (implementation.md 2.1).

architecture.md §3.3 is blunt about this artifact: the relevance threshold is "a
measured parameter, not a guess", and §10 check 1-4 are all comparisons that need
labelled data. The eval set is therefore the *input* to Phase 2's exit gate, not
a deliverable of it, and it is built first (implementation.md 2.1) so the numbers
mean something.

## What counts as a label here

A label is the set of `chunk_id`s that contain the answer, derived from the corpus
generator's own ground truth — the topic, the parameter value, and the document
reference are all known because the documents are generated from templates. That
is a stronger form of labelling than a human annotator working from the text
alone: there is no annotation disagreement to adjudicate and no risk of the
labels inheriting the retriever's own bias, which is the failure mode when an
eval set is built by running a retriever and keeping what it found.

## The four classes

`implementation.md` §2.1 requires ≥20% expected-abstain, ≥10% multi-turn, and
≥10% exact-identifier. Those minimums are enforced here rather than assumed:

* `direct` — a paraphrase of the topic sentence. Should be findable by the vector
  half.
* `identifier` — a question naming `POL-XXX-nnn` or an owner surname. Should be
  findable by *keyword* search and is the class that proves FR-12.
* `multiturn` — a follow-up that is unanswerable alone ("And how long is that
  window?"). Exercises anaphora resolution; labels the same chunks as its turn 1.
* `abstain` — asks for something the corpus does not contain. The correct
  behaviour is refusal, and a retriever that answers these is the failure mode
  FR-14 exists to prevent.

Run with `--questions 200` (the §10 quick-start figure). Deterministic: a fixed
seed, and labels resolved by substring match against the chunk store, so the
same corpus always yields the same dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Allow running as a script from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentState
from app.db.session import session_scope
from scripts.make_corpus import OWNERS, TOPICS

#: Class weights, chosen to clear the §2.1 minimums with margin while leaving the
#: answerable majority dominant: an eval set that is mostly abstains measures the
#: threshold, not retrieval.
#:
#: `abstain` is weighted at 0.26, above its 0.20 minimum. The reason is specific:
#: the abstain class is scored by *refusing*, and refusal is cheap for a
#: retriever that simply fails to find anything. A set where 10% of questions are
#: unanswerable lets a broken retriever score well on them, so the class is
#: over-represented deliberately and the pass condition requires it to abstain
#: rather than merely to fail to retrieve.
_WEIGHTS = {
    "direct": 0.50,
    "identifier": 0.12,
    "multiturn": 0.12,
    "abstain": 0.26,
}

#: Topics the corpus does not contain. Used for `abstain` questions, which must be
#: genuinely unanswerable — an "abstain" question that is actually answerable
## punishes the retriever for being right.
_OUT_OF_SCOPE_TOPICS = [
    "office lease renewal terms",
    "petrol engine maintenance intervals",
    "amateur radio exam syllabuses",
    "medieval cathedral construction contracts",
    "professional surfing competition rules",
    "beekeeping migratory apiary regulations",
    "antique clock restoration standards",
    "orchestra seating arrangements",
    "dry stone wall construction methods",
    "competitive chess tournament schedules",
    "volcanic soil fertilisation trials",
    "whale migration survey methodology",
]


@dataclass
class EvalQuestion:
    """One labelled question."""

    qid: str
    question: str
    #: Classes: direct | identifier | multiturn | abstain
    category: str
    #: True when the correct behaviour is to refuse.
    expect_abstain: bool
    #: chunk_ids that answer the question. Empty for `abstain`.
    relevant_chunk_ids: list[str] = field(default_factory=list)
    #: Prior turn text, for `multiturn`.
    history: list[str] = field(default_factory=list)
    #: The phrase used to locate the answer in the chunk store. Kept in the
    #: dataset so a label can be re-derived and audited, not just trusted.
    evidence: str = ""


@dataclass
class _LabelledDoc:
    """A corpus document plus the facts needed to write questions about it."""

    doc_id: str
    filename: str
    title: str
    topic: int
    index: int
    days: int
    #: Subject scope, e.g. "Digital Goods". Part of the title and the body, and
    #: the discriminator that makes a title-plus-scope question answerable by
    #: exactly one document.
    scope: str
    #: Distinctive sentence from the topic template, with `{days}` and `{scope}`
    #: already substituted. Located per chunk below.
    topic_phrase: str


def _load_documents(session: Session) -> list[_LabelledDoc]:
    """Read live documents and recover which corpus template produced each.

    The generator writes the reference `POL-XXX-nnn` and the owner surname into the
    document text, so the template index is recoverable from the stored content
    without trusting the filename. Matching on content rather than filename
    matters because the eval set must stay correct if the corpus is regenerated
    with different naming.

    `topic_phrase` is the topic sentence **with both placeholders substituted** --
    the scope and the figure. Carrying the raw template instead meant the label
    search looked for the literal text "within {days} days", matched nothing, and
    fell back to matching the first word of the sentence. Every direct and
    multiturn label was therefore "any chunk mentioning 'customers'", which is why
    those questions had 2-3 labels each and scored at chance.
    """
    from scripts.make_corpus import TOPIC_CODES, scope_for

    rows = session.execute(
        select(Document.doc_id, Document.filename, Document.cleaned_text)
        .where(Document.state == DocumentState.LIVE)
    ).all()

    docs: list[_LabelledDoc] = []
    for doc_id, filename, text in rows:
        if not text:
            continue
        topic, index = _identify(text, TOPIC_CODES)
        if topic is None or index is None:
            continue
        scope = scope_for(topic, index)
        days = _figure_of(text, topic, scope)
        if days is None:
            continue
        docs.append(
            _LabelledDoc(
                doc_id=doc_id,
                filename=filename,
                title=TOPICS[topic][0],
                topic=topic,
                index=index,
                days=days,
                scope=scope,
                topic_phrase=TOPICS[topic][1].format(days=days, scope=scope),
            )
        )
    return docs


def _identify(text: str, codes: Sequence[str]) -> tuple[int | None, int | None]:
    """Recover (topic, document index) from the `POL-<CODE>-<nnn>` reference."""
    import re

    code_to_topic = {code: i for i, code in enumerate(codes)}
    match = re.search(r"POL-([A-Z]{3})-(\d{3})", text)
    if not match:
        return None, None
    topic = code_to_topic.get(match.group(1))
    index = int(match.group(2))
    return topic, index


def _figure_of(text: str, topic: int, scope: str) -> int | None:
    """Recover the topic's figure from the document text.

    Matched on the scope rather than taken as "the first integer in the file",
    because the first integer is the document's own reference number
    (`POL-RFD-042` -> 42), not the parameter a direct question asks about. The
    scope-anchored pattern is also self-checking: if the corpus is regenerated
    with a different template, this returns None and the document is skipped
    rather than silently labelled with the wrong figure.
    """
    import re

    from scripts.make_corpus import TOPICS

    pattern = TOPICS[topic][1]
    regex = re.escape(pattern).replace(r"\{days\}", r"(\d+)").replace(
        r"\{scope\}", re.escape(scope)
    )
    match = re.search(regex, text)
    return int(match.group(1)) if match else None


def _chunks_for(session: Session, doc_id: str) -> list[Chunk]:
    return list(
        session.execute(
            select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
        )
        .scalars()
        .all()
    )


def _label_by_phrase(chunks: Sequence[Chunk], needle: str) -> list[str]:
    """chunk_ids whose text contains `needle`.

    Substring match against the authoritative chunk store rather than through the
    retriever. This is the whole point: a label obtained by running retrieval would
    be circular, and any retriever bug would be recorded as ground truth.
    """
    lowered = needle.lower()
    return [c.chunk_id for c in chunks if lowered in c.text.lower()]


def _direct_question(doc: _LabelledDoc, rng: random.Random) -> tuple[str, str]:
    """A paraphrase of the topic sentence. Returns (question, evidence).

    Every template names the document's **scope**, not just its title. A question
    that named only the title would have one correct answer per same-titled
    document -- a dozen, in this corpus -- so labelling a single one of them
    measures which sibling the retriever happened to rank first. Naming the scope
    is also the more realistic shape: a user who wants the digital-goods refund
    window says so.

    The scope must not be quoted verbatim from the body in every template. Two
    templates reword it so the lexical reranker is not simply matching a string
    it can copy, and the vector stage has to do some work.
    """
    title, scope = doc.title, doc.scope
    templates = [
        f"Under the {title} covering {scope}, what is the stated limit?",
        f"How long does the {title} allow for {scope}?",
        f"What timeframe does the {title} set out for {scope}?",
        # Lower-cased scope: "Digital Goods" -> "digital goods". Keeps the
        # discriminator but not the capitalisation the document uses, so the
        # match is not a plain case-sensitive copy of the body text.
        f"State the key figure the {title} gives for {scope.lower()}.",
        f"What is the figure where the {title} applies to {scope.lower()}?",
        f"The {title} sets one figure for {scope.lower()} — what is it?",
    ]
    question = rng.choice(templates)
    return question, doc.topic_phrase


def _identifier_question(doc: _LabelledDoc, rng: random.Random) -> tuple[str, str]:
    """A question naming a unique identifier (FR-12's exact-match case)."""
    from scripts.make_corpus import TOPIC_CODES

    ref = f"POL-{TOPIC_CODES[doc.topic]}-{doc.index:03d}"
    owner = OWNERS[doc.index % len(OWNERS)]
    if rng.random() < 0.5:
        return f"Who is the accountable owner for {ref}?", f"owner for {ref} is {owner}"
    return (
        f"What is the effective date of document {ref}?",
        f"{ref} and takes effect",
    )





def build_dataset(session: Session, count: int, seed: int = 20240101) -> list[EvalQuestion]:
    """Generate `count` labelled questions from the ingested corpus."""
    rng = random.Random(seed)
    docs = _load_documents(session)
    if not docs:
        raise SystemExit(
            "no labelled documents found. Ingest the corpus first:\n"
            "  python scripts/make_corpus.py --out ./samples\n"
            "  # then upload ./samples through POST /admin/documents"
        )

    chunk_cache: dict[str, list[Chunk]] = {
        d.doc_id: _chunks_for(session, d.doc_id) for d in docs
    }

    questions: list[EvalQuestion] = []
    classes = list(_WEIGHTS)
    weights = [_WEIGHTS[c] for c in classes]

    for n in range(count):
        category = rng.choices(classes, weights=weights, k=1)[0]
        doc = rng.choice(docs)
        chunks = chunk_cache[doc.doc_id]
        if not chunks:
            continue

        if category == "abstain":
            scope = rng.choice(_OUT_OF_SCOPE_TOPICS)
            question = (
                f"What are the requirements for {scope}?"
                if rng.random() < 0.5
                else f"Does the corpus say anything about {scope}?"
            )
            questions.append(
                EvalQuestion(
                    qid=f"q{n:04d}",
                    question=question,
                    category="abstain",
                    expect_abstain=True,
                    evidence="(no supporting content in corpus by construction)",
                )
            )
            continue

        if category == "identifier":
            question, evidence = _identifier_question(doc, rng)
            relevant = _label_by_phrase(chunks, evidence)
            category_out = "identifier"
            history: list[str] = []
        elif category == "multiturn":
            # Turn 1 establishes the document by title *and* scope; turn 2 drops
            # both and refers back with "that". Resolution therefore has to
            # recover the scope from history, which is the property G3 wants
            # tested. When turn 2 is resolved to just the title it would be
            # ambiguous across a dozen siblings again.
            title, scope = doc.title, doc.scope
            first = rng.choice(
                [
                    f"What is the {title} for {scope}?",
                    f"Tell me about the {title} covering {scope}.",
                    f"What figure does the {title} give for {scope}?",
                ]
            )
            question = rng.choice(
                [
                    "And how long is that window?",
                    "How long does that period last?",
                    "What is the timeframe for that?",
                    "Is there a limit on that?",
                ]
            )
            evidence = doc.topic_phrase
            relevant = _label_by_phrase(chunks, evidence)
            category_out = "multiturn"
            history = [first]
        else:
            question, evidence = _direct_question(doc, rng)
            relevant = _label_by_phrase(chunks, evidence)
            category_out = "direct"
            history = []

        if not relevant:
            # The topic sentence did not survive chunking intact: a chunk
            # boundary can split it, and the wrapping in make_corpus.py breaks
            # lines mid-sentence. Fall back to the document's scope, which is the
            # one term guaranteed to be unique within a document and is what the
            # question names. Falling back to the sentence's leading token would
            # instead match every chunk sharing a common word -- the bug that
            # made direct labels 2-3 chunks wide.
            relevant = _label_by_phrase(chunks, doc.scope)
        if not relevant:
            # No locatable answer at all. Skip rather than emit an unlabelled
            # question: an answerable question with no label counts as a miss
            # forever, which understates recall and cannot be diagnosed later.
            continue

        questions.append(
            EvalQuestion(
                qid=f"q{n:04d}",
                question=question,
                category=category_out,
                expect_abstain=False,
                relevant_chunk_ids=sorted(relevant),
                history=history,
                evidence=evidence,
            )
        )

    return questions


@dataclass
class DatasetSummary:
    """Class distribution plus the §2.1 minimums, so the gate is self-checking."""

    total: int
    counts: dict[str, int]
    rates: dict[str, float]
    min_abstain_rate: float
    min_multiturn_rate: float
    min_identifier_rate: float


def summarise(questions: Sequence[EvalQuestion]) -> DatasetSummary:
    total = len(questions) or 1
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1
    rates = {k: v / total for k, v in counts.items()}
    return DatasetSummary(
        total=len(questions),
        counts=counts,
        rates=rates,
        min_abstain_rate=rates.get("abstain", 0.0),
        min_multiturn_rate=rates.get("multiturn", 0.0),
        min_identifier_rate=rates.get("identifier", 0.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the labelled retrieval eval set")
    parser.add_argument(
        "--questions", type=int, default=200, help="number of questions to generate"
    )
    parser.add_argument(
        "--out", default="eval/dataset.jsonl", help="output path for the dataset"
    )
    parser.add_argument("--seed", type=int, default=20240101, help="RNG seed")
    args = parser.parse_args()

    with session_scope() as session:
        questions = build_dataset(session, args.questions, seed=args.seed)

    if not questions:
        print("no questions produced; corpus does not match make_corpus.py output",
              file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(asdict(q), sort_keys=True) + "\n")

    stats = summarise(questions)
    print(f"wrote {stats.total} questions to {out_path}")
    print("class distribution:")
    for category, n in sorted(stats.counts.items()):
        print(f"  {category:12s} {n:4d}  ({stats.rates[category]:.1%})")

    print("\nimplementation.md 2.1 minimums:")
    checks = [
        ("abstain >= 20%", stats.min_abstain_rate >= 0.20, stats.min_abstain_rate),
        ("multiturn >= 10%", stats.min_multiturn_rate >= 0.10, stats.min_multiturn_rate),
        ("identifier >= 10%", stats.min_identifier_rate >= 0.10, stats.min_identifier_rate),
    ]
    ok = True
    for label, passed, value in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label:20s} actual {value:.1%}")
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())