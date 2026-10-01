# TTFT benchmark for the chat pipeline (implementation.md 5.6, NFR-1).
#
# Phase 3's exit gate is "TTFT p95 <= 5 s against a local provider". This script
# measures time-to-first-token for the *pipeline*: retrieval, prompt assembly,
# the provider's first delta, and the sentence buffer's first flush. It runs the
# configured `GenerationProvider` in-process, so with the offline provider the
# number is pipeline overhead only. Swapping in a network model adds that
# model's own first-token latency on top, which is usually the dominant term —
# re-run this against the real provider before treating NFR-1 as met.
#
# TTFT here is measured to the first *validated* token the user could see, not
# to the provider's first raw delta. That is the honest number: a provider that
# streams instantly and is then buffered by citation validation has not actually
# delivered a first token early.

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.db.session import session_scope
from app.generation.assembly import AnswerAssembler, ChunkKind
from app.generation.prompt import (
    STYLE_PRESETS,
    AnswerStyle,
    ContextPassage,
    build_messages,
    new_nonce,
)
from app.providers.generation import get_generation_provider
from app.retrieval.retriever import RetrievalConfig, Retriever

DEFAULT_BUDGET_MS = 5000.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[idx]


def _load_questions(path: Path) -> list[str]:
    import json

    questions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line)["question"])
    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat TTFT benchmark")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument(
        "--iterations", type=int, default=100, help="number of timed requests"
    )
    parser.add_argument("--warmup", type=int, default=3, help="untimed warm-up requests")
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--style", choices=["concise", "detailed"], default="concise")
    args = parser.parse_args()

    settings = get_settings()
    config = RetrievalConfig.from_settings(settings)
    provider = get_generation_provider(settings)
    style = AnswerStyle(args.style)
    max_tokens = STYLE_PRESETS[style].max_tokens

    questions = _load_questions(Path(args.dataset))
    if not questions:
        print(f"no questions in {args.dataset}", file=sys.stderr)
        return 1

    ttfts: list[float] = []
    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    refusals = 0

    with session_scope() as session:
        for i in range(args.warmup + args.iterations):
            question = questions[i % len(questions)]
            retriever = Retriever(session, config=config, settings=settings)

            t0 = time.perf_counter()
            result = retriever.retrieve(question, config=config)
            t1 = time.perf_counter()

            first_token: float | None = None
            answer_seen = False
            if not result.abstained and result.candidates:
                passages = [
                    ContextPassage(text=c.text, breadcrumb=c.breadcrumb, page=c.page)
                    for c in result.candidates
                ]
                messages = build_messages(
                    question, passages, style, nonce=new_nonce()
                )
                assembler = AnswerAssembler(passage_count=len(passages))
                for chunk in assembler.run(
                    provider.stream(
                        messages,
                        model=settings.generation_model,
                        max_tokens=max_tokens,
                        temperature=settings.generation_temperature,
                    )
                ):
                    if chunk.kind is ChunkKind.TOKEN and first_token is None:
                        first_token = (time.perf_counter() - t0) * 1000
                        answer_seen = True
                if assembler.abstained:
                    refusals += 1
            t2 = time.perf_counter()

            if i < args.warmup:
                continue
            retrieval_ms.append((t1 - t0) * 1000)
            total_ms.append((t2 - t0) * 1000)
            if first_token is not None:
                ttfts.append(first_token)
            elif not answer_seen:
                # A refusal still reaches the user immediately; its TTFT is the
                # time to the refusal token, which is the total here.
                ttfts.append((t2 - t0) * 1000)

    p50 = _percentile(ttfts, 0.50)
    p95 = _percentile(ttfts, 0.95)
    r_p95 = _percentile(retrieval_ms, 0.95)
    t_p95 = _percentile(total_ms, 0.95)

    print(f"\n=== TTFT benchmark (n={len(ttfts)}, style={args.style}) ===")
    print(f"provider           {settings.generation_provider} (model {settings.generation_model})")
    print(f"TTFT p50           {p50:.1f} ms")
    print(f"TTFT p95           {p95:.1f} ms")
    print(f"TTFT mean          {statistics.fmean(ttfts):.1f} ms" if ttfts else "TTFT mean  n/a")
    print(f"retrieval p95      {r_p95:.1f} ms")
    print(f"total p95          {t_p95:.1f} ms")
    print(f"refusals           {refusals}")
    budget_ok = p95 <= args.budget_ms
    print(
        f"budget             p95 <= {args.budget_ms:.0f} ms: "
        f"{'PASS' if budget_ok else 'FAIL'}"
    )
    print(
        "\nCaveat: with the offline provider this measures pipeline overhead, not "
        "model latency. A real provider's first-token time is additive."
    )
    return 0 if budget_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
