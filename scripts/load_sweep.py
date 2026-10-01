r"""Measure TTFT against VU count to locate saturation, for `docs/perf_report.md` (5.2).

A single 50-VU point cannot distinguish "comfortably within budget" from "saturated,
with the numbers saved by having only just enough users". This sweeps the load and
reports the shape, so the capacity of the deployment is a measured number rather than
an assumption.

Throughput staying flat while latency grows with VU count is the signature of a
saturated single process: requests queue, and each additional user adds its wait to
everyone else's. Throughput rising with VUs means there was spare capacity.

Usage:
    .venv\Scripts\python.exe scripts\load_sweep.py <base_url> [vus,...]

Requires a server already serving the load corpus; see `tests/load/conftest.py` for
the environment. Writes `docs/load_sweep_results.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.load.loadkit import run_load  # noqa: E402

DEFAULT_SWEEP = (1, 5, 10, 20, 30, 40, 50, 75, 100)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    base_url = sys.argv[1].rstrip("/")
    counts = (
        [int(c) for c in sys.argv[2].split(",")]
        if len(sys.argv) > 2
        else list(DEFAULT_SWEEP)
    )

    rows: list[dict[str, object]] = []
    print(f"{'VUs':>5}  {'p50':>8} {'p95':>8} {'max':>8}  {'rps':>6}  "
          f"{'failed':>6}  {'trunc':>5}")
    for vus in counts:
        # 3 requests per VU so the higher-VU points average over enough samples for
        # a p95; a p95 from 1 request is just that request.
        report = run_load(base_url, vus=vus, requests_per_vu=3, warmup_requests=3)
        data = report.as_dict()

        # Latency figures come from answered queries only: a refusal short-circuits
        # before generation, so mixing them in would flatter the curve.
        answered = [r.client_ttft_ms for r in report.answered if r.client_ttft_ms is not None]
        row = {
            "vus": vus,
            "answered_n": len(answered),
            "ttft_p50_ms": round(report.percentile(answered, 0.50), 1),
            "ttft_p95_ms": round(report.percentile(answered, 0.95), 1),
            "ttft_max_ms": round(max(answered, default=0.0), 1),
            "throughput_rps": data["throughput_rps"],
            "failed": data["failed"],
            "truncated": data["truncated"],
        }
        rows.append(row)
        print(
            f"{vus:>5}  {row['ttft_p50_ms']:>8} {row['ttft_p95_ms']:>8} "
            f"{row['ttft_max_ms']:>8}  {row['throughput_rps']:>6}  "
            f"{row['failed']:>6}  {row['truncated']:>5}"
        )

    out = REPO_ROOT / "docs" / "load_sweep_results.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
