import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PilotPage from "@/app/admin/pilot/page";
import type { PilotMetricsResponse, PilotReviewResponse } from "@/lib/types";

const METRICS: PilotMetricsResponse = {
  gate: "not_earned",
  min_samples: 20,
  note: "UNMEASURED is not a pass.",
  raw: {
    total: 4,
    distinct_queries: 1,
    abstained: 0,
    up_votes: 4,
    down_votes: 0,
    votes: 4,
    timed: 4,
    voted_queries: 1,
    ttft_p95: 900,
    ttft_count: 4,
  },
  metrics: [
    {
      name: "answered-helpfully",
      value: 1,
      target: ">= 70%",
      state: "unmeasured",
      detail: "Not reportable.",
      samples: 1,
    },
  ],
};

const REVIEW: PilotReviewResponse = {
  items: [{ query: "escalation policy?", why: "refused", refusals: 1, down_votes: 0, occurrences: 1 }],
  empty_reason: null,
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

describe("PilotPage", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        const path = String(url);
        if (path.includes("/admin/pilot/metrics")) return json(METRICS);
        if (path.includes("/admin/pilot/review/classify") && init?.method === "POST") {
          return json({
            query: "escalation policy?",
            classification: "content_gap",
            confidence: "high",
            fix: "Write the document",
            evidence: "noise",
            notes: [],
          });
        }
        if (path.includes("/admin/pilot/review")) return json(REVIEW);
        if (path.includes("/admin/pilot/threshold-history")) {
          return json({ snapshots: [], note: "One snapshot is not a trend." });
        }
        throw new Error(`unexpected ${url}`);
      }),
    );
  });

  it("renders UNMEASURED instead of treating a perfect rate as a pass", async () => {
    render(<PilotPage />);
    expect(await screen.findByText("UNMEASURED")).toBeInTheDocument();
    expect(screen.getByText("not earned")).toBeInTheDocument();
    expect(screen.queryByText("PASS")).not.toBeInTheDocument();
  });

  it("classifies a review item on demand", async () => {
    render(<PilotPage />);
    await screen.findByText("escalation policy?");
    await userEvent.click(screen.getByRole("button", { name: "Classify" }));
    await waitFor(() =>
      expect(screen.getByText(/content_gap: Write the document/)).toBeInTheDocument(),
    );
  });
});
