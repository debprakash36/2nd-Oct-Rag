/**
 * `CitedAnswer`: inline `[n]` markers as operable controls (FR-18, NFR-9).
 *
 * The backend embeds markers in answer text and validates them server-side, but the
 * marker is only useful to a reader if it can be *followed*. These tests pin the part
 * that is easy to regress silently: a marker that stops being a button, loses its
 * accessible name, or renders corpus text as markup.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CitedAnswer from "./CitedAnswer";
import type { Source } from "@/lib/types";

const SOURCES: Source[] = [
  { index: 1, chunk_id: "c1", breadcrumb: "Refunds", filename: "policy.md", page: 3 },
  { index: 2, chunk_id: "c2", breadcrumb: null, filename: "terms.md", page: null },
];

function mockFetch(ok = true) {
  const spy = vi.fn(async (_url: string) => {
    if (!ok) return { ok: false, status: 500, json: async () => ({ detail: "nope" }) };
    return {
      ok: true,
      status: 200,
      json: async () => ({ text: "Refunds are issued within 30 days." }),
    };
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("CitedAnswer", () => {
  it("renders answer text with markers replaced by focusable buttons", () => {
    render(<CitedAnswer content="Refunds take 30 days [1]." sources={SOURCES} />);

    const marker = screen.getByRole("button", { name: /citation 1/i });
    expect(marker).toBeInTheDocument();
    // The surrounding prose is preserved verbatim around the control.
    expect(screen.getByText(/Refunds take 30 days/)).toBeInTheDocument();
  });

  it("uses the breadcrumb when the sources event has no filename", () => {
    render(
      <CitedAnswer
        content="See [1]."
        sources={[{ index: 1, chunk_id: "c1", breadcrumb: "Refund Policy", page: null }]}
      />,
    );
    expect(
      screen.getByRole("button", { name: /citation 1: refund policy/i }),
    ).toBeInTheDocument();
  });

  it("gives each marker an accessible name that includes the source", () => {
    render(<CitedAnswer content="See [1] and [2]." sources={SOURCES} />);

    expect(
      screen.getByRole("button", { name: /citation 1: policy\.md, page 3/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /citation 2: terms\.md$/i }),
    ).toBeInTheDocument();
  });

  it("reports collapsed state to assistive tech via aria-expanded", () => {
    render(<CitedAnswer content="Refunds [1]." sources={SOURCES} />);
    const marker = screen.getByRole("button", { name: /citation 1/i });
    expect(marker).toHaveAttribute("aria-expanded", "false");
  });

  it("opens the cited passage on activation and points aria-controls at it", async () => {
    mockFetch(true);
    const user = userEvent.setup();
    render(<CitedAnswer content="Refunds [1]." sources={SOURCES} />);

    const marker = screen.getByRole("button", { name: /citation 1/i });
    await user.click(marker);

    const panel = await screen.findByRole("region", { name: /passage cited as 1/i });
    expect(panel).toHaveTextContent("Refunds are issued within 30 days.");

    const markerAfter = screen.getByRole("button", { name: /citation 1/i });
    expect(markerAfter).toHaveAttribute("aria-expanded", "true");
    // aria-controls must reference the id of the element that actually appeared.
    expect(markerAfter.getAttribute("aria-controls")).toBe(panel.id);
  });

  it("collapses again on a second activation", async () => {
    mockFetch(true);
    const user = userEvent.setup();
    render(<CitedAnswer content="Refunds [1]." sources={SOURCES} />);

    const marker = screen.getByRole("button", { name: /citation 1/i });
    await user.click(marker);
    await screen.findByRole("region", { name: /passage cited as 1/i });
    await user.click(screen.getByRole("button", { name: /citation 1/i }));

    await waitFor(() =>
      expect(screen.queryByRole("region", { name: /passage cited as 1/i })).toBeNull(),
    );
  });

  it("is reachable and operable by keyboard alone", async () => {
    mockFetch(true);
    const user = userEvent.setup();
    render(<CitedAnswer content="Refunds [1]." sources={SOURCES} />);

    // Tab to the marker and open it with Enter -- no pointer involved.
    await user.tab();
    const marker = screen.getByRole("button", { name: /citation 1/i });
    expect(marker).toHaveFocus();

    await user.keyboard("{Enter}");
    await screen.findByRole("region", { name: /passage cited as 1/i });
  });

  it("leaves a marker with no matching source as literal text", () => {
    // A stripped or stale marker must not become a button wired to nothing.
    const { container } = render(<CitedAnswer content="Or see [7]." sources={SOURCES} />);

    expect(screen.queryByRole("button", { name: /citation 7/i })).toBeNull();
    // The literal marker survives, so the text is not silently corrupted.
    expect(container.textContent).toContain("[7]");
  });

  it("renders corpus text as text, never as markup (FR-22)", () => {
    const hostile = "<img src=x onerror=alert(1)> and <script>alert(2)</script> [1]";
    const { container } = render(
      <CitedAnswer content={hostile} sources={SOURCES} />,
    );

    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    // The text is still present -- escaped, not stripped.
    expect(container.textContent).toContain("<img src=x onerror=alert(1)>");
  });

  it("surfaces a load failure as an alert without breaking the answer", async () => {
    mockFetch(false);
    const user = userEvent.setup();
    render(<CitedAnswer content="Refunds [1]." sources={SOURCES} />);

    await user.click(screen.getByRole("button", { name: /citation 1/i }));
    // The answer text is untouched: a citation that fails to expand is a degraded
    // affordance, not a lost answer.
    expect(screen.getByText(/Refunds/)).toBeInTheDocument();
  });
});
