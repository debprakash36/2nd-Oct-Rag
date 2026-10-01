/**
 * SourceViewer tests (FR-18, FR-20, NFR-9).
 *
 * The behavior under test is "clicking a citation reveals the exact passage", plus
 * what happens when that passage is unavailable — which is the case that matters,
 * because a citation can outlive the document it came from.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SourceViewer from "@/components/SourceViewer";
import type { Source } from "@/lib/types";

const SOURCES: Source[] = [
  { index: 1, chunk_id: "c1", breadcrumb: "Refund Policy > Digital", filename: "policy.md", page: 2 },
  { index: 2, chunk_id: "c2", breadcrumb: "Shipping", filename: "policy.md", page: null },
];

const PASSAGE_TEXT = "Customers may request a refund within 30 days of purchase.";

function stubApi(_chunkId: string, body: unknown, ok = true) {
  // `_input` is declared purely to type `mock.calls` as `[input]`, which the tests
  // read the requested URL from.
  const fetchMock = vi.fn(async (_input: RequestInfo | URL) =>
    ok
      ? new Response(JSON.stringify(body), { status: 200 })
      : new Response(JSON.stringify({ detail: "chunk not found." }), { status: 404 }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("SourceViewer", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders nothing when there are no sources", () => {
    const { container } = render(<SourceViewer sources={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("lists each source with its index and filename", () => {
    render(<SourceViewer sources={SOURCES} />);
    expect(screen.getByText("[1]")).toBeInTheDocument();
    expect(screen.getAllByText("policy.md")).toHaveLength(2);
  });

  it("shows breadcrumb and page", () => {
    render(<SourceViewer sources={SOURCES} />);
    expect(screen.getByText(/Refund Policy > Digital/)).toBeInTheDocument();
    expect(screen.getByText(/p\. 2/)).toBeInTheDocument();
  });

  it("reveals the exact passage on click", async () => {
    stubApi("c1", {
      chunk_id: "c1",
      document_id: "d1",
      filename: "policy.md",
      breadcrumb: "Refund Policy > Digital",
      page: 2,
      text: PASSAGE_TEXT,
    });

    render(<SourceViewer sources={SOURCES} />);
    // Indexed, not by name: both sources are "policy.md", so a name query is
    // ambiguous and would resolve to whichever the matcher saw first.
    await userEvent.click(screen.getAllByRole("button")[0]);

    await waitFor(() => expect(screen.getByText(PASSAGE_TEXT)).toBeInTheDocument());
  });

  it("requests the specific chunk", async () => {
    const fetchMock = stubApi("c1", {
      chunk_id: "c1",
      document_id: "d1",
      filename: "policy.md",
      breadcrumb: null,
      page: null,
      text: PASSAGE_TEXT,
    });

    render(<SourceViewer sources={SOURCES} />);
    await userEvent.click(screen.getAllByRole("button")[1]);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url] = fetchMock.mock.calls[0] ?? [];
    expect(String(url)).toContain("/chunks/c2");
  });

  it("does not fetch the passage until clicked", () => {
    const fetchMock = stubApi("c1", {});
    render(<SourceViewer sources={SOURCES} />);
    // FR-20's framing payload stays small; the passage is opt-in.
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("caches a passage so re-expanding does not refetch", async () => {
    const fetchMock = stubApi("c1", {
      chunk_id: "c1",
      document_id: "d1",
      filename: "policy.md",
      breadcrumb: null,
      page: null,
      text: PASSAGE_TEXT,
    });

    render(<SourceViewer sources={SOURCES} />);
    const toggle = screen.getAllByRole("button")[0];

    await userEvent.click(toggle);
    await waitFor(() => expect(screen.getByText(PASSAGE_TEXT)).toBeInTheDocument());
    await userEvent.click(toggle);
    expect(screen.queryByText(PASSAGE_TEXT)).not.toBeInTheDocument();
    await userEvent.click(toggle);

    await waitFor(() => expect(screen.getByText(PASSAGE_TEXT)).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("reports a passage that no longer exists", async () => {
    // The document was disabled or deleted since the answer was cached (FR-18).
    stubApi("c1", { detail: "chunk not found." }, false);

    render(<SourceViewer sources={SOURCES} />);
    await userEvent.click(screen.getAllByRole("button")[0]);

    await waitFor(() => expect(screen.getByText("chunk not found.")).toBeInTheDocument());
  });

  it("surfaces a network failure as a user-safe message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );

    render(<SourceViewer sources={SOURCES} />);
    await userEvent.click(screen.getAllByRole("button")[0]);

    // Must not surface the browser's own wording: "Failed to fetch" is JS jargon
    // and tells the user nothing actionable.
    await waitFor(() =>
      expect(screen.getByText("This passage could not be loaded.")).toBeInTheDocument(),
    );
    expect(screen.queryByText(/Failed to fetch/)).not.toBeInTheDocument();
  });

  it("renders passage text literally, never as markup (FR-22)", async () => {
    const hostile = '<img src=x onerror="alert(1)"><script>alert(2)</script>';
    stubApi("c1", {
      chunk_id: "c1",
      document_id: "d1",
      filename: "policy.md",
      breadcrumb: null,
      page: null,
      text: hostile,
    });

    const { container } = render(<SourceViewer sources={SOURCES} />);
    await userEvent.click(screen.getAllByRole("button")[0]);

    await waitFor(() => expect(screen.getByText(hostile)).toBeInTheDocument());
    // Present as text; no element was created from the string.
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
  });

  it("marks the panel when the answer abstained", () => {
    render(<SourceViewer sources={SOURCES} abstained />);
    expect(screen.getByText(/Answer refused/i)).toBeInTheDocument();
  });

  it("exposes expanded state to assistive tech", async () => {
    stubApi("c1", {
      chunk_id: "c1",
      document_id: "d1",
      filename: "policy.md",
      breadcrumb: null,
      page: null,
      text: PASSAGE_TEXT,
    });

    render(<SourceViewer sources={SOURCES} />);
    const toggle = screen.getAllByRole("button")[0];
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    await userEvent.click(toggle);
    await waitFor(() => expect(toggle).toHaveAttribute("aria-expanded", "true"));
  });

  it("gives each delete-free toggle a unique control target", () => {
    // Two sources with the same chunk_id would collide on id and break aria-controls.
    const dupes: Source[] = [
      { index: 1, chunk_id: "same", breadcrumb: null, filename: "a.md", page: null },
      { index: 2, chunk_id: "same", breadcrumb: null, filename: "b.md", page: null },
    ];
    const { container } = render(<SourceViewer sources={dupes} />);
    const ids = Array.from(container.querySelectorAll("[id]")).map((el) => el.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("surfaces an ApiError message when the response is malformed", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("not json", { status: 500 })),
    );

    render(<SourceViewer sources={SOURCES} />);
    await userEvent.click(screen.getAllByRole("button")[0]);

    await waitFor(() => expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument());
  });
});

