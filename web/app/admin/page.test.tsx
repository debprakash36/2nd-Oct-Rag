/**
 * Admin console tests (FR-27).
 *
 * Focused on the parts that are easy to get wrong: state-dependent actions, and
 * reporting an upload honestly. A duplicate that was indexed must not be reported
 * as a failure, and a delete must not silently no-op on the row the user clicked.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AdminPage from "@/app/admin/page";
import type { DocumentRow } from "@/lib/types";

const DOCS: DocumentRow[] = [
  {
    doc_id: "d1",
    filename: "policy.md",
    mime_type: "text/markdown",
    state: "live",
    version: 1,
    byte_size: 2048,
    page_count: null,
    chunk_count: 12,
    content_hash: "abc",
    duplicate_of: null,
    error_reason: null,
    acl_tags: [],
    uploaded_at: "2026-01-01T00:00:00Z",
    indexed_at: "2026-01-01T00:01:00Z",
  },
  {
    doc_id: "d2",
    filename: "broken.pdf",
    mime_type: "application/pdf",
    state: "failed",
    version: 1,
    byte_size: 900,
    page_count: null,
    chunk_count: 0,
    content_hash: "def",
    duplicate_of: null,
    error_reason: "Extraction produced no text.",
    acl_tags: [],
    uploaded_at: "2026-01-02T00:00:00Z",
    indexed_at: null,
  },
];

/** Route fetches by URL so a test can set up only the endpoints it exercises. */
function routeApi(handlers: Record<string, () => Response>) {
  const fetchMock = vi.fn(async (url: string | URL | Request) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    for (const [prefix, respond] of Object.entries(handlers)) {
      if (path.startsWith(prefix)) return respond();
    }
    throw new Error(`unexpected request: ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

/**
 * Attach a file and submit.
 *
 * The submit click is explicit because the component reads the file list from the
 * input's ref inside the form's submit handler. `userEvent.upload` only sets the
 * files and fires `change`, so a test that omits the click never triggers an upload
 * and then asserts against markup that was never rendered.
 */
async function uploadFile(name: string) {
  // The filename must match the input's `accept` list. `userEvent` silently drops a
  // file the accept attribute excludes, so the input keeps zero files, the submit
  // short-circuits, and the test would assert against markup that never rendered.
  const input = screen.getByLabelText("Choose documents to upload") as HTMLInputElement;
  await userEvent.upload(input, new File(["x"], name));
  expect(input.files?.length).toBe(1);
  await userEvent.click(screen.getByRole("button", { name: "Upload" }));
}

describe("AdminPage", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    // jsdom has no confirm(); default to accepting, and override per test.
    vi.stubGlobal("confirm", () => true);
  });

  it("lists documents with state and chunk count", async () => {
    routeApi({ "/admin/documents": () => json(DOCS) });
    render(<AdminPage />);

    await waitFor(() => expect(screen.getByText("policy.md")).toBeInTheDocument());
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText("broken.pdf")).toBeInTheDocument();
  });

  it("shows a failed document's reason", async () => {
    routeApi({ "/admin/documents": () => json(DOCS) });
    render(<AdminPage />);

    await waitFor(() =>
      expect(screen.getByText("Extraction produced no text.")).toBeInTheDocument(),
    );
  });

  it("disables a live document and refreshes", async () => {
    const fetchMock = routeApi({
      "/admin/documents/d1/disable": () => json(DOCS[0]),
      "/admin/documents": () => json(DOCS),
    });
    render(<AdminPage />);
    await waitFor(() => expect(screen.getByText("policy.md")).toBeInTheDocument());

    const row = screen.getByText("policy.md").closest("tr") as HTMLElement;
    await userEvent.click(within(row).getByRole("button", { name: "Disable" }));

    await waitFor(() => expect(screen.getByText(/policy\.md disabled/)).toBeInTheDocument());
    const called = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(called.some((u) => u.includes("/admin/documents/d1/disable"))).toBe(true);
  });

  it("offers enable for a disabled document, not disable", async () => {
    routeApi({
      "/admin/documents": () =>
        json([{ ...DOCS[0], state: "disabled" as const, indexed_at: null }]),
    });
    render(<AdminPage />);

    await waitFor(() => expect(screen.getByText("policy.md")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Enable" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Disable" })).not.toBeInTheDocument();
  });

  it("offers no state action for a document mid-ingest", async () => {
    // A pending document cannot be disabled; showing the button would only produce
    // an error from the server.
    routeApi({
      "/admin/documents": () => json([{ ...DOCS[0], state: "chunking" as const, indexed_at: null }]),
    });
    render(<AdminPage />);

    await waitFor(() => expect(screen.getByText("policy.md")).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Disable" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Enable" })).not.toBeInTheDocument();
    // Delete is always available.
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();
  });

  it("deletes after confirmation", async () => {
    const fetchMock = routeApi({
      "/admin/documents/d1": () => new Response(null, { status: 204 }),
      "/admin/documents": () => json(DOCS),
    });
    render(<AdminPage />);
    await waitFor(() => expect(screen.getByText("policy.md")).toBeInTheDocument());

    const row = screen.getByText("policy.md").closest("tr") as HTMLElement;
    await userEvent.click(within(row).getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(screen.getByText(/policy\.md deleted/)).toBeInTheDocument());
    expect(fetchMock.mock.calls.some((c) => String(c[0]).endsWith("/admin/documents/d1"))).toBe(
      true,
    );
  });

  it("does not delete when confirmation is declined", async () => {
    vi.stubGlobal("confirm", () => false);
    const fetchMock = routeApi({ "/admin/documents": () => json(DOCS) });
    render(<AdminPage />);
    await waitFor(() => expect(screen.getByText("policy.md")).toBeInTheDocument());

    const row = screen.getByText("policy.md").closest("tr") as HTMLElement;
    await userEvent.click(within(row).getByRole("button", { name: "Delete" }));

    expect(
      fetchMock.mock.calls.some((c) => String(c[0]).endsWith("/d1")),
    ).toBe(false);
  });

  it("reports an empty corpus", async () => {
    routeApi({ "/admin/documents": () => json([]) });
    render(<AdminPage />);
    await waitFor(() => expect(screen.getByText(/No documents indexed/)).toBeInTheDocument());
  });

  it("surfaces a load failure with a safe message", async () => {
    routeApi({
      "/admin/documents": () => json({ detail: "Search is temporarily unavailable." }, 503),
    });
    render(<AdminPage />);
    await waitFor(() =>
      expect(screen.getByText("Search is temporarily unavailable.")).toBeInTheDocument(),
    );
  });

  it("reports rejected uploads without calling them ingested", async () => {
    // POST /admin/documents and GET /admin/documents share a prefix, so the router
    // distinguishes by method: a single prefix handler would answer the list refresh
    // with the upload body and crash the table render.
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      if (init?.method === "POST") {
        return json(
          {
            results: [
              {
                filename: "ok.md",
                doc_id: "d9",
                state: "live",
                chunks: 3,
                duplicate_of: null,
                queued: false,
                warning: null,
                error: null,
              },
              {
                filename: "bad.exe.txt",
                doc_id: null,
                state: null,
                chunks: 0,
                duplicate_of: null,
                queued: false,
                warning: null,
                error: "This file could not be read.",
              },
            ],
accepted: 1,
          rejected: 1,
        });
      }
      return json(DOCS);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<AdminPage />);
    // The rejected file must still be an *accepted* type: the input's `accept` list
    // filters the browser's picker, and `userEvent` mirrors that, so a `.exe` would
    // never reach the component at all.
    await uploadFile("bad.exe.txt");

    await waitFor(() => expect(screen.getByText("1 ingested, 1 rejected")).toBeInTheDocument());
    expect(screen.getByText(/bad\.exe\.txt: This file could not be read\./)).toBeInTheDocument();
  });

  it("treats a duplicate as a warning, not a rejection", async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      if (init?.method === "POST") {
        return json({
          results: [
            {
              filename: "policy.md",
              doc_id: "d1",
              state: "duplicate",
              chunks: 0,
              duplicate_of: "d0",
              queued: false,
              warning: "Identical content is already indexed.",
              error: null,
            },
          ],
          accepted: 1,
          rejected: 0,
        });
      }
      return json(DOCS);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<AdminPage />);
    await uploadFile("policy.md");

    await waitFor(() => expect(screen.getByText("1 ingested")).toBeInTheDocument());
    // No "rejected" in the summary: the duplicate WAS indexed (FR-4).
    expect(screen.queryByText(/rejected/)).not.toBeInTheDocument();
    expect(screen.getByText(/Identical content is already indexed/)).toBeInTheDocument();
  });

  it("shows a safe message when upload itself fails", async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      if (init?.method === "POST") {
        return json({ detail: "This file is too large." }, 413);
      }
      return json(DOCS);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<AdminPage />);
    await uploadFile("huge.pdf");

    await waitFor(() => expect(screen.getByText("This file is too large.")).toBeInTheDocument());
  });
});