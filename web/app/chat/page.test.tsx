/**
 * Chat page tests (FR-19, FR-20, FR-23, FR-25, FR-29, FR-34).
 *
 * These drive the component with a stubbed `/chat/stream`, so they cover the wiring
 * the backend cannot verify: that tokens append as they arrive, that a refusal is
 * styled differently from an answer, that the conversation id is threaded through, and
 * that feedback is only offered once an answer has an id to attach to.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ChatPage from "./page";
import type { Conversation, ConversationDetail } from "@/lib/types";

const SOURCES = [
  { index: 1, chunk_id: "c1", breadcrumb: "Refunds", filename: "policy.md", page: 1 },
];

function sse(events: [string, Record<string, unknown>][]): string {
  return events
    .map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`)
    .join("");
}

interface StreamOptions {
  events?: [string, Record<string, unknown>][];
  status?: number;
  errorBody?: Record<string, unknown>;
}

interface RouteOptions {
  stream?: StreamOptions;
  conversations?: Conversation[];
  detail?: ConversationDetail;
  feedbackAccepted?: boolean;
  /** Existing votes keyed by query id, returned from `GET /feedback/{query_id}`. */
  votes?: Record<string, "up" | "down" | "none">;
}

/** Stub every endpoint the page touches, so no test hits the network. */
function route(opts: RouteOptions = {}) {
  const sent: { url: string; method: string; body: unknown }[] = [];

  const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, "");
    const method = init?.method ?? "GET";
    sent.push({ url: path, method, body: init?.body });

    if (path === "/chat/stream") {
      const stream = opts.stream ?? {};
      const status = stream.status ?? 200;
      if (status !== 200) {
        return new Response(JSON.stringify(stream.errorBody ?? { detail: "Nope." }), {
          status,
        });
      }
      const text = sse(
        stream.events ?? [
          ["sources", { sources: SOURCES }],
          ["token", { text: "Refunds " }],
          ["token", { text: "within 30 days. [1]" }],
          ["done", { query_id: "q1", abstained: false, ttft_ms: 10 }],
        ],
      );
      return new Response(new TextEncoder().encode(text), { status: 200 });
    }
    if (path === "/conversations" && method === "POST") {
      return json({
        conversation_id: "conv-1",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
        turn_count: 0,
        preview: "New conversation",
      });
    }
    if (path === "/conversations" && method === "GET") {
      return json(opts.conversations ?? []);
    }
    if (path.startsWith("/conversations/") && method === "GET") {
      return json(opts.detail);
    }
    if (path.startsWith("/conversations/") && method === "DELETE") {
      return new Response(null, { status: 204 });
    }
    if (path === "/feedback") {
      return opts.feedbackAccepted === false
        ? json({ detail: "answer not found." }, 404)
        : json({ query_id: "q1", value: "up" });
    }
    if (path.startsWith("/feedback/") && method === "GET") {
      const queryId = path.slice("/feedback/".length);
      return json({ query_id: queryId, value: opts.votes?.[queryId] ?? "none" });
    }
    if (path.startsWith("/chunks/")) {
      return json({
        chunk_id: "c1",
        document_id: "d1",
        filename: "policy.md",
        breadcrumb: "Refunds",
        page: 1,
        text: "Customers may request a refund within 30 days.",
      });
    }
    return json({ detail: "unhandled" }, 500);
  });

  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, sent };
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

async function ask(text: string) {
  const box = screen.getByLabelText("Message");
  await userEvent.type(box, text);
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
}

describe("ChatPage", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("streams tokens into the assistant turn", async () => {
    route();
    render(<ChatPage />);

    await ask("What is the refund window?");

    // Asserted as one string because that is how it reads: two tokens appended in
    // The concatenation across token boundaries is the point of this test: a
    // dropped or doubled space in the SSE token sequence must not leave a stray
    // space or drop one.
    // The whole answer, with the marker as a control rather than literal text, so
    // the concatenated token string is checked as one continuous reading order.
    // Matching on the container's textContent rather than a node, because the
    // marker is now a separate element and breaks the answer into siblings.
    await waitFor(() =>
      expect(screen.getByTestId("assistant-turn").textContent).toContain(
        "Refunds within 30 days. 1",
      ),
    );
  });

  it("echoes the question immediately", async () => {
    route();
    render(<ChatPage />);
    await ask("What is the refund window?");

    await waitFor(() =>
      expect(screen.getByText("What is the refund window?")).toBeInTheDocument(),
    );
  });

  it("shows sources when the answer streams", async () => {
    route();
    render(<ChatPage />);
    await ask("refund window");

    await waitFor(() => expect(screen.getByText("Sources")).toBeInTheDocument());
    expect(screen.getByText("policy.md")).toBeInTheDocument();
  });

  it("renders a refusal distinctly and keeps the sources", async () => {
    route({
      stream: {
        events: [
          ["sources", { sources: SOURCES }],
          ["token", { text: "I couldn't find anything about that in the corpus." }],
          ["done", { query_id: "q1", abstained: true, ttft_ms: 8 }],
        ],
      },
    });
    render(<ChatPage />);
    await ask("something unrelated");

    await waitFor(() =>
      expect(screen.getByText(/I couldn't find anything/)).toBeInTheDocument(),
    );
    // FR-20: a refusal still shows what was consulted.
    expect(screen.getByText("Sources")).toBeInTheDocument();
    expect(screen.getByText(/Answer refused/)).toBeInTheDocument();
  });

  it("reports a citation warning from the stream", async () => {
    route({
      stream: {
        events: [
          ["sources", { sources: SOURCES }],
          ["citation_warning", { stripped_count: 2 }],
          ["token", { text: "Answer [1] text." }],
          ["done", { query_id: "q1", abstained: false, ttft_ms: 5 }],
        ],
      },
    });
    render(<ChatPage />);
    await ask("refund window");

    await waitFor(() =>
      expect(screen.getByText(/2 invalid citations were removed/)).toBeInTheDocument(),
    );
  });

  it("surfaces a pre-stream 400 as the backend's message (FR-34)", async () => {
    route({
      stream: {
        status: 400,
        errorBody: { detail: "That message is too long. Please keep it under 4000 characters." },
      },
    });
    render(<ChatPage />);
    await ask("x");

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/too long/i),
    );
  });

  it("surfaces a pre-stream 429 as the backend's message (FR-31)", async () => {
    route({
      stream: {
        status: 429,
        errorBody: { detail: "You're sending questions too quickly. Please wait a moment and try again." },
      },
    });
    render(<ChatPage />);
    await ask("refund");

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/too quickly/i));
  });

  it("sends a mid-stream error as an alert", async () => {
    route({
      stream: {
        events: [
          ["token", { text: "partial answer" }],
          ["error", { code: "internal_error", message: "Something went wrong. Please try again." }],
        ],
      },
    });
    render(<ChatPage />);
    await ask("refund");

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Something went wrong/));
    // The partial text stays: what was streamed was real and the user should see it.
    expect(screen.getByText("partial answer")).toBeInTheDocument();
  });

  it("refuses to send an over-long message before any request (FR-34)", async () => {
    const { sent } = route();
    render(<ChatPage />);

    const box = screen.getByLabelText("Message") as HTMLTextAreaElement;
    // Bypass the type loop for a 4001-character value.
    await userEvent.click(box);
    await userEvent.paste("x".repeat(4001));

    // The message is non-empty, so Send stays enabled; the guard lives in the submit
    // handler. Clicking it must surface the error without reaching the backend.
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/too long/i));
    expect(sent.some((s) => s.url === "/chat/stream")).toBe(false);
    // Rejected locally: no turn was appended and the draft was left intact so the
    // user can edit it down. Scope to the thread, since the draft itself holds the x's.
    const thread = screen.getByRole("main");
    expect(within(thread).getByText(/Ask a question about the indexed documents/)).toBeInTheDocument();
    expect((screen.getByLabelText("Message") as HTMLTextAreaElement).value).toHaveLength(4001);
  });

  it("creates a conversation first and threads its id into the request", async () => {
    const { sent } = route();
    render(<ChatPage />);

    await userEvent.click(screen.getByRole("button", { name: /New conversation/ }));
    await waitFor(() => expect(sent.some((s) => s.url === "/conversations" && s.method === "POST")).toBe(true));

    await ask("refund window");

    await waitFor(() => {
      const call = sent.find((s) => s.url === "/chat/stream");
      const body = JSON.parse(String(call?.body)) as { conversation_id?: string };
      expect(body.conversation_id).toBe("conv-1");
    });
  });

  it("opens no conversation id when the user has not started one", async () => {
    const { sent } = route();
    render(<ChatPage />);
    await ask("refund window");

    await waitFor(() => {
      const call = sent.find((s) => s.url === "/chat/stream");
      const body = JSON.parse(String(call?.body)) as Record<string, unknown>;
      expect(body.conversation_id).toBeUndefined();
    });
  });

  it("loads prior turns when a conversation is opened", async () => {
    route({
      conversations: [
        {
          conversation_id: "conv-9",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-02T00:00:00Z",
          turn_count: 2,
          preview: "What is the refund window?",
        },
      ],
      detail: {
        conversation_id: "conv-9",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-02T00:00:00Z",
        turn_count: 2,
        preview: "What is the refund window?",
        turns: [
          {
            turn_id: "t1",
            turn_index: 0,
            role: "user",
            content: "What is the refund window?",
            citations: [],
            query_id: null,
            abstained: false,
            created_at: "2026-01-01T00:00:00Z",
          },
          {
            turn_id: "t2",
            turn_index: 1,
            role: "assistant",
            content: "Within 30 days. [1]",
            citations: ["c1"],
            query_id: "q-old",
            abstained: false,
            created_at: "2026-01-01T00:00:01Z",
          },
        ],
      },
    });
    render(<ChatPage />);

    await userEvent.click(await screen.findByTitle("What is the refund window?"));

    await waitFor(() => expect(screen.getByText("Within 30 days. [1]")).toBeInTheDocument());
    // Scoped to the thread: the same question also appears as the sidebar preview,
    // so an unscoped getByText would match twice and fail for the wrong reason.
    const thread = screen.getByRole("main");
    expect(within(thread).getByText("What is the refund window?")).toBeInTheDocument();
    expect(within(thread).getByText("Within 30 days. [1]")).toBeInTheDocument();
  });

  it("restores an existing vote when a conversation is reopened (FR-29)", async () => {
    route({
      conversations: [
        {
          conversation_id: "conv-9",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-02T00:00:00Z",
          turn_count: 2,
          preview: "Rated answer",
        },
      ],
      votes: { "q-old": "down" },
      detail: {
        conversation_id: "conv-9",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-02T00:00:00Z",
        turn_count: 2,
        preview: "Rated answer",
        turns: [
          {
            turn_id: "t1",
            turn_index: 0,
            role: "user",
            content: "Rated question",
            citations: [],
            query_id: null,
            abstained: false,
            created_at: "2026-01-01T00:00:00Z",
          },
          {
            turn_id: "t2",
            turn_index: 1,
            role: "assistant",
            content: "Rated answer. [1]",
            citations: ["c1"],
            query_id: "q-old",
            abstained: false,
            created_at: "2026-01-01T00:00:01Z",
          },
        ],
      },
    });
    render(<ChatPage />);

    await userEvent.click(await screen.findByTitle("Rated answer"));

    // The vote lives on QueryLog, so it is read back per turn after the load.
    const down = await screen.findByRole("button", { name: "Thumbs down" });
    await waitFor(() => expect(down).toHaveAttribute("aria-pressed", "true"));
    expect(screen.getByRole("button", { name: "Thumbs up" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("opens a conversation whose turns have no votes unpressed", async () => {
    route({
      conversations: [
        {
          conversation_id: "conv-9",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-02T00:00:00Z",
          turn_count: 2,
          preview: "Unrated answer",
        },
      ],
      detail: {
        conversation_id: "conv-9",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-02T00:00:00Z",
        turn_count: 2,
        preview: "Unrated answer",
        turns: [
          {
            turn_id: "t1",
            turn_index: 0,
            role: "user",
            content: "Unrated question",
            citations: [],
            query_id: null,
            abstained: false,
            created_at: "2026-01-01T00:00:00Z",
          },
          {
            turn_id: "t2",
            turn_index: 1,
            role: "assistant",
            content: "Unrated answer. [1]",
            citations: ["c1"],
            query_id: "q-new",
            abstained: false,
            created_at: "2026-01-01T00:00:01Z",
          },
        ],
      },
    });
    render(<ChatPage />);

    await userEvent.click(await screen.findByTitle("Unrated answer"));

    const up = await screen.findByRole("button", { name: "Thumbs up" });
    await waitFor(() => expect(up).toHaveAttribute("aria-pressed", "false"));
  });

  it("offers feedback only after an answer has been persisted (FR-29)", async () => {
    route();
    render(<ChatPage />);

    // Nothing streamed yet: no thumbs.
    expect(screen.queryByRole("button", { name: "Thumbs up" })).not.toBeInTheDocument();

    await ask("refund window");
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Thumbs up" })).toBeInTheDocument(),
    );
  });

  it("sends thumbs up and marks the button pressed", async () => {
    const { sent } = route();
    render(<ChatPage />);
    await ask("refund window");

    const up = await screen.findByRole("button", { name: "Thumbs up" });
    await userEvent.click(up);

    await waitFor(() => {
      const call = sent.find((s) => s.url === "/feedback");
      expect(call?.method).toBe("PUT");
      expect(JSON.parse(String(call?.body))).toEqual({ query_id: "q1", value: "up" });
    });
    await waitFor(() => expect(up).toHaveAttribute("aria-pressed", "true"));
  });

  it("clears a thumb when clicked twice", async () => {
    const { sent } = route();
    render(<ChatPage />);
    await ask("refund window");

    const up = await screen.findByRole("button", { name: "Thumbs up" });
    await userEvent.click(up);
    await waitFor(() => expect(up).toHaveAttribute("aria-pressed", "true"));

    await userEvent.click(up);
    await waitFor(() => {
      const calls = sent.filter((s) => s.url === "/feedback");
      expect(JSON.parse(String(calls.at(-1)?.body))).toEqual({ query_id: "q1", value: "none" });
    });
    await waitFor(() => expect(up).toHaveAttribute("aria-pressed", "false"));
  });

  it("reverts the thumb when the server rejects it", async () => {
    route({ feedbackAccepted: false });
    render(<ChatPage />);
    await ask("refund window");

    const up = await screen.findByRole("button", { name: "Thumbs up" });
    await userEvent.click(up);

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/Feedback could not be saved/),
    );
    // The optimistic update must roll back or the UI would claim a vote that failed.
    await waitFor(() => expect(up).toHaveAttribute("aria-pressed", "false"));
  });

  it("clears the thread when the open conversation is deleted", async () => {
    route({
      conversations: [
        {
          conversation_id: "conv-9",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-02T00:00:00Z",
          turn_count: 1,
          preview: "Refunds question",
        },
      ],
    });
    render(<ChatPage />);

    // The sidebar loads asynchronously, so the item must be awaited rather than
    // queried synchronously.
    const trigger = await screen.findByTitle("Refunds question");
    const item = trigger.closest("li") as HTMLElement;
    await userEvent.click(within(item).getByRole("button", { name: /Delete conversation/ }));

    await waitFor(() =>
      expect(screen.getByText(/Ask a question about the indexed documents/)).toBeInTheDocument(),
    );
  });

  it("re-enables the composer after the stream ends", async () => {
    route();
    render(<ChatPage />);
    await ask("refund window");

    // Busy state clears: the button's label reverts from "Thinking…" to "Send" and
    // the textarea is writable again. Send stays disabled only because the draft was
    // cleared on send, which is the intended empty-state behaviour.
    const send = await screen.findByRole("button", { name: "Send" });
    expect(screen.getByLabelText("Message")).not.toBeDisabled();

    await userEvent.type(screen.getByLabelText("Message"), "next question");
    await waitFor(() => expect(send).not.toBeDisabled());
  });

  it("shows the character counter", async () => {
    route();
    render(<ChatPage />);
    expect(screen.getByText("0/4000")).toBeInTheDocument();
  });

  it("reveals an exact passage when a source is clicked (FR-18)", async () => {
    route();
    render(<ChatPage />);
    await ask("refund window");

    await waitFor(() => expect(screen.getByText("Sources")).toBeInTheDocument());
    // Scoped to the sources panel: the answer body also contains a focusable
    // inline marker for the same source, and both are legitimately named after it.
    const panel = screen.getByRole("complementary");
    await userEvent.click(within(panel).getByRole("button", { name: /policy\.md/ }));

    await waitFor(() =>
      expect(
        screen.getByText("Customers may request a refund within 30 days."),
      ).toBeInTheDocument(),
    );
  });

  it("renders answer text as text, never markup (FR-22)", async () => {
    route({
      stream: {
        events: [
          ["sources", { sources: SOURCES }],
          ["token", { text: '<img src=x onerror="alert(1)">' }],
          ["done", { query_id: "q1", abstained: false, ttft_ms: 1 }],
        ],
      },
    });
    const { container } = render(<ChatPage />);
    await ask("injection attempt");

    await waitFor(() =>
      expect(screen.getByText('<img src=x onerror="alert(1)">')).toBeInTheDocument(),
    );
    expect(container.querySelector("img")).toBeNull();
  });

  it("sends with Enter and does not send on Shift+Enter", async () => {
    const { sent } = route();
    render(<ChatPage />);

    const box = screen.getByLabelText("Message");
    await userEvent.click(box);
    await userEvent.keyboard("first line");
    await userEvent.keyboard("{Shift>}{Enter}{/Shift}");
    await userEvent.keyboard("second line");
    // The newline is in the box; nothing was sent yet.
    expect(sent.some((s) => s.url === "/chat/stream")).toBe(false);

    await userEvent.keyboard("{Enter}");
    await waitFor(() => expect(sent.some((s) => s.url === "/chat/stream")).toBe(true));

    const call = sent.find((s) => s.url === "/chat/stream");
    expect(JSON.parse(String(call?.body)).message).toBe("first line\nsecond line");
  });

  it("does not send an empty message", async () => {
    const { sent } = route();
    render(<ChatPage />);
    // Send is disabled while the box is empty or whitespace-only.
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    await userEvent.type(screen.getByLabelText("Message"), "   ");
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(sent.some((s) => s.url === "/chat/stream")).toBe(false);
  });
});