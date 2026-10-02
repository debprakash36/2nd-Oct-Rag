/**
 * SSE parsing tests.
 *
 * The parser is where the streaming contract actually lives: a bug here produces a
 * chat UI that appears to work while silently dropping events, so the cases below are
 * the ones that have historically broken streaming clients — split chunks, missing
 * blank-line terminators, and events with unknown names.
 */

import { describe, expect, it, vi } from "vitest";
import { streamChat, type StreamHandlers } from "./chatStream";

/** Build a `Response` whose body yields the given SSE text in chunks of `size` bytes. */
function sseResponse(text: string, size = text.length): Response {
  const bytes = new TextEncoder().encode(text);
  const chunks: Uint8Array[] = [];
  for (let i = 0; i < bytes.length; i += size) {
    chunks.push(bytes.slice(i, i + size));
  }
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
  return new Response(stream, { status: 200 });
}

function recorder() {
  const events: { name: string; data: unknown }[] = [];
  const handlers: StreamHandlers = {
    onSources: (s) => events.push({ name: "sources", data: s }),
    onToken: (t) => events.push({ name: "token", data: t }),
    onCitationWarning: (n) => events.push({ name: "citation_warning", data: n }),
    onDone: (id, a, t) => events.push({ name: "done", data: { id, a, t } }),
    onError: (code, message) => events.push({ name: "error", data: { code, message } }),
  };
  return { events, handlers };
}

const SOURCES = [{ index: 1, chunk_id: "c1", breadcrumb: "Refunds", filename: "p.md", page: 1 }];

describe("streamChat", () => {
  it("dispatches events in order", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        sseResponse(
          [
            `event: sources\ndata: ${JSON.stringify({ sources: SOURCES })}\n\n`,
            `event: token\ndata: ${JSON.stringify({ text: "Hello" })}\n\n`,
            `event: token\ndata: ${JSON.stringify({ text: " world" })}\n\n`,
            `event: done\ndata: ${JSON.stringify({ query_id: "q1", abstained: false, ttft_ms: 12 })}\n\n`,
          ].join(""),
        ),
      ),
    );

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events.map((e) => e.name)).toEqual(["sources", "token", "token", "done"]);
    expect(events[1].data).toBe("Hello");
    expect(events[3].data).toEqual({ id: "q1", a: false, t: 12 });
  });

  it("reassembles events split across chunk boundaries", async () => {
    // Size 7 forces the JSON payload to straddle reads — the classic streaming bug.
    const text = [
      `event: sources\ndata: ${JSON.stringify({ sources: SOURCES })}\n\n`,
      `event: token\ndata: ${JSON.stringify({ text: "split" })}\n\n`,
      `event: done\ndata: ${JSON.stringify({ query_id: "q2", abstained: true, ttft_ms: null })}\n\n`,
    ].join("");

    vi.stubGlobal("fetch", vi.fn(async () => sseResponse(text, 7)));

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events.map((e) => e.name)).toEqual(["sources", "token", "done"]);
    expect(events[1].data).toBe("split");
    // A null ttft must survive as null, not become 0 (which would read as "instant").
    expect(events[2].data).toEqual({ id: "q2", a: true, t: null });
  });

  it("handles a final event with no trailing blank line", async () => {
    const text = `event: done\ndata: ${JSON.stringify({ query_id: "q3", abstained: false, ttft_ms: 5 })}`;
    vi.stubGlobal("fetch", vi.fn(async () => sseResponse(text)));

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events).toHaveLength(1);
    expect(events[0].name).toBe("done");
  });

  it("forwards citation_warning", async () => {
    const text = [
      `event: sources\ndata: ${JSON.stringify({ sources: SOURCES })}\n\n`,
      `event: citation_warning\ndata: ${JSON.stringify({ stripped_count: 2 })}\n\n`,
      `event: done\ndata: ${JSON.stringify({ query_id: "q4", abstained: false, ttft_ms: 1 })}\n\n`,
    ].join("");
    vi.stubGlobal("fetch", vi.fn(async () => sseResponse(text)));

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events[1]).toEqual({ name: "citation_warning", data: 2 });
  });

  it("surfaces a mid-stream error event", async () => {
    const text = [
      `event: sources\ndata: ${JSON.stringify({ sources: SOURCES })}\n\n`,
      `event: token\ndata: ${JSON.stringify({ text: "partial" })}\n\n`,
      `event: error\ndata: ${JSON.stringify({ code: "internal_error", message: "Something went wrong." })}\n\n`,
    ].join("");
    vi.stubGlobal("fetch", vi.fn(async () => sseResponse(text)));

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events.at(-1)).toEqual({
      name: "error",
      data: { code: "internal_error", message: "Something went wrong." },
    });
  });

  it("maps a 400 pre-stream rejection to a query_too_large error", async () => {
    // FR-34: the cap is answered before any event stream exists. With EventSource
    // this would look like an empty stream and the UI would hang on "thinking...".
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: "That message is too long." }), { status: 400 }),
      ),
    );

    const { events, handlers } = recorder();
    await streamChat({ message: "x".repeat(9999) }, handlers);

    expect(events).toEqual([
      { name: "error", data: { code: "query_too_large", message: "That message is too long." } },
    ]);
  });

  it("maps a 401 to unauthorized and clears the stored token", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: "Sign in required." }), { status: 401 }),
      ),
    );

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events[0].data).toEqual({
      code: "unauthorized",
      message: "Sign in required.",
    });
  });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: "You're sending questions too quickly." }), {
          status: 429,
        }),
      ),
    );

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events[0].data).toEqual({
      code: "rate_limited",
      message: "You're sending questions too quickly.",
    });
  });

  it("falls back to a safe message when the error body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("<html>502</html>", { status: 502 })),
    );

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events[0].data).toEqual({
      code: "internal_error",
      message: "Something went wrong. Please try again.",
    });
  });

  it("ignores comment lines and unknown events", async () => {
    const text = [
      ": keep-alive\n\n",
      `event: sources\ndata: ${JSON.stringify({ sources: SOURCES })}\n\n`,
      `event: something_new\ndata: ${JSON.stringify({ future: true })}\n\n`,
      `event: done\ndata: ${JSON.stringify({ query_id: "q5", abstained: false, ttft_ms: 3 })}\n\n`,
    ].join("");
    vi.stubGlobal("fetch", vi.fn(async () => sseResponse(text)));

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    // An unknown event is skipped, not fatal: the server may add events later.
    expect(events.map((e) => e.name)).toEqual(["sources", "done"]);
  });

  it("skips a malformed payload without killing the stream", async () => {
    const text = [
      `event: token\ndata: {not json\n\n`,
      `event: token\ndata: ${JSON.stringify({ text: "good" })}\n\n`,
      `event: done\ndata: ${JSON.stringify({ query_id: "q6", abstained: false, ttft_ms: 1 })}\n\n`,
    ].join("");
    vi.stubGlobal("fetch", vi.fn(async () => sseResponse(text)));

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events.map((e) => e.name)).toEqual(["token", "done"]);
    expect(events[0].data).toBe("good");
  });

  it("reports an empty body", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 200 })));

    const { events, handlers } = recorder();
    await streamChat({ message: "hi" }, handlers);

    expect(events[0].name).toBe("error");
  });

  it("sends the conversation id when provided", async () => {
    // Parameters are declared so `mock.calls` is typed as `[input, init]` rather
    // than an empty tuple; the assertions below read the request body off it.
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        sseResponse(`event: done\ndata: ${JSON.stringify({ query_id: "q", abstained: false, ttft_ms: 1 })}\n\n`),
    );
    vi.stubGlobal("fetch", fetchMock);

    const { handlers } = recorder();
    await streamChat({ message: "hi", conversation_id: "conv-1", answer_style: "detailed" }, handlers);

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(init.body as string);
    expect(body.conversation_id).toBe("conv-1");
    expect(body.answer_style).toBe("detailed");
  });
});