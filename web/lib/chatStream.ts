/**
 * SSE reader for `/chat/stream`.
 *
 * `fetch` rather than `EventSource`: EventSource is GET-only, and the endpoint is
 * POST. It also cannot surface a non-2xx response as an error event, which matters
 * here because the size cap and rate limit answer with 400 and 429 *before* any
 * event stream exists (FR-34, FR-31). With EventSource those would look like a
 * silently empty stream, and the user would see "thinking..." forever.
 *
 * The event contract the server emits, in order:
 *   sources          always first, including for a refusal (FR-20)
 *   token            zero or more
 *   citation_warning optional, when the validator stripped markers
 *   done             last, carrying the query_id to attach feedback to (FR-29)
 *   error            instead of done
 */

import type { Source } from "./types";

export interface StreamHandlers {
  onSources(sources: Source[]): void;
  onToken(text: string): void;
  onCitationWarning(stripped: number): void;
  onDone(queryId: string, abstained: boolean, ttftMs: number | null): void;
  onError(code: string, message: string): void;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  answer_style?: "concise" | "detailed";
}

export async function streamChat(
  request: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000"}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  // A pre-stream rejection: 400 for the size cap, 429 for the rate limit. The body
  // is the user-safe message the backend already composed.
  if (!response.ok) {
    const { message, code } = await readErrorBody(response);
    handlers.onError(code, message);
    return;
  }
  if (!response.body) {
    handlers.onError("internal_error", "The response stream was empty. Please try again.");
    return;
  }

  await consumeSse(response.body, handlers);
}

async function readErrorBody(
  response: Response,
): Promise<{ code: string; message: string }> {
  try {
    const body = (await response.json()) as { detail?: string };
    return {
      code: response.status === 429 ? "rate_limited" : "query_too_large",
      message: body.detail ?? "Something went wrong. Please try again.",
    };
  } catch {
    return { code: "internal_error", message: "Something went wrong. Please try again." };
  }
}

/**
 * Parse the SSE byte stream and dispatch events.
 *
 * Decodes incrementally rather than buffering the whole response: the point of
 * streaming is that tokens appear as they are generated, and a buffered read would
 * render the entire answer at once after a visible pause.
 */
async function consumeSse(
  body: ReadableStream<Uint8Array>,
  handlers: StreamHandlers,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Events are separated by a blank line. A partial event stays in `buffer`
      // until the rest arrives, which is what keeps a chunk boundary mid-JSON from
      // producing a parse error.
      let split: number;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        dispatch(raw, handlers);
      }
    }
    // A stream that ends without a trailing blank line still has a final event.
    if (buffer.trim()) {
      dispatch(buffer, handlers);
    }
  } finally {
    // Release the connection on every path, including abort. Leaving it open keeps
    // the server's generator running and billed for tokens nobody will see.
    reader.releaseLock();
  }
}

function dispatch(raw: string, handlers: StreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
    // Comment lines (`: keep-alive`) and unknown fields are ignored, per the SSE spec.
  }
  if (dataLines.length === 0) return;

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
  } catch {
    // A malformed payload must not kill the stream; the server may still send more.
    return;
  }

  switch (event) {
    case "sources":
      handlers.onSources((data.sources as Source[]) ?? []);
      break;
    case "token":
      handlers.onToken(String(data.text ?? ""));
      break;
    case "citation_warning":
      handlers.onCitationWarning(Number(data.stripped_count ?? 0));
      break;
    case "done":
      handlers.onDone(
        String(data.query_id ?? ""),
        Boolean(data.abstained),
        data.ttft_ms === null || data.ttft_ms === undefined
          ? null
          : Number(data.ttft_ms),
      );
      break;
    case "error":
      handlers.onError(String(data.code ?? "internal_error"), String(data.message ?? ""));
      break;
  }
}