"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import SourceViewer from "@/components/SourceViewer";
import CitedAnswer from "@/components/CitedAnswer";
import { apiFetch } from "@/lib/api";
import { streamChat } from "@/lib/chatStream";
import type { Conversation, ConversationDetail, Source } from "@/lib/types";
import type { FeedbackValue } from "@/lib/types";
import styles from "./chat.module.css";

/**
 * A turn as rendered, which is not quite the server shape: the streaming turn exists
 * only in local state until the stream completes and the server's id is known.
 */
interface DisplayTurn {
  role: "user" | "assistant";
  content: string;
  /** Present once persisted; `null` while streaming. */
  queryId: string | null;
  abstained: boolean;
  feedback: FeedbackValue;
}

const MAX_CHARS = 4000;

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [turns, setTurns] = useState<DisplayTurn[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [abstained, setAbstained] = useState(false);
  const [citationStripped, setCitationStripped] = useState(0);
  const [message, setMessage] = useState("");
  const [style, setStyle] = useState<"concise" | "detailed">("concise");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);

  const refreshConversations = useCallback(async () => {
    try {
      const rows = await apiFetch<Conversation[]>("/conversations");
      setConversations(rows);
    } catch {
      // A failed sidebar load is not worth an error banner: the chat still works,
      // and the user can retry by reloading.
    }
  }, []);

  // Load the sidebar on mount. Declared after `refreshConversations` because it is a
  // `const`: putting it first would throw a TDZ error on the dependency array.
  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  // Keep the newest turn in view as tokens arrive.
  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns]);

  async function newConversation() {
    setError(null);
    try {
      const created = await apiFetch<Conversation>("/conversations", { method: "POST" });
      setActiveId(created.conversation_id);
      setTurns([]);
      setSources([]);
      setAbstained(false);
      setCitationStripped(0);
      await refreshConversations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start a conversation.");
    }
  }

  async function openConversation(id: string) {
    if (busy) return;
    setError(null);
    try {
      const detail = await apiFetch<ConversationDetail>(`/conversations/${id}`);
      setActiveId(detail.conversation_id);
      setTurns(
        detail.turns.map((t) => ({
          role: t.role,
          content: t.content,
          queryId: t.query_id,
          abstained: t.abstained,
          feedback: "none" as FeedbackValue,
        })),
      );
      setSources([]);
      setAbstained(false);
      setCitationStripped(0);
      // Votes live on QueryLog, not on the turn, so restoring them costs one request
      // per answered turn (FR-29). Without this a reload silently clears the thumbs.
      await restoreFeedback(detail.turns);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not open that conversation.");
    }
  }

  /**
   * Fill in the thumb state for turns that have a query id.
   *
   * Best-effort by design: a failure here leaves the thumbs unpressed, which is
   * recoverable by clicking, so it must not surface as a conversation-load error.
   */
  async function restoreFeedback(turns: ConversationDetail["turns"]) {
    const voted = await Promise.all(
      turns
        .filter((t) => t.role === "assistant" && t.query_id)
        .map(async (t) => {
          try {
            const r = await apiFetch<{ value: FeedbackValue }>(`/feedback/${t.query_id}`);
            return [t.query_id as string, r.value] as const;
          } catch {
            return null;
          }
        }),
    );
    const byQuery = new Map(voted.filter((v) => v !== null));
    if (byQuery.size === 0) return;

    setTurns((ts) =>
      ts.map((t) => {
        if (!t.queryId) return t;
        const value = byQuery.get(t.queryId);
        return value ? { ...t, feedback: value } : t;
      }),
    );
  }

  async function deleteConversation(id: string) {
    setError(null);
    try {
      await apiFetch<void>(`/conversations/${id}`, { method: "DELETE" });
      // Reset the view if the deleted thread was open; leaving a stale thread on
      // screen would show content the user believes they removed.
      if (activeId === id) {
        setActiveId(null);
        setTurns([]);
        setSources([]);
      }
      await refreshConversations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete that conversation.");
    }
  }

  function setFeedback(index: number, value: FeedbackValue) {
    const turn = turns[index];
    if (!turn || !turn.queryId) return;

    // Optimistic: the UI reflects the click immediately, and a failure reverts it.
    const previous = turn.feedback;
    setTurns((ts) =>
      ts.map((t, i) => (i === index ? { ...t, feedback: value } : t)),
    );

    void (async () => {
      try {
        await apiFetch(`/feedback`, {
          method: "PUT",
          body: JSON.stringify({ query_id: turn.queryId, value }),
        });
      } catch {
        setTurns((ts) =>
          ts.map((t, i) => (i === index ? { ...t, feedback: previous } : t)),
        );
        setError("Feedback could not be saved.");
      }
    })();
  }

  async function send(event: React.FormEvent) {
    event.preventDefault();
    const text = message.trim();
    if (!text || busy) return;
    if (text.length > MAX_CHARS) {
      setError(`That message is too long. Please keep it under ${MAX_CHARS} characters.`);
      return;
    }

    setError(null);
    setBusy(true);
    setCitationStripped(0);
    setAbstained(false);
    setSources([]);

    // Echo the question immediately so the user sees it before the first token.
    // The server records it too; on the next load the thread is server-truth.
    setTurns((ts) => [
      ...ts,
      { role: "user", content: text, queryId: null, abstained: false, feedback: "none" },
      { role: "assistant", content: "", queryId: null, abstained: false, feedback: "none" },
    ]);
    setMessage("");

    const conversationId = activeId;
    const controller = new AbortController();
    abortRef.current = controller;
    const assistantIndex = turns.length + 1; // after the user turn

    const appendToken = (token: string) => {
      setTurns((ts) =>
        ts.map((t, i) => (i === assistantIndex ? { ...t, content: t.content + token } : t)),
      );
    };

    try {
      await streamChat(
        { message: text, answer_style: style, ...(conversationId ? { conversation_id: conversationId } : {}) },
        {
          onSources: setSources,
          onToken: appendToken,
          onCitationWarning: (n) => setCitationStripped(n),
          onDone: (queryId, didAbstain) => {
            setTurns((ts) =>
              ts.map((t, i) =>
                i === assistantIndex ? { ...t, queryId, abstained: didAbstain } : t,
              ),
            );
            setAbstained(didAbstain);
          },
          onError: (_code, msg) => setError(msg),
        },
        controller.signal,
      );
    } catch (e) {
      // Abort is a deliberate user action, not a failure worth reporting.
      if (!controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "The request failed.");
      }
    } finally {
      setBusy(false);
      setStatus("");
      abortRef.current = null;
      // The sidebar's preview and turn count are now stale.
      void refreshConversations();
    }
  }

  return (
    <div className={styles["chat-layout"]}>
      <nav className={styles.sidebar} aria-label="Conversations">
        <div className={styles["sidebar-header"]}>
          <button type="button" className="primary" onClick={newConversation} disabled={busy}>
            + New conversation
          </button>
        </div>
        {conversations.length === 0 ? (
          <p className="empty-state">No conversations yet.</p>
        ) : (
          <ul className={styles["sidebar-list"]}>
            {conversations.map((c) => (
              <li
                key={c.conversation_id}
                className={styles["sidebar-item"]}
                data-active={c.conversation_id === activeId ? "true" : "false"}
              >
                <button
                  type="button"
                  className="select"
                  onClick={() => openConversation(c.conversation_id)}
                  disabled={busy}
                  title={c.preview}
                >
                  {c.preview}
                </button>
                <button
                  type="button"
                  className="delete"
                  aria-label={`Delete conversation: ${c.preview}`}
                  onClick={() => deleteConversation(c.conversation_id)}
                  disabled={busy}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </nav>

      <main className={styles["chat-main"]}>
        <div className={styles.thread} ref={threadRef}>
          {turns.length === 0 && (
            <p className="empty-state">
              Ask a question about the indexed documents. Answers cite their sources.
            </p>
          )}
          {turns.map((turn, i) => (
            <div
              key={i}
              // Lets a test read the whole turn as one string. Needed because an
              // answer is no longer a single text node: inline citation markers are
              // separate buttons, so asserting on one node is no longer possible.
              data-testid={turn.role === "assistant" ? "assistant-turn" : undefined}
              className={turn.role === "user" ? styles["turn-user"] : styles["turn-assistant"]}
            >
              {turn.role === "assistant" && <div className={styles["turn-label"]}>assistant</div>}
              {turn.role === "assistant" && turn.abstained ? (
                <div className={styles["turn-refusal"]}>{turn.content}</div>
              ) : turn.role === "assistant" ? (
                // Inline `[n]` markers become focusable controls (NFR-9). A loaded
                // conversation has no `sources` in state -- the panel is only fed by
                // the live `sources` SSE event -- so markers fall back to plain text
                // there rather than becoming buttons wired to nothing.
                turn.content.includes("[") && sources.length > 0 ? (
                  <CitedAnswer content={turn.content} sources={sources} />
                ) : (
                  turn.content
                )
              ) : (
                turn.content
              )}

              {/* Feedback is only offered once the answer is persisted (FR-29). */}
              {turn.role === "assistant" && turn.queryId && (
                <div className={styles["feedback-row"]}>
                  <button
                    type="button"
                    data-selected={turn.feedback === "up"}
                    aria-label="Thumbs up"
                    aria-pressed={turn.feedback === "up"}
                    onClick={() => setFeedback(i, turn.feedback === "up" ? "none" : "up")}
                  >
                    👍
                  </button>
                  <button
                    type="button"
                    data-selected={turn.feedback === "down"}
                    aria-label="Thumbs down"
                    aria-pressed={turn.feedback === "down"}
                    onClick={() => setFeedback(i, turn.feedback === "down" ? "none" : "down")}
                  >
                    👎
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        <form className={styles.composer} onSubmit={send}>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          {citationStripped > 0 && (
            <p className={styles["citation-warning"]}>
              {citationStripped} invalid citation
              {citationStripped === 1 ? " was" : "s were"} removed from the answer.
            </p>
          )}
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask about the corpus…"
            aria-label="Message"
            disabled={busy}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter newlines — without breaking a textarea's
              // primary purpose of holding a multi-line paste.
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send(e);
              }
            }}
          />
          <div className={styles["composer-row"]}>
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
              <select
                value={style}
                onChange={(e) => setStyle(e.target.value as "concise" | "detailed")}
                aria-label="Answer style"
                disabled={busy}
              >
                <option value="concise">Concise</option>
                <option value="detailed">Detailed</option>
              </select>
              <span className={styles.hint}>{message.length}/{MAX_CHARS}</span>
            </div>
            <button type="submit" className="primary" disabled={busy || !message.trim()}>
              {busy ? "Thinking…" : "Send"}
            </button>
          </div>
          <div className={styles["status-line"]}>{status}</div>
        </form>
      </main>

      {sources.length > 0 && (
        <SourceViewer sources={sources} abstained={abstained} />
      )}
    </div>
  );
}