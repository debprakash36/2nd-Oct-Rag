"use client";

import { useState } from "react";
import type { Source } from "@/lib/types";
import { ApiError, apiFetch } from "@/lib/api";
import type { Passage } from "@/lib/types";
import styles from "./SourceViewer.module.css";

/**
 * Clickable sources with on-demand exact passages (FR-18, FR-20).
 *
 * The SSE stream sends only a trimmed `sources` event (chunk_id, filename,
 * breadcrumb, page) so the framing is small. Clicking a source fetches the full
 * passage text via `GET /chunks/{chunk_id}`. That is the trade-off: no bandwidth for
 * passages that are never viewed, at the cost of one request per click.
 *
 * The passage is rendered as plain text (`<pre>`), never with `dangerouslySetInnerHTML`
 * (FR-22). The API returns raw text — the component treats it as text, not markup.
 */
interface Props {
  sources: Source[];
  /** Whether the answer abstained (refusal). Shown for context, but not actionable. */
  abstained?: boolean;
}

export default function SourceViewer({ sources, abstained = false }: Props) {
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [passages, setPassages] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<Record<string, string>>({});

  if (!sources.length) {
    return null;
  }

  async function toggle(chunkId: string) {
    const next = new Set(open);
    const expanded = next.has(chunkId);
    if (expanded) {
      next.delete(chunkId);
      setOpen(next);
      return;
    }

    next.add(chunkId);
    setOpen(next);

    // Already have it cached. A second expansion after collapsing should reuse the
    // fetched text, so the user is not charged another request and does not wait.
    if (passages[chunkId] !== undefined || loading[chunkId]) {
      return;
    }

    setLoading((s) => ({ ...s, [chunkId]: true }));
    setError((s) => ({ ...s, [chunkId]: "" }));

    try {
      const passage = await apiFetch<Passage>(`/chunks/${chunkId}`);
      setPassages((s) => ({ ...s, [chunkId]: passage.text }));
    } catch (e) {
      // Only an `ApiError` message is safe to show: the backend composed it for a
      // user (NFR-5). Anything else — a `TypeError: Failed to fetch`, a stack
      // message — is browser or JS jargon, so it is replaced rather than displayed.
      const msg =
        e instanceof ApiError ? e.message : "This passage could not be loaded.";
      setError((s) => ({ ...s, [chunkId]: msg }));
    } finally {
      setLoading((s) => ({ ...s, [chunkId]: false }));
    }
  }

  return (
    <aside className={styles["source-panel"]}>
      <h2>Sources</h2>
      <ul className={styles["source-list"]}>
        {sources.map((source) => {
          const isOpen = open.has(source.chunk_id);
          const text = passages[source.chunk_id];
          return (
            <li
              key={source.chunk_id}
              className={styles["source-item"]}
              data-open={isOpen ? "true" : "false"}
            >
              <button
                type="button"
                className={styles["source-toggle"]}
                onClick={() => toggle(source.chunk_id)}
                aria-expanded={isOpen}
                aria-controls={`passage-${source.chunk_id}`}
              >
                <span className={styles["source-index"]}>[{source.index}]</span>
                <div>
                  <span className={styles["source-name"]}>{source.filename}</span>
                  {(source.breadcrumb || source.page !== null) && (
                    <span className={styles["source-location"]}>
                      {source.breadcrumb}
                      {source.breadcrumb && source.page !== null ? " — " : ""}
                      {source.page !== null ? `p. ${source.page}` : ""}
                    </span>
                  )}
                </div>
              </button>

              {isOpen && (
                <div id={`passage-${source.chunk_id}`}>
                  {loading[source.chunk_id] && (
                    <div className={styles["source-status"]}>Loading passage…</div>
                  )}
                  {error[source.chunk_id] && (
                    <div className={`${styles["source-status"]} ${styles.error}`}>
                      {error[source.chunk_id]}
                    </div>
                  )}
                  {text !== undefined && (
                    <pre className={styles["source-passage"]}>{text}</pre>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>

      {abstained && (
        <p className="muted" style={{ marginTop: "0.6rem", fontSize: "0.85rem" }}>
          Answer refused — showing consulted sources only.
        </p>
      )}
    </aside>
  );
}