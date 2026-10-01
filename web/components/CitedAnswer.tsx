"use client";

import { Fragment, useCallback, useMemo, useRef, useState } from "react";
import type { Source } from "@/lib/types";
import styles from "./CitedAnswer.module.css";

/**
 * An answer with its `[n]` citation markers rendered as real controls (FR-18, NFR-9).
 *
 * ## Why this component exists
 *
 * The backend emits answer text with markers embedded in it -- `app/generation/
 * validator.py` matches them with `MARKER_RE = re.compile(r"\[(\d+)\]")` -- and they
 * arrive inside ordinary `token` events as plain text. Rendering `{turn.content}`
 * directly showed a reader "Refunds are issued within 30 days [1]." where `[1]` is
 * indistinguishable from any other bracketed number in the sentence: a pointer to
 * nothing, spoken as "bracket one bracket" by a screen reader.
 *
 * The marker is the only link between a claim and the passage supporting it. If it
 * cannot be operated, the citation is decoration and FR-18 is unmet in practice
 * however well the server validates it.
 *
 * ## What it does
 *
 * Splits the text on the marker pattern and renders each marker as a `<button>` that
 * expands the corresponding source in place. Using a real `<button>` rather than a
 * `<span role="button">` means Space and Enter work, it is in the tab order, and it
 * is announced as a button -- all without reimplementing what the platform already
 * does correctly.
 *
 * ## Security
 *
 * Everything here renders through JSX text nodes; there is no `dangerouslySetInnerHTML`
 * anywhere in this component, and `React` escapes children by default (FR-22). Corpus
 * text is untrusted input and is never treated as markup. The marker indices are
 * parsed with the same `\[(\d+)\]` shape the server validates, then used only as
 * numeric keys into a `Map` -- never as object property access on untrusted input.
 */
interface Props {
  content: string;
  sources: Source[];
}

export default function CitedAnswer({ content, sources }: Props) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [passages, setPassages] = useState<Record<number, string>>({});
  const [loading, setLoading] = useState<Record<number, boolean>>({});
  const [errors, setErrors] = useState<Record<number, string>>({});
  const baseId = useRef(`cited-${Math.random().toString(36).slice(2, 9)}`).current;

  // Memoised because `toggle` depends on it: a fresh Map every render would change
  // the callback's identity every render, and the effect of that is a marker that
  // re-fetches its passage on every parent render.
  const byIndex = useMemo(
    () => new Map(sources.map((s) => [s.index, s])),
    [sources],
  );

  const toggle = useCallback(
    async (index: number) => {
      if (openIndex === index) {
        setOpenIndex(null);
        return;
      }
      setOpenIndex(index);
      // Already fetched: re-opening must not cost another request.
      if (passages[index] !== undefined || loading[index]) return;

      const source = byIndex.get(index);
      if (!source) return;

      setLoading((s) => ({ ...s, [index]: true }));
      setErrors((s) => ({ ...s, [index]: "" }));
      try {
        const { apiFetch } = await import("@/lib/api");
        const passage = await apiFetch<{ text: string }>(`/chunks/${source.chunk_id}`);
        setPassages((s) => ({ ...s, [index]: passage.text }));
      } catch (e) {
        // Only an ApiError message is user-safe (NFR-5); anything else is JS jargon.
        const msg =
          e instanceof Error && e.name === "ApiError"
            ? e.message
            : "This passage could not be loaded.";
        setErrors((s) => ({ ...s, [index]: msg }));
      } finally {
        setLoading((s) => ({ ...s, [index]: false }));
      }
    },
    [openIndex, passages, loading, byIndex],
  );

  // Matches the server's MARKER_RE. A marker with no matching source (stripped, or
  // from a conversation loaded without its sources) stays as literal text rather than
  // becoming a button that does nothing.
  const parts = content.split(/(\[\d+\])/g);

  return (
    <div className={styles.answer}>
      {parts.map((part, i) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (!match) {
          return <Fragment key={i}>{part}</Fragment>;
        }

        const index = Number(match[1]);
        const source = byIndex.get(index);
        if (!source) {
          return <Fragment key={i}>{part}</Fragment>;
        }

        const expanded = openIndex === index;
        const panelId = `${baseId}-panel-${index}`;

        return (
          <Fragment key={i}>
            <button
              type="button"
              className={styles.marker}
              aria-expanded={expanded}
              aria-controls={panelId}
              aria-label={`Citation ${index}: ${source.filename}${
                source.page !== null ? `, page ${source.page}` : ""
              }`}
              onClick={() => void toggle(index)}
            >
              {index}
            </button>
            {expanded && (
              <span
                id={panelId}
                role="region"
                aria-label={`Passage cited as ${index}`}
                className={styles.panel}
              >
                {loading[index] && <span className={styles.status}>Loading passage…</span>}
                {errors[index] && (
                  <span className={styles.error} role="alert">
                    {errors[index]}
                  </span>
                )}
                {passages[index] !== undefined && (
                  <span className={styles.passage}>{passages[index]}</span>
                )}
              </span>
            )}
          </Fragment>
        );
      })}
    </div>
  );
}
