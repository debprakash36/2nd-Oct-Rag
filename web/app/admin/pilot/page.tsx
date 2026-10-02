"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import type {
  PilotDiagnosis,
  PilotMetricsResponse,
  PilotReviewItem,
  PilotReviewResponse,
  ThresholdSnapshot,
} from "@/lib/types";
import styles from "../admin.module.css";

/**
 * Ops view of the Phase 6 scripts. It does not close the exit gate.
 *
 * UNMEASURED is rendered as a gap, never as success. Classifying a query is
 * on-demand because it runs retrieval.
 */
export default function PilotPage() {
  const [metrics, setMetrics] = useState<PilotMetricsResponse | null>(null);
  const [review, setReview] = useState<PilotReviewResponse | null>(null);
  const [history, setHistory] = useState<ThresholdSnapshot[]>([]);
  const [historyNote, setHistoryNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [diagnoses, setDiagnoses] = useState<Record<string, PilotDiagnosis>>({});
  const [busyQuery, setBusyQuery] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [m, r, h] = await Promise.all([
        apiFetch<PilotMetricsResponse>("/admin/pilot/metrics"),
        apiFetch<PilotReviewResponse>("/admin/pilot/review"),
        apiFetch<{ snapshots: ThresholdSnapshot[]; note: string }>(
          "/admin/pilot/threshold-history",
        ),
      ]);
      setMetrics(m);
      setReview(r);
      setHistory(h.snapshots);
      setHistoryNote(h.note);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load pilot data.");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function classify(item: PilotReviewItem) {
    setBusyQuery(item.query);
    setError(null);
    try {
      const diagnosis = await apiFetch<PilotDiagnosis>("/admin/pilot/review/classify", {
        method: "POST",
        body: JSON.stringify({ query: item.query }),
      });
      setDiagnoses((prev) => ({ ...prev, [item.query]: diagnosis }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Classification failed.");
    } finally {
      setBusyQuery(null);
    }
  }

  return (
    <main className={styles["admin-wrap"]}>
      <div className={styles["admin-header"]}>
        <h2>Pilot metrics</h2>
        <button type="button" onClick={() => void load()}>
          Refresh
        </button>
      </div>
      <p className="muted">
        This page reports the PRD §8 traffic gate. It cannot mark Phase 6 complete:
        that needs real users and a classified improvement-loop cycle.
      </p>

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      {metrics && (
        <>
          <p>
            Gate:{" "}
            <strong>
              {metrics.gate === "earned" ? "earned" : "not earned"}
            </strong>
            {metrics.raw.distinct_queries >= 0 && (
              <span className="muted">
                {" "}
                · {metrics.raw.total} log rows, {metrics.raw.distinct_queries} distinct
                queries
              </span>
            )}
          </p>
          <p className="muted">{metrics.note}</p>
          <table className={styles["admin-table"]}>
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">State</th>
                <th scope="col">Value</th>
                <th scope="col">Target</th>
                <th scope="col">Detail</th>
              </tr>
            </thead>
            <tbody>
              {metrics.metrics.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td>
                    <span
                      className={`${styles["state-pill"]} ${
                        m.state === "pass"
                          ? styles["state-live"]
                          : m.state === "fail"
                            ? styles["state-failed"]
                            : styles["state-disabled"]
                      }`}
                    >
                      {m.state === "unmeasured" ? "UNMEASURED" : m.state.toUpperCase()}
                    </span>
                  </td>
                  <td className="numeric">{formatValue(m.value)}</td>
                  <td>{m.target}</td>
                  <td>{m.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <h3 className={styles.section}>Threshold history</h3>
      {historyNote && <p className="muted">{historyNote}</p>}
      {history.length === 0 ? (
        <p className="empty-state">No threshold snapshots recorded yet.</p>
      ) : (
        <table className={styles["admin-table"]}>
          <thead>
            <tr>
              <th scope="col">Recorded</th>
              <th scope="col">Commit</th>
              <th scope="col" className="numeric">
                Chunks
              </th>
              <th scope="col" className="numeric">
                Recall@10
              </th>
              <th scope="col" className="numeric">
                Refusal
              </th>
              <th scope="col">Recommended</th>
            </tr>
          </thead>
          <tbody>
            {history.map((snap, i) => (
              <tr key={`${snap.recorded_at}-${i}`}>
                <td>{snap.recorded_at?.slice(0, 19) ?? "—"}</td>
                <td>{snap.commit ?? "—"}</td>
                <td className="numeric">{snap.chunks ?? "—"}</td>
                <td className="numeric">
                  {snap.recall_at_10 == null ? "—" : snap.recall_at_10.toFixed(3)}
                </td>
                <td className="numeric">
                  {snap.refusal_rate == null ? "—" : `${(snap.refusal_rate * 100).toFixed(1)}%`}
                </td>
                <td>{snap.recommended ?? "none"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3 className={styles.section}>Improvement loop</h3>
      <p className="muted">
        Classify a refused or down-voted query before changing retrieval. A content
        gap needs a document; a retrieval gap needs a different fix.
      </p>
      {review && review.items.length === 0 ? (
        <p className="empty-state">{review.empty_reason}</p>
      ) : (
        <table className={styles["admin-table"]}>
          <thead>
            <tr>
              <th scope="col">Query</th>
              <th scope="col">Why</th>
              <th scope="col" />
            </tr>
          </thead>
          <tbody>
            {review?.items.map((item) => {
              const diagnosis = diagnoses[item.query];
              return (
                <tr key={item.query}>
                  <td>
                    <div>{item.query}</div>
                    {diagnosis && (
                      <div className="muted">
                        {diagnosis.classification}: {diagnosis.fix}
                      </div>
                    )}
                  </td>
                  <td>{item.why}</td>
                  <td>
                    <button
                      type="button"
                      disabled={busyQuery === item.query}
                      onClick={() => void classify(item)}
                    >
                      {busyQuery === item.query ? "Classifying…" : "Classify"}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </main>
  );
}

function formatValue(value: number | null): string {
  if (value == null) return "—";
  if (value <= 1) return `${(value * 100).toFixed(1)}%`;
  return `${Math.round(value)} ms`;
}
