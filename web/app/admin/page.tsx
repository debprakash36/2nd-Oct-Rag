"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { authHeaders } from "@/lib/auth";
import { apiFetch, API_BASE } from "@/lib/api";
import type { DocumentRow, UploadResponse } from "@/lib/types";
import styles from "./admin.module.css";

/**
 * Admin console (FR-27).
 *
 * Shows each document's state and chunk count, with enable/disable and delete. The
 * upload form is included because a console you cannot add to is only half useful
 * during development; batch upload, duplicate handling, and the ingest queue's
 * operational surface are Phase 5 concerns and not built here.
 *
 * Gated by the same bearer token as the rest of the API when `API_TOKEN` is set.
 */
export default function AdminPage() {
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSummary, setUploadSummary] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const refresh = useCallback(async () => {
    try {
      // The endpoint returns a bare list, not an envelope.
      setDocuments(await apiFetch<DocumentRow[]>("/admin/documents"));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load documents.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function setState(doc: DocumentRow, action: "enable" | "disable") {
    setError(null);
    setNotice(null);
    setBusyId(doc.doc_id);
    try {
      await apiFetch(`/admin/documents/${doc.doc_id}/${action}`, { method: "POST" });
      setNotice(`${doc.filename} ${action}d.`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "That action failed.");
    } finally {
      setBusyId(null);
    }
  }

  async function remove(doc: DocumentRow) {
    setError(null);
    setNotice(null);
    // Deletion is destructive and immediate; the confirm is the only gate. A
    // "soft delete" undo would be friendlier but needs state this endpoint lacks.
    if (!confirm(`Delete ${doc.filename}? This cannot be undone.`)) {
      return;
    }
    setBusyId(doc.doc_id);
    try {
      await apiFetch(`/admin/documents/${doc.doc_id}`, { method: "DELETE" });
      setNotice(`${doc.filename} deleted.`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed.");
    } finally {
      setBusyId(null);
    }
  }

  async function upload(files: FileList | null) {
    setError(null);
    setNotice(null);
    setUploadSummary(null);
    if (!files || files.length === 0) return;

    setUploading(true);
    try {
      const form = new FormData();
      for (const file of Array.from(files)) {
        form.append("files", file);
      }
      const response = await fetch(`${API_BASE}/admin/documents`, {
        method: "POST",
        headers: authHeaders(),
        body: form,
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new Error(body.detail ?? "Upload failed.");
      }
      const body = (await response.json()) as UploadResponse;

      // Count on the backend's own tally rather than re-deriving it here, so a file
      // that was stored but queued rather than indexed is not reported as ingested.
      const parts: string[] = [`${body.accepted} ingested`];
      if (body.rejected) parts.push(`${body.rejected} rejected`);
      setUploadSummary(parts.join(", "));

      // Surface rejections and warnings as separate signals: a duplicate is a
      // *success* with a caveat, and conflating the two makes the admin think an
      // upload failed when it did not.
      const problems = body.results.filter((r) => r.error || r.warning);
      if (problems.length) {
        setError(
          problems
            .map((r) => `${r.filename}: ${r.error ?? r.warning}`)
            .join("; "),
        );
      }
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <main className={styles["admin-wrap"]}>
      <div className={styles["admin-header"]}>
        <h2>Documents</h2>
        <button type="button" onClick={() => void refresh()}>
          Refresh
        </button>
      </div>

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      {notice && <p className="muted">{notice}</p>}

      <form
        className={styles["upload-form"]}
        onSubmit={(e) => {
          e.preventDefault();
          void upload(fileRef.current?.files ?? null);
        }}
      >
        <input
          ref={fileRef}
          type="file"
          multiple
          aria-label="Choose documents to upload"
          // `.pdf,.md,.txt,.html` mirrors the backend's accepted MIME types. The
          // server validates independently; this only saves a round trip.
          accept=".pdf,.md,.markdown,.txt,.html,.htm"
        />
        <button type="submit" className="primary" disabled={uploading}>
          {uploading ? "Uploading…" : "Upload"}
        </button>
      </form>
      {uploadSummary && <p className={styles["upload-result"]}>{uploadSummary}</p>}

      {documents.length === 0 ? (
        <p className="empty-state">No documents indexed yet.</p>
      ) : (
        <table className={styles["admin-table"]}>
          <thead>
            <tr>
              <th scope="col">Filename</th>
              <th scope="col">State</th>
              <th scope="col" className="numeric">
                Chunks
              </th>
              <th scope="col" className="numeric">
                Size
              </th>
              <th scope="col">Uploaded</th>
              <th scope="col" />
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.doc_id}>
                <td>
                  <div className={styles.filename}>{doc.filename}</div>
                  {doc.error_reason && (
                    <div className={styles["error-reason"]}>{doc.error_reason}</div>
                  )}
                </td>
                <td>
                  <span className={`${styles["state-pill"]} ${styles[`state-${doc.state}`] ?? ""}`}>
                    {doc.state}
                  </span>
                </td>
                <td className="numeric">{doc.chunk_count}</td>
                <td className="numeric">{formatBytes(doc.byte_size)}</td>
                <td>{new Date(doc.uploaded_at).toLocaleDateString()}</td>
                <td>
                  <div className={styles.actions}>
                    {doc.state === "live" && (
                      <button
                        type="button"
                        disabled={busyId === doc.doc_id}
                        onClick={() => setState(doc, "disable")}
                      >
                        Disable
                      </button>
                    )}
                    {doc.state === "disabled" && (
                      <button
                        type="button"
                        disabled={busyId === doc.doc_id}
                        onClick={() => setState(doc, "enable")}
                      >
                        Enable
                      </button>
                    )}
                    <button
                      type="button"
                      disabled={busyId === doc.doc_id}
                      onClick={() => remove(doc)}
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}