"use client";

import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { TOKEN_KEY, UNAUTHORIZED_EVENT, storeToken } from "@/lib/auth";
import { API_BASE, ApiError } from "@/lib/api";
import styles from "./AuthGate.module.css";

type Mode = "checking" | "open" | "login" | "in";

/**
 * Asks for the API token when the server requires one.
 *
 * The token stays in sessionStorage and is attached by `apiFetch`. A later 401
 * (expired session, token rotated) drops back to this form.
 */
export default function AuthGate({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>("checking");
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const showLogin = () => {
      setMode("login");
      setError(null);
    };
    window.addEventListener(UNAUTHORIZED_EVENT, showLogin);

    void (async () => {
      try {
        const response = await fetch(`${API_BASE}/auth/status`);
        const body = (await response.json()) as { required?: boolean };
        if (cancelled) return;
        if (!body.required) {
          setMode("open");
          return;
        }
        setMode(sessionStorage.getItem(TOKEN_KEY) ? "in" : "login");
      } catch {
        if (!cancelled) setMode("open");
      }
    })();

    return () => {
      cancelled = true;
      window.removeEventListener(UNAUTHORIZED_EVENT, showLogin);
    };
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new ApiError(response.status, body.detail ?? "That sign-in token was not accepted.");
      }
      storeToken(token);
      setToken("");
      setMode("in");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sign-in failed.");
    } finally {
      setBusy(false);
    }
  }

  if (mode === "checking") {
    return <p className="muted">Checking access…</p>;
  }
  if (mode === "login") {
    return (
      <main className={styles.wrap}>
        <h2>Sign in</h2>
        <p className="muted">Enter the access token for this chatbot.</p>
        <form className={styles.form} onSubmit={(e) => void submit(e)}>
          <label>
            Access token
            <input
              type="password"
              autoComplete="current-password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              required
            />
          </label>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          <button type="submit" className="primary" disabled={busy || !token.trim()}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </main>
    );
  }
  return <>{children}</>;
}
