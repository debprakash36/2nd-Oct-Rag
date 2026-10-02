/**
 * Browser-side API client.
 *
 * Every call goes through `apiFetch` so the base URL, the JSON handling, and the
 * error shape are defined once. Backend errors arrive as `{detail: <user-safe text>}`
 * (NFR-5), so `detail` is what the UI shows — never `err.detail`, which is not
 * reachable from here.
 */

import { authHeaders, clearToken } from "./auth";

/**
 * Base URL for API calls.
 *
 * Read from the environment with a localhost default. `NEXT_PUBLIC_*` is inlined at
 * build time, so this cannot be changed at runtime — which is why it is a build
 * argument rather than an env file the container could patch.
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

/** An error whose message is safe to render (NFR-5: the backend already sanitized it). */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * `fetch` with JSON handling and a user-safe error message.
 *
 * Falls back to a generic message when the body is not the expected shape: a proxy
 * returning an HTML error page would otherwise surface as a JSON parse error, which
 * reads to the user as though the app were broken.
 */
export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    if (response.status === 401 && !path.startsWith("/auth/")) {
      clearToken();
    }
    throw new ApiError(response.status, await readError(response));
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Fall through to the generic message below.
  }
  return response.status === 429
    ? "You're sending questions too quickly. Please wait a moment."
    : "Something went wrong. Please try again.";
}