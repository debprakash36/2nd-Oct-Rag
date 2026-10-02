/** Session token for the API gate. Never written into the page source. */

export const TOKEN_KEY = "rag_api_token";
export const UNAUTHORIZED_EVENT = "rag-unauthorized";
export const AUTH_CHANGED_EVENT = "rag-auth-changed";

export function readToken(): string {
  if (typeof window === "undefined") return "";
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

export function storeToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token);
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
  window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
  window.dispatchEvent(new Event(AUTH_CHANGED_EVENT));
}

export function authHeaders(): Record<string, string> {
  const token = readToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
