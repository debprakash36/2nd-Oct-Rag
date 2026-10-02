/**
 * Sign-in gate. A 401 later must return here; an open API must not block.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AuthGate from "@/components/AuthGate";
import { TOKEN_KEY } from "@/lib/auth";

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

describe("AuthGate", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it("renders children when the API is open", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("/auth/status")) return json({ required: false });
        throw new Error(`unexpected ${url}`);
      }),
    );
    render(
      <AuthGate>
        <p>inside</p>
      </AuthGate>,
    );
    await waitFor(() => expect(screen.getByText("inside")).toBeInTheDocument());
  });

  it("asks for a token when one is required and none is stored", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("/auth/status")) return json({ required: true });
        throw new Error(`unexpected ${url}`);
      }),
    );
    render(
      <AuthGate>
        <p>inside</p>
      </AuthGate>,
    );
    expect(await screen.findByLabelText("Access token")).toBeInTheDocument();
    expect(screen.queryByText("inside")).not.toBeInTheDocument();
  });

  it("stores an accepted token and then shows children", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        if (String(url).includes("/auth/status")) return json({ required: true });
        if (String(url).includes("/auth/login")) {
          expect(init?.method).toBe("POST");
          return json({ ok: true });
        }
        throw new Error(`unexpected ${url}`);
      }),
    );
    render(
      <AuthGate>
        <p>inside</p>
      </AuthGate>,
    );
    await screen.findByLabelText("Access token");
    await userEvent.type(screen.getByLabelText("Access token"), "secret-token");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(screen.getByText("inside")).toBeInTheDocument());
    expect(sessionStorage.getItem(TOKEN_KEY)).toBe("secret-token");
  });
});
