import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// When Supabase is not configured (`supabase === null`), the provider resolves immediately to a
// logged-out state: the effect and signOut both early-return, and the admin probe never runs.
vi.mock("../lib/supabase", () => ({ isSupabaseConfigured: false, supabase: null }));
const { adminStatus } = vi.hoisted(() => ({ adminStatus: vi.fn() }));
vi.mock("../api", () => ({ api: { adminStatus } }));

import { AuthProvider, useAuth } from "../lib/auth";

function Probe() {
  const a = useAuth();
  return (
    <div>
      <span data-testid="configured">{String(a.configured)}</span>
      <span data-testid="loading">{String(a.loading)}</span>
      <span data-testid="adminState">{a.adminState}</span>
      <span data-testid="isAdmin">{String(a.isAdmin)}</span>
      <span data-testid="msg" />
      <button onClick={() => void a.signOut()}>signout</button>
      <button
        onClick={async () => {
          try {
            await a.signInWithPassword("e@x.com", "pw");
          } catch (err) {
            screen.getByTestId("msg").textContent = (err as Error).message;
          }
        }}
      >
        signin
      </button>
    </div>
  );
}

beforeEach(() => vi.clearAllMocks());

describe("AuthProvider (Supabase unconfigured)", () => {
  it("resolves immediately to a logged-out, non-admin state and never probes", () => {
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>
    );
    expect(screen.getByTestId("configured")).toHaveTextContent("false");
    expect(screen.getByTestId("loading")).toHaveTextContent("false");
    expect(screen.getByTestId("adminState")).toHaveTextContent("ready");
    expect(screen.getByTestId("isAdmin")).toHaveTextContent("false");
    expect(adminStatus).not.toHaveBeenCalled();
  });

  it("makes signOut a no-op instead of throwing", async () => {
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>
    );
    await userEvent.click(screen.getByText("signout"));
    // Nothing to assert beyond "did not throw"; the button handler ran cleanly.
    expect(screen.getByTestId("adminState")).toHaveTextContent("ready");
  });

  it("throws a clear 'not configured' error from the action methods", async () => {
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>
    );
    await userEvent.click(screen.getByText("signin"));
    await waitFor(() =>
      expect(screen.getByTestId("msg")).toHaveTextContent("Authentication is not configured.")
    );
  });
});
