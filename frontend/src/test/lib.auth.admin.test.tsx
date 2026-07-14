import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Drive the AuthProvider's admin-role probe against a mocked Supabase client + a mocked api.adminStatus.
// The probe only fires when the session carries a real user id (see AuthProvider.refreshAdmin), so each
// test resolves getSession with a session whose user.id is set.
const { mockAuth } = vi.hoisted(() => ({
  mockAuth: {
    getSession: vi.fn(),
    onAuthStateChange: vi.fn(),
    signInWithPassword: vi.fn(),
    signUp: vi.fn(),
    signInWithOAuth: vi.fn(),
    signOut: vi.fn(),
  },
}));
vi.mock("../lib/supabase", () => ({ isSupabaseConfigured: true, supabase: { auth: mockAuth } }));

const { adminStatus } = vi.hoisted(() => ({ adminStatus: vi.fn() }));
vi.mock("../api", () => ({ api: { adminStatus } }));

import { AuthProvider, useAuth } from "../lib/auth";

function AdminProbe() {
  const a = useAuth();
  return (
    <div>
      <span data-testid="isAdmin">{String(a.isAdmin)}</span>
      <span data-testid="adminState">{a.adminState}</span>
      <button onClick={() => a.refreshAdmin()}>refresh</button>
    </div>
  );
}

function renderProbe() {
  return render(
    <AuthProvider>
      <AdminProbe />
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockAuth.onAuthStateChange.mockReturnValue({ data: { subscription: { unsubscribe: vi.fn() } } });
  mockAuth.getSession.mockResolvedValue({ data: { session: { user: { id: "u1", email: "ada@x.com" } } } });
});

describe("AuthProvider admin probe", () => {
  it("probes the admin role for a signed-in user: loading → ready(admin)", async () => {
    // Park the probe in-flight so the "loading" state is observable before resolving it.
    let resolve!: (v: { is_admin: boolean }) => void;
    adminStatus.mockImplementationOnce(() => new Promise((r) => { resolve = r; }));
    renderProbe();

    await waitFor(() => expect(screen.getByTestId("adminState")).toHaveTextContent("loading"));
    await act(async () => resolve({ is_admin: true }));

    await waitFor(() => expect(screen.getByTestId("adminState")).toHaveTextContent("ready"));
    expect(screen.getByTestId("isAdmin")).toHaveTextContent("true");
    expect(adminStatus).toHaveBeenCalledTimes(1);
  });

  it("marks the probe errored when the status request rejects", async () => {
    let reject!: (e: unknown) => void;
    adminStatus.mockImplementationOnce(() => new Promise((_r, rej) => { reject = rej; }));
    renderProbe();

    await waitFor(() => expect(screen.getByTestId("adminState")).toHaveTextContent("loading"));
    await act(async () => reject(new Error("403 forbidden")));

    await waitFor(() => expect(screen.getByTestId("adminState")).toHaveTextContent("error"));
    expect(screen.getByTestId("isAdmin")).toHaveTextContent("false");
  });

  it("re-runs the probe via refreshAdmin: error → ready(admin) without a reload", async () => {
    adminStatus
      .mockImplementationOnce(() => Promise.reject(new Error("boom"))) // initial mount probe fails
      .mockImplementationOnce(() => Promise.resolve({ is_admin: true })); // manual refresh succeeds
    renderProbe();

    await waitFor(() => expect(screen.getByTestId("adminState")).toHaveTextContent("error"));
    await userEvent.click(screen.getByText("refresh"));

    await waitFor(() => expect(screen.getByTestId("adminState")).toHaveTextContent("ready"));
    expect(screen.getByTestId("isAdmin")).toHaveTextContent("true");
    expect(adminStatus).toHaveBeenCalledTimes(2);
  });
});
