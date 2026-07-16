import { describe, it, expect, vi, beforeEach } from "vitest";
import { useState } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// signInWithLine + the in-LIFF auto-login, driven against mocked supabase / liff / api.

const { mockAuth, liffState, mockApi } = vi.hoisted(() => ({
  mockAuth: {
    getSession: vi.fn(),
    onAuthStateChange: vi.fn(),
    signInWithOAuth: vi.fn(),
    setSession: vi.fn(),
    signOut: vi.fn(),
    signInWithPassword: vi.fn(),
    signUp: vi.fn(),
  },
  // Mutable stand-in for the LIFF SDK: null = "not in LIFF / init failed".
  liffState: { liff: null as unknown },
  mockApi: {
    health: vi.fn(),
    lineLogin: vi.fn(),
    adminStatus: vi.fn(),
  },
}));

vi.mock("../lib/supabase", () => ({
  isSupabaseConfigured: true,
  supabase: { auth: mockAuth },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => true,
  initLiff: vi.fn(() => Promise.resolve(liffState.liff)),
}));

// Keep ApiError (and everything else) real; swap only the api object.
vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, api: mockApi };
});

import { ApiError } from "../api";
import { AuthProvider, useAuth } from "../lib/auth";

function fakeLiff(overrides: Record<string, unknown> = {}) {
  return {
    isInClient: vi.fn().mockReturnValue(true),
    getIDToken: vi.fn().mockReturnValue("liff-id-token"),
    login: vi.fn(),
    ...overrides,
  };
}

function Probe() {
  const a = useAuth();
  const [msg, setMsg] = useState("");
  return (
    <div>
      <span data-testid="msg">{msg}</span>
      <button
        onClick={async () => {
          try {
            await a.signInWithLine();
            setMsg("line-done");
          } catch (err) {
            setMsg((err as Error).message);
          }
        }}
      >
        line
      </button>
    </div>
  );
}

function renderProbe() {
  return render(
    <AuthProvider>
      <Probe />
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  sessionStorage.clear();
  liffState.liff = null;
  mockAuth.getSession.mockResolvedValue({ data: { session: null } });
  mockAuth.onAuthStateChange.mockReturnValue({
    data: { subscription: { unsubscribe: vi.fn() } },
  });
  mockApi.adminStatus.mockResolvedValue({ is_admin: false });
  // Default: bridge unconfigured, so the auto-login effect stays quiet unless a test opts in.
  mockApi.health.mockResolvedValue({ status: "ok", line_login_configured: false });
});

describe("signInWithLine — web path", () => {
  it("starts the Supabase custom OIDC flow outside LIFF", async () => {
    mockAuth.signInWithOAuth.mockResolvedValue({ error: null });
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("line-done"));
    expect(mockAuth.signInWithOAuth).toHaveBeenCalledWith(
      expect.objectContaining({ provider: "custom:line" })
    );
    expect(mockApi.lineLogin).not.toHaveBeenCalled();
  });

  it("surfaces an OAuth error", async () => {
    mockAuth.signInWithOAuth.mockResolvedValue({ error: { message: "provider missing" } });
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("provider missing"));
  });
});

describe("signInWithLine — inside LIFF", () => {
  it("exchanges the ID token for a session via the bridge", async () => {
    liffState.liff = fakeLiff();
    mockApi.lineLogin.mockResolvedValue({ access_token: "acc", refresh_token: "ref" });
    mockAuth.setSession.mockResolvedValue({ error: null });
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("line-done"));
    expect(mockApi.lineLogin).toHaveBeenCalledWith("liff-id-token");
    expect(mockAuth.setSession).toHaveBeenCalledWith({ access_token: "acc", refresh_token: "ref" });
    expect(mockAuth.signInWithOAuth).not.toHaveBeenCalled();
  });

  it("re-runs liff.login() once when the bridge rejects a stale token (401)", async () => {
    const liff = fakeLiff();
    liffState.liff = liff;
    mockApi.lineLogin.mockRejectedValue(new ApiError("stale", 401));
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(liff.login).toHaveBeenCalledTimes(1));
    // The one-shot guard is now set: a second attempt errors instead of looping the redirect.
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent(/failed/i));
    expect(liff.login).toHaveBeenCalledTimes(1);
  });

  it("runs liff.login() when there is no ID token at all", async () => {
    const liff = fakeLiff({ getIDToken: vi.fn().mockReturnValue(null) });
    liffState.liff = liff;
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(liff.login).toHaveBeenCalledTimes(1));
    expect(mockApi.lineLogin).not.toHaveBeenCalled();
  });

  it("surfaces a non-401 bridge failure without reloading", async () => {
    const liff = fakeLiff();
    liffState.liff = liff;
    mockApi.lineLogin.mockRejectedValue(new ApiError("LINE login is not configured.", 503));
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() =>
      expect(screen.getByTestId("msg")).toHaveTextContent("LINE login is not configured.")
    );
    expect(liff.login).not.toHaveBeenCalled();
  });

  it("surfaces a setSession failure", async () => {
    liffState.liff = fakeLiff();
    mockApi.lineLogin.mockResolvedValue({ access_token: "acc", refresh_token: "ref" });
    mockAuth.setSession.mockResolvedValue({ error: { message: "invalid refresh token" } });
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() =>
      expect(screen.getByTestId("msg")).toHaveTextContent("invalid refresh token")
    );
  });
});

describe("in-LIFF auto-login", () => {
  it("silently exchanges the token on mount when the bridge is configured", async () => {
    liffState.liff = fakeLiff();
    mockApi.health.mockResolvedValue({ status: "ok", line_login_configured: true });
    mockApi.lineLogin.mockResolvedValue({ access_token: "acc", refresh_token: "ref" });
    mockAuth.setSession.mockResolvedValue({ error: null });
    renderProbe();
    await waitFor(() => expect(mockApi.lineLogin).toHaveBeenCalledWith("liff-id-token"));
    await waitFor(() =>
      expect(mockAuth.setSession).toHaveBeenCalledWith({ access_token: "acc", refresh_token: "ref" })
    );
  });

  it("does nothing when the server bridge is unconfigured", async () => {
    liffState.liff = fakeLiff();
    renderProbe();
    await waitFor(() => expect(mockApi.health).toHaveBeenCalled());
    expect(mockApi.lineLogin).not.toHaveBeenCalled();
  });

  it("does nothing outside the LINE client", async () => {
    liffState.liff = null;
    renderProbe();
    // Settle the initial effects, then confirm no exchange was attempted.
    await waitFor(() => expect(mockAuth.getSession).toHaveBeenCalled());
    expect(mockApi.health).not.toHaveBeenCalled();
    expect(mockApi.lineLogin).not.toHaveBeenCalled();
  });

  it("stays silent when the exchange fails", async () => {
    liffState.liff = fakeLiff();
    mockApi.health.mockResolvedValue({ status: "ok", line_login_configured: true });
    mockApi.lineLogin.mockRejectedValue(new ApiError("boom", 502));
    renderProbe();
    await waitFor(() => expect(mockApi.lineLogin).toHaveBeenCalled());
    expect(mockAuth.setSession).not.toHaveBeenCalled();
    expect(screen.getByTestId("msg")).toHaveTextContent("");
  });
});
