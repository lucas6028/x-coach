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
    isLoggedIn: vi.fn().mockReturnValue(true),
    getIDToken: vi.fn().mockReturnValue("liff-id-token"),
    login: vi.fn(),
    logout: vi.fn(),
    ...overrides,
  };
}

function Probe() {
  const a = useAuth();
  const [msg, setMsg] = useState("");
  return (
    <div>
      <span data-testid="msg">{msg}</span>
      <span data-testid="authing">{String(a.lineAuthenticating)}</span>
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
      <button onClick={() => a.signOut()}>signout</button>
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
  // Default liffState.liff = null keeps the auto-login effect quiet unless a test opts in
  // with a fake LIFF. (health is no longer on the login path, but stays mocked for any
  // other caller.)
  mockApi.health.mockResolvedValue({ status: "ok", line_login_configured: false });
});

describe("signInWithLine — web path (external browser, via LIFF)", () => {
  // An external browser starts logged-out: the first tap hands off to LINE's login
  // redirect (liff.login), NOT Supabase's custom OIDC (which can't verify LINE's HS256
  // ID token). Returning to the app logged-in, the auto-login effect finishes the bridge.
  it("runs liff.login() to start LINE auth when not logged in", async () => {
    const liff = fakeLiff({
      isInClient: vi.fn().mockReturnValue(false),
      isLoggedIn: vi.fn().mockReturnValue(false),
    });
    liffState.liff = liff;
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(liff.login).toHaveBeenCalledTimes(1));
    expect(mockAuth.signInWithOAuth).not.toHaveBeenCalled();
    expect(mockApi.lineLogin).not.toHaveBeenCalled();
  });

  it("exchanges the ID token via the bridge once LINE-authenticated", async () => {
    const liff = fakeLiff({
      isInClient: vi.fn().mockReturnValue(false),
      isLoggedIn: vi.fn().mockReturnValue(true),
    });
    liffState.liff = liff;
    mockApi.lineLogin.mockResolvedValue({ access_token: "acc", refresh_token: "ref" });
    mockAuth.setSession.mockResolvedValue({ error: null });
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("line-done"));
    expect(mockApi.lineLogin).toHaveBeenCalledWith("liff-id-token");
    expect(mockAuth.setSession).toHaveBeenCalledWith({ access_token: "acc", refresh_token: "ref" });
    expect(mockAuth.signInWithOAuth).not.toHaveBeenCalled();
  });

  it("errors clearly when LIFF is unavailable (no VITE_LIFF_ID / init failed)", async () => {
    liffState.liff = null;
    renderProbe();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent(/not available/i));
    expect(mockAuth.signInWithOAuth).not.toHaveBeenCalled();
    expect(mockApi.lineLogin).not.toHaveBeenCalled();
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

  it("also completes in an external browser after LINE login (logged in, not in-client)", async () => {
    // The one-click web flow: liff.login() redirects out and back; on return isLoggedIn()
    // is true even though isInClient() is false, and the effect finishes the exchange.
    liffState.liff = fakeLiff({
      isInClient: vi.fn().mockReturnValue(false),
      isLoggedIn: vi.fn().mockReturnValue(true),
    });
    mockApi.health.mockResolvedValue({ status: "ok", line_login_configured: true });
    mockApi.lineLogin.mockResolvedValue({ access_token: "acc", refresh_token: "ref" });
    mockAuth.setSession.mockResolvedValue({ error: null });
    renderProbe();
    await waitFor(() => expect(mockApi.lineLogin).toHaveBeenCalledWith("liff-id-token"));
    await waitFor(() =>
      expect(mockAuth.setSession).toHaveBeenCalledWith({ access_token: "acc", refresh_token: "ref" })
    );
  });

  it("does nothing when not LINE-authenticated (logged out external browser)", async () => {
    liffState.liff = fakeLiff({
      isInClient: vi.fn().mockReturnValue(false),
      isLoggedIn: vi.fn().mockReturnValue(false),
    });
    mockApi.health.mockResolvedValue({ status: "ok", line_login_configured: true });
    renderProbe();
    await waitFor(() => expect(mockAuth.getSession).toHaveBeenCalled());
    expect(mockApi.lineLogin).not.toHaveBeenCalled();
  });

  it("stays silent when the bridge is unconfigured (503, swallowed)", async () => {
    liffState.liff = fakeLiff();
    mockApi.lineLogin.mockRejectedValue(new ApiError("LINE login is not configured.", 503));
    renderProbe();
    await waitFor(() => expect(mockApi.lineLogin).toHaveBeenCalled());
    expect(mockAuth.setSession).not.toHaveBeenCalled();
    expect(screen.getByTestId("msg")).toHaveTextContent("");
  });

  it("does nothing outside the LINE client", async () => {
    liffState.liff = null;
    renderProbe();
    // Settle the initial effects, then confirm no exchange was attempted.
    await waitFor(() => expect(mockAuth.getSession).toHaveBeenCalled());
    expect(mockApi.lineLogin).not.toHaveBeenCalled();
  });

  it("flags lineAuthenticating while the exchange is in flight, then clears it", async () => {
    liffState.liff = fakeLiff();
    let resolveLogin: (v: { access_token: string; refresh_token: string }) => void = () => {};
    mockApi.lineLogin.mockReturnValue(
      new Promise((resolve) => {
        resolveLogin = resolve;
      })
    );
    mockAuth.setSession.mockResolvedValue({ error: null });
    renderProbe();
    // Header uses this to show "signing in…" instead of the logged-out state.
    await waitFor(() => expect(screen.getByTestId("authing")).toHaveTextContent("true"));
    resolveLogin({ access_token: "acc", refresh_token: "ref" });
    await waitFor(() => expect(screen.getByTestId("authing")).toHaveTextContent("false"));
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

describe("signOut", () => {
  it("clears the LIFF login in an external browser so auto-login won't sign back in", async () => {
    const liff = fakeLiff({
      isInClient: vi.fn().mockReturnValue(false),
      isLoggedIn: vi.fn().mockReturnValue(true),
    });
    liffState.liff = liff;
    mockAuth.signOut.mockResolvedValue({ error: null });
    renderProbe();
    await userEvent.click(screen.getByText("signout"));
    await waitFor(() => expect(mockAuth.signOut).toHaveBeenCalled());
    expect(liff.logout).toHaveBeenCalledTimes(1);
  });

  it("leaves the LIFF session alone inside the LINE app (in-client)", async () => {
    const liff = fakeLiff(); // isInClient: true
    liffState.liff = liff;
    mockAuth.signOut.mockResolvedValue({ error: null });
    renderProbe();
    await userEvent.click(screen.getByText("signout"));
    await waitFor(() => expect(mockAuth.signOut).toHaveBeenCalled());
    expect(liff.logout).not.toHaveBeenCalled();
  });

  // Regression: the web flow finishes in the auto-login effect (no click handler survives
  // the redirect), so the one-shot guard must be cleared there — otherwise sign-out (which
  // flips isLoggedIn to false) leaves the guard poisoned and the NEXT sign-in dead-ends.
  it("web login → signout → sign-in again re-runs liff.login() (guard not poisoned)", async () => {
    // Post-redirect return state: LINE-authenticated in an external browser with the guard
    // set by the pre-redirect liff.login().
    const liff = fakeLiff({
      isInClient: vi.fn().mockReturnValue(false),
      isLoggedIn: vi.fn().mockReturnValue(true),
    });
    liffState.liff = liff;
    sessionStorage.setItem("xcoach.liffReloginTried", "1");
    mockApi.health.mockResolvedValue({ status: "ok", line_login_configured: true });
    mockApi.lineLogin.mockResolvedValue({ access_token: "acc", refresh_token: "ref" });
    mockAuth.setSession.mockResolvedValue({ error: null });
    mockAuth.signOut.mockResolvedValue({ error: null });
    renderProbe();

    // Auto-login completes the exchange and clears the guard.
    await waitFor(() => expect(mockAuth.setSession).toHaveBeenCalled());
    await waitFor(() => expect(sessionStorage.getItem("xcoach.liffReloginTried")).toBeNull());

    // Sign out: LIFF logs out too, so isLoggedIn() is now false.
    liff.isLoggedIn.mockReturnValue(false);
    await userEvent.click(screen.getByText("signout"));
    await waitFor(() => expect(mockAuth.signOut).toHaveBeenCalled());

    // Sign in again: must start a fresh LINE login, not throw the "failed" dead-end.
    liff.login.mockClear();
    await userEvent.click(screen.getByText("line"));
    await waitFor(() => expect(liff.login).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("msg")).not.toHaveTextContent(/failed/i);
  });
});
