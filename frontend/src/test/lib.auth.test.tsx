import { describe, it, expect, vi, beforeEach } from "vitest";
import { useState } from "react";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Drive the auth context against a fully mocked Supabase client.
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

vi.mock("../lib/supabase", () => ({
  isSupabaseConfigured: true,
  supabase: { auth: mockAuth },
}));

import { AuthProvider, useAuth } from "../lib/auth";

function Probe() {
  const a = useAuth();
  const [msg, setMsg] = useState("");
  return (
    <div>
      <span data-testid="configured">{String(a.configured)}</span>
      <span data-testid="user">{a.user?.email ?? "none"}</span>
      <span data-testid="msg">{msg}</span>
      <button
        onClick={async () => {
          try {
            await a.signInWithPassword("e@x.com", "pw");
            setMsg("signed-in");
          } catch (err) {
            setMsg((err as Error).message);
          }
        }}
      >
        signin
      </button>
      <button
        onClick={async () => {
          try {
            const r = await a.signUpWithPassword("e@x.com", "pw");
            setMsg(r.needsConfirmation ? "confirm" : "session");
          } catch (err) {
            setMsg((err as Error).message);
          }
        }}
      >
        signup
      </button>
      <button
        onClick={async () => {
          try {
            await a.signInWithGoogle();
            setMsg("redirect");
          } catch (err) {
            setMsg((err as Error).message);
          }
        }}
      >
        google
      </button>
      <button onClick={() => void a.signOut()}>signout</button>
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
  mockAuth.getSession.mockResolvedValue({ data: { session: null } });
  mockAuth.onAuthStateChange.mockReturnValue({
    data: { subscription: { unsubscribe: vi.fn() } },
  });
});

describe("AuthProvider / useAuth", () => {
  it("reports configured and a logged-out initial state", async () => {
    renderProbe();
    expect(screen.getByTestId("configured")).toHaveTextContent("true");
    await waitFor(() => expect(screen.getByTestId("user")).toHaveTextContent("none"));
  });

  it("hydrates the user from an existing session", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { email: "ada@x.com" } } },
    });
    renderProbe();
    await waitFor(() => expect(screen.getByTestId("user")).toHaveTextContent("ada@x.com"));
  });

  it("updates the user when auth state changes", async () => {
    let cb: (e: string, s: unknown) => void = () => {};
    mockAuth.onAuthStateChange.mockImplementation((fn: typeof cb) => {
      cb = fn;
      return { data: { subscription: { unsubscribe: vi.fn() } } };
    });
    renderProbe();
    await waitFor(() => expect(screen.getByTestId("user")).toHaveTextContent("none"));
    act(() => cb("SIGNED_IN", { user: { email: "grace@x.com" } }));
    await waitFor(() => expect(screen.getByTestId("user")).toHaveTextContent("grace@x.com"));
  });

  it("signs in with a password", async () => {
    mockAuth.signInWithPassword.mockResolvedValue({ error: null });
    renderProbe();
    await userEvent.click(screen.getByText("signin"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("signed-in"));
    expect(mockAuth.signInWithPassword).toHaveBeenCalledWith({ email: "e@x.com", password: "pw" });
  });

  it("surfaces a sign-in error", async () => {
    mockAuth.signInWithPassword.mockResolvedValue({ error: { message: "bad creds" } });
    renderProbe();
    await userEvent.click(screen.getByText("signin"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("bad creds"));
  });

  it("signs up and flags email confirmation when there is no session", async () => {
    mockAuth.signUp.mockResolvedValue({ data: { session: null }, error: null });
    renderProbe();
    await userEvent.click(screen.getByText("signup"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("confirm"));
  });

  it("signs up and proceeds when a session is returned", async () => {
    mockAuth.signUp.mockResolvedValue({ data: { session: { user: {} } }, error: null });
    renderProbe();
    await userEvent.click(screen.getByText("signup"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("session"));
  });

  it("surfaces a sign-up error", async () => {
    mockAuth.signUp.mockResolvedValue({ data: {}, error: { message: "taken" } });
    renderProbe();
    await userEvent.click(screen.getByText("signup"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("taken"));
  });

  it("starts a Google OAuth flow", async () => {
    mockAuth.signInWithOAuth.mockResolvedValue({ error: null });
    renderProbe();
    await userEvent.click(screen.getByText("google"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("redirect"));
    expect(mockAuth.signInWithOAuth).toHaveBeenCalledWith(
      expect.objectContaining({ provider: "google" })
    );
  });

  it("surfaces a Google OAuth error", async () => {
    mockAuth.signInWithOAuth.mockResolvedValue({ error: { message: "oauth off" } });
    renderProbe();
    await userEvent.click(screen.getByText("google"));
    await waitFor(() => expect(screen.getByTestId("msg")).toHaveTextContent("oauth off"));
  });

  it("signs out", async () => {
    mockAuth.signOut.mockResolvedValue({ error: null });
    renderProbe();
    await userEvent.click(screen.getByText("signout"));
    await waitFor(() => expect(mockAuth.signOut).toHaveBeenCalled());
  });

  it("throws when useAuth is used outside the provider", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<Probe />)).toThrow(/within an AuthProvider/);
    spy.mockRestore();
  });
});
