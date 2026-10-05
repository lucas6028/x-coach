import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";

// Mirrors lib.auth.clinician.test.tsx, for the NLF 3D-view enablement probe. Unlike the admin/
// clinician probes, canUse3d exposes no loading state or manual refresh in AuthValue — just the
// settled boolean — so these tests assert its before/after transitions instead of a status field.
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

const { nlfStatus, adminStatus, clinicStatus } = vi.hoisted(() => ({
  nlfStatus: vi.fn(),
  // AuthProvider probes all three flags on the same trigger — a partial mock without these would
  // make the other two calls TypeErrors the moment a real session resolves, unrelated to this file.
  adminStatus: vi.fn().mockResolvedValue({ is_admin: false }),
  clinicStatus: vi.fn().mockResolvedValue({ is_clinician: false }),
}));
vi.mock("../api", () => ({ api: { nlfStatus, adminStatus, clinicStatus } }));

import { AuthProvider, useAuth } from "../lib/auth";

function NlfProbe() {
  const a = useAuth();
  return <span data-testid="canUse3d">{String(a.canUse3d)}</span>;
}

function renderProbe() {
  return render(
    <AuthProvider>
      <NlfProbe />
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockAuth.onAuthStateChange.mockReturnValue({ data: { subscription: { unsubscribe: vi.fn() } } });
  adminStatus.mockResolvedValue({ is_admin: false });
  clinicStatus.mockResolvedValue({ is_clinician: false });
});

describe("AuthProvider NLF 3D-view probe", () => {
  it("resolves canUse3d true for an enabled signed-in user", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { id: "u1", email: "ada@x.com" } } },
    });
    nlfStatus.mockResolvedValue({ enabled: true });
    renderProbe();

    await waitFor(() => expect(screen.getByTestId("canUse3d")).toHaveTextContent("true"));
    expect(nlfStatus).toHaveBeenCalledTimes(1);
  });

  it("resolves canUse3d false for a signed-in user who isn't enabled", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { id: "u1", email: "ada@x.com" } } },
    });
    nlfStatus.mockResolvedValue({ enabled: false });
    renderProbe();

    await waitFor(() => expect(nlfStatus).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("canUse3d")).toHaveTextContent("false");
  });

  it("falls back to false when the status probe rejects", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { id: "u1", email: "ada@x.com" } } },
    });
    nlfStatus.mockRejectedValue(new Error("500 boom"));
    renderProbe();

    await waitFor(() => expect(nlfStatus).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("canUse3d")).toHaveTextContent("false");
  });

  it("resolves logged-out immediately to false and never probes", async () => {
    mockAuth.getSession.mockResolvedValue({ data: { session: null } });
    renderProbe();
    await waitFor(() => expect(screen.getByTestId("canUse3d")).toHaveTextContent("false"));
    expect(nlfStatus).not.toHaveBeenCalled();
  });

  // The monotonic probe-id guard (mirrors the admin/clinician probes' probeIdRef): a stale in-flight
  // request from a superseded user id must not clobber the state the newer probe already settled.
  // Simulated via Supabase's own auth-state-change event, since canUse3d exposes no manual refresh.
  it("ignores a stale in-flight probe superseded by a user-id change", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { id: "u1", email: "ada@x.com" } } },
    });
    let authStateCb: ((event: string, session: unknown) => void) | undefined;
    mockAuth.onAuthStateChange.mockImplementation((cb: (event: string, session: unknown) => void) => {
      authStateCb = cb;
      return { data: { subscription: { unsubscribe: vi.fn() } } };
    });
    let resolveFirst!: (v: { enabled: boolean }) => void;
    nlfStatus
      .mockImplementationOnce(() => new Promise((r) => { resolveFirst = r; }))
      .mockImplementationOnce(() => Promise.resolve({ enabled: true }));

    renderProbe();
    await waitFor(() => expect(nlfStatus).toHaveBeenCalledTimes(1));

    // Switch to a different signed-in user before the first probe resolves.
    await act(async () => {
      authStateCb?.("SIGNED_IN", { user: { id: "u2", email: "bob@x.com" } });
    });
    await waitFor(() => expect(nlfStatus).toHaveBeenCalledTimes(2));

    // The first (u1) probe resolves late with a DIFFERENT answer than the second — it must be
    // ignored, not overwrite the second probe's result.
    await act(async () => resolveFirst({ enabled: false }));

    await waitFor(() => expect(screen.getByTestId("canUse3d")).toHaveTextContent("true"));
  });
});
