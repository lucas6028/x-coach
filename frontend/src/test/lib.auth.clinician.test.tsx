import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Mirrors lib.auth.admin.test.tsx exactly, for the clinician-role probe (WP3). The probe only
// fires when the session carries a real user id (see AuthProvider.refreshClinician), so each
// signed-in test resolves getSession with a session whose user.id is set.
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

const { clinicStatus, adminStatus } = vi.hoisted(() => ({
  clinicStatus: vi.fn(),
  // AuthProvider probes both roles on the same trigger — a partial mock without this would make
  // `api.adminStatus()` a TypeError the moment a real session resolves, unrelated to this file.
  adminStatus: vi.fn().mockResolvedValue({ is_admin: false }),
}));
vi.mock("../api", () => ({ api: { clinicStatus, adminStatus } }));

import { AuthProvider, useAuth } from "../lib/auth";

function ClinicianProbe() {
  const a = useAuth();
  return (
    <div>
      <span data-testid="isClinician">{String(a.isClinician)}</span>
      <span data-testid="clinicianState">{a.clinicianState}</span>
      <button onClick={() => a.refreshClinician()}>refresh</button>
    </div>
  );
}

function renderProbe() {
  return render(
    <AuthProvider>
      <ClinicianProbe />
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockAuth.onAuthStateChange.mockReturnValue({ data: { subscription: { unsubscribe: vi.fn() } } });
  adminStatus.mockResolvedValue({ is_admin: false });
});

describe("AuthProvider clinician probe", () => {
  it("probes the clinician role for a signed-in user: loading → ready(clinician)", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { id: "u1", email: "ada@x.com" } } },
    });
    let resolve!: (v: { is_clinician: boolean }) => void;
    clinicStatus.mockImplementationOnce(() => new Promise((r) => { resolve = r; }));
    renderProbe();

    await waitFor(() => expect(screen.getByTestId("clinicianState")).toHaveTextContent("loading"));
    await act(async () => resolve({ is_clinician: true }));

    await waitFor(() => expect(screen.getByTestId("clinicianState")).toHaveTextContent("ready"));
    expect(screen.getByTestId("isClinician")).toHaveTextContent("true");
    expect(clinicStatus).toHaveBeenCalledTimes(1);
  });

  it("marks the probe errored when the status request rejects", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { id: "u1", email: "ada@x.com" } } },
    });
    let reject!: (e: unknown) => void;
    clinicStatus.mockImplementationOnce(() => new Promise((_r, rej) => { reject = rej; }));
    renderProbe();

    await waitFor(() => expect(screen.getByTestId("clinicianState")).toHaveTextContent("loading"));
    await act(async () => reject(new Error("403 forbidden")));

    await waitFor(() => expect(screen.getByTestId("clinicianState")).toHaveTextContent("error"));
    expect(screen.getByTestId("isClinician")).toHaveTextContent("false");
  });

  it("re-runs the probe via refreshClinician: error → ready(clinician) without a reload", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { id: "u1", email: "ada@x.com" } } },
    });
    clinicStatus
      .mockImplementationOnce(() => Promise.reject(new Error("boom")))
      .mockImplementationOnce(() => Promise.resolve({ is_clinician: true }));
    renderProbe();

    await waitFor(() => expect(screen.getByTestId("clinicianState")).toHaveTextContent("error"));
    await userEvent.click(screen.getByText("refresh"));

    await waitFor(() => expect(screen.getByTestId("clinicianState")).toHaveTextContent("ready"));
    expect(screen.getByTestId("isClinician")).toHaveTextContent("true");
    expect(clinicStatus).toHaveBeenCalledTimes(2);
  });

  // The monotonic probe-id guard (mirrors the admin probe's probeIdRef): a stale in-flight request
  // resolving after a newer one must not clobber the state the newer probe already settled.
  it("ignores a stale in-flight probe superseded by a manual refresh", async () => {
    mockAuth.getSession.mockResolvedValue({
      data: { session: { user: { id: "u1", email: "ada@x.com" } } },
    });
    let resolveFirst!: (v: { is_clinician: boolean }) => void;
    clinicStatus
      .mockImplementationOnce(() => new Promise((r) => { resolveFirst = r; }))
      .mockImplementationOnce(() => Promise.resolve({ is_clinician: true }));
    renderProbe();
    await waitFor(() => expect(clinicStatus).toHaveBeenCalledTimes(1));

    await userEvent.click(screen.getByText("refresh"));
    await waitFor(() => expect(clinicStatus).toHaveBeenCalledTimes(2));

    // The first (now-stale) probe resolves late with a DIFFERENT answer than the second — it must
    // be ignored, not overwrite the second probe's result.
    await act(async () => resolveFirst({ is_clinician: false }));

    await waitFor(() => expect(screen.getByTestId("clinicianState")).toHaveTextContent("ready"));
    expect(screen.getByTestId("isClinician")).toHaveTextContent("true");
  });

  it("resolves logged-out immediately to false/ready and never probes", async () => {
    mockAuth.getSession.mockResolvedValue({ data: { session: null } });
    renderProbe();
    await waitFor(() => expect(screen.getByTestId("clinicianState")).toHaveTextContent("ready"));
    expect(screen.getByTestId("isClinician")).toHaveTextContent("false");
    expect(clinicStatus).not.toHaveBeenCalled();
  });
});
