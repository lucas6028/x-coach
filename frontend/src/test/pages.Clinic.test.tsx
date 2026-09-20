import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { api, type ClinicPatientSummary } from "../api";
import ClinicLayout from "../pages/clinic/ClinicLayout";
import ClinicPatients from "../pages/clinic/ClinicPatients";
import ClinicPatient from "../pages/clinic/ClinicPatient";
import ClinicInvite from "../pages/clinic/ClinicInvite";

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
const mockUseAuth = vi.mocked(useAuth);

// Render the real nested /clinic route tree so ClinicLayout's gate and <Outlet/> resolve exactly
// as in production — same idiom as pages.admin.test.tsx's renderAdmin.
function renderClinic(path: string) {
  return render(
    <I18nProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/clinic" element={<ClinicLayout />}>
            <Route index element={<ClinicPatients />} />
            <Route path="patients/:patientId" element={<ClinicPatient />} />
            <Route path="invite" element={<ClinicInvite />} />
          </Route>
          <Route path="/app" element={<div>app studio</div>} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

function authValue(overrides: Record<string, unknown> = {}) {
  return {
    user: { id: "u1", email: "therapist@x.com" },
    isClinician: true,
    clinicianState: "ready",
    refreshClinician: vi.fn(),
    ...overrides,
  } as unknown as ReturnType<typeof useAuth>;
}

const PATIENTS: ClinicPatientSummary[] = [
  {
    link_id: "l1",
    patient_id: "p1",
    display_name: "Alice",
    email: "alice@x.com",
    accepted_at: "2026-09-01T00:00:00Z",
    last_checkin_at: "2026-09-14T00:00:00Z",
    adherence_7d: 0.8,
    open_flags: 2,
    inactive_7d: false,
  },
  {
    link_id: "l2",
    patient_id: "p2",
    display_name: null,
    email: "bob@x.com",
    accepted_at: "2026-08-01T00:00:00Z",
    last_checkin_at: null,
    adherence_7d: null,
    open_flags: 0,
    inactive_7d: true,
  },
];

beforeEach(() => {
  mockUseAuth.mockReturnValue(authValue());
  vi.spyOn(api, "clinicPatients").mockResolvedValue(PATIENTS);
});
afterEach(() => vi.restoreAllMocks());

// The page renders BOTH the wide table and the phone-card list (CSS `hidden`/`lg:` classes have no
// effect in jsdom, which does not load the real stylesheet) — so anything that shows up in both
// markups is asserted with getAllByText, the same accommodation pages.admin.test.tsx makes for the
// admin rail's desktop-shell + mobile-drawer duplication.
describe("ClinicPatients", () => {
  it("renders both linked patients", async () => {
    renderClinic("/clinic");
    expect((await screen.findAllByText("Alice")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("bob@x.com").length).toBeGreaterThan(0);
  });

  it("shows a red flag badge with the open-flag count on the flagged patient", async () => {
    renderClinic("/clinic");
    expect((await screen.findAllByText("2 flags")).length).toBeGreaterThan(0);
  });

  it("shows the inactive badge for a patient quiet for 7 days", async () => {
    renderClinic("/clinic");
    await screen.findAllByText("bob@x.com");
    expect(screen.getAllByText("7 days quiet").length).toBeGreaterThan(0);
  });

  it("denies a non-clinician the same way AdminLayout does, and never fetches patients", async () => {
    mockUseAuth.mockReturnValue(authValue({ isClinician: false }));
    const patients = vi.spyOn(api, "clinicPatients").mockResolvedValue(PATIENTS);
    renderClinic("/clinic");
    expect(
      await screen.findByText("You don't have access to the clinician dashboard.")
    ).toBeInTheDocument();
    expect(patients).not.toHaveBeenCalled();
  });

  it("shows the loading state while the clinician probe is in flight", () => {
    mockUseAuth.mockReturnValue(authValue({ clinicianState: "loading" }));
    renderClinic("/clinic");
    expect(screen.getByText("Checking your access…")).toBeInTheDocument();
  });

  it("shows an error with retry when the clinician probe fails", async () => {
    const refreshClinician = vi.fn();
    mockUseAuth.mockReturnValue(authValue({ clinicianState: "error", refreshClinician }));
    renderClinic("/clinic");
    const retry = await screen.findByRole("button", { name: "Retry" });
    retry.click();
    expect(refreshClinician).toHaveBeenCalledTimes(1);
  });

  it("shows the disclaimer", async () => {
    renderClinic("/clinic");
    expect(
      await screen.findByText(/is not a medical diagnosis/i)
    ).toBeInTheDocument();
  });
});
