import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Checkin, type ClinicPatientDetail } from "../api";

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useParams: () => ({ patientId: "p1" }) };
});

import ClinicPatient from "../pages/clinic/ClinicPatient";

function checkin(overrides: Partial<Checkin> = {}): Checkin {
  return {
    id: "c1",
    user_id: "p1",
    plan_id: "pl1",
    plan_item_id: null,
    analysis_id: "a1",
    pain_nrs: 7,
    rpe: 6,
    note: "Sore after set 3",
    form_score: 60,
    flagged: true,
    flag_reasons: ["pain_high"],
    acknowledged_at: null,
    acknowledged_by: null,
    created_at: "2026-09-14T00:00:00Z",
    ...overrides,
  };
}

function detail(overrides: Partial<ClinicPatientDetail> = {}): ClinicPatientDetail {
  return {
    patient: {
      link_id: "l1",
      patient_id: "p1",
      display_name: "Alice",
      email: "alice@x.com",
      accepted_at: "2026-09-01T00:00:00Z",
      last_checkin_at: "2026-09-14T00:00:00Z",
      adherence_7d: 0.75,
      open_flags: 1,
      inactive_7d: false,
    },
    plans: [
      {
        id: "pl1",
        name: "Week 1",
        notes: null,
        template_key: null,
        started_at: "2026-09-01T00:00:00Z",
        created_at: "2026-09-01T00:00:00Z",
        updated_at: "2026-09-01T00:00:00Z",
        assigned_by: "u1",
        item_count: 4,
        completed_count: 2,
        day_count: 2,
        movements: ["Squat", "Row"],
      },
    ],
    checkins: [checkin()],
    trend: [
      { created_at: "2026-09-10T00:00:00Z", form_score: 80, pain_nrs: 2 },
      { created_at: "2026-09-14T00:00:00Z", form_score: 60, pain_nrs: 7 },
    ],
    open_flags: [checkin()],
    ...overrides,
  };
}

beforeEach(() => {
  vi.spyOn(api, "planTemplates").mockResolvedValue([]);
});
afterEach(() => vi.restoreAllMocks());

describe("ClinicPatient", () => {
  it("renders the summary tiles from the patient summary", async () => {
    vi.spyOn(api, "clinicPatient").mockResolvedValue(detail());
    renderWithProviders(<ClinicPatient />);
    expect(await screen.findByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument(); // adherence tile
    expect(screen.getByText("1")).toBeInTheDocument(); // open flags tile
  });

  it("renders the assigned-plans panel with the therapist badge, movements, and progress", async () => {
    vi.spyOn(api, "clinicPatient").mockResolvedValue(detail());
    renderWithProviders(<ClinicPatient />);
    expect(await screen.findByText("Week 1")).toBeInTheDocument();
    expect(screen.getByText("Assigned by your therapist")).toBeInTheDocument();
    expect(screen.getByText("2 of 4 done")).toBeInTheDocument();
    expect(screen.getByText("Squat")).toBeInTheDocument();
    expect(screen.getByText("Row")).toBeInTheDocument();
  });

  it("renders no plans-assigned state when there are none", async () => {
    vi.spyOn(api, "clinicPatient").mockResolvedValue(detail({ plans: [] }));
    renderWithProviders(<ClinicPatient />);
    expect(await screen.findByText("No plans assigned yet.")).toBeInTheDocument();
  });

  it("renders the trend chart when there are at least two points", async () => {
    vi.spyOn(api, "clinicPatient").mockResolvedValue(detail());
    renderWithProviders(<ClinicPatient />);
    expect(
      await screen.findByRole("img", { name: "Form score over the last 30 days" })
    ).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Pain over the last 30 days" })).toBeInTheDocument();
    expect(screen.queryByText("Not enough check-ins yet to chart a trend.")).not.toBeInTheDocument();
  });

  it("shows the trend empty state with fewer than two points", async () => {
    vi.spyOn(api, "clinicPatient").mockResolvedValue(
      detail({ trend: [{ created_at: "2026-09-14T00:00:00Z", form_score: 60, pain_nrs: 7 }] })
    );
    renderWithProviders(<ClinicPatient />);
    expect(await screen.findByText("Not enough check-ins yet to chart a trend.")).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /Form score/ })).not.toBeInTheDocument();
  });

  it("acknowledges a flagged check-in and refreshes the patient detail", async () => {
    const getPatient = vi.spyOn(api, "clinicPatient").mockResolvedValue(detail());
    const ack = vi.spyOn(api, "clinicAckFlag").mockResolvedValue({ checkin: checkin({ acknowledged_at: "x" }) });
    renderWithProviders(<ClinicPatient />);

    await userEvent.click(await screen.findByRole("button", { name: "Acknowledge" }));
    await waitFor(() => expect(ack).toHaveBeenCalledWith("c1"));
    await waitFor(() => expect(getPatient.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it("shows an already-acknowledged flag as text with its date, not a button", async () => {
    vi.spyOn(api, "clinicPatient").mockResolvedValue(
      detail({
        checkins: [checkin({ acknowledged_at: "2026-09-15T00:00:00Z" })],
        open_flags: [],
      })
    );
    renderWithProviders(<ClinicPatient />);
    await screen.findByText("Week 1");
    expect(screen.queryByRole("button", { name: "Acknowledge" })).not.toBeInTheDocument();
    expect(screen.getByText(/Acknowledged/)).toBeInTheDocument();
  });

  it("links to the analysis replay when a check-in carries an analysis_id", async () => {
    vi.spyOn(api, "clinicPatient").mockResolvedValue(detail());
    renderWithProviders(<ClinicPatient />);
    const link = await screen.findByRole("link", { name: "View analysis" });
    expect(link).toHaveAttribute("href", "/app?analysis=a1");
  });

  it("shows a not-found state on a 404 (patient not linked)", async () => {
    vi.spyOn(api, "clinicPatient").mockRejectedValue(new Error("404 Not Found for /api/clinic/patients/p1"));
    renderWithProviders(<ClinicPatient />);
    expect(await screen.findByText("This patient isn't linked to you.")).toBeInTheDocument();
  });

  it("shows a load-error state on a non-404 failure", async () => {
    vi.spyOn(api, "clinicPatient").mockRejectedValue(new Error("500 boom"));
    renderWithProviders(<ClinicPatient />);
    expect(await screen.findByText("Couldn't load this. Please try again.")).toBeInTheDocument();
  });
});
