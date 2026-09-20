import { describe, it, expect, vi, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan } from "../api";
import CreatePlanDialog from "../components/plans/CreatePlanDialog";

function plan(overrides: Partial<Plan> = {}): Plan {
  return {
    id: "p1",
    name: "Week 1",
    notes: null,
    template_key: null,
    started_at: null,
    created_at: "2026-09-15T00:00:00Z",
    updated_at: "2026-09-15T00:00:00Z",
    items: [],
    ...overrides,
  };
}

afterEach(() => vi.restoreAllMocks());

describe("CreatePlanDialog — forPatient (WP3 assign flow)", () => {
  it("without forPatient, submits via api.createPlan and never touches clinicAssignPlan", async () => {
    const createPlan = vi.spyOn(api, "createPlan").mockResolvedValue(plan());
    const clinicAssignPlan = vi.spyOn(api, "clinicAssignPlan").mockResolvedValue(plan());
    const onCreated = vi.fn();

    renderWithProviders(
      <CreatePlanDialog open onCancel={vi.fn()} onCreated={onCreated} />
    );
    await userEvent.type(screen.getByLabelText(/name/i), "Week 1");
    await userEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(createPlan).toHaveBeenCalledWith({ name: "Week 1", notes: null }));
    expect(clinicAssignPlan).not.toHaveBeenCalled();
    expect(onCreated).toHaveBeenCalledWith("p1");
  });

  it("with forPatient set, submits via api.clinicAssignPlan(patientId, …) and never calls createPlan", async () => {
    const createPlan = vi.spyOn(api, "createPlan").mockResolvedValue(plan());
    const clinicAssignPlan = vi.spyOn(api, "clinicAssignPlan").mockResolvedValue(plan());
    const onCreated = vi.fn();

    renderWithProviders(
      <CreatePlanDialog
        open
        forPatient={{ id: "patient-1", name: "Alice" }}
        onCancel={vi.fn()}
        onCreated={onCreated}
      />
    );

    // The dialog's own copy names the patient it is assigning to.
    expect(screen.getByText("Assign to Alice")).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/name/i), "Rehab week 1");
    await userEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(clinicAssignPlan).toHaveBeenCalledWith("patient-1", { name: "Rehab week 1", notes: null })
    );
    expect(createPlan).not.toHaveBeenCalled();
    expect(onCreated).toHaveBeenCalledWith("p1");
  });

  it("prefixes a template name when both forPatient and a template are given", async () => {
    const clinicAssignPlan = vi.spyOn(api, "clinicAssignPlan").mockResolvedValue(plan());

    renderWithProviders(
      <CreatePlanDialog
        open
        templateKey="quick_core"
        templateName="Quick core"
        forPatient={{ id: "patient-1", name: "Alice" }}
        onCancel={vi.fn()}
        onCreated={vi.fn()}
      />
    );

    expect(screen.getByLabelText(/name/i)).toHaveValue("Quick core");
    await userEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(clinicAssignPlan).toHaveBeenCalledWith("patient-1", {
        name: "Quick core",
        notes: null,
        template_key: "quick_core",
      })
    );
  });
});
