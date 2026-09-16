import { describe, it, expect, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import TemplatePickerDialog from "../components/clinic/TemplatePickerDialog";
import type { PlanTemplate } from "../api";

function template(overrides: Partial<PlanTemplate> = {}): PlanTemplate {
  return {
    key: "quick_core",
    name: "Quick core session",
    description: "One 15-minute session.",
    items: [{ day_index: 1, movement: "Sit-up", sets: 3, reps: 15 }],
    ...overrides,
  };
}

describe("TemplatePickerDialog — grouping", () => {
  it("lists a rehab template under 'Rehab plans' and a fitness template under 'General training'", () => {
    renderWithProviders(
      <TemplatePickerDialog
        open
        templates={[
          template({ key: "knee_rehab", name: "Knee rehab", category: "rehab" }),
          template({ key: "quick_core", name: "Quick core session", category: "fitness" }),
        ]}
        onPick={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    const rehabGroup = screen.getByText("Rehab plans").closest("div") as HTMLElement;
    expect(within(rehabGroup).getByText("Knee rehab")).toBeInTheDocument();
    expect(within(rehabGroup).queryByText("Quick core session")).not.toBeInTheDocument();

    const fitnessGroup = screen.getByText("General training").closest("div") as HTMLElement;
    expect(within(fitnessGroup).getByText("Quick core session")).toBeInTheDocument();
    expect(within(fitnessGroup).queryByText("Knee rehab")).not.toBeInTheDocument();
  });

  it("buckets a template with no category under 'General training', not dropped", () => {
    renderWithProviders(
      <TemplatePickerDialog
        open
        templates={[template({ key: "quick_core", name: "Quick core session", category: undefined })]}
        onPick={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.queryByText("Rehab plans")).not.toBeInTheDocument();
    const fitnessGroup = screen.getByText("General training").closest("div") as HTMLElement;
    expect(within(fitnessGroup).getByText("Quick core session")).toBeInTheDocument();
  });

  it("calls onPick with the chosen template", async () => {
    const onPick = vi.fn();
    const { default: userEvent } = await import("@testing-library/user-event");
    const knee = template({ key: "knee_rehab", name: "Knee rehab", category: "rehab" });
    renderWithProviders(
      <TemplatePickerDialog open templates={[knee]} onPick={onPick} onCancel={vi.fn()} />
    );
    await userEvent.click(screen.getByRole("button", { name: /knee rehab/i }));
    expect(onPick).toHaveBeenCalledWith(knee);
  });
});
