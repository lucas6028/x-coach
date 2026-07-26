import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import CoachTray from "../components/CoachTray";
import { mockCleanAnalysis, mockUnmeasuredAnalysis } from "./fixtures";

describe("CoachTray — the clean-rep verdict names the movement", () => {
  it("names the movement whose rules ran", () => {
    renderWithProviders(
      <CoachTray
        analysis={{ ...mockCleanAnalysis, movement: "Push-up" }}
        currentTime={0}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText(/No Push-up faults detected/i)).toBeInTheDocument();
  });

  it("falls back to Squat for an analysis predating per-movement selection", () => {
    renderWithProviders(
      <CoachTray analysis={mockCleanAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText(/No Squat faults detected/i)).toBeInTheDocument();
  });

  it("still refuses to claim a clean rep on an unmeasured clip", () => {
    renderWithProviders(
      <CoachTray
        analysis={{ ...mockUnmeasuredAnalysis, movement: "Overhead Press" }}
        currentTime={0}
        onSeek={vi.fn()}
      />
    );
    expect(screen.queryByText(/No Overhead Press faults detected/i)).not.toBeInTheDocument();
    // The brief's draft regex read "could not be measured" — the actual, unchanged
    // feedback.notMeasured copy (frontend/src/lib/i18n.tsx) reads "could be measured", matching
    // the existing assertion in components.CoachTray.test.tsx. Fixed here rather than weakened:
    // still requires the not-measured banner, still refuses the movement-scoped clean-rep text.
    expect(screen.getByText(/could be measured/i)).toBeInTheDocument();
  });
});
