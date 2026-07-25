import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CoachTray from "../components/CoachTray";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis, mockUnmeasuredAnalysis } from "./fixtures";

// The grounded coaching feedback (fault cards / clean-rep) is the top of the unified tray and is
// always visible regardless of auth — only the composer at the foot adapts. These cover the
// feedback half (the chat half lives in components.CoachTray.chat.test.tsx).
describe("CoachTray — coaching feedback", () => {
  it("shows the Lumen coach heading and the rule+GraphRAG provenance badge", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText("Lumen")).toBeInTheDocument();
    expect(screen.getByText("rule + GraphRAG")).toBeInTheDocument();
  });

  it("renders a fault card with the fault name", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();
  });

  it("shows the 'High' severity chip for severity 0.8", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText("High")).toBeInTheDocument();
  });

  it("shows the phase in the header", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    // Header merges timecode and phase: "0:01 · Descent".
    expect(screen.getByText(/Descent/)).toBeInTheDocument();
  });

  it("shows the start time", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText(/0:01/)).toBeInTheDocument();
  });

  it("shows the likely cause, injury risk, and cue from the retrieval", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText("Weak hip abductors")).toBeInTheDocument();
    expect(screen.getByText("ACL strain")).toBeInTheDocument();
    expect(screen.getByText("Drive knees out")).toBeInTheDocument();
  });

  it("shows the key evidence value", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText(/valgus angle 0\.35/)).toBeInTheDocument();
  });

  it("shows the clean-rep message when there are no detections", () => {
    renderWithProviders(<CoachTray analysis={mockCleanAnalysis} currentTime={0} onSeek={vi.fn()} />);
    // Movement-scoped since 2a5d3e64's mitigation (see components.CoachTray.movement.test.tsx):
    // mockCleanAnalysis carries no `movement`, so this exercises the "Squat" fallback branch.
    expect(screen.getByText(/No Squat faults detected/i)).toBeInTheDocument();
  });

  // An empty `detections` list means BOTH "no faults found" and "no frame was measurable". The
  // tray co-renders with MetricsCards (App.tsx), so before this gate a knees-up-cropped clip
  // showed "Faults 0 — not measured" in the HUD and "Clean rep" in the tray, side by side.
  it("shows the not-measured message, NOT the clean-rep one, when no frame was valid", () => {
    renderWithProviders(
      <CoachTray analysis={mockUnmeasuredAnalysis} currentTime={0} onSeek={vi.fn()} />,
    );
    expect(screen.getByText(/could be measured/i)).toBeInTheDocument();
    // Movement-scoped fallback text (see above) — still asserts the clean-rep banner is absent.
    expect(screen.queryByText(/No Squat faults detected/i)).not.toBeInTheDocument();
  });

  it("calls onSeek with the fault's start_time when its card is clicked", async () => {
    const user = userEvent.setup();
    const onSeek = vi.fn();
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={onSeek} />);
    await user.click(screen.getByRole("button", { name: /Knee Valgus/i }));
    expect(onSeek).toHaveBeenCalledWith(mockAnalysis.detections[0].start_time);
  });

  it("marks the fault card active while the playhead is inside its [start, end] window", () => {
    // mockDetection spans [1.0, 2.5] — 1.5 is inside it.
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={1.5} onSeek={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Knee Valgus/i }).className).toContain(
      "border-primary/40"
    );
  });

  it("leaves the fault card inactive once the playhead passes its end time", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={3} onSeek={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Knee Valgus/i }).className).not.toContain(
      "border-primary/40"
    );
  });
});
