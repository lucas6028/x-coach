import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CoachTray from "../components/CoachTray";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";

// The grounded coaching feedback (fault cards / clean-rep) is the top of the unified tray and is
// always visible regardless of auth — only the composer at the foot adapts. These cover the
// feedback half (the chat half lives in components.CoachTray.chat.test.tsx).
describe("CoachTray — coaching feedback", () => {
  it("shows the AI Coach heading and the rule+GraphRAG provenance badge", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText("AI Coach")).toBeInTheDocument();
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

  it("shows the phase tag", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText(/descent phase/i)).toBeInTheDocument();
  });

  it("shows the start time", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />);
    expect(screen.getByText("0:01")).toBeInTheDocument();
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
    expect(screen.getByText(/No biomechanical faults/i)).toBeInTheDocument();
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
      "border-primary/50"
    );
  });

  it("leaves the fault card inactive once the playhead passes its end time", () => {
    renderWithProviders(<CoachTray analysis={mockAnalysis} currentTime={3} onSeek={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Knee Valgus/i }).className).not.toContain(
      "border-primary/50"
    );
  });
});
