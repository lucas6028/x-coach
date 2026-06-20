import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ReasoningLog from "../components/ReasoningLog";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";

describe("ReasoningLog", () => {
  it("shows the coaching feedback heading", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("Coaching Feedback")).toBeInTheDocument();
  });

  it("shows the badge", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("rule + GraphRAG")).toBeInTheDocument();
  });

  it("renders a fault card with the fault name", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();
  });

  it("shows 'High' severity chip for severity 0.8", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("High")).toBeInTheDocument();
  });

  it("shows the phase tag", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText(/descent phase/i)).toBeInTheDocument();
  });

  it("shows the start time", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("0:01")).toBeInTheDocument();
  });

  it("shows the likely cause from the retrieval", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("Likely cause:")).toBeInTheDocument();
    expect(screen.getByText("Weak hip abductors")).toBeInTheDocument();
  });

  it("shows the injury risk from the retrieval", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("Injury risk:")).toBeInTheDocument();
    expect(screen.getByText("ACL strain")).toBeInTheDocument();
  });

  it("shows the cue from the retrieval", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("Cue")).toBeInTheDocument();
    expect(screen.getByText("Drive knees out")).toBeInTheDocument();
  });

  it("shows 'no faults' message when there are no detections", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockCleanAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText(/No biomechanical faults/i)).toBeInTheDocument();
  });

  it("calls onSeek with the fault's start_time when the card is clicked", async () => {
    const user = userEvent.setup();
    const onSeek = vi.fn();
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={onSeek} />
    );
    await user.click(screen.getByRole("button", { name: /Knee Valgus/i }));
    expect(onSeek).toHaveBeenCalledWith(mockAnalysis.detections[0].start_time);
  });

  it("shows the key evidence value", () => {
    renderWithProviders(
      <ReasoningLog analysis={mockAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText(/valgus angle 0\.35/)).toBeInTheDocument();
  });
});
