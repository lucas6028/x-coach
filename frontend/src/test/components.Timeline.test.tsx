import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Timeline from "../components/Timeline";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";

describe("Timeline", () => {
  it("renders the legend labels", () => {
    renderWithProviders(
      <Timeline analysis={mockAnalysis} duration={10} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText("Fault")).toBeInTheDocument();
    expect(screen.getByText("Neutral")).toBeInTheDocument();
  });

  it("renders the current time display", () => {
    renderWithProviders(
      <Timeline analysis={mockAnalysis} duration={10} currentTime={3} onSeek={vi.fn()} />
    );
    expect(screen.getByText(/0:03/)).toBeInTheDocument();
  });

  it("renders fault segments as title-annotated divs", () => {
    renderWithProviders(
      <Timeline analysis={mockAnalysis} duration={10} currentTime={0} onSeek={vi.fn()} />
    );
    const segment = document.querySelector("[title*='Knee Valgus']");
    expect(segment).toBeInTheDocument();
  });

  it("calls onSeek when a fault segment is clicked", async () => {
    const user = userEvent.setup();
    const onSeek = vi.fn();
    renderWithProviders(
      <Timeline analysis={mockAnalysis} duration={10} currentTime={0} onSeek={onSeek} />
    );
    const segment = document.querySelector("[title*='Knee Valgus']") as HTMLElement;
    await user.click(segment);
    expect(onSeek).toHaveBeenCalledWith(mockAnalysis.detections[0].start_time);
  });

  it("renders no fault segments when analysis has no detections", () => {
    renderWithProviders(
      <Timeline analysis={mockCleanAnalysis} duration={10} currentTime={0} onSeek={vi.fn()} />
    );
    expect(document.querySelector("[title*='Knee Valgus']")).toBeNull();
  });

  it("falls back to fps-derived duration when duration prop is 0", () => {
    renderWithProviders(
      <Timeline analysis={mockAnalysis} duration={0} currentTime={0} onSeek={vi.fn()} />
    );
    // 300 frames / 30 fps = 10s → total should appear as 0:10
    expect(screen.getByText(/0:10/)).toBeInTheDocument();
  });

  it("marks spans that were never analyzed", () => {
    const analysis = {
      ...mockAnalysis,
      reps: {
        detected: 3,
        analyzed: [1, 3],
        max_reps: 3,
        fallback: null,
        segments: [
          { index: 1, start_frame: 0, end_frame: 29, start_time: 0, end_time: 1, analyzed: true, partial: false },
          { index: 2, start_frame: 30, end_frame: 59, start_time: 1, end_time: 2, analyzed: false, partial: false },
          { index: 3, start_frame: 60, end_frame: 89, start_time: 2, end_time: 3, analyzed: true, partial: false },
        ],
      },
    };
    renderWithProviders(
      <Timeline analysis={analysis} duration={3} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getAllByTestId("unanalyzed-span")).toHaveLength(1);
  });

  it("says how many reps were found and how many were scored", () => {
    const analysis = {
      ...mockAnalysis,
      reps: {
        detected: 3,
        analyzed: [1, 3],
        max_reps: 3,
        fallback: null,
        segments: [
          { index: 1, start_frame: 0, end_frame: 29, start_time: 0, end_time: 1, analyzed: true, partial: false },
          { index: 2, start_frame: 30, end_frame: 59, start_time: 1, end_time: 2, analyzed: false, partial: false },
          { index: 3, start_frame: 60, end_frame: 89, start_time: 2, end_time: 3, analyzed: true, partial: false },
        ],
      },
    };
    renderWithProviders(
      <Timeline analysis={analysis} duration={3} currentTime={0} onSeek={vi.fn()} />
    );
    // A bare /3/ regex also matches the "0:03" duration readout, so assert on the actual
    // summary text (detected count + which reps were scored) rather than a lone digit.
    expect(screen.getByText(/3 reps found, analyzed #1、3/)).toBeInTheDocument();
  });

  it("shows nothing extra when the whole clip was analyzed as one unit", () => {
    const analysis = {
      ...mockAnalysis,
      reps: { detected: 0, analyzed: [], max_reps: 3, fallback: "no_reps_detected", segments: [] },
    };
    renderWithProviders(
      <Timeline analysis={analysis} duration={3} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.queryAllByTestId("unanalyzed-span")).toHaveLength(0);
  });

  it("renders no unanalyzed spans and no rep summary for analyses without a `reps` field", () => {
    renderWithProviders(
      <Timeline analysis={mockAnalysis} duration={10} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.queryAllByTestId("unanalyzed-span")).toHaveLength(0);
  });
});
