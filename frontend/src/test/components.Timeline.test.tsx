import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Timeline from "../components/Timeline";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";

describe("Timeline", () => {
  // The standalone timeline strip is gone: the reference design carries ONE scrub bar, inside the
  // video card's control pill, and a legend does not fit in a 6px pill. The labels survive as the
  // markers' `title` (asserted below), which is what actually named them for a pointer user.
  it("names each fault segment rather than printing a legend", () => {
    renderWithProviders(
      <Timeline analysis={mockAnalysis} duration={10} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.queryByText("Neutral")).toBeNull();
    expect(document.querySelector("[title^='Fault:']")).toBeInTheDocument();
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
});
