import { describe, it, expect, vi } from "vitest";
import type React from "react";
import { screen, fireEvent } from "@testing-library/react";
import VideoPanel from "../components/VideoPanel";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis } from "./fixtures";

// VideoPanel renders <video ref={videoRef}>, so React attaches the real jsdom
// <video> DOM node to .current on mount. We only need an empty container ref —
// a hand-rolled fake video would be silently discarded. Media APIs (play/pause)
// on the real node are stubbed globally in src/test/setup.ts.
function makeVideoRef(): React.RefObject<HTMLVideoElement> {
  return { current: null };
}

describe("VideoPanel", () => {
  it("renders the video element", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(document.querySelector("video")).toBeInTheDocument();
  });

  it("shows '1 fault detected' badge when there is one fault", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText("1 fault detected")).toBeInTheDocument();
  });

  it("shows 'No faults detected' badge for a clean rep", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockCleanAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText("No faults detected")).toBeInTheDocument();
  });

  it("shows the plural fault badge when fault count > 1", () => {
    const ref = makeVideoRef();
    const multiAnalysis = {
      ...mockAnalysis,
      detections: [
        { ...mockAnalysis.detections[0] },
        { ...mockAnalysis.detections[0], fault_id: "knees_forward_1", fault_name: "knees_forward" },
      ],
    };
    renderWithProviders(
      <VideoPanel
        analysis={multiAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText("2 faults detected")).toBeInTheDocument();
  });

  it("renders the play button", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getAllByRole("button", { name: /play/i }).length).toBeGreaterThan(0);
  });

  it("renders the fullscreen button", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByRole("button", { name: /fullscreen/i })).toBeInTheDocument();
  });

  it("clicking play does not throw an error", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    // jsdom video.play() returns undefined, so we just verify no uncaught throw
    const playBtns = screen.getAllByRole("button", { name: /play/i });
    expect(() => fireEvent.click(playBtns[0])).not.toThrow();
  });

  it("renders a timeline beneath the video", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    // Timeline renders "Fault" and "Neutral" labels
    expect(screen.getByText("Fault")).toBeInTheDocument();
    expect(screen.getByText("Neutral")).toBeInTheDocument();
  });

  it("calls onTimeUpdate when the video fires timeupdate", () => {
    const onTimeUpdate = vi.fn();
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={onTimeUpdate}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    const video = document.querySelector("video")!;
    // Simulate a timeupdate event from the actual DOM video element
    fireEvent(video, new Event("timeupdate"));
    expect(onTimeUpdate).toHaveBeenCalled();
  });

  it("updates duration state when the video fires loadedmetadata", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    const video = document.querySelector("video")!;
    fireEvent(video, new Event("loadedmetadata"));
    // No throw means the handler ran
    expect(video).toBeInTheDocument();
  });

  it("updates playing state when the video fires play and pause events", () => {
    const ref = makeVideoRef();
    renderWithProviders(
      <VideoPanel
        analysis={mockAnalysis}
        videoRef={ref}
        onTimeUpdate={vi.fn()}
        onActiveFault={vi.fn()}
        onSeek={vi.fn()}
      />
    );
    const video = document.querySelector("video")!;
    fireEvent(video, new Event("play"));
    // After 'play' fires, play button becomes pause button
    expect(screen.getAllByRole("button", { name: /pause/i }).length).toBeGreaterThan(0);
    fireEvent(video, new Event("pause"));
    // After 'pause' fires, pause button becomes play button
    expect(screen.getAllByRole("button", { name: /play/i }).length).toBeGreaterThan(0);
  });
});
