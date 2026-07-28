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
    // summary text (detected count + which reps were scored) rather than a lone digit. The
    // separator is ASCII ", " in English — see `timeline.repsListSeparator` — not the zh-Hant
    // full-width "、", which would read wrong inside a Latin sentence.
    expect(screen.getByText(/3 reps found, analyzed #1, 3/)).toBeInTheDocument();
  });

  it("uses the zh-Hant enumeration comma when the summary list has more than one rep", () => {
    localStorage.setItem("lang", "zh-Hant");
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
    try {
      renderWithProviders(
        <Timeline analysis={analysis} duration={3} currentTime={0} onSeek={vi.fn()} />
      );
      expect(screen.getByText(/共 3 下，分析了第 1、3 下/)).toBeInTheDocument();
    } finally {
      localStorage.removeItem("lang");
    }
  });

  // Regression test: on a REAL fallback (e.g. "only_partial_reps" — see
  // src/pose/movements/base.py) `run.reps` is NOT necessarily empty, and the backend marks every
  // segment `analyzed: true` even though `reps.analyzed` itself stays `[]` (nothing was scored
  // PER-REP; the whole clip was scored as one unit — see pose_rule_detector.py:707-713). A span
  // map that reads `reps.analyzed.includes(segment.index)` instead of `segment.analyzed` would
  // wrongly mark this segment unexamined; a fixture with an empty `segments` array can't catch
  // that bug because both a correct and a buggy filter agree there's nothing to mark.
  it("does not mark a segment unanalyzed on a fallback that still reports a rep span", () => {
    const analysis = {
      ...mockAnalysis,
      reps: {
        detected: 1,
        analyzed: [],
        max_reps: 3,
        fallback: "only_partial_reps",
        segments: [
          { index: 1, start_frame: 0, end_frame: 89, start_time: 0, end_time: 3, analyzed: true, partial: true },
        ],
      },
    };
    renderWithProviders(
      <Timeline analysis={analysis} duration={3} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.queryAllByTestId("unanalyzed-span")).toHaveLength(0);
    expect(screen.getByText("Whole clip analyzed")).toBeInTheDocument();
  });

  it("renders no unanalyzed spans and no rep summary for analyses without a `reps` field", () => {
    renderWithProviders(
      <Timeline analysis={mockAnalysis} duration={10} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.queryAllByTestId("unanalyzed-span")).toHaveLength(0);
    expect(screen.queryByText(/reps found/)).not.toBeInTheDocument();
    expect(screen.queryByText("Whole clip analyzed")).not.toBeInTheDocument();
  });
});
