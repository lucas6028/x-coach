import { describe, it, expect, vi, beforeEach } from "vitest";
import type React from "react";
import { screen, fireEvent } from "@testing-library/react";
import type { Analysis } from "../api";
import StudioMobile from "../components/mobile/StudioMobile";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockCleanAnalysis, mockUnmeasuredAnalysis } from "./fixtures";

// The coach thread is stubbed: it owns its own chat/knowledge network calls and has four test files
// of its own (components.CoachTray*.test.tsx). What this suite is about is whether the phone page
// MOUNTS it behind the disclosure — not what it renders once open.
vi.mock("../components/CoachTray", () => ({
  default: () => <div data-testid="coach-tray">coach thread</div>,
}));

// `videoRef` is the studio's own ref, threaded down from App so the page and the scrubber address
// one element. StudioMobile renders <video ref={...}>, so React fills this in on mount.
const makeRef = () => ({ current: null }) as React.RefObject<HTMLVideoElement>;

function renderStudio(analysis: Analysis = mockAnalysis, over: Record<string, unknown> = {}) {
  const props = {
    onTimeUpdate: vi.fn(),
    onActiveFault: vi.fn(),
    onSeek: vi.fn(),
    onNewSession: vi.fn(),
    ...over,
  };
  renderWithProviders(
    <StudioMobile
      analysis={analysis}
      videoRef={makeRef()}
      currentTime={0}
      activeFaultId={null}
      onTimeUpdate={props.onTimeUpdate as () => void}
      onActiveFault={props.onActiveFault as () => void}
      onSeek={props.onSeek as () => void}
      onNewSession={props.onNewSession as () => void}
    />
  );
  return props;
}

// Opens one of the four disclosure rows by its title and hands back the panel body.
function openRow(title: RegExp | string) {
  const row = screen.getByRole("button", { name: new RegExp(title, "i") });
  expect(row).toHaveAttribute("aria-expanded", "false");
  fireEvent.click(row);
  expect(row).toHaveAttribute("aria-expanded", "true");
}

beforeEach(() => vi.clearAllMocks());

describe("StudioMobile — headline", () => {
  it("renders the derived form score and its band", () => {
    renderStudio();
    // One severity-0.8 fault → 100 − 25×0.8 = 80 → "good".
    expect(screen.getByText("80")).toBeInTheDocument();
    expect(screen.getByText("Good")).toBeInTheDocument();
    expect(screen.getByText("Form score")).toBeInTheDocument();
  });

  // The `wasMeasured` invariant, on the phone: an unmeasured clip has an empty detection list for
  // the same reason a flawless one does, so it must not be scored 100/Excellent.
  it("shows no score at all when the clip was never measurable", () => {
    renderStudio(mockUnmeasuredAnalysis);
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("Nothing measurable in this clip")).toBeInTheDocument();
    expect(screen.queryByText("Excellent")).not.toBeInTheDocument();
  });

  it("scores a measured clean rep 100", () => {
    renderStudio(mockCleanAnalysis);
    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.getByText("Excellent")).toBeInTheDocument();
  });

  it("renders the clip duration from its metadata", () => {
    renderStudio();
    // 300 frames at 30fps.
    expect(screen.getByText("Duration")).toBeInTheDocument();
    expect(screen.getAllByText("0:10").length).toBeGreaterThan(0);
  });

  // The mock shows "Reps 8/12 · Target 12". There is no target anywhere in this product, and a rep
  // count only exists when the per-rep detector produced one — so the tile is conditional.
  it("omits the reps tile when the analysis carries no rep count", () => {
    renderStudio();
    expect(screen.queryByText("Reps")).not.toBeInTheDocument();
  });

  it("shows the rep count when the detector produced one", () => {
    const withReps = {
      ...mockAnalysis,
      detections: [{ ...mockAnalysis.detections[0], rep_count: 5 }],
    };
    renderStudio(withReps);
    // Once in the headline row, once in the stage's floating metrics card.
    expect(screen.getAllByText("Reps")).toHaveLength(2);
    expect(screen.getAllByText("5")).toHaveLength(2);
  });
});

describe("StudioMobile — the coach card", () => {
  // Collapsed it is one grounded line, and the line is RETRIEVED, not generated: the corrective cue
  // the knowledge graph returned for the worst fault.
  it("collapses to the corrective cue for the worst fault", () => {
    renderStudio();
    expect(screen.getByText("Drive knees out")).toBeInTheDocument();
  });

  it("falls back to the clean-rep verdict when nothing was detected", () => {
    renderStudio(mockCleanAnalysis);
    expect(screen.getAllByText(/No Squat faults detected/i).length).toBeGreaterThan(0);
  });

  it("mounts the coach thread only once the card is expanded", async () => {
    renderStudio();
    expect(screen.queryByTestId("coach-tray")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Lumen/i }));
    expect(screen.getByTestId("coach-tray")).toBeInTheDocument();
  });
});

describe("StudioMobile — disclosure rows", () => {
  it("summarises the issue count and lists the faults when opened", async () => {
    const { onSeek } = renderStudio();
    expect(screen.getByText("1 detected")).toBeInTheDocument();
    openRow("Top issues");
    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();
    expect(screen.getByText("High")).toBeInTheDocument(); // severity 0.8
    // The evidence line, so the row says what was measured and not just what was named.
    expect(screen.getAllByText(/valgus angle 0.35/).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText("Knee Valgus"));
    expect(onSeek).toHaveBeenCalledWith(1.0);
  });

  it("says 'clean rep' rather than 'no issues' when the clip was measured and clean", async () => {
    renderStudio(mockCleanAnalysis);
    expect(screen.getByText("clean rep")).toBeInTheDocument();
    openRow("Top issues");
    expect(screen.getAllByText(/No Squat faults detected/i).length).toBeGreaterThan(0);
  });

  // Same empty detection list, different meaning — the distinction the whole `wasMeasured` helper
  // exists to keep.
  it("says 'not measured' rather than 'clean rep' when nothing was measurable", async () => {
    renderStudio(mockUnmeasuredAnalysis);
    expect(screen.getByText("not measured")).toBeInTheDocument();
    openRow("Top issues");
    expect(screen.queryByText(/Clean rep/i)).not.toBeInTheDocument();
  });

  it("lists the retrieved causes and corrections under the research row", async () => {
    renderStudio();
    openRow("Based on research");
    expect(screen.getByText("Weak hip abductors")).toBeInTheDocument();
    // "Drive knees out" is also the collapsed coach line, so both render.
    expect(screen.getAllByText("Drive knees out").length).toBe(2);
  });

  it("says so when the knowledge graph returned no cue", async () => {
    renderStudio(mockCleanAnalysis);
    openRow("Based on research");
    expect(
      screen.getByText(/knowledge graph returned no corrective cue/i)
    ).toBeInTheDocument();
  });

  it("mounts the past-sessions list behind its own row", async () => {
    renderStudio();
    expect(screen.queryByText(/Previous/i)).not.toBeInTheDocument();
    openRow("Past sessions");
    // Signed out in this suite, so the card renders its sign-in prompt rather than fetching.
    expect(screen.getByText(/sign in/i)).toBeInTheDocument();
  });

  it("reports the clip's own quality figures under the metrics row", async () => {
    renderStudio();
    openRow("Detailed metrics");
    expect(screen.getByText("Side")).toBeInTheDocument();
    expect(screen.getByText("92%")).toBeInTheDocument(); // valid frames
    expect(screen.getByText("88%")).toBeInTheDocument(); // lower-body visibility
  });

  it("collapses a row again on a second tap", async () => {
    renderStudio();
    const row = screen.getByRole("button", { name: /Detailed metrics/i });
    fireEvent.click(row);
    expect(screen.getByText("Side")).toBeInTheDocument();
    fireEvent.click(row);
    expect(screen.queryByText("Side")).not.toBeInTheDocument();
  });
});

describe("StudioMobile — actions", () => {
  it("starts a fresh session from the primary button", async () => {
    const { onNewSession } = renderStudio();
    fireEvent.click(screen.getByRole("button", { name: /Start \/ upload video/i }));
    expect(onNewSession).toHaveBeenCalledTimes(1);
  });

  it("links the segmented control's other half at the history page", () => {
    renderStudio();
    expect(screen.getByRole("link", { name: /My records/i })).toHaveAttribute("href", "/history");
  });
});

describe("MobileVideoCard — the stage", () => {
  it("renders the clip with its skeleton overlay", () => {
    renderStudio();
    expect(document.querySelector("video")).toBeInTheDocument();
    expect(document.querySelector("canvas")).toBeInTheDocument();
  });

  // The mock's pill says "Live Analysis"; this pipeline is offline, so the pill carries the verdict
  // it actually has.
  it("states the fault verdict in the status pill", () => {
    renderStudio();
    expect(screen.getByText("1 fault detected")).toBeInTheDocument();
  });

  it("pluralises the verdict", () => {
    renderStudio({
      ...mockAnalysis,
      detections: [
        mockAnalysis.detections[0],
        { ...mockAnalysis.detections[0], fault_id: "heel_rise", fault_name: "heel_rise" },
      ],
    });
    expect(screen.getByText("2 faults detected")).toBeInTheDocument();
  });

  it("states a clean verdict when nothing was detected", () => {
    renderStudio(mockCleanAnalysis);
    expect(screen.getByText("No faults detected")).toBeInTheDocument();
  });

  it("swaps play for pause as the element's own events fire", () => {
    renderStudio();
    const video = document.querySelector("video")!;
    fireEvent(video, new Event("play"));
    expect(screen.getByRole("button", { name: /pause/i })).toBeInTheDocument();
    fireEvent(video, new Event("pause"));
    expect(screen.getByRole("button", { name: /play/i })).toBeInTheDocument();
  });

  it("reports the playhead upward as the clip plays", () => {
    const { onTimeUpdate } = renderStudio();
    fireEvent(document.querySelector("video")!, new Event("timeupdate"));
    expect(onTimeUpdate).toHaveBeenCalled();
  });

  it("offers a fullscreen control that does not throw without the API", async () => {
    renderStudio();
    const btn = screen.getByRole("button", { name: /fullscreen/i });
    fireEvent.click(btn);
    expect(btn).toBeInTheDocument();
  });

  it("does not throw when the play control is tapped", async () => {
    renderStudio();
    fireEvent.click(screen.getByRole("button", { name: /play/i }));
    expect(document.querySelector("video")).toBeInTheDocument();
  });

  // The floating metrics card is the mock's Reps / Tempo / Knee Angle panel. Tempo has no
  // counterpart in this pipeline; the surviving rows only render when the data behind them exists,
  // so a clip with neither drops the card rather than showing empty rows.
  it("drops the floating metrics card when there is neither a rep count nor evidence", () => {
    renderStudio(mockCleanAnalysis);
    expect(screen.queryByText("valgus angle")).not.toBeInTheDocument();
    expect(screen.queryByText("Reps")).not.toBeInTheDocument();
  });

  it("shows the detector's own primary evidence in the floating card", () => {
    renderStudio();
    expect(screen.getByText("valgus angle")).toBeInTheDocument();
    expect(screen.getByText("0.35")).toBeInTheDocument();
  });

  // On a phone the markers are the only way to reach a fault in the clip — there is no separate
  // timeline strip under the stage.
  it("marks each fault on the scrub track and seeks to it", async () => {
    const { onSeek } = renderStudio();
    const marker = document.querySelector("[title='Knee Valgus']") as HTMLElement;
    expect(marker).toBeInTheDocument();
    fireEvent.click(marker);
    expect(onSeek).toHaveBeenCalledWith(1.0);
  });

  // The phone draws its own scrub bar instead of mounting Timeline, so RS-SP2's two honesty
  // surfaces had to be ported here rather than inherited. Without them a verdict computed on 3 of
  // 5 reps reads, on a phone, as a verdict on all 5.
  it("hatches the reps it never analyzed and says which ones it did", () => {
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
    renderStudio(analysis as Analysis);
    expect(screen.getAllByTestId("unanalyzed-span")).toHaveLength(1);
    expect(screen.getByText(/3 reps found, analyzed #1, 3/)).toBeInTheDocument();
  });

  it("says the whole clip was analyzed on a segmentation fallback, and nothing at all without reps", () => {
    const fallback = {
      ...mockAnalysis,
      reps: {
        detected: 1, analyzed: [], max_reps: 3, fallback: "only_partial_reps",
        segments: [
          { index: 1, start_frame: 0, end_frame: 89, start_time: 0, end_time: 3, analyzed: true, partial: true },
        ],
      },
    };
    renderStudio(fallback as Analysis);
    expect(screen.queryAllByTestId("unanalyzed-span")).toHaveLength(0);
    expect(screen.getByText("Whole clip analyzed")).toBeInTheDocument();
  });

  it("shows no rep summary for an analysis that carries no `reps` field", () => {
    renderStudio();
    expect(screen.queryAllByTestId("unanalyzed-span")).toHaveLength(0);
    expect(screen.queryByText(/reps found/)).not.toBeInTheDocument();
    expect(screen.queryByText("Whole clip analyzed")).not.toBeInTheDocument();
  });

  it("seeks when the track itself is clicked", () => {
    const { onSeek } = renderStudio();
    const track = document.querySelector("[title='Knee Valgus']")!.parentElement!;
    // jsdom reports a zero-width rect, so the computed target is not meaningful — what is being
    // pinned is that scrubbing is wired at all, and that the marker's own click does not also
    // bubble into it (asserted by the single call in the test above).
    fireEvent.click(track);
    expect(onSeek).toHaveBeenCalled();
  });

  it("pins the fault callouts onto the body", () => {
    renderStudio({
      ...mockAnalysis,
      // The detector emits the bare rule name as `fault_id`; the shared fixture's suffixed id
      // misses FAULT_LANDMARKS, and an unanchored fault is deliberately skipped.
      detections: [{ ...mockAnalysis.detections[0], fault_id: "knees_inward", start_frame: 0 }],
      pose: {
        ...mockAnalysis.pose,
        frames: [{ i: 0, lm: Array.from({ length: 33 }, () => [0.5, 0.5, 0.9] as [number, number, number]) }],
      },
    });
    // Re-keying the detection also detaches it from the fixture's retrieval, so the coach card's
    // headline falls back to the fault's own name — hence two "Knee Valgus" on screen. The callout
    // is the positioned one: a chip carries the anchor this test is about.
    const chip = screen.getAllByText("Knee Valgus").find((el) => el.style.left !== "");
    expect(chip).toBeDefined();
    expect(chip!.style.top).not.toBe("");
  });
});
