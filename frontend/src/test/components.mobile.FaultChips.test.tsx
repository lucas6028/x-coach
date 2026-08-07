import { describe, it, expect } from "vitest";
import type React from "react";
import { screen } from "@testing-library/react";
import type { Analysis, Detection, PoseFrame } from "../api";
import FaultChips from "../components/mobile/FaultChips";
import { renderWithProviders } from "./renderWithProviders";
import { mockAnalysis, mockDetection } from "./fixtures";

// `fault_id` — not `fault_name` — is what FAULT_LANDMARKS is keyed on (SkeletonOverlay reads the
// same map the same way), and the detector emits the bare rule name there: `fault_id="knees_inward"`
// (src/pose/pose_rule_detector.py). The shared fixture's suffixed "knees_inward_1" would silently
// miss the map and skip every chip, so these build their own detections.
const detection = (over: Partial<Detection> = {}): Detection => ({
  ...mockDetection,
  fault_id: "knees_inward",
  fault_name: "knees_inward",
  start_frame: 0,
  end_frame: 60,
  ...over,
});

// A landmark at the centre of the frame, comfortably visible.
const CENTRE: [number, number, number] = [0.5, 0.5, 0.9];

const frames = (lm: [number, number, number][] | null = null): PoseFrame[] =>
  Array.from({ length: 100 }, (_, i) => ({
    i,
    lm: (lm ?? Array.from({ length: 33 }, () => CENTRE)) as [number, number, number][],
  }));

function build(over: Partial<Analysis> = {}): Analysis {
  return {
    ...mockAnalysis,
    detections: [detection()],
    // 1280x720 metadata: the aspect the chip positions below are computed against.
    pose: { ...mockAnalysis.pose, frames: frames() },
    ...over,
  };
}

// `videoRef.current` is null here on purpose: jsdom reports videoWidth/videoHeight as 0 even on a
// mounted <video>, so the component takes its metadata fallback either way. Null makes that
// explicit rather than pretending a real element is being measured.
const nullRef = { current: null } as React.RefObject<HTMLVideoElement>;

const renderChips = (analysis: Analysis, time = 0.5) =>
  renderWithProviders(<FaultChips analysis={analysis} videoRef={nullRef} time={time} />);

describe("FaultChips", () => {
  it("labels the fault the playhead is inside", () => {
    renderChips(build());
    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();
  });

  // The whole point of the extraction: the chip is placed on the video's RENDERED pixels, not on
  // the letterboxed box around them. A 16:9 clip in the card's box is letterboxed, so a landmark at
  // the vertical centre of the CLIP must not land at the vertical centre of the BOX.
  //
  // containRect(100, 100, 1280/720) → height 56.25, offsetY 21.875. A centroid at (0.5, 0.5) maps to
  // left 50 / top 50; the chip then applies the mock's own anchor nudge (-34 left, -4 up for the
  // first chip) and clamps into the card.
  it("maps the anchor through the letterbox rather than the raw box", () => {
    renderChips(build());
    const chip = screen.getByText("Knee Valgus");
    expect(chip.style.left).toBe("16%");
    expect(chip.style.top).toBe("46%");
  });

  it("renders nothing once the playhead leaves the fault's span", () => {
    // fps 30, so t=3s is frame 90 — past end_frame 60.
    renderChips(build(), 3);
    expect(screen.queryByText("Knee Valgus")).not.toBeInTheDocument();
  });

  it("renders nothing when the clip carries no pose frames", () => {
    const { container } = renderChips(build({ pose: { ...mockAnalysis.pose, frames: [] } }));
    expect(container).toBeEmptyDOMElement();
  });

  // `lm` is nullable on the wire: a frame the detector could not resolve carries no landmarks at
  // all, and the playhead can sit on one.
  it("renders nothing when the frame at the playhead has no landmarks", () => {
    const bare: PoseFrame[] = [{ i: 0, lm: null }];
    const { container } = renderChips(build({ pose: { ...mockAnalysis.pose, frames: bare } }));
    expect(container).toBeEmptyDOMElement();
  });

  // A label floating over nothing is worse than no label: if the detector could not see the joints
  // the fault names, the chip is dropped rather than pinned to a guess.
  it("skips a fault whose landmarks are all below the visibility threshold", () => {
    const dim = Array.from({ length: 33 }, () => [0.5, 0.5, 0.1] as [number, number, number]);
    const { container } = renderChips(
      build({ pose: { ...mockAnalysis.pose, frames: frames(dim) } })
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("anchors to the visible landmarks only, ignoring the occluded ones", () => {
    // Every knee-group landmark low in frame and visible EXCEPT one planted high and invisible —
    // if the invisible one were averaged in it would drag the anchor upward.
    const lm = Array.from({ length: 33 }, () => [0.5, 0.8, 0.9] as [number, number, number]);
    lm[23] = [0.5, 0.0, 0.05];
    renderChips(build({ pose: { ...mockAnalysis.pose, frames: frames(lm) } }));
    // cy 0.8 → top = 21.875 + 0.8*56.25 - 4 = 62.875. Averaging the occluded hip in would give
    // cy 0.667 and a visibly higher chip.
    expect(screen.getByText("Knee Valgus").style.top).toBe("62.875%");
  });

  it("skips a fault with no landmark group to anchor to", () => {
    const { container } = renderChips(
      build({ detections: [detection({ fault_id: "butt_wink", fault_name: "butt_wink" })] })
    );
    expect(container).toBeEmptyDOMElement();
  });

  // The clip is the subject; a third label starts covering the athlete the labels exist to explain.
  it("shows at most two chips even when three faults are active at once", () => {
    renderChips(
      build({
        detections: [
          detection(),
          detection({ fault_id: "excessive_forward_lean", fault_name: "excessive_forward_lean" }),
          detection({ fault_id: "heel_rise", fault_name: "heel_rise" }),
        ],
      })
    );
    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();
    expect(screen.getByText("Excessive Forward Lean")).toBeInTheDocument();
    expect(screen.queryByText("Heel Rise")).not.toBeInTheDocument();
  });

  it("alternates the horizontal anchor so two chips on the same joints do not stack", () => {
    renderChips(
      build({
        detections: [
          detection(),
          detection({ fault_id: "knees_forward", fault_name: "knees_forward" }),
        ],
      })
    );
    expect(screen.getByText("Knee Valgus").style.left).not.toBe(
      screen.getByText("Knees Forward").style.left
    );
  });

  // Decorative: every fault named here is also listed as real, focusable text in the studio's
  // issues row, so announcing the overlay too would read the same list twice.
  it("hides the overlay from assistive tech", () => {
    const { container } = renderChips(build());
    expect(container.querySelector("[aria-hidden='true']")).toBeInTheDocument();
  });
});
