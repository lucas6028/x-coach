import { describe, it, expect, vi, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api } from "../api";
import DetectedErrorsCard from "../components/studio/DetectedErrorsCard";
import FormScoreCard from "../components/studio/FormScoreCard";
import KeyMetricsCard from "../components/studio/KeyMetricsCard";
import PreviousSessionsCard from "../components/studio/PreviousSessionsCard";
import TipsCard from "../components/studio/TipsCard";
import { renderWithProviders } from "./renderWithProviders";
import {
  mockAnalysis,
  mockCleanAnalysis,
  mockDetection,
  mockUnmeasuredAnalysis,
} from "./fixtures";

afterEach(() => vi.restoreAllMocks());

describe("DetectedErrorsCard", () => {
  it("renders nothing when there is no fault to list", () => {
    const { container } = renderWithProviders(
      <DetectedErrorsCard detections={[]} onSeek={vi.fn()} activeFaultId={null} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("lists the fault with its measured evidence", () => {
    renderWithProviders(
      <DetectedErrorsCard
        detections={mockAnalysis.detections}
        onSeek={vi.fn()}
        activeFaultId={null}
      />
    );
    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();
    expect(screen.getByText("valgus angle 0.35")).toBeInTheDocument();
  });

  it("seeks to the fault's start when a row is clicked", async () => {
    const onSeek = vi.fn();
    renderWithProviders(
      <DetectedErrorsCard
        detections={mockAnalysis.detections}
        onSeek={onSeek}
        activeFaultId={null}
      />
    );
    await userEvent.click(screen.getByText("Knee Valgus"));
    expect(onSeek).toHaveBeenCalledWith(mockAnalysis.detections[0].start_time);
  });

  // The overlay is small, so it shows two rows and says how many it did not — silently dropping
  // the rest would understate what was detected.
  it("counts the faults it could not fit", () => {
    const many = [0, 1, 2, 3].map((i) => ({ ...mockDetection, fault_id: `f${i}` }));
    renderWithProviders(
      <DetectedErrorsCard detections={many} onSeek={vi.fn()} activeFaultId={null} />
    );
    expect(screen.getByText("+2 more in the coach panel")).toBeInTheDocument();
  });
});

describe("FormScoreCard", () => {
  it("shows the derived score and its band", () => {
    renderWithProviders(<FormScoreCard analysis={mockAnalysis} />);
    expect(screen.getByText("80%")).toBeInTheDocument();
    expect(screen.getByText("Good")).toBeInTheDocument();
    // The card must say where the number comes from — it is not a backend measurement.
    expect(screen.getByText("From detected faults")).toBeInTheDocument();
  });

  it("shows a dash instead of a score when nothing was measurable", () => {
    renderWithProviders(<FormScoreCard analysis={mockUnmeasuredAnalysis} />);
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("Nothing measurable in this clip")).toBeInTheDocument();
    expect(screen.queryByText("Excellent")).toBeNull();
  });
});

describe("KeyMetricsCard", () => {
  it("leads with the detector's own primary evidence", () => {
    renderWithProviders(<KeyMetricsCard analysis={mockAnalysis} />);
    expect(screen.getByText("valgus angle")).toBeInTheDocument();
    expect(screen.getByText("0.35")).toBeInTheDocument();
  });

  it("fills the remaining cells with clip quality, always three wide", () => {
    const { container } = renderWithProviders(<KeyMetricsCard analysis={mockAnalysis} />);
    expect(container.querySelectorAll(".grid > div")).toHaveLength(3);
    expect(screen.getByText("92%")).toBeInTheDocument(); // valid frames
    expect(screen.getByText("88%")).toBeInTheDocument(); // lower-body visibility
  });

  it("falls back entirely to quality cells for a clean rep", () => {
    renderWithProviders(<KeyMetricsCard analysis={mockCleanAnalysis} />);
    expect(screen.queryByText("valgus angle")).toBeNull();
    expect(screen.getByText("Side")).toBeInTheDocument(); // camera view
  });

  // Ported from the deleted MetricsCards suite: the RS-SP2 denominator moved into this card (and
  // into lib/quality.ts's `validFrameStat`) when the muse-spark redesign removed that component.
  it("counts valid frames against the frames that were EXTRACTED, not the whole clip", () => {
    // Under RS-SP2 only the scored reps carry landmarks, so a whole-clip denominator would show
    // "30%" for a deliberately partial extraction and read as bad tracking.
    const analysis = {
      ...mockAnalysis,
      quality: {
        ...mockAnalysis.quality,
        total_frames: 900,
        valid_frames: 260,
        extracted_frames: 270,
        extracted_frame_ratio: 0.3,
        valid_frame_ratio: 0.289,
      },
    };
    const { container } = renderWithProviders(<KeyMetricsCard analysis={analysis} />);
    expect(screen.getByText("96%")).toBeInTheDocument();
    expect(screen.getByText("260 / 270 extracted")).toBeInTheDocument();
    // The BAR has to agree with the number: a 96% figure over a 29% bar is a card contradicting
    // itself, and only the styling carries that — hence the width assertion.
    const bars = container.querySelectorAll(".grid > div .h-full");
    expect((bars[1] as HTMLElement).style.width).toBe("96%");
  });

  it("falls back to the whole-clip denominator for analyses with no extracted_frames", () => {
    renderWithProviders(<KeyMetricsCard analysis={mockAnalysis} />);
    // mockAnalysis has no extracted_frames — old stored analyses and CLI output look like this.
    expect(
      screen.getByText(`${Math.round((mockAnalysis.quality.valid_frame_ratio ?? 0) * 100)}%`)
    ).toBeInTheDocument();
    expect(screen.getByText("276/300 frames")).toBeInTheDocument();
  });
});

describe("TipsCard", () => {
  it("lists the corrective cue the knowledge graph returned for the fault", () => {
    renderWithProviders(<TipsCard analysis={mockAnalysis} />);
    expect(screen.getByText("Drive knees out")).toBeInTheDocument();
    expect(screen.getByText("Knee Valgus")).toBeInTheDocument();
  });

  it("says so rather than inventing advice when retrieval returned no cue", () => {
    renderWithProviders(<TipsCard analysis={{ ...mockAnalysis, retrievals: [] }} />);
    expect(
      screen.getByText("The knowledge graph returned no corrective cue for these faults.")
    ).toBeInTheDocument();
  });

  it("distinguishes a clean rep from a retrieval that came back empty", () => {
    renderWithProviders(<TipsCard analysis={mockCleanAnalysis} />);
    expect(screen.getByText("Nothing to correct on this rep.")).toBeInTheDocument();
  });
});

describe("PreviousSessionsCard", () => {
  // Signed out there is no history to fetch; the card invites sign-in instead of erroring.
  it("invites sign-in when there is no session", () => {
    const list = vi.spyOn(api, "listAnalyses");
    renderWithProviders(<PreviousSessionsCard />);
    expect(screen.getByText("Sign in to keep a history of your sessions.")).toBeInTheDocument();
    expect(list).not.toHaveBeenCalled();
  });
});
