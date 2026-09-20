import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan } from "../api";
import App from "../App";
import { mockAnalysis } from "./fixtures";

// Same stubs as App.planItem.test.tsx: the client-capture path needs a <video>/WASM pipeline
// jsdom cannot run, and this suite is about the WP4 rehab disclaimer, not pose extraction.
vi.mock("../lib/poseExtract", () => ({
  extractPoseWithReps: vi.fn().mockResolvedValue({
    pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] },
    reps: { max_reps: 3, fallback: null, segments: [] },
  }),
}));
vi.mock("../lib/thumbnail", () => ({ captureThumbnail: () => Promise.resolve(null) }));

function plan(assignedBy: string | null): Plan {
  return {
    id: "p1",
    name: "Upper week",
    notes: null,
    template_key: null,
    started_at: "2026-08-13T00:00:00Z",
    created_at: "2026-08-13T00:00:00Z",
    updated_at: "2026-08-13T00:00:00Z",
    assigned_by: assignedBy,
    items: [
      {
        id: "i1",
        plan_id: "p1",
        day_index: 3,
        position: 0,
        movement: "Squat",
        sets: 3,
        reps: 10,
        notes: null,
        completed_at: null,
        analysis_id: null,
        created_at: "2026-08-13T00:00:00Z",
      },
    ],
  };
}

const PLAN_ROUTE = "/app?movement=Squat&plan=p1&plan_item=i1";
const DISCLAIMER = /This analysis is exercise feedback, not a medical diagnosis/i;

async function findFileInput(): Promise<HTMLInputElement> {
  await vi.waitFor(() => expect(document.querySelector('input[type="file"]')).not.toBeNull());
  return document.querySelector('input[type="file"]') as HTMLInputElement;
}

beforeEach(() => {
  vi.spyOn(api, "getMovements").mockResolvedValue([{ name: "Squat", validated: true }]);
  vi.spyOn(api, "updatePlanItem").mockResolvedValue(plan("clinician-1").items[0]);
  vi.spyOn(api, "analyzePose").mockResolvedValue({
    ...mockAnalysis,
    video_id: "v1",
    movement: "Squat",
    analysis_id: "an-7",
  });
});
afterEach(() => vi.restoreAllMocks());

describe("studio rehab disclaimer (WP4)", () => {
  it("appears at the result for a THERAPIST-ASSIGNED plan", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(plan("clinician-1"));
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

    expect(await screen.findByText(DISCLAIMER)).toBeInTheDocument();
  });

  describe("on the phone", () => {
    // Driven by `useIsMobile`, so this forces its media query to match — the global setup stub
    // (src/test/setup.ts) answers `false` to everything else, which is why every other test in
    // this file exercises the DESKTOP tree. Same idiom as pages.PlanBuilder.test.tsx.
    beforeEach(() => {
      vi.spyOn(window, "matchMedia").mockImplementation(
        (query: string) =>
          ({
            matches: query === "(max-width: 1023px)",
            media: query,
            onchange: null,
            addListener: vi.fn(),
            removeListener: vi.fn(),
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            dispatchEvent: vi.fn(),
          }) as unknown as MediaQueryList
      );
    });

    it("appears below the phone result (StudioMobile) for a THERAPIST-ASSIGNED plan", async () => {
      vi.spyOn(api, "getPlan").mockResolvedValue(plan("clinician-1"));
      renderWithProviders(<App />, { route: PLAN_ROUTE });
      await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

      // The phone layout's own headline stat, so this really is StudioMobile (not the desktop tree).
      expect(await screen.findByText("Form score")).toBeInTheDocument();
      expect(screen.getByText(DISCLAIMER)).toBeInTheDocument();
    });

    it("is absent on the phone for a SELF-MADE plan", async () => {
      vi.spyOn(api, "getPlan").mockResolvedValue(plan(null));
      renderWithProviders(<App />, { route: PLAN_ROUTE });
      await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

      expect(await screen.findByText("Form score")).toBeInTheDocument();
      expect(screen.queryByText(DISCLAIMER)).not.toBeInTheDocument();
    });
  });

  it("is absent for a SELF-MADE plan (assigned_by null)", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(plan(null));
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

    // Wait for the result to render (the coach heading is always there once analysis lands),
    // then assert the disclaimer specifically never joined it.
    await screen.findByText("Lumen");
    expect(screen.queryByText(DISCLAIMER)).not.toBeInTheDocument();
  });

  it("is absent for an ordinary studio visit with no plan context at all", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Squat" });
    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

    await screen.findByText("Lumen");
    expect(screen.queryByText(DISCLAIMER)).not.toBeInTheDocument();
  });
});
