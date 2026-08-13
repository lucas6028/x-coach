import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan } from "../api";
import App from "../App";
import { mockAnalysis } from "./fixtures";

// Same two stubs as App.movement.test.tsx, for the same reason: the client-capture path needs a
// <video>/WASM pipeline jsdom cannot run, and these tests are about the PLAN write-back, not about
// pose extraction or thumbnail capture.
vi.mock("../lib/poseExtract", () => ({
  extractPoseFromBlob: vi.fn().mockResolvedValue({
    metadata: { fps: 30, width: 1, height: 1, total_frames: 0 },
    frames: [],
  }),
}));
vi.mock("../lib/thumbnail", () => ({ captureThumbnail: () => Promise.resolve(null) }));

const PLAN: Plan = {
  id: "p1",
  name: "Upper week",
  notes: null,
  template_key: null,
  started_at: "2026-08-13T00:00:00Z",
  created_at: "2026-08-13T00:00:00Z",
  updated_at: "2026-08-13T00:00:00Z",
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

const PLAN_ROUTE = "/app?movement=Squat&plan=p1&plan_item=i1";

async function findFileInput(): Promise<HTMLInputElement> {
  await vi.waitFor(() => expect(document.querySelector('input[type="file"]')).not.toBeNull());
  return document.querySelector('input[type="file"]') as HTMLInputElement;
}

beforeEach(() => {
  vi.spyOn(api, "getMovements").mockResolvedValue([{ name: "Squat", validated: true }]);
  vi.spyOn(api, "getPlan").mockResolvedValue(PLAN);
});
afterEach(() => vi.restoreAllMocks());

describe("studio entered from a plan item", () => {
  it("names the plan and the day it came from", async () => {
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    expect(await screen.findByText(/From Upper week · Day 3/)).toBeInTheDocument();
  });

  it("offers a way back to the plan", async () => {
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    expect(await screen.findByRole("link", { name: /back to plan/i })).toHaveAttribute(
      "href",
      "/plans/p1"
    );
  });

  it("shows no banner for an ordinary studio visit", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Squat" });
    await findFileInput();
    expect(screen.queryByText(/From /)).not.toBeInTheDocument();
    expect(api.getPlan).not.toHaveBeenCalled();
  });

  it("still renders the studio when the plan fetch fails", async () => {
    // The banner is context. A signed-out or failed fetch must not stop the studio from analysing
    // the clip the user came here to analyse.
    vi.mocked(api.getPlan).mockRejectedValue(new Error("401"));
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    expect(await findFileInput()).toBeInTheDocument();
  });

  it("ticks the item off and links the analysis once the upload persists", async () => {
    const patch = vi.spyOn(api, "updatePlanItem").mockResolvedValue(PLAN.items[0]);
    vi.spyOn(api, "analyzePose").mockResolvedValue({
      ...mockAnalysis,
      video_id: "v1",
      movement: "Squat",
      analysis_id: "an-7",
    });
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

    await vi.waitFor(() =>
      expect(patch).toHaveBeenCalledWith("p1", "i1", { completed: true, analysis_id: "an-7" })
    );
  });

  it("does nothing for an anonymous upload, which has no analysis to link", async () => {
    // `analysis_id` is present only for a signed-in upload. An anonymous visitor who lands here
    // with a ?plan_item= in the URL gets their analysis and no failed write.
    const patch = vi.spyOn(api, "updatePlanItem");
    vi.spyOn(api, "analyzePose").mockResolvedValue({
      ...mockAnalysis,
      video_id: "v1",
      movement: "Squat",
      analysis_id: undefined,
    });
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

    await screen.findByText(/From Upper week/);
    expect(patch).not.toHaveBeenCalled();
  });

  it("keeps the analysis when the plan write-back fails", async () => {
    // A failed tick is worth neither an error over a successful analysis nor losing the result.
    vi.spyOn(api, "updatePlanItem").mockRejectedValue(new Error("offline"));
    vi.spyOn(api, "analyzePose").mockResolvedValue({
      ...mockAnalysis,
      video_id: "v1",
      movement: "Squat",
      analysis_id: "an-7",
    });
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

    await vi.waitFor(() => expect(api.updatePlanItem).toHaveBeenCalled());
    expect(screen.queryByText("offline")).not.toBeInTheDocument();
  });

  it("reports an already-ticked item as linked on arrival", async () => {
    // A re-record or a back-button return must not claim the item is still outstanding.
    vi.mocked(api.getPlan).mockResolvedValue({
      ...PLAN,
      items: [{ ...PLAN.items[0], completed_at: "2026-08-13T01:00:00Z" }],
    });
    renderWithProviders(<App />, { route: PLAN_ROUTE });
    expect(await screen.findByText(/ticked off in your plan/i)).toBeInTheDocument();
  });
});
