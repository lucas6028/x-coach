import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan } from "../api";
import App from "../App";
import { mockAnalysis } from "./fixtures";

// Same stubs as App.planItem.test.tsx: the client-capture path needs a <video>/WASM pipeline
// jsdom cannot run, and these tests are about the post-tick check-in prompt, not pose extraction.
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

describe("check-in prompt after a plan-item tick", () => {
  it("opens the dialog with the plan, item and analysis ids for an ASSIGNED plan", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(plan("clinician-1"));
    const createCheckin = vi.spyOn(api, "createCheckin").mockResolvedValue({
      id: "c1",
      user_id: "u1",
      plan_id: "p1",
      plan_item_id: "i1",
      analysis_id: "an-7",
      pain_nrs: 3,
      rpe: null,
      note: null,
      form_score: null,
      flagged: false,
      flag_reasons: [],
      acknowledged_at: null,
      created_at: "2026-09-15T00:00:00Z",
    });

    renderWithProviders(<App />, { route: PLAN_ROUTE });
    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

    const dialog = await screen.findByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: "3" }));
    await userEvent.click(within(dialog).getByRole("button", { name: /submit/i }));

    await vi.waitFor(() =>
      expect(createCheckin).toHaveBeenCalledWith({
        plan_id: "p1",
        plan_item_id: "i1",
        analysis_id: "an-7",
        pain_nrs: 3,
      })
    );
  });

  it("shows no dialog for a SELF-MADE plan (assigned_by null)", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(plan(null));

    renderWithProviders(<App />, { route: PLAN_ROUTE });
    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));

    await vi.waitFor(() => expect(api.updatePlanItem).toHaveBeenCalled());
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
