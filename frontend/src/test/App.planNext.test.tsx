import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api, type Plan, type PlanItem } from "../api";
import App from "../App";
import { mockAnalysis } from "./fixtures";

// Same two stubs as App.planItem.test.tsx, for the same reason: the client-capture path needs a
// <video>/WASM pipeline jsdom cannot run, and these tests are about the plan BANNER, not about
// pose extraction or thumbnail capture.
vi.mock("../lib/poseExtract", () => ({
  extractPoseWithReps: vi.fn().mockResolvedValue({
    pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] },
    reps: { max_reps: 3, fallback: null, segments: [] },
  }),
}));
vi.mock("../lib/thumbnail", () => ({ captureThumbnail: () => Promise.resolve(null) }));

function item(overrides: Partial<PlanItem> & { id: string; day_index: number }): PlanItem {
  return {
    plan_id: "p1",
    position: 0,
    movement: "Squat",
    sets: 3,
    reps: 10,
    notes: null,
    completed_at: null,
    analysis_id: null,
    created_at: "2026-09-01T00:00:00Z",
    ...overrides,
  };
}

function plan(items: PlanItem[]): Plan {
  return {
    id: "p1",
    name: "Upper week",
    notes: null,
    template_key: null,
    started_at: "2026-09-01T00:00:00Z",
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    items,
  };
}

const ROUTE = "/app?movement=Squat&plan=p1&plan_item=i1";

async function findFileInput(): Promise<HTMLInputElement> {
  await vi.waitFor(() => expect(document.querySelector('input[type="file"]')).not.toBeNull());
  return document.querySelector('input[type="file"]') as HTMLInputElement;
}

/** Run one successful, persisted analysis through the studio, which ticks the plan item off. */
async function analyse() {
  vi.spyOn(api, "updatePlanItem").mockResolvedValue(item({ id: "i1", day_index: 3 }));
  vi.spyOn(api, "analyzePose").mockResolvedValue({
    ...mockAnalysis,
    video_id: "v1",
    movement: "Squat",
    analysis_id: "an-7",
  });
  renderWithProviders(<App />, { route: ROUTE });
  await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));
}

beforeEach(() => {
  vi.spyOn(api, "getMovements").mockResolvedValue([
    { name: "Squat", validated: true },
    { name: "Row", validated: false },
  ]);
});
afterEach(() => vi.restoreAllMocks());

describe("studio — what the plan asks for next", () => {
  it("says nothing about the next exercise before this one is done", async () => {
    // Until the current item is ticked, the next exercise is a distraction from the one the user
    // came here to record.
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan([item({ id: "i1", day_index: 3 }), item({ id: "i2", day_index: 4, movement: "Row" })])
    );
    renderWithProviders(<App />, { route: ROUTE });
    await screen.findByText(/From Upper week/);
    expect(screen.queryByRole("link", { name: /next:/i })).not.toBeInTheDocument();
  });

  it("names the next uncompleted exercise and links the studio at it", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan([item({ id: "i1", day_index: 3 }), item({ id: "i2", day_index: 4, movement: "Row" })])
    );
    await analyse();

    const next = await screen.findByRole("link", { name: /next: row · day 4/i });
    expect(next).toHaveAttribute("href", "/app?movement=Row&plan=p1&plan_item=i2");
  });

  it("skips the item just completed, even though the fetched plan still calls it outstanding", async () => {
    // The plan was fetched BEFORE the tick, so the current item is skipped by id, not by its flag.
    // Without that it would offer the exercise the user has this second finished.
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan([item({ id: "i1", day_index: 3 }), item({ id: "i2", day_index: 5, movement: "Row" })])
    );
    await analyse();

    await screen.findByRole("link", { name: /next: row/i });
    expect(screen.queryByRole("link", { name: /next: squat/i })).not.toBeInTheDocument();
  });

  it("orders by day, then by position within the day", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan([
        item({ id: "i1", day_index: 3 }),
        item({ id: "late", day_index: 4, position: 1, movement: "Squat" }),
        item({ id: "early", day_index: 4, position: 0, movement: "Row" }),
      ])
    );
    await analyse();

    expect(await screen.findByRole("link", { name: /next: row · day 4/i })).toHaveAttribute(
      "href",
      "/app?movement=Row&plan=p1&plan_item=early"
    );
  });

  it("prefers an analysable exercise, but offers an unanalysable one rather than nothing", async () => {
    // Jumping Jacks has no detector, so the studio cannot grade it — it is still the next thing
    // the plan asks for, and the plan page can tick it off by hand.
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan([
        item({ id: "i1", day_index: 3 }),
        item({ id: "jj", day_index: 4, movement: "Jumping Jacks" }),
      ])
    );
    await analyse();
    expect(await screen.findByRole("link", { name: /next: jumping jacks · day 4/i })).toBeInTheDocument();
  });

  it("prefers the analysable exercise when the plan offers both", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan([
        item({ id: "i1", day_index: 3 }),
        item({ id: "jj", day_index: 4, position: 0, movement: "Jumping Jacks" }),
        item({ id: "row", day_index: 4, position: 1, movement: "Row" }),
      ])
    );
    await analyse();
    expect(await screen.findByRole("link", { name: /next: row/i })).toBeInTheDocument();
  });

  it("actually moves the studio onto the next exercise when followed", async () => {
    // The only /app -> /app link in the app, so React Router re-renders instead of remounting.
    // The movement and the plan context follow the URL by themselves; the RESULT does not, and a
    // next exercise opened under the previous one's report is the failure this guards.
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan([item({ id: "i1", day_index: 3 }), item({ id: "i2", day_index: 4, movement: "Row" })])
    );
    await analyse();
    await userEvent.click(await screen.findByRole("link", { name: /next: row · day 4/i }));

    expect(await screen.findByText(/From Upper week · Day 4/)).toBeInTheDocument();
    // The previous item's tick does not carry over to the new one...
    expect(screen.queryByText(/ticked off in your plan/i)).not.toBeInTheDocument();
    // ...and the studio is back on the dropzone rather than the finished report.
    expect(await findFileInput()).toBeInTheDocument();
  });

  it("says the plan is complete when nothing is left", async () => {
    vi.spyOn(api, "getPlan").mockResolvedValue(
      plan([
        item({ id: "i1", day_index: 3 }),
        item({ id: "old", day_index: 1, completed_at: "2026-09-02T00:00:00Z" }),
      ])
    );
    await analyse();

    expect(await screen.findByText(/this plan is complete/i)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /next:/i })).not.toBeInTheDocument();
    // The way back to the plan is still there — it is where a finished plan gets restarted.
    expect(screen.getByRole("link", { name: /back to plan/i })).toHaveAttribute("href", "/plans/p1");
  });
});
