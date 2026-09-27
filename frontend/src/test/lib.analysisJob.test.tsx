import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { Link, MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider } from "../lib/auth";
import { I18nProvider } from "../lib/i18n";
import {
  AnalysisJobProvider,
  jobMatchesStudio,
  jobStudioHref,
  useAnalysisJob,
  type AnalysisJobState,
  type JobPlanSnapshot,
} from "../lib/analysisJob";
import BackgroundAnalysisPill from "../components/BackgroundAnalysisPill";
import { api, type Plan } from "../api";
import App from "../App";
import { extractPoseWithReps } from "../lib/poseExtract";
import { mockAnalysis } from "./fixtures";

// Same two stubs every other App upload test uses: the client-capture path needs a <video>/WASM
// pipeline jsdom cannot run. `extractPoseWithReps` is a bare `vi.fn()` here (rather than resolving
// immediately) because the key test below needs to hold it PENDING while the user navigates away.
vi.mock("../lib/poseExtract", () => ({ extractPoseWithReps: vi.fn() }));
vi.mock("../lib/thumbnail", () => ({ captureThumbnail: () => Promise.resolve(null) }));

/** A promise plus its own resolve/reject, so a test can control exactly when a mocked async step
 *  settles — the mechanism the key test below uses to hold "extraction" and "upload" open across
 *  a navigation. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function PLAN(): Plan {
  return {
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
}

function PLAN_TWO_ITEMS(): Plan {
  return {
    id: "p1",
    name: "Upper week",
    notes: null,
    template_key: null,
    started_at: "2026-08-13T00:00:00Z",
    created_at: "2026-08-13T00:00:00Z",
    updated_at: "2026-08-13T00:00:00Z",
    // Truthy so a completed item's tick sets `checkinPending` — the reviewer's repro needs that to
    // prove a check-in dialog for item A never pops open while the user is on item B's studio.
    assigned_by: "clinician-1",
    items: [
      {
        id: "i1", plan_id: "p1", day_index: 3, position: 0, movement: "Squat",
        sets: 3, reps: 10, notes: null, completed_at: null, analysis_id: null,
        created_at: "2026-08-13T00:00:00Z",
      },
      {
        id: "i2", plan_id: "p1", day_index: 4, position: 0, movement: "Squat",
        sets: 3, reps: 10, notes: null, completed_at: null, analysis_id: null,
        created_at: "2026-08-13T00:00:00Z",
      },
    ],
  };
}

/** Stands in for /plans/:planId — the point of this suite is that navigating there and back must
 *  not disturb the analysis running in the background, not that the real plan page renders. */
function PlanStub() {
  return <p>Plan detail stub</p>;
}

/** Rendered outside <Routes> (a sibling of it, inside the same Router) so a test can read the
 *  CURRENT location without that read itself being torn down by whichever route just unmounted. */
function LocationProbe() {
  const loc = useLocation();
  return (
    <>
      <div data-testid="loc-pathname">{loc.pathname}</div>
      <div data-testid="loc-search">{loc.search}</div>
    </>
  );
}

/** Stands in for the app's persistent sidebar: a handful of /app-to-/app links a user can click
 *  WITHOUT the analysis job that's running (for a different plan item / movement) ever offering a
 *  way there itself — exactly the "opened a different studio some other way" repro. Because these
 *  all resolve to the same "/app" route, React Router re-renders App in place rather than
 *  remounting it, matching how the real sidebar/plan-banner links behave (see App.tsx's own
 *  comment on its "next plan item" link). */
function StudioSwitcherLinks() {
  return (
    <nav>
      <Link to="/app?movement=Squat&plan=p1&plan_item=i1">Item A</Link>
      <Link to="/app?movement=Squat&plan=p1&plan_item=i2">Item B</Link>
      <Link to="/app?movement=Push-up">Push-up studio</Link>
    </nav>
  );
}

function renderAppWithSwitcher(entry: string) {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <AuthProvider>
        <I18nProvider>
          <AnalysisJobProvider>
            <Routes>
              <Route path="/app" element={<App />} />
            </Routes>
            <StudioSwitcherLinks />
          </AnalysisJobProvider>
          <LocationProbe />
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
}

/** The real provider tree, with a real two-route switch — App at /app (where the job starts) and
 *  the plan stub at /plans/:planId (where the key test navigates the user to). Mirrors main.tsx's
 *  nesting order (AnalysisJobProvider inside AuthProvider/I18nProvider, wrapping the routes). */
function renderAppAndPlan(entry: string) {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <AuthProvider>
        <I18nProvider>
          <AnalysisJobProvider>
            <Routes>
              <Route path="/app" element={<App />} />
              <Route path="/plans/:planId" element={<PlanStub />} />
            </Routes>
          </AnalysisJobProvider>
          <LocationProbe />
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
}

async function findFileInput(): Promise<HTMLInputElement> {
  await vi.waitFor(() => expect(document.querySelector('input[type="file"]')).not.toBeNull());
  return document.querySelector('input[type="file"]') as HTMLInputElement;
}

afterEach(() => {
  vi.restoreAllMocks();
  // `extractPoseWithReps` is a bare `vi.fn()` from the `vi.mock` factory above, not a spy on a
  // real implementation — `restoreAllMocks` only rewinds spies, so its call log and any
  // `mockImplementationOnce` queue would otherwise leak into the next test.
  vi.mocked(extractPoseWithReps).mockReset();
});

describe("background analysis job — survives navigating away from /app", () => {
  it(
    "keeps running, ticks the plan item once, and the result is viewable after navigating away and back",
    async () => {
      const d1 = deferred<{ pose: unknown; reps: unknown }>();
      vi.mocked(extractPoseWithReps).mockImplementationOnce(() => d1.promise as never);
      const d2 = deferred<typeof mockAnalysis>();

      vi.spyOn(api, "getMovements").mockResolvedValue([{ name: "Squat", validated: true }]);
      vi.spyOn(api, "getPlan").mockResolvedValue(PLAN());
      const analyzePose = vi.spyOn(api, "analyzePose").mockImplementation(() => d2.promise);
      const updatePlanItem = vi
        .spyOn(api, "updatePlanItem")
        .mockResolvedValue(PLAN().items[0]);

      renderAppAndPlan("/app?movement=Squat&plan=p1&plan_item=i1");

      // Start the analysis and let it get as far as "extraction in flight".
      await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));
      // The job's own progress caption ("Reading your pose… 0%") — proof the job is actually
      // running before we navigate away from it.
      await screen.findByText(/reading your pose/i);

      // Navigate away via the studio's OWN "back to plan" link — the real path a user takes, not a
      // synthetic route push.
      await userEvent.click(await screen.findByRole("link", { name: /back to plan/i }));
      expect(await screen.findByText("Plan detail stub")).toBeInTheDocument();
      expect(screen.getByTestId("loc-pathname").textContent).toBe("/plans/p1");

      // Let extraction, then the upload, resolve now that the studio (App) is unmounted.
      d1.resolve({ pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] }, reps: { max_reps: 3, fallback: null, segments: [] } });
      d2.resolve({ ...mockAnalysis, video_id: "v1", movement: "Squat", analysis_id: "an-7" });

      // (c) The plan item is ticked exactly once, even though the tab that started the tick is gone.
      await vi.waitFor(() =>
        expect(updatePlanItem).toHaveBeenCalledWith("p1", "i1", { completed: true, analysis_id: "an-7" })
      );
      expect(updatePlanItem).toHaveBeenCalledTimes(1);
      expect(analyzePose).toHaveBeenCalledTimes(1);

      // (b) The pill surfaces the finished job — visible here because we're NOT on /app.
      const pill = await screen.findByRole("status");
      const viewLink = await within(pill).findByRole("link", { name: /view result/i });

      // (a) The app never navigated itself back to /app — the user is still exactly where they
      // clicked to. This is the whole point: the old behaviour unmounted the analysis WITH the page;
      // the fixed behaviour must not force a navigation the other way either.
      expect(screen.getByTestId("loc-pathname").textContent).toBe("/plans/p1");
      expect(screen.getByText("Plan detail stub")).toBeInTheDocument();

      // (d) Following the pill's link back to /app renders the result — from memory, no re-fetch.
      const getStoredAnalysis = vi.spyOn(api, "getStoredAnalysis");
      await userEvent.click(viewLink);
      // Renders twice by design (the video card's floating list AND the coach panel's fault card
      // both cite it) — see the same pattern in App.test.tsx.
      expect((await screen.findAllByText(/valgus angle 0\.35/))[0]).toBeInTheDocument();
      expect(getStoredAnalysis).not.toHaveBeenCalled();
      expect(screen.getByTestId("loc-pathname").textContent).toBe("/app");
    },
    15000
  );
});

describe("background analysis job — does not leak into a DIFFERENT studio context", () => {
  it(
    "stays invisible on plan item B's studio while a job for plan item A runs, and does not hijack it on completion",
    async () => {
      const d1 = deferred<{ pose: unknown; reps: unknown }>();
      vi.mocked(extractPoseWithReps).mockImplementationOnce(() => d1.promise as never);

      vi.spyOn(api, "getMovements").mockResolvedValue([{ name: "Squat", validated: true }]);
      vi.spyOn(api, "getPlan").mockResolvedValue(PLAN_TWO_ITEMS());
      vi.spyOn(api, "analyzePose").mockResolvedValue({
        ...mockAnalysis,
        video_id: "vA",
        movement: "Squat",
        analysis_id: "an-A",
      });
      const updatePlanItem = vi
        .spyOn(api, "updatePlanItem")
        .mockResolvedValue(PLAN_TWO_ITEMS().items[0]);

      renderAppWithSwitcher("/app?movement=Squat&plan=p1&plan_item=i1");

      // Start a job for plan item A and let it get as far as "extraction in flight".
      await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));
      await screen.findByText(/reading your pose/i);

      // The reviewer's repro: open plan item B's studio some OTHER way (not a link the running job
      // itself offers) — a fresh /app?plan_item=i2 visit with no ?analysis= of its own.
      await userEvent.click(screen.getByRole("link", { name: "Item B" }));
      await screen.findByText(/From Upper week · Day 4/);
      expect(screen.getByTestId("loc-search").textContent).toBe(
        "?movement=Squat&plan=p1&plan_item=i2"
      );

      // Let job A finish while the user is on item B's studio.
      d1.resolve({
        pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] },
        reps: { max_reps: 3, fallback: null, segments: [] },
      });
      await vi.waitFor(() =>
        expect(updatePlanItem).toHaveBeenCalledWith("p1", "i1", { completed: true, analysis_id: "an-A" })
      );

      // Item B's URL is untouched — job A must never rewrite it back to plan_item=i1.
      expect(screen.getByTestId("loc-pathname").textContent).toBe("/app");
      expect(screen.getByTestId("loc-search").textContent).toBe(
        "?movement=Squat&plan=p1&plan_item=i2"
      );
      // Job A's result is not swapped in under item B...
      expect(screen.queryByText(/valgus angle 0\.35/)).not.toBeInTheDocument();
      // ...and its check-in prompt (assigned_by is truthy on this plan) never pops open here.
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
      expect(screen.queryByText(/ticked off in your plan/i)).not.toBeInTheDocument();

      // The pill still surfaces job A, since /app's CURRENT url (item B) does not match it.
      const pill = await screen.findByRole("status");
      const viewLink = within(pill).getByRole("link", { name: /view result/i });
      expect(viewLink).toHaveAttribute("href", "/app?analysis=an-A&plan=p1&plan_item=i1");

      // Following it actually reaches job A's result.
      await userEvent.click(viewLink);
      expect((await screen.findAllByText(/valgus angle 0\.35/))[0]).toBeInTheDocument();
    },
    15000
  );
});

describe("background analysis job — movement mismatch on /app", () => {
  it("shows the studio's own view (not the job's loader) on a different movement, and refuses a second upload", async () => {
    const d1 = deferred<{ pose: unknown; reps: unknown }>();
    vi.mocked(extractPoseWithReps).mockImplementationOnce(() => d1.promise as never);
    vi.spyOn(api, "getMovements").mockResolvedValue([
      { name: "Squat", validated: true },
      { name: "Push-up", validated: true },
    ]);
    vi.spyOn(api, "analyzePose").mockResolvedValue({ ...mockAnalysis, analysis_id: undefined });

    renderAppWithSwitcher("/app?movement=Squat");

    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));
    await screen.findByText(/reading your pose/i);

    await userEvent.click(screen.getByRole("link", { name: "Push-up studio" }));

    // The Push-up studio shows ITS OWN normal dropzone, not the Squat job's loader — proven by the
    // dropzone's file input actually being there (the loader state renders no such input at all).
    await screen.findByRole("heading", { level: 2, name: /push-up/i });
    const input = await findFileInput();

    // The job is still surfaced — just via the pill (which legitimately shows the SAME "Reading
    // your pose…" caption as the studio's own loader would have — that overlap is expected, this
    // is asserting WHERE it renders, not that the text is unique to one or the other).
    const pill = await screen.findByRole("status");
    expect(pill).toHaveTextContent(/reading your pose/i);

    // Supplying a clip here hits the global "one job at a time" limit (there is only ever one job,
    // matching or not) — this must say so, not silently swallow the upload.
    await userEvent.upload(input, new File(["y"], "clip2.mp4", { type: "video/mp4" }));
    expect(await screen.findByText(/another analysis is still running/i)).toBeInTheDocument();
    expect(extractPoseWithReps).toHaveBeenCalledTimes(1);

    d1.resolve({
      pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] },
      reps: { max_reps: 1, fallback: null, segments: [] },
    });
  });

  it("clears a stale job.busy message once a later successful start begins", async () => {
    const d1 = deferred<{ pose: unknown; reps: unknown }>();
    vi.mocked(extractPoseWithReps).mockImplementationOnce(() => d1.promise as never);
    vi.spyOn(api, "getMovements").mockResolvedValue([
      { name: "Squat", validated: true },
      { name: "Push-up", validated: true },
    ]);
    const analyzePose = vi
      .spyOn(api, "analyzePose")
      .mockResolvedValue({ ...mockAnalysis, analysis_id: undefined });

    renderAppWithSwitcher("/app?movement=Squat");

    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));
    await screen.findByText(/reading your pose/i);

    await userEvent.click(screen.getByRole("link", { name: "Push-up studio" }));
    await userEvent.upload(await findFileInput(), new File(["y"], "clip2.mp4", { type: "video/mp4" }));
    await screen.findByText(/another analysis is still running/i);

    // Free the one global job slot.
    d1.resolve({
      pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] },
      reps: { max_reps: 1, fallback: null, segments: [] },
    });
    await vi.waitFor(() => expect(analyzePose).toHaveBeenCalledTimes(1));

    // The old `runPoseAnalysis` cleared `error` unconditionally at the top of every attempt;
    // `onBlob` must do the same on a SUCCESSFUL start, or this stale "another analysis is still
    // running" text would sit on screen forever, masking whatever the new job goes on to show.
    vi.mocked(extractPoseWithReps).mockResolvedValueOnce({
      pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] },
      reps: { max_reps: 1, fallback: null, segments: [] },
    });
    await userEvent.upload(await findFileInput(), new File(["z"], "clip3.mp4", { type: "video/mp4" }));
    await vi.waitFor(() =>
      expect(screen.queryByText(/another analysis is still running/i)).not.toBeInTheDocument()
    );
  });
});

describe("background analysis job — error link reaches the studio for a plan-snapshotted job", () => {
  it("clicking the pill's 'back to studio' link from another route shows the job's own error on /app", async () => {
    const d1 = deferred<{ pose: unknown; reps: unknown }>();
    vi.mocked(extractPoseWithReps).mockImplementationOnce(() => d1.promise as never);
    vi.spyOn(api, "getMovements").mockResolvedValue([{ name: "Squat", validated: true }]);
    vi.spyOn(api, "getPlan").mockResolvedValue(PLAN());

    renderAppAndPlan("/app?movement=Squat&plan=p1&plan_item=i1");

    await userEvent.upload(await findFileInput(), new File(["x"], "clip.mp4", { type: "video/mp4" }));
    await screen.findByText(/reading your pose/i);

    await userEvent.click(await screen.findByRole("link", { name: /back to plan/i }));
    expect(await screen.findByText("Plan detail stub")).toBeInTheDocument();

    // The job fails while the user is on an entirely different route.
    d1.reject(new Error("could not decode the clip"));

    const pill = await screen.findByRole("status");
    const backLink = within(pill).getByRole("link", { name: /back to studio/i });
    // Carries the job's own plan snapshot — a bare "/app" here is exactly the bug this closes: it
    // would read (via jobMatchesStudio) as "not this job" and the studio would never show the error.
    expect(backLink).toHaveAttribute("href", "/app?movement=Squat&plan=p1&plan_item=i1");

    await userEvent.click(backLink);
    expect(await screen.findByText("could not decode the clip")).toBeInTheDocument();
  });
});

describe("jobStudioHref — always round-trips into a StudioContext jobMatchesStudio accepts", () => {
  const PLAN_SNAPSHOT: JobPlanSnapshot = { planId: "p1", planItemId: "i1", assignedBy: null };

  function baseJob(overrides: Partial<AnalysisJobState>): AnalysisJobState {
    return {
      status: "running",
      movement: "Squat",
      statusMsg: "",
      progress: null,
      result: null,
      error: "",
      viewed: false,
      plan: null,
      planLinked: false,
      checkinPending: false,
      ...overrides,
    };
  }

  const cases: Record<string, AnalysisJobState> = {
    "running, no plan": baseJob({ status: "running", plan: null }),
    "running, with plan": baseJob({ status: "running", plan: PLAN_SNAPSHOT }),
    "error, no plan": baseJob({ status: "error", error: "boom", plan: null }),
    "error, with plan": baseJob({ status: "error", error: "boom", plan: PLAN_SNAPSHOT }),
    "done, no plan, anonymous (no analysis_id)": baseJob({
      status: "done",
      plan: null,
      result: { ...mockAnalysis, analysis_id: undefined },
    }),
    "done, with plan, anonymous (no analysis_id)": baseJob({
      status: "done",
      plan: PLAN_SNAPSHOT,
      result: { ...mockAnalysis, analysis_id: undefined },
    }),
    "done, no plan, with analysis_id": baseJob({
      status: "done",
      plan: null,
      result: { ...mockAnalysis, analysis_id: "an-1" },
    }),
    "done, with plan, with analysis_id": baseJob({
      status: "done",
      plan: PLAN_SNAPSHOT,
      result: { ...mockAnalysis, analysis_id: "an-1" },
    }),
  };

  for (const [name, job] of Object.entries(cases)) {
    it(`matches for: ${name}`, () => {
      const href = jobStudioHref(job);
      const [, qs = ""] = href.split("?");
      const params = new URLSearchParams(qs);
      const ctx = {
        storedId: params.get("analysis"),
        planItemId: params.get("plan_item"),
        requestedMovement: params.get("movement"),
      };
      expect(jobMatchesStudio(job, ctx)).toBe(true);
    });
  }
});

describe("jobMatchesStudio", () => {
  const PLAN_JOB: AnalysisJobState = {
    status: "running",
    movement: "Squat",
    statusMsg: "",
    progress: null,
    result: null,
    error: "",
    viewed: false,
    plan: { planId: "p1", planItemId: "i1", assignedBy: null },
    planLinked: false,
    checkinPending: false,
  };

  it("an explicit ?analysis= matches only the job that produced it", () => {
    const done: AnalysisJobState = {
      ...PLAN_JOB,
      status: "done",
      result: { ...mockAnalysis, analysis_id: "an-1" },
    };
    expect(jobMatchesStudio(done, { storedId: "an-1", planItemId: null, requestedMovement: null })).toBe(
      true
    );
    expect(
      jobMatchesStudio(done, { storedId: "an-2", planItemId: "i1", requestedMovement: "Squat" })
    ).toBe(false);
  });

  it("an explicit ?analysis= never matches a job with no result yet", () => {
    expect(
      jobMatchesStudio(PLAN_JOB, { storedId: "an-1", planItemId: "i1", requestedMovement: "Squat" })
    ).toBe(false);
  });

  it("without ?analysis=, matches by plan item — movement match is case-insensitive, and optional", () => {
    expect(
      jobMatchesStudio(PLAN_JOB, { storedId: null, planItemId: "i1", requestedMovement: "Squat" })
    ).toBe(true);
    expect(
      jobMatchesStudio(PLAN_JOB, { storedId: null, planItemId: "i1", requestedMovement: "squat" })
    ).toBe(true);
    expect(
      jobMatchesStudio(PLAN_JOB, { storedId: null, planItemId: "i1", requestedMovement: null })
    ).toBe(true);
  });

  it("without ?analysis=, a different plan item never matches", () => {
    expect(
      jobMatchesStudio(PLAN_JOB, { storedId: null, planItemId: "i2", requestedMovement: "Squat" })
    ).toBe(false);
  });

  it("without ?analysis=, a different requested movement never matches, even on the same plan item", () => {
    expect(
      jobMatchesStudio(PLAN_JOB, { storedId: null, planItemId: "i1", requestedMovement: "Push-up" })
    ).toBe(false);
  });

  it("a plan-less job matches a plan-less studio by movement alone, never a plan item's studio", () => {
    const anon: AnalysisJobState = { ...PLAN_JOB, plan: null };
    expect(
      jobMatchesStudio(anon, { storedId: null, planItemId: null, requestedMovement: "Squat" })
    ).toBe(true);
    expect(
      jobMatchesStudio(anon, { storedId: null, planItemId: null, requestedMovement: "Push-up" })
    ).toBe(false);
    expect(
      jobMatchesStudio(anon, { storedId: null, planItemId: "i1", requestedMovement: "Squat" })
    ).toBe(false);
  });
});

describe("AnalysisJobProvider — one job at a time", () => {
  function JobProbe() {
    const job = useAnalysisJob();
    const [lastStart, setLastStart] = useState<string>("");
    return (
      <div>
        <div data-testid="status">{job.status}</div>
        <div data-testid="last-start">{lastStart}</div>
        <button
          type="button"
          onClick={() =>
            setLastStart(
              String(
                job.start({
                  blob: new Blob(["x"]),
                  tier: "full",
                  movement: "Squat",
                  plan: null,
                })
              )
            )
          }
        >
          start
        </button>
      </div>
    );
  }

  function renderProbe() {
    return render(
      <MemoryRouter initialEntries={["/somewhere-else"]}>
        <I18nProvider>
          <AnalysisJobProvider>
            <JobProbe />
          </AnalysisJobProvider>
        </I18nProvider>
      </MemoryRouter>
    );
  }

  it("refuses a second start while one is running", async () => {
    const d1 = deferred<{ pose: unknown; reps: unknown }>();
    vi.mocked(extractPoseWithReps).mockImplementation(() => d1.promise as never);

    renderProbe();
    const start = screen.getByRole("button", { name: "start" });

    await userEvent.click(start);
    await vi.waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("running"));
    expect(screen.getByTestId("last-start")).toHaveTextContent("true");

    await userEvent.click(start);
    expect(screen.getByTestId("last-start")).toHaveTextContent("false");
    expect(extractPoseWithReps).toHaveBeenCalledTimes(1);

    d1.resolve({ pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] }, reps: { max_reps: 1, fallback: null, segments: [] } });
  });
});

describe("BackgroundAnalysisPill", () => {
  const IDLE: AnalysisJobState = {
    status: "idle",
    movement: "",
    statusMsg: "",
    progress: null,
    result: null,
    error: "",
    viewed: false,
    plan: null,
    planLinked: false,
    checkinPending: false,
  };

  function renderPill(job: AnalysisJobState, entry = "/history") {
    return render(
      <MemoryRouter initialEntries={[entry]}>
        <I18nProvider>
          <BackgroundAnalysisPill job={job} onDismiss={vi.fn()} />
        </I18nProvider>
      </MemoryRouter>
    );
  }

  it("shows the error and a way back to the studio", () => {
    renderPill({ ...IDLE, status: "error", error: "That clip did not go through" });
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText("That clip did not go through")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /back to studio/i })).toHaveAttribute("href", "/app");
  });

  it("renders nothing on the /app route, running or not", () => {
    renderPill({ ...IDLE, status: "running", movement: "Squat" }, "/app");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders nothing while idle", () => {
    renderPill(IDLE);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders nothing once a done result has been viewed", () => {
    renderPill({ ...IDLE, status: "done", viewed: true, result: mockAnalysis });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});

describe("AnalysisJobProvider — beforeunload guard", () => {
  function fireBeforeUnload(): boolean {
    const event = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(event);
    return event.defaultPrevented;
  }

  it("blocks unload only while a job is running", async () => {
    const d1 = deferred<{ pose: unknown; reps: unknown }>();
    vi.mocked(extractPoseWithReps).mockImplementationOnce(() => d1.promise as never);
    vi.spyOn(api, "analyzePose").mockResolvedValue({ ...mockAnalysis, analysis_id: undefined });

    function JobProbe() {
      const job = useAnalysisJob();
      return (
        <button
          type="button"
          onClick={() => job.start({ blob: new Blob(["x"]), tier: "full", movement: "Squat", plan: null })}
        >
          start
        </button>
      );
    }

    render(
      <MemoryRouter>
        <I18nProvider>
          <AnalysisJobProvider>
            <JobProbe />
          </AnalysisJobProvider>
        </I18nProvider>
      </MemoryRouter>
    );

    expect(fireBeforeUnload()).toBe(false);

    await userEvent.click(screen.getByRole("button", { name: "start" }));
    expect(fireBeforeUnload()).toBe(true);

    d1.resolve({ pose: { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] }, reps: { max_reps: 1, fallback: null, segments: [] } });
    await vi.waitFor(() => expect(fireBeforeUnload()).toBe(false));
  });
});

describe("useAnalysisJob — outside the provider", () => {
  it("is inert: idle state, and start() is a no-op that returns false", () => {
    function Bare() {
      const job = useAnalysisJob();
      return (
        <div>
          <div data-testid="status">{job.status}</div>
          <button
            type="button"
            onClick={() => {
              const ok = job.start({ blob: new Blob(["x"]), tier: "full", movement: "Squat", plan: null });
              // Nothing to assert on visibly other than that this never throws and never starts a
              // job — extractPoseWithReps below is asserted un-called.
              expect(ok).toBe(false);
            }}
          >
            start
          </button>
        </div>
      );
    }

    render(<Bare />);
    expect(screen.getByTestId("status")).toHaveTextContent("idle");
    expect(() => screen.getByRole("button", { name: "start" }).click()).not.toThrow();
    expect(extractPoseWithReps).not.toHaveBeenCalled();
  });
});
