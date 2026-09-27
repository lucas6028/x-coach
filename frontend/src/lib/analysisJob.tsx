// A single, app-wide "background analysis" job that survives navigating away from /app.
//
// Before this file, the whole client-capture pipeline (extraction -> thumbnail -> upload -> plan
// tick) lived inside App.tsx's own `runPoseAnalysis`, closed over React state owned by that one
// component instance. The pipeline itself kept running after a navigation away (its promise chain
// has nothing to do with the component tree), but every `setX` call inside it landed on state that
// had already unmounted — the progress bar, the result, even the plan tick's success all vanished
// silently. Lifting the job up to a provider mounted once, above the router's <Routes>, means the
// SAME state survives whichever page is on screen; App.tsx becomes a VIEWER of the job rather than
// its owner.
//
// Deliberately NOT a navigation-aware component: this file must never import `useNavigate` or call
// `setSearchParams`. Rewriting the URL is App's job (it owns the ?analysis= / ?plan= contract with
// the rest of the studio); a background job doing that itself could yank the address bar out from
// under whatever page the user is actually looking at.
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, UploadLimitError, type Analysis } from "../api";
import BackgroundAnalysisPill from "../components/BackgroundAnalysisPill";
import { createProgressTracker, type AnalysisProgress } from "./analysisProgress";
import { useI18n } from "./i18n";
import type { PoseTier } from "./poseTier";
import { DEFAULT_MAX_REPS } from "./repSpans";
import { captureThumbnail } from "./thumbnail";

export type JobStatus = "idle" | "running" | "done" | "error";

/** The plan context a job was started from, captured at `start()` time — NOT re-read from the URL
 *  later, because by the time the job finishes the user may be looking at an entirely different
 *  page (or a different plan item on this same one). */
export interface JobPlanSnapshot {
  planId: string;
  planItemId: string;
  /** The therapist's user id when a linked clinician assigned this plan, else null. Mirrors
   *  `Plan.assigned_by` — see App.tsx's own `assignedByTherapist`. */
  assignedBy: string | null;
}

export interface AnalysisJobState {
  status: JobStatus;
  /** The movement being analyzed by the CURRENT (or just-finished) job — independent of whatever
   *  movement the studio's own selector shows, which may have moved on since this job started. */
  movement: string;
  statusMsg: string;
  /** Null outside a running upload — nothing to measure before extraction starts or after the job
   *  settles. */
  progress: AnalysisProgress | null;
  result: Analysis | null;
  error: string;
  /** Whether the studio (or the pill) has already shown this job's DONE result to the user. Gates
   *  both the pill (hidden once viewed) and App's "adopt this result" effect (adopted at most once,
   *  though an explicit ?analysis= in the URL can still ask for it again). */
  viewed: boolean;
  plan: JobPlanSnapshot | null;
  planLinked: boolean;
  /** Set on success only when `plan.assignedBy` was truthy — a self-made plan never nudges for a
   *  check-in. Consumed (cleared) once App has opened the dialog for it. */
  checkinPending: boolean;
}

/** The URL context a studio page (App.tsx) or the background pill is currently looking at — enough
 *  to decide whether a given job BELONGS there. Field names deliberately echo the query params
 *  they come from (`?analysis=`, `?plan_item=`, `?movement=`). */
export interface StudioContext {
  storedId: string | null;
  planItemId: string | null;
  requestedMovement: string | null;
}

/** Whether `job` is the one this studio context should show inline (the running loader, an error,
 *  the job's own movement) or adopt on completion — as opposed to a DIFFERENT job that happens to
 *  be running/done/errored in the background while this page shows something else entirely.
 *
 *  Without this, a job started from one plan item (or movement) would surface on ANY /app visit
 *  with no ?analysis= of its own — swapping in the wrong result, rewriting the URL back to the
 *  plan item it belongs to, and popping its check-in dialog while the user is looking at a
 *  different exercise. See the regression this closes: a job for plan item A finishing while the
 *  user has since opened plan item B's studio must never touch B's page.
 *
 *  Pure and exported so both App.tsx (loader/adopt) and BackgroundAnalysisPill (its own
 *  hide-on-/app check) apply exactly the same rule rather than two hand-rolled ones drifting apart. */
export function jobMatchesStudio(job: AnalysisJobState, ctx: StudioContext): boolean {
  // An explicit ?analysis=<id> is a deliberate ask for THAT analysis — it matches a job iff the
  // job is the one that produced it. A running job has no result yet, so it never matches here
  // (which is correct: an explicit history/clinician link always wins over a job in flight).
  if (ctx.storedId) return ctx.storedId === (job.result?.analysis_id ?? null);

  // Otherwise, match by WHERE the job was started: the same plan item (or no plan item on both
  // sides), and — when the URL asks for a specific movement — the same movement. An idle job (no
  // plan, no movement) trivially "matches" a plan-less, movement-less studio, which is harmless:
  // callers only consult this for a running/done/error job, never an idle one.
  const planMatches = (job.plan?.planItemId ?? null) === (ctx.planItemId ?? null);
  if (!planMatches) return false;

  const requested = ctx.requestedMovement?.trim();
  if (!requested) return true;
  return requested.toLowerCase() === job.movement.toLowerCase();
}

/** The /app URL a studio visit needs to be at for `jobMatchesStudio` to recognize it as `job`'s own
 *  page — i.e. what every BackgroundAnalysisPill link should point at, whatever the job's status.
 *  Built from the SAME fields `jobMatchesStudio` reads, rather than each link hand-rolling its own
 *  query string, which is what closes this bug: the pill's running/error links (and an anonymous
 *  done result's link) used to fall back to a bare "/app" — for a plan-linked job that has no
 *  `?plan_item=` on it, so `jobMatchesStudio` read it as "not this job" and the studio ignored it,
 *  leaving the pill with no way to ever reach (or clear) that job from its own link. */
export function jobStudioHref(job: AnalysisJobState): string {
  const params = new URLSearchParams();
  const analysisId = job.status === "done" ? job.result?.analysis_id ?? null : null;
  if (analysisId) {
    // The durable identifier: once present, `jobMatchesStudio`'s ?analysis= branch doesn't even
    // consult movement, so there's nothing left to add for it to match.
    params.set("analysis", analysisId);
  } else if (job.movement) {
    // No analysis id yet (running/error, or an anonymous done result) — carry the movement so the
    // studio's own selector names the right thing on arrival, not whatever it last showed.
    params.set("movement", job.movement);
  }
  if (job.plan) {
    params.set("plan", job.plan.planId);
    params.set("plan_item", job.plan.planItemId);
  }
  const qs = params.toString();
  return qs ? `/app?${qs}` : "/app";
}

export interface StartJobArgs {
  blob: Blob;
  tier: PoseTier;
  movement: string;
  plan: JobPlanSnapshot | null;
}

export interface AnalysisJobActions {
  /** Starts a job. Returns false (and does nothing else) if one is already running — there is no
   *  cancel, so the only way to free up a slot is to let the running job finish. */
  start: (args: StartJobArgs) => boolean;
  /** Marks a DONE job's result as shown. A no-op on any other status (there is nothing to mark). */
  markViewed: () => void;
  /** Resets to idle. Refuses while a job is RUNNING — clearing state out from under an in-flight
   *  promise would just orphan its eventual `setState` calls, which is exactly the bug this file
   *  exists to fix. */
  clear: () => void;
  /** Clears `checkinPending` once the caller has opened (or decided not to open) the dialog for it. */
  consumeCheckin: () => void;
}

const IDLE_STATE: AnalysisJobState = {
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

// Inert defaults: any component rendered OUTSIDE the provider (a test that mounts a page directly,
// a future route that forgets to wrap it) reads an idle job and gets no-op actions rather than a
// crash. This is a real path — several existing App tests render <App/> with their own minimal
// provider stack and must keep working.
const noopActions: AnalysisJobActions = {
  start: () => false,
  markViewed: () => {},
  clear: () => {},
  consumeCheckin: () => {},
};

// Split in two so a component that only dispatches actions (there is none today, but the pill is
// one candidate) never re-renders on the ~5 Hz progress ticks a running job emits. `useAnalysisJob`
// below recombines them for the common case (a consumer that needs both).
const AnalysisJobStateContext = createContext<AnalysisJobState>(IDLE_STATE);
const AnalysisJobActionsContext = createContext<AnalysisJobActions>(noopActions);

export function useAnalysisJobState(): AnalysisJobState {
  return useContext(AnalysisJobStateContext);
}

export function useAnalysisJobActions(): AnalysisJobActions {
  return useContext(AnalysisJobActionsContext);
}

/** The common case: current state plus the actions to drive it. */
export function useAnalysisJob(): AnalysisJobState & AnalysisJobActions {
  const state = useAnalysisJobState();
  const actions = useAnalysisJobActions();
  return useMemo(() => ({ ...state, ...actions }), [state, actions]);
}

export function AnalysisJobProvider({ children }: { children: ReactNode }) {
  const { t } = useI18n();
  // `start` must stay a STABLE callback (see the actions useMemo below) so it never becomes a new
  // function on every render — but it also needs the CURRENT `t` when a job settles, possibly many
  // renders and one language switch after it began. A ref sidesteps the tradeoff: read the latest
  // translator without `start` depending on it.
  const tRef = useRef(t);
  tRef.current = t;

  const [state, setState] = useState<AnalysisJobState>(IDLE_STATE);
  // Guards "one job at a time" synchronously — `state.status` is only updated by `setState`, which
  // is not guaranteed to have landed yet the instant a second `start()` call race-checks it.
  const runningRef = useRef(false);

  // The server's 413 detail is English and structured; the message the user reads is neither. Moved
  // here verbatim from App.tsx's old `errorMessage`.
  const errorMessage = useCallback((e: unknown): string => {
    const tt = tRef.current;
    if (e instanceof UploadLimitError) {
      return e.code === "upload_too_large"
        ? tt("upload.tooLarge", { limit: e.limitMb })
        : tt("upload.quotaFull", { used: e.usedMb ?? 0, limit: e.limitMb });
    }
    return e instanceof Error ? e.message : String(e);
  }, []);

  const start = useCallback(
    (args: StartJobArgs): boolean => {
      if (runningRef.current) return false;
      runningRef.current = true;

      setState({
        status: "running",
        movement: args.movement,
        statusMsg: tRef.current("app.analysing"),
        progress: { phase: "extract", fraction: 0, remainingSec: null },
        result: null,
        error: "",
        viewed: false,
        plan: args.plan,
        planLinked: false,
        checkinPending: false,
      });

      const tracker = createProgressTracker((p) => setState((s) => ({ ...s, progress: p })));

      void (async () => {
        try {
          // MediaPipe is a cold path: defer its WASM graph until the user explicitly supplies video.
          const { extractPoseWithReps } = await import("./poseExtract");
          const { pose, reps } = await extractPoseWithReps(
            args.blob,
            args.tier,
            args.movement,
            DEFAULT_MAX_REPS,
            tracker.extract
          );
          // Captured from the same blob just decoded for MediaPipe. Resolves to null on any
          // failure — a missing thumbnail never blocks analysis.
          const thumbnail = await captureThumbnail(args.blob);
          const data = await api.analyzePose(
            args.movement,
            pose,
            args.blob,
            thumbnail,
            reps,
            tracker.upload
          );

          // Tick the plan item off using the snapshot taken at `start()`, never the live URL — by
          // now the user may be on a different page, or back on /app against a different item.
          // Guarded on `analysis_id`, which ONLY a signed-in upload has.
          let planLinked = false;
          let checkinPending = false;
          if (data.analysis_id && args.plan) {
            try {
              await api.updatePlanItem(args.plan.planId, args.plan.planItemId, {
                completed: true,
                analysis_id: data.analysis_id,
              });
              planLinked = true;
              // Only a THERAPIST-ASSIGNED plan gets the automatic check-in prompt.
              checkinPending = !!args.plan.assignedBy;
            } catch {
              // The analysis is saved either way — a failed tick costs neither an error banner nor
              // the result the user just waited for.
            }
          }

          setState((s) => ({
            ...s,
            status: "done",
            result: data,
            planLinked,
            checkinPending,
            progress: null,
            statusMsg: "",
          }));
        } catch (e) {
          setState((s) => ({
            ...s,
            status: "error",
            error: errorMessage(e),
            progress: null,
            statusMsg: "",
          }));
        } finally {
          tracker.stop();
          // `args.blob` (and everything else this closure captured) becomes unreachable once this
          // async function returns — nothing keeps a reference to it beyond this IIFE, so the raw
          // clip is free to be garbage-collected from here rather than pinned for the rest of the
          // job's (possibly long) "done, unviewed" lifetime.
          runningRef.current = false;
        }
      })();

      return true;
    },
    [errorMessage]
  );

  const markViewed = useCallback(() => {
    setState((s) => (s.status === "done" && !s.viewed ? { ...s, viewed: true } : s));
  }, []);

  const clear = useCallback(() => {
    setState((s) => (s.status === "running" ? s : IDLE_STATE));
  }, []);

  const consumeCheckin = useCallback(() => {
    setState((s) => (s.checkinPending ? { ...s, checkinPending: false } : s));
  }, []);

  // Memoized on its own, separately from `state`: the actions never change identity across the
  // whole life of the provider, so a consumer that reads only `useAnalysisJobActions()` never
  // re-renders on the progress ticks a running job emits several times a second.
  const actions = useMemo<AnalysisJobActions>(
    () => ({ start, markViewed, clear, consumeCheckin }),
    [start, markViewed, clear, consumeCheckin]
  );

  // Warn before a refresh/close would silently discard an in-flight analysis. This is the one way
  // to lose a running job that navigating the SPA no longer causes — leaving the page (or the tab)
  // entirely still does, and the browser gives no other hook for it.
  useEffect(() => {
    if (state.status !== "running") return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [state.status]);

  return (
    <AnalysisJobStateContext.Provider value={state}>
      <AnalysisJobActionsContext.Provider value={actions}>
        {children}
        {/* Rendered here, inside the provider, rather than by App or AppLayout: it needs to be
            visible from EVERY page (including ones with no AppLayout at all, like the landing page
            and the games), and it needs `useLocation` to hide itself on /app — both of which this
            spot gets for free once the provider sits above <AppRoutes/> but still inside the
            Router. */}
        <BackgroundAnalysisPill job={state} onDismiss={clear} />
      </AnalysisJobActionsContext.Provider>
    </AnalysisJobStateContext.Provider>
  );
}
