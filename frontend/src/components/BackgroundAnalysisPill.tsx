// The one place a background analysis stays visible once its own page (/app) has been navigated
// away from. Fixed to the viewport, independent of whatever chrome is under it — the landing page
// and the games render with no AppLayout at all, so this cannot live inside that layout (see the
// routing note on AnalysisJobProvider, which mounts this directly above <AppRoutes/>).
//
// Takes the job state and a `clear` action as PROPS rather than reading them via `useAnalysisJob`
// itself: that hook lives in ../lib/analysisJob.tsx, which is also this component's own importer
// (it renders <BackgroundAnalysisPill/> directly). Reading the context back here would make the
// two files import each other at runtime; threading props one level down does not.
import { Link, useLocation } from "react-router-dom";
import { CheckCircle, WarningCircle, X } from "@phosphor-icons/react";
import { movementLabel, useI18n } from "../lib/i18n";
import { jobMatchesStudio, jobStudioHref, type AnalysisJobState } from "../lib/analysisJob";

interface Props {
  job: AnalysisJobState;
  onDismiss: () => void;
}

export default function BackgroundAnalysisPill({ job, onDismiss }: Props) {
  const { t } = useI18n();
  const { pathname, search } = useLocation();

  if (job.status === "idle") return null;
  if (job.status === "done" && job.viewed) return null;

  // /app usually already shows this job inline (DemoIntro's own loader while it runs; the actual
  // result or error once adopted) — a floating copy on top of that would duplicate it, not rescue
  // it. But that is only true when THIS job is the one /app's current URL is actually about; a job
  // for a different plan item or movement is invisible on /app (App.tsx never shows it there), so
  // hiding the pill too would strand the user with no way to know it is still running, or to reach
  // its result. Same rule App.tsx uses to decide what to show inline — see jobMatchesStudio.
  if (pathname === "/app") {
    const params = new URLSearchParams(search);
    const matchesThisPage = jobMatchesStudio(job, {
      storedId: params.get("analysis"),
      planItemId: params.get("plan_item"),
      requestedMovement: params.get("movement"),
    });
    if (matchesThisPage) return null;
  }

  // Every link below points here — the one URL that makes `jobMatchesStudio` recognize THIS job as
  // this studio visit's own, whatever its status. See jobStudioHref's own comment for the bug this
  // closes (a bare "/app" fallback used to strand a plan-linked running/error job).
  const studioHref = jobStudioHref(job);

  return (
    <div
      role="status"
      aria-live="polite"
      // Below `lg` (phone web + the LINE in-app shell, both `useIsMobile`'s breakpoint): pinned
      // below the TOP bar, not the bottom — the mobile tab bar (and the LINE shell's own bottom
      // nav) live there, and this pill has no way to know which chrome, if any, is under it on
      // whatever page it happens to be floating over. The offset is `env(safe-area-inset-top)` plus
      // MobileTopBar's own real height (`py-3` + its `h-10` round buttons = 0.75rem*2 + 2.5rem =
      // 4rem) — both AppLayout's phone branch and LiffAppShell render that exact header, so a
      // guessed constant would only be one refactor away from covering its back/account buttons
      // again. At `lg` and up the desktop shell has no bottom chrome and top-center would sit over
      // the page's own header controls, so the pill moves to a fixed bottom-right corner instead.
      className="fixed inset-x-0 top-[calc(env(safe-area-inset-top)+4rem)] z-[70] mx-auto w-[calc(100%-2rem)] max-w-sm lg:inset-x-auto lg:left-auto lg:right-6 lg:top-auto lg:bottom-6 lg:mx-0 lg:w-80"
    >
      <div className="glass-panel xc-pop flex items-start gap-3 rounded-2xl border border-white/70 p-3 shadow-[0_16px_40px_rgba(105,112,175,0.25)]">
        {job.status === "running" && (
          <>
            <span
              aria-hidden="true"
              className="mt-1.5 h-2.5 w-2.5 shrink-0 animate-pulse rounded-full bg-primary"
            />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold text-[#1e2142]">
                {t("job.running", { movement: movementLabel(t, job.movement) })}
              </p>
              <p className="mt-0.5 truncate text-xs text-[#59648f]">
                {job.progress
                  ? t(`app.progress.${job.progress.phase}`, {
                      pct: Math.round(job.progress.fraction * 100),
                    })
                  : job.statusMsg}
              </p>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/60">
                <div
                  className="h-full rounded-full bg-primary transition-[width]"
                  style={{ width: `${Math.round((job.progress?.fraction ?? 0) * 100)}%` }}
                />
              </div>
            </div>
          </>
        )}

        {job.status === "done" && (
          <>
            <CheckCircle size={20} weight="fill" className="mt-0.5 shrink-0 text-secondary" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-[#1e2142]">{t("job.ready")}</p>
              <Link
                to={studioHref}
                className="mt-1 inline-block text-xs font-semibold text-primary underline-offset-2 hover:underline"
              >
                {t("job.view")}
              </Link>
            </div>
            <button
              type="button"
              onClick={onDismiss}
              aria-label={t("job.dismiss")}
              className="shrink-0 rounded-lg p-1 text-[#8b8fa8] hover:bg-white/60 hover:text-[#1e2142]"
            >
              <X size={16} />
            </button>
          </>
        )}

        {job.status === "error" && (
          <>
            <WarningCircle size={20} weight="fill" className="mt-0.5 shrink-0 text-danger" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-[#1e2142]">{t("job.failed")}</p>
              <p className="mt-0.5 line-clamp-2 text-xs text-[#59648f]">{job.error}</p>
              <Link
                to={studioHref}
                className="mt-1 inline-block text-xs font-semibold text-primary underline-offset-2 hover:underline"
              >
                {t("job.backToStudio")}
              </Link>
            </div>
            <button
              type="button"
              onClick={onDismiss}
              aria-label={t("job.dismiss")}
              className="shrink-0 rounded-lg p-1 text-[#8b8fa8] hover:bg-white/60 hover:text-[#1e2142]"
            >
              <X size={16} />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
