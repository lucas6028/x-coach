import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ArrowRight, ListChecks, X } from "@phosphor-icons/react";
import AppLayout from "../components/AppLayout";
import PlanCoach from "../components/plans/PlanCoach";
import PlanPreview from "../components/plans/PlanPreview";
import { LumenAvatar } from "../components/LumenLoader";
import type { Plan } from "../api";
import { useI18n } from "../lib/i18n";
import { useIsMobile, useMediaQuery } from "../lib/useIsMobile";

/**
 * "和 Lumen 一起規劃" — build a training plan by talking to the coach.
 *
 * Two panes that watch the same thing from different angles: the conversation on the left, and the
 * plan it is writing on the right, filled in live from the `tool_done.plan` frames the chat
 * streams. Nothing here fetches: the page opens on an empty plan by definition, so there is no
 * loading state and no skeleton — the placeholder IS the empty state.
 *
 * The plan's id arrives mid-conversation (Lumen creates it), which is the only reason this page
 * holds state at all: once it exists, the footer can offer the way out to the real plan page.
 */
export default function PlanBuilder() {
  const { t } = useI18n();
  const isMobile = useIsMobile();
  // The chat column is a FIXED 440px, so every pixel the window loses comes out of the preview
  // beside it: at a 1024px window that column is 232px, and the muscle card's fixed-width legend
  // then rendered OUTSIDE the card's own border. Same container-not-viewport problem the plan
  // page has, and the same answer -- the preview is told it is narrow and drops to its compact
  // shape. 1360px is where the column clears ~520px, enough for the bodies and the legend.
  const narrowPreview = useMediaQuery("(max-width: 1359px)");

  const [plan, setPlan] = useState<Plan | null>(null);
  // Handed back down to PlanCoach so a reload-free follow-up turn is scoped to the created plan.
  // PlanCoach keys its restore on the id it mounted with, so this never disturbs the live thread.
  const [planId, setPlanId] = useState<string | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  // Escape closes the mobile preview sheet, as any dialog should; the listener only exists while
  // the sheet is open so the page carries nothing on desktop or while the sheet is closed.
  useEffect(() => {
    if (!sheetOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSheetOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sheetOpen]);

  const suggestions = [
    t("plans.builder.chipFullBody"),
    t("plans.builder.chipUpper"),
    t("plans.builder.chipRehab"),
  ];

  const coach = (
    <PlanCoach
      planId={planId}
      onPlan={setPlan}
      onCreated={setPlanId}
      suggestions={suggestions}
      greeting={t("plans.builder.greeting")}
      className="h-full"
    />
  );

  // The empty preview is composed, not a blank box: Lumen says what she needs, in the place the
  // answer will appear.
  const emptyPreview = (
    // Tall enough to hold the column beside the chat rather than leave a void under it: a 150px
    // placeholder next to a full-height conversation reads as a page that failed to load.
    <div className="flex min-h-[26rem] flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-border-dark bg-surface/60 px-6 py-14 text-center">
      <LumenAvatar size={44} />
      <p className="max-w-xs text-sm leading-relaxed text-muted">{t("plans.builder.empty")}</p>
    </div>
  );

  const preview = (
    <div>
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-faint">
        {t("plans.builder.previewTitle")}
      </h2>
      {plan ? (
        <>
          <p className="mb-3 font-display text-lg font-bold text-content">{plan.name}</p>
          <PlanPreview plan={plan} compact={narrowPreview} />
        </>
      ) : (
        emptyPreview
      )}
    </div>
  );

  // Once the plan exists there is somewhere to go, and it is the same destination on both layouts.
  const openCta = planId && (
    <Link
      to={`/plans/${encodeURIComponent(planId)}`}
      className="inline-flex items-center gap-2 rounded-full bg-primary px-5 py-2.5 text-[13px] font-semibold text-primary-content shadow-accent transition-colors hover:bg-primary/90 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
    >
      {t("plans.builder.openPlan")}
      <ArrowRight size={14} weight="bold" />
    </Link>
  );

  const header = (
    <>
      <Link
        to="/plans"
        className="inline-flex items-center gap-1.5 text-[13px] font-medium text-muted transition-colors hover:text-content active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
      >
        <ArrowLeft size={14} weight="bold" />
        {t("plans.back")}
      </Link>
      <h1 className="mt-4 font-display text-2xl font-bold text-content">
        {t("plans.builder.title")}
      </h1>
      <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">
        {t("plans.builder.subtitle")}
      </p>
    </>
  );

  // MOBILE: the conversation is the page. The plan lives one tap away in a sheet, because a phone
  // cannot show both and the chat is where the work happens.
  if (isMobile) {
    return (
      <AppLayout>
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="shrink-0 px-4 pb-3 pt-6">{header}</div>
          <div className="min-h-0 flex-1 px-4 pb-3">{coach}</div>

          <div className="flex shrink-0 items-center justify-between gap-3 border-t border-border-dark bg-surface px-4 py-3">
            <button
              type="button"
              onClick={() => setSheetOpen(true)}
              className="inline-flex items-center gap-2 rounded-full border border-border-dark px-4 py-2 text-[13px] font-medium text-content transition-colors hover:border-primary/40 hover:text-primary active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
            >
              <ListChecks size={15} weight="duotone" />
              {t("plans.builder.showPreview")}
            </button>
            {openCta}
          </div>

          {sheetOpen && (
            <div
              role="dialog"
              aria-modal="true"
              aria-label={t("plans.builder.previewTitle")}
              className="fixed inset-0 z-40 flex flex-col justify-end bg-content/30"
            >
              <button
                type="button"
                aria-label={t("plans.builder.hidePreview")}
                onClick={() => setSheetOpen(false)}
                className="flex-1 focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-4 focus-visible:outline-primary"
              />
              <div className="max-h-[80vh] overflow-y-auto rounded-t-2xl border-t border-border-dark bg-surface px-4 pb-6 pt-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <span className="text-xs font-semibold uppercase tracking-wider text-faint">
                    {t("plans.builder.previewTitle")}
                  </span>
                  <button
                    type="button"
                    onClick={() => setSheetOpen(false)}
                    aria-label={t("plans.builder.hidePreview")}
                    className="rounded-full p-2.5 text-faint transition-colors hover:bg-content/[0.06] hover:text-content active:scale-[0.96] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                  >
                    <X size={16} weight="bold" />
                  </button>
                </div>
                {plan ? <PlanPreview plan={plan} compact /> : emptyPreview}
              </div>
            </div>
          )}
        </div>
      </AppLayout>
    );
  }

  // DESKTOP: chat left at a fixed comfortable reading width, plan right in the remaining space.
  // The chat column is sticky and full-height with its own scroll, so a long plan scrolls past a
  // conversation that stays put.
  return (
    <AppLayout>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-12">
          {header}

          <div className="mt-6 grid items-start gap-6 lg:grid-cols-[440px_minmax(0,1fr)]">
            <div className="h-[600px] min-h-0 lg:sticky lg:top-6 lg:h-[min(calc(100vh-14rem),46rem)]">
              {coach}
            </div>
            <div>
              {preview}
              {/* Sticky: a seven-day preview is taller than the viewport, and the only way out of
                  the builder must not scroll off the bottom of it. */}
              {openCta && (
                <div className="sticky bottom-4 z-10 mt-6 flex justify-end">{openCta}</div>
              )}
            </div>
          </div>
        </main>
      </div>
    </AppLayout>
  );
}
