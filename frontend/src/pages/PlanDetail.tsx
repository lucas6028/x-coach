import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Moon, Play, Plus, Trash, WarningCircle, X } from "@phosphor-icons/react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import ConfirmDialog from "../components/ConfirmDialog";
import CheckinDialog from "../components/checkin/CheckinDialog";
import AddExerciseForm from "../components/plans/AddExerciseForm";
import PlanCoach from "../components/plans/PlanCoach";
import PlanItemRow from "../components/plans/PlanItemRow";
import { LumenAvatar } from "../components/LumenLoader";
import { api, type Checkin, type NewPlanItem, type Plan, type PlanItem } from "../api";
import { movementLabel, useI18n } from "../lib/i18n";
import type { AnalyzableMovement } from "../lib/movements";
import { useIsMobile, useMediaQuery } from "../lib/useIsMobile";
import PlanMuscleCoverage from "../components/plans/PlanMuscleCoverage";
import WeekStrip from "../components/plans/WeekStrip";
import {
  PLAN_DAYS,
  currentDay,
  isAnalyzable,
  itemsByDay,
  progressRatio,
  usedDays,
} from "../lib/plans";
import { dayMuscles } from "../lib/planMuscles";

type Status = "loading" | "ready" | "error";

/** The `tabpanel` the week strip drives, and how each tile's `tab` is named. Module-level rather
 *  than generated: there is exactly one plan page at a time, and a stable id is what lets the
 *  panel point back at the tab that opened it. */
const DAY_PANEL_ID = "plan-day-panel";
const dayTabId = (day: number) => `plan-day-tab-${day}`;

// One plan, as a week strip plus one focused day. Editing is immediate — every tick, add and remove is its own
// request and the local copy is patched from the response, rather than a save button over a draft:
// a plan is edited while standing in a gym, and a draft that needs saving is a draft that gets lost.
export default function PlanDetail() {
  const { t, lang } = useI18n();
  const { planId = "" } = useParams();
  const isMobile = useIsMobile();
  // WHY A SECOND, WIDER BREAKPOINT THAN `useIsMobile`: the coach column is 400px wide, and the
  // plan beside it needs room for a real exercise row. Below 1280px it does not have it -- at
  // 1024px the plan column lands at ~232px against the ~241px one row needs, which is the exact
  // overflow recorded on the day-band comment below: the rows collapse and the "add exercise"
  // button escapes the band and ends up UNDER the panel, unreachable. So between 1024 and 1279
  // the coach opens as the same bottom sheet the phone uses -- an overlay borrows no width -- and
  // only from `xl` does it become a column of the page.
  const sheetCoach = useMediaQuery("(max-width: 1279px)");
  const [searchParams] = useSearchParams();

  // `?coach=1` opens the panel — read ONCE, into the initial state. Kept as an effect it would
  // re-open the panel every time the user closed it, since the param is still in the URL.
  const [coachOpen, setCoachOpen] = useState(() => searchParams.get("coach") === "1");

  const [plan, setPlan] = useState<Plan | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState("");

  // Which movements the pipeline can actually analyse. Same source and same fallback as the studio
  // and the movement menu: on failure assume Squat only, because the alternative — offering every
  // movement — sends the user to record a clip we would then grade with the wrong rules.
  const [analyzable, setAnalyzable] = useState<AnalyzableMovement[]>([
    { name: "Squat", validated: true },
  ]);

  // Per-row write lock, keyed by item id, so ticking one exercise does not freeze the rest of the
  // plan. `null` for the plan-level actions (start, delete).
  const [busyItem, setBusyItem] = useState<string | null>(null);
  const [addingTo, setAddingTo] = useState<number | null>(null);

  // WHICH DAY THE PANEL SHOWS, as "the one the user picked, or else the derived one" — NOT as a
  // day number seeded by an effect. The distinction matters twice. Before the user has picked
  // anything, `null` lets the selection track the plan: tick off the last exercise of day 1 and
  // the panel moves to day 2, and a week Lumen rewrote opens on ITS current day rather than on
  // whatever the old plan's day happened to be. After they have picked, the number wins and
  // nothing moves under them — an effect that recomputed a stored default would yank the panel
  // away mid-edit every time a tick changed `currentDay`.
  const [pickedDay, setPickedDay] = useState<number | null>(null);
  const [confirming, setConfirming] = useState<"delete" | "restart" | null>(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [itemError, setItemError] = useState("");
  const [deleted, setDeleted] = useState(false);

  // The most recent check-in against this plan, newest-first from the server — only its `flagged`
  // state and reasons matter here, for the red-flag banner. `null` covers both "not loaded yet"
  // and "no check-ins exist", which is fine: either way there is nothing to warn about.
  const [latestCheckin, setLatestCheckin] = useState<Checkin | null>(null);
  const refreshLatestCheckin = useCallback(() => {
    api
      .listCheckins(planId, 1)
      .then((list) => setLatestCheckin(list[0] ?? null))
      .catch(() => undefined);
  }, [planId]);
  useEffect(() => {
    refreshLatestCheckin();
  }, [refreshLatestCheckin]);

  // Which item the check-in dialog is reporting on, or null when it is closed. Carries the item's
  // own analysis id (if it has one) so a flagged response can be scored against the form_score of
  // the session that prompted it.
  const [checkinTarget, setCheckinTarget] = useState<{
    itemId: string;
    analysisId?: string;
    movement: string;
  } | null>(null);

  const load = useCallback(async () => {
    setStatus("loading");
    setError("");
    try {
      setPlan(await api.getPlan(planId));
      setStatus("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, [planId]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    let cancelled = false;
    api
      .getMovements()
      .then((ms) => {
        if (!cancelled && ms.length) setAnalyzable(ms);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  // The sheet is a modal, so Escape dismisses it like every other dialog in the app. The desktop
  // panel is a column of the page, not an overlay, and deliberately ignores Escape.
  useEffect(() => {
    if (!sheetCoach || !coachOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setCoachOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [sheetCoach, coachOpen]);

  // Splice the changed item into the local copy rather than refetching the plan: a refetch after
  // every tick would reorder nothing and cost a round trip, and it would also blank the day columns
  // for as long as it took.
  const patchItem = async (itemId: string, patch: Parameters<typeof api.updatePlanItem>[2]) => {
    setBusyItem(itemId);
    setItemError("");
    try {
      const updated = await api.updatePlanItem(planId, itemId, patch);
      setPlan((prev) =>
        prev
          ? { ...prev, items: prev.items.map((it) => (it.id === itemId ? updated : it)) }
          : prev
      );
    } catch (e) {
      setItemError(e instanceof Error ? e.message : t("plans.itemFailed"));
    } finally {
      setBusyItem(null);
    }
  };

  const removeItem = async (itemId: string) => {
    setBusyItem(itemId);
    setItemError("");
    try {
      await api.deletePlanItem(planId, itemId);
      setPlan((prev) =>
        prev ? { ...prev, items: prev.items.filter((it) => it.id !== itemId) } : prev
      );
    } catch (e) {
      setItemError(e instanceof Error ? e.message : t("plans.itemFailed"));
    } finally {
      setBusyItem(null);
    }
  };

  const addItem = async (item: NewPlanItem) => {
    setPlanBusy(true);
    setItemError("");
    try {
      const created = await api.addPlanItem(planId, item);
      setPlan((prev) => (prev ? { ...prev, items: [...prev.items, created] } : prev));
      setAddingTo(null);
    } catch (e) {
      setItemError(e instanceof Error ? e.message : t("plans.itemFailed"));
    } finally {
      setPlanBusy(false);
    }
  };

  const start = async () => {
    setPlanBusy(true);
    setItemError("");
    try {
      setPlan(await api.startPlan(planId));
      setConfirming(null);
    } catch (e) {
      setItemError(e instanceof Error ? e.message : t("plans.itemFailed"));
    } finally {
      setPlanBusy(false);
    }
  };

  const removePlan = async () => {
    setPlanBusy(true);
    try {
      await api.deletePlan(planId);
      // Rendered as a "gone" state rather than navigating: a redirect from inside a confirm dialog
      // leaves the user unsure whether the delete or a stray click took them back to the list.
      setDeleted(true);
      setConfirming(null);
    } catch (e) {
      setItemError(e instanceof Error ? e.message : t("plans.itemFailed"));
    } finally {
      setPlanBusy(false);
    }
  };

  const backLink = (
    <Link
      to="/plans"
      className="inline-flex items-center gap-1.5 text-[13px] font-medium text-muted transition-colors hover:text-content active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
    >
      <ArrowLeft size={14} weight="bold" />
      {t("plans.back")}
    </Link>
  );

  if (status === "error" || deleted) {
    return (
      <AppLayout>
        <div className="flex-1 min-h-0 overflow-y-auto">
          <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-12">
            {backLink}
            <div className="mt-6 flex flex-col items-center gap-3 rounded-2xl border border-border-dark bg-surface px-6 py-12 text-center">
              <WarningCircle size={22} className="text-faint" weight="duotone" />
              <p className="text-sm text-muted">{t("plans.notFound")}</p>
              {!deleted && error && <p className="text-xs text-faint">{error}</p>}
            </div>
          </main>
        </div>
      </AppLayout>
    );
  }

  if (status === "loading" || !plan) {
    return (
      <AppLayout>
        <div className="flex-1 min-h-0 overflow-y-auto">
          <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-12">
            {backLink}
            {/* Shaped like what arrives — a title block, the week strip, then one day panel — so
                the page does not visibly re-flow the moment the plan lands. */}
            <div className="mt-4 h-8 w-56 animate-pulse rounded-lg bg-content/[0.06]" />
            <div className="mt-6 flex gap-2">
              {PLAN_DAYS.map((day) => (
                <div
                  key={day}
                  className="h-[76px] min-w-[7.5rem] flex-1 basis-0 animate-pulse rounded-2xl border border-border-dark bg-surface"
                />
              ))}
            </div>
            <div className="mt-3 h-[152px] animate-pulse rounded-2xl border border-border-dark bg-surface" />
          </main>
        </div>
      </AppLayout>
    );
  }

  const days = itemsByDay(plan.items);
  const completed = plan.items.filter((it) => it.completed_at).length;
  const today = currentDay(plan.items);
  const ratio = progressRatio(completed, plan.items.length);

  // The default the brief asks for, in order: the day the user is on, else the first day that
  // holds anything, else day 1. `currentDay` is null on a FINISHED plan, which is why the second
  // fallback exists — otherwise completing a week would drop the panel onto an empty day 1.
  const selectedDay = pickedDay ?? today ?? usedDays(plan.items)[0] ?? 1;
  const selectedItems = days[selectedDay - 1];

  // One component in two frames: a column beside the plan on a wide desktop, a sheet over it
  // anywhere narrower. Exactly ONE of the two call sites below renders at a time -- two mounted
  // copies would be two live conversations. `onPlan` replaces the page's plan in place — a tool that rewrote the week has already
  // handed back the whole fresh plan, so a refetch would only blank the bands to learn what we
  // were just told.
  // The sheet IS the card, so the panel inside it drops its own shell — a bordered, rounded,
  // shadowed card sitting inside a bordered, rounded sheet is one frame too many.
  const renderCoach = (className: string) => (
    <PlanCoach
      planId={plan.id}
      onPlan={setPlan}
      suggestions={[
        t("plans.coach.chipFourDays"),
        t("plans.coach.chipFewerSets"),
        t("plans.coach.chipSwap"),
        t("plans.coach.chipCore"),
      ]}
      className={className}
    />
  );

  const panelOpen = coachOpen && !sheetCoach;

  // COLUMN COUNTS FOR A CARD, NOT A ROW. The exercise used to be a thin horizontal band whose
  // controls sat side by side with its label, which is why this grid used to guard a ~241px
  // minimum cell and run only one across beside the coach panel. A `PlanItemRow` is now a card
  // with a SQUARE art stage and its controls stacked underneath, so the binding constraint moved:
  // the floor is the ~115px its action row needs (chip + 36px remove), and the CEILING is what
  // matters instead — three across on a wide screen would make each figure ~350px tall, which is a
  // poster, not a plan. So the counts go UP.
  //
  // The counts came DOWN again, on the user's instruction ("make the figure bigger in every day").
  // The note above argued that three across makes a poster rather than a plan; that judgement was
  // overruled, and the figure won. Keeping the old counts while this comment said otherwise would
  // leave the next reader trusting a rule the code no longer follows.
  //
  // Widths, measured against the shell (max-w-[1500px], 240px sidebar from `lg`, main `p-5`,
  // content `max-w-6xl` with `px-4`/`lg:px-6`, day panel `p-4`, `gap-3`):
  //   1500 viewport, 3 across → ~350px a cell (~330px of art), was ~239px at four across
  //   1024 viewport, 3 across → ~305px
  //    768 viewport, 2 across → ~345px, was ~226px at three across
  //    375 viewport, 2 across → ~151px, unchanged
  // The phone stays at two across. One would give a ~300px figure and fit a single exercise on
  // screen, turning a day into a scroll — that part of the old reasoning still holds, and nobody
  // complained about the phone.
  //
  // WITH THE PANEL the plan column is ~483px at 1280 and ~560px at 1360, so two across is what
  // fits at either width; the extra column it used to gain at 1360 is what made those figures
  // small, so it is gone.
  const itemGrid = panelOpen
    ? "mt-3 grid grid-cols-2 gap-3"
    : "mt-3 grid grid-cols-2 gap-3 lg:grid-cols-3";

  return (
    <AppLayout>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-12">
          {backLink}

          {/* Fixed copy only — no advice generated here. The care loop's job is to notice and
              route the reader to their therapist, not to diagnose. */}
          {latestCheckin?.flagged && (
            <div className="mt-4 flex flex-col gap-1 rounded-xl border border-danger/30 bg-danger/[0.05] px-3 py-2 text-xs text-danger">
              <p className="flex items-start gap-1.5 font-medium">
                <WarningCircle size={14} weight="duotone" className="mt-px shrink-0" />
                {t("plans.redflag.title")}
              </p>
              <p>{t("plans.redflag.body")}</p>
              {latestCheckin.flag_reasons.length > 0 && (
                <ul className="list-disc pl-5">
                  {latestCheckin.flag_reasons.map((reason) => (
                    <li key={reason}>{t(`checkin.reason.${reason}`)}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div
            className={
              panelOpen ? "grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_400px]" : undefined
            }
          >
          <div className="min-w-0">
          <div className="mt-4 flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <h1 className="flex flex-wrap items-center gap-2 font-display text-2xl font-bold text-content">
                {plan.name}
                {plan.assigned_by && (
                  <span className="inline-flex shrink-0 items-center rounded-full bg-primary/10 px-2 py-0.5 text-[10.5px] font-medium text-primary">
                    {t("plans.assignedByTherapist")}
                  </span>
                )}
              </h1>
              {plan.notes && (
                <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">{plan.notes}</p>
              )}
              <p className="mt-2 text-xs tabular-nums text-muted">
                {plan.started_at
                  ? `${t("plans.progress", { done: completed, total: plan.items.length })} · ${
                      today === null
                        ? t("plans.finished")
                        : t("plans.onDay", { n: today })
                    } · ${t("plans.startedOn", {
                      date: new Date(plan.started_at).toLocaleDateString(lang),
                    })}`
                  : t("plans.notStarted")}
              </p>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              {/* Desktop only: on a phone this same panel is reached from the floating button
                  below, because a fifth control in this row would wrap onto its own line. */}
              {!isMobile && (
                <button
                  type="button"
                  onClick={() => setCoachOpen((v) => !v)}
                  aria-expanded={coachOpen}
                  className="inline-flex items-center gap-2 rounded-full border border-border-dark bg-surface px-4 py-2 text-[13px] font-semibold text-content transition-all hover:border-primary/40 hover:text-primary active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                >
                  <LumenAvatar size={17} />
                  {coachOpen ? t("plans.coach.close") : t("plans.coach.open")}
                </button>
              )}
              <button
                type="button"
                onClick={() => (plan.started_at ? setConfirming("restart") : void start())}
                disabled={planBusy || plan.items.length === 0}
                className="inline-flex items-center gap-2 rounded-full bg-primary px-5 py-2 text-[13px] font-semibold text-primary-content shadow-accent transition-all hover:bg-primary/90 active:scale-[0.98] disabled:opacity-50 disabled:active:scale-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                <Play size={14} weight="fill" />
                {planBusy
                  ? t("plans.starting")
                  : plan.started_at
                    ? t("plans.restart")
                    : t("plans.start")}
              </button>
              <button
                type="button"
                onClick={() => setConfirming("delete")}
                aria-label={t("plans.deletePlan")}
                className="rounded-full border border-border-dark p-2 text-faint transition-all hover:border-danger/40 hover:text-danger active:scale-[0.95] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-danger"
              >
                <Trash size={15} weight="duotone" />
              </button>
            </div>
          </div>

          {plan.started_at && plan.items.length > 0 && (
            <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-content/[0.06]">
              <div
                className="h-full rounded-full bg-primary transition-[width]"
                style={{ width: `${Math.round(ratio * 100)}%` }}
              />
            </div>
          )}

          {itemError && (
            <p className="mt-4 flex items-start gap-1.5 rounded-xl border border-danger/30 bg-danger/[0.05] px-3 py-2 text-xs text-danger">
              <WarningCircle size={14} weight="duotone" className="mt-px shrink-0" />
              {itemError}
            </p>
          )}

          {/* THE WEEK AS A STRIP OF SEVEN SUMMARY TILES, PLUS ONE FOCUSED DAY.
              Two earlier layouts are buried under this one, and both failures come down to the
              same measurement, so it is recorded here as well as in WeekStrip.

              (1) SEVEN COLUMNS, each holding its own exercises. At >=1280px a column is ~147px
              while one exercise ROW needs ~241px — a checkbox, an icon, a name, an action chip and
              a delete button — so rows overflowed their column by ~120px into the next day and the
              flex-1 movement NAME was squeezed to zero width and vanished. Truncation cannot save
              it: the fixed parts alone exceed the column. THE RULE THAT COMES OUT OF THAT, and the
              one thing not to regress: an exercise row never goes in a narrow per-day column. A
              narrow tile holding only a SUMMARY — a label, a count, a bar, one muscle name — is
              fine, because every part of it can shrink.

              (2) SEVEN FULL-WIDTH BANDS stacked down the page. Legal, but it gave a rest day the
              same visual weight as a training day: a three-day plan drew three bands of content and
              then four near-empty ones, each still carrying a label, a hairline and a button, so
              most of the page was furniture — and there was no "week" anywhere, only a list you
              scrolled.

              The strip is the summary layer (narrow is fine there) and the panel below it is the
              row layer, at the FULL page width the rows need. Every day still exists in the strip,
              rest days included, so "Day 3" means the same thing in every plan. */}
          <div className="mt-6">
            <WeekStrip
              days={days}
              selected={selectedDay}
              today={today}
              onSelect={(day) => {
                setPickedDay(day);
                // The add form belongs to the day it was opened on. Leaving it open across a
                // selection change would let someone fill it in while looking at Wednesday and
                // land the exercise on Monday.
                setAddingTo(null);
              }}
              panelId={DAY_PANEL_ID}
              tabId={dayTabId}
            />

            <section
              id={DAY_PANEL_ID}
              role="tabpanel"
              aria-labelledby={dayTabId(selectedDay)}
              tabIndex={-1}
              className="mt-3 rounded-2xl border border-border-dark bg-surface p-4"
            >
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                <h2 className="shrink-0 text-sm font-semibold text-content">
                  {t("plans.day", { n: selectedDay })}
                </h2>
                <span className="shrink-0 text-xs text-faint">
                  {selectedItems.length === 0
                    ? t("plans.rest")
                    : selectedItems.length === 1
                      ? t("plans.exerciseCountOne")
                      : t("plans.exerciseCount", { n: selectedItems.length })}
                </span>
                <DayMuscles items={selectedItems} />
                <span className="h-px min-w-6 flex-1 bg-border-dark" />
                {/* A rest day's only add action is the one inside its empty state below — two
                    buttons saying "Add exercise" a hundred pixels apart is a choice the user does
                    not have to make. */}
                {addingTo !== selectedDay && selectedItems.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setAddingTo(selectedDay)}
                    // `ml-auto` for the wrapped case: this header wraps, and when the chips push
                    // the button onto a second line the hairline spacer stays behind on the first
                    // one, so without it the button lands hard against the left margin instead of
                    // where its whole line expects it.
                    className="ml-auto inline-flex shrink-0 items-center gap-1.5 rounded-full border border-dashed border-border-dark px-3.5 py-1.5 text-xs font-medium text-muted transition-colors hover:border-primary/40 hover:text-primary active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                  >
                    <Plus size={12} weight="bold" />
                    {t("plans.addExercise")}
                  </button>
                )}
              </div>

              {selectedItems.length > 0 && (
                // Up to four exercises across on a wide screen, two on a phone — see the
                // `itemGrid` arithmetic above for why the counts are what they are.
                <ul className={itemGrid}>
                  {selectedItems.map((item) => (
                    <PlanItemRow
                      key={item.id}
                      item={item}
                      planId={plan.id}
                      analyzable={isAnalyzable(item.movement, analyzable)}
                      busy={busyItem === item.id}
                      onToggle={() => void patchItem(item.id, { completed: !item.completed_at })}
                      onRemove={() => void removeItem(item.id)}
                      onCheckin={() =>
                        setCheckinTarget({
                          itemId: item.id,
                          analysisId: item.analysis_id ?? undefined,
                          movement: item.movement,
                        })
                      }
                    />
                  ))}
                </ul>
              )}

              {/* A composed empty state, not a blank box: it says what this day IS and offers the
                  one action that changes it. */}
              {selectedItems.length === 0 && addingTo !== selectedDay && (
                <div className="mt-3 flex flex-col items-center gap-3 rounded-xl border border-dashed border-border-dark px-4 py-8 text-center">
                  <Moon size={22} weight="duotone" className="text-faint" />
                  <p className="max-w-sm text-sm leading-relaxed text-muted">
                    {t("plans.restEmptyBody")}
                  </p>
                  <button
                    type="button"
                    onClick={() => setAddingTo(selectedDay)}
                    // `py-2.5` rather than the header pill's `py-1.5`: this is the only way out of
                    // a rest day and it is meant to be tapped, so it clears the 36px touch target
                    // (16px line + 2 x 10px) instead of the 28px a header-sized pill would give.
                    className="inline-flex items-center gap-1.5 rounded-full border border-dashed border-border-dark bg-surface px-4 py-2.5 text-xs font-medium text-muted transition-colors hover:border-primary/40 hover:text-primary active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                  >
                    <Plus size={12} weight="bold" />
                    {t("plans.addExercise")}
                  </button>
                </div>
              )}

              {addingTo === selectedDay && (
                <div className="mt-3 max-w-xs">
                  <AddExerciseForm
                    day={selectedDay}
                    busy={planBusy}
                    onAdd={(item) => void addItem(item)}
                    onCancel={() => setAddingTo(null)}
                  />
                </div>
              )}
            </section>
          </div>

          {/* Below the week, not beside it: it is a summary OF the bands above, and it sits inside
              the same column so the sticky Lumen panel keeps its own full height. */}
          <PlanMuscleCoverage items={plan.items} className="mt-3" compact={panelOpen} />
          </div>

            {/* Sticky and full-height with its own scroll, like the builder's chat column: a long
                plan scrolls past a conversation that stays put. */}
            {panelOpen && (
              <aside className="h-[600px] min-h-0 xl:sticky xl:top-6 xl:h-[min(calc(100vh-14rem),46rem)]">
                {renderCoach("h-full")}
              </aside>
            )}
          </div>
        </main>
      </div>

      {/* PHONE: a floating way in, and a sheet rather than a column — there is no second column on
          a phone, and the plan is what the user came to look at. Lifted clear of the tab bar's
          own safe-area inset so it never sits on top of the navigation. */}
      {isMobile && !coachOpen && (
        <button
          type="button"
          onClick={() => setCoachOpen(true)}
          aria-label={t("plans.coach.open")}
          className="fixed bottom-[calc(max(env(safe-area-inset-bottom),0.75rem)+5.25rem)] right-4 z-30 flex h-14 w-14 items-center justify-center rounded-full bg-primary shadow-accent transition-transform active:scale-95 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
        >
          <LumenAvatar size={30} />
        </button>
      )}

      {sheetCoach && coachOpen && (
        <div className="fixed inset-0 z-40 flex flex-col justify-end bg-content/30">
          {/* The backdrop is a dismiss target, not a control: `aria-hidden` keeps it out of the
              accessibility tree, where it would otherwise be a second element named "Close
              Lumen" competing with the button in the sheet. */}
          <div aria-hidden="true" onClick={() => setCoachOpen(false)} className="flex-1" />
          <div
            role="dialog"
            aria-modal="true"
            aria-label={t("chat.coach")}
            className="flex h-[78vh] flex-col rounded-t-2xl border-t border-border-dark bg-surface pb-[max(env(safe-area-inset-bottom),0.75rem)]"
          >
            <div className="mb-1 flex shrink-0 justify-end px-2 pt-2">
              <button
                type="button"
                onClick={() => setCoachOpen(false)}
                aria-label={t("plans.coach.close")}
                className="rounded-full p-2.5 text-faint transition-colors hover:bg-content/[0.06] hover:text-content focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                <X size={16} weight="bold" />
              </button>
            </div>
            <div className="min-h-0 flex-1">
              {renderCoach("h-full rounded-none border-0 shadow-none")}
            </div>
          </div>
        </div>
      )}

      <CheckinDialog
        open={checkinTarget !== null}
        planId={plan.id}
        planItemId={checkinTarget?.itemId}
        analysisId={checkinTarget?.analysisId}
        movementLabel={checkinTarget ? movementLabel(t, checkinTarget.movement) : undefined}
        onClose={() => setCheckinTarget(null)}
        onSubmitted={() => {
          setCheckinTarget(null);
          refreshLatestCheckin();
        }}
      />

      <ConfirmDialog
        open={confirming === "restart"}
        title={t("plans.restartTitle")}
        description={t("plans.restartBody")}
        detail={plan.name}
        confirmLabel={t("plans.restart")}
        cancelLabel={t("plans.cancel")}
        busy={planBusy}
        onConfirm={() => void start()}
        onCancel={() => setConfirming(null)}
      />
      <ConfirmDialog
        open={confirming === "delete"}
        title={t("plans.deletePlanTitle")}
        description={t("plans.deletePlanBody")}
        detail={plan.name}
        confirmLabel={t("plans.delete")}
        cancelLabel={t("plans.cancel")}
        busy={planBusy}
        onConfirm={() => void removePlan()}
        onCancel={() => setConfirming(null)}
      />
    </AppLayout>
  );
}

/**
 * What the FOCUSED day trains, as up to three muted chips on the day panel's header line. Three
 * and then a count: a full session touches eight groups, and naming them all would out-shout the
 * day label and the add button sharing this row.
 *
 * VISIBLE AT EVERY WIDTH, unlike the version that lived on the old day bands. That one was hidden
 * below `sm` because a band header was a single NON-WRAPPING row and at 375px the chips shoved the
 * add button off the card. This header wraps (`flex-wrap` on the parent), and there is now exactly
 * one of these on the page instead of seven, so the chips can take a second line on a phone
 * instead of disappearing from it.
 *
 * A rest day has no items, so `dayMuscles` is empty and nothing renders. The strip's tiles name
 * only the top group, and the coverage card below names the whole week.
 */
function DayMuscles({ items }: { items: PlanItem[] }) {
  const { t } = useI18n();
  const muscles = dayMuscles(items);
  if (muscles.length === 0) return null;
  const shown = muscles.slice(0, 3);
  const overflow = muscles.length - shown.length;
  return (
    <ul className="flex shrink-0 items-center gap-1.5">
      {shown.map((m) => (
        <li
          key={m}
          className="rounded-full bg-content/[0.05] px-2 py-0.5 text-[10.5px] font-medium text-faint"
        >
          {t(`muscle.${m}`)}
        </li>
      ))}
      {overflow > 0 && (
        <li className="text-[10.5px] font-medium tabular-nums text-faint">
          {t("plans.muscles.more", { n: overflow })}
        </li>
      )}
    </ul>
  );
}
