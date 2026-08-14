import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Play, Plus, Trash, WarningCircle } from "@phosphor-icons/react";
import { Link, useParams } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import ConfirmDialog from "../components/ConfirmDialog";
import AddExerciseForm from "../components/plans/AddExerciseForm";
import PlanItemRow from "../components/plans/PlanItemRow";
import { api, type NewPlanItem, type Plan } from "../api";
import { useI18n } from "../lib/i18n";
import type { AnalyzableMovement } from "../lib/movements";
import { PLAN_DAYS, currentDay, isAnalyzable, itemsByDay, progressRatio } from "../lib/plans";

type Status = "loading" | "ready" | "error";

// One plan, as one full-width band per day. Editing is immediate — every tick, add and remove is its own
// request and the local copy is patched from the response, rather than a save button over a draft:
// a plan is edited while standing in a gym, and a draft that needs saving is a draft that gets lost.
export default function PlanDetail() {
  const { t, lang } = useI18n();
  const { planId = "" } = useParams();

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
  const [confirming, setConfirming] = useState<"delete" | "restart" | null>(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [itemError, setItemError] = useState("");
  const [deleted, setDeleted] = useState(false);

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
      className="inline-flex items-center gap-1.5 text-[13px] font-medium text-muted transition-colors hover:text-content"
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
            <div className="mt-6 h-[420px] animate-pulse rounded-2xl border border-border-dark bg-surface" />
          </main>
        </div>
      </AppLayout>
    );
  }

  const days = itemsByDay(plan.items);
  const completed = plan.items.filter((it) => it.completed_at).length;
  const today = currentDay(plan.items);
  const ratio = progressRatio(completed, plan.items.length);

  return (
    <AppLayout>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-12">
          {backLink}

          <div className="mt-4 flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <h1 className="font-display text-2xl font-bold text-content">{plan.name}</h1>
              {plan.notes && (
                <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">{plan.notes}</p>
              )}
              <p className="mt-2 text-xs text-muted">
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
              <button
                type="button"
                onClick={() => (plan.started_at ? setConfirming("restart") : void start())}
                disabled={planBusy || plan.items.length === 0}
                className="inline-flex items-center gap-2 rounded-full bg-primary px-5 py-2 text-[13px] font-semibold text-primary-content shadow-accent transition-colors hover:bg-primary/90 disabled:opacity-50"
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
                className="rounded-full border border-border-dark p-2 text-faint transition-colors hover:border-danger/40 hover:text-danger"
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

          {/* ONE FULL-WIDTH BAND PER DAY, not seven columns across.
              Seven columns was the first attempt and it broke: at >=1280px each day got ~147px
              while a single exercise row needs ~241px, so the row overflowed its card by 120px
              (spilling over the next day) and the flex-1 label was crushed to ZERO width -- the
              movement name simply disappeared. A day column cannot hold a horizontal row carrying a
              checkbox, an icon, a name, an action and a delete button, and no amount of truncation
              fixes that; the row's fixed parts alone exceed the column.
              Bands also spend the vertical space where it is earned: a rest day is one thin line
              here instead of a full-height empty column, and the days that hold work get the whole
              page width for their exercises. Every day is still shown, rest days included, so
              "Day 3" means the same thing in every plan. */}
          <div className="mt-6 flex flex-col gap-3">
            {PLAN_DAYS.map((day) => {
              const items = days[day - 1];
              const isToday = today === day;
              return (
                <section
                  key={day}
                  className={`rounded-2xl border p-4 ${
                    isToday ? "border-primary/40 bg-primary/[0.03]" : "border-border-dark bg-surface"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <h2
                      className={`shrink-0 text-xs font-semibold uppercase tracking-wider ${
                        isToday ? "text-primary" : "text-faint"
                      }`}
                    >
                      {t("plans.day", { n: day })}
                    </h2>
                    {items.length === 0 && addingTo !== day && (
                      <span className="shrink-0 text-[11px] text-faint">{t("plans.rest")}</span>
                    )}
                    <span className="h-px flex-1 bg-border-dark" />
                    {addingTo !== day && (
                      <button
                        type="button"
                        onClick={() => setAddingTo(day)}
                        className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-dashed border-border-dark px-3 py-1 text-[11px] font-medium text-muted transition-colors hover:border-primary/40 hover:text-primary"
                      >
                        <Plus size={12} weight="bold" />
                        {t("plans.addExercise")}
                      </button>
                    )}
                  </div>

                  {items.length > 0 && (
                    // Up to three exercises across on a wide screen. The narrowest cell this
                    // produces is the phone's full width; the widest day still never squeezes a row
                    // below what its own controls need.
                    <ul className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-3">
                      {items.map((item) => (
                        <PlanItemRow
                          key={item.id}
                          item={item}
                          planId={plan.id}
                          analyzable={isAnalyzable(item.movement, analyzable)}
                          busy={busyItem === item.id}
                          onToggle={() =>
                            void patchItem(item.id, { completed: !item.completed_at })
                          }
                          onRemove={() => void removeItem(item.id)}
                        />
                      ))}
                    </ul>
                  )}

                  {addingTo === day && (
                    <div className="mt-3 max-w-xs">
                      <AddExerciseForm
                        day={day}
                        busy={planBusy}
                        onAdd={(item) => void addItem(item)}
                        onCancel={() => setAddingTo(null)}
                      />
                    </div>
                  )}
                </section>
              );
            })}
          </div>
        </main>
      </div>

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
