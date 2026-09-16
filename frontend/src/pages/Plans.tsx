import { useCallback, useEffect, useState } from "react";
import { ArrowRight, CalendarBlank, Plus, Sparkle, WarningCircle } from "@phosphor-icons/react";
import { Link, useNavigate } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import PlanCard from "../components/plans/PlanCard";
import CreatePlanDialog from "../components/plans/CreatePlanDialog";
import MovementIcon from "../components/movements/MovementIcon";
import { LumenAvatar } from "../components/LumenLoader";
import { api, type Plan, type PlanSummary, type PlanTemplate } from "../api";
import { movementLabel, useI18n } from "../lib/i18n";
import { MuscleSummaryChips } from "../components/plans/PlanMuscleCoverage";
import { currentDay, itemsByDay, progressRatio, templateText } from "../lib/plans";
import { coverageOf, planCoverage } from "../lib/planMuscles";

type Status = "loading" | "ready" | "error";

// "訓練菜單": the signed-in user's own routines, plus the built-in templates to start from.
//
// Two ways in, and they are not the same offer: talking to Lumen (`/plans/new`) is the one that
// writes a whole week for you, so it leads; building it by hand stays beside it for the user who
// already knows what they want and would rather not negotiate for it.
//
// The template gallery is always on the page, not tucked behind the create dialog: for someone
// with no plans it IS the empty state, and for someone with three it is still the fastest way to
// add a fourth. Each template now offers both endings — copy it as-is, or hand it to Lumen to
// adapt — because a template that ALMOST fits used to mean editing it row by row afterwards.
export default function Plans() {
  const { t } = useI18n();
  const navigate = useNavigate();

  const [plans, setPlans] = useState<PlanSummary[]>([]);
  const [templates, setTemplates] = useState<PlanTemplate[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState("");

  // Which template the create dialog opens with. `null` closes it; `""` is "start blank" — an
  // explicit third state, because `undefined` would be indistinguishable from "closed".
  const [creatingFrom, setCreatingFrom] = useState<string | null>(null);

  // The template currently being handed to Lumen, so its card can say so and cannot be double-fired.
  const [customising, setCustomising] = useState<string | null>(null);
  const [customiseError, setCustomiseError] = useState("");

  const load = useCallback(async () => {
    setStatus("loading");
    setError("");
    try {
      const mine = await api.listPlans();
      setPlans(mine);
      setStatus("ready");
      try {
        setTemplates(await api.planTemplates());
      } catch {
        // The templates are a suggestion, not the page. A failure here must not turn a readable
        // list of the user's own plans into an error state.
        setTemplates([]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // The run already under way: the first started plan that still has something left in it. The
  // LIST rows carry counts but no items, so the exercises this card offers need the plan itself.
  const resuming = plans.find(
    (p) => p.started_at && p.item_count > 0 && p.completed_count < p.item_count
  );
  const resumingId = resuming?.id;

  // Fetched separately and allowed to fail: the card's headline facts (name, progress) come from
  // the summary the list already returned, so a failed detail fetch costs the exercise chips and
  // nothing else — never the card, and never the page.
  const [detail, setDetail] = useState<Plan | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  useEffect(() => {
    if (!resumingId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setDetail(null);
    api
      .getPlan(resumingId)
      .then((p) => !cancelled && setDetail(p))
      .catch(() => undefined)
      .finally(() => !cancelled && setDetailLoading(false));
    return () => {
      cancelled = true;
    };
  }, [resumingId]);

  // A freshly created plan opens straight into its detail page: the user's next action is always
  // to look at it (a blank plan needs exercises; a template plan needs starting), never to admire
  // it in the list.
  const onCreated = (id: string) => {
    setCreatingFrom(null);
    navigate(`/plans/${id}`);
  };

  // "請 Lumen 客製": copy the template, then open its page with the coach panel already up. Two
  // steps rather than one, deliberately — the plan exists and is the user's before any
  // conversation happens, so an abandoned chat still leaves them a usable plan.
  const customise = async (template: PlanTemplate) => {
    setCustomising(template.key);
    setCustomiseError("");
    try {
      const created = await api.createPlan({
        name: templateText(t, template.key, "name", template.name),
        template_key: template.key,
      });
      navigate(`/plans/${created.id}?coach=1`);
    } catch (e) {
      setCustomiseError(e instanceof Error ? e.message : t("plans.itemFailed"));
    } finally {
      setCustomising(null);
    }
  };

  // Rehab-first grouping (WP4): the same bucketing TemplatePickerDialog applies for a clinician's
  // own picker. A template with no `category` (an older backend, or one the migration hasn't
  // tagged yet) reads as "fitness" — never dropped, just bucketed with the rest.
  const rehabTemplates = templates.filter((tpl) => tpl.category === "rehab");
  const fitnessTemplates = templates.filter((tpl) => tpl.category !== "rehab");

  const renderTemplateCard = (template: PlanTemplate) => {
    const days = new Set(template.items.map((i) => i.day_index)).size;
    // Distinct movements only, and only the first few: the point of the pills is
    // "what does this train", which repeating "Squat" three times does not answer.
    // Same cap and same `+n` ending as PlanCard — the full-body template holds nine
    // distinct movements and would otherwise be all pills.
    const movements = [...new Set(template.items.map((i) => i.movement))];
    const shown = movements.slice(0, 4);
    const overflow = movements.length - shown.length;
    // Ranked over DISTINCT movements, exactly as PlanCard ranks a plan summary's
    // `movements` — the two lines then mean the same thing on both cards.
    const muscles = coverageOf(movements).primary;
    const busy = customising === template.key;
    return (
      <li
        key={template.key}
        className="flex h-full min-w-0 flex-col rounded-2xl border border-border-dark bg-surface p-4 shadow-card transition-all hover:border-primary/35 hover:shadow-card-hover focus-within:border-primary/35 focus-within:shadow-card-hover"
      >
        <h3 className="truncate font-display text-[15px] font-semibold text-content">
          {templateText(t, template.key, "name", template.name)}
        </h3>
        {/* The same meta line PlanCard draws, down to the icon and `tabular-nums`:
            it is the single strongest signal that these are the same kind of thing. */}
        <p className="mt-1 flex items-center gap-1.5 text-xs tabular-nums text-muted">
          <CalendarBlank size={13} weight="duotone" className="shrink-0" />
          {t(days === 1 ? "plans.templateItemsOneDay" : "plans.templateItems", {
            n: template.items.length,
            days,
          })}
        </p>
        <p className="mt-2 text-xs leading-relaxed text-muted">
          {templateText(t, template.key, "desc", template.description)}
        </p>

        <ul className="mt-3 flex flex-wrap gap-1.5">
          {shown.map((movement) => (
            <li
              key={movement}
              className="inline-flex items-center gap-1 rounded-full bg-content/[0.04] px-2 py-1 text-[11px] font-medium text-muted"
            >
              <MovementIcon movement={movement} size={13} />
              {movementLabel(t, movement)}
            </li>
          ))}
          {overflow > 0 && (
            <li className="inline-flex items-center rounded-full bg-content/[0.04] px-2 py-1 text-[11px] font-medium text-faint">
              +{overflow}
            </li>
          )}
        </ul>

        <MuscleSummaryChips muscles={muscles} className="mt-2" />

        <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-4">
          <button
            type="button"
            onClick={() => setCreatingFrom(template.key)}
            className="inline-flex min-h-[36px] items-center gap-1.5 whitespace-nowrap rounded-full border border-border-dark px-3 py-2 text-xs font-semibold text-content transition-all hover:border-primary/40 hover:text-primary active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
          >
            {t("plans.useTemplate")}
          </button>
          <button
            type="button"
            onClick={() => void customise(template)}
            disabled={busy}
            className="inline-flex min-h-[36px] items-center gap-1.5 whitespace-nowrap rounded-full bg-primary/10 px-3 py-2 text-xs font-semibold text-primary transition-all hover:bg-primary/15 active:scale-[0.98] disabled:opacity-60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
          >
            <LumenAvatar size={15} />
            {busy ? t("plans.creating") : t("plans.customiseWithLumen")}
          </button>
        </div>
      </li>
    );
  };

  const primaryCta = (
    <Link
      to="/plans/new"
      className="inline-flex shrink-0 items-center gap-2 rounded-full bg-primary px-5 py-2.5 text-[13px] font-semibold text-primary-content shadow-accent transition-all hover:bg-primary/90 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
    >
      <LumenAvatar size={18} />
      {t("plans.planWithLumen")}
    </Link>
  );

  // The exercises the current day still asks for. Ticked ones are dropped: the card is a list of
  // what is LEFT, and re-offering finished work is how a "continue" card stops being one.
  const today = detail ? currentDay(detail.items) : null;
  // Derived from the FETCHED plan only, never from the list row's distinct movement names as a
  // stop-gap: the two rank differently (see `coverageOf`), so a fallback would render one order and
  // then visibly reshuffle into another the moment the fetch landed — beside a skeleton that exists
  // precisely to keep item-derived content off the card until it is real.
  const resumingMuscles = detail ? planCoverage(detail.items).primary : [];
  const todayItems =
    detail && today !== null
      ? itemsByDay(detail.items)[today - 1].filter((it) => !it.completed_at)
      : [];

  const continueCard = resuming && (
    <section className="mt-8 rounded-2xl border border-primary/30 bg-primary/[0.03] p-5 shadow-card">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wider text-primary">
            {t("plans.continueTitle")}
          </p>
          <h2 className="mt-1.5 truncate font-display text-xl font-bold text-content">
            {resuming.name}
          </h2>
          <p className="mt-1 text-xs tabular-nums text-muted">
            {today !== null && (
              <>
                {t("plans.dayOf", { n: today, total: resuming.day_count })}
                <span aria-hidden="true"> · </span>
              </>
            )}
            {t("plans.progress", {
              done: resuming.completed_count,
              total: resuming.item_count,
            })}
          </p>
        </div>
        <Link
          to={`/plans/${resuming.id}`}
          className="inline-flex shrink-0 items-center gap-2 rounded-full bg-primary px-5 py-2 text-[13px] font-semibold text-primary-content shadow-accent transition-all hover:bg-primary/90 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
        >
          {t("plans.open")}
          <ArrowRight size={14} weight="bold" />
        </Link>
      </div>

      <div
        className="mt-4 h-1.5 overflow-hidden rounded-full bg-content/[0.08]"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={resuming.item_count}
        aria-valuenow={resuming.completed_count}
        aria-label={t("plans.progress", {
          done: resuming.completed_count,
          total: resuming.item_count,
        })}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width]"
          style={{
            width: `${Math.round(progressRatio(resuming.completed_count, resuming.item_count) * 100)}%`,
          }}
        />
      </div>

      <MuscleSummaryChips muscles={resumingMuscles} className="mt-3" />

      {detailLoading && <div className="mt-4 h-8 animate-pulse rounded-xl bg-content/[0.05]" />}

      {todayItems.length > 0 && (
        <>
          <p className="mt-4 text-[11px] font-semibold uppercase tracking-wider text-faint">
            {t("plans.todayExercises")}
          </p>
          <ul className="mt-2 flex flex-wrap gap-2">
            {todayItems.map((item) => (
              <li key={item.id}>
                <Link
                  to={`/app?movement=${encodeURIComponent(item.movement)}&plan=${encodeURIComponent(
                    resuming.id
                  )}&plan_item=${encodeURIComponent(item.id)}`}
                  className="inline-flex items-center gap-1.5 rounded-full border border-border-dark bg-surface px-3 py-1.5 text-[12px] font-medium text-content transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:text-primary active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                >
                  <MovementIcon movement={item.movement} size={14} />
                  {movementLabel(t, item.movement)}
                  <span className="tabular-nums text-faint">
                    {t("plans.setsReps", { sets: item.sets, reps: item.reps })}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );

  return (
    <AppLayout>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-12">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 className="font-display text-2xl font-bold text-content">{t("plans.title")}</h1>
              <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">
                {t("plans.subtitle")}
              </p>
            </div>
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              {primaryCta}
              <button
                type="button"
                onClick={() => setCreatingFrom("")}
                className="inline-flex shrink-0 items-center gap-2 rounded-full border border-border-dark bg-surface px-4 py-2.5 text-[13px] font-semibold text-content transition-all hover:border-primary/40 hover:text-primary active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                <Plus size={15} weight="bold" />
                {t("plans.buildMyself")}
              </button>
            </div>
          </div>

          {status === "error" && (
            <div className="mt-8 flex flex-col items-center gap-3 rounded-2xl border border-border-dark bg-surface px-6 py-12 text-center">
              <WarningCircle size={22} className="text-danger" weight="duotone" />
              <p className="text-sm text-muted">{t("plans.loadFailed")}</p>
              <p className="text-xs text-faint">{error}</p>
              <button
                type="button"
                onClick={() => void load()}
                className="mt-1 rounded-full border border-border-dark px-4 py-1.5 text-xs font-medium text-content transition-colors hover:border-primary/40 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                {t("plans.retry")}
              </button>
            </div>
          )}

          {/* The skeleton is the shape that arrives, not a generic three cards: a started plan
              puts a wide continue band above the grid, so the placeholder does too — otherwise
              the whole page jumps down by a band's height the moment the fetch lands. */}
          {status === "loading" && (
            <>
              <div className="mt-8 h-[168px] animate-pulse rounded-2xl border border-border-dark bg-surface" />
              <div className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="h-[150px] animate-pulse rounded-2xl border border-border-dark bg-surface"
                  />
                ))}
              </div>
            </>
          )}

          {status === "ready" && continueCard}

          {status === "ready" && plans.length > 0 && (
            <section className="mt-10">
              <div className="flex items-center gap-3">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">
                  {t("plans.mineTitle")}
                </h2>
                <span className="h-px flex-1 bg-border-dark" />
              </div>
              <ul className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 lg:gap-4">
                {plans.map((plan) => (
                  <li key={plan.id}>
                    <PlanCard plan={plan} />
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* The empty state is a composed invitation, not a sentence in a box: the page has
              nothing else on it at this point, so it is the whole first impression. Its CTA is
              worded differently from the header's on purpose — both are on screen together. */}
          {status === "ready" && plans.length === 0 && (
            <div className="mt-8 flex flex-col items-center gap-3 rounded-2xl border border-dashed border-border-dark bg-surface/60 px-6 py-14 text-center">
              <LumenAvatar size={48} />
              <h2 className="font-display text-lg font-bold text-content">
                {t("plans.emptyTitle")}
              </h2>
              <p className="max-w-sm text-sm leading-relaxed text-muted">{t("plans.empty")}</p>
              <Link
                to="/plans/new"
                className="mt-1 inline-flex items-center gap-2 rounded-full bg-primary px-5 py-2.5 text-[13px] font-semibold text-primary-content shadow-accent transition-all hover:bg-primary/90 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
              >
                {t("plans.emptyCta")}
                <ArrowRight size={14} weight="bold" />
              </Link>
            </div>
          )}

          {status === "ready" && templates.length > 0 && (
            <section className="mt-10">
              <div className="flex items-center gap-3">
                <h2 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-faint">
                  <Sparkle size={13} weight="duotone" />
                  {t("plans.templatesTitle")}
                </h2>
                <span className="h-px flex-1 bg-border-dark" />
              </div>
              <p className="mt-2 text-xs text-faint">{t("plans.templatesSubtitle")}</p>

              {customiseError && (
                <p className="mt-3 flex items-start gap-1.5 rounded-xl border border-danger/30 bg-danger/[0.05] px-3 py-2 text-xs text-danger">
                  <WarningCircle size={14} weight="duotone" className="mt-px shrink-0" />
                  {customiseError}
                </p>
              )}

              {/* Cards, on the same grid as the user's own plans above — asked for, and the old
                  objection to them no longer applies. That objection assumed a card cell holding
                  nothing but two stacked buttons; the cell now carries the body PlanCard carries
                  (meta line, movement pills, muscle line), so the two actions are a FOOTER under
                  content rather than the content, and they sit side by side in one row at every
                  column width from about 1150px up. `mt-auto` on that row lines the actions up
                  across cards whose descriptions run to different lengths; every card's two
                  labels are identical, so where the column IS too narrow for one row (the
                  1024-1150 band, where three columns are at their tightest) all five wrap the
                  same way and the rows still line up. That fit is why "Use this" lost its `+`
                  glyph here: with it, the pair overran a 1280px three-up column by two pixels.

                  What is deliberately NOT copied from PlanCard: the whole-card lift and
                  `active:scale`. PlanCard is one link, and the lift is what says so — a card with
                  two actions must not claim to be a single click target, and `:active` on the cell
                  would fire from a mousedown on either button. Hover and focus-within move the
                  border and the shadow instead, and the real focus rings stay on the buttons.

                  Grouped rehab-first under the same `clinic.rehabTemplates` / `clinic.fitnessTemplates`
                  headings TemplatePickerDialog uses for the clinician's own picker (WP4) — one
                  vocabulary for "which bucket is this template in", on both sides of the care loop.
                  A template with no category (an older backend, or one the migration hasn't tagged
                  yet) reads as "fitness", the same fallback TemplatePickerDialog applies. */}
              {rehabTemplates.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-faint">
                    {t("clinic.rehabTemplates")}
                  </h3>
                  <ul className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 lg:gap-4">
                    {rehabTemplates.map(renderTemplateCard)}
                  </ul>
                </div>
              )}
              {fitnessTemplates.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-faint">
                    {t("clinic.fitnessTemplates")}
                  </h3>
                  <ul className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 lg:gap-4">
                    {fitnessTemplates.map(renderTemplateCard)}
                  </ul>
                </div>
              )}
            </section>
          )}
        </main>
      </div>

      <CreatePlanDialog
        open={creatingFrom !== null}
        templateKey={creatingFrom || undefined}
        templateName={
          creatingFrom
            ? templateText(
                t,
                creatingFrom,
                "name",
                templates.find((tpl) => tpl.key === creatingFrom)?.name ?? ""
              )
            : undefined
        }
        onCancel={() => setCreatingFrom(null)}
        onCreated={onCreated}
      />
    </AppLayout>
  );
}
