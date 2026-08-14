import { useCallback, useEffect, useState } from "react";
import { Plus, Sparkle, WarningCircle } from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import PlanCard from "../components/plans/PlanCard";
import CreatePlanDialog from "../components/plans/CreatePlanDialog";
import { api, type PlanSummary, type PlanTemplate } from "../api";
import { movementLabel, useI18n } from "../lib/i18n";
import { templateText } from "../lib/plans";

type Status = "loading" | "ready" | "error";

// "訓練菜單": the signed-in user's own routines, plus the built-in templates to start from.
//
// The template gallery is always on the page, not tucked behind the create dialog: for someone
// with no plans it IS the empty state, and for someone with three it is still the fastest way to
// add a fourth. The create dialog opens with a template preselected when a template card is what
// was clicked, so both routes end in the same place.
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

  // A freshly created plan opens straight into its detail page: the user's next action is always
  // to look at it (a blank plan needs exercises; a template plan needs starting), never to admire
  // it in the list.
  const onCreated = (id: string) => {
    setCreatingFrom(null);
    navigate(`/plans/${id}`);
  };

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
            <button
              type="button"
              onClick={() => setCreatingFrom("")}
              className="inline-flex shrink-0 items-center gap-2 rounded-full bg-primary px-5 py-2 text-[13px] font-semibold text-primary-content shadow-accent transition-colors hover:bg-primary/90 active:scale-[0.99]"
            >
              <Plus size={15} weight="bold" />
              {t("plans.new")}
            </button>
          </div>

          {status === "error" && (
            <div className="mt-8 flex flex-col items-center gap-3 rounded-2xl border border-border-dark bg-surface px-6 py-12 text-center">
              <WarningCircle size={22} className="text-danger" weight="duotone" />
              <p className="text-sm text-muted">{t("plans.loadFailed")}</p>
              <p className="text-xs text-faint">{error}</p>
              <button
                type="button"
                onClick={() => void load()}
                className="mt-1 rounded-full border border-border-dark px-4 py-1.5 text-xs font-medium text-content transition-colors hover:border-primary/40"
              >
                {t("plans.retry")}
              </button>
            </div>
          )}

          {status === "loading" && (
            <div className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="h-[150px] animate-pulse rounded-2xl border border-border-dark bg-surface"
                />
              ))}
            </div>
          )}

          {status === "ready" && plans.length > 0 && (
            <ul className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 lg:gap-4">
              {plans.map((plan) => (
                <li key={plan.id}>
                  <PlanCard plan={plan} />
                </li>
              ))}
            </ul>
          )}

          {status === "ready" && plans.length === 0 && (
            <p className="mt-8 rounded-2xl border border-border-dark bg-surface px-6 py-10 text-center text-sm text-muted">
              {t("plans.empty")}
            </p>
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

              <ul className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {templates.map((template) => {
                  const days = new Set(template.items.map((i) => i.day_index)).size;
                  // Distinct movements only, and only the first few: the point of the line is
                  // "what does this train", which repeating "Squat" three times does not answer.
                  const movements = [...new Set(template.items.map((i) => i.movement))].slice(0, 3);
                  return (
                    <li key={template.key}>
                      <button
                        type="button"
                        onClick={() => setCreatingFrom(template.key)}
                        className="group flex h-full w-full flex-col rounded-2xl border border-border-dark bg-surface p-4 text-left transition-all hover:-translate-y-0.5 hover:border-primary/35 hover:shadow-card-hover"
                      >
                        <h3 className="font-display text-[15px] font-semibold text-content">
                          {templateText(t, template.key, "name", template.name)}
                        </h3>
                        <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted">
                          {templateText(t, template.key, "desc", template.description)}
                        </p>
                        <p className="mt-3 text-[11px] text-faint">
                          {t("plans.templateItems", { n: template.items.length, days })}
                        </p>
                        <p className="mt-1 text-[11px] text-faint">
                          {movements.map((m) => movementLabel(t, m)).join(" · ")}
                        </p>
                        <span className="mt-3 inline-flex items-center justify-center gap-1.5 rounded-xl bg-content/[0.04] py-2 text-xs font-semibold text-content transition-colors group-hover:bg-primary group-hover:text-primary-content">
                          <Plus size={13} weight="bold" />
                          {t("plans.useTemplate")}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
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
