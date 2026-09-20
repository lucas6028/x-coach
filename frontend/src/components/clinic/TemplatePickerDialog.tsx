import { useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";
import { CalendarBlank, X } from "@phosphor-icons/react";
import type { PlanTemplate } from "../../api";
import { templateText } from "../../lib/plans";
import { useI18n } from "../../lib/i18n";

interface Props {
  open: boolean;
  templates: PlanTemplate[];
  /** True while the catalog is still being fetched — shown instead of an empty state. */
  loading?: boolean;
  onPick: (template: PlanTemplate) => void;
  onCancel: () => void;
}

// Step 1 of the clinician "assign a plan" flow: pick a starting template, grouped so the rehab
// catalog leads for a therapist (the same list Plans.tsx shows a patient, just re-sorted). Choosing
// one hands off to CreatePlanDialog (with `forPatient` set) for the name/notes step.
//
// Portalled to <body>, unlike CreatePlanDialog: this one is new, and every new dialog in this
// feature follows CheckinDialog's idiom because the pages that open it (ClinicPatients,
// ClinicPatient) render inside AppLayout's `glass-shell`, whose backdrop-filter would otherwise
// clip a `position: fixed` descendant to the card's own bounds.
export default function TemplatePickerDialog({ open, templates, loading, onPick, onCancel }: Props) {
  const { t } = useI18n();
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    return () => restoreRef.current?.focus();
  }, [open]);

  if (!open) return null;

  const rehab = templates.filter((tpl) => tpl.category === "rehab");
  // Anything not explicitly "rehab" — including a template whose category is absent because the
  // backend hasn't tagged it yet — reads as "fitness", per the contract.
  const fitness = templates.filter((tpl) => tpl.category !== "rehab");

  const group = (title: string, items: PlanTemplate[]) =>
    items.length === 0 ? null : (
      <div className="mt-4 first:mt-0">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-faint">{title}</p>
        <ul className="mt-2 flex flex-col gap-1.5">
          {items.map((tpl) => {
            const days = new Set(tpl.items.map((i) => i.day_index)).size;
            return (
              <li key={tpl.key}>
                <button
                  type="button"
                  onClick={() => onPick(tpl)}
                  className="flex w-full flex-col items-start gap-0.5 rounded-xl border border-border-dark bg-surface px-3 py-2.5 text-left transition-colors hover:border-primary/40 active:scale-[0.99] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                >
                  <span className="text-[13px] font-semibold text-content">
                    {templateText(t, tpl.key, "name", tpl.name)}
                  </span>
                  <span className="flex items-center gap-1 text-[11px] tabular-nums text-faint">
                    <CalendarBlank size={12} weight="duotone" className="shrink-0" />
                    {t(days === 1 ? "plans.templateItemsOneDay" : "plans.templateItems", {
                      n: tpl.items.length,
                      days,
                    })}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    );

  return createPortal(
    <div
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="max-h-[80vh] w-full max-w-sm overflow-y-auto rounded-2xl border border-border-dark bg-surface-dark p-5 shadow-lg outline-none"
      >
        <div className="flex items-start justify-between gap-3">
          <h2 id={titleId} className="text-base font-semibold text-content">
            {t("clinic.pickTemplate")}
          </h2>
          <button
            type="button"
            onClick={onCancel}
            aria-label={t("checkin.close")}
            className="rounded-lg p-1 text-muted transition-colors hover:bg-content/5 hover:text-content"
          >
            <X size={18} />
          </button>
        </div>

        {loading && <p className="mt-4 text-sm text-muted">{t("clinic.loadingTemplates")}</p>}
        {!loading && templates.length === 0 && (
          <p className="mt-4 text-sm text-muted">{t("clinic.noTemplates")}</p>
        )}

        {group(t("clinic.rehabTemplates"), rehab)}
        {group(t("clinic.fitnessTemplates"), fitness)}
      </div>
    </div>,
    document.body
  );
}
