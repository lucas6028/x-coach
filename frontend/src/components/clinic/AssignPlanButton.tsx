import { useState } from "react";
import { ClipboardText } from "@phosphor-icons/react";
import { api, type PlanTemplate } from "../../api";
import { templateText } from "../../lib/plans";
import { useI18n } from "../../lib/i18n";
import CreatePlanDialog from "../plans/CreatePlanDialog";
import TemplatePickerDialog from "./TemplatePickerDialog";

interface Props {
  patientId: string;
  patientName: string;
  /** Called once a plan was actually created, so the caller can refresh its own data. The dialog
   *  never navigates — the clinician cannot open a patient's plan (owner-only route), so there is
   *  nowhere for it to send them. */
  onAssigned: () => void;
  className?: string;
}

// The "指派菜單" action shared by the patients list (one per row) and the patient detail page (one
// per plans panel): pick a template, then name/notes it via CreatePlanDialog with `forPatient` set.
// Owns both dialogs so neither caller has to re-wire the two-step flow.
export default function AssignPlanButton({ patientId, patientName, onAssigned, className }: Props) {
  const { t } = useI18n();
  // null = not yet fetched. Fetched lazily on first open rather than on every page's mount, since
  // most visits to the list never open the assign flow at all.
  const [templates, setTemplates] = useState<PlanTemplate[] | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [chosen, setChosen] = useState<PlanTemplate | null>(null);

  const open = async () => {
    setPickerOpen(true);
    if (templates === null) {
      try {
        setTemplates(await api.planTemplates());
      } catch {
        // The picker's own "no templates" empty state covers a failed fetch too — there is
        // nothing actionable to add beyond "nothing to choose from right now".
        setTemplates([]);
      }
    }
  };

  const cancel = () => {
    setPickerOpen(false);
    setChosen(null);
  };

  return (
    <>
      <button
        type="button"
        onClick={() => void open()}
        className={
          className ??
          "inline-flex items-center gap-1.5 rounded-full border border-border-dark px-3 py-1.5 text-xs font-semibold text-content transition-colors hover:border-primary/40 hover:text-primary active:scale-[0.98]"
        }
      >
        <ClipboardText size={14} weight="duotone" />
        {t("clinic.assignPlan")}
      </button>

      <TemplatePickerDialog
        open={pickerOpen}
        templates={templates ?? []}
        loading={templates === null}
        onCancel={cancel}
        onPick={(tpl) => {
          setChosen(tpl);
          setPickerOpen(false);
        }}
      />

      <CreatePlanDialog
        open={chosen !== null}
        templateKey={chosen?.key}
        templateName={chosen ? templateText(t, chosen.key, "name", chosen.name) : undefined}
        forPatient={{ id: patientId, name: patientName }}
        onCancel={cancel}
        onCreated={() => {
          setChosen(null);
          onAssigned();
        }}
      />
    </>
  );
}
