import { useState } from "react";
import { Plus, X } from "@phosphor-icons/react";
import { MOVEMENT_GROUPS } from "../../lib/movements";
import { movementLabel, useI18n } from "../../lib/i18n";
import type { NewPlanItem } from "../../api";

interface Props {
  /** Which day the new exercise lands on. The form is rendered per-day, so this is fixed. */
  day: number;
  busy: boolean;
  onAdd: (item: NewPlanItem) => void;
  onCancel: () => void;
}

// Add one exercise to one day. Movement, sets, reps — nothing else, because everything else about
// a plan item is editable by removing it and adding it again, and a five-field form inline in a day
// column is worse than a three-field one.
//
// The movement list is the FULL sixteen-name catalog grouped by body region (MOVEMENT_GROUPS), not
// the analysable fourteen: a plan is a training schedule first, and refusing to let someone write
// down "jumping jacks" because we cannot yet grade their form would be the tail wagging the dog.
// The row renders a tick-only note for those two afterwards.
export default function AddExerciseForm({ day, busy, onAdd, onCancel }: Props) {
  const { t } = useI18n();
  const [movement, setMovement] = useState<string>("Squat");
  const [sets, setSets] = useState(3);
  const [reps, setReps] = useState(10);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    onAdd({ day_index: day, movement, sets, reps });
  };

  const field =
    "h-[36px] rounded-lg border border-border-dark bg-background px-2 text-[13px] text-content outline-none transition-colors focus:border-primary/50";

  return (
    <form
      onSubmit={submit}
      className="rounded-xl border border-dashed border-primary/40 bg-primary/[0.03] p-3"
    >
      <label className="block text-[11px] font-medium text-muted" htmlFor={`movement-${day}`}>
        {t("plans.movementLabel")}
      </label>
      <select
        id={`movement-${day}`}
        value={movement}
        onChange={(e) => setMovement(e.target.value)}
        className={`${field} mt-1 w-full`}
      >
        {MOVEMENT_GROUPS.map((group) => (
          <optgroup key={group.key} label={t(group.key)}>
            {group.items.map((name) => (
              <option key={name} value={name}>
                {movementLabel(t, name)}
              </option>
            ))}
          </optgroup>
        ))}
      </select>

      <div className="mt-2 flex gap-2">
        <div className="min-w-0 flex-1">
          <label className="block text-[11px] font-medium text-muted" htmlFor={`sets-${day}`}>
            {t("plans.setsLabel")}
          </label>
          <input
            id={`sets-${day}`}
            type="number"
            min={1}
            max={20}
            value={sets}
            onChange={(e) => setSets(Number(e.target.value))}
            className={`${field} mt-1 w-full`}
          />
        </div>
        <div className="min-w-0 flex-1">
          <label className="block text-[11px] font-medium text-muted" htmlFor={`reps-${day}`}>
            {t("plans.repsLabel")}
          </label>
          <input
            id={`reps-${day}`}
            type="number"
            min={1}
            max={200}
            value={reps}
            onChange={(e) => setReps(Number(e.target.value))}
            className={`${field} mt-1 w-full`}
          />
        </div>
      </div>

      <div className="mt-3 flex gap-2">
        <button
          type="submit"
          disabled={busy}
          className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-primary py-2 text-xs font-semibold text-primary-content transition-colors hover:bg-primary/90 disabled:opacity-50"
        >
          <Plus size={13} weight="bold" />
          {busy ? t("plans.adding") : t("plans.add")}
        </button>
        <button
          type="button"
          onClick={onCancel}
          aria-label={t("plans.cancel")}
          className="rounded-lg border border-border-dark px-2.5 text-muted transition-colors hover:text-content"
        >
          <X size={14} />
        </button>
      </div>
    </form>
  );
}
