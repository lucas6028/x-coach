import { useEffect, useId, useRef, useState } from "react";
import { WarningCircle } from "@phosphor-icons/react";
import { api } from "../../api";
import { useI18n } from "../../lib/i18n";

interface Props {
  open: boolean;
  /** Copy this built-in template's exercises into the new plan. Omitted = start blank. */
  templateKey?: string;
  /** The template's localized name, prefilled into the name field so the common case is one click
   *  from "use this" to a created plan. */
  templateName?: string;
  onCancel: () => void;
  onCreated: (planId: string) => void;
}

// The new-plan form. Follows ConfirmDialog's overlay idiom (`fixed inset-0 z-50`, no portal) rather
// than adding a portal layer for a second dialog, and refuses to close while the create is in
// flight for the same reason: a dismissed-mid-request dialog leaves the user unable to tell whether
// a plan was made.
export default function CreatePlanDialog({
  open,
  templateKey,
  templateName,
  onCancel,
  onCreated,
}: Props) {
  const { t } = useI18n();
  const titleId = useId();
  const nameId = useId();
  const notesId = useId();
  const inputRef = useRef<HTMLInputElement>(null);

  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // Reset on every OPEN, not on mount: the dialog stays mounted between uses, so without this the
  // second plan someone creates starts with the first one's name still in the field. Keyed on
  // `templateKey` too, so switching from one template card to another refreshes the prefill.
  useEffect(() => {
    if (!open) return;
    setName(templateName ?? "");
    setNotes("");
    setError("");
    setBusy(false);
    // Focus the field the user is going to type in. The template case prefills it, so the caret
    // lands somewhere useful either way.
    const id = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(id);
  }, [open, templateKey, templateName]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, busy, onCancel]);

  if (!open) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError("");
    try {
      const plan = await api.createPlan({
        name: trimmed,
        notes: notes.trim() || null,
        ...(templateKey ? { template_key: templateKey } : {}),
      });
      onCreated(plan.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/30 backdrop-blur-[2px]"
        onClick={() => !busy && onCancel()}
        aria-hidden="true"
      />
      <form
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onSubmit={submit}
        className="relative w-full max-w-md rounded-2xl border border-border-dark bg-surface p-5 shadow-card-hover"
      >
        <h2 id={titleId} className="font-display text-lg font-bold text-content">
          {t("plans.createTitle")}
        </h2>

        <label htmlFor={nameId} className="mt-4 block text-xs font-medium text-muted">
          {t("plans.nameLabel")}
        </label>
        <input
          id={nameId}
          ref={inputRef}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={t("plans.namePlaceholder")}
          maxLength={80}
          required
          className="mt-1.5 h-[42px] w-full rounded-xl border border-border-dark bg-background px-3 text-sm text-content outline-none transition-colors placeholder:text-faint focus:border-primary/50"
        />

        <label htmlFor={notesId} className="mt-3 block text-xs font-medium text-muted">
          {t("plans.notesLabel")}
        </label>
        <textarea
          id={notesId}
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder={t("plans.notesPlaceholder")}
          maxLength={500}
          rows={2}
          className="mt-1.5 w-full resize-none rounded-xl border border-border-dark bg-background px-3 py-2 text-sm text-content outline-none transition-colors placeholder:text-faint focus:border-primary/50"
        />

        {error && (
          <p className="mt-3 flex items-start gap-1.5 text-xs text-danger">
            <WarningCircle size={14} weight="duotone" className="mt-px shrink-0" />
            {error}
          </p>
        )}

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="rounded-full border border-border-dark px-4 py-2 text-[13px] font-medium text-content transition-colors hover:border-primary/40 disabled:opacity-50"
          >
            {t("plans.cancel")}
          </button>
          <button
            type="submit"
            disabled={busy || !name.trim()}
            className="rounded-full bg-primary px-5 py-2 text-[13px] font-semibold text-primary-content shadow-accent transition-colors hover:bg-primary/90 disabled:opacity-50"
          >
            {busy ? t("plans.creating") : t("plans.create")}
          </button>
        </div>
      </form>
    </div>
  );
}
