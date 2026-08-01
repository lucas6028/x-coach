import { useEffect, useId, useRef } from "react";
import { WarningCircle } from "@phosphor-icons/react";

type Props = {
  open: boolean;
  title: string;
  description: string;
  /** What is being acted on, e.g. "Side Squat · 10:24" — so the user can see they picked the right one. */
  detail?: string;
  confirmLabel: string;
  cancelLabel: string;
  /** The action is in flight: the confirm button locks and the dialog refuses to close. */
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
};

// A confirm dialog for destructive actions. Follows KnowledgeGraphWidget's overlay idiom
// (`fixed inset-0 z-50`, no portal) rather than introducing a portal layer for one dialog.
//
// While `busy`, the dialog refuses to close — Escape and backdrop clicks are ignored — so an
// in-flight delete can't be dismissed into a state where the user can't tell what happened.
export default function ConfirmDialog({
  open,
  title,
  description,
  detail,
  confirmLabel,
  cancelLabel,
  busy = false,
  onConfirm,
  onCancel,
}: Props) {
  const titleId = useId();
  const cancelRef = useRef<HTMLButtonElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  // Escape cancels. Bound only while open, so a closed dialog never swallows the key.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, busy, onCancel]);

  // Focus lands on Cancel — the safe choice for a destructive action — and goes back to whatever
  // opened the dialog once it closes, so the list doesn't dump keyboard users at the top of the page.
  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    cancelRef.current?.focus();
    return () => restoreRef.current?.focus();
  }, [open]);

  if (!open) return null;

  return (
    <div
      // mousedown, not click: a drag that starts inside the card and releases on the backdrop
      // would otherwise read as a backdrop click and discard the dialog.
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !busy) onCancel();
      }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="w-full max-w-sm rounded-2xl border border-border-dark bg-surface-dark p-5 shadow-lg"
      >
        <h2 id={titleId} className="flex items-center gap-2 font-semibold text-content">
          <WarningCircle size={20} weight="duotone" className="shrink-0 text-danger" />
          {title}
        </h2>
        {detail && <p className="mt-2 truncate font-mono text-xs text-muted">{detail}</p>}
        <p className="mt-2 text-sm text-muted">{description}</p>

        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            ref={cancelRef}
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="rounded-xl px-4 py-2 text-sm font-medium text-muted transition-colors hover:bg-content/5 hover:text-content disabled:cursor-not-allowed disabled:opacity-60"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className="rounded-xl bg-red-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-700 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
