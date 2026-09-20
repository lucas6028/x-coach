import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { WarningCircle, X } from "@phosphor-icons/react";
import { api, type Checkin, type CheckinCreate } from "../../api";
import { useI18n } from "../../lib/i18n";

const PAIN_SCALE = Array.from({ length: 11 }, (_, i) => i); // 0..10
const RPE_SCALE = Array.from({ length: 11 }, (_, i) => i); // 0..10

interface Props {
  open: boolean;
  planId: string;
  planItemId?: string;
  analysisId?: string;
  /** The already-localized movement name, for the "How did X feel today?" intro line. Omitted
   *  entirely when the caller has no single exercise in view (e.g. a plan-level prompt). */
  movementLabel?: string;
  onClose: () => void;
  onSubmitted?: (checkin: Checkin) => void;
}

type SubmitState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "error"; message: string }
  | { kind: "flagged"; checkin: Checkin };

// Patient-reported status after a training session: pain (required), RPE and a note (both
// optional). Portalled to <body> like SettingsDialog and ConfirmDialog need to be when rendered
// under AppLayout's `glass-shell` — its backdrop-filter makes that card the containing block for
// `position: fixed` descendants, which would otherwise clip this overlay to the card's bounds.
//
// A FLAGGED response (server-computed — the client never decides this itself) keeps the dialog
// open with fixed, non-advisory copy: the loop is "notice and route to the therapist", not
// "diagnose". Anything else closes the dialog and hands the fresh Checkin to the caller.
export default function CheckinDialog({
  open,
  planId,
  planItemId,
  analysisId,
  movementLabel,
  onClose,
  onSubmitted,
}: Props) {
  const { t } = useI18n();
  const [pain, setPain] = useState<number | null>(null);
  const [rpe, setRpe] = useState<number | null>(null);
  const [note, setNote] = useState("");
  const [state, setState] = useState<SubmitState>({ kind: "idle" });
  const [showRequired, setShowRequired] = useState(false);
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  // A re-open (a different item, or the same one again) starts clean rather than showing the
  // previous attempt's values or its flagged/error state.
  useEffect(() => {
    if (!open) return;
    setPain(null);
    setRpe(null);
    setNote("");
    setState({ kind: "idle" });
    setShowRequired(false);
  }, [open]);

  // Escape closes. Bound only while open, so a closed dialog never swallows the key.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Focus moves into the dialog on open and goes back to whatever opened it on close.
  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    return () => restoreRef.current?.focus();
  }, [open]);

  if (!open) return null;

  const submit = async () => {
    if (pain === null) {
      setShowRequired(true);
      return;
    }
    setState({ kind: "submitting" });
    const body: CheckinCreate = { plan_id: planId, pain_nrs: pain };
    if (planItemId) body.plan_item_id = planItemId;
    if (analysisId) body.analysis_id = analysisId;
    if (rpe !== null) body.rpe = rpe;
    if (note.trim()) body.note = note.trim();
    try {
      const checkin = await api.createCheckin(body);
      if (checkin.flagged) {
        setState({ kind: "flagged", checkin });
      } else {
        onSubmitted?.(checkin);
        onClose();
      }
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const submitting = state.kind === "submitting";
  const flagged = state.kind === "flagged" ? state.checkin : null;

  return createPortal(
    <div
      // mousedown, not click: a drag that starts inside the card and releases on the backdrop
      // would otherwise read as a backdrop click and dismiss the dialog.
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="w-full max-w-sm rounded-2xl border border-border-dark bg-surface-dark p-5 shadow-lg outline-none"
      >
        <div className="flex items-start justify-between gap-3">
          <h2 id={titleId} className="text-base font-semibold text-content">
            {t("checkin.title")}
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label={t("checkin.close")}
            className="rounded-lg p-1 text-muted transition-colors hover:bg-content/5 hover:text-content"
          >
            <X size={18} />
          </button>
        </div>

        {flagged ? (
          // Fixed copy only — no advice generated here, per the care-loop's "notice, don't
          // diagnose" boundary.
          <div className="mt-4 flex flex-col gap-2 rounded-xl border border-danger/30 bg-danger/[0.05] px-3 py-3 text-sm text-danger">
            <p className="flex items-start gap-1.5 font-medium">
              <WarningCircle size={16} weight="duotone" className="mt-px shrink-0" />
              {t("checkin.flaggedTitle")}
            </p>
            <p>{t("checkin.flaggedBody")}</p>
            <ul className="list-disc pl-5">
              {flagged.flag_reasons.map((reason) => (
                <li key={reason}>{t(`checkin.reason.${reason}`)}</li>
              ))}
            </ul>
            <button
              type="button"
              onClick={onClose}
              className="mt-2 self-end rounded-xl bg-content/[0.08] px-4 py-2 text-sm font-medium text-content transition-colors hover:bg-content/[0.12]"
            >
              {t("checkin.close")}
            </button>
          </div>
        ) : (
          <>
            {movementLabel && (
              <p className="mt-1 text-sm text-muted">
                {t("checkin.movementIntro", { movement: movementLabel })}
              </p>
            )}

            <div className="mt-4">
              <p className="text-[13px] font-medium text-content">{t("checkin.painLabel")}</p>
              <div
                role="group"
                aria-label={t("checkin.painLabel")}
                className="mt-2 flex flex-wrap gap-1.5"
              >
                {PAIN_SCALE.map((n) => (
                  <button
                    key={n}
                    type="button"
                    aria-pressed={pain === n}
                    onClick={() => {
                      setPain(n);
                      setShowRequired(false);
                    }}
                    className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold transition-colors ${
                      pain === n
                        ? "bg-primary text-primary-content"
                        : "bg-content/[0.06] text-content hover:bg-content/10"
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
              <div className="mt-1 flex justify-between text-[11px] text-faint">
                <span>{t("checkin.painAnchorNone")}</span>
                <span>{t("checkin.painAnchorWorst")}</span>
              </div>
              {showRequired && (
                <p className="mt-1 text-xs text-danger">{t("checkin.painRequired")}</p>
              )}
            </div>

            <div className="mt-4">
              <label htmlFor="checkin-rpe" className="text-[13px] font-medium text-content">
                {t("checkin.rpeLabel")}
              </label>
              <select
                id="checkin-rpe"
                value={rpe === null ? "" : rpe}
                onChange={(e) => setRpe(e.target.value === "" ? null : Number(e.target.value))}
                className="mt-1.5 h-9 w-full rounded-lg border border-border-dark bg-background px-2 text-[13px] text-content outline-none transition-colors focus:border-primary/50"
              >
                <option value="">—</option>
                {RPE_SCALE.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>

            <div className="mt-4">
              <label htmlFor="checkin-note" className="text-[13px] font-medium text-content">
                {t("checkin.noteLabel")}
              </label>
              <textarea
                id="checkin-note"
                value={note}
                maxLength={500}
                onChange={(e) => setNote(e.target.value)}
                placeholder={t("checkin.notePlaceholder")}
                rows={3}
                className="mt-1.5 w-full resize-none rounded-lg border border-border-dark bg-background px-2 py-1.5 text-[13px] text-content outline-none transition-colors focus:border-primary/50"
              />
            </div>

            <p className="mt-4 text-[11px] leading-relaxed text-faint">
              {t("checkin.disclaimer")}
            </p>

            {state.kind === "error" && (
              <p className="mt-2 flex items-start gap-1.5 text-xs text-danger">
                <WarningCircle size={13} weight="duotone" className="mt-px shrink-0" />
                {t("checkin.error")}
              </p>
            )}

            <div className="mt-5 flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={onClose}
                disabled={submitting}
                className="rounded-xl px-4 py-2 text-sm font-medium text-muted transition-colors hover:bg-content/5 hover:text-content disabled:cursor-not-allowed disabled:opacity-60"
              >
                {t("checkin.skip")}
              </button>
              <button
                type="button"
                onClick={() => void submit()}
                disabled={submitting}
                className="rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {submitting ? t("checkin.submitting") : t("checkin.submit")}
              </button>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body
  );
}
