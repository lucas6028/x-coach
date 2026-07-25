import { Brain, FilmSlate, Graph, PersonSimpleRun, Sparkle, WarningCircle, type Icon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { movementLabel, useI18n } from "../lib/i18n";
import type { AnalyzableMovement } from "../lib/movements";
import UploadDropzone from "./UploadDropzone";
import { LumenLoader } from "./LumenLoader";

interface Props {
  onFile: (file: File) => void;
  onOpenLibrary: () => void;
  loading: boolean;
  statusMsg: string;
  error: string;
  movements: AnalyzableMovement[];
  movement: string;
  onMovementChange: (movement: string) => void;
  /** Non-empty when the requested movement is KNOWN not to be analyzable; the dropzone stays
   *  hidden. Empty while the catalog is still in flight. */
  movementError: string;
  /** False until GET /api/movements settles. The dropzone waits, so a slow network cannot let
   *  someone upload against a movement we have not confirmed. */
  movementsLoaded: boolean;
}

const STEPS: { Icon: Icon; titleKey: string; bodyKey: string }[] = [
  { Icon: PersonSimpleRun, titleKey: "demo.get1.title", bodyKey: "demo.get1.body" },
  { Icon: Brain, titleKey: "demo.get2.title", bodyKey: "demo.get2.body" },
  { Icon: Graph, titleKey: "demo.get3.title", bodyKey: "demo.get3.body" },
];

// The demo's pre-analysis onboarding. Theme-aware (uses semantic tokens so it
// follows the app's light/dark toggle). Asymmetric split: actions on the left,
// an expectation-setting "what comes back" panel on the right.
export default function DemoIntro({
  onFile,
  onOpenLibrary,
  loading,
  statusMsg,
  error,
  movements,
  movement,
  onMovementChange,
  movementError,
  movementsLoaded,
}: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="mx-auto flex min-h-full max-w-5xl flex-col justify-start gap-8 px-5 py-8 sm:gap-12 sm:px-6 sm:py-14 lg:flex-row lg:items-center lg:justify-center lg:gap-16"
      >
        {/* Left: message + actions */}
        <div className="lg:flex-1">
          <h2 className="font-display text-3xl font-bold leading-tight tracking-tight text-content md:text-4xl">
            {t("demo.heading", { movement: movementLabel(t, movement) })}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-muted">{t("demo.sub")}</p>

          <div className="mt-8 max-w-md">
            <div className="mb-4 flex items-center gap-2">
              <label htmlFor="movement-select" className="text-sm font-medium text-muted">
                {t("studio.movement")}
              </label>
              <select
                id="movement-select"
                value={movement}
                onChange={(e) => onMovementChange(e.target.value)}
                className="rounded-lg border border-border-dark bg-surface px-2.5 py-1.5 text-sm text-content"
              >
                {movements.map((m) => (
                  <option key={m.name} value={m.name}>
                    {movementLabel(t, m.name)}
                  </option>
                ))}
                {/* Keep an unanalyzable URL-supplied movement visible rather than silently
                    snapping the control to something the user did not choose. */}
                {!movements.some((m) => m.name === movement) && (
                  <option value={movement}>{movementLabel(t, movement)}</option>
                )}
              </select>
              {movements.find((m) => m.name === movement)?.validated === false && (
                <span
                  title={t("movements.betaNote")}
                  className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-warning ring-1 ring-warning/40"
                >
                  {t("movements.beta")}
                </span>
              )}
            </div>

            {movementError ? (
              <p className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-content">
                {movementError}
              </p>
            ) : loading || !movementsLoaded ? (
              // Analysis waiting state: Lumen takes over the upload target and narrates the
              // pipeline (read pose → check mechanics → surface the reason) while it runs. The
              // loader carries its own navy stage, so no wrapper card is needed. The same loader
              // also covers the brief window before GET /api/movements settles, so the dropzone
              // never appears against an unconfirmed movement.
              <LumenLoader variant="scan" caption={statusMsg} />
            ) : (
              <UploadDropzone onFile={onFile} movement={movement} />
            )}

            <div className="my-4 flex items-center gap-3 text-[11px] uppercase tracking-wider text-faint">
              <span className="h-px flex-1 bg-border-dark" />
              {t("demo.or")}
              <span className="h-px flex-1 bg-border-dark" />
            </div>

            <button
              onClick={onOpenLibrary}
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-2xl border border-border-dark bg-content/[0.02] px-5 py-3.5 text-sm font-medium text-content transition-colors hover:bg-content/[0.05] active:scale-[0.99] disabled:opacity-50"
            >
              <FilmSlate size={18} weight="duotone" className="text-primary" />
              {t("demo.sampleBtn")}
            </button>

            {error && (
              <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-danger/30 bg-danger/[0.06] p-3.5 text-sm text-danger">
                <WarningCircle size={18} className="shrink-0" />
                <div className="min-w-0">
                  <p className="font-medium">{t("demo.errorTitle")}</p>
                  <p className="mt-0.5 break-words text-danger/80">{error}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: what the demo returns — a dashboard-style section panel: a faintly
            tinted icon-header strip over a divided list, lifted by a soft card shadow. */}
        <div className="lg:w-80 lg:shrink-0">
          <div className="overflow-hidden rounded-2xl border border-border-dark bg-surface-dark shadow-card">
            <div className="flex items-center gap-2.5 border-b border-border-dark bg-content/[0.02] px-5 py-4">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <Sparkle size={18} weight="duotone" />
              </span>
              <p className="font-display text-sm font-semibold text-content">{t("demo.getTitle")}</p>
            </div>
            <div className="divide-y divide-border-dark">
              {STEPS.map((s) => (
                <div key={s.titleKey} className="flex items-start gap-4 p-5">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                    <s.Icon size={22} weight="duotone" />
                  </span>
                  <div className="min-w-0">
                    <p className="font-medium text-content">{t(s.titleKey)}</p>
                    <p className="mt-1 text-sm leading-snug text-muted">{t(s.bodyKey)}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
