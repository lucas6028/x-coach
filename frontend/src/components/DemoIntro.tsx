import { Brain, FilmSlate, Graph, PersonSimpleRun, Sparkle, WarningCircle, type Icon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { useI18n } from "../lib/i18n";
import UploadDropzone from "./UploadDropzone";
import { LumenLoader } from "./LumenLoader";

interface Props {
  onFile: (file: File) => void;
  onOpenLibrary: () => void;
  loading: boolean;
  statusMsg: string;
  error: string;
}

const STEPS: { Icon: Icon; titleKey: string; bodyKey: string }[] = [
  { Icon: PersonSimpleRun, titleKey: "demo.get1.title", bodyKey: "demo.get1.body" },
  { Icon: Brain, titleKey: "demo.get2.title", bodyKey: "demo.get2.body" },
  { Icon: Graph, titleKey: "demo.get3.title", bodyKey: "demo.get3.body" },
];

// The demo's pre-analysis onboarding. Theme-aware (uses semantic tokens so it
// follows the app's light/dark toggle). Asymmetric split: actions on the left,
// an expectation-setting "what comes back" panel on the right.
export default function DemoIntro({ onFile, onOpenLibrary, loading, statusMsg, error }: Props) {
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
            {t("demo.heading")}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-muted">{t("demo.sub")}</p>

          <div className="mt-8 max-w-md">
            {loading ? (
              // Analysis waiting state: Lumen takes over the upload target and narrates the
              // pipeline (read pose → check mechanics → surface the reason) while it runs. The
              // loader carries its own navy stage, so no wrapper card is needed.
              <LumenLoader variant="scan" caption={statusMsg} />
            ) : (
              <UploadDropzone onFile={onFile} />
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
