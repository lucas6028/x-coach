import { motion, useReducedMotion } from "motion/react";
import { useI18n } from "../lib/i18n";
import UploadDropzone from "./UploadDropzone";

interface Props {
  onFile: (file: File) => void;
  onOpenLibrary: () => void;
  loading: boolean;
  statusMsg: string;
  error: string;
}

const STEPS: { icon: string; titleKey: string; bodyKey: string }[] = [
  { icon: "directions_run", titleKey: "demo.get1.title", bodyKey: "demo.get1.body" },
  { icon: "psychology", titleKey: "demo.get2.title", bodyKey: "demo.get2.body" },
  { icon: "hub", titleKey: "demo.get3.title", bodyKey: "demo.get3.body" },
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
        className="mx-auto flex min-h-full max-w-5xl flex-col justify-center gap-12 px-6 py-14 lg:flex-row lg:items-center lg:gap-16"
      >
        {/* Left: message + actions */}
        <div className="lg:flex-1">
          <h2 className="font-display text-3xl font-bold leading-tight tracking-tight text-content md:text-4xl">
            {t("demo.heading")}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-muted">{t("demo.sub")}</p>

          <div className="mt-8 max-w-md">
            <UploadDropzone onFile={onFile} loading={loading} statusMsg={statusMsg} />

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
              <span className="material-symbols-outlined text-lg text-primary">video_library</span>
              {t("demo.sampleBtn")}
            </button>

            {error && (
              <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-danger/30 bg-danger/[0.06] p-3.5 text-sm text-danger">
                <span className="material-symbols-outlined text-lg leading-none">error</span>
                <div className="min-w-0">
                  <p className="font-medium">{t("demo.errorTitle")}</p>
                  <p className="mt-0.5 break-words text-danger/80">{error}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: what the demo returns */}
        <div className="lg:w-80 lg:shrink-0">
          <p className="mb-3 font-mono text-xs uppercase tracking-wider text-faint">{t("demo.getTitle")}</p>
          <div className="divide-y divide-border-dark overflow-hidden rounded-2xl border border-border-dark bg-surface-dark">
            {STEPS.map((s) => (
              <div key={s.titleKey} className="flex items-start gap-4 p-5">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <span className="material-symbols-outlined">{s.icon}</span>
                </span>
                <div className="min-w-0">
                  <p className="font-medium text-content">{t(s.titleKey)}</p>
                  <p className="mt-1 text-sm leading-snug text-muted">{t(s.bodyKey)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
