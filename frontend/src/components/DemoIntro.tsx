import { Brain, FilmSlate, Graph, PersonSimpleRun, Sparkle, WarningCircle, type Icon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { movementLabel, useI18n } from "../lib/i18n";
import CaptureStudio from "./CaptureStudio";
import type { PoseTier } from "../lib/poseTier";
import { LumenLoader } from "./LumenLoader";

interface Props {
  onBlob: (blob: Blob, tier: PoseTier) => void;
  onError: (msg: string) => void;
  onOpenLibrary: () => void;
  loading: boolean;
  statusMsg: string;
  error: string;
  movement: string;
  /** Non-empty when the requested movement is KNOWN not to be analyzable; the dropzone stays
   *  hidden. Empty while the catalog is still in flight. */
  movementError: string;
  /** False until GET /api/movements settles. The dropzone waits, so a slow network cannot let
   *  someone upload against a movement we have not confirmed. */
  movementsLoaded: boolean;
  /** The extraction tier, owned by the studio header (StudioTitleBar) since the reference design
   *  moved every analysis control up into the page header. */
  tier: PoseTier;
}

const STEPS: { Icon: Icon; titleKey: string; bodyKey: string }[] = [
  { Icon: PersonSimpleRun, titleKey: "demo.get1.title", bodyKey: "demo.get1.body" },
  { Icon: Brain, titleKey: "demo.get2.title", bodyKey: "demo.get2.body" },
  { Icon: Graph, titleKey: "demo.get3.title", bodyKey: "demo.get3.body" },
];

// The studio's pre-analysis state, in the reference palette: actions on the left, an
// expectation-setting "what comes back" panel on the right. The movement selector that used to
// sit above the dropzone now lives in the page header, so there is exactly one of it.
export default function DemoIntro({
  onBlob,
  onError,
  onOpenLibrary,
  loading,
  statusMsg,
  error,
  movement,
  movementError,
  movementsLoaded,
  tier,
}: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();

  return (
    <div className="mt-2 flex-1 overflow-y-auto scrollbar-thin">
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="mx-auto flex min-h-full max-w-5xl flex-col justify-start gap-8 px-1 py-6 sm:gap-12 sm:py-10 lg:flex-row lg:items-center lg:justify-center lg:gap-16"
      >
        {/* Left: message + actions */}
        <div className="lg:flex-1">
          <h2 className="font-display text-3xl font-bold leading-tight tracking-tight text-[#1e2142] md:text-4xl">
            {t("demo.heading", { movement: movementLabel(t, movement) })}
          </h2>
          <p className="mt-4 max-w-md leading-relaxed text-[#59648f]">{t("demo.sub")}</p>

          <div className="mt-8 max-w-md">
            {movementError ? (
              <p className="rounded-2xl border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-[#1e2142]">
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
              // Extraction-progress bar is a deferred follow-up (SP1 Task 9 note): the studio is
              // never busy here — DemoIntro's own `loading` branch above already covers the
              // whole waiting state with Lumen.
              // `movement` is forwarded so the dropzone inside names what is being uploaded
              // ("Drop a Push-up video…") rather than a hardcoded squat.
              <CaptureStudio
                onBlob={onBlob}
                busy={false}
                progress={0}
                onError={onError}
                movement={movement}
                tier={tier}
              />
            )}

            <div className="my-4 flex items-center gap-3 text-[11px] uppercase tracking-wider text-[#63709f]">
              <span className="h-px flex-1 bg-[#ebeaf6]" />
              {t("demo.or")}
              <span className="h-px flex-1 bg-[#ebeaf6]" />
            </div>

            <button
              onClick={onOpenLibrary}
              disabled={loading}
              className="glass-control flex w-full items-center justify-center gap-2 rounded-2xl px-5 py-3.5 text-sm font-medium text-[#1e2142] transition-colors active:scale-[0.99] disabled:opacity-50"
            >
              <FilmSlate size={18} weight="duotone" className="text-primary" />
              {t("demo.sampleBtn")}
            </button>

            {error && (
              <div className="mt-4 flex items-start gap-2.5 rounded-2xl border border-[#ffe0e0] bg-[#fff5f5] p-3.5 text-sm text-[#e05252]">
                <WarningCircle size={18} className="shrink-0" />
                <div className="min-w-0">
                  <p className="font-medium">{t("demo.errorTitle")}</p>
                  <p className="mt-0.5 break-words opacity-80">{error}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: what the demo returns — the reference's tinted icon-header strip over a
            divided list, lifted by a soft card shadow. */}
        <div className="lg:w-80 lg:shrink-0">
          <div className="glass-panel xc-pop overflow-hidden rounded-[18px]">
            <div className="flex items-center gap-2.5 border-b border-white/70 bg-white/50 px-5 py-4">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#f0eaff] text-primary">
                <Sparkle size={18} weight="duotone" />
              </span>
              <p className="font-display text-sm font-semibold text-[#1e2142]">
                {t("demo.getTitle")}
              </p>
            </div>
            <div className="divide-y divide-white/70">
              {STEPS.map((s) => (
                <div key={s.titleKey} className="flex items-start gap-4 p-5">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#f0eaff] text-primary">
                    <s.Icon size={22} weight="duotone" />
                  </span>
                  <div className="min-w-0">
                    <p className="font-medium text-[#1e2142]">{t(s.titleKey)}</p>
                    <p className="mt-1 text-sm leading-snug text-[#59648f]">{t(s.bodyKey)}</p>
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
