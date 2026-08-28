import { Brain, Graph, PersonSimpleRun, Sparkle, WarningCircle, type Icon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { movementLabel, useI18n } from "../lib/i18n";
import CaptureStudio from "./CaptureStudio";
import type { PoseTier } from "../lib/poseTier";
import { LumenLoader } from "./LumenLoader";
import WhileYouWait, { hasMovementGuide } from "./WhileYouWait";

interface Props {
  onBlob: (blob: Blob, tier: PoseTier) => void;
  onError: (msg: string) => void;
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
  /** True when the studio was opened with `?capture=record` — the movement detail page's "Start
   *  recording" card. Opens the capture panel on the camera instead of the dropzone. */
  record?: boolean;
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
  loading,
  statusMsg,
  error,
  movement,
  movementError,
  movementsLoaded,
  tier,
  record,
}: Props) {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  // The guide only replaces the "what comes back" card once an analysis is actually running AND the
  // catalog has confirmed the movement — see the note on the right column below. Hoisted because
  // the column's width depends on it too: the guide is a reading panel and gets more room than the
  // three-line promise it stands in for.
  const showGuide = loading && movementsLoaded && !movementError && hasMovementGuide(movement);

  return (
    <div className="mt-2 flex-1 overflow-y-auto scrollbar-thin">
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        // px-4 below `lg` because that is where the phone shell renders (useIsMobile is the same
        // 1023px query) and its <main> carries no padding of its own — 16px is what every other
        // phone page inset uses. From `lg` up this sits inside the shell's already-padded card,
        // where the old 4px is all that's wanted.
        className="mx-auto flex min-h-full max-w-5xl flex-col justify-start gap-8 px-4 py-6 sm:gap-12 sm:py-10 lg:flex-row lg:items-center lg:justify-center lg:gap-16 lg:px-1"
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
                initialMode={record ? "record" : "upload"}
              />
            )}

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
            divided list, lifted by a soft card shadow. While an analysis is actually running this
            column switches to the movement's own steps and common mistakes (WhileYouWait): the
            promise of what comes back is worth reading before you upload, not during the ~20s wait.
            The `!movementsLoaded` branch is deliberately NOT included — that runs before the
            catalog confirms the movement, so movement-specific content there could name the wrong
            thing. Movements with no authored guide keep the default panel. */}
        <div className={`lg:shrink-0 ${showGuide ? "lg:w-[26rem]" : "lg:w-80"}`}>
          {showGuide ? (
            <WhileYouWait movement={movement} />
          ) : (
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
          )}
        </div>
      </motion.div>
    </div>
  );
}
