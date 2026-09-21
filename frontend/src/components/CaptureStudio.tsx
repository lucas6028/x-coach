import { useState } from "react";
import ComplexitySelector from "./ComplexitySelector";
import RecordPanel from "./RecordPanel";
import UploadDropzone from "./UploadDropzone";
import { useI18n } from "../lib/i18n";
import { useIsMobile } from "../lib/useIsMobile";
import { loadAnalysisTier, saveAnalysisTier, type PoseTier } from "../lib/poseTier";

type Mode = "upload" | "record";

// The pre-analysis capture screen: pick upload vs live record and the analysis tier, then hand the
// resulting video blob up. Extraction + the API call happen in the parent (App.runPoseAnalysis).
export default function CaptureStudio({
  onBlob,
  busy,
  progress,
  onError,
  movement,
  tier: controlledTier,
  initialMode = "upload",
}: {
  onBlob: (blob: Blob, tier: PoseTier) => void;
  busy: boolean;
  progress: number;
  onError?: (msg: string) => void;
  /** The movement the user selected upstream, forwarded so the upload prompt names what is
   *  actually being uploaded. Required rather than defaulted: a default is how the hardcoded
   *  "squat" copy survived unnoticed next to a movement selector. */
  movement: string;
  /** When supplied, the tier is owned upstream (the studio's page header carries the picker in
   *  the reference design) and the panel's own selector is hidden — two controls for one setting
   *  is how they drift apart. Omitted, the panel stays self-contained. */
  tier?: PoseTier;
  /** Which tab the panel opens on. The seam exists so a link can point at the camera: the
   *  movement detail page offers "Record live" and "Upload video" as two separate cards, and
   *  without this both would land on the dropzone. Initial state only — the tabs own it after. */
  initialMode?: Mode;
}) {
  const { t } = useI18n();
  const mobile = useIsMobile();
  const [mode, setMode] = useState<Mode>(initialMode);
  const [ownTier, setOwnTier] = useState<PoseTier>(() => loadAnalysisTier());
  const controlled = controlledTier !== undefined;
  const tier = controlledTier ?? ownTier;

  const setTierPersist = (t: PoseTier) => {
    setOwnTier(t);
    saveAnalysisTier(t);
  };

  if (busy) {
    return (
      <div className="flex flex-col items-center gap-3 py-10">
        <p className="text-sm text-muted">{t("capture.progress", { pct: Math.round(progress * 100) })}</p>
        <div className="h-2 w-64 overflow-hidden rounded-full bg-content/10">
          <div className="h-full bg-primary transition-[width]" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      {/* One control strip: input mode on the left, the precision picker on the right. */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div role="tablist" className="flex gap-2">
          {(["upload", "record"] as Mode[]).map((m) => (
            <button
              key={m}
              role="tab"
              aria-selected={mode === m}
              onClick={() => setMode(m)}
              className={`rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                mode === m
                  ? "bg-[#232535] text-white"
                  : "bg-[#f5f6fb] text-[#59648f] hover:text-[#1e2142]"
              }`}
            >
              {t(m === "upload" ? "capture.upload" : "capture.record")}
            </button>
          ))}
        </div>

        {!controlled && <ComplexitySelector value={tier} onChange={setTierPersist} />}
      </div>

      {mode === "upload" ? (
        <UploadDropzone onFile={(file) => onBlob(file, tier)} movement={movement} />
      ) : (
        // On a phone the camera goes fullscreen — the header, this page's copy and the bottom nav
        // otherwise leave the preview half off-screen. Keyed on `mobile` because the fullscreen
        // and inline layouts are different trees: a flip must restart the camera, not strand it.
        <RecordPanel
          key={mobile ? "fullscreen" : "inline"}
          fullscreen={mobile}
          onClose={() => setMode("upload")}
          onRecorded={(blob) => onBlob(blob, tier)}
          onError={(msg) => {
            setMode("upload");
            onError?.(msg);
          }}
        />
      )}
    </div>
  );
}
