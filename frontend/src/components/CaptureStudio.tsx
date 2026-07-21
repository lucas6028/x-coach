import { useState } from "react";
import ComplexitySelector from "./ComplexitySelector";
import RecordPanel from "./RecordPanel";
import UploadDropzone from "./UploadDropzone";
import { loadAnalysisTier, saveAnalysisTier, type PoseTier } from "../lib/poseTier";

type Mode = "upload" | "record";

// The pre-analysis capture screen: pick upload vs live record and the analysis tier, then hand the
// resulting video blob up. Extraction + the API call happen in the parent (App.runPoseAnalysis).
export default function CaptureStudio({
  onBlob,
  busy,
  progress,
  onError,
}: {
  onBlob: (blob: Blob, tier: PoseTier) => void;
  busy: boolean;
  progress: number;
  onError?: (msg: string) => void;
}) {
  const [mode, setMode] = useState<Mode>("upload");
  const [tier, setTier] = useState<PoseTier>(() => loadAnalysisTier());

  const setTierPersist = (t: PoseTier) => {
    setTier(t);
    saveAnalysisTier(t);
  };

  if (busy) {
    return (
      <div className="flex flex-col items-center gap-3 py-10">
        <p className="text-sm text-muted">分析中… {Math.round(progress * 100)}%</p>
        <div className="h-2 w-64 overflow-hidden rounded-full bg-content/10">
          <div className="h-full bg-primary transition-[width]" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <div role="tablist" className="flex gap-2">
        {(["upload", "record"] as Mode[]).map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            onClick={() => setMode(m)}
            className={`rounded-lg px-4 py-2 text-sm font-medium ${
              mode === m ? "bg-primary text-white" : "bg-content/5 text-muted"
            }`}
          >
            {m === "upload" ? "上傳影片" : "即時錄影"}
          </button>
        ))}
      </div>

      <ComplexitySelector value={tier} onChange={setTierPersist} />

      {mode === "upload" ? (
        <UploadDropzone onFile={(file) => onBlob(file, tier)} />
      ) : (
        <RecordPanel
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
