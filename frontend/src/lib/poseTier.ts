// Which MediaPipe model tier to run. The LIVE overlay is always Lite (perf, visual only);
// the ANALYSIS extraction (offline, from the recorded/uploaded blob) is user-selectable.
export type PoseTier = "lite" | "full" | "heavy";

const MODEL_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker";
export const MODEL_URL: Record<PoseTier, string> = {
  lite: `${MODEL_BASE}/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  full: `${MODEL_BASE}/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
  heavy: `${MODEL_BASE}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
};

// Live overlay: never anything but Lite.
export const LIVE_OVERLAY_TIER: PoseTier = "lite";
// Analysis default = "heavy": Task 1 measured only 50% Lite==Heavy squat verdict agreement
// (notes/mediapipe_complexity_squat_verdicts.md), far below the 95% bar, so Lite would materially
// change the fault verdicts. The live overlay stays Lite (LIVE_OVERLAY_TIER) — this only governs
// the offline analysis extraction.
export const DEFAULT_ANALYSIS_TIER: PoseTier = "heavy";

const KEY = "xcoach.poseTier";
const TIERS: readonly PoseTier[] = ["lite", "full", "heavy"];

export function loadAnalysisTier(): PoseTier {
  const raw = localStorage.getItem(KEY);
  return (TIERS as readonly string[]).includes(raw ?? "") ? (raw as PoseTier) : DEFAULT_ANALYSIS_TIER;
}

export function saveAnalysisTier(tier: PoseTier): void {
  localStorage.setItem(KEY, tier);
}
