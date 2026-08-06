import type { Analysis, Detection } from "../api";
import { wasMeasured } from "./quality";

export type FormScoreBand = "excellent" | "good" | "fair" | "poor";

export interface FormScore {
  /** 0–100, rounded. */
  value: number;
  band: FormScoreBand;
}

// How much one fault can cost. A severity-1.0 fault removes a quarter of the score, so a clip
// needs several serious faults before the ring bottoms out.
const PENALTY_PER_SEVERITY = 25;
// The floor. A clip that was measured and came back full of faults is still a clip with signal in
// it; driving the ring to 0 would read as "the analysis failed", which is a different state
// (see `formScore` returning null).
const FLOOR = 20;

/**
 * A single headline number for the analysis, derived from the detections that produced it.
 *
 * THIS IS NOT A BACKEND FIELD. The analyze pipeline returns detections, per-fault severities and
 * clip quality — it has never returned a "form score", and no scoring rubric in this repo has been
 * validated against expert ratings. So this is a presentation-layer summary of data the user can
 * already see on the same screen (the fault cards list every deduction), deliberately built from
 * ONE published rule rather than a tuned model:
 *
 *     score = 100 − 25 × Σ severity,  clamped to [20, 100]
 *
 * `severity` is the detector's own 0–1 output, so a clean rep scores 100, one moderate fault
 * (0.5) scores 88, and two severe faults (0.8 each) score 60.
 *
 * Returns `null` when the clip was never measurable (`wasMeasured` — the SHARED criterion, see
 * src/lib/quality.ts). An unmeasured clip has an empty detection list for the same reason a
 * flawless one does, and printing "100 — Excellent" over a clip nothing was measured on is
 * exactly the fabricated verdict that helper exists to prevent.
 */
export function formScore(analysis: Analysis): FormScore | null {
  if (!wasMeasured(analysis.quality)) return null;
  return scoreFromDetections(analysis.detections);
}

// Split out so the rule is testable without assembling a whole Analysis.
export function scoreFromDetections(detections: Detection[]): FormScore {
  const penalty = detections.reduce(
    (sum, d) => sum + PENALTY_PER_SEVERITY * Math.min(1, Math.max(0, d.severity)),
    0
  );
  const value = Math.round(Math.min(100, Math.max(FLOOR, 100 - penalty)));
  return { value, band: bandFor(value) };
}

export function bandFor(value: number): FormScoreBand {
  if (value >= 90) return "excellent";
  if (value >= 75) return "good";
  if (value >= 55) return "fair";
  return "poor";
}
