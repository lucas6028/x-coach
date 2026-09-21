// Progress + rough time-remaining for the upload analysis wait (App.runPoseAnalysis). The wait has
// three phases and only two of them report anything real:
//   extract — in-browser pose extraction; real progress from extractPoseWithReps' onProgress.
//   upload  — the multipart POST; real byte progress from XMLHttpRequest.upload.
//   server  — /api/analyze/pose is one synchronous request with no progress channel, so this phase
//             is an asymptotic fill that approaches but never reaches 100%, and it deliberately
//             reports NO seconds: a countdown there would be invented, not estimated.

export type AnalysisPhase = "extract" | "upload" | "server";

export interface AnalysisProgress {
  phase: AnalysisPhase;
  /** Overall 0..1 across all three phases. Never decreases within one run. */
  fraction: number;
  /** Whole seconds left, or null when there is no honest estimate (too early, or server phase). */
  remainingSec: number | null;
}

// Shares of the overall bar. Extraction dominates the wall clock on every device measured so far;
// the split only has to be roughly right, because the seconds come from measured rates, not from it.
export const PHASE_SHARE: Record<AnalysisPhase, number> = { extract: 0.7, upload: 0.15, server: 0.15 };
const PHASE_START: Record<AnalysisPhase, number> = {
  extract: 0,
  upload: PHASE_SHARE.extract,
  server: PHASE_SHARE.extract + PHASE_SHARE.upload,
};

// What the phases AFTER the current one are assumed to cost, added to the measured remainder. The
// pose is already extracted, so the server only runs rules + storage — a few seconds.
const UPLOAD_GUESS_SEC = 2;
const SERVER_GUESS_SEC = 3;
// Time constant of the server phase's asymptotic fill: ~63% of its share after this long.
const SERVER_TAU_MS = 4000;
const SERVER_TICK_MS = 250;

// The rate is measured over a sliding window rather than since the start: extraction changes speed
// at the coarse→dense boundary (different model, different frame count behind a fixed 30/70 split),
// and an average over the whole run would keep quoting the old pass's pace.
const RATE_WINDOW_MS = 4000;
// No estimate until the window spans at least this long — the first few frames say nothing.
const MIN_SPAN_MS = 1000;
// Per-frame progress callbacks would re-render the studio dozens of times a second.
const EMIT_INTERVAL_MS = 200;
// Weight of the newest estimate in the displayed seconds; the rest is the previous display.
const SMOOTHING = 0.3;

/** Seconds left for one phase from a sliding window of (time, progress) samples, or null. */
export function windowedRemainingSec(
  samples: readonly { t: number; p: number }[]
): number | null {
  if (samples.length < 2) return null;
  const first = samples[0];
  const last = samples[samples.length - 1];
  const span = last.t - first.t;
  const gained = last.p - first.p;
  if (span < MIN_SPAN_MS || gained <= 0) return null;
  return ((1 - last.p) / gained) * span / 1000;
}

export interface ProgressTracker {
  /** Extraction progress, 0..1. */
  extract(p: number): void;
  /** Upload progress, 0..1. Reaching 1 starts the server phase. */
  upload(p: number): void;
  /** Stops the server-phase timer. Safe to call more than once. */
  stop(): void;
}

export function createProgressTracker(
  emit: (progress: AnalysisProgress) => void,
  now: () => number = () => performance.now()
): ProgressTracker {
  let phase: AnalysisPhase = "extract";
  let samples: { t: number; p: number }[] = [];
  let shownSec: number | null = null;
  let fraction = 0;
  let lastEmit = -Infinity;
  let timer: ReturnType<typeof setInterval> | null = null;

  const publish = (phaseFraction: number, remainingSec: number | null, force: boolean) => {
    // Monotonic on purpose: a bar that steps backwards reads as a failure even when it is only a
    // late progress event from the phase before.
    fraction = Math.max(fraction, PHASE_START[phase] + PHASE_SHARE[phase] * phaseFraction);
    const t = now();
    if (!force && t - lastEmit < EMIT_INTERVAL_MS) return;
    lastEmit = t;
    emit({ phase, fraction, remainingSec });
  };

  const measured = (next: AnalysisPhase, raw: number, tailSec: number) => {
    const p = Math.min(1, Math.max(0, raw));
    const changed = next !== phase;
    if (changed) {
      phase = next;
      samples = [];
    }
    const t = now();
    samples.push({ t, p });
    // Keep one sample older than the window so the span never collapses to nothing.
    while (samples.length > 2 && t - samples[1].t >= RATE_WINDOW_MS) samples.shift();
    const est = windowedRemainingSec(samples);
    if (est !== null) {
      const total = est + tailSec;
      shownSec = shownSec === null ? total : shownSec * (1 - SMOOTHING) + total * SMOOTHING;
    }
    publish(p, shownSec === null ? null : Math.max(1, Math.round(shownSec)), changed);
  };

  const stop = () => {
    if (timer !== null) clearInterval(timer);
    timer = null;
  };

  const startServer = () => {
    if (phase === "server") return;
    phase = "server";
    const start = now();
    publish(0, null, true);
    timer = setInterval(() => {
      publish(1 - Math.exp(-(now() - start) / SERVER_TAU_MS), null, true);
    }, SERVER_TICK_MS);
  };

  return {
    extract: (p) => {
      if (phase === "extract") measured("extract", p, UPLOAD_GUESS_SEC + SERVER_GUESS_SEC);
    },
    upload: (p) => {
      if (phase === "server") return;
      measured("upload", p, SERVER_GUESS_SEC);
      if (p >= 1) startServer();
    },
    stop,
  };
}
