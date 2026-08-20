import type { Analysis } from "../../api";
import { FAULT_LANDMARKS } from "../../lib/pose";
import { containRect } from "../../lib/videoRect";
import { faultLabel, useI18n } from "../../lib/i18n";

const VIS_THRESHOLD = 0.4;
// At most two at once. The mock shows two and the clip is the subject — a third chip starts
// covering the athlete the labels exist to explain.
const SHOWN = 2;

interface Props {
  analysis: Analysis;
  videoRef: React.RefObject<HTMLVideoElement>;
  /** The playhead, in seconds. Already driven at frame rate by VideoPanel's rAF loop, so these
   *  chips ride that update instead of starting a second one. */
  time: number;
}

/**
 * The mock's labelled callouts pinned to the body — "Knees caving inward" at the knees, and so on.
 *
 * A chip is anchored to the centroid of the landmarks its fault implicates (`FAULT_LANDMARKS`, the
 * same map that decides which limbs SkeletonOverlay reddens), mapped through `containRect` so it
 * lands on the video's rendered pixels rather than the letterboxed box around them.
 *
 * Only faults active at the current frame render, so the callouts appear and clear with the fault
 * they name. A fault whose landmarks are all below the visibility threshold is skipped rather than
 * pinned to a guessed position: a label floating over nothing is worse than no label.
 */
export default function FaultChips({ analysis, videoRef, time }: Props) {
  const { t } = useI18n();
  const fps = analysis.pose.fps || 30;
  const frames = analysis.pose.frames;
  if (!frames.length) return null;

  const idx = Math.min(frames.length - 1, Math.max(0, Math.round(time * fps)));
  const lm = frames[idx]?.lm;
  if (!lm) return null;

  const video = videoRef.current;
  const vw = video?.videoWidth || analysis.metadata.width || analysis.pose.width || 1;
  const vh = video?.videoHeight || analysis.metadata.height || analysis.pose.height || 1;
  // Percentages, so the chips scale with the card and need no resize observer.
  const rect = containRect(100, 100, vw / vh);

  const active = analysis.detections.filter(
    (d) => idx >= d.start_frame && idx <= d.end_frame
  );

  const chips: { key: string; label: string; left: number; top: number }[] = [];
  for (const d of active) {
    if (chips.length === SHOWN) break;
    const points = (FAULT_LANDMARKS[d.fault_id] ?? [])
      .map((i) => lm[i])
      .filter((p): p is [number, number, number] => !!p && p[2] >= VIS_THRESHOLD);
    if (!points.length) continue;
    const cx = points.reduce((s, p) => s + p[0], 0) / points.length;
    const cy = points.reduce((s, p) => s + p[1], 0) / points.length;
    chips.push({
      key: d.fault_id,
      label: faultLabel(t, d.fault_name),
      left: rect.offsetX + cx * rect.width,
      top: rect.offsetY + cy * rect.height,
    });
  }
  if (!chips.length) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-20" aria-hidden="true">
      {chips.map((c, i) => (
        <span
          key={c.key}
          // Alternate the horizontal anchor so two chips on the same limb group do not stack on
          // top of each other, and clamp into the card so one near an edge stays readable.
          style={{
            left: `${Math.min(78, Math.max(4, c.left + (i % 2 ? 6 : -34)))}%`,
            top: `${Math.min(82, Math.max(6, c.top - 4))}%`,
          }}
          className="absolute max-w-[42%] rounded-2xl bg-[#ef5a5a]/92 px-3 py-2 text-[11px] font-semibold leading-tight text-white shadow-[0_8px_20px_rgba(20,24,60,0.35)]"
        >
          {c.label}
        </span>
      ))}
    </div>
  );
}
