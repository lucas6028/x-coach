// Pure helpers for the NLF "3D view" turntable strip (components/nlf/TurntableViewer.tsx). Kept
// framework-free and i18n-free on purpose: everything here is arithmetic on tile indices or CSS
// values, unit-tested without React, and the two callers (drag handling, keyboard stepping) share
// the same wraparound math instead of each re-deriving it.
//
// A strip is `angles` square tiles side by side (one WebP per key frame): tile 0 is the camera's
// own view, tile i is the mesh rotated i * (360/angles) degrees about the vertical axis.
import type { NlfKeyFrame } from "../api";

/**
 * Step from `startIndex` by the whole tile-steps implied by a horizontal drag of `dxPx` pixels,
 * wrapping around in both directions. Dragging right (`dxPx > 0`) rotates the mesh positively
 * (increasing index), matching a turntable being pushed the way the hand moves.
 */
export function angleIndexFromDrag(
  startIndex: number,
  dxPx: number,
  pxPerStep: number,
  angles: number
): number {
  if (angles <= 0) return 0;
  const steps = Math.round(dxPx / pxPerStep);
  return ((startIndex + steps) % angles + angles) % angles;
}

/** The tile index for the 90-degree side view (a quarter turn from the camera's own view). */
export function sideViewIndex(angles: number): number {
  return Math.floor(angles / 4);
}

/** CSS to show tile `index` of an `angles`-tile strip, each tile rendered at `displayPx` square. */
export function tileBackground(
  index: number,
  angles: number,
  displayPx: number
): { backgroundSize: string; backgroundPosition: string } {
  return {
    backgroundSize: `${angles * displayPx}px ${displayPx}px`,
    backgroundPosition: `-${index * displayPx}px 0`,
  };
}

/** What a key-frame label needs, split out so the i18n/fault-label rendering stays in the
 *  component while the "what kind of moment is this" logic is unit-tested here. */
export type KeyFrameLabelParts =
  | { kind: "rep_bottom"; repIndex: number | null }
  | { kind: "fault_peak"; faultIds: string[] };

export function keyFrameLabelParts(kf: NlfKeyFrame): KeyFrameLabelParts {
  if (kf.kind === "rep_bottom") return { kind: "rep_bottom", repIndex: kf.rep_index };
  return { kind: "fault_peak", faultIds: kf.fault_ids };
}
