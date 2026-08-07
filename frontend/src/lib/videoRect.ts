/**
 * Where an `object-contain` video actually renders inside its box.
 *
 * The clip keeps its own aspect inside a fixed-ratio stage (see VideoPanel), so it is letterboxed
 * or pillarboxed with dead space on two sides. Normalized [0,1] pose landmarks are relative to the
 * VIDEO, not the box — mapping them onto the full box drifts the skeleton off the body, and the
 * further the two aspects diverge the worse it gets (a portrait phone clip in the landscape stage
 * is the extreme case).
 *
 * Extracted so the canvas overlay and the DOM-positioned fault chips share ONE definition: they
 * draw over the same pixels, and two copies of this arithmetic would let the chips drift off the
 * joints the skeleton is highlighting.
 */
export interface ContentRect {
  /** Offset of the rendered video's left edge inside the box. */
  offsetX: number;
  /** Offset of the rendered video's top edge inside the box. */
  offsetY: number;
  width: number;
  height: number;
}

export function containRect(boxWidth: number, boxHeight: number, videoAspect: number): ContentRect {
  // A zero/NaN aspect (metadata not in yet) would produce a degenerate rect; fall back to the box
  // so callers render something sane rather than collapsing everything into a corner.
  if (!Number.isFinite(videoAspect) || videoAspect <= 0) {
    return { offsetX: 0, offsetY: 0, width: boxWidth, height: boxHeight };
  }
  const boxAspect = boxWidth / boxHeight;
  if (boxAspect > videoAspect) {
    // Pillarboxed: full height, bars left and right.
    const height = boxHeight;
    const width = height * videoAspect;
    return { offsetX: (boxWidth - width) / 2, offsetY: 0, width, height };
  }
  // Letterboxed: full width, bars top and bottom.
  const width = boxWidth;
  const height = width / videoAspect;
  return { offsetX: 0, offsetY: (boxHeight - height) / 2, width, height };
}
