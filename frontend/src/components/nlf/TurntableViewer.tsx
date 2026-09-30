import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowsHorizontal } from "@phosphor-icons/react";
import type { NlfKeyFrame } from "../../api";
import { fmtTime } from "../../lib/format";
import { faultLabel, useI18n } from "../../lib/i18n";
import {
  angleIndexFromDrag,
  keyFrameLabelParts,
  sideViewIndex,
  tileBackground,
} from "../../lib/turntable";

// The viewer renders every strip at this CSS box size regardless of the source tile's actual pixel
// size (`tilePx`) — the browser downsamples the (larger) source tile to fit, same as any other
// responsive image. Square, matching the source tiles.
const DISPLAY_PX = 288;
// How many drag pixels move the turntable by one tile step. Chosen so a full box-width drag (about
// 288px) covers roughly a third of a full turn — enough to feel like a drag, not a hair-trigger.
const PX_PER_STEP = 14;

interface Props {
  keyFrames: NlfKeyFrame[];
  angles: number;
  tilePx: number;
  onSeek?: (t_s: number) => void;
  onImageError?: () => void;
}

function keyFrameLabel(t: ReturnType<typeof useI18n>["t"], kf: NlfKeyFrame): string {
  const parts = keyFrameLabelParts(kf);
  if (parts.kind === "rep_bottom") {
    return parts.repIndex != null
      ? t("nlf.keyframe.repBottom", { n: parts.repIndex })
      : t("nlf.keyframe.repBottomUnknown");
  }
  const faults = parts.faultIds.length
    ? parts.faultIds.map((id) => faultLabel(t, id)).join(", ")
    : t("nlf.keyframe.faultUnknown");
  return t("nlf.keyframe.faultPeak", { fault: faults });
}

/**
 * One key frame's rendered turntable strip (24 square tiles side by side), with pointer-drag and
 * arrow-key rotation, a side/front-view shortcut, and the key-frame list that switches strips.
 *
 * `tilePx` is carried on the job (the source tile's pixel size) but is not needed for layout here:
 * the CSS background is always scaled to `DISPLAY_PX`, so the prop exists only so callers can pass
 * the job through without picking it apart — kept in the signature for that reason.
 */
export default function TurntableViewer({ keyFrames, angles, onSeek, onImageError }: Props) {
  const { t } = useI18n();
  const [selected, setSelected] = useState(0);
  const [index, setIndex] = useState(0);
  const dragRef = useRef<{ startX: number; startIndex: number } | null>(null);

  const kf = keyFrames[selected] as NlfKeyFrame | undefined;

  const selectKeyFrame = useCallback(
    (i: number) => {
      setSelected(i);
      setIndex(0); // tile 0 is the camera's own view for THAT frame's strip.
      const next = keyFrames[i];
      if (next) onSeek?.(next.t_s);
    },
    [keyFrames, onSeek]
  );

  const step = useCallback(
    (delta: number) => {
      setIndex((i) => ((i + delta) % angles + angles) % angles);
    },
    [angles]
  );

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    // Best-effort: jsdom (tests) and some non-pointer environments either lack this API or throw
    // for a pointerId it didn't itself dispatch — dragging still works locally via the move/up
    // handlers below without capture, just without the "keep tracking outside the box" guarantee.
    try {
      (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    } catch {
      /* see above */
    }
    dragRef.current = { startX: e.clientX, startIndex: index };
  }, [index]);

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current;
      if (!drag) return;
      const dx = e.clientX - drag.startX;
      setIndex(angleIndexFromDrag(drag.startIndex, dx, PX_PER_STEP, angles));
    },
    [angles]
  );

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    try {
      if ((e.target as HTMLElement).hasPointerCapture?.(e.pointerId)) {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      }
    } catch {
      /* see onPointerDown */
    }
    dragRef.current = null;
  }, []);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        step(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        step(1);
      }
    },
    [step]
  );

  // Reset the retry budget on image-error re-fetches only happens in the parent (Nlf3dPanel); this
  // component just reports the failure once per strip URL it is handed.
  const reportedErrorForRef = useRef<string | null>(null);
  useEffect(() => {
    reportedErrorForRef.current = null;
  }, [kf?.strip_url]);
  const handleImgError = useCallback(() => {
    if (!kf || reportedErrorForRef.current === kf.strip_url) return;
    reportedErrorForRef.current = kf.strip_url;
    onImageError?.();
  }, [kf, onImageError]);

  if (!kf) return null;

  const bg = tileBackground(index, angles, DISPLAY_PX);
  // Degrees for THIS tile, matching the render contract in lib/turntable.ts's module doc: tile i
  // is the mesh rotated i * (360/angles) degrees.
  const degrees = Math.round((index * 360) / angles);

  return (
    <div data-testid="turntable-viewer" className="space-y-3">
      <div
        data-testid="turntable-tile"
        role="slider"
        aria-label={t("nlf.viewerAlt")}
        aria-valuemin={0}
        aria-valuemax={angles - 1}
        aria-valuenow={index}
        aria-valuetext={`${degrees}°`}
        aria-orientation="horizontal"
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onKeyDown={onKeyDown}
        className="mx-auto touch-none select-none rounded-[14px] border border-[#ececf8] bg-[#f5f4fb] outline-none focus-visible:ring-2 focus-visible:ring-primary"
        style={{
          width: DISPLAY_PX,
          height: DISPLAY_PX,
          backgroundImage: `url(${kf.strip_url})`,
          backgroundSize: bg.backgroundSize,
          backgroundPosition: bg.backgroundPosition,
          backgroundRepeat: "no-repeat",
          cursor: "grab",
        }}
      />
      {/* Hidden preload: the only way to learn a presigned strip URL already expired, since the
          background-image above fails silently in CSS. */}
      <img src={kf.strip_url} alt="" className="hidden" onError={handleImgError} />

      <div className="flex items-center justify-center gap-2">
        <button
          type="button"
          onClick={() => setIndex(0)}
          className="rounded-full border border-[#ddd8f5] px-3 py-1 text-[11px] font-semibold text-[#59648f] hover:bg-[#f3f0ff]"
        >
          {t("nlf.frontView")}
        </button>
        <button
          type="button"
          onClick={() => setIndex(sideViewIndex(angles))}
          className="flex items-center gap-1 rounded-full border border-[#ddd8f5] px-3 py-1 text-[11px] font-semibold text-[#59648f] hover:bg-[#f3f0ff]"
        >
          <ArrowsHorizontal size={12} weight="bold" />
          {t("nlf.sideView")}
        </button>
      </div>

      <p className="text-center text-[10.5px] leading-relaxed text-[#8a90ac]">{t("nlf.caption")}</p>

      <ul className="space-y-1.5">
        {keyFrames.map((k, i) => (
          <li key={`${k.t_s}-${i}`}>
            <button
              type="button"
              onClick={() => selectKeyFrame(i)}
              className={`flex w-full items-center justify-between gap-2 rounded-lg border px-3 py-1.5 text-left text-[11px] transition-colors ${
                i === selected
                  ? "border-primary/50 bg-primary/[0.07] text-[#1e2142]"
                  : "border-transparent bg-[#f7f6fc] text-[#59648f] hover:border-[#ddd8f5]"
              }`}
            >
              <span className="min-w-0 truncate font-semibold">{keyFrameLabel(t, k)}</span>
              <span className="shrink-0 font-mono text-[10px] text-[#9aa0b8]">{fmtTime(k.t_s)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
