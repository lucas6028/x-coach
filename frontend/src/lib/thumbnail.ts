// One frame of an upload, captured in the browser and sent alongside it so the history page has
// something to show. The pure sizing/timing decisions are exported and unit-tested; the <video>
// and canvas glue below them cannot run under jsdom and is coverage-excluded, matching the split
// in lib/poseExtract.ts.
import { resolveDuration } from "./mediaDuration";

/** Longest edge of the stored thumbnail. A history card renders it at ~40px; 480 covers a
 *  retina card and any future larger use without approaching the backend's 512KB cap. */
export const THUMBNAIL_MAX_EDGE = 480;

/** Where in the clip to grab the frame: a quarter in is usually mid-movement, and past the
 *  black or motion-blurred frames a clip tends to open on. */
const THUMBNAIL_POSITION = 0.25;

const CAPTURE_TIMEOUT_MS = 5000;
const JPEG_QUALITY = 0.8;

/** How close to the requested timestamp counts as "the seek landed". Browsers snap to the
 *  nearest keyframe, so an exact match is not something this may depend on. */
const SEEK_TOLERANCE_S = 0.05;

/** Downscale a frame to fit THUMBNAIL_MAX_EDGE, preserving aspect. Never returns 0 in either
 *  dimension — a 0-width canvas throws on drawImage. */
export function thumbnailSize(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, THUMBNAIL_MAX_EDGE / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

/** The timestamp to seek to. A recorded MediaRecorder clip can report NaN or Infinity for its
 *  length (no Duration element in a live-muxed WebM — see poseExtract.ts); seeking to NaN is a
 *  no-op that would hang the capture, so fall back to the opening frame. */
export function thumbnailTime(duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  return duration * THUMBNAIL_POSITION;
}

/** Per-channel spread at or below which a frame counts as one flat colour. */
const BLANK_SPREAD = 2;

/** Whether an RGBA pixel buffer is one flat colour — a frame that was never decoded.
 *
 *  iPhone Safari recordings stored thumbnails of solid black (every pixel 0, std 0) while the pose
 *  extractor read real frames from the same clip. A real camera frame always carries sensor
 *  noise, so "flat" rather than "dark" is the test: it rejects the failed draw (including the
 *  transparent canvas a draw that did nothing leaves behind) without rejecting a dim gym. */
export function isBlankFrame(data: Uint8ClampedArray): boolean {
  if (data.length < 4) return true;
  for (let channel = 0; channel < 3; channel += 1) {
    let min = 255;
    let max = 0;
    for (let i = channel; i < data.length; i += 4) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (max - min > BLANK_SPREAD) return false;
  }
  return true;
}

/** Reject if `promise` has not settled within `ms`. Exported for test only — it is pure and
 *  DOM-free, so it does not belong behind the `c8 ignore` marker that covers the decode glue. */
export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("thumbnail capture timed out")), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

/* c8 ignore start — <video>/canvas decode glue, unrunnable under jsdom */
/**
 * Grab one frame of `video` as a JPEG blob.
 *
 * Resolves to `null` on ANY failure. A thumbnail is a nicety; a decode problem must never block
 * an analysis, so every error path here is a silent degradation rather than a thrown one.
 */
export async function captureThumbnail(video: Blob): Promise<Blob | null> {
  // Declared out here so `finally` can clean up whatever got as far as being created, but
  // ASSIGNED inside the try: `URL.createObjectURL` and `document.createElement` can themselves
  // throw, and this function's contract is to resolve null on ANY failure. Setup sitting above
  // the try would reject instead — and the only caller awaits this inside its own try, so a
  // rejection would surface as an ANALYSIS failure rather than a missing thumbnail, which is
  // exactly what "a thumbnail must never block an analysis" forbids.
  let url: string | null = null;
  let el: HTMLVideoElement | null = null;
  try {
    url = URL.createObjectURL(video);
    const media = document.createElement("video");
    el = media;
    media.muted = true;
    media.playsInline = true;

    const loaded = new Promise<void>((resolve, reject) => {
      media.onloadedmetadata = () => resolve();
      media.onerror = () => reject(new Error("could not decode the clip"));
    });
    media.src = url;
    await withTimeout(loaded, CAPTURE_TIMEOUT_MS);

    // A recorded clip's duration is not known until probed — reuse the same recovery the pose
    // extractor needs, so both paths behave the same on a live recording.
    const duration = await resolveDuration(media, CAPTURE_TIMEOUT_MS).catch(() => Number.NaN);
    const target = thumbnailTime(duration);

    const seekToTarget = async () => {
      const seeked = new Promise<void>((resolve, reject) => {
        // GUARDED ON POSITION, not just on the event firing. `resolveDuration` rewinds to
        // currentTime = 0 as its last act (both on success and on timeout), and that write can emit
        // a `seeked` that lands after this handler is attached but before our own seek takes
        // effect. Resolving on it would capture the clip's OPENING frame — usually black — which is
        // exactly the frame this whole 25% offset exists to avoid. The recorded-clip path is where
        // resolveDuration does its probe-seek dance, so this is the app's live path, not a corner.
        media.onseeked = () => {
          if (Math.abs(media.currentTime - target) < SEEK_TOLERANCE_S) resolve();
        };
        media.onerror = () => reject(new Error("could not seek the clip"));
      });
      if (Math.abs(media.currentTime - target) >= SEEK_TOLERANCE_S) {
        media.currentTime = target;
        await withTimeout(seeked, CAPTURE_TIMEOUT_MS);
      }
      // else: already at the target (the unusable-duration fallback leaves us at 0). Assigning
      // currentTime the value it already holds fires no `seeked`, so awaiting one would only time out.
    };
    await seekToTarget();

    if (!media.videoWidth || !media.videoHeight) return null;
    const { width, height } = thumbnailSize(media.videoWidth, media.videoHeight);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    const draw = (source: CanvasImageSource): boolean => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(source, 0, 0, width, height);
      return !isBlankFrame(ctx.getImageData(0, 0, width, height).data);
    };
    // Three ways to the frame, cheapest-proven first, and a blank frame is never uploaded: a
    // missing thumbnail falls back in the UI, a black one becomes the player's poster.
    //  1. createImageBitmap — how the pose worker reads frames, which worked on iPhone Safari
    //     recordings whose direct drawImage below came out solid black.
    //  2. drawImage straight off the element — the original path, fine on desktop browsers.
    //  3. Prime the decoder with a brief muted play, then seek back and draw again — WebKit may
    //     not paint a frame for an off-page element that has never played.
    let drawn = false;
    if (typeof createImageBitmap === "function") {
      const bitmap = await createImageBitmap(media).catch(() => null);
      if (bitmap) {
        drawn = draw(bitmap);
        bitmap.close();
      }
    }
    if (!drawn) drawn = draw(media);
    if (!drawn) {
      await withTimeout(media.play(), CAPTURE_TIMEOUT_MS).catch(() => undefined);
      await new Promise((resolve) => setTimeout(resolve, 150));
      media.pause();
      await seekToTarget();
      drawn = draw(media);
    }
    if (!drawn) return null;
    return await new Promise<Blob | null>((resolve) =>
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", JPEG_QUALITY)
    );
  } catch {
    return null;
  } finally {
    // Guarded: setup can fail partway, and revoking a URL that was never created — or calling
    // load() on an element that was never made — would throw out of the finally, defeating the
    // catch above.
    if (url) URL.revokeObjectURL(url);
    if (el) {
      el.removeAttribute("src");
      el.load();
    }
  }
}
/* c8 ignore stop */
