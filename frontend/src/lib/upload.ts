// Client-side upload guardrails. These mirror the backend defaults (backend/app/config.py:
// MAX_UPLOAD_BYTES / MAX_UPLOAD_DURATION_S) so an oversized or over-long clip is rejected before it
// is uploaded and analysed — saving the round-trip and the wasted compute. The backend re-checks
// both (size hard, duration from pose metadata), so these are UX, not the security boundary.

export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB
export const MAX_UPLOAD_DURATION_S = 60;

export const MAX_UPLOAD_MB = Math.round(MAX_UPLOAD_BYTES / (1024 * 1024));

/** i18n vars shared by the dropzone hint and the limit-exceeded errors. */
export const uploadLimitVars = { maxMb: MAX_UPLOAD_MB, maxS: MAX_UPLOAD_DURATION_S } as const;

export interface UploadCheck {
  ok: boolean;
  /** i18n key for the failure message (absent when ok). */
  errorKey?: "upload.tooLarge" | "upload.tooLong";
}

// Give up waiting on metadata after this long and let the upload proceed (the server backstops an
// over-long clip). Guards against a browser that never fires loadedmetadata/error for some codec.
const PROBE_TIMEOUT_MS = 15000;

/** Read a video file's duration (seconds) from its metadata, or NaN if it can't be determined. */
export function probeDuration(file: File): Promise<number> {
  return new Promise((resolve) => {
    let url: string;
    try {
      url = URL.createObjectURL(file);
    } catch {
      resolve(NaN); // no object-URL support (or a non-blob input) — can't probe; allow it.
      return;
    }
    const video = document.createElement("video");
    video.preload = "metadata";
    let done = false;
    const finish = (d: number) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      URL.revokeObjectURL(url);
      resolve(d);
    };
    const timer = setTimeout(() => finish(NaN), PROBE_TIMEOUT_MS);
    video.onloadedmetadata = () => finish(video.duration);
    video.onerror = () => finish(NaN);
    video.src = url;
  });
}

/**
 * Validate a file against the size and duration limits before upload. Size is a hard reject;
 * duration is best-effort — if the browser can't read it (NaN/Infinity), we allow the upload and
 * let the server backstop catch an over-long clip.
 */
export async function validateUpload(file: File): Promise<UploadCheck> {
  if (file.size > MAX_UPLOAD_BYTES) return { ok: false, errorKey: "upload.tooLarge" };
  const duration = await probeDuration(file);
  if (Number.isFinite(duration) && duration > MAX_UPLOAD_DURATION_S) {
    return { ok: false, errorKey: "upload.tooLong" };
  }
  return { ok: true };
}
