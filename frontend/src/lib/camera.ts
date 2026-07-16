// Camera acquisition with a timeout, because of one specific field bug: inside the LINE
// (LIFF) in-app browser on iOS, `getUserMedia` can neither resolve NOR reject — it just
// hangs, leaving the caller's spinner up forever (line/line-platform-feedback#98). Racing
// a timer turns that hang into a catchable error so the UI can tell the player to open the
// page in a real browser instead. Plain browsers are unaffected: they answer (or throw)
// long before the timeout.

/** Why the camera could not be acquired (`timeout` ≈ the LIFF-on-iOS hang). */
export type CameraFailure = "unsupported" | "timeout";

export class CameraError extends Error {
  constructor(
    readonly reason: CameraFailure,
    message: string
  ) {
    super(message);
    this.name = "CameraError";
  }
}

const DEFAULT_TIMEOUT_MS = 12_000;

/**
 * `getUserMedia` that always settles: rejects with `CameraError("unsupported")` when the
 * API is missing (some in-app webviews), `CameraError("timeout")` when the browser never
 * answers, and rethrows real answers (e.g. permission denied) untouched. A stream that
 * arrives after the timeout already fired is stopped so its tracks don't leak.
 */
export function getCameraStream(
  constraints: MediaStreamConstraints,
  { timeoutMs = DEFAULT_TIMEOUT_MS }: { timeoutMs?: number } = {}
): Promise<MediaStream> {
  const getUserMedia = navigator.mediaDevices?.getUserMedia?.bind(navigator.mediaDevices);
  if (!getUserMedia) {
    return Promise.reject(
      new CameraError("unsupported", "getUserMedia is not available in this browser.")
    );
  }
  return new Promise<MediaStream>((resolve, reject) => {
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      reject(new CameraError("timeout", `Camera did not answer within ${timeoutMs}ms.`));
    }, timeoutMs);
    getUserMedia(constraints).then(
      (stream) => {
        clearTimeout(timer);
        if (timedOut) {
          // Lost the race — nobody is listening for this stream anymore.
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        resolve(stream);
      },
      (err) => {
        clearTimeout(timer);
        if (!timedOut) reject(err);
      }
    );
  });
}

export interface CameraProbeResult {
  ok: boolean;
  reason: "ok" | CameraFailure | "denied" | "error";
  message: string;
}

/**
 * Try to open (then immediately release) the camera and report what happened — the LIFF
 * diagnostics page uses this to answer "does live camera work inside LINE on this device?".
 */
export async function probeCamera(timeoutMs = 8_000): Promise<CameraProbeResult> {
  try {
    const stream = await getCameraStream({ video: true, audio: false }, { timeoutMs });
    stream.getTracks().forEach((track) => track.stop());
    return { ok: true, reason: "ok", message: "Camera stream opened and released." };
  } catch (err) {
    if (err instanceof CameraError) {
      return { ok: false, reason: err.reason, message: err.message };
    }
    const name = err instanceof DOMException ? err.name : "";
    if (name === "NotAllowedError" || name === "SecurityError") {
      return { ok: false, reason: "denied", message: `${name}: permission was denied.` };
    }
    return { ok: false, reason: "error", message: err instanceof Error ? err.message : String(err) };
  }
}
