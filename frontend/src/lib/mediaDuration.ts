// Duration recovery for live-muxed WebM recordings. Kept independent of MediaPipe so thumbnail
// capture and the app shell do not load the pose/WASM bundle just to inspect a video duration.
const SEEK_PROBE = 1e101;

export interface DurationProbe {
  duration: number;
  currentTime: number;
  addEventListener(type: string, listener: () => void): void;
  removeEventListener(type: string, listener: () => void): void;
}

export function resolveDuration(video: DurationProbe, timeoutMs = 5000): Promise<number> {
  if (Number.isFinite(video.duration) && video.duration > 0) return Promise.resolve(video.duration);
  return new Promise<number>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener("durationchange", check);
      video.removeEventListener("seeked", check);
    };
    function check() {
      if (settled || !Number.isFinite(video.duration) || video.duration <= 0) return;
      settled = true;
      cleanup();
      video.currentTime = 0;
      resolve(video.duration);
    }
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      video.currentTime = 0;
      reject(new Error("Could not read the clip's length."));
    }, timeoutMs);
    video.addEventListener("durationchange", check);
    video.addEventListener("seeked", check);
    video.currentTime = SEEK_PROBE;
  });
}
