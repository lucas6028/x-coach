// Resolve once a <video> has a decoded frame a pose model can read (readyState >=
// HAVE_CURRENT_DATA). On mobile, `video.play()` resolving does NOT guarantee a decoded frame, so
// callers that need one (e.g. a GPU warmup inference) should wait for `loadeddata` — with a
// timeout so a stalled camera never wedges the loading phase.
export function waitForVideoFrame(video: HTMLVideoElement, timeoutMs = 3000): Promise<void> {
  if (video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve) => {
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      video.removeEventListener("loadeddata", done);
      resolve();
    };
    const timer = setTimeout(done, timeoutMs);
    video.addEventListener("loadeddata", done);
  });
}
