import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Analysis } from "../api";

/** Everything the player needs to show this analysis's clip without downloading it up front. */
export interface VideoSource {
  /** The clip itself. `null` while resolving and after an unrecoverable failure. */
  src: string | null;
  /** A stored frame to paint while the clip is still just a URL. `null` when there is none. */
  poster: string | null;
  /**
   * Re-sign and swap in a fresh URL after the element reported an error.
   *
   * This exists because of `preload="metadata"`: the bytes are now fetched when the user presses
   * play, which may be an hour after the page resolved the URL — past `DEFAULT_URL_TTL`. The old
   * eager-download behaviour hid that, because the fetch happened immediately on mount.
   */
  recover: () => void;
}

/** How many times one clip may be re-signed before the player gives up.
 *
 * Bounded, because `<video>` fires `error` for a deleted object and a corrupt file too, not just
 * an expired signature — and those repeat forever. Two is enough for the case this is for (one
 * expiry, one to cover a signature minted in the same second it lapsed). */
const MAX_RECOVERIES = 2;

/**
 * Where this analysis's video actually lives.
 *
 * Three sources, resolved in order of what is already known:
 *  - a library demo clip is a public file the backend streams directly;
 *  - a fresh upload's presigned URL rides along on the analyze response;
 *  - a history replay has neither, because storing a presigned URL in the row would mean
 *    replaying an expired one — so it re-signs through the ownership-checked endpoint.
 *
 * `src` is `null` while resolving and after a failure: callers render the analysis without
 * playback rather than blocking the page on storage.
 *
 * Lives here rather than inside VideoPanel because the phone card resolves the same three cases.
 */
export function useVideoSrc(analysis: Analysis): VideoSource {
  const {
    source,
    video_id: videoId,
    video_url: videoUrl,
    thumbnail_url: thumbnailUrl,
    analysis_id: analysisId,
  } = analysis;
  // Whether there is anywhere to re-sign FROM. `/api/uploads/{id}/url` resolves the storage key
  // through the caller's own JWT, so it can only answer for a clip that has a persisted `videos`
  // row owned by a signed-in user. Two analyses never do:
  //  - an anonymous upload, which is deliberately never persisted (`analysis_id` absent);
  //  - a signed-in upload whose persist failed (`analysis_id` explicitly null), whose objects are
  //    orphaned with no row pointing at them.
  // Both would get a 401/404, and asking anyway would replace a still-visible poster with an
  // empty player. A history replay has no `video_url` at all and by construction came from a row,
  // so it must always be allowed to sign.
  //
  // KNOWN LIMITATION: an anonymous upload left unplayed for longer than DEFAULT_URL_TTL therefore
  // cannot be played at all. Accepted rather than fixed, because the fix would be an unauthenticated
  // re-signing route — precisely the IDOR that was deliberately removed from `/api/video-file`
  // (see its docstring). Anonymous uploads are a try-it-now path; watching happens immediately.
  const canReSign = source !== "library" && (!videoUrl || Boolean(analysisId));
  const [src, setSrc] = useState<string | null>(null);
  const [poster, setPoster] = useState<string | null>(null);
  // Bumping this re-runs the effect. It also selects the RE-SIGN branch below: the URL that just
  // failed is the one that came with the analysis, so retrying it would fail identically.
  const [attempt, setAttempt] = useState(0);
  const attemptRef = useRef(0);
  attemptRef.current = attempt;

  // Reset the retry budget when the clip changes — the cap is per-clip, not per-session. Done
  // DURING RENDER rather than in an effect so the resolving effect below already sees `attempt`
  // back at 0; an effect would run it once with the previous clip's exhausted count first, which
  // sends the new clip straight down the re-sign branch for no reason.
  const [resolvedFor, setResolvedFor] = useState(videoId);
  if (resolvedFor !== videoId) {
    setResolvedFor(videoId);
    setAttempt(0);
    attemptRef.current = 0;
  }

  useEffect(() => {
    if (source === "library") {
      // A shared demo asset served straight off disk by the backend, which answers Range
      // requests, so the browser seeks it without a signature. There is no stored frame for it.
      setSrc(api.videoFileUrl(videoId));
      setPoster(null);
      return;
    }
    if (videoUrl && attempt === 0) {
      setSrc(videoUrl);
      setPoster(thumbnailUrl ?? null);
      return;
    }
    let cancelled = false;
    // Clear only on a FIRST resolve, where `src` would otherwise still hold the previous
    // analysis's URL — showing one clip under another analysis's verdict. On a recovery the
    // element is already displaying this clip, so blanking it would throw away a working poster
    // and a rendered timeline for the duration of a round trip.
    if (attempt === 0) setSrc(null);
    api
      .uploadMedia(videoId)
      .then((media) => {
        if (cancelled) return;
        setSrc(media.video_url);
        setPoster(media.thumbnail_url);
      })
      .catch(() => {
        // Same asymmetry on the failure side: a recovery that could not re-sign leaves the stale
        // URL in place. It is unplayable either way, and keeping it means the viewer still sees
        // the poster frame rather than an empty black stage.
        if (!cancelled && attempt === 0) setSrc(null);
      });
    return () => {
      cancelled = true;
    };
  }, [source, videoId, videoUrl, thumbnailUrl, attempt]);

  const recover = useCallback(() => {
    // No route to a fresh signature — a library clip carries none, and an unpersisted upload has
    // no row to resolve. An `error` on those is a real one, not an expiry.
    if (!canReSign) return;
    if (attemptRef.current >= MAX_RECOVERIES) return;
    setAttempt((n) => n + 1);
  }, [canReSign]);

  return { src, poster, recover };
}
