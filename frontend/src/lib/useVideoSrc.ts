import { useEffect, useState } from "react";
import { api, type Analysis } from "../api";

/**
 * Where this analysis's video actually lives.
 *
 * Three sources, resolved in order of what is already known:
 *  - a library demo clip is a public file the backend streams directly;
 *  - a fresh upload's presigned URL rides along on the analyze response;
 *  - a history replay has neither, because storing a presigned URL in the row would mean
 *    replaying an expired one — so it re-signs through the ownership-checked endpoint.
 *
 * `null` while resolving and after a failure: callers render the analysis without playback rather
 * than blocking the page on storage.
 *
 * Lives here rather than inside VideoPanel because the phone card resolves the same three cases.
 */
export function useVideoSrc(analysis: Analysis): string | null {
  const { source, video_id: videoId, video_url: videoUrl } = analysis;
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (source === "library") {
      setSrc(api.videoFileUrl(videoId));
      return;
    }
    if (videoUrl) {
      setSrc(videoUrl);
      return;
    }
    let cancelled = false;
    setSrc(null);
    api
      .uploadMedia(videoId)
      .then((media) => {
        if (!cancelled) setSrc(media.video_url);
      })
      .catch(() => {
        if (!cancelled) setSrc(null);
      });
    return () => {
      cancelled = true;
    };
  }, [source, videoId, videoUrl]);

  return src;
}
