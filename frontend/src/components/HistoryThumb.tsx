import { useEffect, useState } from "react";
import { PersonSimpleRun } from "@phosphor-icons/react";

/**
 * A history row's leading tile: the upload's captured frame, falling back to the movement icon.
 *
 * The fallback covers three cases with one branch — a row from before thumbnails existed, a
 * capture that failed in the browser, and a signed URL that 404s because the object is not
 * there. Probing for existence before rendering would cost a request per row to avoid an
 * occasional broken image, so the `onError` handler carries it instead.
 *
 * `failed` is reset whenever `src` changes: signed URLs expire and a history page re-fetches
 * them in batches, so the same row can be handed a working URL after an earlier one 404'd.
 * Without the reset the component keeps the fallback icon forever — React reuses the instance
 * across renders, so the failure of a URL that no longer exists would outlive it.
 */
export default function HistoryThumb({ src }: { src?: string }) {
  const [failed, setFailed] = useState(false);

  useEffect(() => setFailed(false), [src]);

  if (!src || failed) {
    return (
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
        <PersonSimpleRun size={22} weight="duotone" />
      </span>
    );
  }
  return (
    <img
      src={src}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
      className="h-10 w-10 shrink-0 rounded-lg object-cover bg-content/5"
    />
  );
}
