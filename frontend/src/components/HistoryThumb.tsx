import { useEffect, useState } from "react";
import { PersonSimpleRun } from "@phosphor-icons/react";

/**
 * A history card's media tile: the upload's captured frame, falling back to the movement icon.
 *
 * Fills its container at a fixed 3:4 portrait ratio. The ratio is not arbitrary — captured
 * frames come from phone clips held upright (720x1280, stored at a 480px longest edge), so a
 * landscape tile would crop away the top and bottom of the very body the frame exists to show.
 * `object-cover` still centre-crops a landscape clip, which is the rarer case and loses less.
 *
 * The fallback covers three cases with one branch — a card from before thumbnails existed, a
 * capture that failed in the browser, and a signed URL that 404s because the object is not
 * there. Probing for existence before rendering would cost a request per card to avoid an
 * occasional broken image, so the `onError` handler carries it instead.
 *
 * `failed` is reset whenever `src` changes: signed URLs expire and a history page re-fetches
 * them in batches, so the same card can be handed a working URL after an earlier one 404'd.
 * Without the reset the component keeps the fallback icon forever — React reuses the instance
 * across renders, so the failure of a URL that no longer exists would outlive it.
 */
export default function HistoryThumb({ src }: { src?: string }) {
  const [failed, setFailed] = useState(false);

  useEffect(() => setFailed(false), [src]);

  if (!src || failed) {
    return (
      <span className="flex aspect-[3/4] w-full items-center justify-center bg-primary/10 text-primary">
        <PersonSimpleRun size={44} weight="duotone" />
      </span>
    );
  }
  return (
    <img
      src={src}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
      className="aspect-[3/4] w-full bg-content/5 object-cover"
    />
  );
}
