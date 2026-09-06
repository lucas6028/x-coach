import { useEffect, useState } from "react";

// Tailwind's `lg` — the breakpoint the desktop studio's 8/4 grid is written against.
const MOBILE_QUERY = "(max-width: 1023px)";

/**
 * Whether a CSS media query currently matches, as state.
 *
 * A media QUERY rather than a `lg:hidden` / `hidden lg:block` pair on two trees, because the trees
 * this drives are not interchangeable: the studio's mount a <video> and a skeleton canvas, so
 * rendering both would decode the clip twice and run two rAF loops with only CSS hiding one of
 * them. This picks one. The same reasoning covers the plans coach, which holds a live conversation:
 * two mounted copies would be two threads, and CSS cannot pick which one the user typed into.
 *
 * Defaults to false during SSR-less first paint on a server-rendered-free app is moot, but jsdom
 * has no `matchMedia` in some setups — hence the guard, which keeps tests on the desktop tree
 * unless they opt in.
 *
 * `query` is read on every render but only SUBSCRIBED to when it changes, so a caller passing a
 * literal (all of them, today) never re-subscribes.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia(query).matches
      : false
  );

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

/** Whether the phone layout should render. See `useMediaQuery` for why this is a query. */
export function useIsMobile(): boolean {
  return useMediaQuery(MOBILE_QUERY);
}
