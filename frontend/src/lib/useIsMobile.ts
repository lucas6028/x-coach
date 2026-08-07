import { useEffect, useState } from "react";

// Tailwind's `lg` — the breakpoint the desktop studio's 8/4 grid is written against.
const QUERY = "(max-width: 1023px)";

/**
 * Whether the phone layout should render.
 *
 * A media QUERY rather than `lg:hidden` / `hidden lg:block` on two trees, because both trees mount
 * a <video> and a skeleton canvas: rendering them together would decode the clip twice and run two
 * rAF loops, with only CSS hiding one of them. This picks one.
 *
 * Defaults to false during SSR-less first paint on a server-rendered-free app is moot, but jsdom
 * has no `matchMedia` in some setups — hence the guard, which keeps tests on the desktop tree
 * unless they opt in.
 */
export function useIsMobile(): boolean {
  const [mobile, setMobile] = useState(() =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia(QUERY).matches
      : false
  );

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia(QUERY);
    const onChange = () => setMobile(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return mobile;
}
