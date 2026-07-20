// LIFF situation awareness for the layout layer. `lib/liff.isInLiffClient()` is async (it
// awaits the SDK init), but a layout has to decide which shell to render on its very first
// paint — so this provider resolves the real answer once and shares it with the whole app.
//
// The pending window is real: on a LINE redirect-return the SDK load + init takes ~1-1.5s.
// Rendering the web shell for that second and then swapping to the app shell is a visible
// flash of the wrong UI, so the initial state is a SYNCHRONOUS guess from two signals LINE
// leaves lying around (the in-app browser's user agent, and the liff.state/liff-referrer
// params LINE appends when it opens a LIFF URL). The guess is corrected — silently, and
// almost never in practice — once the SDK answers.

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { initLiff, isLiffConfigured } from "./liff";

export interface LiffContextValue {
  /** Whether the SDK has answered yet (false while `isInClient` is still the guess). */
  ready: boolean;
  /** Running inside the LINE app's in-app browser. */
  isInClient: boolean;
}

// Default for consumers rendered outside the provider (every existing unit test): plain web.
const LiffCtx = createContext<LiffContextValue>({ ready: true, isInClient: false });

/** The synchronous first-paint guess. */
function guessInClient(): boolean {
  // No LIFF id configured — the app is a plain web page and no signal can change that.
  if (!isLiffConfigured()) return false;
  if (/\bLine\//i.test(navigator.userAgent || "")) return true;
  const query = window.location.search;
  return query.includes("liff.state") || query.includes("liff-referrer");
}

export function LiffProvider({ children }: { children: ReactNode }) {
  const [value, setValue] = useState<LiffContextValue>(() => ({
    // Nothing to wait for when LIFF is unconfigured: that state is final.
    ready: !isLiffConfigured(),
    isInClient: guessInClient(),
  }));

  useEffect(() => {
    if (!isLiffConfigured()) return;
    let cancelled = false;
    // Any LIFF-side failure degrades to plain web (this module's doc comment, lib/liff.ts).
    // initLiff() itself never rejects, but two paths can still strand `ready` at false forever
    // if left unguarded: `liff.isInClient()` throwing inside the callback below, or the dynamic
    // `import("@line/liff")` never settling on a bad network. On the app's entry route that
    // would now mean a permanent spinner, so both paths must settle `ready`. Race pattern
    // matches lib/camera.ts's `getCameraStream` — same idiom, same "clean up the timer" care.
    const settle = (isInClient: boolean) => {
      if (cancelled) return;
      cancelled = true;
      clearTimeout(timer);
      setValue({ ready: true, isInClient });
    };
    // Comfortably longer than the ~1-1.5s init this is designed around (per the module doc
    // comment) so it essentially never fires in practice, but short enough that a real hang
    // doesn't leave a human staring at a spinner: 6s.
    const timer = setTimeout(() => settle(false), 6_000);
    initLiff().then(
      (liff) => {
        let isInClient = false;
        try {
          isInClient = Boolean(liff?.isInClient());
        } catch {
          // liff.isInClient() itself threw — degrade to plain web, same as any other LIFF failure.
        }
        settle(isInClient);
      },
      () => settle(false)
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  return <LiffCtx.Provider value={value}>{children}</LiffCtx.Provider>;
}

/** `{ ready, isInClient }` — safe to call anywhere, including outside the provider. */
export function useLiffContext(): LiffContextValue {
  return useContext(LiffCtx);
}
