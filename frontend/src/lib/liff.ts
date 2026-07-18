// LIFF (LINE Front-end Framework) bootstrap. Configured from a Vite env var:
//   VITE_LIFF_ID — the LIFF app id from the LINE Developers console.
//
// LIFF is OPTIONAL, mirroring how lib/supabase treats auth: with no env set every helper
// resolves to null/false and the app runs as a plain web page. The SDK is imported
// dynamically so its bytes are only downloaded when a LIFF id is actually configured, and
// a failed init (bad id, network, opened outside LINE with strict settings) degrades to
// "not in LIFF" instead of crashing the app — the fallback-to-web behaviour is the spec.

import type { Liff } from "@line/liff";

// Read lazily (not at module scope) so unit tests can stub the env per test case.
function configuredLiffId(): string | undefined {
  return (import.meta.env.VITE_LIFF_ID as string | undefined) || undefined;
}

/** Whether a LIFF id is configured at all (the whole module no-ops without one). */
export function isLiffConfigured(): boolean {
  return Boolean(configuredLiffId());
}

// One init per page load, shared by every caller (init is idempotent but not free).
let liffPromise: Promise<Liff | null> | null = null;

/** Initialise the LIFF SDK once and return it, or null when unconfigured / init failed. */
export function initLiff(): Promise<Liff | null> {
  const liffId = configuredLiffId();
  if (!liffId) return Promise.resolve(null);
  if (!liffPromise) {
    liffPromise = import("@line/liff")
      .then(async ({ default: liff }) => {
        await liff.init({ liffId });
        return liff;
      })
      .catch((err) => {
        // Degrade, don't break: the same bundle must keep working as a plain web page.
        console.warn("liff: init failed — continuing as a plain web page", err);
        return null;
      });
  }
  return liffPromise;
}

/** True only when running inside the LINE app's LIFF browser (never in an external browser). */
export async function isInLiffClient(): Promise<boolean> {
  const liff = await initLiff();
  return Boolean(liff?.isInClient());
}

/**
 * The current LINE ID token, or null (not in LIFF / not logged in / no `openid` scope).
 * Works in the LINE in-app browser AND in an external browser once the user has completed
 * `liff.login()` — both surface a standard LINE ID token the backend bridge can verify
 * (see lib/auth.signInWithLine). Keyed on `isLoggedIn()`, not `isInClient()`, so the web
 * login path is covered too.
 */
export async function getLiffIdToken(): Promise<string | null> {
  const liff = await initLiff();
  if (!liff?.isLoggedIn()) return null;
  try {
    return liff.getIDToken();
  } catch {
    return null;
  }
}

/** Drop the cached init so unit tests can exercise fresh configurations. */
export function _resetLiffForTests(): void {
  liffPromise = null;
}
