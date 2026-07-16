// Auth context over Supabase Auth. Stateless on the backend: the session lives in the browser
// (supabase-js persists + refreshes it); api.ts reads the access token straight from the client
// for each request. This provider exposes the user/session to React and the sign-in/out actions.
//
// When Supabase is not configured (`supabase === null`), the provider resolves immediately to a
// logged-out state and the action methods throw a clear error — the public demo still works.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { Session, User } from "@supabase/supabase-js";
import { api, ApiError } from "../api";
import { initLiff, isLiffConfigured } from "./liff";
import { isSupabaseConfigured, supabase } from "./supabase";

interface AuthValue {
  user: User | null;
  session: Session | null;
  /** True until the initial session lookup resolves (so guards don't flash the login page). */
  loading: boolean;
  /** Whether Supabase Auth is wired up at all (env present). */
  configured: boolean;
  /** Whether the signed-in user holds the admin role (false when logged out). Probed ONCE per
   *  session so the shell's Admin link and the /admin gate never re-hit the endpoint on every
   *  navigation — the server re-checks on each admin API call, so this is UX gating only. */
  isAdmin: boolean;
  /** Status of the admin-role probe for the current user ("ready" also covers the logged-out case). */
  adminState: "loading" | "ready" | "error";
  /** Re-run the admin-role probe for the current user (used to recover from a transient probe error
   *  without a full reload). No-op when logged out. */
  refreshAdmin: () => void;
  signInWithPassword: (email: string, password: string) => Promise<void>;
  /** Returns whether the account still needs email confirmation (no session yet). */
  signUpWithPassword: (email: string, password: string) => Promise<{ needsConfirmation: boolean }>;
  signInWithGoogle: () => Promise<void>;
  /** LINE login: silent LIFF-token exchange inside the LINE app, OAuth redirect on the web. */
  signInWithLine: () => Promise<void>;
  signOut: () => Promise<void>;
}

// Supabase custom OIDC provider slug for LINE — register the provider as "line" in the
// Supabase dashboard (Auth → Sign In / Up → Custom providers) for the web login path.
const LINE_OAUTH_PROVIDER = "custom:line" as const;

// One liff.login() reload per browser session: a stale cached LINE ID token 401s on the
// bridge and liff.login() re-mints it via a full-page redirect — but if the exchange keeps
// failing after that, looping the redirect forever would brick the tab. sessionStorage
// (not state) because the redirect wipes the JS heap.
const LIFF_RELOGIN_KEY = "xcoach.liffReloginTried";

const AuthContext = createContext<AuthValue | null>(null);

function requireClient() {
  if (!supabase) throw new Error("Authentication is not configured.");
  return supabase;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  // Only "loading" while a real client exists to query; otherwise we already know we're logged out.
  const [loading, setLoading] = useState(isSupabaseConfigured);
  // Admin role, resolved once per signed-in user (keyed on user id below), not per component mount.
  const [isAdmin, setIsAdmin] = useState(false);
  const [adminState, setAdminState] = useState<"loading" | "ready" | "error">("ready");

  useEffect(() => {
    if (!supabase) return;
    let active = true;
    supabase.auth.getSession().then(({ data }) => {
      if (active) {
        setSession(data.session);
        setLoading(false);
      }
    });
    const { data } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next);
    });
    return () => {
      active = false;
      data.subscription.unsubscribe();
    };
  }, []);

  // Resolve the admin role once whenever the signed-in identity changes (keyed on the user id, a
  // stable primitive — not the session object, which changes reference on every token refresh). This
  // replaces the per-mount probes the Sidebar and AdminLayout used to make on every navigation.
  const userId = session?.user?.id ?? null;
  // Monotonic probe id: only the newest probe's result is applied, so a stale in-flight probe (from a
  // previous user id, or superseded by a manual refresh) can never clobber the current state.
  const probeIdRef = useRef(0);
  const refreshAdmin = useCallback(() => {
    if (!userId) {
      setIsAdmin(false);
      setAdminState("ready");
      return;
    }
    const probeId = ++probeIdRef.current;
    setAdminState("loading");
    api
      .adminStatus()
      .then((res) => {
        if (probeId !== probeIdRef.current) return;
        setIsAdmin(res.is_admin);
        setAdminState("ready");
      })
      .catch(() => {
        if (probeId !== probeIdRef.current) return;
        setIsAdmin(false);
        setAdminState("error");
      });
  }, [userId]);

  // Probe once on mount and whenever the signed-in identity changes.
  useEffect(() => {
    refreshAdmin();
  }, [refreshAdmin]);

  const signInWithPassword = useCallback(async (email: string, password: string) => {
    const { error } = await requireClient().auth.signInWithPassword({ email, password });
    if (error) throw new Error(error.message);
  }, []);

  const signUpWithPassword = useCallback(async (email: string, password: string) => {
    const { data, error } = await requireClient().auth.signUp({ email, password });
    if (error) throw new Error(error.message);
    // With email confirmation enabled, signUp returns a user but no session yet.
    return { needsConfirmation: !data.session };
  }, []);

  const signInWithGoogle = useCallback(async () => {
    const { error } = await requireClient().auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/app` },
    });
    if (error) throw new Error(error.message);
  }, []);

  const signInWithLine = useCallback(async () => {
    const client = requireClient();

    // Inside the LINE app the user is already LINE-authenticated: exchange the LIFF ID
    // token for a Supabase session through the backend bridge — no redirect, no form.
    const liff = await initLiff();
    if (liff?.isInClient()) {
      const idToken = liff.getIDToken();
      if (idToken) {
        try {
          const minted = await api.lineLogin(idToken);
          const { error } = await client.auth.setSession(minted);
          if (error) throw new Error(error.message);
          sessionStorage.removeItem(LIFF_RELOGIN_KEY);
          return;
        } catch (err) {
          // 401 = the cached LINE ID token went stale; fall through to one liff.login().
          if (!(err instanceof ApiError && err.status === 401)) {
            throw err instanceof Error ? err : new Error(String(err));
          }
        }
      }
      // Missing/stale ID token: re-run LINE auth ONCE (full-page redirect). A second
      // failure in the same browser session surfaces as an error instead of a reload loop.
      if (sessionStorage.getItem(LIFF_RELOGIN_KEY) !== "1") {
        sessionStorage.setItem(LIFF_RELOGIN_KEY, "1");
        liff.login();
        return;
      }
      throw new Error("LINE sign-in failed inside LIFF. Please try again later.");
    }

    // Plain web: Supabase's custom OIDC provider handles the whole redirect dance, exactly
    // like the Google path (detectSessionInUrl picks the session up on return).
    const { error } = await client.auth.signInWithOAuth({
      provider: LINE_OAUTH_PROVIDER,
      options: { redirectTo: `${window.location.origin}/app` },
    });
    if (error) throw new Error(error.message);
  }, []);

  // Auto sign-in inside the LINE app: a LIFF user already proved who they are to LINE, so
  // the first anonymous render silently exchanges their ID token for a session (once per
  // mount — NOT retried after sign-out, and never via liff.login() here, so the auto path
  // can't cause redirect loops). Failures stay silent: the Login button remains the
  // explicit, error-surfacing path.
  const autoLineTriedRef = useRef(false);
  useEffect(() => {
    if (!supabase || !isLiffConfigured() || loading || session || autoLineTriedRef.current) return;
    autoLineTriedRef.current = true;
    let active = true;
    (async () => {
      const liff = await initLiff();
      if (!active || !liff?.isInClient()) return;
      const idToken = liff.getIDToken();
      if (!idToken) return;
      // Only attempt the exchange when the server says the bridge is configured.
      const health = await api.health().catch(() => null);
      if (!active || !health?.line_login_configured) return;
      try {
        const minted = await api.lineLogin(idToken);
        if (!active) return;
        await supabase!.auth.setSession(minted);
      } catch {
        /* silent — see above */
      }
    })();
    return () => {
      active = false;
    };
  }, [loading, session]);

  const signOut = useCallback(async () => {
    if (!supabase) return;
    await supabase.auth.signOut();
  }, []);

  const value = useMemo<AuthValue>(
    () => ({
      user: session?.user ?? null,
      session,
      loading,
      configured: isSupabaseConfigured,
      isAdmin,
      adminState,
      refreshAdmin,
      signInWithPassword,
      signUpWithPassword,
      signInWithGoogle,
      signInWithLine,
      signOut,
    }),
    [
      session,
      loading,
      isAdmin,
      adminState,
      refreshAdmin,
      signInWithPassword,
      signUpWithPassword,
      signInWithGoogle,
      signInWithLine,
      signOut,
    ]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
