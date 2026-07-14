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
  useState,
  type ReactNode,
} from "react";
import type { Session, User } from "@supabase/supabase-js";
import { api } from "../api";
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
  signInWithPassword: (email: string, password: string) => Promise<void>;
  /** Returns whether the account still needs email confirmation (no session yet). */
  signUpWithPassword: (email: string, password: string) => Promise<{ needsConfirmation: boolean }>;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
}

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
  useEffect(() => {
    if (!userId) {
      setIsAdmin(false);
      setAdminState("ready");
      return;
    }
    let active = true;
    setAdminState("loading");
    api
      .adminStatus()
      .then((res) => {
        if (!active) return;
        setIsAdmin(res.is_admin);
        setAdminState("ready");
      })
      .catch(() => {
        if (!active) return;
        setIsAdmin(false);
        setAdminState("error");
      });
    return () => {
      active = false;
    };
  }, [userId]);

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
      signInWithPassword,
      signUpWithPassword,
      signInWithGoogle,
      signOut,
    }),
    [
      session,
      loading,
      isAdmin,
      adminState,
      signInWithPassword,
      signUpWithPassword,
      signInWithGoogle,
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
