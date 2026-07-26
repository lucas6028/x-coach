import { describe, it, expect, vi, afterEach } from "vitest";

// `isSupabaseConfigured` and `supabase` are computed ONCE at module load from import.meta.env, so
// a static top-level import bakes in whatever the ambient environment happened to hold. That is
// exactly how this file used to fail: it asserted the unconfigured state while a developer
// machine's .env supplied real VITE_SUPABASE_* values, so the assertion inverted and the
// public-demo degradation path stopped being tested on precisely the machines that have auth set
// up. Stub the env first, then import -- and state both cases rather than assuming either.
async function loadSupabase(url: string, anonKey: string) {
  vi.resetModules();
  vi.stubEnv("VITE_SUPABASE_URL", url);
  vi.stubEnv("VITE_SUPABASE_ANON_KEY", anonKey);
  return import("../lib/supabase");
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("lib/supabase", () => {
  it("is unconfigured and null when the env vars are absent", async () => {
    const { supabase, isSupabaseConfigured } = await loadSupabase("", "");
    expect(isSupabaseConfigured).toBe(false);
    // Null rather than throwing at import time: auth is optional and the app runs as a public
    // demo without it, so every consumer treats the client as nullable.
    expect(supabase).toBeNull();
  });

  it("is unconfigured when only one of the two vars is set", async () => {
    // A half-configured environment must degrade to the demo state rather than construct a client
    // with a missing credential -- `createClient` would otherwise be handed an empty string.
    const partial = await loadSupabase("https://test.supabase.co", "");
    expect(partial.isSupabaseConfigured).toBe(false);
    expect(partial.supabase).toBeNull();

    const other = await loadSupabase("", "test-anon-key");
    expect(other.isSupabaseConfigured).toBe(false);
    expect(other.supabase).toBeNull();
  });

  it("builds a client when both vars are set", async () => {
    // The branch that was never covered before, because the ambient environment decided which one
    // ran. Both are now stated explicitly.
    const { supabase, isSupabaseConfigured } = await loadSupabase(
      "https://test.supabase.co",
      "test-anon-key"
    );
    expect(isSupabaseConfigured).toBe(true);
    expect(supabase).not.toBeNull();
    expect(supabase?.auth).toBeDefined();
  });
});
