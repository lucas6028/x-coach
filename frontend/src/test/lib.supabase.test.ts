import { describe, it, expect } from "vitest";
import { supabase, isSupabaseConfigured } from "../lib/supabase";

// In the test environment the VITE_SUPABASE_* vars are unset, so the client must degrade to the
// public-demo state (null client) rather than throwing at import time.
describe("lib/supabase", () => {
  it("is unconfigured and null without env vars", () => {
    expect(isSupabaseConfigured).toBe(false);
    expect(supabase).toBeNull();
  });
});
