// Supabase browser client. Configured from Vite env vars (see frontend/.env.example):
//   VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY
//
// Auth is OPTIONAL: with no env set, `supabase` is null and the app runs as a public demo
// (uploads are analyzed but not saved, history is unavailable). Every consumer must treat the
// client as nullable. The anon key is safe to ship — row access is governed by Postgres RLS.

import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

export const isSupabaseConfigured = Boolean(url && anonKey);

export const supabase: SupabaseClient | null = isSupabaseConfigured
  ? createClient(url as string, anonKey as string, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        // Pick up the access token Supabase appends to the URL after an OAuth redirect.
        detectSessionInUrl: true,
      },
    })
  : null;
