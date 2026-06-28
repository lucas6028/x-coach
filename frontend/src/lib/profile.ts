// Helpers for reading display fields off a Supabase user. OAuth providers (Google) populate
// user_metadata with an avatar URL and name under a few possible keys; email/password accounts
// have neither, so callers fall back to the email / an initial.
import type { User } from "@supabase/supabase-js";

export function avatarUrl(user: User): string | null {
  const m = user.user_metadata ?? {};
  return (m.avatar_url as string) || (m.picture as string) || null;
}

export function displayName(user: User): string {
  const m = user.user_metadata ?? {};
  return (m.full_name as string) || (m.name as string) || user.email || "";
}

export function initial(user: User): string {
  return (displayName(user) || "?").charAt(0).toUpperCase();
}
