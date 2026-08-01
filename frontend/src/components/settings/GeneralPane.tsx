import type { User } from "@supabase/supabase-js";
import ProfileSection from "./ProfileSection";
import PreferencesSection from "./PreferencesSection";

// "General" holds more than one section, stacked in a single scrolling pane — the reference
// layout's shape, where a category is a scroll target rather than a single block of rows.
export default function GeneralPane({ user }: { user: User }) {
  return (
    <div className="space-y-12">
      <ProfileSection user={user} />
      <PreferencesSection />
    </div>
  );
}
