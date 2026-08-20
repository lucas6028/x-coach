import { useState } from "react";
import type { User } from "@supabase/supabase-js";
import { useI18n } from "../../lib/i18n";
import { avatarUrl, displayName, initial } from "../../lib/profile";
import { PaneRows, PaneTitle, SettingRow } from "./parts";

// Read-only identity: avatar, name, email, and which provider the session came from.
// Nothing here is editable — the profile lives in Supabase auth, not in our own tables.
export default function ProfileSection({ user }: { user: User }) {
  const { t } = useI18n();
  const [imgError, setImgError] = useState(false);

  const url = imgError ? null : avatarUrl(user);
  const provider = (user.app_metadata?.provider as string) ?? "email";
  // Every LINE user now arrives through the LIFF bridge — web and in-app alike (provider
  // "email" + a line_sub in user_metadata). The provider-name checks remain as a harmless
  // fallback. Exact matches only — `includes("line")` would also catch "linkedin".
  const isLineUser =
    provider === "custom:line" || provider === "line" || Boolean(user.user_metadata?.line_sub);
  const providerLabel = isLineUser
    ? t("settings.provider.line")
    : provider === "google"
      ? t("settings.provider.google")
      : t("settings.provider.email");

  return (
    <section>
      <PaneTitle>{t("settings.profile")}</PaneTitle>
      <PaneRows>
        <SettingRow label={t("settings.avatar")}>
          {url ? (
            <img
              src={url}
              alt=""
              referrerPolicy="no-referrer"
              onError={() => setImgError(true)}
              className="h-12 w-12 rounded-full object-cover ring-1 ring-border-dark"
            />
          ) : (
            <span className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/15 text-base font-semibold text-primary ring-1 ring-border-dark">
              {initial(user)}
            </span>
          )}
        </SettingRow>
        <SettingRow label={t("settings.name")}>
          <span className="text-[15px] text-muted">{displayName(user)}</span>
        </SettingRow>
        <SettingRow label={t("settings.email")}>
          <span className="text-[15px] text-muted">{user.email}</span>
        </SettingRow>
        <SettingRow label={t("settings.provider")}>
          <span className="text-[15px] text-muted">{providerLabel}</span>
        </SettingRow>
      </PaneRows>
    </section>
  );
}
