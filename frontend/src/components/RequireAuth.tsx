import { type ReactNode } from "react";
import { CircleNotch } from "@phosphor-icons/react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";

// Gate a route on a signed-in session. While the initial session check is in flight we show a
// quiet placeholder (not the login page) so a logged-in user never sees a flash of /login on
// refresh. Once resolved, anonymous users are redirected to the login page (default `/login`,
// overridable via `redirectTo` — e.g. the admin tree points at `/admin/login`) with a `from`
// location so we can return them after sign-in.
//
// `lineAuthenticating` gets the same treatment as `loading`: inside LINE, lib/auth's silent
// token exchange runs with `loading` already false but `user` still null. Without this, tapping
// a guarded tab (History, Settings) during that window bounces to /login — which itself redirects
// straight back to /app in-client (see Login.tsx), so the one screen that could explain the wait
// is unreachable. Worse, if the exchange fails outright those tabs would bounce forever.
export default function RequireAuth({
  children,
  redirectTo = "/login",
}: {
  children: ReactNode;
  redirectTo?: string;
}) {
  const { user, loading, lineAuthenticating } = useAuth();
  const { t } = useI18n();
  const location = useLocation();

  if (loading || lineAuthenticating) {
    return (
      <div className="grid min-h-[100dvh] place-items-center bg-background-dark text-muted">
        <CircleNotch size={24} className="animate-spin" />
        <span className="sr-only">{t("auth.checking")}</span>
      </div>
    );
  }

  if (!user) {
    return <Navigate to={redirectTo} replace state={{ from: location.pathname + location.search }} />;
  }

  return <>{children}</>;
}
