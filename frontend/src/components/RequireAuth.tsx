import { type ReactNode } from "react";
import { CircleNotch } from "@phosphor-icons/react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";

// Gate a route on a signed-in session. While the initial session check is in flight we show a
// quiet placeholder (not the login page) so a logged-in user never sees a flash of /login on
// refresh. Once resolved, anonymous users are redirected to /login with a `from` location so we
// can return them after sign-in.
export default function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const { t } = useI18n();
  const location = useLocation();

  if (loading) {
    return (
      <div className="grid min-h-[100dvh] place-items-center bg-background-dark text-muted">
        <CircleNotch size={24} className="animate-spin" />
        <span className="sr-only">{t("auth.checking")}</span>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  }

  return <>{children}</>;
}
