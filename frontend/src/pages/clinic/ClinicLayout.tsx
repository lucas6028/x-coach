import { Link, Outlet } from "react-router-dom";
import { ShieldWarning, WarningCircle } from "@phosphor-icons/react";
import AppLayout from "../../components/AppLayout";
import AdminLoading from "../admin/AdminLoading";
import { useAuth } from "../../lib/auth";
import { useI18n } from "../../lib/i18n";

// The clinician dashboard's gate, mirroring pages/admin/AdminLayout.tsx's loading/error/denied/ready
// states exactly — but around the ordinary app shell (AppLayout + Sidebar), not a separate console:
// a therapist is still a user of the app, so /clinic keeps the same rail, and the sidebar's own
// 「個案管理」 link is how they get here (see components/Sidebar.tsx). The child routes (/clinic,
// /clinic/patients/:id, /clinic/invite) render into <Outlet/> and do NOT wrap themselves in
// AppLayout again — this layout owns the shell and the scroll container for all three.
export default function ClinicLayout() {
  const { t } = useI18n();
  const { isClinician, clinicianState, refreshClinician } = useAuth();

  return (
    <AppLayout title={t("clinic.title")}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-12">
          {clinicianState === "loading" && <AdminLoading labelKey="clinic.loading" />}

          {clinicianState === "error" && (
            <div className="flex flex-col gap-3 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
              <div className="flex items-start gap-2.5">
                <WarningCircle size={18} className="shrink-0" />
                <p className="font-medium">{t("clinic.error")}</p>
              </div>
              <button
                onClick={() => refreshClinician()}
                className="inline-flex w-fit items-center gap-1.5 rounded-xl border border-danger/40 px-3 py-1.5 text-sm font-semibold text-danger transition-colors hover:bg-danger/10 active:scale-[0.99]"
              >
                {t("clinic.retry")}
              </button>
            </div>
          )}

          {clinicianState === "ready" && !isClinician && (
            <div className="flex flex-col items-center gap-4 rounded-2xl border border-dashed border-border-dark bg-content/[0.02] px-6 py-16 text-center">
              <span className="flex h-14 w-14 items-center justify-center rounded-full bg-danger/10 text-danger">
                <ShieldWarning size={30} weight="duotone" />
              </span>
              <p className="font-medium text-content">{t("clinic.denied")}</p>
              <Link
                to="/app"
                className="inline-flex items-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99]"
              >
                {t("nav.analyse")}
              </Link>
            </div>
          )}

          {clinicianState === "ready" && isClinician && <Outlet />}
        </main>
      </div>
    </AppLayout>
  );
}
