import { useEffect, useState, type ReactNode } from "react";
import {
  Brain,
  Database,
  Gauge,
  ShieldCheck,
  SlidersHorizontal,
  Users,
  WarningCircle,
} from "@phosphor-icons/react";
import { api, type AdminOverview as AdminOverviewData } from "../../api";
import { useI18n } from "../../lib/i18n";

type Status = "loading" | "ready" | "error";

// Read-only health + usage dashboard (mirrors /api/health + totals). Admin-only; gated by AdminLayout.
export default function AdminOverview() {
  const { t } = useI18n();
  const [data, setData] = useState<AdminOverviewData | null>(null);
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    let active = true;
    api
      .getAdminOverview()
      .then((res) => {
        if (!active) return;
        setData(res);
        setStatus("ready");
      })
      .catch(() => active && setStatus("error"));
    return () => {
      active = false;
    };
  }, []);

  if (status === "loading") return <p className="text-sm text-muted">{t("admin.loading")}</p>;
  if (status === "error" || !data)
    return (
      <div className="flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
        <WarningCircle size={18} className="shrink-0" />
        <p className="font-medium">{t("admin.overview.loadError")}</p>
      </div>
    );

  const storeCount = Object.keys(data.stores).length;
  const storesReady = Object.values(data.stores).filter(Boolean).length;

  return (
    <div>
      <div className="flex items-center gap-2">
        <Gauge size={18} weight="duotone" className="text-primary" />
        <h2 className="text-sm font-semibold text-content">{t("admin.overview.title")}</h2>
      </div>
      <p className="mt-1 text-xs text-muted">{t("admin.overview.desc")}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <OverviewCard
          icon={<ShieldCheck size={16} weight="duotone" />}
          label={t("admin.overview.auth")}
          value={data.auth_configured ? t("admin.overview.configured") : t("admin.overview.notConfigured")}
          ok={data.auth_configured}
        />
        <OverviewCard
          icon={<Brain size={16} weight="duotone" />}
          label={t("admin.overview.chat")}
          value={data.chat_configured ? t("admin.overview.configured") : t("admin.overview.notConfigured")}
          ok={data.chat_configured}
        />
        <OverviewCard
          icon={<Database size={16} weight="duotone" />}
          label={t("admin.overview.stores")}
          value={t("admin.overview.storesReady", { ready: storesReady, total: storeCount })}
          ok={storesReady === storeCount}
        />
        <OverviewCard
          icon={<Users size={16} weight="duotone" />}
          label={t("admin.overview.totalUsers")}
          value={String(data.total_users)}
        />
        <OverviewCard
          icon={<SlidersHorizontal size={16} weight="duotone" />}
          label={t("admin.overview.totalAnalyses")}
          value={String(data.total_analyses)}
        />
      </div>
    </div>
  );
}

function OverviewCard({
  icon,
  label,
  value,
  ok,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  ok?: boolean;
}) {
  const tone = ok === undefined ? "text-content" : ok ? "text-secondary" : "text-danger";
  return (
    <div className="rounded-2xl border border-border-dark bg-surface-dark p-4">
      <div className="flex items-center gap-1.5 text-faint">
        {icon}
        <span className="text-xs font-medium">{label}</span>
      </div>
      <p className={`mt-2 text-base font-semibold tabular-nums ${tone}`}>{value}</p>
    </div>
  );
}
