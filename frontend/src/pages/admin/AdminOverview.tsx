import { useEffect, useState, type ReactNode } from "react";
import {
  Brain,
  ChatCircleText,
  Database,
  Gauge,
  PaperPlaneTilt,
  Plugs,
  Robot,
  ShieldCheck,
  SlidersHorizontal,
  Users,
  WarningCircle,
} from "@phosphor-icons/react";
import {
  api,
  type AdminOverview as AdminOverviewData,
  type LineStatus,
  type LineWebhookTestResponse,
} from "../../api";
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
      <LineSection />
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

// LINE connection status + push-quota, fetched independently so a slow/failed LINE call never
// blocks or breaks the main overview. On error the section renders nothing (the page stays intact).
function LineSection() {
  const { t } = useI18n();
  const [data, setData] = useState<LineStatus | null>(null);
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    let active = true;
    api
      .getLineStatus()
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

  const [testing, setTesting] = useState(false);
  const [testMsg, setTestMsg] = useState<string | null>(null);

  async function runWebhookTest() {
    setTesting(true);
    setTestMsg(null);
    let res: LineWebhookTestResponse;
    try {
      res = await api.testLineWebhook();
    } catch {
      setTesting(false);
      setTestMsg(t("admin.line.webhookTestError"));
      return;
    }
    setTesting(false);
    if (!res.result) {
      setTestMsg(t("admin.line.webhookTestError"));
    } else if (res.result.success) {
      setTestMsg(t("admin.line.webhookReachable", { code: res.result.status_code ?? 0 }));
    } else {
      setTestMsg(t("admin.line.webhookFailed", { reason: res.result.reason ?? "" }));
    }
  }

  if (status !== "ready" || !data) return null;

  const q = data.quota;
  const limited = q?.type === "limited";

  return (
    <div className="mt-8">
      <div className="flex items-center gap-2">
        <ChatCircleText size={18} weight="duotone" className="text-primary" />
        <h2 className="text-sm font-semibold text-content">{t("admin.line.title")}</h2>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <OverviewCard
          icon={<ShieldCheck size={16} weight="duotone" />}
          label={t("admin.line.loginBridge")}
          value={
            data.login_configured
              ? t("admin.overview.configured")
              : t("admin.overview.notConfigured")
          }
          ok={data.login_configured}
        />
        <OverviewCard
          icon={<ChatCircleText size={16} weight="duotone" />}
          label={t("admin.line.bot")}
          value={
            data.messaging_configured
              ? t("admin.overview.configured")
              : t("admin.overview.notConfigured")
          }
          ok={data.messaging_configured}
        />
        {data.quota_error === "unreachable" ? (
          <div className="rounded-2xl border border-border-dark bg-surface-dark p-4">
            <p className="text-xs text-muted">{t("admin.line.unreachable")}</p>
          </div>
        ) : q ? (
          <>
            <OverviewCard
              icon={<Gauge size={16} weight="duotone" />}
              label={t("admin.line.pushUsed")}
              value={limited ? `${q.used} / ${q.value}` : String(q.used)}
            />
            <OverviewCard
              icon={<SlidersHorizontal size={16} weight="duotone" />}
              label={t("admin.line.remaining")}
              value={limited ? String(q.remaining) : "—"}
            />
          </>
        ) : null}
      </div>
      {q && !limited ? <p className="mt-3 text-xs text-muted">{t("admin.line.noCapNote")}</p> : null}

      {data.bot_info ? (
        <div className="mt-3 rounded-2xl border border-border-dark bg-surface-dark p-4">
          <div className="flex items-center gap-1.5 text-faint">
            <Robot size={16} weight="duotone" />
            <span className="text-xs font-medium">{t("admin.line.oaName")}</span>
          </div>
          <p className="mt-2 text-base font-semibold text-content">{data.bot_info.display_name}</p>
          <p className="text-xs text-muted">{data.bot_info.basic_id}</p>
          {data.bot_info.chat_mode !== "bot" ? (
            <p className="mt-2 text-xs font-medium text-danger">{t("admin.line.chatModeWarn")}</p>
          ) : null}
        </div>
      ) : null}

      {data.webhook ? (
        <div className="mt-3 rounded-2xl border border-border-dark bg-surface-dark p-4">
          <div className="flex items-center gap-1.5 text-faint">
            <Plugs size={16} weight="duotone" />
            <span className="text-xs font-medium">{t("admin.line.webhook")}</span>
          </div>
          <p className="mt-2 truncate text-sm text-content" title={data.webhook.endpoint}>
            {data.webhook.endpoint}
          </p>
          <p className={`text-xs font-medium ${data.webhook.active ? "text-secondary" : "text-danger"}`}>
            {data.webhook.active ? t("admin.line.webhookActive") : t("admin.line.webhookInactive")}
          </p>
          <button
            type="button"
            onClick={runWebhookTest}
            disabled={testing}
            className="mt-3 rounded-lg border border-border-dark px-3 py-1.5 text-xs font-medium text-content disabled:opacity-50"
          >
            {testing ? t("admin.line.webhookTesting") : t("admin.line.webhookTest")}
          </button>
          {testMsg ? <p className="mt-2 text-xs text-muted">{testMsg}</p> : null}
        </div>
      ) : null}

      {data.delivery ? (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <OverviewCard
            icon={<PaperPlaneTilt size={16} weight="duotone" />}
            label={t("admin.line.replyYesterday")}
            value={data.delivery.reply === null ? t("admin.line.deliveryUnready") : String(data.delivery.reply)}
          />
          <OverviewCard
            icon={<PaperPlaneTilt size={16} weight="duotone" />}
            label={t("admin.line.pushYesterday")}
            value={data.delivery.push === null ? t("admin.line.deliveryUnready") : String(data.delivery.push)}
          />
        </div>
      ) : null}
    </div>
  );
}
