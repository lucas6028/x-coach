import { useEffect, useState } from "react";
import {
  ChatCircleText,
  Gauge,
  PaperPlaneTilt,
  Plugs,
  Robot,
  ShieldCheck,
  SlidersHorizontal,
  WarningCircle,
} from "@phosphor-icons/react";
import {
  api,
  type LineStatus,
  type LineWebhookTestError,
  type LineWebhookTestResponse,
} from "../../api";
import { useI18n } from "../../lib/i18n";
import { OverviewCard } from "./AdminOverview";

type Status = "loading" | "ready" | "error";

// The webhook test is the panel's one ACTIVE probe, so it's where an admin learns WHY a read failed.
// Collapsing 401/429/404 into "couldn't reach LINE" would send them chasing connectivity instead.
const WEBHOOK_TEST_ERROR_KEY: Record<LineWebhookTestError, string> = {
  not_configured: "admin.line.webhookTestNotConfigured",
  unauthorized: "admin.line.webhookTestUnauthorized",
  rate_limited: "admin.line.webhookTestRateLimited",
  no_endpoint: "admin.line.webhookTestNoEndpoint",
  unreachable: "admin.line.webhookTestError",
};

// "yyyymmdd" (LINE's delivery-date format) -> "YYYY-MM-DD". Falls back to the raw value if it
// doesn't match the expected shape rather than rendering something misleading.
function formatDeliveryDate(yyyymmdd: string): string {
  const m = /^(\d{4})(\d{2})(\d{2})$/.exec(yyyymmdd);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : yyyymmdd;
}

// LINE connection status + push-quota, delivery, and webhook diagnostics. Its own dedicated admin
// page (moved out of the overview) so a slow/failed LINE call never blocks the main overview, and a
// failed read now surfaces a proper loading/error state instead of silently rendering nothing.
export default function AdminLine() {
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
    if (res.error) {
      setTestMsg(t(WEBHOOK_TEST_ERROR_KEY[res.error] ?? "admin.line.webhookTestError"));
    } else if (!res.result) {
      setTestMsg(t("admin.line.webhookTestError"));
    } else if (res.result.success) {
      setTestMsg(t("admin.line.webhookReachable", { code: res.result.status_code ?? 0 }));
    } else {
      setTestMsg(
        t("admin.line.webhookFailed", {
          code: res.result.status_code ?? 0,
          reason: res.result.reason ?? "",
        })
      );
    }
  }

  if (status === "loading") return <p className="text-sm text-muted">{t("admin.loading")}</p>;
  if (status === "error" || !data)
    return (
      <div className="flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
        <WarningCircle size={18} className="shrink-0" />
        <p className="font-medium">{t("admin.line.loadError")}</p>
      </div>
    );

  const q = data.quota;
  const limited = q?.type === "limited";

  return (
    <div>
      <div className="flex items-center gap-2">
        <ChatCircleText size={18} weight="duotone" className="text-primary" />
        <h2 className="text-sm font-semibold text-content">{t("admin.line.title")}</h2>
      </div>
      <p className="mt-1 text-xs text-muted">{t("admin.line.desc")}</p>
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

      {/* Each card renders whenever its read was ATTEMPTED (data or error) — never only on success.
          A card that vanishes on failure hides the very misconfiguration this panel is for. */}
      {data.bot_info || data.bot_info_error ? (
        <div className="mt-3 rounded-2xl border border-border-dark bg-surface-dark p-4">
          <div className="flex items-center gap-1.5 text-faint">
            <Robot size={16} weight="duotone" />
            <span className="text-xs font-medium">{t("admin.line.oaName")}</span>
          </div>
          {data.bot_info ? (
            <>
              <p className="mt-2 text-base font-semibold text-content">{data.bot_info.display_name}</p>
              <p className="text-xs text-muted">{data.bot_info.basic_id}</p>
              {data.bot_info.chat_mode !== "bot" ? (
                <p className="mt-2 text-xs font-medium text-danger">{t("admin.line.chatModeWarn")}</p>
              ) : null}
            </>
          ) : (
            <p className="mt-2 text-xs text-muted">{t("admin.line.botInfoUnavailable")}</p>
          )}
        </div>
      ) : null}

      {data.webhook || data.webhook_error ? (
        <div className="mt-3 rounded-2xl border border-border-dark bg-surface-dark p-4">
          <div className="flex items-center gap-1.5 text-faint">
            <Plugs size={16} weight="duotone" />
            <span className="text-xs font-medium">{t("admin.line.webhook")}</span>
          </div>
          {data.webhook ? (
            <>
              <p className="mt-2 truncate text-sm text-content" title={data.webhook.endpoint}>
                {data.webhook.endpoint}
              </p>
              <p className={`text-xs font-medium ${data.webhook.active ? "text-secondary" : "text-danger"}`}>
                {data.webhook.active ? t("admin.line.webhookActive") : t("admin.line.webhookInactive")}
              </p>
            </>
          ) : (
            <p className="mt-2 text-xs text-muted">{t("admin.line.webhookUnavailable")}</p>
          )}
          {/* Deliberately OUTSIDE the data.webhook branch: when the passive read failed, this active
              probe is the only way left to find out whether it's the token, the endpoint, or LINE. */}
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
        <>
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
          <p className="mt-2 text-xs text-faint">
            {t("admin.line.deliveryDate", { date: formatDeliveryDate(data.delivery.date) })}
          </p>
        </>
      ) : data.delivery_error ? (
        <div className="mt-3 rounded-2xl border border-border-dark bg-surface-dark p-4">
          <div className="flex items-center gap-1.5 text-faint">
            <PaperPlaneTilt size={16} weight="duotone" />
            <span className="text-xs font-medium">{t("admin.line.replyYesterday")}</span>
          </div>
          <p className="mt-2 text-xs text-muted">{t("admin.line.deliveryUnavailable")}</p>
        </div>
      ) : null}
    </div>
  );
}
