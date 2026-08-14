import { useEffect, useState, type ReactNode } from "react";
import {
  ArrowSquareOut,
  ChatCircleText,
  CheckCircle,
  Copy,
  Gauge,
  PaperPlaneTilt,
  Plugs,
  Robot,
  ShieldCheck,
  WarningCircle,
} from "@phosphor-icons/react";
import {
  api,
  type LineStatus,
  type LineWebhookTestError,
  type LineWebhookTestResponse,
} from "../../api";
import { useI18n, type TFunc } from "../../lib/i18n";
import LineLogo from "../../components/LineLogo";
import AdminLoading from "./AdminLoading";

type Status = "loading" | "ready" | "error";

// LINE's own console. Deliberately the manager HOME and not a per-account deep link: the OA manager
// addresses accounts by an internal id we never receive, and `basic_id` (@xcoach) is not it — a
// constructed deep link would 404 the admin instead of taking them anywhere.
const OA_MANAGER_URL = "https://manager.line.biz/";

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

/* ---------- Presentational building blocks (local to this page) ---------- */

// The violet glyph tile every card leads with. Local rather than shared: AdminOverview's compact
// OverviewCard has no room for it, and this page's cards are taller by design.
function IconBadge({ children }: { children: ReactNode }) {
  return (
    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
      {children}
    </span>
  );
}

type Tone = "ok" | "warn" | "bad";

const TONE_CLASS: Record<Tone, string> = {
  ok: "bg-secondary/10 text-secondary",
  warn: "bg-warning/10 text-warning",
  bad: "bg-danger/10 text-danger",
};

function Pill({ tone, icon, children }: { tone: Tone; icon?: ReactNode; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${TONE_CLASS[tone]}`}
    >
      {icon}
      {children}
    </span>
  );
}

function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`flex flex-col rounded-2xl border border-border-dark bg-surface-dark p-5 shadow-card ${className}`}>
      {children}
    </div>
  );
}

function CardHead({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-3">
      <IconBadge>{icon}</IconBadge>
      <h3 className="text-sm font-semibold text-content">{title}</h3>
    </div>
  );
}

// Ring gauge for the monthly push allowance. Rendered ONLY when LINE reports a real cap — a donut
// needs a denominator, and inventing one for an uncapped account would draw a limit that isn't there.
function Donut({ percent }: { percent: number }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  const filled = Math.min(100, Math.max(0, percent));
  return (
    <div className="relative h-[68px] w-[68px] shrink-0">
      <svg viewBox="0 0 68 68" className="h-full w-full -rotate-90" aria-hidden="true">
        <circle cx="34" cy="34" r={r} fill="none" stroke="rgb(var(--c-track))" strokeWidth="7" />
        <circle
          cx="34"
          cy="34"
          r={r}
          fill="none"
          stroke="#7b61ff"
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${(filled / 100) * c} ${c}`}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-xs font-bold tabular-nums text-content">
        {filled}%
      </div>
    </div>
  );
}

// A labelled figure: the small caption plus the number underneath. `tone` colours the number when
// the value is a verdict (configured / not) rather than a count.
function Figure({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div>
      <p className="text-xs text-faint">{label}</p>
      <p className={`mt-0.5 text-base font-semibold tabular-nums ${tone ?? "text-content"}`}>{value}</p>
    </div>
  );
}

/* ---------- Page ---------- */

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
  const [testResult, setTestResult] = useState<TestOutcome | null>(null);
  const [copied, setCopied] = useState<"idle" | "ok" | "fail">("idle");

  async function runWebhookTest() {
    setTesting(true);
    setTestResult(null);
    let res: LineWebhookTestResponse;
    try {
      res = await api.testLineWebhook();
    } catch {
      setTesting(false);
      setTestResult({ ok: false, text: t("admin.line.webhookTestError") });
      return;
    }
    setTesting(false);
    if (res.error) {
      setTestResult({ ok: false, text: t(WEBHOOK_TEST_ERROR_KEY[res.error] ?? "admin.line.webhookTestError") });
    } else if (!res.result) {
      setTestResult({ ok: false, text: t("admin.line.webhookTestError") });
    } else if (res.result.success) {
      setTestResult({ ok: true, text: t("admin.line.webhookReachable", { code: res.result.status_code ?? 0 }) });
    } else {
      setTestResult({
        ok: false,
        text: t("admin.line.webhookFailed", {
          code: res.result.status_code ?? 0,
          reason: res.result.reason ?? "",
        }),
      });
    }
  }

  // `navigator.clipboard` is absent on insecure origins and in some embedded webviews; the throw is
  // caught and reported so the button never looks like it worked when it didn't.
  async function copyEndpoint(endpoint: string) {
    try {
      await navigator.clipboard.writeText(endpoint);
      setCopied("ok");
    } catch {
      setCopied("fail");
    }
  }

  if (status === "loading") return <AdminLoading />;
  if (status === "error" || !data)
    return (
      <div className="flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
        <WarningCircle size={18} className="shrink-0" />
        <p className="font-medium">{t("admin.line.loadError")}</p>
      </div>
    );

  const q = data.quota;
  const limited = q?.type === "limited";
  const usedPercent = limited && q.value ? Math.round((q.used / q.value) * 100) : 0;

  // Login and messaging are two SEPARATE LINE channels. Half-wired is its own state: a green
  // "Enabled" on a server where only one is configured is exactly the misreport this page exists
  // to prevent.
  const wired = Number(data.login_configured) + Number(data.messaging_configured);
  const headerTone: Tone = wired === 2 ? "ok" : wired === 1 ? "warn" : "bad";
  const headerKey =
    wired === 2 ? "admin.line.stateEnabled" : wired === 1 ? "admin.line.statePartial" : "admin.line.stateDisabled";

  return (
    <div>
      {/* Header: the LINE mark is aria-hidden so the heading's accessible name stays "LINE". */}
      <div className="flex flex-wrap items-center gap-3">
        <LineLogo size={30} />
        <h2 className="text-lg font-bold tracking-tight text-content">{t("admin.line.title")}</h2>
        <Pill tone={headerTone}>{t(headerKey)}</Pill>
      </div>
      <p className="mt-2 text-xs text-muted">{t("admin.line.desc")}</p>

      {/* The four status cards share one row from `xl` — below that the column is too narrow to
          give the quota card room for its donut alongside the used/remaining figures. */}
      <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <ChannelCard
          icon={<ShieldCheck size={20} weight="duotone" />}
          title={t("admin.line.loginBridge")}
          description={t("admin.line.loginBridgeDesc")}
          configured={data.login_configured}
          okLabel={t("admin.overview.configured")}
          badLabel={t("admin.overview.notConfigured")}
        />
        <ChannelCard
          icon={<ChatCircleText size={20} weight="duotone" />}
          title={t("admin.line.bot")}
          description={t("admin.line.botDesc")}
          configured={data.messaging_configured}
          okLabel={t("admin.overview.configured")}
          badLabel={t("admin.overview.notConfigured")}
        />

        {/* Quota has THREE renderings, not one: a capped account (donut + remaining), an uncapped
            account (used only, no denominator to draw), and a failed read. Collapsing them would
            hide either a misconfiguration or the fact that we simply couldn't ask. */}
        {data.quota_error === "unreachable" ? (
          <Card>
            <CardHead icon={<Gauge size={20} weight="duotone" />} title={t("admin.line.pushUsed")} />
            <p className="mt-4 text-xs text-muted">{t("admin.line.unreachable")}</p>
          </Card>
        ) : q ? (
          <Card>
            <CardHead icon={<Gauge size={20} weight="duotone" />} title={t("admin.line.pushUsed")} />
            <p className="mt-3 text-xs text-muted">
              {limited ? t("admin.line.quotaFree", { limit: q.value ?? 0 }) : t("admin.line.quotaNoCap")}
            </p>
            <div className="mt-3 flex items-center gap-4">
              {limited ? <Donut percent={usedPercent} /> : null}
              <div className="min-w-0 space-y-2">
                <p className="text-xl font-extrabold tabular-nums leading-none text-content">
                  {limited ? `${q.used} / ${q.value}` : String(q.used)}
                </p>
                <Figure label={t("admin.line.remaining")} value={limited ? String(q.remaining) : "—"} />
              </div>
            </div>
          </Card>
        ) : null}

        {/* Each card renders whenever its read was ATTEMPTED (data or error) — never only on success.
            A card that vanishes on failure hides the very misconfiguration this panel is for. */}
        {data.bot_info || data.bot_info_error ? (
          <Card>
            <CardHead icon={<Robot size={20} weight="duotone" />} title={t("admin.line.oaName")} />
            {data.bot_info ? (
              <>
                <div className="mt-3 space-y-2">
                  <p className="text-base font-semibold text-content">{data.bot_info.display_name}</p>
                  <p className="text-xs text-muted">{data.bot_info.basic_id}</p>
                </div>
                {data.bot_info.chat_mode !== "bot" ? (
                  <p className="mt-2 text-xs font-medium text-danger">{t("admin.line.chatModeWarn")}</p>
                ) : null}
              </>
            ) : (
              <p className="mt-3 text-xs text-muted">{t("admin.line.botInfoUnavailable")}</p>
            )}
            <a
              href={OA_MANAGER_URL}
              target="_blank"
              rel="noreferrer noopener"
              className="mt-auto inline-flex items-center justify-center gap-1.5 rounded-xl border border-primary/25 px-3 py-2 text-xs font-semibold text-primary transition-colors hover:bg-primary/5"
            >
              <ArrowSquareOut size={14} weight="bold" />
              {t("admin.line.oaManager")}
            </a>
          </Card>
        ) : null}
      </div>

      {q && !limited ? <p className="mt-3 text-xs text-muted">{t("admin.line.noCapNote")}</p> : null}

      {data.webhook || data.webhook_error ? (
        <Card className="mt-3">
          <CardHead icon={<Plugs size={20} weight="duotone" />} title={t("admin.line.webhook")} />
          <div className="mt-4 grid gap-5 sm:grid-cols-[1.4fr_1fr]">
            <div className="min-w-0">
              <p className="text-xs font-medium text-content">{t("admin.line.webhookUrl")}</p>
              {data.webhook ? (
                <>
                  <div className="mt-1.5 flex items-center gap-2 rounded-xl border border-border-dark bg-background-dark px-3 py-2">
                    <code className="min-w-0 flex-1 truncate font-mono text-xs text-content" title={data.webhook.endpoint}>
                      {data.webhook.endpoint}
                    </code>
                    <button
                      type="button"
                      onClick={() => copyEndpoint(data.webhook!.endpoint)}
                      aria-label={t("admin.line.copy")}
                      className="shrink-0 rounded-md p-1 text-muted transition-colors hover:bg-content/5 hover:text-content"
                    >
                      <Copy size={16} />
                    </button>
                  </div>
                  {copied === "ok" ? (
                    <p className="mt-1.5 text-xs font-medium text-secondary">{t("admin.line.copied")}</p>
                  ) : copied === "fail" ? (
                    <p className="mt-1.5 text-xs font-medium text-danger">{t("admin.line.copyFailed")}</p>
                  ) : (
                    <p className="mt-1.5 text-xs text-muted">{t("admin.line.webhookUrlNote")}</p>
                  )}
                </>
              ) : (
                <p className="mt-1.5 text-xs text-muted">{t("admin.line.webhookUnavailable")}</p>
              )}
            </div>

            <div>
              <p className="text-xs font-medium text-content">{t("admin.line.webhookStatus")}</p>
              <div className="mt-1.5">
                {data.webhook ? (
                  <Pill
                    tone={data.webhook.active ? "ok" : "bad"}
                    icon={data.webhook.active ? <CheckCircle size={13} weight="fill" /> : undefined}
                  >
                    {data.webhook.active ? t("admin.line.webhookActive") : t("admin.line.webhookInactive")}
                  </Pill>
                ) : (
                  <span className="text-xs text-muted">—</span>
                )}
              </div>
              {/* Deliberately OUTSIDE the data.webhook branch: when the passive read failed, this
                  active probe is the only way left to find out whether it's the token, the endpoint,
                  or LINE. */}
              <p className="mt-4 text-xs text-muted">{t("admin.line.webhookTestNote")}</p>
              <button
                type="button"
                onClick={runWebhookTest}
                disabled={testing}
                className="mt-2 inline-flex items-center gap-1.5 rounded-xl border border-primary/25 px-3 py-2 text-xs font-semibold text-primary transition-colors hover:bg-primary/5 disabled:opacity-50"
              >
                <PaperPlaneTilt size={14} weight="fill" />
                {testing ? t("admin.line.webhookTesting") : t("admin.line.webhookTest")}
              </button>
            </div>
          </div>
        </Card>
      ) : null}

      {data.delivery ? (
        <>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <DeliveryCard
              label={t("admin.line.replyYesterday")}
              value={data.delivery.reply === null ? t("admin.line.deliveryUnready") : String(data.delivery.reply)}
              muted={data.delivery.reply === null}
            />
            <DeliveryCard
              label={t("admin.line.pushYesterday")}
              value={data.delivery.push === null ? t("admin.line.deliveryUnready") : String(data.delivery.push)}
              muted={data.delivery.push === null}
            />
          </div>
          <p className="mt-2 text-xs text-faint">
            {t("admin.line.deliveryDate", { date: formatDeliveryDate(data.delivery.date) })}
          </p>
        </>
      ) : data.delivery_error ? (
        <Card className="mt-3">
          <CardHead icon={<PaperPlaneTilt size={20} weight="duotone" />} title={t("admin.line.replyYesterday")} />
          <p className="mt-3 text-xs text-muted">{t("admin.line.deliveryUnavailable")}</p>
        </Card>
      ) : null}

      {/* The webhook test is an ACTION with a side effect (LINE really posts to the endpoint), so
          its verdict gets a dialog rather than a line of grey text under the button — the outcome
          is the whole point of pressing it, and the button sits in the card's narrow right column
          where a wrapped status code is easy to miss. */}
      {testResult ? (
        <WebhookTestDialog outcome={testResult} onClose={() => setTestResult(null)} t={t} />
      ) : null}
    </div>
  );
}

// The webhook test's verdict. `ok` drives the tone; `text` is already localized by the caller,
// which is the only place that knows how to name each failure cause.
interface TestOutcome {
  ok: boolean;
  text: string;
}

// Modal verdict for the webhook probe. Dismissed by the button, the backdrop, or Escape.
function WebhookTestDialog({
  outcome,
  onClose,
  t,
}: {
  outcome: TestOutcome;
  onClose: () => void;
  t: TFunc;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="line-webhook-test-title"
        className="relative w-full max-w-sm rounded-2xl border border-white/80 bg-surface-dark p-6 text-center shadow-[0_22px_58px_rgba(105,112,175,0.28)]"
      >
        <span
          className={`mx-auto flex h-12 w-12 items-center justify-center rounded-full ${
            outcome.ok ? "bg-secondary/10 text-secondary" : "bg-danger/10 text-danger"
          }`}
        >
          {outcome.ok ? (
            <CheckCircle size={28} weight="duotone" />
          ) : (
            <WarningCircle size={28} weight="duotone" />
          )}
        </span>
        <h3 id="line-webhook-test-title" className="mt-3 text-sm font-semibold text-content">
          {t("admin.line.webhookTestResult")}
        </h3>
        <p className="mt-1.5 text-sm text-muted">{outcome.text}</p>
        <button
          type="button"
          onClick={onClose}
          // Focus moves into the dialog on open so a keyboard user isn't left behind on the page
          // under it. The UA's default outline is replaced with the app's own ring.
          autoFocus
          className="mt-5 w-full rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:ring-offset-2 active:scale-[0.99]"
        >
          {t("a11y.close")}
        </button>
      </div>
    </div>
  );
}

// One of the two LINE channels (login / messaging), with its own configured verdict.
function ChannelCard({
  icon,
  title,
  description,
  configured,
  okLabel,
  badLabel,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  configured: boolean;
  okLabel: string;
  badLabel: string;
}) {
  return (
    <Card>
      <CardHead icon={icon} title={title} />
      <p className="mt-3 text-xs text-muted">{description}</p>
      <div className="mt-auto pt-3">
        <Pill tone={configured ? "ok" : "bad"} icon={configured ? <CheckCircle size={13} weight="fill" /> : undefined}>
          {configured ? okLabel : badLabel}
        </Pill>
      </div>
    </Card>
  );
}

// Yesterday's reply / push count. A single day's number and nothing more: the status endpoint
// returns no history, so there is no series to chart and no previous day to compare against.
function DeliveryCard({ label, value, muted }: { label: string; value: string; muted: boolean }) {
  return (
    <Card>
      <div className="flex items-center gap-1.5 text-faint">
        <PaperPlaneTilt size={16} weight="duotone" />
        <span className="text-xs font-medium">{label}</span>
      </div>
      <p
        className={`mt-2 tabular-nums font-extrabold leading-none ${
          muted ? "text-base text-muted" : "text-3xl text-content"
        }`}
      >
        {value}
      </p>
    </Card>
  );
}
