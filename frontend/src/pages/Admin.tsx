import { useEffect, useState, type ReactNode } from "react";
import {
  Brain,
  CheckCircle,
  Database,
  FloppyDisk,
  Gauge,
  Graph,
  ShieldCheck,
  ShieldWarning,
  SlidersHorizontal,
  Users,
  WarningCircle,
  XCircle,
} from "@phosphor-icons/react";
import {
  api,
  type AdminOverview as AdminOverviewData,
  type AdminSettingsResponse,
  type AdminSettingsUpdate,
  type AdminUserRow,
} from "../api";
import AppLayout from "../components/AppLayout";
import { useAuth } from "../lib/auth";
import { useI18n, type TFunc } from "../lib/i18n";

type Status = "loading" | "ready" | "error";

// Admin panel. RequireAuth-gated by the router, then this page re-checks the admin role server-side
// (api.adminStatus) — the backend is the real defence; this gating is only UX. Non-admin signed-in
// users get an access-denied card. Admins additionally get the runtime-settings editor (P2).
export default function Admin() {
  const { t } = useI18n();
  const { user } = useAuth();

  const [isAdmin, setIsAdmin] = useState(false);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setStatus("loading");
    setError("");
    api
      .adminStatus()
      .then((res) => {
        if (!active) return;
        setIsAdmin(res.is_admin);
        setStatus("ready");
      })
      .catch((e) => {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      });
    return () => {
      active = false;
    };
  }, [user]);

  return (
    <AppLayout title={t("admin.title")}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-3xl px-4 py-8 lg:px-6 lg:py-12">
          {status === "loading" && <p className="text-sm text-muted">{t("admin.loading")}</p>}

          {status === "error" && (
            <div className="flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
              <WarningCircle size={18} className="shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="font-medium">{t("admin.error")}</p>
                <p className="mt-0.5 break-words text-danger/80">{error}</p>
              </div>
            </div>
          )}

          {status === "ready" && !isAdmin && (
            <div className="flex flex-col items-center gap-4 rounded-2xl border border-dashed border-border-dark bg-content/[0.02] px-6 py-16 text-center">
              <span className="flex h-14 w-14 items-center justify-center rounded-full bg-danger/10 text-danger">
                <ShieldWarning size={30} weight="duotone" />
              </span>
              <p className="font-medium text-content">{t("admin.denied")}</p>
            </div>
          )}

          {status === "ready" && isAdmin && (
            <section className="space-y-8">
              <div>
                <h1 className="flex items-center gap-2 text-lg font-semibold text-content">
                  <ShieldCheck size={22} weight="duotone" className="text-primary" />
                  {t("admin.title")}
                </h1>
                <p className="mt-1.5 text-sm text-muted">{t("admin.subtitle")}</p>
              </div>
              <AdminOverview t={t} />
              <AdminUsers t={t} currentUserId={user?.id} />
              <AdminSettings t={t} />
            </section>
          )}
        </main>
      </div>
    </AppLayout>
  );
}

// --- System overview dashboard (admin-only, read-only) --------------------------------------------

function fmtDate(iso: string | null, never: string): string {
  if (!iso) return never;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? never : d.toLocaleDateString();
}

// Read-only health + usage cards at the top of the admin panel (mirrors /api/health + totals).
function AdminOverview({ t }: { t: TFunc }) {
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

  if (status === "loading") return null;
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

// --- Users oversight + in-app role assignment (admin-only) ----------------------------------------

// Read-only users table with a per-row admin toggle. The current user's own row is non-toggleable so
// an admin cannot lock themselves out (the backend also rejects self-demotion with a 400).
function AdminUsers({ t, currentUserId }: { t: TFunc; currentUserId?: string }) {
  const [rows, setRows] = useState<AdminUserRow[] | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [rowError, setRowError] = useState<{ id: string; message: string } | null>(null);

  const load = () => {
    setStatus("loading");
    api
      .listAdminUsers()
      .then((res) => {
        setRows(res.users);
        setStatus("ready");
      })
      .catch(() => setStatus("error"));
  };

  useEffect(load, []);

  const onToggle = async (row: AdminUserRow) => {
    setPendingId(row.id);
    setRowError(null);
    try {
      await api.setUserRole(row.id, !row.is_admin);
      const res = await api.listAdminUsers();
      setRows(res.users);
    } catch (e) {
      setRowError({ id: row.id, message: e instanceof Error ? e.message : String(e) });
    } finally {
      setPendingId(null);
    }
  };

  return (
    <div>
      <div className="flex items-center gap-2">
        <Users size={18} weight="duotone" className="text-primary" />
        <h2 className="text-sm font-semibold text-content">{t("admin.users.title")}</h2>
      </div>
      <p className="mt-1 text-xs text-muted">{t("admin.users.desc")}</p>

      {status === "loading" && <p className="mt-4 text-sm text-muted">{t("admin.users.loading")}</p>}
      {status === "error" && (
        <div className="mt-4 flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
          <WarningCircle size={18} className="shrink-0" />
          <p className="font-medium">{t("admin.users.loadError")}</p>
        </div>
      )}

      {status === "ready" && rows && rows.length === 0 && (
        <p className="mt-4 text-sm text-muted">{t("admin.users.empty")}</p>
      )}

      {status === "ready" && rows && rows.length > 0 && (
        <div className="mt-4 overflow-x-auto rounded-2xl border border-border-dark bg-surface-dark">
          <table className="w-full min-w-[40rem] text-left text-sm">
            <thead>
              <tr className="border-b border-border-dark text-xs font-medium text-faint">
                <th className="px-4 py-3">{t("admin.users.email")}</th>
                <th className="px-4 py-3">{t("admin.users.created")}</th>
                <th className="px-4 py-3">{t("admin.users.lastSignIn")}</th>
                <th className="px-4 py-3 text-right tabular-nums">{t("admin.users.analyses")}</th>
                <th className="px-4 py-3 text-right tabular-nums">{t("admin.users.conversations")}</th>
                <th className="px-4 py-3 text-right">{t("admin.users.role")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isSelf = row.id === currentUserId;
                const pending = pendingId === row.id;
                return (
                  <tr key={row.id} className="border-b border-border-dark/60 last:border-0">
                    <td className="px-4 py-3 text-content">
                      <span className="break-all">{row.email ?? row.id}</span>
                      {isSelf && (
                        <span className="ml-2 rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                          {t("admin.users.you")}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted">{fmtDate(row.created_at, t("admin.users.never"))}</td>
                    <td className="px-4 py-3 text-muted">{fmtDate(row.last_sign_in_at, t("admin.users.never"))}</td>
                    <td className="px-4 py-3 text-right tabular-nums text-content">{row.analyses_count}</td>
                    <td className="px-4 py-3 text-right tabular-nums text-content">{row.conversations_count}</td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => void onToggle(row)}
                        disabled={isSelf || pending}
                        aria-label={row.is_admin ? t("admin.users.revokeAdmin") : t("admin.users.makeAdmin")}
                        className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                          row.is_admin
                            ? "border-secondary/40 bg-secondary/10 text-secondary hover:bg-secondary/20"
                            : "border-border-dark bg-content/[0.02] text-faint hover:bg-content/[0.05]"
                        }`}
                      >
                        {row.is_admin ? <CheckCircle size={14} weight="fill" /> : <XCircle size={14} />}
                        {row.is_admin ? t("admin.users.revokeAdmin") : t("admin.users.makeAdmin")}
                      </button>
                      {rowError?.id === row.id && (
                        <p className="mt-1 text-[11px] text-danger">{t("admin.users.updateError")}</p>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

type SaveState =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "done" }
  | { kind: "error"; message: string };

// Editable string mirror of the settings so inputs stay controlled; parsed into the API payload on save.
interface FormState {
  llm_models: string;
  llm_followup_model: string;
  llm_base_url: string;
  chat_temperature: string;
  chat_timeout: string;
  followup_timeout: string;
  rag_top_k: string;
  kg_hops: string;
  kg_seeds: string;
  allowed_upload_suffixes: string;
  max_concurrent_analyses: string;
}

function toForm(s: AdminSettingsResponse): FormState {
  const { llm, rag_kg, analyze } = s.effective;
  return {
    llm_models: llm.llm_models.join("\n"),
    llm_followup_model: llm.llm_followup_model,
    llm_base_url: llm.llm_base_url,
    chat_temperature: llm.chat_temperature === null ? "" : String(llm.chat_temperature),
    chat_timeout: String(llm.chat_timeout),
    followup_timeout: String(llm.followup_timeout),
    rag_top_k: String(rag_kg.rag_top_k),
    kg_hops: String(rag_kg.kg_hops),
    kg_seeds: String(rag_kg.kg_seeds),
    allowed_upload_suffixes: analyze.allowed_upload_suffixes.join(", "),
    max_concurrent_analyses: String(analyze.max_concurrent_analyses),
  };
}

const splitList = (raw: string): string[] =>
  raw
    .split(/[\n,]/)
    .map((x) => x.trim())
    .filter(Boolean);

function toPayload(f: FormState): AdminSettingsUpdate {
  return {
    llm_models: splitList(f.llm_models),
    llm_followup_model: f.llm_followup_model.trim(),
    llm_base_url: f.llm_base_url.trim(),
    chat_temperature: f.chat_temperature.trim() === "" ? null : Number(f.chat_temperature),
    chat_timeout: Number(f.chat_timeout),
    followup_timeout: Number(f.followup_timeout),
    rag_top_k: Number(f.rag_top_k),
    kg_hops: Number(f.kg_hops),
    kg_seeds: Number(f.kg_seeds),
    allowed_upload_suffixes: splitList(f.allowed_upload_suffixes),
    max_concurrent_analyses: Number(f.max_concurrent_analyses),
  };
}

// The runtime-settings editor (admin-only). Loads the effective values + defaults, lets the admin
// retune the LLM / RAG-KG / analysis knobs, and persists them (mirrors Settings.tsx's clear-state save).
function AdminSettings({ t }: { t: TFunc }) {
  const [data, setData] = useState<AdminSettingsResponse | null>(null);
  const [form, setForm] = useState<FormState | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [save, setSave] = useState<SaveState>({ kind: "idle" });

  useEffect(() => {
    let active = true;
    api
      .getAdminSettings()
      .then((res) => {
        if (!active) return;
        setData(res);
        setForm(toForm(res));
        setStatus("ready");
      })
      .catch(() => {
        if (!active) return;
        setStatus("error");
      });
    return () => {
      active = false;
    };
  }, []);

  if (status === "loading") return <p className="mt-6 text-sm text-muted">{t("admin.settings.loading")}</p>;
  if (status === "error" || !form || !data)
    return (
      <div className="mt-6 flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
        <WarningCircle size={18} className="shrink-0" />
        <p className="font-medium">{t("admin.settings.loadError")}</p>
      </div>
    );

  const set = (key: keyof FormState) => (value: string) => setForm((f) => (f ? { ...f, [key]: value } : f));

  const onSave = async () => {
    setSave({ kind: "saving" });
    try {
      const res = await api.updateAdminSettings(toPayload(form));
      setData(res);
      setForm(toForm(res));
      setSave({ kind: "done" });
    } catch (e) {
      setSave({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const d = data.defaults;

  return (
    <div className="mt-6 space-y-6">
      {/* LLM chat settings */}
      <SettingsCard icon={<Brain size={18} weight="duotone" className="text-primary" />} title={t("admin.settings.llm")} desc={t("admin.settings.llmDesc")}>
        <Field id="llm_models" label={t("admin.settings.models")} hint={t("admin.settings.modelsHint")}>
          <textarea
            id="llm_models"
            rows={4}
            value={form.llm_models}
            onChange={(e) => set("llm_models")(e.target.value)}
            className={textareaClass}
          />
        </Field>
        <Field id="llm_followup_model" label={t("admin.settings.followupModel")} hint={defaultHint(t, d.llm.llm_followup_model)}>
          <input id="llm_followup_model" value={form.llm_followup_model} onChange={(e) => set("llm_followup_model")(e.target.value)} className={inputClass} />
        </Field>
        <Field id="llm_base_url" label={t("admin.settings.baseUrl")} hint={defaultHint(t, d.llm.llm_base_url)}>
          <input id="llm_base_url" value={form.llm_base_url} onChange={(e) => set("llm_base_url")(e.target.value)} className={inputClass} />
        </Field>
        <div className="grid gap-4 sm:grid-cols-3">
          <Field id="chat_temperature" label={t("admin.settings.temperature")} hint={t("admin.settings.temperatureHint")}>
            <input id="chat_temperature" inputMode="decimal" value={form.chat_temperature} onChange={(e) => set("chat_temperature")(e.target.value)} className={inputClass} />
          </Field>
          <Field id="chat_timeout" label={t("admin.settings.chatTimeout")} hint={defaultHint(t, d.llm.chat_timeout)}>
            <input id="chat_timeout" inputMode="numeric" value={form.chat_timeout} onChange={(e) => set("chat_timeout")(e.target.value)} className={inputClass} />
          </Field>
          <Field id="followup_timeout" label={t("admin.settings.followupTimeout")} hint={defaultHint(t, d.llm.followup_timeout)}>
            <input id="followup_timeout" inputMode="numeric" value={form.followup_timeout} onChange={(e) => set("followup_timeout")(e.target.value)} className={inputClass} />
          </Field>
        </div>
      </SettingsCard>

      {/* RAG / KG */}
      <SettingsCard icon={<Graph size={18} weight="duotone" className="text-primary" />} title={t("admin.settings.ragkg")} desc={t("admin.settings.ragkgDesc")}>
        <div className="grid gap-4 sm:grid-cols-3">
          <Field id="rag_top_k" label={t("admin.settings.ragTopK")} hint={defaultHint(t, d.rag_kg.rag_top_k)}>
            <input id="rag_top_k" inputMode="numeric" value={form.rag_top_k} onChange={(e) => set("rag_top_k")(e.target.value)} className={inputClass} />
          </Field>
          <Field id="kg_hops" label={t("admin.settings.kgHops")} hint={defaultHint(t, d.rag_kg.kg_hops)}>
            <input id="kg_hops" inputMode="numeric" value={form.kg_hops} onChange={(e) => set("kg_hops")(e.target.value)} className={inputClass} />
          </Field>
          <Field id="kg_seeds" label={t("admin.settings.kgSeeds")} hint={defaultHint(t, d.rag_kg.kg_seeds)}>
            <input id="kg_seeds" inputMode="numeric" value={form.kg_seeds} onChange={(e) => set("kg_seeds")(e.target.value)} className={inputClass} />
          </Field>
        </div>
      </SettingsCard>

      {/* Analysis pipeline */}
      <SettingsCard icon={<SlidersHorizontal size={18} weight="duotone" className="text-primary" />} title={t("admin.settings.analyze")} desc={t("admin.settings.analyzeDesc")}>
        <Field id="allowed_upload_suffixes" label={t("admin.settings.uploadFormats")} hint={t("admin.settings.uploadFormatsHint")}>
          <input id="allowed_upload_suffixes" value={form.allowed_upload_suffixes} onChange={(e) => set("allowed_upload_suffixes")(e.target.value)} className={inputClass} />
        </Field>
        <Field
          id="max_concurrent_analyses"
          label={t("admin.settings.maxConcurrent")}
          hint={t("admin.settings.restartRequired")}
          hintDanger
        >
          <input id="max_concurrent_analyses" inputMode="numeric" value={form.max_concurrent_analyses} onChange={(e) => set("max_concurrent_analyses")(e.target.value)} className={inputClass} />
        </Field>
      </SettingsCard>

      {/* Save */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => void onSave()}
          disabled={save.kind === "saving"}
          className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-primary/90 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
        >
          <FloppyDisk size={16} weight="fill" />
          {save.kind === "saving" ? t("admin.settings.saving") : t("admin.settings.save")}
        </button>
        {save.kind === "done" && (
          <p className="flex items-center gap-1.5 text-sm text-secondary">
            <CheckCircle size={16} weight="fill" />
            {t("admin.settings.saved")}
          </p>
        )}
        {save.kind === "error" && (
          <p className="flex items-center gap-1.5 text-sm text-danger">
            <WarningCircle size={16} weight="fill" />
            {t("admin.settings.saveError")}
          </p>
        )}
      </div>
    </div>
  );
}

const inputClass =
  "w-full rounded-xl border border-border-dark bg-content/[0.02] px-3 py-2 text-sm text-content outline-none transition-colors focus:border-primary/50 focus:bg-content/[0.04]";
const textareaClass = `${inputClass} font-mono resize-y`;

function defaultHint(t: TFunc, value: string | number): string {
  return t("admin.settings.defaultLabel", { value: String(value) });
}

function SettingsCard({
  icon,
  title,
  desc,
  children,
}: {
  icon: ReactNode;
  title: string;
  desc: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-border-dark bg-surface-dark p-5">
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-sm font-semibold text-content">{title}</h2>
      </div>
      <p className="mt-1 text-xs text-muted">{desc}</p>
      <div className="mt-4 space-y-4">{children}</div>
    </div>
  );
}

function Field({
  id,
  label,
  hint,
  hintDanger,
  children,
}: {
  id: string;
  label: string;
  hint?: string;
  hintDanger?: boolean;
  children: ReactNode;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-faint">
        {label}
      </label>
      <div className="mt-1.5">{children}</div>
      {hint && (
        <p className={`mt-1 text-xs ${hintDanger ? "text-danger/80" : "text-faint"}`}>{hint}</p>
      )}
    </div>
  );
}
