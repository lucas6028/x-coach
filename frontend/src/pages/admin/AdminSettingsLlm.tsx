import { useEffect, useState } from "react";
import { Brain, Warning } from "@phosphor-icons/react";
import {
  api,
  type AdminLlmModelEntry,
  type AdminLlmModelRole,
  type AdminLlmModelStatus,
  type AdminLlmModelsResponse,
  type AdminSettingsResponse,
} from "../../api";
import { useI18n, type TFunc } from "../../lib/i18n";
import {
  Field,
  SaveBar,
  SettingsCard,
  SettingsLoadError,
  SettingsLoading,
  defaultHint,
  inputClass,
  parseNumber,
  parseRequiredNumbers,
  splitList,
  textareaClass,
  type SaveState,
} from "./settingsShared";

type Status = "loading" | "ready" | "error";

interface LlmForm {
  llm_models: string;
  llm_followup_model: string;
  llm_base_url: string;
  chat_temperature: string;
  chat_timeout: string;
  followup_timeout: string;
}

function toForm(s: AdminSettingsResponse): LlmForm {
  const { llm } = s.effective;
  return {
    llm_models: llm.llm_models.join("\n"),
    llm_followup_model: llm.llm_followup_model,
    llm_base_url: llm.llm_base_url,
    chat_temperature: llm.chat_temperature === null ? "" : String(llm.chat_temperature),
    chat_timeout: String(llm.chat_timeout),
    followup_timeout: String(llm.followup_timeout),
  };
}

// Availability status → the badge tone + i18n key, distinct per state so a glance tells the admin
// what to do: available needs nothing, not_listed/unavailable mean users won't get this model,
// unknown means the catalog check itself failed (fail-open — the model is still offered).
const MODEL_STATUS_STYLE: Record<AdminLlmModelStatus, string> = {
  available: "bg-secondary/10 text-secondary",
  not_listed: "bg-content/10 text-faint",
  unavailable: "bg-danger/10 text-danger",
  unknown: "bg-warning/10 text-warning",
};
const MODEL_STATUS_KEY: Record<AdminLlmModelStatus, string> = {
  available: "admin.settings.modelStatusAvailable",
  not_listed: "admin.settings.modelStatusNotListed",
  unavailable: "admin.settings.modelStatusUnavailable",
  unknown: "admin.settings.modelStatusUnknown",
};
const MODEL_ROLE_KEY: Record<AdminLlmModelRole, string> = {
  default: "admin.settings.modelRoleDefault",
  option: "admin.settings.modelRoleOption",
  followup: "admin.settings.modelRoleFollowup",
};

function ModelStatusBadge({ status, t }: { status: AdminLlmModelStatus; t: TFunc }) {
  return (
    <span
      className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold ${MODEL_STATUS_STYLE[status]}`}
    >
      {t(MODEL_STATUS_KEY[status])}
    </span>
  );
}

// One row of the availability table: id, its role chips, the status badge, and — when present — an
// expiry warning and a status detail (e.g. "HTTP 410").
function ModelRow({ model, t }: { model: AdminLlmModelEntry; t: TFunc }) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-border-dark bg-content/[0.02] px-3 py-2">
      <span className="min-w-0 flex-1 truncate font-mono text-xs text-content">{model.id}</span>
      {model.roles.map((role) => (
        <span
          key={role}
          className="shrink-0 rounded-full bg-content/5 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-faint"
        >
          {t(MODEL_ROLE_KEY[role])}
        </span>
      ))}
      <ModelStatusBadge status={model.status} t={t} />
      {model.expires && (
        <span className="flex shrink-0 items-center gap-1 text-[11px] font-medium text-warning">
          <Warning size={12} weight="fill" />
          {t("admin.settings.modelExpires", { date: model.expires })}
        </span>
      )}
      {model.detail && <span className="w-full text-xs text-faint">{model.detail}</span>}
    </div>
  );
}

// The catalog fetch's own summary line: when it last succeeded/failed and how many models it saw.
function CatalogLine({ catalog, t }: { catalog: AdminLlmModelsResponse["catalog"]; t: TFunc }) {
  if (catalog.status === "ok") {
    return (
      <p className="text-xs text-muted">
        {t("admin.settings.modelCatalogOk", {
          count: catalog.model_count ?? 0,
          when: catalog.checked_at ?? "—",
        })}
      </p>
    );
  }
  if (catalog.status === "error") {
    return (
      <p className="text-xs text-danger">
        {t("admin.settings.modelCatalogError", { error: catalog.error ?? "" })}
      </p>
    );
  }
  return <p className="text-xs text-muted">{t("admin.settings.modelCatalogUnknown")}</p>;
}

// The "model availability" section: catalog summary, a manual re-check, and every configured
// model's live status. Fetched independently of the settings form above (own loading/error state)
// so a slow/failed provider check never blocks the editable form from rendering.
function ModelAvailabilitySection({ data, t }: { data: AdminSettingsResponse; t: TFunc }) {
  const [models, setModels] = useState<AdminLlmModelsResponse | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    let active = true;
    api
      .getAdminLlmModels()
      .then((res) => {
        if (!active) return;
        setModels(res);
        setStatus("ready");
      })
      .catch(() => active && setStatus("error"));
    return () => {
      active = false;
    };
  }, []);

  async function checkNow() {
    setRefreshing(true);
    try {
      const res = await api.getAdminLlmModels(true);
      setModels(res);
      setStatus("ready");
    } catch {
      setStatus("error");
    } finally {
      setRefreshing(false);
    }
  }

  const configuredDefault = data.effective.llm.llm_models[0] ?? "";
  const configuredFollowup = data.effective.llm.llm_followup_model;

  return (
    <SettingsCard
      icon={<Brain size={18} weight="duotone" className="text-primary" />}
      title={t("admin.settings.modelAvailability")}
      desc={t("admin.settings.modelAvailabilityDesc")}
    >
      <div className="flex items-center justify-between gap-3">
        {status === "ready" && models ? (
          <CatalogLine catalog={models.catalog} t={t} />
        ) : status === "error" ? (
          <p className="text-xs text-danger">{t("admin.settings.modelLoadError")}</p>
        ) : (
          <p className="text-xs text-muted">{t("admin.settings.modelChecking")}</p>
        )}
        <button
          type="button"
          onClick={() => void checkNow()}
          disabled={refreshing}
          className="shrink-0 rounded-xl border border-primary/25 px-3 py-1.5 text-xs font-semibold text-primary transition-colors hover:bg-primary/5 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {refreshing ? t("admin.settings.modelChecking") : t("admin.settings.modelCheckNow")}
        </button>
      </div>

      {models && (
        <>
          {models.effective_default !== configuredDefault && (
            <p className="flex items-start gap-1.5 text-xs font-medium text-warning">
              <Warning size={14} weight="fill" className="mt-0.5 shrink-0" />
              {t("admin.settings.modelDefaultFallback", {
                configured: configuredDefault,
                effective: models.effective_default,
              })}
            </p>
          )}
          {models.effective_followup !== configuredFollowup && (
            <p className="flex items-start gap-1.5 text-xs font-medium text-warning">
              <Warning size={14} weight="fill" className="mt-0.5 shrink-0" />
              {t("admin.settings.modelFollowupFallback", {
                configured: configuredFollowup,
                effective: models.effective_followup,
              })}
            </p>
          )}
          <div className="space-y-1.5">
            {models.models.map((m) => (
              <ModelRow key={m.id} model={m} t={t} />
            ))}
          </div>
        </>
      )}
    </SettingsCard>
  );
}

// LLM chat runtime settings (admin-only): model list, follow-up model, base URL, temperature, timeouts.
export default function AdminSettingsLlm() {
  const { t } = useI18n();
  const [data, setData] = useState<AdminSettingsResponse | null>(null);
  const [form, setForm] = useState<LlmForm | null>(null);
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
      .catch(() => active && setStatus("error"));
    return () => {
      active = false;
    };
  }, []);

  if (status === "loading") return <SettingsLoading />;
  if (status === "error" || !form || !data) return <SettingsLoadError t={t} />;

  const set = (key: keyof LlmForm) => (value: string) => setForm((f) => (f ? { ...f, [key]: value } : f));

  const onSave = async () => {
    // Guard the required positive-integer timeouts on the client (empty/typo → honest error, not a
    // silent NaN reset or an opaque 422). chat_temperature stays optional: empty = "use default".
    const nums = parseRequiredNumbers({
      chat_timeout: form.chat_timeout,
      followup_timeout: form.followup_timeout,
    });
    if (!nums) {
      setSave({ kind: "error", message: t("admin.settings.invalidNumber") });
      return;
    }
    setSave({ kind: "saving" });
    try {
      const res = await api.updateAdminSettings({
        llm_models: splitList(form.llm_models),
        llm_followup_model: form.llm_followup_model.trim(),
        llm_base_url: form.llm_base_url.trim(),
        chat_temperature: parseNumber(form.chat_temperature),
        ...nums,
      });
      setData(res);
      setForm(toForm(res));
      setSave({ kind: "done" });
    } catch {
      setSave({ kind: "error", message: t("admin.settings.saveError") });
    }
  };

  const d = data.defaults;

  // Long forms keep their own reading measure — see the note in AdminSettingsRag.
  return (
    <div className="max-w-3xl space-y-6">
      <SettingsCard
        icon={<Brain size={18} weight="duotone" className="text-primary" />}
        title={t("admin.settings.llm")}
        desc={t("admin.settings.llmDesc")}
      >
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

      <ModelAvailabilitySection data={data} t={t} />

      <SaveBar t={t} save={save} onSave={() => void onSave()} />
    </div>
  );
}
