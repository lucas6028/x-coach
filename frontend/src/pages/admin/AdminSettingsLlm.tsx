import { useEffect, useState } from "react";
import { Brain } from "@phosphor-icons/react";
import { api, type AdminSettingsResponse, type AdminSettingsUpdate } from "../../api";
import { useI18n } from "../../lib/i18n";
import {
  Field,
  SaveBar,
  SettingsCard,
  SettingsLoadError,
  SettingsLoading,
  defaultHint,
  inputClass,
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

// Sends only the LLM group's keys — getAdminSettings returns everything, updateAdminSettings upserts
// just what we pass.
function toPayload(f: LlmForm): AdminSettingsUpdate {
  return {
    llm_models: splitList(f.llm_models),
    llm_followup_model: f.llm_followup_model.trim(),
    llm_base_url: f.llm_base_url.trim(),
    chat_temperature: f.chat_temperature.trim() === "" ? null : Number(f.chat_temperature),
    chat_timeout: Number(f.chat_timeout),
    followup_timeout: Number(f.followup_timeout),
  };
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

  if (status === "loading") return <SettingsLoading t={t} />;
  if (status === "error" || !form || !data) return <SettingsLoadError t={t} />;

  const set = (key: keyof LlmForm) => (value: string) => setForm((f) => (f ? { ...f, [key]: value } : f));

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
    <div className="space-y-6">
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

      <SaveBar t={t} save={save} onSave={() => void onSave()} />
    </div>
  );
}
