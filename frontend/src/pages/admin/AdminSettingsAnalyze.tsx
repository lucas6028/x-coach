import { useEffect, useState } from "react";
import { SlidersHorizontal } from "@phosphor-icons/react";
import { api, type AdminSettingsResponse, type AdminSettingsUpdate } from "../../api";
import { useI18n } from "../../lib/i18n";
import {
  Field,
  SaveBar,
  SettingsCard,
  SettingsLoadError,
  SettingsLoading,
  inputClass,
  splitList,
  type SaveState,
} from "./settingsShared";

type Status = "loading" | "ready" | "error";

interface AnalyzeForm {
  allowed_upload_suffixes: string;
  max_concurrent_analyses: string;
}

function toForm(s: AdminSettingsResponse): AnalyzeForm {
  const { analyze } = s.effective;
  return {
    allowed_upload_suffixes: analyze.allowed_upload_suffixes.join(", "),
    max_concurrent_analyses: String(analyze.max_concurrent_analyses),
  };
}

// Sends only the analyze group's keys.
function toPayload(f: AnalyzeForm): AdminSettingsUpdate {
  return {
    allowed_upload_suffixes: splitList(f.allowed_upload_suffixes),
    max_concurrent_analyses: Number(f.max_concurrent_analyses),
  };
}

// Analyze-pipeline settings (admin-only): allowed upload formats + max concurrent analyses. The
// concurrency knob only takes effect on restart (worker pool is built at startup) — surfaced inline.
export default function AdminSettingsAnalyze() {
  const { t } = useI18n();
  const [form, setForm] = useState<AnalyzeForm | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [save, setSave] = useState<SaveState>({ kind: "idle" });

  useEffect(() => {
    let active = true;
    api
      .getAdminSettings()
      .then((res) => {
        if (!active) return;
        setForm(toForm(res));
        setStatus("ready");
      })
      .catch(() => active && setStatus("error"));
    return () => {
      active = false;
    };
  }, []);

  if (status === "loading") return <SettingsLoading t={t} />;
  if (status === "error" || !form) return <SettingsLoadError t={t} />;

  const set = (key: keyof AnalyzeForm) => (value: string) => setForm((f) => (f ? { ...f, [key]: value } : f));

  const onSave = async () => {
    setSave({ kind: "saving" });
    try {
      const res = await api.updateAdminSettings(toPayload(form));
      setForm(toForm(res));
      setSave({ kind: "done" });
    } catch (e) {
      setSave({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  return (
    <div className="space-y-6">
      <SettingsCard
        icon={<SlidersHorizontal size={18} weight="duotone" className="text-primary" />}
        title={t("admin.settings.analyze")}
        desc={t("admin.settings.analyzeDesc")}
      >
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

      <SaveBar t={t} save={save} onSave={() => void onSave()} />
    </div>
  );
}
