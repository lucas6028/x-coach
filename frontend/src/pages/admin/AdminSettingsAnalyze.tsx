import { useEffect, useState } from "react";
import { SlidersHorizontal } from "@phosphor-icons/react";
import { api, type AdminSettingsResponse } from "../../api";
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
}

function toForm(s: AdminSettingsResponse): AnalyzeForm {
  const { analyze } = s.effective;
  return {
    allowed_upload_suffixes: analyze.allowed_upload_suffixes.join(", "),
  };
}

// Analyze-pipeline settings (admin-only): allowed upload formats are editable; max concurrent
// analyses is env-var-driven (XCOACH_MAX_CONCURRENT_ANALYSES, applied at startup) so it's shown
// read-only and never included in the update payload.
export default function AdminSettingsAnalyze() {
  const { t } = useI18n();
  const [data, setData] = useState<AdminSettingsResponse | null>(null);
  const [form, setForm] = useState<AnalyzeForm | null>(null);
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

  const set = (key: keyof AnalyzeForm) => (value: string) => setForm((f) => (f ? { ...f, [key]: value } : f));

  const onSave = async () => {
    setSave({ kind: "saving" });
    try {
      const res = await api.updateAdminSettings({
        allowed_upload_suffixes: splitList(form.allowed_upload_suffixes),
      });
      setData(res);
      setForm(toForm(res));
      setSave({ kind: "done" });
    } catch {
      setSave({ kind: "error", message: t("admin.settings.saveError") });
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
        {/* Read-only: max concurrency is fixed at startup from an env var, not editable at runtime. */}
        <Field label={t("admin.settings.maxConcurrent")} hint={t("admin.settings.maxConcurrentReadonly")}>
          <p className="rounded-xl border border-border-dark bg-content/[0.04] px-3 py-2 text-sm text-muted">
            {data.effective.analyze.max_concurrent_analyses}
          </p>
        </Field>
      </SettingsCard>

      <SaveBar t={t} save={save} onSave={() => void onSave()} />
    </div>
  );
}
