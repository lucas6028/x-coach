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
  defaultHint,
  inputClass,
  parseRequiredNumbers,
  splitList,
  type SaveState,
} from "./settingsShared";

type Status = "loading" | "ready" | "error";

interface AnalyzeForm {
  allowed_upload_suffixes: string;
  max_upload_bytes: string;
  user_storage_quota_bytes: string;
}

function toForm(s: AdminSettingsResponse): AnalyzeForm {
  const { analyze } = s.effective;
  return {
    allowed_upload_suffixes: analyze.allowed_upload_suffixes.join(", "),
    max_upload_bytes: String(analyze.max_upload_bytes),
    user_storage_quota_bytes: String(analyze.user_storage_quota_bytes),
  };
}

// Analyze-pipeline settings (admin-only): allowed upload formats and the two upload limits are
// editable; max concurrent analyses is env-var-driven (XCOACH_MAX_CONCURRENT_ANALYSES, applied at
// startup) so it's shown read-only and never included in the update payload.
//
// The two limits are edited in BYTES, the same unit the backend stores and validates in, so the
// form can't introduce a rounding mismatch against the server's ge/le bounds.
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
    // Same guard the RAG page uses for its required integers: bail with an honest error rather than
    // sending NaN (→ stored null → silent reset) or 0 (→ backend ge → opaque 422).
    const nums = parseRequiredNumbers({
      max_upload_bytes: form.max_upload_bytes,
      user_storage_quota_bytes: form.user_storage_quota_bytes,
    });
    if (!nums) {
      setSave({ kind: "error", message: t("admin.settings.invalidNumber") });
      return;
    }
    setSave({ kind: "saving" });
    try {
      const res = await api.updateAdminSettings({
        allowed_upload_suffixes: splitList(form.allowed_upload_suffixes),
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
        <div className="grid gap-4 sm:grid-cols-2">
          <Field id="max_upload_bytes" label={t("admin.settings.maxUploadBytes")} hint={defaultHint(t, d.analyze.max_upload_bytes)}>
            <input id="max_upload_bytes" inputMode="numeric" value={form.max_upload_bytes} onChange={(e) => set("max_upload_bytes")(e.target.value)} className={inputClass} />
          </Field>
          <Field id="user_storage_quota_bytes" label={t("admin.settings.userStorageQuotaBytes")} hint={defaultHint(t, d.analyze.user_storage_quota_bytes)}>
            <input id="user_storage_quota_bytes" inputMode="numeric" value={form.user_storage_quota_bytes} onChange={(e) => set("user_storage_quota_bytes")(e.target.value)} className={inputClass} />
          </Field>
        </div>
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
