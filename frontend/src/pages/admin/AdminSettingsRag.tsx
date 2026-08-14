import { useEffect, useState } from "react";
import { Graph } from "@phosphor-icons/react";
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
  type SaveState,
} from "./settingsShared";

type Status = "loading" | "ready" | "error";

interface RagForm {
  rag_top_k: string;
  kg_hops: string;
  kg_seeds: string;
}

function toForm(s: AdminSettingsResponse): RagForm {
  const { rag_kg } = s.effective;
  return {
    rag_top_k: String(rag_kg.rag_top_k),
    kg_hops: String(rag_kg.kg_hops),
    kg_seeds: String(rag_kg.kg_seeds),
  };
}

// RAG / KG retrieval settings (admin-only): rag_top_k, kg_hops, kg_seeds.
export default function AdminSettingsRag() {
  const { t } = useI18n();
  const [data, setData] = useState<AdminSettingsResponse | null>(null);
  const [form, setForm] = useState<RagForm | null>(null);
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

  const set = (key: keyof RagForm) => (value: string) => setForm((f) => (f ? { ...f, [key]: value } : f));

  const onSave = async () => {
    // Guard the required positive integers on the client: bail with an honest error rather than
    // sending NaN (→ stored null → silent reset) or 0 (→ backend ge=1 → opaque 422).
    const nums = parseRequiredNumbers({
      rag_top_k: form.rag_top_k,
      kg_hops: form.kg_hops,
      kg_seeds: form.kg_seeds,
    });
    if (!nums) {
      setSave({ kind: "error", message: t("admin.settings.invalidNumber") });
      return;
    }
    setSave({ kind: "saving" });
    try {
      const res = await api.updateAdminSettings(nums);
      setData(res);
      setForm(toForm(res));
      setSave({ kind: "done" });
    } catch {
      setSave({ kind: "error", message: t("admin.settings.saveError") });
    }
  };

  const d = data.defaults;

  // Long forms keep their own reading measure: the admin column is now wide enough for the LINE
  // page's four-across card row, which would otherwise stretch these label+input rows across it.
  return (
    <div className="max-w-3xl space-y-6">
      <SettingsCard
        icon={<Graph size={18} weight="duotone" className="text-primary" />}
        title={t("admin.settings.ragkg")}
        desc={t("admin.settings.ragkgDesc")}
      >
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

      <SaveBar t={t} save={save} onSave={() => void onSave()} />
    </div>
  );
}
