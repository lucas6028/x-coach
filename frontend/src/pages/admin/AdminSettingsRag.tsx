import { useEffect, useState } from "react";
import { Graph } from "@phosphor-icons/react";
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

// Sends only the RAG / KG group's keys.
function toPayload(f: RagForm): AdminSettingsUpdate {
  return {
    rag_top_k: Number(f.rag_top_k),
    kg_hops: Number(f.kg_hops),
    kg_seeds: Number(f.kg_seeds),
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

  if (status === "loading") return <SettingsLoading t={t} />;
  if (status === "error" || !form || !data) return <SettingsLoadError t={t} />;

  const set = (key: keyof RagForm) => (value: string) => setForm((f) => (f ? { ...f, [key]: value } : f));

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
