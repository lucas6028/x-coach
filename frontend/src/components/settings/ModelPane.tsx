import { useEffect, useState } from "react";
import { CheckCircle } from "@phosphor-icons/react";
import { api } from "../../api";
import { useI18n } from "../../lib/i18n";
import { getStoredModel, setStoredModel } from "../../lib/model";
import ModelIcon, { modelLabel } from "../ModelIcon";
import { PaneTitle } from "./parts";

// The LLM that answers follow-up chat, chosen per user (localStorage).
//
// The catalog and the default are server-driven (env-configurable) and fetched from /api/health.
// The fetch lives here rather than in the dialog shell so opening the popup on any other pane
// costs nothing; the shell mounts this pane only when the user selects it.
export default function ModelPane() {
  const { t } = useI18n();
  // `model` is the user's pinned choice ("" = follow the server default).
  const [model, setModel] = useState(getStoredModel);
  const [models, setModels] = useState<string[]>([]);
  const [chatDefault, setChatDefault] = useState("");

  useEffect(() => {
    let active = true;
    api
      .health()
      .then((h) => {
        if (!active) return;
        setModels(h.chat_models ?? []);
        setChatDefault(h.chat_default ?? "");
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  const chooseModel = (id: string) => {
    setStoredModel(id);
    setModel(id);
  };
  // What's shown as selected: the user's pin, or the server default when they haven't pinned one.
  const selectedModel = model || chatDefault;

  return (
    <section>
      <PaneTitle>{t("settings.model")}</PaneTitle>
      <p className="mt-1.5 text-sm text-muted">{t("settings.modelDesc")}</p>
      <fieldset className="mt-4 space-y-1.5">
        <legend className="sr-only">{t("settings.model")}</legend>
        {models.length === 0 ? (
          <p className="text-sm text-muted">{t("settings.modelLoading")}</p>
        ) : (
          models.map((id) => {
            const selected = id === selectedModel;
            const isDefault = id === chatDefault;
            const label = modelLabel(id);
            return (
              <label
                key={id}
                className={`flex cursor-pointer items-center gap-3 rounded-xl border p-3 transition-colors ${
                  selected
                    ? "border-primary/40 bg-primary/[0.06]"
                    : "border-border-dark hover:bg-content/[0.03]"
                }`}
              >
                <input
                  type="radio"
                  name="coach-model"
                  value={id}
                  checked={selected}
                  onChange={() => chooseModel(id)}
                  className="sr-only"
                />
                <span
                  className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                    selected ? "bg-primary/10 ring-1 ring-primary/30" : "bg-content/5"
                  }`}
                >
                  <ModelIcon id={id} size={20} />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2">
                    <span className="truncate font-medium text-content">{label}</span>
                    {isDefault && (
                      <span className="shrink-0 rounded-full bg-content/5 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-faint">
                        {t("settings.modelDefault")}
                      </span>
                    )}
                  </span>
                  {/* Show the raw slug only when it differs from the friendly label. */}
                  {label !== id && (
                    <span className="block truncate font-mono text-xs text-faint">{id}</span>
                  )}
                </span>
                {selected && (
                  <CheckCircle size={20} weight="fill" className="shrink-0 text-primary" />
                )}
              </label>
            );
          })
        )}
      </fieldset>
    </section>
  );
}
