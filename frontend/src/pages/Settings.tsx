import { useEffect, useState } from "react";
import { CheckCircle, ClockCounterClockwise, Trash, WarningCircle } from "@phosphor-icons/react";
import { api } from "../api";
import AppLayout from "../components/AppLayout";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";
import { getStoredModel, setStoredModel } from "../lib/model";
import { avatarUrl, displayName, initial } from "../lib/profile";
import ModelIcon, { modelLabel } from "../components/ModelIcon";

type ClearState =
  | { kind: "idle" }
  | { kind: "confirm" }
  | { kind: "working" }
  | { kind: "done"; deleted: number }
  | { kind: "error"; message: string };

// Account settings: profile display + a danger zone to clear saved analyses.
// Product UI — kept in the app's token system, RequireAuth-gated by the router.
export default function Settings() {
  const { t } = useI18n();
  const { user } = useAuth();
  const [imgError, setImgError] = useState(false);
  const [clear, setClear] = useState<ClearState>({ kind: "idle" });
  // Model picker: the catalog + default are server-driven (env-configurable), fetched from health;
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

  if (!user) return null;

  const url = imgError ? null : avatarUrl(user);
  const name = displayName(user);
  const provider = (user.app_metadata?.provider as string) ?? "email";
  // LINE users arrive two ways: Supabase custom OIDC on the web (provider "custom:line")
  // or the LIFF bridge (provider "email" + a line_sub in user_metadata). Exact matches
  // only — `includes("line")` would also catch "linkedin".
  const isLineUser =
    provider === "custom:line" || provider === "line" || Boolean(user.user_metadata?.line_sub);
  const providerLabel = isLineUser
    ? t("settings.provider.line")
    : provider === "google"
      ? t("settings.provider.google")
      : t("settings.provider.email");

  const runClear = async () => {
    setClear({ kind: "working" });
    try {
      const { deleted } = await api.deleteAnalyses();
      setClear({ kind: "done", deleted });
    } catch (e) {
      setClear({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  return (
    <AppLayout title={t("settings.title")}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-3xl px-4 py-8 lg:px-6 lg:py-12">
          <p className="text-sm text-muted">{t("settings.subtitle")}</p>

        {/* Profile */}
        <section className="mt-8">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-faint">
            {t("settings.profile")}
          </h2>
          <div className="mt-3 flex items-center gap-4 rounded-2xl border border-border-dark bg-surface-dark p-5">
            {url ? (
              <img
                src={url}
                alt=""
                referrerPolicy="no-referrer"
                onError={() => setImgError(true)}
                className="h-16 w-16 shrink-0 rounded-full object-cover ring-1 ring-border-dark"
              />
            ) : (
              <span className="flex h-16 w-16 shrink-0 items-center justify-center rounded-full bg-primary/15 text-xl font-semibold text-primary ring-1 ring-border-dark">
                {initial(user)}
              </span>
            )}
            <dl className="min-w-0 flex-1 space-y-1.5">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <dt className="w-20 shrink-0 text-xs text-faint">{t("settings.name")}</dt>
                <dd className="min-w-0 truncate font-medium text-content">{name}</dd>
              </div>
              <div className="flex flex-wrap items-baseline gap-x-2">
                <dt className="w-20 shrink-0 text-xs text-faint">{t("settings.email")}</dt>
                <dd className="min-w-0 truncate text-sm text-muted">{user.email}</dd>
              </div>
              <div className="flex flex-wrap items-baseline gap-x-2">
                <dt className="w-20 shrink-0 text-xs text-faint">{t("settings.provider")}</dt>
                <dd className="min-w-0 truncate text-sm text-muted">{providerLabel}</dd>
              </div>
            </dl>
          </div>
        </section>

        {/* Coach model — the LLM that answers follow-up chat, chosen per user (localStorage). */}
        <section className="mt-10">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-faint">
            {t("settings.model")}
          </h2>
          <p className="mt-1.5 text-sm text-muted">{t("settings.modelDesc")}</p>
          <fieldset className="mt-3 divide-y divide-border-dark overflow-hidden rounded-2xl border border-border-dark bg-surface-dark">
            <legend className="sr-only">{t("settings.model")}</legend>
            {models.length === 0 ? (
              <p className="p-4 text-sm text-muted">{t("settings.modelLoading")}</p>
            ) : (
              models.map((id) => {
                const selected = id === selectedModel;
                const isDefault = id === chatDefault;
                const label = modelLabel(id);
                return (
                  <label
                    key={id}
                    className="flex cursor-pointer items-center gap-3 p-4 transition-colors hover:bg-content/[0.03]"
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

        {/* Danger zone */}
        <section className="mt-10">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-danger/80">
            {t("settings.danger")}
          </h2>
          <div className="mt-3 divide-y divide-danger/15 rounded-2xl border border-danger/30 bg-danger/[0.04]">
            {/* Clear saved analyses */}
            <div className="flex flex-wrap items-center justify-between gap-4 p-5">
              <div className="min-w-0 flex-1">
                <p className="flex items-center gap-2 font-medium text-content">
                  <ClockCounterClockwise size={18} weight="duotone" className="text-danger" />
                  {t("settings.clearTitle")}
                </p>
                <p className="mt-1 text-sm text-muted">{t("settings.clearDesc")}</p>

                {clear.kind === "done" && (
                  <p className="mt-2 flex items-center gap-1.5 text-sm text-secondary">
                    <CheckCircle size={16} weight="fill" />
                    {clear.deleted === 0
                      ? t("settings.clearedNone")
                      : clear.deleted === 1
                        ? t("settings.clearedOne")
                        : t("settings.clearedMany", { count: clear.deleted })}
                  </p>
                )}
                {clear.kind === "error" && (
                  <p className="mt-2 flex items-center gap-1.5 text-sm text-danger">
                    <WarningCircle size={16} weight="fill" />
                    {t("settings.clearError")}
                  </p>
                )}
              </div>

              {clear.kind === "confirm" ? (
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    onClick={() => setClear({ kind: "idle" })}
                    className="rounded-xl px-4 py-2 text-sm font-medium text-muted transition-colors hover:bg-content/5 hover:text-content"
                  >
                    {t("settings.clearCancel")}
                  </button>
                  <button
                    onClick={() => void runClear()}
                    className="inline-flex items-center gap-1.5 rounded-xl bg-red-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-700 active:scale-[0.99]"
                  >
                    <Trash size={16} weight="fill" />
                    {t("settings.clearConfirm")}
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setClear({ kind: "confirm" })}
                  disabled={clear.kind === "working"}
                  className="inline-flex shrink-0 items-center gap-1.5 rounded-xl border border-danger/40 px-4 py-2 text-sm font-semibold text-danger transition-colors hover:bg-danger/10 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <Trash size={16} />
                  {clear.kind === "working" ? t("settings.clearing") : t("settings.clearCta")}
                </button>
              )}
            </div>

            {/* Account deletion: not wired (backend holds no service-role key). */}
            <div className="flex flex-wrap items-center justify-between gap-4 p-5">
              <div className="min-w-0 flex-1">
                <p className="flex items-center gap-2 font-medium text-content">
                  <Trash size={18} weight="duotone" className="text-danger" />
                  {t("settings.deleteAccount")}
                </p>
                <p className="mt-1 text-sm text-muted">{t("settings.deleteAccountDesc")}</p>
              </div>
            </div>
          </div>
        </section>
        </main>
      </div>
    </AppLayout>
  );
}
