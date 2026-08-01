import { useState } from "react";
import { CheckCircle, Trash, WarningCircle } from "@phosphor-icons/react";
import { api } from "../../api";
import { useI18n } from "../../lib/i18n";
import { PaneRows, PaneTitle, SettingRow } from "./parts";

type ClearState =
  | { kind: "idle" }
  | { kind: "confirm" }
  | { kind: "working" }
  | { kind: "done"; deleted: number }
  | { kind: "error"; message: string };

// Destructive account actions, in the same row idiom as every other pane — the reference layout
// has no boxed "danger zone", so the destructive weight sits on the control instead: a red button,
// and a two-step confirmation before anything is deleted.
//
// That confirmation stays INLINE (a button swap) rather than opening ConfirmDialog: both overlays
// are `fixed inset-0 z-50` with no portal (see components/ConfirmDialog), so stacking one inside
// the settings popup would be fragile.
export default function AccountPane() {
  const { t } = useI18n();
  const [clear, setClear] = useState<ClearState>({ kind: "idle" });

  const runClear = async () => {
    setClear({ kind: "working" });
    try {
      const { deleted } = await api.deleteAnalyses();
      setClear({ kind: "done", deleted });
    } catch (e) {
      setClear({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const clearHint = (
    <>
      {t("settings.clearDesc")}
      {clear.kind === "done" && (
        <p className="mt-2 flex items-center gap-1.5 text-secondary">
          <CheckCircle size={16} weight="fill" />
          {clear.deleted === 0
            ? t("settings.clearedNone")
            : clear.deleted === 1
              ? t("settings.clearedOne")
              : t("settings.clearedMany", { count: clear.deleted })}
        </p>
      )}
      {clear.kind === "error" && (
        <p className="mt-2 flex items-center gap-1.5 text-danger">
          <WarningCircle size={16} weight="fill" />
          {t("settings.clearError")}
        </p>
      )}
    </>
  );

  return (
    <section>
      <PaneTitle>{t("settings.account")}</PaneTitle>
      <PaneRows>
        <SettingRow label={t("settings.clearTitle")} hint={clearHint}>
          {clear.kind === "confirm" ? (
            <div className="flex items-center gap-2">
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
              className="inline-flex items-center gap-1.5 rounded-xl border border-danger/40 px-4 py-2 text-sm font-semibold text-danger transition-colors hover:bg-danger/10 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
            >
              <Trash size={16} />
              {clear.kind === "working" ? t("settings.clearing") : t("settings.clearCta")}
            </button>
          )}
        </SettingRow>

        {/* Account deletion: not wired (backend holds no service-role key), so this row carries
            the explanation and no control. */}
        <SettingRow label={t("settings.deleteAccount")} hint={t("settings.deleteAccountDesc")} />
      </PaneRows>
    </section>
  );
}
