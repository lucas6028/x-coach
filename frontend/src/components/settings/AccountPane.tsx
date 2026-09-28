import { useEffect, useState, type ReactNode } from "react";
import { CheckCircle, Trash, WarningCircle } from "@phosphor-icons/react";
import { api, type StorageUsage } from "../../api";
import { fmtStorage } from "../../lib/format";
import { useI18n } from "../../lib/i18n";
import { PaneRows, PaneTitle, SettingRow } from "./parts";

type ClearState =
  | { kind: "idle" }
  | { kind: "confirm" }
  | { kind: "working" }
  | { kind: "done"; deleted: number }
  | { kind: "error"; message: string };

type UsageState = { kind: "loading" } | { kind: "error" } | { kind: "ready"; usage: StorageUsage };

// The bar turns red from here: close enough to the quota that an ordinary clip may be refused.
const NEARLY_FULL = 0.9;

// How much of the upload quota the user has used — the same two numbers the analyze route checks
// an upload against, so "full" here means the next upload really is refused. Refetched whenever
// `reloadKey` changes, which the pane bumps after a clear frees space; a refetch keeps the old
// figures on screen rather than flashing back to "loading".
function StorageRow({ reloadKey }: { reloadKey: number }) {
  const { t } = useI18n();
  const [state, setState] = useState<UsageState>({ kind: "loading" });

  useEffect(() => {
    let alive = true;
    api
      .getStorageUsage()
      .then((usage) => alive && setState({ kind: "ready", usage }))
      .catch(() => alive && setState({ kind: "error" }));
    return () => {
      alive = false;
    };
  }, [reloadKey]);

  let hint: ReactNode;
  if (state.kind === "loading") {
    hint = t("settings.storageLoading");
  } else if (state.kind === "error") {
    hint = (
      <p className="flex items-center gap-1.5 text-danger">
        <WarningCircle size={16} weight="fill" />
        {t("settings.storageError")}
      </p>
    );
  } else {
    const { used_bytes: used, quota_bytes: quota } = state.usage;
    // Clamped for display only: an admin can lower the quota below what a user already stores,
    // and the bar must not overflow its track nor "free" go negative.
    const ratio = quota > 0 ? Math.min(used / quota, 1) : 1;
    const usedText = t("settings.storageUsage", { used: fmtStorage(used), quota: fmtStorage(quota) });
    hint = (
      <>
        {usedText} · {t("settings.storageFree", { free: fmtStorage(Math.max(quota - used, 0)) })}
        <div
          className="mt-2 h-1.5 overflow-hidden rounded-full bg-content/[0.06]"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(ratio * 100)}
          aria-valuetext={usedText}
          aria-label={t("settings.storage")}
        >
          <div
            className={`h-full rounded-full transition-[width] ${ratio >= NEARLY_FULL ? "bg-danger" : "bg-primary"}`}
            style={{ width: `${Math.round(ratio * 100)}%` }}
          />
        </div>
        {used >= quota && (
          <p className="mt-2 flex items-center gap-1.5 text-danger">
            <WarningCircle size={16} weight="fill" />
            {t("settings.storageFull")}
          </p>
        )}
      </>
    );
  }

  return <SettingRow label={t("settings.storage")} hint={hint} />;
}

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
  const [usageKey, setUsageKey] = useState(0);

  const runClear = async () => {
    setClear({ kind: "working" });
    try {
      const { deleted } = await api.deleteAnalyses();
      setClear({ kind: "done", deleted });
      setUsageKey((k) => k + 1);
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
        <StorageRow reloadKey={usageKey} />

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
