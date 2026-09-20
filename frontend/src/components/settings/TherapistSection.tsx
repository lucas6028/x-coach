import { useCallback, useEffect, useState, type FormEvent } from "react";
import ConfirmDialog from "../ConfirmDialog";
import { api, ApiError, type CareClinician } from "../../api";
import { useI18n } from "../../lib/i18n";
import { PaneRows, PaneTitle, SettingRow } from "./parts";

type LinkState =
  | { kind: "idle" }
  | { kind: "linking" }
  | { kind: "linked" }
  | { kind: "error"; message: string };

// 「我的治療師」(WP2 care loop): accept a therapist's invite code, list who the caller is linked
// to, and unlink. What a linked therapist can and cannot see is fixed by the DB policies written
// in the WP0/WP1 migrations — the privacy line below states exactly that boundary and nothing more.
export default function TherapistSection() {
  const { t, lang } = useI18n();

  const [clinicians, setClinicians] = useState<CareClinician[] | null>(null);
  const [listError, setListError] = useState(false);
  const refresh = useCallback(() => {
    api
      .careClinicians()
      .then((list) => {
        setClinicians(list);
        setListError(false);
      })
      .catch(() => setListError(true));
  }, []);
  useEffect(() => {
    refresh();
  }, [refresh]);

  const [code, setCode] = useState("");
  const [linkState, setLinkState] = useState<LinkState>({ kind: "idle" });

  const submitCode = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = code.trim();
    if (!trimmed || linkState.kind === "linking") return;
    setLinkState({ kind: "linking" });
    try {
      await api.careAccept(trimmed);
      setCode("");
      setLinkState({ kind: "linked" });
      refresh();
    } catch (err) {
      const message =
        err instanceof ApiError && err.status === 404
          ? t("settings.therapist.invalidCode")
          : err instanceof ApiError && err.status === 400
            ? t("settings.therapist.ownCode")
            : t("settings.therapist.genericError");
      setLinkState({ kind: "error", message });
    }
  };

  const [confirmingUnlink, setConfirmingUnlink] = useState<CareClinician | null>(null);
  const [unlinkBusy, setUnlinkBusy] = useState(false);
  const [unlinkError, setUnlinkError] = useState(false);

  const doUnlink = async () => {
    if (!confirmingUnlink) return;
    setUnlinkBusy(true);
    setUnlinkError(false);
    try {
      await api.careUnlink(confirmingUnlink.link_id);
      setConfirmingUnlink(null);
      refresh();
    } catch {
      setUnlinkError(true);
    } finally {
      setUnlinkBusy(false);
    }
  };

  const codeHint =
    linkState.kind === "error" ? (
      <span className="text-danger">{linkState.message}</span>
    ) : linkState.kind === "linked" ? (
      <span className="text-secondary">{t("settings.therapist.linked")}</span>
    ) : undefined;

  return (
    <section>
      <PaneTitle>{t("settings.therapist")}</PaneTitle>
      <p className="mt-2 text-sm leading-relaxed text-muted">{t("settings.therapist.privacy")}</p>

      <PaneRows>
        <SettingRow label={t("settings.therapist.codeLabel")} hint={codeHint}>
          <form onSubmit={(e) => void submitCode(e)} className="flex items-center gap-2">
            <input
              type="text"
              value={code}
              onChange={(e) => {
                setCode(e.target.value);
                if (linkState.kind !== "idle") setLinkState({ kind: "idle" });
              }}
              placeholder={t("settings.therapist.codePlaceholder")}
              aria-label={t("settings.therapist.codeLabel")}
              className="h-9 w-44 min-w-0 rounded-lg border border-border-dark bg-background px-2 text-[13px] text-content outline-none transition-colors focus:border-primary/50"
            />
            <button
              type="submit"
              disabled={linkState.kind === "linking" || !code.trim()}
              className="shrink-0 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {linkState.kind === "linking"
                ? t("settings.therapist.linking")
                : t("settings.therapist.link")}
            </button>
          </form>
        </SettingRow>
      </PaneRows>

      <div className="mt-9">
        <h4 className="text-[15px] font-semibold text-content">
          {t("settings.therapist.listTitle")}
        </h4>
        {listError ? (
          <p className="mt-2 text-sm text-danger">{t("settings.therapist.genericError")}</p>
        ) : clinicians !== null && clinicians.length === 0 ? (
          <p className="mt-2 text-sm text-muted">{t("settings.therapist.empty")}</p>
        ) : (
          <ul className="mt-3 space-y-2">
            {(clinicians ?? []).map((c) => (
              <li
                key={c.link_id}
                className="flex items-center justify-between gap-3 rounded-xl border border-border-dark bg-surface px-3 py-2.5"
              >
                <div className="min-w-0">
                  <p className="truncate text-[13px] font-medium text-content">
                    {c.display_name || c.email || t("settings.therapist.fallbackName")}
                  </p>
                  <p className="text-[11px] text-faint">
                    {t("settings.therapist.linkedOn", {
                      date: new Date(c.accepted_at).toLocaleDateString(lang),
                    })}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setConfirmingUnlink(c)}
                  className="shrink-0 rounded-xl border border-danger/40 px-3 py-1.5 text-xs font-semibold text-danger transition-colors hover:bg-danger/10"
                >
                  {t("settings.therapist.unlink")}
                </button>
              </li>
            ))}
          </ul>
        )}
        {unlinkError && (
          <p className="mt-2 text-xs text-danger">{t("settings.therapist.unlinkError")}</p>
        )}
      </div>

      <ConfirmDialog
        open={confirmingUnlink !== null}
        title={t("settings.therapist.unlinkTitle")}
        description={t("settings.therapist.unlinkBody")}
        detail={
          confirmingUnlink
            ? confirmingUnlink.display_name || confirmingUnlink.email || undefined
            : undefined
        }
        confirmLabel={t("settings.therapist.unlink")}
        cancelLabel={t("plans.cancel")}
        busy={unlinkBusy}
        onConfirm={() => void doUnlink()}
        onCancel={() => setConfirmingUnlink(null)}
      />
    </section>
  );
}
