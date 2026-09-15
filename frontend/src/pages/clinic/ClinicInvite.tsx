import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Check, Copy, Sparkle, Trash, WarningCircle } from "@phosphor-icons/react";
import { api, type ClinicInvite as ClinicInviteRow } from "../../api";
import { useI18n } from "../../lib/i18n";

type ListStatus = "loading" | "ready" | "error";

function fmtExpiry(iso: string, lang: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(lang, { dateStyle: "medium", timeStyle: "short" });
}

// 「邀請個案」: generate a fresh invite code and manage the still-pending ones. The code itself is
// the only credential — there is no email/SMS delivery here, so the therapist reads it out or
// copies it to send however they already talk to the patient.
export default function ClinicInvite() {
  const { t, lang } = useI18n();

  const [invites, setInvites] = useState<ClinicInviteRow[]>([]);
  const [listStatus, setListStatus] = useState<ListStatus>("loading");

  const [justCreated, setJustCreated] = useState<ClinicInviteRow | null>(null);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");
  const [copied, setCopied] = useState(false);

  const [revokingId, setRevokingId] = useState<string | null>(null);
  const [revokeError, setRevokeError] = useState("");

  const loadInvites = useCallback(() => {
    setListStatus("loading");
    api
      .clinicInvites()
      .then((res) => {
        setInvites(res);
        setListStatus("ready");
      })
      .catch(() => setListStatus("error"));
  }, []);

  useEffect(() => loadInvites(), [loadInvites]);

  const create = async () => {
    setCreating(true);
    setCreateError("");
    setCopied(false);
    try {
      const invite = await api.clinicCreateInvite();
      setJustCreated(invite);
      loadInvites();
    } catch (e) {
      setCreateError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreating(false);
    }
  };

  const copyCode = async () => {
    if (!justCreated) return;
    try {
      await navigator.clipboard.writeText(justCreated.invite_code);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const revoke = async (id: string) => {
    setRevokingId(id);
    setRevokeError("");
    try {
      await api.clinicRevokeLink(id);
      if (justCreated?.id === id) setJustCreated(null);
      loadInvites();
    } catch (e) {
      setRevokeError(e instanceof Error ? e.message : String(e));
    } finally {
      setRevokingId(null);
    }
  };

  return (
    <div>
      <Link to="/clinic" className="text-xs font-medium text-muted transition-colors hover:text-primary">
        {t("clinic.backToPatients")}
      </Link>

      <h1 className="mt-3 font-display text-2xl font-bold text-content">{t("clinic.inviteTitle")}</h1>
      <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">
        {t("clinic.inviteInstructions", { settings: t("nav.settings"), therapist: t("settings.therapist") })}
      </p>

      <div className="mt-6 rounded-2xl border border-border-dark bg-surface p-5 shadow-card">
        <button
          type="button"
          onClick={() => void create()}
          disabled={creating}
          className="inline-flex items-center gap-2 rounded-full bg-primary px-5 py-2.5 text-[13px] font-semibold text-primary-content shadow-accent transition-all hover:bg-primary/90 active:scale-[0.98] disabled:opacity-60"
        >
          <Sparkle size={15} weight="bold" />
          {creating ? t("clinic.generating") : t("clinic.generateCode")}
        </button>

        {createError && (
          <p className="mt-3 flex items-start gap-1.5 rounded-xl border border-danger/30 bg-danger/[0.05] px-3 py-2 text-xs text-danger">
            <WarningCircle size={14} weight="duotone" className="mt-px shrink-0" />
            {createError}
          </p>
        )}

        {justCreated && (
          <div className="mt-5 rounded-xl border border-primary/30 bg-primary/[0.04] p-4">
            <p className="text-xs font-medium text-muted">{t("clinic.newCodeLabel")}</p>
            <div className="mt-1.5 flex flex-wrap items-center gap-3">
              <span className="font-display text-3xl font-bold tracking-[0.2em] text-content">
                {justCreated.invite_code}
              </span>
              <button
                type="button"
                onClick={() => void copyCode()}
                className="inline-flex items-center gap-1.5 rounded-full border border-border-dark bg-surface px-3 py-1.5 text-xs font-semibold text-content transition-colors hover:border-primary/40 hover:text-primary active:scale-[0.98]"
              >
                {copied ? <Check size={14} weight="bold" /> : <Copy size={14} />}
                {copied ? t("clinic.copied") : t("clinic.copyCode")}
              </button>
            </div>
            <p className="mt-2 text-xs text-faint">
              {t("clinic.expiresOn", { date: fmtExpiry(justCreated.expires_at, lang) })}
            </p>
          </div>
        )}
      </div>

      <section className="mt-8">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">
          {t("clinic.pendingInvitesTitle")}
        </h2>

        {listStatus === "loading" && <p className="mt-3 text-sm text-muted">{t("clinic.loadingInvites")}</p>}
        {listStatus === "error" && (
          <p className="mt-3 text-sm text-danger">{t("clinic.loadFailed")}</p>
        )}
        {listStatus === "ready" && invites.length === 0 && (
          <p className="mt-3 text-sm text-muted">{t("clinic.noPendingInvites")}</p>
        )}

        {listStatus === "ready" && invites.length > 0 && (
          <ul className="mt-3 flex flex-col gap-2">
            {invites.map((inv) => (
              <li
                key={inv.id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border-dark bg-surface px-4 py-3"
              >
                <div>
                  <p className="font-mono text-sm font-semibold tracking-widest text-content">
                    {inv.invite_code}
                  </p>
                  <p className="mt-0.5 text-[11px] text-faint">
                    {t("clinic.expiresOn", { date: fmtExpiry(inv.expires_at, lang) })}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void revoke(inv.id)}
                  disabled={revokingId === inv.id}
                  className="inline-flex items-center gap-1.5 rounded-full border border-danger/30 px-3 py-1.5 text-xs font-semibold text-danger transition-colors hover:bg-danger/10 disabled:opacity-60"
                >
                  <Trash size={13} weight="bold" />
                  {t("clinic.revoke")}
                </button>
              </li>
            ))}
          </ul>
        )}
        {revokeError && <p className="mt-2 text-xs text-danger">{revokeError}</p>}
      </section>
    </div>
  );
}
