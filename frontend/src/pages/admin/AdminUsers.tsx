import { useEffect, useState } from "react";
import { CheckCircle, Users, WarningCircle, XCircle } from "@phosphor-icons/react";
import { useOutletContext } from "react-router-dom";
import { api, type AdminUserRow } from "../../api";
import { useI18n } from "../../lib/i18n";

type Status = "loading" | "ready" | "error";

function fmtDate(iso: string | null, never: string): string {
  if (!iso) return never;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? never : d.toLocaleDateString();
}

// Read-only users table with a per-row admin toggle. The current user's own row is non-toggleable so
// an admin cannot lock themselves out (the backend also rejects self-demotion with a 400). The
// signed-in user id is threaded through the AdminLayout <Outlet/> context.
export default function AdminUsers() {
  const { t } = useI18n();
  const { currentUserId } = useOutletContext<{ currentUserId?: string }>();

  const [rows, setRows] = useState<AdminUserRow[] | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [rowError, setRowError] = useState<{ id: string; message: string } | null>(null);

  const load = () => {
    setStatus("loading");
    api
      .listAdminUsers()
      .then((res) => {
        setRows(res.users);
        setStatus("ready");
      })
      .catch(() => setStatus("error"));
  };

  useEffect(load, []);

  const onToggle = async (row: AdminUserRow) => {
    setPendingId(row.id);
    setRowError(null);
    try {
      await api.setUserRole(row.id, !row.is_admin);
      const res = await api.listAdminUsers();
      setRows(res.users);
    } catch (e) {
      setRowError({ id: row.id, message: e instanceof Error ? e.message : String(e) });
    } finally {
      setPendingId(null);
    }
  };

  return (
    <div>
      <div className="flex items-center gap-2">
        <Users size={18} weight="duotone" className="text-primary" />
        <h2 className="text-sm font-semibold text-content">{t("admin.users.title")}</h2>
      </div>
      <p className="mt-1 text-xs text-muted">{t("admin.users.desc")}</p>

      {status === "loading" && <p className="mt-4 text-sm text-muted">{t("admin.users.loading")}</p>}
      {status === "error" && (
        <div className="mt-4 flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
          <WarningCircle size={18} className="shrink-0" />
          <p className="font-medium">{t("admin.users.loadError")}</p>
        </div>
      )}

      {status === "ready" && rows && rows.length === 0 && (
        <p className="mt-4 text-sm text-muted">{t("admin.users.empty")}</p>
      )}

      {status === "ready" && rows && rows.length > 0 && (
        <div className="mt-4 overflow-x-auto rounded-2xl border border-border-dark bg-surface-dark">
          <table className="w-full min-w-[40rem] text-left text-sm">
            <thead>
              <tr className="border-b border-border-dark text-xs font-medium text-faint">
                <th className="px-4 py-3">{t("admin.users.email")}</th>
                <th className="px-4 py-3">{t("admin.users.created")}</th>
                <th className="px-4 py-3">{t("admin.users.lastSignIn")}</th>
                <th className="px-4 py-3 text-right tabular-nums">{t("admin.users.analyses")}</th>
                <th className="px-4 py-3 text-right tabular-nums">{t("admin.users.conversations")}</th>
                <th className="px-4 py-3 text-right">{t("admin.users.role")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isSelf = row.id === currentUserId;
                const pending = pendingId === row.id;
                return (
                  <tr key={row.id} className="border-b border-border-dark/60 last:border-0">
                    <td className="px-4 py-3 text-content">
                      <span className="break-all">{row.email ?? row.id}</span>
                      {isSelf && (
                        <span className="ml-2 rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                          {t("admin.users.you")}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted">{fmtDate(row.created_at, t("admin.users.never"))}</td>
                    <td className="px-4 py-3 text-muted">{fmtDate(row.last_sign_in_at, t("admin.users.never"))}</td>
                    <td className="px-4 py-3 text-right tabular-nums text-content">{row.analyses_count}</td>
                    <td className="px-4 py-3 text-right tabular-nums text-content">{row.conversations_count}</td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => void onToggle(row)}
                        disabled={isSelf || pending}
                        aria-label={row.is_admin ? t("admin.users.revokeAdmin") : t("admin.users.makeAdmin")}
                        className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                          row.is_admin
                            ? "border-secondary/40 bg-secondary/10 text-secondary hover:bg-secondary/20"
                            : "border-border-dark bg-content/[0.02] text-faint hover:bg-content/[0.05]"
                        }`}
                      >
                        {row.is_admin ? <CheckCircle size={14} weight="fill" /> : <XCircle size={14} />}
                        {row.is_admin ? t("admin.users.revokeAdmin") : t("admin.users.makeAdmin")}
                      </button>
                      {rowError?.id === row.id && (
                        <p className="mt-1 text-[11px] text-danger">{t("admin.users.updateError")}</p>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
