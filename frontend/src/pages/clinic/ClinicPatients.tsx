import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ChatCircleDots, UserPlus, WarningCircle } from "@phosphor-icons/react";
import { api, type ClinicPatientSummary } from "../../api";
import { useI18n } from "../../lib/i18n";
import AssignPlanButton from "../../components/clinic/AssignPlanButton";

type Status = "loading" | "ready" | "error";

/** display_name, else email, else a localized placeholder — the same fallback chain every row uses. */
function patientName(p: ClinicPatientSummary, unnamed: string): string {
  return p.display_name ?? p.email ?? unnamed;
}

/** Flagged-or-inactive patients sort first; `Array.prototype.sort` is stable, so ties keep the
 *  server's own (newest-first-ish) order. */
function needsAttention(p: ClinicPatientSummary): boolean {
  return p.open_flags > 0 || p.inactive_7d;
}

function sortPatients(patients: ClinicPatientSummary[]): ClinicPatientSummary[] {
  return [...patients].sort((a, b) => Number(needsAttention(b)) - Number(needsAttention(a)));
}

function fmtCheckin(iso: string | null, lang: string, never: string): string {
  if (!iso) return never;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? never : d.toLocaleString(lang, { dateStyle: "short", timeStyle: "short" });
}

// "個案管理": the therapist's landing page — every linked patient, sorted so the ones needing
// attention (an open red flag, or seven days of silence) lead the list.
export default function ClinicPatients() {
  const { t, lang } = useI18n();
  const navigate = useNavigate();

  const [patients, setPatients] = useState<ClinicPatientSummary[]>([]);
  const [status, setStatus] = useState<Status>("loading");

  const load = useCallback(() => {
    setStatus("loading");
    api
      .clinicPatients()
      .then((res) => {
        setPatients(sortPatients(res));
        setStatus("ready");
      })
      .catch(() => setStatus("error"));
  }, []);

  useEffect(() => load(), [load]);

  const unnamed = t("clinic.unnamed");

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-content">{t("clinic.patientsTitle")}</h1>
          <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">
            {t("clinic.patientsSubtitle")}
          </p>
        </div>
        <Link
          to="/clinic/invite"
          className="inline-flex shrink-0 items-center gap-2 rounded-full bg-primary px-5 py-2.5 text-[13px] font-semibold text-primary-content shadow-accent transition-all hover:bg-primary/90 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
        >
          <UserPlus size={16} weight="bold" />
          {t("clinic.invitePatient")}
        </Link>
      </div>

      {status === "loading" && (
        <div className="mt-8 grid grid-cols-1 gap-3 lg:grid-cols-2">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-20 animate-pulse rounded-2xl border border-border-dark bg-surface" />
          ))}
        </div>
      )}

      {status === "error" && (
        <div className="mt-8 flex flex-col items-center gap-3 rounded-2xl border border-border-dark bg-surface px-6 py-12 text-center">
          <WarningCircle size={22} className="text-danger" weight="duotone" />
          <p className="text-sm text-muted">{t("clinic.loadFailed")}</p>
          <button
            type="button"
            onClick={load}
            className="mt-1 rounded-full border border-border-dark px-4 py-1.5 text-xs font-medium text-content transition-colors hover:border-primary/40 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
          >
            {t("plans.retry")}
          </button>
        </div>
      )}

      {status === "ready" && patients.length === 0 && (
        <div className="mt-8 flex flex-col items-center gap-3 rounded-2xl border border-dashed border-border-dark bg-surface/60 px-6 py-14 text-center">
          <ChatCircleDots size={40} weight="duotone" className="text-primary" />
          <h2 className="font-display text-lg font-bold text-content">{t("clinic.emptyTitle")}</h2>
          <p className="max-w-sm text-sm leading-relaxed text-muted">{t("clinic.emptyBody")}</p>
          <Link
            to="/clinic/invite"
            className="mt-1 inline-flex items-center gap-2 rounded-full bg-primary px-5 py-2.5 text-[13px] font-semibold text-primary-content shadow-accent transition-all hover:bg-primary/90 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
          >
            {t("clinic.invitePatient")}
          </Link>
        </div>
      )}

      {status === "ready" && patients.length > 0 && (
        <>
          {/* Wide screens: a table, its own horizontal-scroll container so a narrow-but-not-phone
              width never pushes the page itself sideways. */}
          <div className="mt-6 hidden overflow-x-auto rounded-2xl border border-border-dark bg-surface-dark lg:block">
            <table className="w-full min-w-[48rem] text-left text-sm">
              <thead>
                <tr className="border-b border-border-dark text-xs font-medium text-faint">
                  <th className="px-4 py-3">{t("clinic.colPatient")}</th>
                  <th className="px-4 py-3">{t("clinic.colLastCheckin")}</th>
                  <th className="px-4 py-3 text-right tabular-nums">{t("clinic.colAdherence")}</th>
                  <th className="px-4 py-3 text-right">{t("clinic.colFlags")}</th>
                  <th className="px-4 py-3 text-right">{t("clinic.colAssign")}</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p) => (
                  <tr
                    key={p.link_id}
                    onClick={() => navigate(`/clinic/patients/${encodeURIComponent(p.patient_id)}`)}
                    className="cursor-pointer border-b border-border-dark/60 transition-colors last:border-0 hover:bg-content/[0.03]"
                  >
                    <td className="px-4 py-3">
                      <p className="font-medium text-content">{patientName(p, unnamed)}</p>
                      {p.inactive_7d && (
                        <span className="mt-1 inline-flex items-center rounded-full bg-danger/10 px-2 py-0.5 text-[10.5px] font-medium text-danger">
                          {t("clinic.inactiveBadge")}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted">
                      {fmtCheckin(p.last_checkin_at, lang, t("clinic.never"))}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-content">
                      {p.adherence_7d === null ? "—" : `${Math.round(p.adherence_7d * 100)}%`}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {p.open_flags > 0 && (
                        <span className="inline-flex items-center rounded-full bg-danger/10 px-2 py-0.5 text-[11px] font-semibold text-danger">
                          {t("clinic.flagCount", { n: p.open_flags })}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right" onClick={(e) => e.stopPropagation()}>
                      <AssignPlanButton
                        patientId={p.patient_id}
                        patientName={patientName(p, unnamed)}
                        onAssigned={load}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Phone width: cards instead of a squeezed table. */}
          <ul className="mt-6 flex flex-col gap-3 lg:hidden">
            {patients.map((p) => (
              <li
                key={p.link_id}
                className="rounded-2xl border border-border-dark bg-surface p-4 shadow-card"
              >
                <button
                  type="button"
                  onClick={() => navigate(`/clinic/patients/${encodeURIComponent(p.patient_id)}`)}
                  className="flex w-full flex-col items-start gap-2 text-left"
                >
                  <div className="flex w-full flex-wrap items-center gap-2">
                    <p className="font-display text-[15px] font-semibold text-content">
                      {patientName(p, unnamed)}
                    </p>
                    {p.open_flags > 0 && (
                      <span className="inline-flex items-center rounded-full bg-danger/10 px-2 py-0.5 text-[10.5px] font-semibold text-danger">
                        {t("clinic.flagCount", { n: p.open_flags })}
                      </span>
                    )}
                    {p.inactive_7d && (
                      <span className="inline-flex items-center rounded-full bg-danger/10 px-2 py-0.5 text-[10.5px] font-medium text-danger">
                        {t("clinic.inactiveBadge")}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-muted">
                    {t("clinic.colLastCheckin")}: {fmtCheckin(p.last_checkin_at, lang, t("clinic.never"))}
                  </p>
                  <p className="text-xs tabular-nums text-muted">
                    {t("clinic.colAdherence")}:{" "}
                    {p.adherence_7d === null ? "—" : `${Math.round(p.adherence_7d * 100)}%`}
                  </p>
                </button>
                <div className="mt-3">
                  <AssignPlanButton
                    patientId={p.patient_id}
                    patientName={patientName(p, unnamed)}
                    onAssigned={load}
                  />
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      <p className="mt-10 text-[11px] leading-relaxed text-faint">{t("clinic.disclaimer")}</p>
    </div>
  );
}
