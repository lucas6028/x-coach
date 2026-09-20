import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  CalendarBlank,
  ChatCircleText,
  ClipboardText,
  LinkBreak,
  UserCircle,
  WarningCircle,
} from "@phosphor-icons/react";
import { api, type ClinicPatientDetail, type Checkin } from "../../api";
import { movementLabel, useI18n } from "../../lib/i18n";
import MovementIcon from "../../components/movements/MovementIcon";
import AssignPlanButton from "../../components/clinic/AssignPlanButton";
import TrendChart from "../../components/clinic/TrendChart";
import ConfirmDialog from "../../components/ConfirmDialog";
import { progressRatio } from "../../lib/plans";

type Status = "loading" | "ready" | "notfound" | "error";

function isNotFound(e: unknown): boolean {
  return e instanceof Error && e.message.startsWith("404");
}

function fmtDateTime(iso: string | null, lang: string, fallback: string): string {
  if (!iso) return fallback;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? fallback : d.toLocaleString(lang, { dateStyle: "medium", timeStyle: "short" });
}

function fmtDate(iso: string, lang: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString(lang);
}

function Tile({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-border-dark bg-surface-dark p-4 shadow-card">
      <p className="text-[11px] tracking-wide text-faint">{label}</p>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

// One linked patient's clinic view: summary tiles, assigned plans (no link into them — that route
// is owner-only), the form-score/pain trend, and the check-in log with red-flag acknowledgement.
export default function ClinicPatient() {
  const { t, lang } = useI18n();
  const navigate = useNavigate();
  const { patientId } = useParams<{ patientId: string }>();

  const [status, setStatus] = useState<Status>("loading");
  const [detail, setDetail] = useState<ClinicPatientDetail | null>(null);

  const [acking, setAcking] = useState<string | null>(null);
  const [ackError, setAckError] = useState("");

  const [confirmingUnlink, setConfirmingUnlink] = useState(false);
  const [unlinking, setUnlinking] = useState(false);

  const load = useCallback(() => {
    if (!patientId) return;
    setStatus("loading");
    api
      .clinicPatient(patientId)
      .then((res) => {
        setDetail(res);
        setStatus("ready");
      })
      .catch((e) => setStatus(isNotFound(e) ? "notfound" : "error"));
  }, [patientId]);

  useEffect(() => load(), [load]);

  const ack = async (checkinId: string) => {
    setAcking(checkinId);
    setAckError("");
    try {
      await api.clinicAckFlag(checkinId);
      load();
    } catch (e) {
      setAckError(e instanceof Error ? e.message : String(e));
    } finally {
      setAcking(null);
    }
  };

  const unlink = async () => {
    if (!detail) return;
    setUnlinking(true);
    try {
      await api.clinicRevokeLink(detail.patient.link_id);
      navigate("/clinic");
    } catch (e) {
      setUnlinking(false);
      setAckError(e instanceof Error ? e.message : String(e));
      setConfirmingUnlink(false);
    }
  };

  if (status === "loading") {
    return <p className="text-sm text-muted">{t("clinic.loadingPatient")}</p>;
  }

  if (status === "notfound") {
    return (
      <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-border-dark bg-surface/60 px-6 py-16 text-center">
        <UserCircle size={40} weight="duotone" className="text-faint" />
        <p className="font-medium text-content">{t("clinic.patientNotFound")}</p>
        <Link
          to="/clinic"
          className="mt-1 inline-flex items-center gap-2 rounded-full border border-border-dark px-4 py-2 text-xs font-medium text-content transition-colors hover:border-primary/40"
        >
          {t("clinic.backToPatients")}
        </Link>
      </div>
    );
  }

  if (status === "error" || !detail) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-2xl border border-border-dark bg-surface px-6 py-12 text-center">
        <WarningCircle size={22} className="text-danger" weight="duotone" />
        <p className="text-sm text-muted">{t("clinic.loadFailed")}</p>
        <button
          type="button"
          onClick={load}
          className="mt-1 rounded-full border border-border-dark px-4 py-1.5 text-xs font-medium text-content transition-colors hover:border-primary/40"
        >
          {t("plans.retry")}
        </button>
      </div>
    );
  }

  const { patient, plans, checkins, trend } = detail;
  const name = patient.display_name ?? patient.email ?? t("clinic.unnamed");

  return (
    <div>
      <Link to="/clinic" className="text-xs font-medium text-muted transition-colors hover:text-primary">
        {t("clinic.backToPatients")}
      </Link>

      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-content">{name}</h1>
          {patient.email && patient.display_name && (
            <p className="mt-1 text-xs text-muted">{patient.email}</p>
          )}
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <AssignPlanButton patientId={patient.patient_id} patientName={name} onAssigned={load} />
          <button
            type="button"
            onClick={() => setConfirmingUnlink(true)}
            className="inline-flex items-center gap-1.5 rounded-full border border-danger/30 px-3 py-1.5 text-xs font-semibold text-danger transition-colors hover:bg-danger/10"
          >
            <LinkBreak size={14} weight="bold" />
            {t("clinic.unlink")}
          </button>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Tile label={t("clinic.colAdherence")}>
          <span className="text-[22px] font-bold leading-none text-content">
            {patient.adherence_7d === null ? "—" : `${Math.round(patient.adherence_7d * 100)}%`}
          </span>
        </Tile>
        <Tile label={t("clinic.colFlags")}>
          <span
            className={`text-[22px] font-bold leading-none ${patient.open_flags > 0 ? "text-danger" : "text-content"}`}
          >
            {patient.open_flags}
          </span>
        </Tile>
        <Tile label={t("clinic.colLastCheckin")}>
          <span className="text-[15px] font-semibold leading-none text-content">
            {fmtDateTime(patient.last_checkin_at, lang, t("clinic.never"))}
          </span>
        </Tile>
      </div>

      {/* Panel 1: assigned plans. No links into them — GET /api/plans/{id} is owner-only, so a
          clinician clicking through would only reach a 404. */}
      <section className="mt-8">
        <div className="flex items-center gap-2">
          <ClipboardText size={16} weight="duotone" className="text-primary" />
          <h2 className="text-sm font-semibold text-content">{t("clinic.plansTitle")}</h2>
        </div>

        {plans.length === 0 ? (
          <p className="mt-3 text-sm text-muted">{t("clinic.noPlans")}</p>
        ) : (
          <ul className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {plans.map((plan) => {
              const shown = plan.movements.slice(0, 4);
              const overflow = plan.movements.length - shown.length;
              return (
                <li key={plan.id} className="rounded-2xl border border-border-dark bg-surface p-4 shadow-card">
                  <h3 className="flex flex-wrap items-center gap-2 font-display text-[14px] font-semibold text-content">
                    {plan.name}
                    {plan.assigned_by && (
                      <span className="inline-flex shrink-0 items-center rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                        {t("plans.assignedByTherapist")}
                      </span>
                    )}
                  </h3>
                  <p className="mt-1 text-xs tabular-nums text-muted">
                    {t("plans.progress", { done: plan.completed_count, total: plan.item_count })}
                  </p>
                  <ul className="mt-2 flex flex-wrap gap-1.5">
                    {shown.map((m) => (
                      <li
                        key={m}
                        className="inline-flex items-center gap-1 rounded-full bg-content/[0.04] px-2 py-1 text-[11px] font-medium text-muted"
                      >
                        <MovementIcon movement={m} size={12} />
                        {movementLabel(t, m)}
                      </li>
                    ))}
                    {overflow > 0 && (
                      <li className="inline-flex items-center rounded-full bg-content/[0.04] px-2 py-1 text-[11px] font-medium text-faint">
                        +{overflow}
                      </li>
                    )}
                  </ul>
                  {plan.item_count > 0 && (
                    <div
                      className="mt-3 h-1.5 overflow-hidden rounded-full bg-content/[0.08]"
                      role="progressbar"
                      aria-valuemin={0}
                      aria-valuemax={plan.item_count}
                      aria-valuenow={plan.completed_count}
                    >
                      <div
                        className="h-full rounded-full bg-primary transition-[width]"
                        style={{
                          width: `${Math.round(progressRatio(plan.completed_count, plan.item_count) * 100)}%`,
                        }}
                      />
                    </div>
                  )}
                  <p className="mt-2 flex items-center gap-1 text-[11px] tabular-nums text-faint">
                    <CalendarBlank size={12} weight="duotone" className="shrink-0" />
                    {plan.started_at
                      ? t("plans.startedOn", { date: fmtDate(plan.started_at, lang) })
                      : t("plans.notStarted")}
                  </p>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {/* Panel 2: the 30-day form-score / pain trend. */}
      <section className="mt-8">
        <div className="flex items-center gap-2">
          <ChatCircleText size={16} weight="duotone" className="text-primary" />
          <h2 className="text-sm font-semibold text-content">{t("clinic.trendTitle")}</h2>
        </div>
        <div className="mt-3 rounded-2xl border border-border-dark bg-surface p-4 shadow-card">
          <TrendChart trend={trend} />
        </div>
      </section>

      {/* Panel 3: the check-in log. */}
      <section className="mt-8">
        <h2 className="text-sm font-semibold text-content">{t("clinic.checkinsTitle")}</h2>
        {ackError && <p className="mt-2 text-xs text-danger">{ackError}</p>}
        {checkins.length === 0 ? (
          <p className="mt-3 text-sm text-muted">{t("clinic.noCheckins")}</p>
        ) : (
          <ul className="mt-3 flex flex-col gap-2">
            {checkins.map((c) => (
              <CheckinRow key={c.id} checkin={c} acking={acking === c.id} onAck={() => void ack(c.id)} />
            ))}
          </ul>
        )}
      </section>

      <ConfirmDialog
        open={confirmingUnlink}
        title={t("clinic.unlinkConfirmTitle")}
        description={t("clinic.unlinkConfirmBody")}
        detail={name}
        confirmLabel={t("clinic.unlink")}
        cancelLabel={t("plans.cancel")}
        busy={unlinking}
        onConfirm={() => void unlink()}
        onCancel={() => setConfirmingUnlink(false)}
      />
    </div>
  );
}

function CheckinRow({
  checkin,
  acking,
  onAck,
}: {
  checkin: Checkin;
  acking: boolean;
  onAck: () => void;
}) {
  const { t, lang } = useI18n();
  const needsAck = checkin.flagged && !checkin.acknowledged_at;
  const wasAcked = checkin.flagged && !!checkin.acknowledged_at;

  return (
    <li
      className={`rounded-xl border px-4 py-3 ${
        checkin.flagged ? "border-danger/30 bg-danger/[0.04]" : "border-border-dark bg-surface"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium text-muted">{fmtDateTime(checkin.created_at, lang, "")}</p>
        {needsAck && (
          <button
            type="button"
            onClick={onAck}
            disabled={acking}
            className="inline-flex items-center gap-1.5 rounded-full bg-danger px-3 py-1 text-[11px] font-semibold text-white transition-colors hover:bg-danger/90 disabled:opacity-60"
          >
            {t("clinic.acknowledge")}
          </button>
        )}
        {wasAcked && (
          <span className="text-[11px] font-medium text-muted">
            {t("clinic.acknowledgedOn", { date: fmtDateTime(checkin.acknowledged_at, lang, "") })}
          </span>
        )}
      </div>

      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-content">
        <span>
          {t("checkin.painLabel")}: <span className="font-semibold tabular-nums">{checkin.pain_nrs}</span>
        </span>
        <span>
          {t("checkin.rpeLabel")}:{" "}
          <span className="font-semibold tabular-nums">{checkin.rpe ?? "—"}</span>
        </span>
        <span>
          {t("studio.formScore")}:{" "}
          <span className="font-semibold tabular-nums">{checkin.form_score ?? "—"}</span>
        </span>
      </div>

      {checkin.note && <p className="mt-1.5 text-xs leading-relaxed text-muted">{checkin.note}</p>}

      {checkin.flag_reasons.length > 0 && (
        <ul className="mt-1.5 flex flex-wrap gap-1.5">
          {checkin.flag_reasons.map((reason) => (
            <li
              key={reason}
              className="inline-flex items-center rounded-full bg-danger/10 px-2 py-0.5 text-[10.5px] font-medium text-danger"
            >
              {t(`checkin.reason.${reason}`)}
            </li>
          ))}
        </ul>
      )}

      {checkin.analysis_id && (
        <Link
          to={`/app?analysis=${encodeURIComponent(checkin.analysis_id)}`}
          className="mt-1.5 inline-block text-[11px] font-medium text-primary hover:underline"
        >
          {t("clinic.viewAnalysis")}
        </Link>
      )}
    </li>
  );
}
