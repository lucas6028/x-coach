import type { Analysis } from "../../api";
import { formScore, type FormScoreBand } from "../../lib/formScore";
import { useI18n } from "../../lib/i18n";

const R = 26;
const CIRCUMFERENCE = 2 * Math.PI * R;

const BAND_COLOR: Record<FormScoreBand, string> = {
  excellent: "#22c55e",
  good: "#22c55e",
  fair: "#e0a33a",
  poor: "#ff6b6b",
};
const BAND_TEXT: Record<FormScoreBand, string> = {
  excellent: "text-[#1a9e5a]",
  good: "text-[#1a9e5a]",
  fair: "text-[#b8922e]",
  poor: "text-[#e05252]",
};

// The reference's floating "Form Score" ring. The number is DERIVED on the client from this
// clip's detections (see lib/formScore.ts) — the backend has no such field — so the card names
// where it comes from rather than presenting it as a measurement.
export default function FormScoreCard({ analysis }: { analysis: Analysis }) {
  const { t } = useI18n();
  const score = formScore(analysis);

  return (
    <div
      title={t("studio.formScoreNote")}
      className="glass-over-video rounded-[14px] p-3"
    >
      <div className="mb-2 text-[11px] font-bold text-[#1e2142]">{t("studio.formScore")}</div>
      <div className="flex items-center gap-3">
        <div className="relative h-[66px] w-[66px] shrink-0">
          <svg width="66" height="66" viewBox="0 0 66 66" className="-rotate-90">
            <circle cx="33" cy="33" r={R} fill="none" stroke="#e8e8f0" strokeWidth="6" />
            {score && (
              <circle
                cx="33"
                cy="33"
                r={R}
                fill="none"
                stroke={BAND_COLOR[score.band]}
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={`${(score.value / 100) * CIRCUMFERENCE} ${CIRCUMFERENCE}`}
                style={{ transition: "stroke-dasharray 0.5s" }}
              />
            )}
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-[15px] font-extrabold text-[#1e2142]">
              {score ? `${score.value}%` : "—"}
            </span>
          </div>
        </div>
        <div className="min-w-0">
          {score ? (
            <>
              <div className={`text-[13px] font-bold leading-none ${BAND_TEXT[score.band]}`}>
                {t(`studio.band.${score.band}`)}
              </div>
              <div className="mt-1 text-[11px] font-medium leading-tight text-[#63709f]">
                {t("studio.formScoreFrom")}
              </div>
            </>
          ) : (
            <div className="text-[11px] font-medium leading-tight text-[#63709f]">
              {t("studio.formScoreUnknown")}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
