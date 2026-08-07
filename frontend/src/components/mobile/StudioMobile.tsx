import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  BookOpen,
  CalendarBlank,
  CaretDown,
  ChartBar,
  VideoCamera,
  WarningCircle,
} from "@phosphor-icons/react";
import type { Analysis } from "../../api";
import { formScore } from "../../lib/formScore";
import { fmtTime } from "../../lib/format";
import { faultLabel, movementLabel, severityText, useI18n, viewLabel } from "../../lib/i18n";
import { keyEvidence, retrievalByFault, summaryCategory } from "../../lib/retrieval";
import { useVideoSrc } from "../../lib/useVideoSrc";
import { wasMeasured } from "../../lib/quality";
import CoachTray from "../CoachTray";
import PreviousSessionsCard from "../studio/PreviousSessionsCard";
import { LumenAvatar } from "../LumenLoader";
import MobileVideoCard from "./MobileVideoCard";

interface Props {
  analysis: Analysis;
  videoRef: React.RefObject<HTMLVideoElement>;
  currentTime: number;
  onTimeUpdate: (t: number) => void;
  onActiveFault: (faultId: string | null) => void;
  /** Which fault the playhead is inside — forwarded to the coach so its cards mark it. */
  activeFaultId: string | null;
  onSeek: (t: number) => void;
  /** Start a fresh session — the mock's full-width primary button. */
  onNewSession: () => void;
}

// One of the mock's disclosure rows: icon, title, muted subtitle, chevron. Expands in place.
function Row({
  icon,
  title,
  subtitle,
  children,
}: {
  icon: ReactNode;
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-b border-[#f0f0f8] last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-3 px-4 py-3.5 text-left"
      >
        <span className="shrink-0 text-primary">{icon}</span>
        <span className="text-[13.5px] font-bold text-[#1e2142]">{title}</span>
        <span className="min-w-0 flex-1 truncate text-[11.5px] text-[#63709f]">{subtitle}</span>
        <CaretDown
          size={15}
          weight="bold"
          className={`shrink-0 text-[#9aa0b8] transition-transform ${open ? "rotate-180" : "-rotate-90"}`}
        />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  );
}

/**
 * The phone layout from `motion_analysis_mobile.png`: stage, headline stats, the coach's opening
 * line, four disclosure rows, and a primary action.
 *
 * WHAT THE MOCK SHOWS THAT THIS PRODUCT DOES NOT HAVE, and what happened to it:
 *  - "Reps 8/12 · Target 12" — rep counts are real (per-rep detection) but no target exists
 *    anywhere, so the count renders alone.
 *  - "Tempo 2.1s ↑1.8s" — nothing computes tempo. Dropped.
 *  - the "1x" speed pill — no playback-rate control behind it. Dropped.
 *  - "Live Analysis" — this pipeline is offline; the pill carries the real fault verdict instead.
 *  - "Finish & Save Session" — uploads already persist on their own, so there is no session to
 *    finish. The slot carries the action a finished analysis actually wants: start another.
 * Everything else on the screen is backed by the analysis.
 */
export default function StudioMobile({
  analysis,
  videoRef,
  currentTime,
  onTimeUpdate,
  onActiveFault,
  activeFaultId,
  onSeek,
  onNewSession,
}: Props) {
  const { t } = useI18n();
  const [coachOpen, setCoachOpen] = useState(false);
  const videoSrc = useVideoSrc(analysis);
  const score = formScore(analysis);
  const byFault = retrievalByFault(analysis.retrievals);

  const duration = analysis.metadata.total_frames / (analysis.metadata.fps || 30);
  const reps = analysis.detections.find((d) => typeof d.rep_count === "number")?.rep_count;
  const q = analysis.quality;

  const cites = analysis.detections.flatMap((d) => {
    const r = byFault.get(d.fault_id);
    return [...summaryCategory(r, "causes"), ...summaryCategory(r, "corrections")];
  });

  // The coach's opening line, collapsed to one sentence: the corrective cue for the worst fault,
  // or the clean-rep / unmeasured verdict. Grounded, not generated — expanding shows the full
  // coach thread that produced it.
  const worst = [...analysis.detections].sort((a, b) => b.severity - a.severity)[0];
  const headline = worst
    ? summaryCategory(byFault.get(worst.fault_id), "corrections")[0] ??
      faultLabel(t, worst.fault_name)
    : wasMeasured(q)
      ? t("feedback.noFaults", { movement: movementLabel(t, analysis.movement ?? "Squat") })
      : t("feedback.notMeasured");

  return (
    <div className="scrollbar-none flex-1 space-y-3 overflow-y-auto px-3 pb-4">
      {/* Analyze | History — the mock's segmented control. Both halves are real destinations. */}
      <div className="glass-control flex rounded-[16px] p-1">
        <span className="flex flex-1 items-center justify-center gap-2 rounded-[12px] bg-white py-2 text-[13px] font-semibold text-primary shadow-sm">
          <VideoCamera size={17} weight="duotone" />
          {t("mobile.analyze")}
        </span>
        <Link
          to="/history"
          className="flex flex-1 items-center justify-center gap-2 rounded-[12px] py-2 text-[13px] font-medium text-[#59648f]"
        >
          <ChartBar size={17} weight="duotone" />
          {t("nav.history")}
        </Link>
      </div>

      <MobileVideoCard
        analysis={analysis}
        videoRef={videoRef}
        videoSrc={videoSrc}
        onTimeUpdate={onTimeUpdate}
        onActiveFault={onActiveFault}
        onSeek={onSeek}
      />

      {/* Headline stats: the score ring beside the two figures the clip actually carries. */}
      <div className="glass-panel flex items-stretch rounded-[18px] px-2 py-3.5">
        <div className="flex flex-1 items-center gap-3 px-2">
          <div className="relative h-[64px] w-[64px] shrink-0">
            <svg width="64" height="64" viewBox="0 0 64 64" className="-rotate-90">
              <circle cx="32" cy="32" r="27" fill="none" stroke="#ece8ff" strokeWidth="6" />
              {score && (
                <circle
                  cx="32"
                  cy="32"
                  r="27"
                  fill="none"
                  stroke="#7b5cff"
                  strokeWidth="6"
                  strokeLinecap="round"
                  strokeDasharray={`${(score.value / 100) * 2 * Math.PI * 27} 999`}
                />
              )}
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center leading-none">
              <span className="text-[17px] font-extrabold text-[#1e2142]">
                {score ? score.value : "—"}
              </span>
              <span className="text-[9px] font-medium text-[#9aa0b8]">/100</span>
            </div>
          </div>
          <div className="min-w-0">
            <div className="text-[11px] font-semibold text-[#65719f]">{t("studio.formScore")}</div>
            <div className="mt-0.5 text-[12px] font-bold leading-tight text-[#1e2142]">
              {score ? t(`studio.band.${score.band}`) : t("studio.formScoreUnknown")}
            </div>
            <div className="text-[10.5px] leading-tight text-[#63709f]">
              {t("studio.formScoreFrom")}
            </div>
          </div>
        </div>

        {reps !== undefined && (
          <div className="flex w-[84px] flex-col items-center justify-center border-l border-[#ebeaf6]">
            <div className="text-[11px] font-semibold text-[#65719f]">{t("mobile.reps")}</div>
            <div className="text-[19px] font-extrabold text-[#1e2142]">{reps}</div>
          </div>
        )}

        <div className="flex w-[92px] flex-col items-center justify-center border-l border-[#ebeaf6]">
          <div className="text-[11px] font-semibold text-[#65719f]">{t("mobile.duration")}</div>
          <div className="text-[19px] font-extrabold text-[#1e2142]">{fmtTime(duration)}</div>
        </div>
      </div>

      {/* The coach's card. Collapsed it is one grounded line; expanded it is the whole coach —
          the causal ladder per fault, the knowledge graph, and the follow-up conversation. */}
      <div className="glass-panel overflow-hidden rounded-[18px]">
        <button
          type="button"
          onClick={() => setCoachOpen((v) => !v)}
          aria-expanded={coachOpen}
          className="flex w-full items-start gap-3 p-4 text-left"
        >
          <LumenAvatar size={34} className="shrink-0" />
          <span className="min-w-0 flex-1">
            <span className="block text-[13.5px] font-bold text-[#1e2142]">{t("chat.heading")}</span>
            <span className="mt-0.5 block text-[12px] leading-snug text-[#59648f]">{headline}</span>
          </span>
          <span
            className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#f3f0ff] text-primary transition-transform ${
              coachOpen ? "rotate-180" : ""
            }`}
          >
            <CaretDown size={15} weight="bold" />
          </span>
        </button>
        {coachOpen && (
          // Bounded so the coach thread scrolls inside the card instead of stretching the page.
          <div className="flex h-[70vh] border-t border-[#ebeaf6]">
            <CoachTray
              analysis={analysis}
              currentTime={currentTime}
              onSeek={onSeek}
              activeFaultId={activeFaultId}
            />
          </div>
        )}
      </div>

      {/* Four disclosure rows, expanding in place. */}
      <div className="glass-panel overflow-hidden rounded-[18px]">
        <Row
          icon={<WarningCircle size={19} weight="duotone" className="text-[#ff5a5a]" />}
          title={t("mobile.topIssues")}
          subtitle={
            analysis.detections.length === 0
              ? wasMeasured(q)
                ? t("metric.cleanRep")
                : t("metric.notMeasured")
              : t("mobile.issuesN", { n: analysis.detections.length })
          }
        >
          {analysis.detections.length === 0 ? (
            <p className="text-[12px] leading-relaxed text-[#59648f]">
              {wasMeasured(q)
                ? t("feedback.noFaults", {
                    movement: movementLabel(t, analysis.movement ?? "Squat"),
                  })
                : t("feedback.notMeasured")}
            </p>
          ) : (
            // Compact rows, not the full fault cards: the causal ladder lives in the coach card
            // above, and repeating it here would say the same thing twice on one scroll.
            <ul className="space-y-1.5">
              {analysis.detections.map((d, i) => {
                const active = currentTime >= d.start_time && currentTime <= d.end_time;
                return (
                  <li key={i}>
                    <button
                      onClick={() => onSeek(d.start_time)}
                      className={`flex w-full items-center gap-2.5 rounded-xl border px-3 py-2 text-left transition-colors ${
                        active
                          ? "border-primary/50 bg-primary/[0.07]"
                          : "border-[#ffe0e0] bg-[#fff5f5]/85"
                      }`}
                    >
                      <span className="shrink-0 font-mono text-[10px] text-[#9aa0b8]">
                        {fmtTime(d.start_time)}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[12px] font-semibold text-[#1e2142]">
                          {faultLabel(t, d.fault_name)}
                        </span>
                        {keyEvidence(d) && (
                          <span className="block truncate font-mono text-[10px] text-[#63709f]">
                            {keyEvidence(d)}
                          </span>
                        )}
                      </span>
                      <span className="shrink-0 text-[10px] font-semibold text-[#e05252]">
                        {severityText(t, d.severity)}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </Row>

        <Row
          icon={<BookOpen size={19} weight="duotone" />}
          title={t("mobile.research")}
          subtitle={t("mobile.researchSub")}
        >
          {cites.length === 0 ? (
            <p className="text-[12px] leading-relaxed text-[#59648f]">{t("studio.tipsNone")}</p>
          ) : (
            <ul className="space-y-1.5">
              {Array.from(new Set(cites)).map((c) => (
                <li key={c} className="flex gap-2 text-[12px] leading-relaxed text-[#59648f]">
                  <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-primary" />
                  {c}
                </li>
              ))}
            </ul>
          )}
        </Row>

        <Row
          icon={<CalendarBlank size={19} weight="duotone" />}
          title={t("mobile.pastSessions")}
          subtitle={t("mobile.pastSessionsSub")}
        >
          <PreviousSessionsCard currentVideoId={analysis.video_id} />
        </Row>

        <Row
          icon={<ChartBar size={19} weight="duotone" />}
          title={t("mobile.detailedMetrics")}
          subtitle={t("mobile.detailedMetricsSub")}
        >
          <dl className="space-y-2 text-[12px]">
            {[
              [t("metric.cameraView"), viewLabel(t, analysis.view.view_type)],
              [t("metric.validFrames"), `${((q.valid_frame_ratio ?? 0) * 100).toFixed(0)}%`],
              [
                t("metric.lowerBodyVis"),
                `${((q.lower_body_visibility_mean ?? 0) * 100).toFixed(0)}%`,
              ],
              ...analysis.detections
                .map((d) => keyEvidence(d))
                .filter((e): e is string => !!e)
                .map((e) => [t("mobile.evidence"), e] as [string, string]),
            ].map(([k, v], i) => (
              <div key={i} className="flex items-baseline justify-between gap-3">
                <dt className="text-[#65719f]">{k}</dt>
                <dd className="font-semibold tabular-nums text-[#1e2142]">{v}</dd>
              </div>
            ))}
          </dl>
        </Row>
      </div>

      <button
        onClick={onNewSession}
        className="w-full rounded-[18px] bg-gradient-to-r from-[#a48bff] to-[#7b5cff] py-3.5 text-[14px] font-semibold text-white shadow-[0_10px_24px_rgba(123,92,255,0.35)] transition-transform active:scale-[0.99]"
      >
        {t("studio.newSession")}
      </button>
    </div>
  );
}
