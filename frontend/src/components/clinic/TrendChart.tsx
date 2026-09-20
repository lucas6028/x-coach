import { useCallback, useRef, useState } from "react";
import type { ClinicTrendPoint } from "../../api";
import { useI18n } from "../../lib/i18n";

interface Props {
  trend: ClinicTrendPoint[];
}

// Hand-drawn SVG, following components/history/HistoryStats.tsx's Spark/Ring idiom (no chart
// library in this app). Per the dataviz skill: ONE axis per plot — form score (0-100) and pain
// (0-10) are different scales, so they are two stacked charts, never one dual-axis plot. Colors are
// the app's own tokens, not the skill's reference palette: `text-primary` for form score (a single
// series needs no legend), a recessive `text-muted` for pain (a magnitude track, not an identity),
// and `text-danger` reserved for the flagged-point marker — a status, not a value color, and never
// the ONLY way a flagged point reads: the marker also gets a heavier stroke, and the flag itself is
// named in the accessible table below.
// The viewBox width tracks the rendered CSS width (one user unit = one px) instead of being a
// constant. A fixed `viewBox="0 0 320 72"` on a `w-full` svg keeps its own aspect ratio, so in a
// wide card the whole plot scales to the 72px height and sits as a 320px island in the middle of
// the panel; `preserveAspectRatio="none"` would fill the width but stretch every dot into an
// ellipse. Measuring avoids both. 320 stays the pre-measurement fallback (also what jsdom, where
// clientWidth is 0 and there is no ResizeObserver, renders with).
const VB_W_FALLBACK = 320;
const H_FORM = 72;
const H_PAIN = 40;
const PAD_X = 6;
const PAD_Y_FORM = 8;
const PAD_Y_PAIN = 6;

function xAt(i: number, n: number, width: number): number {
  if (n <= 1) return width / 2;
  return PAD_X + (i * (width - PAD_X * 2)) / (n - 1);
}

function yAt(value: number, min: number, max: number, height: number, padY: number): number {
  const clamped = Math.min(max, Math.max(min, value));
  const t = (clamped - min) / (max - min || 1);
  return height - padY - t * (height - padY * 2);
}

export default function TrendChart({ trend }: Props) {
  const { t, lang } = useI18n();

  const [vbW, setVbW] = useState(VB_W_FALLBACK);
  const observer = useRef<ResizeObserver | null>(null);
  // A callback ref, not an effect: the measured element only exists once there are >= 2 points, so
  // a mount-time effect would run while the empty state is still on screen and never see it.
  const measure = useCallback((el: HTMLDivElement | null) => {
    observer.current?.disconnect();
    observer.current = null;
    if (!el) return;
    const apply = (w: number) => setVbW(w > 0 ? Math.round(w) : VB_W_FALLBACK);
    apply(el.clientWidth);
    if (typeof ResizeObserver === "undefined") return; // jsdom, and older browsers
    const ro = new ResizeObserver((entries) => apply(entries[0].contentRect.width));
    ro.observe(el);
    observer.current = ro;
  }, []);

  if (trend.length < 2) {
    return (
      <div className="rounded-xl border border-dashed border-border-dark bg-content/[0.02] px-4 py-8 text-center">
        <p className="text-sm text-muted">{t("clinic.trendEmpty")}</p>
      </div>
    );
  }

  const n = trend.length;
  const formPoints = trend
    .map((p, i) =>
      p.form_score === null
        ? null
        : {
            i,
            x: xAt(i, n, vbW),
            y: yAt(p.form_score, 0, 100, H_FORM, PAD_Y_FORM),
            flagged: p.flagged,
          }
    )
    .filter((p): p is { i: number; x: number; y: number; flagged: boolean } => p !== null);
  const formPath = formPoints.map((p, idx) => `${idx === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");

  const painPoints = trend.map((p, i) => ({
    i,
    x: xAt(i, n, vbW),
    y: yAt(p.pain_nrs, 0, 10, H_PAIN, PAD_Y_PAIN),
    flagged: p.flagged,
  }));
  const painPath = painPoints.map((p, idx) => `${idx === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");

  return (
    <div ref={measure}>
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("clinic.formScoreAxis")}
        </p>
        <svg
          viewBox={`0 0 ${vbW} ${H_FORM}`}
          className="mt-1.5 h-[72px] w-full"
          role="img"
          aria-label={t("clinic.formChartLabel")}
        >
          <line
            x1={PAD_X}
            x2={vbW - PAD_X}
            y1={H_FORM - PAD_Y_FORM}
            y2={H_FORM - PAD_Y_FORM}
            stroke="rgb(var(--c-border))"
            strokeWidth="1"
          />
          {formPoints.length > 0 && (
            <path
              d={formPath}
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="text-primary"
            />
          )}
          {formPoints.map((p) => (
            <circle
              key={p.i}
              cx={p.x}
              cy={p.y}
              r={p.flagged ? 4.5 : 3}
              fill="currentColor"
              className={p.flagged ? "text-danger" : "text-primary"}
            />
          ))}
        </svg>
      </div>

      <div className="mt-3">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-faint">
          {t("clinic.painAxis")}
        </p>
        <svg
          viewBox={`0 0 ${vbW} ${H_PAIN}`}
          className="mt-1.5 h-[40px] w-full"
          role="img"
          aria-label={t("clinic.painChartLabel")}
        >
          <line
            x1={PAD_X}
            x2={vbW - PAD_X}
            y1={H_PAIN - PAD_Y_PAIN}
            y2={H_PAIN - PAD_Y_PAIN}
            stroke="rgb(var(--c-border))"
            strokeWidth="1"
          />
          <path
            d={painPath}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-muted"
          />
          {painPoints.map((p) => (
            <circle
              key={p.i}
              cx={p.x}
              cy={p.y}
              r={p.flagged ? 4.5 : 3}
              fill="currentColor"
              className={p.flagged ? "text-danger" : "text-muted"}
            />
          ))}
        </svg>
      </div>

      {/* Accessible summary: everything the two plots draw, as a real (visually hidden) table —
          the SVGs above are decorative beyond their own aria-label. */}
      <table className="sr-only">
        <caption>{t("clinic.trendTableCaption")}</caption>
        <thead>
          <tr>
            <th>{t("clinic.colDate")}</th>
            <th>{t("studio.formScore")}</th>
            <th>{t("checkin.painLabel")}</th>
            <th>{t("clinic.colFlagged")}</th>
          </tr>
        </thead>
        <tbody>
          {trend.map((p, i) => (
            <tr key={i}>
              <td>{new Date(p.created_at).toLocaleDateString(lang)}</td>
              <td>{p.form_score ?? "—"}</td>
              <td>{p.pain_nrs}</td>
              <td>{p.flagged ? t("clinic.flaggedYes") : t("clinic.flaggedNo")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
