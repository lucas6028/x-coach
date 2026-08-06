import { Warning } from "@phosphor-icons/react";
import type { Detection } from "../../api";
import { fmtTime } from "../../lib/format";
import { faultLabel, phaseLabel, useI18n } from "../../lib/i18n";
import { keyEvidence } from "../../lib/retrieval";

const SHOWN = 2;

// The reference's floating "Detected Errors" card, overlaid on the top-right of the video. Each
// row is a real detection; clicking it seeks the clip to where it happened.
export default function DetectedErrorsCard({
  detections,
  onSeek,
  activeFaultId,
}: {
  detections: Detection[];
  onSeek: (t: number) => void;
  activeFaultId: string | null;
}) {
  const { t } = useI18n();
  if (detections.length === 0) return null;
  const shown = detections.slice(0, SHOWN);
  const rest = detections.length - shown.length;

  return (
    <div className="glass-over-video rounded-[14px] p-3">
      <div className="mb-2.5 flex items-center gap-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#ffe8e8] text-[#ff5a5a]">
          <Warning size={12} weight="fill" />
        </span>
        <span className="text-[11px] font-bold text-[#1e2142]">{t("studio.detectedErrors")}</span>
      </div>
      <ul className="space-y-2">
        {shown.map((d, i) => (
          <li key={i}>
            <button
              onClick={() => onSeek(d.start_time)}
              className={`flex w-full gap-2 rounded-lg border p-2 text-left transition-colors ${
                activeFaultId === d.fault_id
                  ? "border-[#ffc9c9] bg-[#ffecec]"
                  : "border-[#ffe0e0] bg-[#fff5f5] hover:border-[#ffc9c9]"
              }`}
            >
              <span className="mt-0.5 flex h-3 w-3 shrink-0 items-center justify-center rounded-full border-[1.5px] border-[#ff6b6b] bg-white">
                <span className="h-1 w-1 rounded-full bg-[#ff6b6b]" />
              </span>
              <span className="min-w-0">
                <span className="block text-[11px] font-semibold leading-tight text-[#1e2142]">
                  {faultLabel(t, d.fault_name)}
                </span>
                <span className="block truncate text-[10px] leading-tight text-[#63709f]">
                  {keyEvidence(d) ?? `${fmtTime(d.start_time)} · ${phaseLabel(t, d.phase)}`}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      {rest > 0 && (
        <p className="mt-2 text-[10px] font-medium text-[#63709f]">
          {t("studio.moreFaults", { n: rest })}
        </p>
      )}
    </div>
  );
}
