import { Lightbulb } from "@phosphor-icons/react";
import type { Analysis } from "../../api";
import { faultLabel, useI18n } from "../../lib/i18n";
import { retrievalByFault, summaryCategory } from "../../lib/retrieval";
import StudioCard from "./StudioCard";

const SHOWN = 3;

// The reference's "Tips for Improvement" list. Every line is a corrective cue the knowledge graph
// actually returned for one of THIS clip's faults (`summaryCategory(..., "corrections")` — the
// same helper the fault cards read, so the panel and the cards cannot disagree). Faults whose
// retrieval carried no correction are skipped rather than padded with generic advice; if none of
// them did, the panel says so.
export default function TipsCard({ analysis }: { analysis: Analysis }) {
  const { t } = useI18n();
  const byFault = retrievalByFault(analysis.retrievals);

  const tips: { title: string; desc: string }[] = [];
  for (const d of analysis.detections) {
    if (tips.length === SHOWN) break;
    const corrections = summaryCategory(byFault.get(d.fault_id), "corrections");
    if (!corrections.length) continue;
    tips.push({ title: corrections[0], desc: faultLabel(t, d.fault_name) });
  }

  return (
    <StudioCard icon={<Lightbulb size={14} weight="bold" />} title={t("studio.tips")} index={2}>
      {tips.length === 0 ? (
        <p className="text-[11px] leading-relaxed text-[#63709f]">
          {analysis.detections.length === 0 ? t("studio.tipsClean") : t("studio.tipsNone")}
        </p>
      ) : (
        <ol className="space-y-3">
          {tips.map((tip, i) => (
            <li key={i} className="flex gap-2.5">
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-[#d0f0dc] bg-[#e6f7ed] text-[10px] font-bold text-[#1a9e5a]">
                {i + 1}
              </span>
              <span className="min-w-0">
                <span className="block text-[11px] font-bold leading-tight text-[#1e2142]">
                  {tip.title}
                </span>
                <span className="mt-0.5 block text-[10px] leading-[1.4] text-[#59648f]">
                  {tip.desc}
                </span>
              </span>
            </li>
          ))}
        </ol>
      )}
    </StudioCard>
  );
}
