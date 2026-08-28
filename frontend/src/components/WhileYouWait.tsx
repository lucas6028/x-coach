import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight, CheckCircle, ListNumbers, WarningOctagon } from "@phosphor-icons/react";
import { movementLabel, useI18n } from "../lib/i18n";
import { movementDetail } from "../lib/movementDetail";
import { movementMistakes } from "../lib/movementMistakes";

// The studio's "beside the loader" panel: while an analysis runs, the right column stops promising
// what comes back and instead shows what the user was just filmed doing — this movement's steps and
// the mistakes the analyzer can actually report. The wait is ~20s of dead time next to a running
// mascot; this is the one moment the user is guaranteed to be looking at the screen with nothing to
// do, so it is where a reminder of the correct pattern is worth the most.
//
// It reads the SAME two content modules the movement detail page reads (lib/movementDetail.ts,
// lib/movementMistakes.ts) rather than lifting that page's cards: those are built for a full-width
// tabbed page and this is a 20rem column beside a 200px stage. Sharing the DATA is the part that
// must not fork; the layout deliberately does. The "full guide" link hands off to the real page for
// anything longer than a glance.

/** True when this movement has authored content worth showing. Callers use it to decide whether to
 *  render this panel at all instead of getting an empty shell back — an uncatalogued movement (or
 *  one with no registered detector rules) has nothing to say here. */
export function hasMovementGuide(movement: string): boolean {
  return (movementDetail(movement)?.steps.length ?? 0) > 0 || movementMistakes(movement).length > 0;
}

type Tab = "steps" | "mistakes";

export default function WhileYouWait({ movement }: { movement: string }) {
  const { t, lang } = useI18n();
  const detail = movementDetail(movement);
  const steps = detail?.steps ?? [];
  const mistakes = movementMistakes(movement);

  // Open on whichever half exists. Local state, not the URL: this panel lives inside a transient
  // waiting state, and a ?tab= here would collide with the movement detail page's own tab param.
  const [tab, setTab] = useState<Tab>(steps.length ? "steps" : "mistakes");

  if (!steps.length && !mistakes.length) return null;
  const showTabs = steps.length > 0 && mistakes.length > 0;
  const active = showTabs ? tab : steps.length ? "steps" : "mistakes";

  return (
    <div className="glass-panel xc-pop overflow-hidden rounded-[18px]">
      <div className="flex items-center gap-2.5 border-b border-white/70 bg-white/50 px-5 py-4">
        {/* The header mark follows the open half, so the card's identity matches what is under it. */}
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#f0eaff] text-primary">
          {active === "steps" ? (
            <ListNumbers size={18} weight="duotone" />
          ) : (
            <WarningOctagon size={18} weight="duotone" />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-display text-[15px] font-semibold text-[#1e2142]">{t("wait.title")}</p>
          {/* Wraps rather than truncates: at this width the line only just overflows, and clipping
              it mid-word ("what usually goe…") looks like a bug rather than a summary. */}
          <p className="text-[12.5px] leading-snug text-[#59648f]">
            {t("wait.sub", { movement: movementLabel(t, movement) })}
          </p>
        </div>
        <Link
          to={`/movements/${encodeURIComponent(movement)}`}
          className="flex shrink-0 items-center gap-0.5 rounded-lg px-1.5 py-1 text-[11.5px] font-medium text-primary hover:bg-primary/10"
        >
          {t("wait.full")}
          <ArrowUpRight size={13} />
        </Link>
      </div>

      {showTabs && (
        <div className="flex gap-1 border-b border-white/70 px-3 py-2" role="tablist">
          {(["steps", "mistakes"] as const).map((k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={active === k}
              onClick={() => setTab(k)}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-[13px] font-medium transition-colors ${
                active === k
                  ? "bg-primary/10 text-primary"
                  : "text-[#59648f] hover:bg-white/60 hover:text-[#1e2142]"
              }`}
            >
              {k === "steps" ? <ListNumbers size={15} /> : <WarningOctagon size={15} />}
              {t(k === "steps" ? "detail.tabHowTo" : "detail.tabMistakes")}
            </button>
          ))}
        </div>
      )}

      {/* Capped and scrollable, but the cap is generous: this is the one panel the user has time to
          actually read, and a squeezed list makes them scroll through the very wait it is meant to
          fill. The viewport half of the min() is what keeps it honest — on a short window the cap
          has to stay below the fold's worth of room, or the loader this sits beside (and sits BELOW,
          on phones) gets pushed off the top of the screen. */}
      <div className="max-h-[min(30rem,58vh)] overflow-y-auto scrollbar-thin">
        {active === "steps" ? (
          <ol className="divide-y divide-white/70">
            {steps.map((s, i) => (
              <li key={i} className="flex items-center gap-3.5 px-5 py-3.5">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#f0eaff] font-mono text-[12px] font-semibold text-primary">
                  {i + 1}
                </span>
                {s.image && (
                  <img
                    src={s.image}
                    alt=""
                    loading="lazy"
                    className="h-16 w-16 shrink-0 rounded-xl bg-white/60 object-contain"
                  />
                )}
                <p className="min-w-0 text-[13.5px] leading-relaxed text-[#1e2142]">{s.text[lang]}</p>
              </li>
            ))}
          </ol>
        ) : (
          <ul className="divide-y divide-white/70">
            {mistakes.map((m) => (
              <li key={m.id} className="flex items-start gap-3.5 px-5 py-3.5">
                {m.art ? (
                  <img
                    src={m.art.wrong}
                    alt=""
                    loading="lazy"
                    className="h-16 w-16 shrink-0 rounded-xl bg-white/60 object-contain"
                  />
                ) : (
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-warning/15 text-warning">
                    <WarningOctagon size={16} weight="fill" />
                  </span>
                )}
                <div className="min-w-0">
                  <p className="text-[13.5px] font-medium leading-snug text-[#1e2142]">
                    {m.title[lang]}
                  </p>
                  <p className="mt-1 text-[12.5px] leading-snug text-[#59648f]">
                    {m.subtitle[lang]}
                  </p>
                  {/* One cue, not the whole fix list — this is a glance while waiting, and the
                      rest is one tap away behind "full guide". */}
                  {m.fixes[lang][0] && (
                    <p className="mt-2 flex items-start gap-1.5 text-[12.5px] leading-snug text-[#1e2142]">
                      <CheckCircle size={14} weight="fill" className="mt-[2px] shrink-0 text-primary" />
                      <span className="min-w-0">{m.fixes[lang][0]}</span>
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
