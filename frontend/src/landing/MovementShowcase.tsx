import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import {
  BarbellIcon,
  PersonArmsSpreadIcon,
  PersonSimpleRunIcon,
  PersonSimpleTaiChiIcon,
  type Icon,
} from "@phosphor-icons/react";
import Reveal from "./Reveal";
import SkeletonStage from "./SkeletonStage";
import { useI18n } from "../lib/i18n";

const SECTION = "mx-auto w-full max-w-6xl px-5 sm:px-8";
const ROTATE_MS = 6000;

// Real Pexels footage (license-clear, see public/demo) with real MediaPipe pose tracks
// (public/demo/<id>.pose.json). Each clip wipes from original footage to the skeleton overlay.
const CLIPS: { id: string; src: string; poster: string; icon: Icon }[] = [
  { id: "squat", src: "/demo/squat.mp4", poster: "/demo/squat.jpg", icon: BarbellIcon },
  { id: "pushups", src: "/demo/pushups.mp4", poster: "/demo/pushups.jpg", icon: PersonArmsSpreadIcon },
  { id: "highknee", src: "/demo/highknee.mp4", poster: "/demo/highknee.jpg", icon: PersonSimpleRunIcon },
  { id: "situps", src: "/demo/situps.mp4", poster: "/demo/situps.jpg", icon: PersonSimpleTaiChiIcon },
];

export default function MovementShowcase() {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);

  const clip = CLIPS[active];

  // Auto-advance through the library; pauses on hover and under reduced motion.
  useEffect(() => {
    if (reduce || paused) return;
    const id = setTimeout(() => setActive((i) => (i + 1) % CLIPS.length), ROTATE_MS);
    return () => clearTimeout(id);
  }, [active, paused, reduce]);

  return (
    <section className={`${SECTION} border-t border-white/10 py-24`}>
      <Reveal>
        <h2 className="max-w-2xl font-display text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">
          {t("landing.showcase.title")}
        </h2>
        <p className="mt-4 max-w-xl text-zinc-400">{t("landing.showcase.sub")}</p>
      </Reveal>

      <div className="mt-12 grid items-center gap-8 lg:grid-cols-12 lg:gap-10">
        <Reveal className="lg:col-span-7" delay={0.05}>
          <SkeletonStage
            key={clip.id}
            clipId={clip.id}
            src={clip.src}
            poster={clip.poster}
            name={t(`landing.showcase.${clip.id}.name`)}
            analyzingLabel={t("landing.showcase.analyzing")}
            playLabel={t("a11y.play")}
            pauseLabel={t("a11y.pause")}
            onHoverChange={setPaused}
          />
        </Reveal>

        {/* Selector: pick a movement, drives the stage */}
        <div className="lg:col-span-5">
          <ul className="grid gap-2.5">
            {CLIPS.map((c, i) => {
              const isActive = i === active;
              const Ic = c.icon;
              return (
                <li key={c.id}>
                  <button
                    onClick={() => setActive(i)}
                    onMouseEnter={() => setPaused(true)}
                    onMouseLeave={() => setPaused(false)}
                    aria-pressed={isActive}
                    className={`relative w-full overflow-hidden rounded-xl border p-4 text-left transition-colors ${
                      isActive
                        ? "border-[#16b8a8]/40 bg-gradient-to-br from-[#16b8a8]/[0.12] to-transparent"
                        : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
                    }`}
                  >
                    <div className="flex items-center gap-3.5">
                      <span
                        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border ${
                          isActive
                            ? "border-[#3ee07a]/30 bg-[#3ee07a]/10 text-[#3ee07a]"
                            : "border-white/10 bg-[#15191b] text-zinc-400"
                        }`}
                      >
                        <Ic size={20} weight="duotone" />
                      </span>
                      <div className="min-w-0">
                        <p
                          className={`font-display text-base font-semibold ${
                            isActive ? "text-zinc-50" : "text-zinc-200"
                          }`}
                        >
                          {t(`landing.showcase.${c.id}.name`)}
                        </p>
                        <p className="mt-0.5 truncate text-[13px] text-zinc-400">
                          {t(`landing.showcase.${c.id}.note`)}
                        </p>
                      </div>
                    </div>

                    {/* auto-rotate progress, only on the active row */}
                    {isActive && !reduce && !paused && (
                      <motion.span
                        key={active}
                        aria-hidden
                        className="absolute bottom-0 left-0 h-0.5 origin-left bg-gradient-to-r from-[#5ffb6f] to-[#16b8a8]"
                        style={{ width: "100%" }}
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: 1 }}
                        transition={{ duration: ROTATE_MS / 1000, ease: "linear" }}
                      />
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </section>
  );
}
