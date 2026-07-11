import { useEffect, useState } from "react";
import { useI18n } from "../lib/i18n";

// Lumen — the AI coach's on-brand waiting states. The transparent full-body cutout and head
// avatar live in /public/lumen/ (see docs/mascot for the source). Styles are in index.css under
// the "Lumen loader" section (app convention; keeps this a markup-only leaf, no runtime <style>).
const BODY = "/lumen/lumen-full.png"; // transparent full-body cutout, drives the scan silhouette
const HEAD = "/lumen/lumen-head.png"; // transparent head cutout, the chat avatar

// The scan stage cycles these three micro-labels — they mirror the analysis pipeline (read the
// pose, check it against mechanics, surface the reason) so the wait narrates what Lumen is doing.
const STEP_KEYS = ["loader.step1", "loader.step2", "loader.step3"] as const;

// "scan" — full 220px stage: Lumen bobbing under a warm glow with a light band sweeping her
//          silhouette, plus a cycling status line. For the analysis waiting state.
// "dots"  — three brand-colour dots. For the inline "Lumen is thinking…" chat indicator.
export function LumenLoader({
  variant = "scan",
  caption,
}: {
  variant?: "scan" | "dots";
  caption?: string;
}) {
  const { t } = useI18n();
  const [i, setI] = useState(0);

  useEffect(() => {
    if (variant !== "scan") return;
    // Don't cycle the status under reduced motion — the label just holds on the first step.
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const id = setInterval(() => setI((n) => (n + 1) % STEP_KEYS.length), 1600);
    return () => clearInterval(id);
  }, [variant]);

  if (variant === "dots")
    return (
      <span className="lm-dots" role="status" aria-label={t("loader.aria")}>
        <b />
        <b />
        <b />
      </span>
    );

  return (
    <div className="lm-scan-wrap" role="status" aria-label={caption || t("loader.aria")}>
      <div className="lm-stage">
        <div className="lm-glow" />
        <div className="lm-char" style={{ backgroundImage: `url(${BODY})` }} />
        <div
          className="lm-scan"
          style={{ WebkitMaskImage: `url(${BODY})`, maskImage: `url(${BODY})` }}
        />
        <div className="lm-status">
          <span>{t(STEP_KEYS[i])}</span>
          <span className="lm-el">
            <i />
            <i />
            <i />
          </span>
        </div>
      </div>
      {caption && <p className="lm-caption">{caption}</p>}
    </div>
  );
}

// Lumen's face, used as the coach's identity mark in the tray (header, sign-in prompt, each coach
// turn). Decorative by default — a "Lumen" text label always sits alongside it, so the empty alt
// avoids a doubled announcement for screen readers.
export function LumenAvatar({ size = 24, className = "" }: { size?: number; className?: string }) {
  return (
    <span className={`lm-avatar ${className}`} style={{ width: size, height: size }}>
      <img src={HEAD} alt="" width={size} height={size} />
    </span>
  );
}
