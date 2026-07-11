/**
 * LumenLoader — Lumen 品牌載入動畫（lift-ready，尚未接進 app）。
 *
 * 用法：把 docs/mascot/icons/web-full.png 複製到 frontend/public/lumen/lumen-full.png，
 * 然後 <LumenLoader variant="scan" /> 即可。variant:
 *   "scan"  — 全螢幕／卡片不確定型（Lumen 掃描態，預設）
 *   "ring"  — 行內火環 spinner（純 CSS，無圖）
 *   "dots"  — 三色點（聊天輸入中／輕量等待）
 * 進度條屬確定型，另用 <LumenBar value={0.68} />。
 *
 * 樣式內嵌於元件（styled-jsx 風格的 <style>），無外部相依；若專案用 CSS Modules /
 * Tailwind，可把 CSS 抽出。色票對齊角色設定：gold #EC9B0C, glow #FFCF5B, ember #B76A0A,
 * night #20223F, cream #FDF7E7。
 */
import { useEffect, useState } from "react";

const IMG = "/lumen/lumen-full.png"; // place the transparent cutout here
const STATUS = ["讀取姿勢", "對照力學", "照亮原因"];

export function LumenLoader({
  variant = "scan",
  label = "Lumen 正在照亮你的這一組。",
}: {
  variant?: "scan" | "ring" | "dots";
  label?: string;
}) {
  const [i, setI] = useState(0);
  useEffect(() => {
    if (variant !== "scan") return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    const id = setInterval(() => setI((n) => (n + 1) % STATUS.length), 1600);
    return () => clearInterval(id);
  }, [variant]);

  if (variant === "ring") return <span className="lm-ring" role="status" aria-label="載入中" />;
  if (variant === "dots")
    return (
      <span className="lm-dots" role="status" aria-label="載入中">
        <b /><b /><b />
      </span>
    );

  return (
    <div className="lm-stage" role="status" aria-label={label}>
      <div className="lm-glow" />
      <div className="lm-char" style={{ background: `url(${IMG}) center/contain no-repeat` }} />
      <div
        className="lm-scan"
        style={{
          WebkitMaskImage: `url(${IMG})`,
          maskImage: `url(${IMG})`,
        }}
      />
      <div className="lm-status">
        <span>{STATUS[i]}</span>
        <span className="lm-el"><i /><i /><i /></span>
      </div>
    </div>
  );
}

export function LumenBar({ value }: { value: number }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div className="lm-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div className="lm-bar-track">
        <i style={{ transform: `scaleX(${value})` }} />
      </div>
    </div>
  );
}

/* Scoped styles — inject once. In a real app, move to a .css / CSS Module. */
export const lumenLoaderCss = `
.lm-stage{position:relative;width:220px;height:220px;display:grid;place-items:center;margin:0 auto;}
.lm-glow{position:absolute;inset:-6% -6% 8%;border-radius:50%;
  background:radial-gradient(circle at 50% 46%,rgba(255,207,91,.5),rgba(236,155,12,.12) 45%,transparent 68%);
  animation:lm-pulse 2.6s ease-in-out infinite;}
.lm-char{position:relative;width:86%;height:86%;filter:drop-shadow(0 12px 18px rgba(0,0,0,.25));
  animation:lm-bob 3.4s ease-in-out infinite;}
.lm-scan{position:absolute;width:86%;height:86%;overflow:hidden;
  -webkit-mask:center/contain no-repeat;mask:center/contain no-repeat;animation:lm-bob 3.4s ease-in-out infinite;}
.lm-scan::after{content:"";position:absolute;left:-10%;right:-10%;height:26%;top:-26%;
  background:linear-gradient(180deg,transparent,rgba(255,247,231,.9),transparent);
  animation:lm-sweep 2.4s cubic-bezier(.6,0,.4,1) infinite;}
.lm-status{position:absolute;bottom:-6px;font-weight:700;color:#20223F;display:flex;gap:2px;align-items:center;}
.lm-el i{width:4px;height:4px;border-radius:50%;background:#EC9B0C;display:inline-block;margin-left:3px;animation:lm-blink 1.4s infinite;}
.lm-el i:nth-child(2){animation-delay:.2s;}.lm-el i:nth-child(3){animation-delay:.4s;}
.lm-ring{display:inline-block;width:1.4em;height:1.4em;border-radius:50%;vertical-align:-.3em;
  background:conic-gradient(from 0deg,transparent 0 25%,#EC9B0C 90%,#FFCF5B);
  -webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 3px),#000 calc(100% - 2px));
  mask:radial-gradient(farthest-side,transparent calc(100% - 3px),#000 calc(100% - 2px));animation:lm-spin 1s linear infinite;}
.lm-dots{display:inline-flex;gap:6px;vertical-align:middle;}
.lm-dots b{width:9px;height:9px;border-radius:50%;background:#EC9B0C;animation:lm-pop 1.1s ease-in-out infinite;}
.lm-dots b:nth-child(2){animation-delay:.15s;background:#FFCF5B;}
.lm-dots b:nth-child(3){animation-delay:.3s;background:#B76A0A;}
.lm-bar-track{height:9px;border-radius:99px;background:#EFE8D8;overflow:hidden;}
.lm-bar-track i{display:block;height:100%;transform-origin:left;border-radius:99px;transition:transform .3s ease;
  background:linear-gradient(90deg,#B76A0A,#EC9B0C 60%,#FFCF5B);}
@keyframes lm-bob{0%,100%{transform:translateY(-3%);}50%{transform:translateY(3%);}}
@keyframes lm-pulse{0%,100%{opacity:.55;transform:scale(.96);}50%{opacity:1;transform:scale(1.04);}}
@keyframes lm-spin{to{transform:rotate(360deg);}}
@keyframes lm-sweep{0%{top:-26%;}100%{top:100%;}}
@keyframes lm-blink{0%,100%{opacity:.25;}50%{opacity:1;}}
@keyframes lm-pop{0%,100%{transform:scale(.55);opacity:.5;}50%{transform:scale(1);opacity:1;}}
@media (prefers-reduced-motion:reduce){
  .lm-char,.lm-scan,.lm-glow,.lm-ring,.lm-dots b,.lm-bar-track i{animation:none!important;}
  .lm-scan::after{opacity:0;}.lm-el{display:none;}
}`;
