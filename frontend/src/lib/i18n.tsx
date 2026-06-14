import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { titleCase } from "./format";

export type Lang = "en" | "zh-Hant";

const STORAGE_KEY = "lang";

// <html lang="…"> value per language (helps the browser pick fonts / hyphenation).
const HTML_LANG: Record<Lang, string> = {
  en: "en",
  "zh-Hant": "zh-Hant",
};

type Dict = Record<string, string>;

// English is the source/fallback dictionary; every key here should have a zh-Hant counterpart.
const en: Dict = {
  // Sidebar
  "nav.analyse": "Analyse",
  "nav.library": "Library",
  "nav.hide": "Hide navigation",
  "nav.show": "Show navigation",
  "sidebar.version": "Prototype v0.1",
  "sidebar.tagline": "Pose · Rules · GraphRAG",

  // Header
  "header.session": "Session: {id}",
  "header.title": "Squat Analysis",
  "header.processing": "PROCESSING",
  "header.complete": "ANALYSIS COMPLETE",
  "header.awaiting": "AWAITING INPUT",
  "header.view": "{type} view",

  // Camera views
  "view.front": "Front",
  "view.side": "Side",
  "view.rear": "Rear",
  "view.left": "Left",
  "view.right": "Right",
  "view.unknown": "Unknown",

  // Video panel
  "video.faultOne": "1 fault detected",
  "video.faultMany": "{count} faults detected",
  "video.noFaults": "No faults detected",
  "a11y.play": "Play",
  "a11y.pause": "Pause",
  "a11y.fullscreen": "Toggle fullscreen",

  // Timeline
  "timeline.fault": "Fault",
  "timeline.neutral": "Neutral",

  // Metrics
  "metric.cameraView": "Camera View",
  "metric.faults": "Faults",
  "metric.lowerBodyVis": "Lower-body Vis.",
  "metric.validFrames": "Valid Frames",
  "metric.conf": "conf {v}",
  "metric.peakSeverity": "peak severity {v}",
  "metric.cleanRep": "clean rep",
  "metric.landmarkConf": "landmark confidence",
  "metric.framesRatio": "{valid}/{total} frames",

  // Chat input
  "chat.placeholder": "Ask the AI Coach… (LLM layer coming soon)",
  "chat.title": "Conversational coaching arrives with the LLM layer.",

  // Library picker
  "library.title": "Sample Library",
  "library.loading": "Loading…",
  "library.clean": "clean",
  "library.filter.all": "All",
  "library.filter.knees_inward": "Knee Valgus",
  "library.filter.knees_forward": "Knees Forward",
  "library.filter.shallow_depth": "Shallow",
  "library.filter.excessive_forward_lean": "Forward Lean",
  "a11y.close": "Close",

  // Reasoning / coaching feedback
  "feedback.title": "Coaching Feedback",
  "feedback.badge": "rule + GraphRAG",
  "feedback.noFaults": "No biomechanical faults detected. Clean rep.",
  "feedback.graphragContext": "GraphRAG Context",
  "feedback.likelyCause": "Likely cause:",
  "feedback.injuryRisk": "Injury risk:",
  "feedback.cue": "Cue",
  "feedback.phaseTag": "during {phase} phase",

  // Severity
  "severity.high": "High",
  "severity.moderate": "Moderate",
  "severity.mild": "Mild",

  // Squat phases
  "phase.descent": "Descent",
  "phase.ascent": "Ascent",
  "phase.bottom": "Bottom",
  "phase.top": "Top",
  "phase.eccentric": "Eccentric",
  "phase.concentric": "Concentric",
  "phase.hold": "Hold",
  "phase.transition": "Transition",
  "phase.setup": "Setup",
  "phase.full": "Full",

  // Fault names
  "fault.knees_inward": "Knee Valgus",
  "fault.knees_forward": "Knees Forward",
  "fault.shallow_depth": "Shallow Depth",
  "fault.excessive_forward_lean": "Excessive Forward Lean",
  "fault.heel_lift": "Heel Lift",
  "fault.butt_wink": "Butt Wink",
  "fault.asymmetric_shift": "Asymmetric Shift",

  // Knowledge graph
  "kg.title": "Knowledge Graph",
  "kg.empty": "No graph context for this clip.",

  // App shell
  "app.pickSample": "…or pick a clip from the sample library",
  "app.loading": "Loading {id}…",
  "app.analysing": "Extracting pose & analysing… (this can take ~20s)",
  "tab.coaching": "Coaching",
  "tab.graph": "Knowledge Graph",

  // Upload dropzone
  "upload.analysing": "Analysing…",
  "upload.prompt": "Drop a squat video or tap to upload",
  "upload.hint": "MP4 / MOV · single athlete · side or rear view",

  // Demo onboarding (empty state)
  "demo.heading": "Analyze a squat in about 20 seconds.",
  "demo.sub": "Upload a clip or open a labeled sample. You get a skeleton overlay, a fault timeline, and coaching feedback x-coach can trace to the cause.",
  "demo.sampleBtn": "Open a sample clip",
  "demo.or": "or",
  "demo.getTitle": "What comes back",
  "demo.get1.title": "Skeleton and faults",
  "demo.get1.body": "Pose overlay with every detected fault marked on the timeline.",
  "demo.get2.title": "Grounded feedback",
  "demo.get2.body": "An observation, a likely cause, and a corrective cue per fault.",
  "demo.get3.title": "Knowledge graph",
  "demo.get3.body": "The retrieval path that links each symptom to its cause.",
  "demo.errorTitle": "That clip did not go through",

  // Theme toggle
  "theme.light": "Light",
  "theme.system": "System",
  "theme.dark": "Dark",
  "theme.label": "Theme: {name} (click to change)",
  "theme.aria": "Theme: {name}",

  // Language toggle
  "lang.label": "Language",
  "lang.en": "English",
  "lang.zh-Hant": "繁體中文",
};

const zhHant: Dict = {
  // Sidebar
  "nav.analyse": "分析",
  "nav.library": "資料庫",
  "nav.hide": "隱藏導覽列",
  "nav.show": "顯示導覽列",
  "sidebar.version": "原型 v0.1",
  "sidebar.tagline": "姿態 · 規則 · GraphRAG",

  // Header
  "header.session": "工作階段：{id}",
  "header.title": "深蹲分析",
  "header.processing": "處理中",
  "header.complete": "分析完成",
  "header.awaiting": "等待輸入",
  "header.view": "{type}視角",

  // Camera views
  "view.front": "正面",
  "view.side": "側面",
  "view.rear": "背面",
  "view.left": "左側",
  "view.right": "右側",
  "view.unknown": "未知",

  // Video panel
  "video.faultOne": "偵測到 1 個錯誤",
  "video.faultMany": "偵測到 {count} 個錯誤",
  "video.noFaults": "未偵測到錯誤",
  "a11y.play": "播放",
  "a11y.pause": "暫停",
  "a11y.fullscreen": "切換全螢幕",

  // Timeline
  "timeline.fault": "錯誤",
  "timeline.neutral": "正常",

  // Metrics
  "metric.cameraView": "拍攝視角",
  "metric.faults": "錯誤數",
  "metric.lowerBodyVis": "下肢可見度",
  "metric.validFrames": "有效影格",
  "metric.conf": "信心 {v}",
  "metric.peakSeverity": "最高嚴重度 {v}",
  "metric.cleanRep": "標準動作",
  "metric.landmarkConf": "關鍵點信心度",
  "metric.framesRatio": "{valid}/{total} 影格",

  // Chat input
  "chat.placeholder": "詢問 AI 教練…（LLM 功能即將推出）",
  "chat.title": "對話式教練功能將隨 LLM 層推出。",

  // Library picker
  "library.title": "範例資料庫",
  "library.loading": "載入中…",
  "library.clean": "標準",
  "library.filter.all": "全部",
  "library.filter.knees_inward": "膝蓋內夾",
  "library.filter.knees_forward": "膝蓋前移",
  "library.filter.shallow_depth": "深度不足",
  "library.filter.excessive_forward_lean": "軀幹前傾",
  "a11y.close": "關閉",

  // Reasoning / coaching feedback
  "feedback.title": "教練回饋",
  "feedback.badge": "規則 + GraphRAG",
  "feedback.noFaults": "未偵測到生物力學錯誤，標準動作。",
  "feedback.graphragContext": "GraphRAG 脈絡",
  "feedback.likelyCause": "可能原因：",
  "feedback.injuryRisk": "受傷風險：",
  "feedback.cue": "提示",
  "feedback.phaseTag": "（{phase}階段）",

  // Severity
  "severity.high": "高",
  "severity.moderate": "中",
  "severity.mild": "低",

  // Squat phases
  "phase.descent": "下降",
  "phase.ascent": "上升",
  "phase.bottom": "最低點",
  "phase.top": "頂點",
  "phase.eccentric": "離心",
  "phase.concentric": "向心",
  "phase.hold": "停頓",
  "phase.transition": "轉換",
  "phase.setup": "準備",
  "phase.full": "完整",

  // Fault names
  "fault.knees_inward": "膝蓋內夾",
  "fault.knees_forward": "膝蓋前移",
  "fault.shallow_depth": "深度不足",
  "fault.excessive_forward_lean": "軀幹過度前傾",
  "fault.heel_lift": "腳跟離地",
  "fault.butt_wink": "骨盆後傾",
  "fault.asymmetric_shift": "左右不對稱",

  // Knowledge graph
  "kg.title": "知識圖譜",
  "kg.empty": "此片段沒有圖譜脈絡。",

  // App shell
  "app.pickSample": "…或從範例資料庫挑選片段",
  "app.loading": "載入 {id} 中…",
  "app.analysing": "擷取姿態並分析中…（約需 20 秒）",
  "tab.coaching": "教練回饋",
  "tab.graph": "知識圖譜",

  // Upload dropzone
  "upload.analysing": "分析中…",
  "upload.prompt": "拖放深蹲影片或點擊上傳",
  "upload.hint": "MP4 / MOV · 單一運動員 · 側面或背面視角",

  // Demo onboarding (empty state)
  "demo.heading": "約 20 秒，分析一段深蹲。",
  "demo.sub": "上傳影片或開啟已標註的範例。你會得到骨架疊圖、錯誤時間軸，以及能追溯成因的教練回饋。",
  "demo.sampleBtn": "開啟範例片段",
  "demo.or": "或",
  "demo.getTitle": "你會得到",
  "demo.get1.title": "骨架與錯誤",
  "demo.get1.body": "骨架疊圖，並在時間軸上標出每個偵測到的錯誤。",
  "demo.get2.title": "有據回饋",
  "demo.get2.body": "每個錯誤都附上觀察、可能成因與修正提示。",
  "demo.get3.title": "知識圖譜",
  "demo.get3.body": "連結每個現象到其成因的檢索路徑。",
  "demo.errorTitle": "這段片段沒有成功處理",

  // Theme toggle
  "theme.light": "淺色",
  "theme.system": "系統",
  "theme.dark": "深色",
  "theme.label": "主題：{name}（點擊切換）",
  "theme.aria": "主題：{name}",

  // Language toggle
  "lang.label": "語言",
  "lang.en": "English",
  "lang.zh-Hant": "繁體中文",
};

const DICTS: Record<Lang, Dict> = { en, "zh-Hant": zhHant };

export const LANGS: { value: Lang; short: string }[] = [
  { value: "en", short: "EN" },
  { value: "zh-Hant", short: "中" },
];

export function getStoredLang(): Lang {
  const l = (typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY)) as Lang | null;
  if (l === "en" || l === "zh-Hant") return l;
  // First visit: honour the browser preference (any Chinese locale -> Traditional Chinese).
  if (typeof navigator !== "undefined" && /^zh/i.test(navigator.language)) return "zh-Hant";
  return "en";
}

export type TFunc = (key: string, vars?: Record<string, string | number>) => string;

interface I18nValue {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: TFunc;
}

const I18nContext = createContext<I18nValue | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(getStoredLang);

  useEffect(() => {
    document.documentElement.lang = HTML_LANG[lang];
  }, [lang]);

  const setLang = (l: Lang) => {
    localStorage.setItem(STORAGE_KEY, l);
    setLangState(l);
  };

  const t: TFunc = (key, vars) => {
    let s = DICTS[lang][key] ?? en[key] ?? key;
    if (vars) {
      for (const [k, v] of Object.entries(vars)) {
        s = s.replace(new RegExp(`\\{${k}\\}`, "g"), String(v));
      }
    }
    return s;
  };

  return <I18nContext.Provider value={{ lang, setLang, t }}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within an I18nProvider");
  return ctx;
}

// Data-driven labels: look up a namespaced key, else fall back to humanised raw text.
function dataLabel(t: TFunc, prefix: string, raw: string): string {
  if (!raw) return titleCase(raw);
  const key = `${prefix}.${raw}`;
  const v = t(key);
  return v === key ? titleCase(raw) : v;
}

export const faultLabel = (t: TFunc, raw: string) => dataLabel(t, "fault", raw);
export const viewLabel = (t: TFunc, raw: string) => dataLabel(t, "view", raw);
export const phaseLabel = (t: TFunc, raw: string) => dataLabel(t, "phase", raw);

export function severityText(t: TFunc, sev: number): string {
  if (sev >= 0.75) return t("severity.high");
  if (sev >= 0.4) return t("severity.moderate");
  return t("severity.mild");
}
