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
  "nav.newAnalysis": "New analysis",
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

  // Chat input — disabled fallback (auth/LLM not configured) + the working grounded chat.
  "chat.placeholder": "Ask the AI Coach… (LLM layer coming soon)",
  "chat.title": "Conversational coaching arrives with the LLM layer.",
  "chat.heading": "AI Coach",
  "chat.grounded": "grounded in your analysis",
  "chat.groundedShort": "grounded",
  "chat.intro": "Ask a follow-up about your squat. Answers stay grounded in the detected faults and retrieved cues.",
  "chat.suggestFix": "What should I fix first?",
  "chat.suggestDrill": "Show me a drill for this",
  "chat.suggestWhy": "Why does this matter?",
  "chat.placeholderActive": "Ask a follow-up…",
  "chat.send": "Send message",
  "chat.thinking": "Coach is thinking…",
  "chat.signIn": "Sign in to chat with the AI coach about this analysis.",
  "chat.error": "Couldn't reach the coach. Please try again.",
  "chat.sessionExpired": "Your session expired. Please sign in again to keep chatting.",
  "chat.you": "You",
  "chat.coach": "Coach",
  "coach.followUp": "Follow-up",

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
  "feedback.cause": "Cause",
  "feedback.risk": "Risk",
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
  "kg.chain": "Fault → cause → fix",
  "kg.nodes": "{count} nodes",
  "kg.empty": "No graph context for this clip.",
  "kg.focus": "Showing",
  "kg.expand": "Expand to full screen",
  "kg.collapse": "Close full screen",
  "kg.cause": "Cause",
  "kg.risk": "Risk",
  "kg.correction": "Fix",
  "kg.evidence": "Evidence",

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

  // Landing — nav
  "landing.nav.how": "How it works",
  "landing.nav.pipeline": "The pipeline",
  "landing.nav.eval": "Evaluation",
  "landing.cta.open": "Open the demo",

  // Landing — hero
  "landing.hero.titlePre": "Coaching cues you can ",
  "landing.hero.titleAccent": "trace to the joint",
  "landing.hero.titlePost": ".",
  "landing.hero.sub":
    "x-coach reads a squat video, locates the fault, traces its cause in a biomechanics knowledge graph, and explains the fix.",
  "landing.hero.readMethod": "Read the method",

  // Landing — problem
  "landing.problem.title": "Scores don't coach. Generic models guess.",
  "landing.problem.sub":
    "Action-quality models hand back a number with no instruction. Ask a general language model and it sounds confident while inventing the biomechanics.",
  "landing.problem.aqs.label": "Action quality scoring",
  "landing.problem.aqs.body":
    "Returns a 0 to 100 rating. The lifter learns they scored a 71, not what to change or why.",
  "landing.problem.llm.label": "General language models",
  "landing.problem.llm.body":
    "Produce fluent advice untethered from the video, and hallucinate causes that the footage never showed.",
  "landing.problem.xcoach.title": "Grounded by construction",
  "landing.problem.point1": "Sees the fault in the actual frames",
  "landing.problem.point2": "Retrieves the cause from a sourced knowledge graph",
  "landing.problem.point3": "Explains the fix it can point back to",

  // Landing — pipeline
  "landing.pipeline.kicker": "The system",
  "landing.pipeline.title": "Four modules, one closed loop from pixels to prescription.",
  "landing.stage.perceive.title": "Perceive",
  "landing.stage.perceive.body":
    "Pose landmarks and VideoMAE motion features extract geometry, then localize the fault in time.",
  "landing.stage.retrieve.title": "Retrieve",
  "landing.stage.retrieve.body":
    "GraphRAG walks the fitness knowledge graph from the visible symptom to its deeper cause.",
  "landing.stage.reason.title": "Reason",
  "landing.stage.reason.body":
    "A chain of thought moves from observation to attribution to prescription, grounded in the retrieved evidence.",
  "landing.stage.coach.title": "Coach",
  "landing.stage.coach.body":
    "A diagnosis report and corrective cues come back, with the exact frames highlighted.",

  // Landing — diagnosis
  "landing.diagnosis.title": "Every cue carries its reasoning.",
  "landing.diagnosis.sub":
    "One detected fault, walked from what the camera saw to the exercise you should do about it.",
  "landing.step.observation.tag": "Perception",
  "landing.step.observation.title": "Observation",
  "landing.step.observation.body":
    "The left knee crosses inward of the foot through the bottom of the rep, flagged across frames 96 to 118.",
  "landing.step.attribution.tag": "Knowledge graph",
  "landing.step.attribution.title": "Attribution",
  "landing.step.attribution.body":
    "Multi-hop retrieval links medial knee travel to weak hip abductors, with glute medius as the primary node.",
  "landing.step.prescription.tag": "Reasoning",
  "landing.step.prescription.title": "Prescription",
  "landing.step.prescription.body":
    "Cue the lifter to drive the knees out over the toes, and program banded goblet squats as the accessory.",
  "landing.frame.alt": "Sampled video frame under analysis",

  // Landing — bento
  "landing.bento.kicker": "Under the hood",
  "landing.bento.title": "Four signals, read locally and fused.",
  "landing.bento.pose.title": "Pose perception",
  "landing.bento.pose.body":
    "MediaPipe and RTMPose landmarks on 33 keypoints. Joint geometry like knee valgus maps straight into language the graph understands.",
  "landing.bento.kg.title": "Knowledge graph",
  "landing.bento.kg.body":
    "A fitness knowledge graph links fault to cause to fix, with multi-hop retrieval over sourced biomechanics.",
  "landing.bento.rules.title": "Interpretable rules",
  "landing.bento.videomae.title": "VideoMAE motion",
  "landing.bento.videomae.body":
    "Spatio-temporal features tell a clean rep from a subtle error.",

  // Landing — evaluation
  "landing.eval.title": "Held to a measurable bar.",
  "landing.eval.sub":
    "Explainability only counts if it is checkable. x-coach is validated on agreement, grounding, and whether lifters actually act on it.",
  "landing.eval.m1.label": "Score agreement",
  "landing.eval.m1.body":
    "Spearman correlation against expert ranking, so the model's ordering tracks a human judge.",
  "landing.eval.m2.label": "Grounding and hallucination",
  "landing.eval.m2.body":
    "RAGAS faithfulness checks that every claim stays anchored to the retrieved evidence.",
  "landing.eval.m3.label": "Usefulness",
  "landing.eval.m3.body":
    "A user study with lifters across experience levels rates the skeleton overlay and the written advice.",

  // Landing — closing CTA
  "landing.cta.title": "See it analyze a real squat.",
  "landing.cta.sub":
    "Upload a clip or open a labeled sample, and watch the skeleton, the faults, and the grounded feedback come back together.",

  // Landing — footer
  "landing.footer.pipeline": "Pipeline",
  "landing.footer.tagline": "Explainable squat coaching, research prototype.",

  // Landing — movement showcase
  "landing.showcase.title": "One pipeline, the whole movement library.",
  "landing.showcase.sub":
    "The same perceive, retrieve, reason loop that reads a squat reads the rest of the library, on real footage.",
  "landing.showcase.analyzing": "Analyzing",
  "landing.showcase.squat.name": "Squat",
  "landing.showcase.squat.note": "Depth, knee tracking, and torso angle at the bottom of the rep.",
  "landing.showcase.pushups.name": "Push-ups",
  "landing.showcase.pushups.note": "Elbow path and hip line, tracked through every press.",
  "landing.showcase.highknee.name": "High Knees",
  "landing.showcase.highknee.note": "Knee-drive height and left-right landing symmetry.",
  "landing.showcase.situps.name": "Sit-ups",
  "landing.showcase.situps.note": "Trunk flexion and neck compensation, rep by rep.",

  // Auth — login + session
  "auth.checking": "Checking your session…",
  "auth.back": "Back to home",
  "auth.signInTitle": "Sign in",
  "auth.signInSub": "Welcome back. Pick up your saved analyses.",
  "auth.signUpTitle": "Create your account",
  "auth.signUpSub": "Save every analysis to your history.",
  "auth.email": "Email",
  "auth.password": "Password",
  "auth.signInBtn": "Sign in",
  "auth.signUpBtn": "Create account",
  "auth.or": "or",
  "auth.google": "Continue with Google",
  "auth.noAccount": "New here?",
  "auth.toSignup": "Create an account",
  "auth.haveAccount": "Already have an account?",
  "auth.toSignin": "Sign in",
  "auth.demoLink": "Continue without an account",
  "auth.errorTitle": "Couldn't sign you in",
  "auth.confirmEmail": "Check your inbox to confirm your email, then sign in.",
  "auth.notConfigured": "Sign-in isn't set up on this server yet. You can still use the demo.",
  "auth.brandHeadline": "Coaching you can revisit.",
  "auth.brandSub":
    "Sign in to keep every squat analysis, with its skeleton, faults, and grounded feedback.",
  "auth.point1": "Every analysis saved to your history",
  "auth.point2": "Reopen any past rep, exactly as analyzed",
  "auth.point3": "Private to you, enforced at the database",

  // Account / nav
  "nav.history": "My records",
  "account.signin": "Sign in",
  "account.signout": "Sign out",
  "account.menu": "Account menu",
  "account.settings": "Settings",

  // Settings page
  "settings.title": "Settings",
  "settings.subtitle": "Manage your account.",
  "settings.backToStudio": "Back to studio",
  "settings.profile": "Profile",
  "settings.name": "Name",
  "settings.email": "Email",
  "settings.provider": "Signed in with",
  "settings.provider.google": "Google",
  "settings.provider.email": "Email & password",
  "settings.model": "Coach model",
  "settings.modelDesc": "Choose which LLM answers your follow-up questions. Applies to new messages.",
  "settings.modelDefault": "Default",
  "settings.modelLoading": "Loading models…",
  "settings.danger": "Danger zone",
  "settings.clearTitle": "Clear saved analyses",
  "settings.clearDesc": "Permanently delete all of your saved analyses. This cannot be undone.",
  "settings.clearCta": "Clear all",
  "settings.clearConfirm": "Yes, delete everything",
  "settings.clearCancel": "Cancel",
  "settings.clearing": "Clearing…",
  "settings.clearedNone": "You had no saved analyses to clear.",
  "settings.clearedOne": "Deleted 1 saved analysis.",
  "settings.clearedMany": "Deleted {count} saved analyses.",
  "settings.clearError": "Couldn't clear your analyses. Please try again.",
  "settings.deleteAccount": "Delete account",
  "settings.deleteAccountDesc":
    "Removing the login itself isn't available here yet — contact support to delete your account.",

  // History page (我的紀錄)
  "history.title": "My records",
  "history.subtitle": "Saved analyses for {email}.",
  "history.subtitleAnon": "Your saved analyses.",
  "history.newAnalysis": "New analysis",
  "history.empty": "No saved analyses yet.",
  "history.emptyHint": "Analyze a squat while signed in and it lands here.",
  "history.startCta": "Analyze a squat",
  "history.errorTitle": "Couldn't load your history",
  "history.retry": "Retry",
  "history.rowTitle": "{view} squat",
  "history.clean": "clean rep",
  "history.faultOne": "1 fault",
  "history.faultMany": "{count} faults",

  // Pose Match game
  "nav.play": "Pose game",
  "game.title": "Pose Match Rush",
  "game.badge": "Live · MediaPipe",
  "game.heading": "Strike the pose. Beat the clock.",
  "game.sub":
    "Your camera reads your body in real time with the same pose engine that powers X-Coach's squat analysis. Match each target shape to score — the closer your joint angles, the bigger the points.",
  "game.how1": "Stand back so your whole body is in frame.",
  "game.how2": "Copy the target pose shown at the top and hold it steady.",
  "game.how3": "Chain poses without missing to build a combo multiplier.",
  "game.startBtn": "Enable camera & play",
  "game.starting": "Starting camera…",
  "game.cameraNote": "{s}-second round · runs on-device, nothing is uploaded.",
  "game.error": "Couldn't start the camera. Grant camera access and try again.",
  "game.posesTitle": "Poses in the deck",
  "game.pose.tPose": "T-Pose",
  "game.pose.cactus": "Cactus",
  "game.pose.cheer": "Cheer",
  "game.pose.flex": "Flex",
  "game.pose.stand": "Stand tall",
  "game.pose.squat": "Squat hold",
  "game.hud.score": "Score",
  "game.hud.strike": "Strike this",
  "game.hud.time": "Time",
  "game.hud.combo": "{n}× combo",
  "game.hud.holding": "Hold it…",
  "game.hud.matchPrompt": "Match the pose",
  "game.grade.perfect": "Perfect!",
  "game.grade.great": "Great!",
  "game.grade.good": "Good",
  "game.grade.miss": "Miss",
  "game.board.title": "Local leaderboard",
  "game.board.empty": "No scores yet — be the first!",
  "game.board.you": "You",
  "game.over.title": "Round over",
  "game.over.poses": "poses",
  "game.over.combo": "best combo",
  "game.over.nameLabel": "Add your name to the board",
  "game.over.namePlaceholder": "Your name",
  "game.over.save": "Save",
  "game.over.anon": "Anonymous",
  "game.over.ranked": "You're #{rank} on the local board!",
  "game.over.notRanked": "Saved — keep practising to crack the top 10.",
  "game.over.replay": "Play again",
};

const zhHant: Dict = {
  // Sidebar
  "nav.newAnalysis": "新增分析",
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

  // Chat input — disabled fallback (auth/LLM not configured) + the working grounded chat.
  "chat.placeholder": "詢問 AI 教練…（LLM 功能即將推出）",
  "chat.title": "對話式教練功能將隨 LLM 層推出。",
  "chat.heading": "AI 教練",
  "chat.grounded": "根據你的分析",
  "chat.groundedShort": "有依據",
  "chat.intro": "針對你的深蹲追問後續問題，回答將僅根據偵測到的錯誤與檢索到的提示。",
  "chat.suggestFix": "我該先修正什麼？",
  "chat.suggestDrill": "給我一個矯正動作",
  "chat.suggestWhy": "這為什麼重要？",
  "chat.placeholderActive": "追問後續問題…",
  "chat.send": "傳送訊息",
  "chat.thinking": "教練思考中…",
  "chat.signIn": "登入即可就本次分析與 AI 教練對話。",
  "chat.error": "無法連線至教練，請再試一次。",
  "chat.sessionExpired": "登入階段已過期，請重新登入以繼續對話。",
  "chat.you": "你",
  "chat.coach": "教練",
  "coach.followUp": "後續追問",

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
  "feedback.cause": "原因",
  "feedback.risk": "風險",
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
  "kg.chain": "錯誤 → 成因 → 修正",
  "kg.nodes": "{count} 個節點",
  "kg.empty": "此片段沒有圖譜脈絡。",
  "kg.focus": "顯示",
  "kg.expand": "全螢幕檢視",
  "kg.collapse": "關閉全螢幕",
  "kg.cause": "成因",
  "kg.risk": "風險",
  "kg.correction": "修正",
  "kg.evidence": "證據",

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

  // Landing — nav
  "landing.nav.how": "運作方式",
  "landing.nav.pipeline": "處理流程",
  "landing.nav.eval": "評估",
  "landing.cta.open": "開啟示範",

  // Landing — hero
  "landing.hero.titlePre": "每個訓練提示，都能",
  "landing.hero.titleAccent": "追溯到關節",
  "landing.hero.titlePost": "。",
  "landing.hero.sub":
    "x-coach 讀取深蹲影片，定位動作錯誤，在生物力學知識圖譜中追溯成因，並說明修正方式。",
  "landing.hero.readMethod": "了解方法",

  // Landing — problem
  "landing.problem.title": "分數不會教學，通用模型只能猜測。",
  "landing.problem.sub":
    "動作品質模型只回傳一個分數，沒有任何指導。問通用語言模型，它聽起來很有把握，卻在杜撰生物力學。",
  "landing.problem.aqs.label": "動作品質評分",
  "landing.problem.aqs.body":
    "回傳 0 到 100 的評分。運動員只知道自己得了 71 分，卻不知道該改什麼、為什麼改。",
  "landing.problem.llm.label": "通用語言模型",
  "landing.problem.llm.body":
    "產生與影片脫節的流暢建議，並憑空捏造影片中根本沒有出現的成因。",
  "landing.problem.xcoach.title": "從設計上就有依據",
  "landing.problem.point1": "在實際影格中看見錯誤",
  "landing.problem.point2": "從具來源的知識圖譜檢索成因",
  "landing.problem.point3": "說明可回溯依據的修正方式",

  // Landing — pipeline
  "landing.pipeline.kicker": "系統架構",
  "landing.pipeline.title": "四個模組，構成從畫面到處方的完整閉環。",
  "landing.stage.perceive.title": "感知",
  "landing.stage.perceive.body":
    "姿態關鍵點與 VideoMAE 動作特徵擷取幾何資訊，再於時間軸上定位錯誤。",
  "landing.stage.retrieve.title": "檢索",
  "landing.stage.retrieve.body":
    "GraphRAG 在健身知識圖譜中，從可見徵狀走向更深層的成因。",
  "landing.stage.reason.title": "推理",
  "landing.stage.reason.body":
    "一條思考鏈從觀察、歸因到處方，全程以檢索到的證據為依據。",
  "landing.stage.coach.title": "指導",
  "landing.stage.coach.body":
    "回傳診斷報告與修正提示，並標出確切的影格。",

  // Landing — diagnosis
  "landing.diagnosis.title": "每個提示都帶著推理依據。",
  "landing.diagnosis.sub":
    "一個偵測到的錯誤，從鏡頭所見一路走到你該採取的訓練動作。",
  "landing.step.observation.tag": "感知",
  "landing.step.observation.title": "觀察",
  "landing.step.observation.body":
    "在動作最低點，左膝向內越過腳掌，於第 96 到 118 影格被標記。",
  "landing.step.attribution.tag": "知識圖譜",
  "landing.step.attribution.title": "歸因",
  "landing.step.attribution.body":
    "多跳檢索將膝蓋內移連結到髖外展肌無力，並以臀中肌為主要節點。",
  "landing.step.prescription.tag": "推理",
  "landing.step.prescription.title": "處方",
  "landing.step.prescription.body":
    "提示運動員將膝蓋向腳尖外推，並安排彈力帶高腳杯深蹲作為輔助訓練。",
  "landing.frame.alt": "分析中的取樣影格",

  // Landing — bento
  "landing.bento.kicker": "技術細節",
  "landing.bento.title": "四種訊號，於本地讀取並融合。",
  "landing.bento.pose.title": "姿態感知",
  "landing.bento.pose.body":
    "MediaPipe 與 RTMPose 在 33 個關鍵點上的標記。像膝外翻這類關節幾何，可直接對應成知識圖譜理解的語言。",
  "landing.bento.kg.title": "知識圖譜",
  "landing.bento.kg.body":
    "健身知識圖譜將錯誤、成因與修正串連起來，並在具來源的生物力學上進行多跳檢索。",
  "landing.bento.rules.title": "可解釋規則",
  "landing.bento.videomae.title": "VideoMAE 動作",
  "landing.bento.videomae.body":
    "時空特徵能分辨標準動作與細微錯誤。",

  // Landing — evaluation
  "landing.eval.title": "以可衡量的標準檢驗。",
  "landing.eval.sub":
    "可解釋性唯有能被驗證才有意義。x-coach 在一致性、依據紮實度，以及運動員是否真的採用建議上接受驗證。",
  "landing.eval.m1.label": "評分一致性",
  "landing.eval.m1.body":
    "與專家排名的 Spearman 相關性，讓模型的排序貼近人類評審。",
  "landing.eval.m2.label": "依據與幻覺",
  "landing.eval.m2.body":
    "以 RAGAS 忠實度檢查，確保每項主張都緊扣檢索到的證據。",
  "landing.eval.m3.label": "實用性",
  "landing.eval.m3.body":
    "由不同經驗程度的運動員參與使用者研究，評比骨架疊圖與文字建議。",

  // Landing — closing CTA
  "landing.cta.title": "看它分析一段真實深蹲。",
  "landing.cta.sub":
    "上傳影片或開啟已標註的範例，看著骨架、錯誤與有據回饋一同呈現。",

  // Landing — footer
  "landing.footer.pipeline": "處理流程",
  "landing.footer.tagline": "可解釋的深蹲教練，研究原型。",

  // Landing — movement showcase
  "landing.showcase.title": "同一套流程，涵蓋整個動作庫。",
  "landing.showcase.sub":
    "讀懂深蹲的「感知、檢索、推理」流程，同樣能分析動作庫中的其他項目，全部基於真實影片。",
  "landing.showcase.analyzing": "分析中",
  "landing.showcase.squat.name": "深蹲",
  "landing.showcase.squat.note": "檢視最低點的深度、膝蓋軌跡與軀幹角度。",
  "landing.showcase.pushups.name": "伏地挺身",
  "landing.showcase.pushups.note": "逐次追蹤手肘軌跡與髖部直線。",
  "landing.showcase.highknee.name": "高抬腿",
  "landing.showcase.highknee.note": "抬膝高度與左右落地的對稱性。",
  "landing.showcase.situps.name": "仰臥起坐",
  "landing.showcase.situps.note": "逐下檢視軀幹屈曲與頸部代償。",

  // Auth — login + session
  "auth.checking": "確認登入狀態中…",
  "auth.back": "返回首頁",
  "auth.signInTitle": "登入",
  "auth.signInSub": "歡迎回來，繼續查看你儲存的分析。",
  "auth.signUpTitle": "建立帳號",
  "auth.signUpSub": "把每一次分析都存進你的紀錄。",
  "auth.email": "電子郵件",
  "auth.password": "密碼",
  "auth.signInBtn": "登入",
  "auth.signUpBtn": "建立帳號",
  "auth.or": "或",
  "auth.google": "使用 Google 繼續",
  "auth.noAccount": "還沒有帳號？",
  "auth.toSignup": "建立帳號",
  "auth.haveAccount": "已經有帳號了？",
  "auth.toSignin": "登入",
  "auth.demoLink": "不登入，直接試用",
  "auth.errorTitle": "無法登入",
  "auth.confirmEmail": "請至信箱點擊確認連結，再回來登入。",
  "auth.notConfigured": "此伺服器尚未設定登入功能，你仍可使用示範。",
  "auth.brandHeadline": "可以回顧的教練回饋。",
  "auth.brandSub": "登入後，每一段深蹲分析（骨架、錯誤與有據回饋）都會為你保留。",
  "auth.point1": "每次分析都存進你的紀錄",
  "auth.point2": "隨時重開任何一次動作，完整重現",
  "auth.point3": "僅你可見，由資料庫層級保護",

  // Account / nav
  "nav.history": "我的紀錄",
  "account.signin": "登入",
  "account.signout": "登出",
  "account.menu": "帳號選單",
  "account.settings": "設定",

  // Settings page
  "settings.title": "設定",
  "settings.subtitle": "管理你的帳號。",
  "settings.backToStudio": "返回工作室",
  "settings.profile": "個人資料",
  "settings.name": "名稱",
  "settings.email": "電子郵件",
  "settings.provider": "登入方式",
  "settings.provider.google": "Google",
  "settings.provider.email": "電子郵件與密碼",
  "settings.model": "教練模型",
  "settings.modelDesc": "選擇回答你追問的 LLM 模型，套用於之後的新訊息。",
  "settings.modelDefault": "預設",
  "settings.modelLoading": "載入模型中…",
  "settings.danger": "危險區域",
  "settings.clearTitle": "清除已存分析",
  "settings.clearDesc": "永久刪除你所有已存的分析，此操作無法復原。",
  "settings.clearCta": "全部清除",
  "settings.clearConfirm": "是的，全部刪除",
  "settings.clearCancel": "取消",
  "settings.clearing": "清除中…",
  "settings.clearedNone": "沒有可清除的已存分析。",
  "settings.clearedOne": "已刪除 1 筆已存分析。",
  "settings.clearedMany": "已刪除 {count} 筆已存分析。",
  "settings.clearError": "無法清除你的分析，請再試一次。",
  "settings.deleteAccount": "刪除帳號",
  "settings.deleteAccountDesc": "目前無法在此移除登入帳號本身，如需刪除帳號請聯絡客服。",

  // History page (我的紀錄)
  "history.title": "我的紀錄",
  "history.subtitle": "{email} 的已存分析。",
  "history.subtitleAnon": "你的已存分析。",
  "history.newAnalysis": "新分析",
  "history.empty": "還沒有已存的分析。",
  "history.emptyHint": "登入狀態下分析一段深蹲，就會出現在這裡。",
  "history.startCta": "分析深蹲",
  "history.errorTitle": "無法載入你的紀錄",
  "history.retry": "重試",
  "history.rowTitle": "{view}深蹲",
  "history.clean": "標準動作",
  "history.faultOne": "1 個錯誤",
  "history.faultMany": "{count} 個錯誤",

  // 體感對招遊戲
  "nav.play": "體感遊戲",
  "game.title": "體感對招",
  "game.badge": "即時 · MediaPipe",
  "game.heading": "擺出動作，跟時間賽跑。",
  "game.sub":
    "鏡頭即時讀取你的身體，使用與 X-Coach 深蹲分析相同的姿態引擎。模仿每個目標動作即可得分——關節角度越接近，分數越高。",
  "game.how1": "往後站，讓全身入鏡。",
  "game.how2": "模仿上方顯示的目標動作並穩定維持。",
  "game.how3": "連續命中不失誤，累積連擊倍率。",
  "game.startBtn": "開啟鏡頭並開始",
  "game.starting": "啟動鏡頭中…",
  "game.cameraNote": "{s} 秒一回合 · 全程在裝置端運算，不會上傳。",
  "game.error": "無法啟動鏡頭，請允許鏡頭權限後再試一次。",
  "game.posesTitle": "動作牌組",
  "game.pose.tPose": "大字",
  "game.pose.cactus": "仙人掌",
  "game.pose.cheer": "歡呼",
  "game.pose.flex": "健美",
  "game.pose.stand": "立正",
  "game.pose.squat": "深蹲維持",
  "game.hud.score": "分數",
  "game.hud.strike": "擺出這個",
  "game.hud.time": "時間",
  "game.hud.combo": "{n} 連擊",
  "game.hud.holding": "維持住…",
  "game.hud.matchPrompt": "模仿動作",
  "game.grade.perfect": "完美！",
  "game.grade.great": "很棒！",
  "game.grade.good": "不錯",
  "game.grade.miss": "失誤",
  "game.board.title": "本地排行榜",
  "game.board.empty": "還沒有紀錄——來當第一名吧！",
  "game.board.you": "你",
  "game.over.title": "回合結束",
  "game.over.poses": "個動作",
  "game.over.combo": "最高連擊",
  "game.over.nameLabel": "把名字加入排行榜",
  "game.over.namePlaceholder": "你的名字",
  "game.over.save": "儲存",
  "game.over.anon": "匿名玩家",
  "game.over.ranked": "你在本地排行榜排名第 {rank}！",
  "game.over.notRanked": "已儲存——繼續練習擠進前十名。",
  "game.over.replay": "再玩一次",
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
