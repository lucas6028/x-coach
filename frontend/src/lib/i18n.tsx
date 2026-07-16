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
  "nav.games": "Games",
  "nav.hide": "Hide navigation",
  "nav.show": "Show navigation",
  "sidebar.version": "Prototype v0.1",
  "sidebar.tagline": "Pose · Rules · GraphRAG",

  // Games hub — the App-Store-style catalog of the pose mini-games
  "games.title": "Games",
  "games.badge": "Move to play",
  "games.heading": "Pose Arcade",
  "games.sub":
    "Camera mini-games powered by the same MediaPipe pose engine behind x-coach. Get up and move — we'll estimate the calories you burn.",
  "games.totalTitle": "Calories burned",
  "games.kcalUnit": "kcal",
  "games.totalSub": "Estimated across {n} rounds",
  "games.totalSubOne": "Estimated across 1 round",
  "games.totalEmpty": "Play a round to start burning.",
  "games.play": "Play",
  "games.stat.bestScore": "best",
  "games.stat.bestCount": "best 67s",
  "games.ninja.desc": "Your hands are the blades — slice the flying fruit, dodge the bombs.",
  "games.six.desc": "Do the 6-7 bob: raise one hand then the other and rack up as many as you can.",
  "games.blast.desc": "Charge a Kamehameha with your palms, then fling your arms apart to blast the meme orbs.",

  // Per-round calorie estimate (shown on each game's over screen)
  "game.kcal.est": "≈ {n} kcal",
  "game.kcal.note": "estimated",

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
  "chat.placeholder": "Ask Lumen… (LLM layer coming soon)",
  "chat.title": "Conversational coaching arrives with the LLM layer.",
  "chat.heading": "Lumen",
  "chat.grounded": "grounded in your analysis",
  "chat.groundedShort": "grounded",
  "chat.intro": "Ask a follow-up about your squat. Answers stay grounded in the detected faults and retrieved cues.",
  "chat.suggestFix": "What should I fix first?",
  "chat.suggestDrill": "Show me a drill for this",
  "chat.suggestWhy": "Why does this matter?",
  "chat.placeholderActive": "Ask a follow-up…",
  "chat.send": "Send message",
  "chat.thinking": "Lumen is thinking…",
  "chat.signIn": "Sign in to chat with Lumen about this analysis.",
  "chat.error": "Couldn't reach Lumen. Please try again.",
  "chat.sessionExpired": "Your session expired. Please sign in again to keep chatting.",
  "chat.you": "You",
  "chat.coach": "Lumen",
  "coach.followUp": "Follow-up",
  "loader.aria": "Lumen is working",
  "loader.step1": "Reading pose",
  "loader.step2": "Checking mechanics",
  "loader.step3": "Lighting the why",

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

  // Meme Blaster game
  "nav.blast": "Meme Blaster",
  "blast.title": "Meme Blaster",
  "blast.badge": "Live · MediaPipe",
  "blast.heading": "Charge up. Blast the memes.",
  "blast.sub":
    "The classic \"charging energy\" meme, made real. Your camera tracks your hands with the same pose engine behind X-Coach — bring your palms together to charge a Kamehameha, then throw your arms apart to fire an energy beam and vaporise the meme orbs.",
  "blast.how1": "Stand back so your hands and shoulders are in frame.",
  "blast.how2": "Bring both hands together to charge — the meter fills up.",
  "blast.how3": "Fling your arms apart to fire. Aim with how high you hold your hands.",
  "blast.startBtn": "Enable camera & play",
  "blast.starting": "Starting camera…",
  "blast.cameraNote": "{s}-second round · runs on-device, nothing is uploaded.",
  "blast.error": "Couldn't start the camera. Grant camera access and try again.",
  "blast.orbsTitle": "Meme orbs incoming",
  "blast.hud.score": "Score",
  "blast.hud.time": "Time",
  "blast.hud.combo": "{n}× combo",
  "blast.hud.charge": "Hands together — charge!",
  "blast.hud.fire": "Charged! Fling arms apart to FIRE",
  "blast.hud.whiff": "Whiff!",
  "blast.board.title": "Local leaderboard",
  "blast.board.empty": "No scores yet — be the first!",
  "blast.board.you": "You",
  "blast.over.title": "Round over",
  "blast.over.hits": "orbs blasted",
  "blast.over.combo": "best combo",
  "blast.over.nameLabel": "Add your name to the board",
  "blast.over.namePlaceholder": "Your name",
  "blast.over.save": "Save",
  "blast.over.anon": "Anonymous",
  "blast.over.ranked": "You're #{rank} on the local board!",
  "blast.over.notRanked": "Saved — keep blasting to crack the top 10.",
  "blast.over.replay": "Play again",

  // Admin panel (P1 shell — role-gated; sections arrive in later phases)
  "admin.nav": "Admin",
  "admin.title": "Admin",
  "admin.subtitle": "Manage users, LLM settings, and pipeline parameters. More sections coming soon.",
  "admin.denied": "You don't have access to the admin panel.",
  "admin.loading": "Checking your access…",
  "admin.error": "Couldn't verify your admin access. Please try again.",
  "admin.retry": "Retry",

  // Admin console standalone shell (nav + section titles)
  "admin.console.title": "Admin console",
  "admin.nav.overview": "Overview",
  "admin.nav.users": "Users",
  "admin.nav.settingsLlm": "LLM chat",
  "admin.nav.settingsRag": "RAG / KG",
  "admin.nav.settingsAnalyze": "Analyze pipeline",
  "admin.nav.backToApp": "Back to app",

  // Dedicated admin console sign-in (email + password only; no OAuth)
  "adminLogin.title": "Admin console",
  "adminLogin.subtitle": "Sign in with your admin account to continue.",
  "adminLogin.submit": "Sign in",
  "adminLogin.submitting": "Signing in…",
  "adminLogin.error": "Couldn't sign you in. Check your email and password.",
  "adminLogin.switchAccount": "Sign in as admin",
  "adminLogin.notConfigured": "Sign-in isn't set up on this server yet.",

  // Admin panel P2 — runtime settings
  "admin.settings.loading": "Loading settings…",
  "admin.settings.loadError": "Couldn't load the current settings.",
  "admin.settings.save": "Save changes",
  "admin.settings.saving": "Saving…",
  "admin.settings.saved": "Settings saved.",
  "admin.settings.saveError": "Couldn't save the settings.",
  "admin.settings.defaultLabel": "Default: {value}",
  "admin.settings.llm": "LLM chat settings",
  "admin.settings.llmDesc": "Tune the conversational-coaching model list and request behaviour.",
  "admin.settings.models": "Selectable models",
  "admin.settings.modelsHint": "One id per line or comma-separated. The first is the default.",
  "admin.settings.followupModel": "Follow-up model",
  "admin.settings.baseUrl": "Provider base URL",
  "admin.settings.temperature": "Temperature",
  "admin.settings.temperatureHint": "Leave blank to omit (0–2).",
  "admin.settings.chatTimeout": "Answer timeout (s)",
  "admin.settings.followupTimeout": "Follow-up timeout (s)",
  "admin.settings.ragkg": "RAG / KG",
  "admin.settings.ragkgDesc": "Default retrieval breadth (endpoint query params still override).",
  "admin.settings.ragTopK": "RAG top-k",
  "admin.settings.kgHops": "KG hops",
  "admin.settings.kgSeeds": "KG seed nodes",
  "admin.settings.analyze": "Analysis pipeline",
  "admin.settings.analyzeDesc": "Upload constraints and concurrency.",
  "admin.settings.uploadFormats": "Allowed upload formats",
  "admin.settings.uploadFormatsHint": "Comma-separated .ext values.",
  "admin.settings.maxConcurrent": "Max concurrent analyses",
  "admin.settings.restartRequired": "Restart required to take effect",
  "admin.settings.maxConcurrentReadonly": "Read-only: set via the XCOACH_MAX_CONCURRENT_ANALYSES environment variable; requires a restart to change.",
  "admin.settings.invalidNumber": "Please enter a valid number for every field.",

  // Admin panel P3 — system overview dashboard
  "admin.overview.title": "System overview",
  "admin.overview.desc": "Live health of the deployment and how much it's being used.",
  "admin.overview.loadError": "Couldn't load the system overview.",
  "admin.overview.auth": "Authentication",
  "admin.overview.chat": "LLM chat",
  "admin.overview.configured": "Configured",
  "admin.overview.notConfigured": "Not configured",
  "admin.overview.stores": "Data stores",
  "admin.overview.storesReady": "{ready}/{total} ready",
  "admin.overview.totalUsers": "Total users",
  "admin.overview.totalAnalyses": "Total analyses",

  // Admin panel P3 — user oversight
  "admin.users.title": "Users",
  "admin.users.desc": "Read-only activity overview. Assign or revoke admin access per user.",
  "admin.users.loading": "Loading users…",
  "admin.users.loadError": "Couldn't load the users list.",
  "admin.users.empty": "No users yet.",
  "admin.users.email": "Email",
  "admin.users.created": "Registered",
  "admin.users.lastSignIn": "Last sign-in",
  "admin.users.analyses": "Analyses",
  "admin.users.conversations": "Conversations",
  "admin.users.role": "Admin",
  "admin.users.makeAdmin": "Make admin",
  "admin.users.revokeAdmin": "Revoke admin",
  "admin.users.you": "You",
  "admin.users.never": "Never",
  "admin.users.updateError": "Couldn't update this user's role.",

  // 67 (mini-game) — the "six seven" meme gesture counter
  "nav.six": "67",
  "six.title": "67",
  "six.badge": "Brainrot · one camera",
  "six.heading": "How many 67s can you hit?",
  "six.sub":
    "The 6-7 bob, counted. Hold both hands out and bounce them up and down, one at a time — six… seven… — and rack up as many as you can. Same MediaPipe pose tracking x-coach uses, pointed at the meme.",
  "six.how1": "Stand back so both hands and shoulders are in frame.",
  "six.how2": "Raise one hand, then the other — every switch is one 67.",
  "six.how3": "Keep the rhythm going for a combo. {s} seconds on the clock.",
  "six.leftHand": "left hand up",
  "six.rightHand": "right hand up",
  "six.startBtn": "Enable camera & go",
  "six.starting": "Starting camera…",
  "six.cameraNote": "Runs on-device. {s}-second sprint.",
  "six.error": "Couldn't start the camera.",

  // In-round HUD
  "six.hud.time": "Time",
  "six.hud.label": "sixty-sevens",
  "six.hud.combo": "{n}× rhythm",

  // Over screen
  "six.over.title": "Time! You hit",
  "six.over.combo": "Best rhythm streak: {n}",
  "six.over.nameLabel": "Add your name to the board",
  "six.over.namePlaceholder": "Your name",
  "six.over.save": "Save",
  "six.over.anon": "Anonymous",
  "six.over.ranked": "You're #{rank} on the local board!",
  "six.over.notRanked": "Saved — keep bobbing to crack the top 10.",
  "six.over.replay": "Go again",

  // Leaderboard
  "six.board.title": "Top 67s",
  "six.board.empty": "No scores yet — be the first!",
  "six.board.you": "You",
  "six.board.count": "{n} × 67",

  // Fruit Ninja (mini-game)
  "nav.ninja": "Fruit Ninja",
  "ninja.title": "Fruit Ninja",
  "ninja.badge": "Slice · MediaPipe",
  "ninja.heading": "Your hands are the blades.",
  "ninja.sub":
    "Fruit flies up, you slice it with your bare hands. MediaPipe tracks both wrists in real time — the same pose engine behind x-coach's squat analysis — and turns every swipe into a blade.",
  "ninja.how1": "Stand back so your hands and shoulders are in frame.",
  "ninja.how2": "Swipe a hand through the flying fruit to slice it. Fast swipes cut.",
  "ninja.how3": "Dodge the 💣, and don't drop {lives} fruits — chain cuts for a combo.",
  "ninja.startBtn": "Enable camera & slice",
  "ninja.starting": "Starting camera…",
  "ninja.cameraNote": "Runs on-device. Two-handed play works great.",
  "ninja.deckTitle": "On the board",
  "ninja.error": "Couldn't start the camera.",

  // In-round HUD
  "ninja.hud.score": "Score",
  "ninja.hud.lives": "Lives",
  "ninja.hud.combo": "{n} combo!",
  "ninja.hud.boom": "💥 BOOM",
  "ninja.hud.slice": "Swipe to slice",

  // Over screen
  "ninja.over.title": "Round over",
  "ninja.over.bombed": "You hit a bomb!",
  "ninja.over.combo": "Best combo: {n}",
  "ninja.over.nameLabel": "Add your name to the board",
  "ninja.over.namePlaceholder": "Your name",
  "ninja.over.save": "Save",
  "ninja.over.anon": "Anonymous",
  "ninja.over.ranked": "You're #{rank} on the local board!",
  "ninja.over.notRanked": "Saved — keep slicing to crack the top 10.",
  "ninja.over.replay": "Play again",

  // Leaderboard
  "ninja.board.title": "Top slicers",
  "ninja.board.empty": "No scores yet — be the first!",
  "ninja.board.you": "You",
};

const zhHant: Dict = {
  // Sidebar
  "nav.newAnalysis": "新增分析",
  "nav.analyse": "分析",
  "nav.library": "資料庫",
  "nav.games": "小遊戲",
  "nav.hide": "隱藏導覽列",
  "nav.show": "顯示導覽列",
  "sidebar.version": "原型 v0.1",
  "sidebar.tagline": "姿態 · 規則 · GraphRAG",

  // 小遊戲中心 — App Store 風格的姿態小遊戲總覽
  "games.title": "小遊戲",
  "games.badge": "動起來就能玩",
  "games.heading": "姿態遊戲場",
  "games.sub":
    "用與 x-coach 同一套 MediaPipe 姿態引擎打造的鏡頭小遊戲。站起來動一動——我們會估算你消耗的卡路里。",
  "games.totalTitle": "累計消耗",
  "games.kcalUnit": "大卡",
  "games.totalSub": "估計自 {n} 場遊戲",
  "games.totalSubOne": "估計自 1 場遊戲",
  "games.totalEmpty": "玩一場開始燃燒吧。",
  "games.play": "開始遊玩",
  "games.stat.bestScore": "最高分",
  "games.stat.bestCount": "最多 67",
  "games.ninja.desc": "你的雙手就是刀——切開飛來的水果，閃避炸彈。",
  "games.six.desc": "做「6-7」抖手：一手上一手下輪流擺動，盡量累積次數。",
  "games.blast.desc": "雙手合十蓄氣功，再張開雙臂發射能量波轟爆迷因光球。",

  // 每回合卡路里估計（顯示於各遊戲的結算畫面）
  "game.kcal.est": "≈ {n} 大卡",
  "game.kcal.note": "估計值",

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
  "chat.placeholder": "詢問 Lumen…（LLM 功能即將推出）",
  "chat.title": "對話式教練功能將隨 LLM 層推出。",
  "chat.heading": "Lumen",
  "chat.grounded": "根據你的分析",
  "chat.groundedShort": "有依據",
  "chat.intro": "針對你的深蹲追問後續問題，回答將僅根據偵測到的錯誤與檢索到的提示。",
  "chat.suggestFix": "我該先修正什麼？",
  "chat.suggestDrill": "給我一個矯正動作",
  "chat.suggestWhy": "這為什麼重要？",
  "chat.placeholderActive": "追問後續問題…",
  "chat.send": "傳送訊息",
  "chat.thinking": "Lumen 思考中…",
  "chat.signIn": "登入即可就本次分析與 Lumen 對話。",
  "chat.error": "無法連線至 Lumen，請再試一次。",
  "chat.sessionExpired": "登入階段已過期，請重新登入以繼續對話。",
  "chat.you": "你",
  "chat.coach": "Lumen",
  "coach.followUp": "後續追問",
  "loader.aria": "Lumen 分析中",
  "loader.step1": "讀取姿勢",
  "loader.step2": "對照力學",
  "loader.step3": "照亮原因",

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

  // 迷因發射器遊戲
  "nav.blast": "迷因發射器",
  "blast.title": "迷因發射器",
  "blast.badge": "即時 · MediaPipe",
  "blast.heading": "蓄力，轟爆迷因！",
  "blast.sub":
    "把經典的「蓄力氣功」迷因照片變成真的！鏡頭用與 X-Coach 相同的姿態引擎追蹤你的雙手——把手掌合在一起蓄力發龜派氣功，再張開雙臂發射能量波，把飄過來的迷因光球轟爆。",
  "blast.how1": "往後站，讓雙手和肩膀都入鏡。",
  "blast.how2": "把雙手合在一起蓄力——能量條會逐漸充滿。",
  "blast.how3": "猛然張開雙臂發射！用手舉的高度來瞄準。",
  "blast.startBtn": "開啟鏡頭並開始",
  "blast.starting": "啟動鏡頭中…",
  "blast.cameraNote": "{s} 秒一回合 · 全程在裝置端運算，不會上傳。",
  "blast.error": "無法啟動鏡頭，請允許鏡頭權限後再試一次。",
  "blast.orbsTitle": "迷因光球來襲",
  "blast.hud.score": "分數",
  "blast.hud.time": "時間",
  "blast.hud.combo": "{n} 連擊",
  "blast.hud.charge": "雙手合十——蓄力！",
  "blast.hud.fire": "蓄滿了！張開雙臂發射！",
  "blast.hud.whiff": "沒打中！",
  "blast.board.title": "本地排行榜",
  "blast.board.empty": "還沒有紀錄——來當第一名吧！",
  "blast.board.you": "你",
  "blast.over.title": "回合結束",
  "blast.over.hits": "顆光球",
  "blast.over.combo": "最高連擊",
  "blast.over.nameLabel": "把名字加入排行榜",
  "blast.over.namePlaceholder": "你的名字",
  "blast.over.save": "儲存",
  "blast.over.anon": "匿名玩家",
  "blast.over.ranked": "你在本地排行榜排名第 {rank}！",
  "blast.over.notRanked": "已儲存——繼續轟爆擠進前十名。",
  "blast.over.replay": "再玩一次",

  // 後台管理面板（P1 空殼——依角色授權；其餘分頁於後續階段推出）
  "admin.nav": "後台管理",
  "admin.title": "後台管理",
  "admin.subtitle": "管理使用者、LLM 設定與分析管線參數。更多功能即將推出。",
  "admin.denied": "你沒有存取後台管理面板的權限。",
  "admin.loading": "確認你的存取權限中…",
  "admin.error": "無法確認你的後台權限，請再試一次。",
  "admin.retry": "重試",

  // 後台管理獨立主控台外框（導覽＋分頁標題）
  "admin.console.title": "後台管理主控台",
  "admin.nav.overview": "總覽",
  "admin.nav.users": "使用者",
  "admin.nav.settingsLlm": "LLM 對話",
  "admin.nav.settingsRag": "RAG / KG",
  "admin.nav.settingsAnalyze": "分析管線",
  "admin.nav.backToApp": "返回 app",

  // 後台管理專用登入（僅電子郵件 + 密碼，無 OAuth）
  "adminLogin.title": "後台管理主控台",
  "adminLogin.subtitle": "請以你的管理員帳號登入以繼續。",
  "adminLogin.submit": "登入",
  "adminLogin.submitting": "登入中…",
  "adminLogin.error": "無法登入，請確認電子郵件與密碼。",
  "adminLogin.switchAccount": "以其他帳號登入",
  "adminLogin.notConfigured": "此伺服器尚未設定登入功能。",

  // 後台管理 P2 — 執行階段設定
  "admin.settings.loading": "載入設定中…",
  "admin.settings.loadError": "無法載入目前的設定。",
  "admin.settings.save": "儲存變更",
  "admin.settings.saving": "儲存中…",
  "admin.settings.saved": "設定已儲存。",
  "admin.settings.saveError": "無法儲存設定。",
  "admin.settings.defaultLabel": "預設值：{value}",
  "admin.settings.llm": "LLM 對話設定",
  "admin.settings.llmDesc": "調整對話式教練的模型清單與請求行為。",
  "admin.settings.models": "可選模型",
  "admin.settings.modelsHint": "每行一個 id 或以逗號分隔，第一個為預設值。",
  "admin.settings.followupModel": "追問模型",
  "admin.settings.baseUrl": "供應商 Base URL",
  "admin.settings.temperature": "Temperature",
  "admin.settings.temperatureHint": "留白則不送出（0–2）。",
  "admin.settings.chatTimeout": "回覆逾時（秒）",
  "admin.settings.followupTimeout": "追問逾時（秒）",
  "admin.settings.ragkg": "RAG / KG",
  "admin.settings.ragkgDesc": "檢索預設廣度（端點 query 參數仍可覆寫）。",
  "admin.settings.ragTopK": "RAG top-k",
  "admin.settings.kgHops": "KG 跳數",
  "admin.settings.kgSeeds": "KG 種子節點數",
  "admin.settings.analyze": "分析管線",
  "admin.settings.analyzeDesc": "上傳限制與並發數。",
  "admin.settings.uploadFormats": "允許的上傳格式",
  "admin.settings.uploadFormatsHint": "以逗號分隔的 .ext 值。",
  "admin.settings.maxConcurrent": "最大同時分析數",
  "admin.settings.restartRequired": "需重啟才生效",
  "admin.settings.maxConcurrentReadonly": "唯讀：透過 XCOACH_MAX_CONCURRENT_ANALYSES 環境變數設定，需重啟才能變更。",
  "admin.settings.invalidNumber": "請為每個欄位輸入有效的數字。",

  // 後台管理 P3 — 系統狀態總覽
  "admin.overview.title": "系統總覽",
  "admin.overview.desc": "部署的即時健康狀態與使用量。",
  "admin.overview.loadError": "無法載入系統總覽。",
  "admin.overview.auth": "身分驗證",
  "admin.overview.chat": "LLM 對話",
  "admin.overview.configured": "已設定",
  "admin.overview.notConfigured": "未設定",
  "admin.overview.stores": "資料儲存",
  "admin.overview.storesReady": "{ready}/{total} 就緒",
  "admin.overview.totalUsers": "使用者總數",
  "admin.overview.totalAnalyses": "分析總數",

  // 後台管理 P3 — 使用者監看
  "admin.users.title": "使用者",
  "admin.users.desc": "唯讀活動總覽。可逐一指派或取消管理員權限。",
  "admin.users.loading": "載入使用者中…",
  "admin.users.loadError": "無法載入使用者清單。",
  "admin.users.empty": "尚無使用者。",
  "admin.users.email": "電子郵件",
  "admin.users.created": "註冊時間",
  "admin.users.lastSignIn": "最後登入",
  "admin.users.analyses": "分析數",
  "admin.users.conversations": "對話數",
  "admin.users.role": "管理員",
  "admin.users.makeAdmin": "設為管理員",
  "admin.users.revokeAdmin": "取消管理員",
  "admin.users.you": "你",
  "admin.users.never": "從未",
  "admin.users.updateError": "無法更新這位使用者的角色。",

  // 67（小遊戲）— 「six seven」迷因手勢計數
  "nav.six": "67",
  "six.title": "67",
  "six.badge": "迷因 · 一台鏡頭",
  "six.heading": "你能做出幾次 67？",
  "six.sub":
    "把「6-7」抖手動作變成計數遊戲。雙手往前伸，一上一下輪流擺動——six… seven…——盡量累積次數。用的正是 x-coach 分析姿態的同一套 MediaPipe 追蹤，對準了這個迷因。",
  "six.how1": "站遠一點，讓雙手與肩膀都進到畫面。",
  "six.how2": "先舉一隻手，再換另一隻——每次交替算一個 67。",
  "six.how3": "保持節奏就能累積連擊。限時 {s} 秒。",
  "six.leftHand": "左手在上",
  "six.rightHand": "右手在上",
  "six.startBtn": "開啟鏡頭開始",
  "six.starting": "啟動鏡頭中…",
  "six.cameraNote": "全程裝置端運算。{s} 秒衝刺。",
  "six.error": "無法啟動鏡頭。",

  // 回合中 HUD
  "six.hud.time": "時間",
  "six.hud.label": "六七次數",
  "six.hud.combo": "{n}× 節奏",

  // 結算畫面
  "six.over.title": "時間到！你完成了",
  "six.over.combo": "最佳節奏連擊：{n}",
  "six.over.nameLabel": "把名字加進排行榜",
  "six.over.namePlaceholder": "你的名字",
  "six.over.save": "儲存",
  "six.over.anon": "匿名",
  "six.over.ranked": "你在本地排行榜第 {rank} 名！",
  "six.over.notRanked": "已儲存——繼續抖手擠進前 10 名。",
  "six.over.replay": "再玩一次",

  // 排行榜
  "six.board.title": "67 排行榜",
  "six.board.empty": "還沒有紀錄——搶第一個吧！",
  "six.board.you": "你",
  "six.board.count": "{n} × 67",

  // 水果忍者（小遊戲）
  "nav.ninja": "水果忍者",
  "ninja.title": "水果忍者",
  "ninja.badge": "切水果 · MediaPipe",
  "ninja.heading": "你的雙手就是刀。",
  "ninja.sub":
    "水果往上飛，用你的雙手把它切開。MediaPipe 即時追蹤雙手手腕——與 x-coach 深蹲分析同一套姿態引擎——把每一次揮動變成一把刀。",
  "ninja.how1": "站遠一點，讓雙手與肩膀都進到畫面。",
  "ninja.how2": "揮手劃過飛來的水果就能切開，要夠快才切得動。",
  "ninja.how3": "閃避 💣，別漏接 {lives} 顆水果——連續切開累積連擊。",
  "ninja.startBtn": "開啟鏡頭開切",
  "ninja.starting": "啟動鏡頭中…",
  "ninja.cameraNote": "全程裝置端運算。雙手一起玩最過癮。",
  "ninja.deckTitle": "會出現的東西",
  "ninja.error": "無法啟動鏡頭。",

  // 回合中 HUD
  "ninja.hud.score": "分數",
  "ninja.hud.lives": "生命",
  "ninja.hud.combo": "{n} 連擊！",
  "ninja.hud.boom": "💥 爆炸",
  "ninja.hud.slice": "揮手切開",

  // 結算畫面
  "ninja.over.title": "回合結束",
  "ninja.over.bombed": "你切到炸彈了！",
  "ninja.over.combo": "最高連擊：{n}",
  "ninja.over.nameLabel": "把名字加進排行榜",
  "ninja.over.namePlaceholder": "你的名字",
  "ninja.over.save": "儲存",
  "ninja.over.anon": "匿名",
  "ninja.over.ranked": "你在本地排行榜第 {rank} 名！",
  "ninja.over.notRanked": "已儲存——繼續切擠進前 10 名。",
  "ninja.over.replay": "再玩一次",

  // 排行榜
  "ninja.board.title": "切水果排行榜",
  "ninja.board.empty": "還沒有紀錄——搶第一個吧！",
  "ninja.board.you": "你",
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
