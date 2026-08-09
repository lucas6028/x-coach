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
  "nav.games": "Games",
  "nav.hide": "Hide navigation",
  "nav.show": "Show navigation",
  // The desktop rail's own width toggle. Deliberately NOT worded "hide/show navigation" — that
  // pair names the tablet drawer's ✕ and the navbar's ☰, and two controls sharing an accessible
  // name makes every by-name query in the layout tests ambiguous.
  "nav.collapse": "Collapse navigation",
  "nav.expand": "Expand navigation",
  "nav.tabBar": "Primary navigation",

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
  "games.liffCameraHint":
    "The live camera may not work inside LINE. If a game won't start, tap ⋮ at the top right and choose \"Open in browser\".",
  "games.play": "Play",
  "games.stat.bestScore": "best",
  "games.stat.bestCount": "best 67s",
  "games.ninja.desc": "Your hands are the blades — slice the flying fruit, dodge the bombs.",
  "games.six.desc": "Do the 6-7 bob: raise one hand then the other and rack up as many as you can.",

  // Per-round calorie estimate (shown on each game's over screen)
  "game.kcal.est": "≈ {n} kcal",
  "game.kcal.note": "estimated",

  // Header

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
  // Shown instead of "clean rep" when zero frames were measurable: an empty detection list then
  // means "never measured", not "nothing wrong". See MetricsCards.
  "metric.notMeasured": "not measured",
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
  "chat.tool.get_analysis": "Re-reading the analysis",
  "chat.tool.kg_query": "Searching the knowledge graph",
  "chat.tool.rag_search": "Searching the literature",
  "chat.tool.generic": "Looking something up",
  // The separator between the tool label and its query subject ("Searching the knowledge graph{sep}knee
  // valgus"). Localised because zh-Hant wants the fullwidth "：" — the only user-facing string on this
  // branch that isn't already routed through t(), before this fix.
  "chat.tool.sep": ": ",
  "chat.tool.sourcesN": "Sources · {n}",
  "chat.tool.conceptsN": "Knowledge-graph concepts · {n}",
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
  "loader.neutral": "Loading…",

  "a11y.close": "Close",

  // Reasoning / coaching feedback
  "feedback.title": "Coaching Feedback",
  "feedback.badge": "rule + GraphRAG",
  // Names the movement because it is USER-ASSERTED: the studio lets the user pick, so a clip can
  // be measured by rules that do not describe it. Naming it makes the claim true relative to the
  // user's own assertion and puts that assertion in front of them at the verdict.
  "feedback.noFaults": "No {movement} faults detected. Clean rep.",
  // Shown instead of the clean-rep banner when no frame was measurable: an empty fault list then
  // means "never measured", not "nothing wrong". See lib/quality.ts.
  "feedback.notMeasured":
    "No frame in this clip could be measured, so no form verdict was produced. Re-record with your whole body in frame.",
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
  "fault.heel_rise": "Heel Rise",
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
  "app.loading": "Loading {id}…",
  "app.analysing": "Extracting pose & analysing… (this can take ~20s)",
  "tab.coaching": "Coaching",
  "tab.graph": "Knowledge Graph",

  // Upload dropzone
  "upload.analysing": "Analysing…",
  "upload.prompt": "Drop a {movement} video or tap to upload",
  "upload.hint": "MP4 / MOV · single athlete · side or rear view",
  "upload.tooLarge": "That clip is too large — the limit is {limit} MB. Try a shorter recording.",
  "upload.quotaFull": "Your storage is full ({used} MB of {limit} MB). Delete a saved analysis to make room.",

  // Capture studio — input mode + the MediaPipe model-tier picker
  "capture.upload": "Upload video",
  "capture.record": "Record live",
  "capture.progress": "Analysing… {pct}%",
  "tier.label": "Precision",
  "tier.aria": "Analysis precision: {name}",
  "tier.default": "Default",
  "tier.lite.hint": "Fastest. Agrees with Heavy on only about half of squat verdicts.",
  "tier.full.hint": "Middle ground between speed and accuracy.",
  "tier.heavy.hint": "Most accurate — what the fault thresholds were validated against.",
  "tier.note": "Applies to the analysis only; the live skeleton overlay always runs Lite.",

  // Demo onboarding (empty state)
  "demo.heading": "Analyze your {movement} in about 20 seconds.",
  "demo.sub": "Upload or record a clip. You get a skeleton overlay, a fault timeline, and coaching feedback x-coach can trace to the cause.",
  "demo.getTitle": "What comes back",
  "demo.get1.title": "Skeleton and faults",
  "demo.get1.body": "Pose overlay with every detected fault marked on the timeline.",
  "demo.get2.title": "Grounded feedback",
  "demo.get2.body": "An observation, a likely cause, and a corrective cue per fault.",
  "demo.get3.title": "Knowledge graph",
  "demo.get3.body": "The retrieval path that links each symptom to its cause.",
  "demo.errorTitle": "That clip did not go through",

  // Studio movement selector
  "studio.movement": "Movement",
  "studio.movementUnavailable":
    "\"{movement}\" cannot be analysed yet. Pick one of the available movements.",

  // Studio page header
  "studio.crumbHome": "Home",
  "studio.crumbWorkout": "Workout",
  "studio.crumbCurrent": "{movement} Analysis",
  "studio.title": "{movement} Motion Analysis",
  "studio.subtitle": "Get grounded feedback and improve your form",
  "studio.newSession": "Start / upload video",

  // Studio dashboard cards
  "studio.previous": "Your previous sessions",
  "studio.viewAll": "View all",
  "studio.previousEmpty": "No earlier sessions yet — this is your first.",
  "studio.previousSignIn": "Sign in to keep a history of your sessions.",
  "studio.previousError": "Couldn't load your earlier sessions.",
  "studio.sessionClean": "Clean",
  "studio.sessionFaults": "{n} faults",
  "studio.keyMetrics": "Key metrics",
  "studio.metricLimit": "(limit {op} {v})",
  "studio.tips": "Tips for improvement",
  "studio.tipsNone": "The knowledge graph returned no corrective cue for these faults.",
  "studio.tipsClean": "Nothing to correct on this rep.",
  "studio.detectedErrors": "Detected errors",
  "studio.moreFaults": "+{n} more in the coach panel",

  // Derived form score (client-side, see lib/formScore.ts)
  "studio.formScore": "Form score",
  "studio.formScoreFrom": "From detected faults",
  "studio.formScoreNote":
    "Derived on this device from the detected faults and their severities — not a backend measurement.",
  "studio.formScoreUnknown": "Nothing measurable in this clip",
  "studio.band.excellent": "Excellent",
  "studio.band.good": "Good",
  "studio.band.fair": "Fair",
  "studio.band.poor": "Needs work",

  // Phone layout (motion_analysis_mobile)
  "mobile.back": "Back",
  "mobile.analyze": "Analyze",
  "mobile.reps": "Reps",
  "mobile.duration": "Duration",
  "mobile.topIssues": "Top issues",
  "mobile.issuesN": "{n} detected",
  "mobile.research": "Based on research",
  "mobile.researchSub": "Evidence-backed suggestions",
  "mobile.pastSessions": "Past sessions",
  "mobile.pastSessionsSub": "Your progress over time",
  "mobile.detailedMetrics": "Detailed metrics",
  "mobile.detailedMetricsSub": "Clip quality & measurements",
  "mobile.evidence": "Measured",

  // Language
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
    "x-coach reads a squat, push-up or overhead-press video, locates the fault, traces its cause in a biomechanics knowledge graph spanning 16 movements, and explains the fix.",
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
    "Upload or record a clip, and watch the skeleton, the faults, and the grounded feedback come back together.",

  // Landing — footer
  "landing.footer.pipeline": "Pipeline",
  "landing.footer.tagline": "Explainable movement coaching, research prototype.",

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
  "auth.back": "Back to app",
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
  "auth.lineBtn": "Continue with LINE",

  // Camera-in-LIFF fallback + the /liff/diag device check page
  "camera.liffHint":
    "The LINE in-app browser couldn't open the camera on this device. Tap the ⋯ menu and choose \"Open in browser\" to play with the camera.",
  "diag.title": "LIFF device check",
  "diag.subtitle":
    "Open this page inside LINE on your phone to verify login and camera support before rolling out the LIFF app.",
  "diag.env": "Environment",
  "diag.session": "Login session",
  "diag.signedIn": "Signed in",
  "diag.signedOut": "Not signed in",
  "diag.camera": "Live camera (getUserMedia)",
  "diag.probeBtn": "Test camera",
  "diag.probing": "Testing…",
  "diag.cameraOk": "Camera works in this browser.",
  "diag.pose": "Live pose (camera + MediaPipe)",
  "diag.poseBtn": "Test camera + pose",
  "diag.poseGood": "Smooth — the live-camera games should be playable here.",
  "diag.poseMarginal": "Runs, but slowly — expect a degraded game experience.",
  "diag.poseBad": "Too slow — keep live-camera features out of LIFF on this device.",
  "diag.poseNoLandmarks":
    "Camera and model both ran, but no person was detected — aim the camera at yourself and retest.",
  "diag.capture": "Video file capture (tap to confirm the camera recorder opens)",
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
  "nav.movements": "Movements",
  "nav.settings": "Settings",
  "account.signin": "Sign in",
  "account.signout": "Sign out",
  "account.menu": "Account menu",
  "account.settings": "Settings",
  "account.lineSigningIn": "Signing in with LINE…",

  // Movements menu
  "movements.title": "Movement library",
  "movements.subtitle":
    "Every movement the coach knows. Squat, Push-up and Overhead Press analysis are live today; the rest are on the way.",
  "movements.groupLower": "Lower body",
  "movements.groupUpper": "Upper body",
  "movements.groupCore": "Core",
  "movements.groupFullBody": "Full body",
  "movements.filterAll": "All",
  "movements.searchPlaceholder": "Search movements",
  "movements.noMatch": "No movements match your search.",
  "movements.uploadCta": "Upload a video",
  "movements.analyze": "Analyze a video",
  "movements.soon": "Soon",
  "movements.beta": "Beta",
  "movements.betaNote":
    "Rules for this movement are literature-derived and have not yet been validated against labeled data.",

  // Movement display names. Only the ones titleCase() would mangle need an entry here; the rest
  // fall back to their canonical spelling.
  "movement.Push-up": "Push-up",
  "movement.Sit-up": "Sit-up",

  // Settings popup
  "settings.title": "Settings",
  "settings.nav": "Settings sections",
  "settings.close": "Close settings",
  "settings.search": "Search",
  "settings.searchEmpty": "No matching settings.",
  "settings.groupSettings": "Settings",
  "settings.groupCustomize": "Customize",
  "settings.general": "General",
  "settings.preferences": "Preferences",
  "settings.account": "Account",
  "settings.avatar": "Avatar",
  "settings.language": "Language",
  "settings.profile": "Profile",
  "settings.name": "Name",
  "settings.email": "Email",
  "settings.provider": "Signed in with",
  "settings.provider.google": "Google",
  "settings.provider.line": "LINE",
  "settings.provider.email": "Email & password",
  "settings.model": "Coach model",
  "settings.modelDesc": "Choose which LLM answers your follow-up questions. Applies to new messages.",
  "settings.modelDefault": "Default",
  "settings.modelLoading": "Loading models…",
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
  "history.rowTitle": "{view} {movement}",
  "history.clean": "clean rep",
  "history.faultOne": "1 fault",
  "history.faultMany": "{count} faults",
  "history.deleteAria": "Delete this record",
  "history.deleteCta": "Delete",
  "history.deleteTitle": "Delete this record?",
  "history.deleteDesc": "This can't be undone.",
  "history.deleteConfirm": "Delete",
  "history.deleteCancel": "Cancel",
  "history.deleting": "Deleting…",
  "history.deleteError": "Couldn't delete this record. Please try again.",
  // Summary strip. Only `statTotal` is a true all-time figure (the API returns it alongside the
  // page); the rest are derived from the rows actually loaded, which `statsScope` says out loud.
  // The header CTA. Distinct from `startCta` ("Analyze a squat"), which the empty state uses to
  // pitch the one movement a newcomer can try; once there are records the button is just the way
  // to add another, and naming a movement there would read as a filter.
  "history.uploadCta": "Upload a video",
  "history.statTotal": "Total analyses",
  "history.statCleanRate": "Clean-rep rate",
  "history.statTopMovement": "Most trained",
  "history.statLatest": "Latest analysis",
  "history.statTimes": "{count} analyses",
  "history.statNone": "—",
  "history.statsScope": "Rates below are from the {loaded} most recent of {total} analyses.",
  // Filter bar
  "history.searchPlaceholder": "Search movements",
  "history.filterMovement": "Movement",
  "history.filterStatus": "Result",
  "history.filterRange": "Period",
  // Each menu's "no filter" entry. Worded as a value ("All movements"), not repeated as the
  // caption — the caption is already above it on the control.
  "history.filterAllMovements": "All movements",
  "history.filterAllResults": "All results",
  // The phone layout's disclosure for the three menus. `{count}` is how many are set, so the
  // button says whether anything is hidden behind it.
  "history.filterToggle": "Filters",
  "history.filterToggleActive": "Filters ({count} active)",
  "history.filterClean": "Clean reps",
  "history.filterFaults": "Needs work",
  "history.rangeAll": "All time",
  "history.rangeToday": "Today",
  "history.range7": "Last 7 days",
  "history.range30": "Last 30 days",
  "history.clearFilters": "Clear filters",
  "history.noMatch": "No records match these filters.",
  // Day separators
  "history.today": "Today",
  "history.yesterday": "Yesterday",

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
  "admin.nav.line": "LINE",
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
  "admin.settings.maxUploadBytes": "Max upload size (bytes)",
  "admin.settings.userStorageQuotaBytes": "Per-user storage quota (bytes)",
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
  "admin.line.title": "LINE",
  "admin.line.desc": "LINE Messaging status — connection, push quota, webhook health, and delivery counts.",
  "admin.line.loadError": "Couldn't load the LINE status.",
  "admin.line.loginBridge": "LINE login bridge",
  "admin.line.bot": "LINE bot",
  "admin.line.pushUsed": "Push used this month",
  "admin.line.remaining": "Free remaining",
  "admin.line.unreachable": "Couldn't reach LINE for quota.",
  "admin.line.noCapNote":
    "No monthly limit set in LINE Official Account Manager, so remaining can't be shown. Set a monthly message limit there to track your free allowance.",
  "admin.line.oaName": "Official account",
  "admin.line.chatModeWarn": "Chat mode — the webhook won't receive message events",
  "admin.line.webhook": "Webhook",
  "admin.line.webhookActive": "Active",
  "admin.line.webhookInactive": "Inactive",
  "admin.line.webhookTest": "Test webhook",
  "admin.line.webhookTesting": "Testing…",
  "admin.line.webhookReachable": "Reachable ({code})",
  "admin.line.webhookFailed": "Failed ({code}: {reason})",
  "admin.line.webhookTestError": "Couldn't reach LINE.",
  "admin.line.webhookTestUnauthorized": "LINE rejected the channel access token — check or reissue it.",
  "admin.line.webhookTestRateLimited": "LINE rate-limited the test. Wait a minute and try again.",
  "admin.line.webhookTestNoEndpoint": "No webhook endpoint is set on this channel yet.",
  "admin.line.webhookTestNotConfigured": "LINE messaging isn't configured on this server.",
  "admin.line.botInfoUnavailable": "Couldn't read the official-account info.",
  "admin.line.webhookUnavailable": "Couldn't read the webhook setting — run the test to find out why.",
  "admin.line.deliveryUnavailable": "Couldn't read yesterday's delivery counts.",
  "admin.line.replyYesterday": "Replies yesterday",
  "admin.line.pushYesterday": "Pushes yesterday",
  "admin.line.deliveryUnready": "Not ready yet",
  "admin.line.deliveryDate": "Counts for {date}",

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

  // Spider-Man Web Slinger (mini-game)
  "games.web.desc": "Play as Spider-Man and fire web silk from your hands to tag moving drones.",
  "web.title": "Spider-Man Web Slinger",
  "web.badge": "Web shots · MediaPipe",
  "web.heading": "Your hands shoot the webs.",
  "web.sub":
    "Aim with either arm, snap your wrist outward, and web every drone before time runs out.",
  "web.how1": "Stand back so both wrists and elbows stay in frame.",
  "web.how2": "Point your forearm at a target, then flick your wrist outward to fire.",
  "web.how3": "Web targets in a row to build a combo. You have {s} seconds.",
  "web.startBtn": "Enable camera & swing",
  "web.starting": "Starting camera...",
  "web.cameraNote": "Runs on-device. Use either hand or alternate both.",
  "web.error": "Couldn't start the camera.",
  "web.hud.score": "Score",
  "web.hud.time": "Time",
  "web.hud.combo": "{n}x web combo",
  "web.hud.hint": "Point, then snap your wrist outward",
  "web.over.title": "City secured",
  "web.over.stats": "{hits} drones webbed · best combo {combo}",
  "web.over.nameLabel": "Add your name to the board",
  "web.over.namePlaceholder": "Your name",
  "web.over.save": "Save",
  "web.over.anon": "Anonymous",
  "web.over.ranked": "You're #{rank} on the local board!",
  "web.over.saved": "Score saved. Keep swinging for the top 10.",
  "web.over.replay": "Play again",
  "web.board.title": "Top web slingers",
  "web.board.empty": "No scores yet. Be the first hero on the board.",
};

const zhHant: Dict = {
  "games.web.desc": "化身蜘蛛人，從雙手射出蛛絲，擊中移動中的無人機。",
  "web.title": "蜘蛛人蛛絲射手",
  "web.badge": "蛛絲射擊 · MediaPipe",
  "web.heading": "你的雙手就是蛛絲發射器。",
  "web.sub": "用任一手臂瞄準，手腕向外甩動，在時間結束前纏住所有無人機。",
  "web.how1": "站遠一點，讓雙手腕與手肘都留在畫面中。",
  "web.how2": "前臂對準目標，手腕快速向外甩動即可射出蛛絲。",
  "web.how3": "連續命中可累積連擊。你有 {s} 秒。",
  "web.startBtn": "開啟鏡頭開始射擊",
  "web.starting": "啟動鏡頭中...",
  "web.cameraNote": "全程在裝置端運算，可用任一手或雙手交替。",
  "web.error": "無法啟動鏡頭。",
  "web.hud.score": "分數",
  "web.hud.time": "時間",
  "web.hud.combo": "{n}x 蛛絲連擊",
  "web.hud.hint": "先瞄準，再將手腕向外甩動",
  "web.over.title": "城市安全了",
  "web.over.stats": "纏住 {hits} 架無人機 · 最佳連擊 {combo}",
  "web.over.nameLabel": "把名字加進排行榜",
  "web.over.namePlaceholder": "你的名字",
  "web.over.save": "儲存",
  "web.over.anon": "匿名",
  "web.over.ranked": "你在本地排行榜第 {rank} 名！",
  "web.over.saved": "分數已儲存。繼續擺盪，挑戰前 10 名。",
  "web.over.replay": "再玩一次",
  "web.board.title": "蛛絲射手排行榜",
  "web.board.empty": "還沒有分數，成為排行榜上的第一位英雄。",
  // Sidebar
  "nav.newAnalysis": "新增分析",
  "nav.analyse": "分析",
  "nav.games": "小遊戲",
  "nav.hide": "隱藏導覽列",
  "nav.show": "顯示導覽列",
  "nav.collapse": "收合導覽列",
  "nav.expand": "展開導覽列",
  "nav.tabBar": "主要導覽",

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
  "games.liffCameraHint":
    "在 LINE 內即時相機可能無法使用。若遊戲開不起來，請點右上角 ⋮ 選「用其他瀏覽器開啟」。",
  "games.play": "開始遊玩",
  "games.stat.bestScore": "最高分",
  "games.stat.bestCount": "最多 67",
  "games.ninja.desc": "你的雙手就是刀——切開飛來的水果，閃避炸彈。",
  "games.six.desc": "做「6-7」抖手：一手上一手下輪流擺動，盡量累積次數。",

  // 每回合卡路里估計（顯示於各遊戲的結算畫面）
  "game.kcal.est": "≈ {n} 大卡",
  "game.kcal.note": "估計值",

  // Header

  // Camera views
  "view.front": "正面",
  "view.side": "側面",
  "view.rear": "背面",
  "view.left": "左側",
  "view.right": "右側",
  "view.unknown": "未知",

  // Video panel
  "video.faultOne": "偵測到 1 個動作問題",
  "video.faultMany": "偵測到 {count} 個動作問題",
  "video.noFaults": "未偵測到動作問題",
  "a11y.play": "播放",
  "a11y.pause": "暫停",
  "a11y.fullscreen": "切換全螢幕",

  // Timeline
  "timeline.fault": "動作問題",
  "timeline.neutral": "正常",

  // Metrics
  "metric.cameraView": "拍攝視角",
  "metric.faults": "問題數",
  "metric.lowerBodyVis": "下肢可見度",
  "metric.validFrames": "有效影格",
  "metric.conf": "可信度 {v}",
  "metric.peakSeverity": "最高問題程度 {v}",
  "metric.cleanRep": "動作穩定",
  "metric.notMeasured": "無法判讀",
  "metric.landmarkConf": "關鍵點可信度",
  "metric.framesRatio": "{valid}/{total} 影格",

  // Chat input — disabled fallback (auth/LLM not configured) + the working grounded chat.
  "chat.placeholder": "詢問 Lumen…（LLM 功能即將推出）",
  "chat.title": "對話式教練功能將隨 LLM 層推出。",
  "chat.heading": "Lumen",
  "chat.grounded": "根據你的分析",
  "chat.groundedShort": "有依據",
  "chat.intro": "針對這次深蹲繼續提問；回答只會根據偵測到的動作問題和找到的建議。",
  "chat.suggestFix": "我該先修正什麼？",
  "chat.suggestDrill": "給我一個矯正動作",
  "chat.suggestWhy": "這為什麼重要？",
  "chat.placeholderActive": "追問後續問題…",
  "chat.send": "傳送訊息",
  "chat.thinking": "Lumen 思考中…",
  "chat.tool.get_analysis": "重讀分析細節",
  "chat.tool.kg_query": "搜尋知識圖譜",
  "chat.tool.rag_search": "查詢文獻",
  "chat.tool.generic": "查詢中",
  "chat.tool.sep": "：",
  "chat.tool.sourcesN": "引用來源 {n} 筆",
  "chat.tool.conceptsN": "知識圖譜概念 {n} 筆",
  "chat.signIn": "登入即可就本次分析與 Lumen 對話。",
  "chat.error": "無法連線至 Lumen，請再試一次。",
  "chat.sessionExpired": "登入階段已過期，請重新登入以繼續對話。",
  "chat.you": "你",
  "chat.coach": "Lumen",
  "coach.followUp": "後續追問",
  "loader.aria": "Lumen 分析中",
  "loader.step1": "讀取動作",
  "loader.step2": "檢查動作細節",
  "loader.step3": "找出可能原因",
  "loader.neutral": "載入中…",

  "a11y.close": "關閉",

  // Reasoning / coaching feedback
  "feedback.title": "教練回饋",
  "feedback.badge": "規則 + GraphRAG",
  "feedback.noFaults": "這次{movement}沒有偵測到明顯的動作問題，做得很穩。",
  "feedback.notMeasured":
    "這支影片沒有可判讀的畫面，所以暫時無法評估動作。請確認全身都入鏡後再錄一次。",
  "feedback.graphragContext": "建議依據",
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
  "phase.descent": "下蹲",
  "phase.ascent": "起身",
  "phase.bottom": "最低點",
  "phase.top": "頂點",
  "phase.eccentric": "下放",
  "phase.concentric": "發力起身",
  "phase.hold": "停頓",
  "phase.transition": "轉換",
  "phase.setup": "準備",
  "phase.full": "完整",

  // Fault names
  "fault.knees_inward": "膝蓋內夾",
  "fault.knees_forward": "膝蓋前移",
  "fault.shallow_depth": "深度不足",
  "fault.excessive_forward_lean": "軀幹過度前傾",
  "fault.heel_rise": "腳跟離地",
  "fault.butt_wink": "骨盆捲起",
  "fault.asymmetric_shift": "左右不對稱",

  // Knowledge graph
  "kg.title": "知識圖譜",
  "kg.chain": "動作問題 → 原因 → 改善方式",
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
  "app.loading": "載入 {id} 中…",
  "app.analysing": "擷取姿態並分析中…（約需 20 秒）",
  "tab.coaching": "教練回饋",
  "tab.graph": "知識圖譜",

  // Upload dropzone
  "upload.analysing": "分析中…",
  "upload.prompt": "拖放 {movement} 影片，或點一下上傳",
  "upload.hint": "MP4 / MOV · 單一運動員 · 側面或背面視角",
  "upload.tooLarge": "這支影片太大了，上限是 {limit} MB。請改用較短的片段。",
  "upload.quotaFull": "儲存空間已滿（{used} MB / {limit} MB）。請先刪除部分已存分析來騰出空間。",

  // Capture studio — input mode + the MediaPipe model-tier picker
  "capture.upload": "上傳影片",
  "capture.record": "即時錄影",
  "capture.progress": "分析中… {pct}%",
  "tier.label": "分析精度",
  "tier.aria": "分析精度：{name}",
  "tier.default": "預設",
  "tier.lite.hint": "最快，但深蹲判定僅約半數與 Heavy 一致。",
  "tier.full.hint": "速度和準確度之間的平衡。",
  "tier.heavy.hint": "最準確；目前的問題判定標準就是用它驗證的。",
  "tier.note": "僅影響分析；即時骨架疊圖一律使用 Lite。",

  // Demo onboarding (empty state)
  "demo.heading": "約 20 秒，分析一段 {movement}。",
  "demo.sub": "上傳影片或直接錄一段。你會看到骨架疊圖、問題時間軸，以及附上原因的教練建議。",
  "demo.getTitle": "你會得到",
  "demo.get1.title": "骨架與動作問題",
  "demo.get1.body": "骨架疊圖會在時間軸上標出每個偵測到的動作問題。",
  "demo.get2.title": "有據回饋",
  "demo.get2.body": "每個動作問題都有觀察結果、可能原因和改善提示。",
  "demo.get3.title": "知識圖譜",
  "demo.get3.body": "帶你看系統怎麼從動作問題找到可能原因。",
  "demo.errorTitle": "這段片段沒有成功處理",

  // Studio movement selector
  "studio.movement": "動作",
  "studio.movementUnavailable": "「{movement}」尚未支援分析，請選擇其他已開放的動作。",

  // Studio page header
  "studio.crumbHome": "首頁",
  "studio.crumbWorkout": "訓練",
  "studio.crumbCurrent": "{movement} 分析",
  "studio.title": "{movement} 動作分析",
  "studio.subtitle": "看懂動作問題，把每一下做得更好",
  "studio.newSession": "開始／上傳影片",

  // Studio dashboard cards
  "studio.previous": "先前的紀錄",
  "studio.viewAll": "查看全部",
  "studio.previousEmpty": "還沒有更早的紀錄，這是第一次。",
  "studio.previousSignIn": "登入後即可保留每次分析的紀錄。",
  "studio.previousError": "無法載入先前的紀錄。",
  "studio.sessionClean": "動作穩定",
  "studio.sessionFaults": "{n} 個動作問題",
  "studio.keyMetrics": "關鍵指標",
  "studio.metricLimit": "（門檻 {op} {v}）",
  "studio.tips": "改善建議",
  "studio.tipsNone": "目前找不到能對應這些動作問題的改善提示。",
  "studio.tipsClean": "這一下沒有需要修正的地方。",
  "studio.detectedErrors": "偵測到的動作問題",
  "studio.moreFaults": "教練面板還有 {n} 項",

  // Derived form score (client-side, see lib/formScore.ts)
  "studio.formScore": "動作分數",
  "studio.formScoreFrom": "依偵測到的動作問題計算",
  "studio.formScoreNote": "這個分數會依偵測到的動作問題和程度在你的裝置上計算，不是後端直接量出來的數值。",
  "studio.formScoreUnknown": "這段影片沒有可量測的畫面",
  "studio.band.excellent": "優秀",
  "studio.band.good": "良好",
  "studio.band.fair": "尚可",
  "studio.band.poor": "需加強",

  // Phone layout (motion_analysis_mobile)
  "mobile.back": "返回",
  "mobile.analyze": "分析",
  "mobile.reps": "次數",
  "mobile.duration": "時長",
  "mobile.topIssues": "主要問題",
  "mobile.issuesN": "偵測到 {n} 項",
  "mobile.research": "文獻依據",
  "mobile.researchSub": "有實證支持的建議",
  "mobile.pastSessions": "先前的紀錄",
  "mobile.pastSessionsSub": "看看這段時間的進步",
  "mobile.detailedMetrics": "詳細指標",
  "mobile.detailedMetricsSub": "影片品質與量測值",
  "mobile.evidence": "量測值",

  // Language
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
    "x-coach 會分析深蹲、伏地挺身和肩上推舉影片，找出動作問題，從涵蓋 16 種動作的知識圖譜找原因，再告訴你怎麼改善。",
  "landing.hero.readMethod": "了解方法",

  // Landing — problem
  "landing.problem.title": "分數不會告訴你怎麼練，通用模型也只能猜。",
  "landing.problem.sub":
    "只會打分的動作模型只給你一個數字，沒有改善方向。通用語言模型聽起來很有把握，卻可能把原因講錯。",
  "landing.problem.aqs.label": "只會打分的模型",
  "landing.problem.aqs.body":
    "它回傳 0 到 100 分。你只知道自己拿了 71 分，卻不知道哪裡要改、為什麼要改。",
  "landing.problem.llm.label": "通用語言模型",
  "landing.problem.llm.body":
    "它能給出流暢的建議，卻不一定真的看懂你的影片，還可能猜錯原因。",
  "landing.problem.xcoach.title": "每個建議都有根據",
  "landing.problem.point1": "從實際畫面找出動作問題",
  "landing.problem.point2": "從有來源的知識圖譜找可能原因",
  "landing.problem.point3": "每個改善方式都說得出依據",

  // Landing — pipeline
  "landing.pipeline.kicker": "系統架構",
  "landing.pipeline.title": "四個步驟，從畫面一路帶你找到改善方式。",
  "landing.stage.perceive.title": "看動作",
  "landing.stage.perceive.body":
    "姿態關鍵點和 VideoMAE 動作特徵讀取身體位置，再找出問題發生的時間點。",
  "landing.stage.retrieve.title": "找資料",
  "landing.stage.retrieve.body":
    "GraphRAG 會在健身知識圖譜中，從看得到的動作問題找出可能原因。",
  "landing.stage.reason.title": "判斷原因",
  "landing.stage.reason.body":
    "系統把看到的情況、可能原因和改善方式串起來，並以找到的資料為依據。",
  "landing.stage.coach.title": "給建議",
  "landing.stage.coach.body":
    "提供改善建議，並標出問題出現的畫面。",

  // Landing — diagnosis
  "landing.diagnosis.title": "每個建議都看得到原因。",
  "landing.diagnosis.sub":
    "從鏡頭看到的動作問題，一路帶到你可以怎麼練。",
  "landing.step.observation.tag": "看動作",
  "landing.step.observation.title": "觀察",
  "landing.step.observation.body":
    "在動作最低點，左膝向內越過腳掌，於第 96 到 118 影格被標記。",
  "landing.step.attribution.tag": "知識圖譜",
  "landing.step.attribution.title": "找原因",
  "landing.step.attribution.body":
    "系統把膝蓋內夾連結到髖部外側力量不足，並指出臀中肌可能是關鍵。",
  "landing.step.prescription.tag": "給建議",
  "landing.step.prescription.title": "怎麼改善",
  "landing.step.prescription.body":
    "讓膝蓋跟著腳尖方向往外推，並加上彈力帶高腳杯深蹲輔助練習。",
  "landing.frame.alt": "分析中的取樣影格",

  // Landing — bento
  "landing.bento.kicker": "技術細節",
  "landing.bento.title": "四種訊號都在裝置上分析，再整合成結果。",
  "landing.bento.pose.title": "姿態感知",
  "landing.bento.pose.body":
    "MediaPipe 和 RTMPose 追蹤 33 個身體關鍵點，像膝蓋內夾這類問題都能轉成系統看得懂的資訊。",
  "landing.bento.kg.title": "知識圖譜",
  "landing.bento.kg.body":
    "健身知識圖譜把動作問題、原因和改善方式串起來，讓建議有資料可追溯。",
  "landing.bento.rules.title": "看得懂的判定規則",
  "landing.bento.videomae.title": "VideoMAE 動作",
  "landing.bento.videomae.body":
    "影片中的動作變化能幫系統分辨穩定動作和細微問題。",

  // Landing — evaluation
  "landing.eval.title": "不只說得好聽，也要經得起檢驗。",
  "landing.eval.sub":
    "建議能不能解釋不夠，還要能驗證。x-coach 會檢查結果是否一致、建議是否有根據，以及使用者是否真的覺得有幫助。",
  "landing.eval.m1.label": "評分一致性",
  "landing.eval.m1.body":
    "和專家的排序比對，確認模型的判斷是否接近真人教練。",
  "landing.eval.m2.label": "建議是否有根據",
  "landing.eval.m2.body":
    "用 RAGAS 檢查每個說法是否都對得上找到的資料。",
  "landing.eval.m3.label": "實用性",
  "landing.eval.m3.body":
    "邀請不同程度的訓練者實際使用，評估骨架疊圖和文字建議是否好懂、有幫助。",

  // Landing — closing CTA
  "landing.cta.title": "看它分析一段真實深蹲。",
  "landing.cta.sub":
    "上傳或直接錄一段影片，看著骨架、錯誤與有據回饋一同呈現。",

  // Landing — footer
  "landing.footer.pipeline": "處理流程",
  "landing.footer.tagline": "看得懂原因的動作教練，研究原型。",

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
  "auth.back": "返回應用程式",
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
  "auth.lineBtn": "使用 LINE 繼續",

  // LIFF 內相機的備援提示 + /liff/diag 裝置檢測頁
  "camera.liffHint":
    "LINE 內建瀏覽器在此裝置無法開啟相機。請點右上角「⋯」選單,選擇「用瀏覽器開啟」再使用相機功能。",
  "diag.title": "LIFF 裝置檢測",
  "diag.subtitle": "請在手機的 LINE 內開啟本頁,以在正式導入 LIFF 前確認登入與相機支援程度。",
  "diag.env": "執行環境",
  "diag.session": "登入 Session",
  "diag.signedIn": "已登入",
  "diag.signedOut": "未登入",
  "diag.camera": "即時相機 (getUserMedia)",
  "diag.probeBtn": "測試相機",
  "diag.probing": "測試中…",
  "diag.cameraOk": "此瀏覽器可正常使用相機。",
  "diag.pose": "即時姿態(相機 + MediaPipe)",
  "diag.poseBtn": "測試相機＋姿態",
  "diag.poseGood": "流暢——此裝置可在 LIFF 內玩即時相機遊戲。",
  "diag.poseMarginal": "跑得動但偏慢——遊戲體驗會打折。",
  "diag.poseBad": "太慢——此裝置的即時相機功能建議留在外部瀏覽器。",
  "diag.poseNoLandmarks": "相機與模型都正常,但未偵測到人——請將相機對準自己再測一次。",
  "diag.capture": "影片檔案拍攝(點擊確認能否開啟相機錄影)",
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
  "nav.movements": "動作庫",
  "nav.settings": "設定",
  "account.signin": "登入",
  "account.signout": "登出",
  "account.menu": "帳號選單",
  "account.settings": "設定",
  "account.lineSigningIn": "LINE 登入中…",

  // Movements menu
  "movements.title": "動作庫",
  "movements.subtitle": "教練目前認識的所有動作。深蹲、伏地挺身與肩上推舉分析已經上線，其餘動作陸續開放。",
  "movements.groupLower": "下肢",
  "movements.groupUpper": "上肢",
  "movements.groupCore": "核心",
  "movements.groupFullBody": "全身",
  "movements.filterAll": "全部",
  "movements.searchPlaceholder": "搜尋動作",
  "movements.noMatch": "找不到符合的動作。",
  "movements.uploadCta": "上傳影片",
  "movements.analyze": "分析影片",
  "movements.soon": "即將開放",
  "movements.beta": "Beta",
  "movements.betaNote": "此動作的規則來自文獻推導，尚未以標註資料驗證。",

  // Movement display names.
  "movement.Squat": "深蹲",
  "movement.Lunge": "弓步蹲",
  "movement.Deadlift": "硬舉",
  "movement.Leg Abduction": "腿部外展",
  "movement.Shoulder Bridge": "臀橋",
  "movement.Push-up": "伏地挺身",
  "movement.Overhead Press": "肩上推舉",
  "movement.Row": "划船",
  "movement.Bicep Curl": "二頭彎舉",
  "movement.Band Pull Apart": "彈力帶擴胸",
  "movement.Arm Abduction": "手臂外展",
  "movement.Arm VW": "手臂 VW 開合",
  "movement.Sit-up": "仰臥起坐",
  "movement.Torso Twist": "軀幹旋轉",
  "movement.Jumping Jacks": "開合跳",
  "movement.High Knee": "高抬腿",

  // Settings popup
  "settings.title": "設定",
  "settings.nav": "設定分類",
  "settings.close": "關閉設定",
  "settings.search": "搜尋",
  "settings.searchEmpty": "找不到符合的設定。",
  "settings.groupSettings": "設定",
  "settings.groupCustomize": "自訂",
  "settings.general": "一般",
  "settings.preferences": "偏好設定",
  "settings.account": "帳號",
  "settings.avatar": "頭像",
  "settings.language": "語言",
  "settings.profile": "個人資料",
  "settings.name": "名稱",
  "settings.email": "電子郵件",
  "settings.provider": "登入方式",
  "settings.provider.google": "Google",
  "settings.provider.line": "LINE",
  "settings.provider.email": "電子郵件與密碼",
  "settings.model": "教練模型",
  "settings.modelDesc": "選擇回答你追問的 LLM 模型，套用於之後的新訊息。",
  "settings.modelDefault": "預設",
  "settings.modelLoading": "載入模型中…",
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
  "history.rowTitle": "{view} {movement}",
  "history.clean": "標準動作",
  "history.faultOne": "1 個錯誤",
  "history.faultMany": "{count} 個錯誤",
  "history.deleteAria": "刪除這筆紀錄",
  "history.deleteCta": "刪除",
  "history.deleteTitle": "刪除這筆紀錄？",
  "history.deleteDesc": "此操作無法復原。",
  "history.deleteConfirm": "刪除",
  "history.deleteCancel": "取消",
  "history.deleting": "刪除中…",
  "history.deleteError": "無法刪除這筆紀錄，請再試一次。",
  "history.uploadCta": "上傳影片",
  "history.statTotal": "總分析次數",
  "history.statCleanRate": "動作正確率",
  "history.statTopMovement": "最常訓練動作",
  "history.statLatest": "最近分析",
  "history.statTimes": "共 {count} 次分析",
  "history.statNone": "—",
  "history.statsScope": "下方比率取自最近 {loaded} 次分析（共 {total} 次）。",
  "history.searchPlaceholder": "搜尋動作",
  "history.filterMovement": "動作類型",
  "history.filterStatus": "分析結果",
  "history.filterRange": "時間範圍",
  "history.filterAllMovements": "全部動作",
  "history.filterAllResults": "全部結果",
  "history.filterToggle": "篩選",
  "history.filterToggleActive": "篩選（已套用 {count} 項）",
  "history.filterClean": "動作正確",
  "history.filterFaults": "需要調整",
  "history.rangeAll": "全部時間",
  "history.rangeToday": "今天",
  "history.range7": "近 7 天",
  "history.range30": "近 30 天",
  "history.clearFilters": "清除篩選",
  "history.noMatch": "沒有符合篩選的紀錄。",
  "history.today": "今天",
  "history.yesterday": "昨天",

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
  "admin.nav.line": "LINE",
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
  "admin.settings.maxUploadBytes": "單檔上傳上限（位元組）",
  "admin.settings.userStorageQuotaBytes": "每位使用者的儲存配額（位元組）",
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
  "admin.line.title": "LINE",
  "admin.line.desc": "LINE 整合狀態:連線、推播額度、Webhook 健康度與訊息發送統計。",
  "admin.line.loadError": "無法載入 LINE 狀態。",
  "admin.line.loginBridge": "LINE 登入橋接",
  "admin.line.bot": "LINE Bot",
  "admin.line.pushUsed": "本月推播已用",
  "admin.line.remaining": "剩餘免費額度",
  "admin.line.unreachable": "無法取得 LINE 額度。",
  "admin.line.noCapNote":
    "未在 LINE Official Account Manager 設定每月上限,無法計算剩餘。請在該後台設定每月訊息上限,即可追蹤免費額度。",
  "admin.line.oaName": "官方帳號",
  "admin.line.chatModeWarn": "聊天模式 — webhook 不會收到訊息事件",
  "admin.line.webhook": "Webhook",
  "admin.line.webhookActive": "運作中",
  "admin.line.webhookInactive": "未啟用",
  "admin.line.webhookTest": "測試 webhook",
  "admin.line.webhookTesting": "測試中…",
  "admin.line.webhookReachable": "連得到（{code}）",
  "admin.line.webhookFailed": "失敗（{code}：{reason}）",
  "admin.line.webhookTestError": "無法連到 LINE。",
  "admin.line.webhookTestUnauthorized": "LINE 不接受這組 access token，請確認或重新發行。",
  "admin.line.webhookTestRateLimited": "測試被 LINE 限流，請稍候再試。",
  "admin.line.webhookTestNoEndpoint": "這個 channel 還沒設定 webhook 網址。",
  "admin.line.webhookTestNotConfigured": "這台伺服器沒有設定 LINE Messaging。",
  "admin.line.botInfoUnavailable": "無法讀取官方帳號資訊。",
  "admin.line.webhookUnavailable": "無法讀取 webhook 設定 — 執行測試可以看出原因。",
  "admin.line.deliveryUnavailable": "無法讀取昨日發送統計。",
  "admin.line.replyYesterday": "昨日回覆",
  "admin.line.pushYesterday": "昨日推播",
  "admin.line.deliveryUnready": "資料尚未就緒",
  "admin.line.deliveryDate": "統計日期 {date}",

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

// Exported so the suite can enforce key-set PARITY between the locales. Both dicts are typed
// `Record<string, string>`, so TypeScript cannot catch a key added to one locale and forgotten in
// the other: `t()` falls back to returning the raw key, and the zh UI silently renders
// "metric.notMeasured" as literal text. That failure is invisible in review and in every
// English-locale test — see `lib.i18n.test.ts`.
export const DICTS: Record<Lang, Dict> = { en, "zh-Hant": zhHant };

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
// Canonical movement names ("Squat", "Overhead Press", ...) title-case cleanly, so movement.* keys
// are optional; the fallback renders the display text.
export const movementLabel = (t: TFunc, raw: string) => dataLabel(t, "movement", raw);
export const viewLabel = (t: TFunc, raw: string) => dataLabel(t, "view", raw);
export const phaseLabel = (t: TFunc, raw: string) => dataLabel(t, "phase", raw);

export function severityText(t: TFunc, sev: number): string {
  if (sev >= 0.75) return t("severity.high");
  if (sev >= 0.4) return t("severity.moderate");
  return t("severity.mild");
}
