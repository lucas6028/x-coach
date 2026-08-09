import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Crosshair, FloppyDisk, MaskHappy, Trophy } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import AppLayout from "../components/AppLayout";
import { useI18n } from "../lib/i18n";
import { LM } from "../lib/pose";
import { CameraError, getCameraStream } from "../lib/camera";
import { isInLiffClient } from "../lib/liff";
import { waitForVideoFrame } from "../lib/videoFrame";
import { createLivePoseSchedule, shouldRunLivePoseInference } from "../lib/livePoseScheduler";
import { addCalories } from "../lib/calorieStore";
import { EFFORT, estimateKcal } from "../lib/calories";
import {
  ROUND_SECONDS,
  advanceWorld,
  createWebGameState,
  detectWebFlick,
  fireWeb,
  type Point,
} from "../lib/webslinger/engine";
import {
  loadLeaderboard,
  saveScore,
  type WebSlingerEntry,
} from "../lib/webslinger/leaderboard";
import {
  createPoseLandmarker,
  drawWebSlingerScene,
} from "../components/webslinger/webSlingerDetector";

type Phase = "intro" | "countdown" | "playing" | "over";
const WRISTS = [LM.LEFT_WRIST, LM.RIGHT_WRIST];
const ELBOWS = [LM.LEFT_ELBOW, LM.RIGHT_ELBOW];
const SHOT_COOLDOWN_MS = 280;

export default function WebSlinger() {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const [phase, setPhase] = useState<Phase>("intro");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState(3);
  const [score, setScore] = useState(0);
  const [combo, setCombo] = useState(0);
  const [timeLeft, setTimeLeft] = useState(ROUND_SECONDS);
  const [hits, setHits] = useState(0);
  const [result, setResult] = useState({ score: 0, bestCombo: 0, hits: 0, kcal: 0 });
  const [leaderboard, setLeaderboard] = useState<WebSlingerEntry[]>(() => loadLeaderboard());
  const [name, setName] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [rank, setRank] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  const game = useRef({
    running: false,
    roundEnd: 0,
    lastFrameAt: 0,
    lastDetectionAt: 0,
    lastUiAt: 0,
    poseSchedule: createLivePoseSchedule(),
    state: createWebGameState(0),
    previousWrists: [null, null] as (Point | null)[],
    wrists: [null, null] as (Point | null)[],
    lastShotAt: [0, 0],
  });

  const teardown = useCallback(() => {
    game.current.running = false;
    cancelAnimationFrame(rafRef.current);
    if (countdownRef.current) clearInterval(countdownRef.current);
    countdownRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      teardown();
    };
  }, [teardown]);

  const endRound = useCallback(() => {
    const state = game.current.state;
    teardown();
    const kcal = estimateKcal({
      durationSec: ROUND_SECONDS,
      moves: state.hits,
      effort: EFFORT.webslinger,
    });
    addCalories("webslinger", kcal);
    setResult({ score: state.score, bestCombo: state.bestCombo, hits: state.hits, kcal });
    setLeaderboard(loadLeaderboard());
    setName("");
    setSubmitted(false);
    setRank(null);
    setPhase("over");
  }, [teardown]);

  const tick = useCallback(() => {
    const current = game.current;
    if (!current.running) return;
    rafRef.current = requestAnimationFrame(tick);
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker || video.readyState < 2) return;

    const now = performance.now();
    const dtMs = current.lastFrameAt ? now - current.lastFrameAt : 33;
    current.lastFrameAt = now;
    current.state = advanceWorld(current.state, dtMs, now);
    const seconds = Math.max(0, Math.ceil((current.roundEnd - now) / 1000));

    if (shouldRunLivePoseInference(current.poseSchedule, video.currentTime, now)) {
      const detectionDtMs = current.lastDetectionAt ? now - current.lastDetectionAt : 33;
      current.lastDetectionAt = now;
      const landmarks = landmarker.detectForVideo(video, now).landmarks?.[0] ?? null;
      WRISTS.forEach((wristIndex, hand) => {
        const wristLandmark = landmarks?.[wristIndex];
        const elbowLandmark = landmarks?.[ELBOWS[hand]];
        const wrist =
          wristLandmark && (wristLandmark.visibility ?? 1) >= 0.5
            ? { x: wristLandmark.x, y: wristLandmark.y }
            : null;
        const elbow =
          elbowLandmark && (elbowLandmark.visibility ?? 1) >= 0.5
            ? { x: elbowLandmark.x, y: elbowLandmark.y }
            : null;
        current.wrists[hand] = wrist;
        if (wrist && elbow && now - current.lastShotAt[hand] >= SHOT_COOLDOWN_MS) {
          const ray = detectWebFlick(elbow, wrist, current.previousWrists[hand], detectionDtMs);
          if (ray) {
            current.state = fireWeb(current.state, ray);
            current.lastShotAt[hand] = now;
          }
        }
        current.previousWrists[hand] = wrist;
      });
    }

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    }
    const context = canvas.getContext("2d");
    if (context) {
      drawWebSlingerScene(
        context,
        { targets: current.state.targets, traces: current.state.traces, wrists: current.wrists },
        canvas.width,
        canvas.height
      );
    }

    if (now - current.lastUiAt > 50) {
      current.lastUiAt = now;
      setScore(current.state.score);
      setCombo(current.state.combo);
      setHits(current.state.hits);
      setTimeLeft(seconds);
    }
    if (seconds <= 0) endRound();
  }, [endRound]);

  const beginCountdown = useCallback(() => {
    setPhase("countdown");
    let next = 3;
    setCountdown(next);
    const interval = setInterval(() => {
      next -= 1;
      if (next <= 0) {
        clearInterval(interval);
        countdownRef.current = null;
        const now = performance.now();
        const current = game.current;
        current.running = true;
        current.roundEnd = now + ROUND_SECONDS * 1000;
        current.lastFrameAt = 0;
        current.lastDetectionAt = 0;
        current.lastUiAt = 0;
        current.poseSchedule = createLivePoseSchedule();
        current.state = createWebGameState(now);
        current.previousWrists = [null, null];
        current.wrists = [null, null];
        current.lastShotAt = [0, 0];
        setScore(0);
        setCombo(0);
        setHits(0);
        setTimeLeft(ROUND_SECONDS);
        setPhase("playing");
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setCountdown(next);
      }
    }, 800);
    countdownRef.current = interval;
  }, [tick]);

  const start = useCallback(async () => {
    setError("");
    setStarting(true);
    try {
      const landmarkerPromise = createPoseLandmarker();
      const stream = await getCameraStream({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });
      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        landmarkerPromise.then((model) => model.close()).catch(() => {});
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }
      const landmarker = await landmarkerPromise;
      if (!mountedRef.current) {
        landmarker.close();
        teardown();
        return;
      }
      landmarkerRef.current = landmarker;
      if (video) {
        await waitForVideoFrame(video);
        if (video.readyState >= 2) {
          try {
            landmarker.detectForVideo(video, performance.now());
          } catch (warmupError) {
            console.error("web slinger: warmup inference failed", warmupError);
          }
        }
      }
      setStarting(false);
      beginCountdown();
    } catch (startError) {
      teardown();
      setStarting(false);
      const liffCameraFailed = startError instanceof CameraError && (await isInLiffClient());
      setError(
        liffCameraFailed
          ? t("camera.liffHint")
          : startError instanceof Error
            ? startError.message
            : t("web.error")
      );
    }
  }, [beginCountdown, t, teardown]);

  const submit = () => {
    if (submitted) return;
    const saved = saveScore({
      name: name.trim() || t("web.over.anon"),
      score: result.score,
      bestCombo: result.bestCombo,
      ts: Date.now(),
    });
    setLeaderboard(saved.board);
    setRank(saved.rank > 0 ? saved.rank : null);
    setSubmitted(true);
  };

  const replay = () => {
    setLeaderboard(loadLeaderboard());
    setPhase("intro");
  };

  const showCamera = phase === "playing" || phase === "countdown";

  return (
    <AppLayout initialSidebarOpen={false}>
      {phase === "intro" && (
        <div className="flex-1 overflow-y-auto bg-surface scrollbar-thin">
          <motion.div
            initial={reduce ? false : { opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            className="mx-auto grid min-h-full max-w-5xl grid-cols-1 gap-10 px-5 py-8 sm:px-6 sm:py-12 lg:grid-cols-[1fr_20rem] lg:items-center lg:gap-16"
          >
            <section>
              <span className="inline-flex items-center gap-2 rounded-full border border-rose-400/30 bg-rose-500/10 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-rose-300">
                <MaskHappy size={14} weight="fill" />
                {t("web.badge")}
              </span>
              <h1 className="mt-4 max-w-xl font-display text-4xl font-black leading-[0.98] tracking-tight text-content sm:text-5xl">
                {t("web.heading")}
              </h1>
              <p className="mt-4 max-w-lg leading-relaxed text-muted">{t("web.sub")}</p>
              <ol className="mt-7 grid gap-3 text-sm text-muted">
                {["web.how1", "web.how2", "web.how3"].map((key, index) => (
                  <li key={key} className="flex items-start gap-3">
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-rose-500/15 font-bold text-rose-300">
                      {index + 1}
                    </span>
                    <span>{t(key, { s: ROUND_SECONDS })}</span>
                  </li>
                ))}
              </ol>
              <button
                onClick={start}
                disabled={starting}
                className="mt-7 inline-flex items-center gap-2 rounded-2xl bg-rose-600 px-6 py-3.5 font-semibold text-white transition-transform hover:-translate-y-0.5 active:scale-[0.98] disabled:opacity-60"
              >
                <Camera size={19} weight="duotone" />
                {starting ? t("web.starting") : t("web.startBtn")}
              </button>
              <p className="mt-2 text-xs text-faint">{t("web.cameraNote")}</p>
              {error && (
                <p className="mt-4 rounded-2xl border border-danger/30 bg-danger/[0.06] p-3 text-sm text-danger">
                  {error}
                </p>
              )}
            </section>

            <aside className="rounded-2xl border border-border-dark bg-surface-dark p-5">
              <h2 className="flex items-center gap-2 font-display text-lg font-bold text-content">
                <Trophy size={19} className="text-rose-300" weight="duotone" />
                {t("web.board.title")}
              </h2>
              {leaderboard.length === 0 ? (
                <p className="mt-5 text-sm text-muted">{t("web.board.empty")}</p>
              ) : (
                <ol className="mt-4 space-y-3">
                  {leaderboard.slice(0, 5).map((entry, index) => (
                    <li key={`${entry.ts}-${entry.name}`} className="flex items-center gap-3 text-sm">
                      <span className="w-5 font-mono text-faint">{index + 1}</span>
                      <span className="min-w-0 flex-1 truncate text-muted">{entry.name}</span>
                      <strong className="tabular-nums text-content">{entry.score}</strong>
                    </li>
                  ))}
                </ol>
              )}
            </aside>
          </motion.div>
        </div>
      )}

      {phase === "over" && (
        <div className="flex flex-1 items-center justify-center overflow-y-auto bg-surface px-5 py-10">
          <motion.div
            initial={reduce ? false : { opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-lg rounded-2xl border border-border-dark bg-surface-dark p-6 text-center sm:p-8"
          >
            <MaskHappy size={42} weight="fill" className="mx-auto text-rose-400" />
            <h1 className="mt-3 font-display text-3xl font-black text-content">{t("web.over.title")}</h1>
            <p className="mt-4 font-display text-6xl font-black tabular-nums text-rose-400">{result.score}</p>
            <p className="mt-2 text-sm text-muted">
              {t("web.over.stats", { hits: result.hits, combo: result.bestCombo })}
            </p>
            <p className="mt-1 text-xs text-faint">{t("game.kcal.est", { n: result.kcal })}</p>
            {!submitted ? (
              <div className="mt-6 flex gap-2">
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  aria-label={t("web.over.nameLabel")}
                  placeholder={t("web.over.namePlaceholder")}
                  maxLength={20}
                  className="min-w-0 flex-1 rounded-xl border border-border-dark bg-content/[0.04] px-4 py-3 text-content outline-none placeholder:text-faint focus:border-rose-400"
                />
                <button
                  onClick={submit}
                  className="inline-flex items-center gap-2 rounded-xl bg-rose-600 px-4 py-3 font-semibold text-white active:scale-[0.98]"
                >
                  <FloppyDisk size={18} />
                  {t("web.over.save")}
                </button>
              </div>
            ) : (
              <p className="mt-6 text-sm font-semibold text-rose-300">
                {rank ? t("web.over.ranked", { rank }) : t("web.over.saved")}
              </p>
            )}
            <button
              onClick={replay}
              className="mt-5 w-full rounded-xl border border-border-dark px-5 py-3 font-semibold text-content transition-colors hover:bg-content/[0.05] active:scale-[0.98]"
            >
              {t("web.over.replay")}
            </button>
          </motion.div>
        </div>
      )}

      <div className={`relative flex-1 overflow-hidden bg-zinc-950 ${showCamera ? "" : "hidden"}`}>
        <video ref={videoRef} muted playsInline className="absolute inset-0 h-full w-full -scale-x-100 object-contain" />
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full object-contain" />
        {phase === "countdown" && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/60">
            <span className="font-display text-8xl font-black text-white">{countdown}</span>
          </div>
        )}
        {phase === "playing" && (
          <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-3 p-4 sm:p-6">
            <div className="rounded-2xl border border-white/15 bg-zinc-950/70 px-4 py-3 text-white backdrop-blur-md">
              <p className="text-xs font-semibold text-white/65">{t("web.hud.score")}</p>
              <p className="font-display text-3xl font-black tabular-nums">{score}</p>
            </div>
            {combo > 1 && (
              <div className="rounded-2xl bg-rose-600 px-4 py-3 text-center text-white">
                <p className="font-display text-xl font-black">{t("web.hud.combo", { n: combo })}</p>
              </div>
            )}
            <div className="rounded-2xl border border-white/15 bg-zinc-950/70 px-4 py-3 text-right text-white backdrop-blur-md">
              <p className="text-xs font-semibold text-white/65">{t("web.hud.time")}</p>
              <p className="font-display text-3xl font-black tabular-nums">{timeLeft}</p>
            </div>
          </div>
        )}
        {phase === "playing" && hits === 0 && (
          <div className="pointer-events-none absolute inset-x-0 bottom-6 flex justify-center px-4">
            <p className="flex items-center gap-2 rounded-2xl border border-white/15 bg-zinc-950/75 px-4 py-2 text-sm font-semibold text-white backdrop-blur-md">
              <Crosshair size={18} />
              {t("web.hud.hint")}
            </p>
          </div>
        )}
      </div>
    </AppLayout>
  );
}
