import { useCallback, useEffect, useRef, useState } from "react";
import AppLayout from "../components/AppLayout";
import GameStartScreen from "../components/game/GameStartScreen";
import GameHud from "../components/game/GameHud";
import GameOverScreen, { type RoundResult } from "../components/game/GameOverScreen";
import { useI18n } from "../lib/i18n";
import { poseSignature } from "../lib/game/angles";
import { POSES, scorePose, poseById, type GamePose } from "../lib/game/poses";
import { HIT_THRESHOLD, HOLD_MS, ROUND_SECONDS, type Grade } from "../lib/game/scoring";
import { createGameState, stepGame, type GameState } from "../lib/game/engine";
import {
  loadLeaderboard,
  saveScore,
  type ScoreEntry,
} from "../lib/game/leaderboard";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { createPoseLandmarker, drawSkeleton } from "../components/game/poseDetector";

type Phase = "intro" | "countdown" | "playing" | "over";

// Pose Match Rush — the live camera mini-game. React state drives the UI; a ref-backed
// rAF loop drives detection + scoring so the hot path never waits on re-renders. All the
// scoring/leaderboard reasoning lives in lib/game/* (unit-tested) — this file wires it to
// the camera, canvas, and MediaPipe, and owns only those impure edges.
export default function PoseGame() {
  const { t } = useI18n();

  const [phase, setPhase] = useState<Phase>("intro");
  const [leaderboard, setLeaderboard] = useState<ScoreEntry[]>(() => loadLeaderboard());
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState(3);

  // UI-facing snapshots (throttled from the loop).
  const [score, setScore] = useState(0);
  const [combo, setCombo] = useState(0);
  const [timeLeft, setTimeLeft] = useState(ROUND_SECONDS);
  const [target, setTarget] = useState<GamePose>(POSES[0]);
  const [quality, setQuality] = useState(0);
  const [holdProgress, setHoldProgress] = useState(0);
  const [lastGrade, setLastGrade] = useState<Grade | null>(null);

  const [result, setResult] = useState<RoundResult>({ score: 0, poses: 0, bestCombo: 0 });
  const [submitted, setSubmitted] = useState(false);
  const [rank, setRank] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);

  // Loop bookkeeping the rAF callback mutates without re-rendering; `game` holds the pure
  // scoring state advanced by stepGame.
  const loop = useRef({
    running: false,
    roundEnd: 0,
    lastVideoTime: -1,
    lastUi: 0,
    game: createGameState() as GameState,
  });

  const teardown = useCallback(() => {
    loop.current.running = false;
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((tr) => tr.stop());
    streamRef.current = null;
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
  }, []);

  // Stop everything when leaving the page.
  useEffect(() => teardown, [teardown]);

  const endRound = useCallback(() => {
    const game = loop.current.game;
    teardown();
    setResult({ score: game.score, poses: game.poses, bestCombo: game.bestCombo });
    setLeaderboard(loadLeaderboard());
    setSubmitted(false);
    setRank(null);
    setPhase("over");
  }, [teardown]);

  const tick = useCallback(() => {
    const g = loop.current;
    if (!g.running) return;
    rafRef.current = requestAnimationFrame(tick);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker || video.readyState < 2) return;

    const now = performance.now();
    const secLeft = Math.max(0, Math.ceil((g.roundEnd - now) / 1000));

    let liveQuality = 0;
    // Only run detection on a fresh camera frame (MediaPipe needs rising timestamps).
    if (video.currentTime !== g.lastVideoTime) {
      g.lastVideoTime = video.currentTime;
      const res = landmarker.detectForVideo(video, now);
      const lm = res.landmarks?.[0];

      const ctx = canvas.getContext("2d");
      if (ctx) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      const target = poseById(g.game.targetId) ?? POSES[0];
      if (lm) {
        liveQuality = scorePose(poseSignature(lm), target).score;
        if (ctx) drawSkeleton(ctx, lm, canvas.width, canvas.height, liveQuality >= HIT_THRESHOLD);
      }

      // Advance scoring with one pure step; a returned grade means a pose just locked in.
      const { state, grade } = stepGame(g.game, {
        quality: liveQuality,
        hasLandmarks: !!lm,
        now,
        rng: Math.random,
      });
      g.game = state;
      if (grade) {
        setLastGrade(grade);
        setTarget(poseById(state.targetId) ?? POSES[0]);
      }
    }

    // Throttle React updates to ~15/s; the loop itself runs at display rate.
    if (now - g.lastUi > 66) {
      g.lastUi = now;
      const game = g.game;
      setScore(game.score);
      setCombo(game.combo);
      setTimeLeft(secLeft);
      setQuality(liveQuality);
      setHoldProgress(game.holdStart ? Math.min(1, (now - game.holdStart) / HOLD_MS) : 0);
      if (game.gradeUntil && now > game.gradeUntil) {
        g.game = { ...game, gradeUntil: 0 };
        setLastGrade(null);
      }
    }

    if (secLeft <= 0) endRound();
  }, [endRound]);

  const beginCountdown = useCallback(() => {
    setPhase("countdown");
    let n = 3;
    setCountdown(n);
    const iv = setInterval(() => {
      n -= 1;
      if (n <= 0) {
        clearInterval(iv);
        const g = loop.current;
        g.running = true;
        g.roundEnd = performance.now() + ROUND_SECONDS * 1000;
        g.lastVideoTime = -1;
        g.lastUi = 0;
        g.game = createGameState();
        setScore(0);
        setCombo(0);
        setTimeLeft(ROUND_SECONDS);
        setTarget(POSES[0]);
        setLastGrade(null);
        setPhase("playing");
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setCountdown(n);
      }
    }, 800);
  }, [tick]);

  const start = useCallback(async () => {
    setError("");
    setStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }
      landmarkerRef.current = await createPoseLandmarker();
      setStarting(false);
      beginCountdown();
    } catch (e) {
      teardown();
      setStarting(false);
      setError(e instanceof Error ? e.message : t("game.error"));
    }
  }, [beginCountdown, teardown, t]);

  const submit = useCallback((name: string) => {
    const entry: ScoreEntry = {
      name,
      score: result.score,
      poses: result.poses,
      bestCombo: result.bestCombo,
      ts: Date.now(),
    };
    const { board, rank: r } = saveScore(entry);
    setLeaderboard(board);
    setRank(r > 0 ? r : null);
    setSubmitted(true);
  }, [result]);

  const replay = useCallback(() => {
    setLeaderboard(loadLeaderboard());
    setPhase("intro");
  }, []);

  const showCamera = phase === "playing" || phase === "countdown";

  return (
    <AppLayout title={t("game.title")}>
      {phase === "intro" && (
        <GameStartScreen
          leaderboard={leaderboard}
          onStart={start}
          starting={starting}
          error={error}
        />
      )}

      {phase === "over" && (
        <GameOverScreen
          result={result}
          leaderboard={leaderboard}
          rank={rank}
          submitted={submitted}
          onSubmit={submit}
          onReplay={replay}
        />
      )}

      {/* The camera stage stays mounted for countdown + play so the video keeps streaming. */}
      <div className={`relative flex-1 bg-black ${showCamera ? "" : "hidden"}`}>
        <video
          ref={videoRef}
          muted
          playsInline
          className="absolute inset-0 h-full w-full -scale-x-100 object-contain"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full object-contain"
        />

        {phase === "countdown" && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/50">
            <span className="font-display text-8xl font-black text-white drop-shadow-lg">
              {countdown}
            </span>
          </div>
        )}

        {phase === "playing" && (
          <GameHud
            score={score}
            combo={combo}
            timeLeft={timeLeft}
            target={target}
            quality={quality}
            holdProgress={holdProgress}
            lastGrade={lastGrade}
          />
        )}
      </div>
    </AppLayout>
  );
}
