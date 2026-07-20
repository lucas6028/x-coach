import { useCallback, useEffect, useRef, useState } from "react";
import AppLayout from "../components/AppLayout";
import BlastStartScreen from "../components/blast/BlastStartScreen";
import BlastHud from "../components/blast/BlastHud";
import BlastOverScreen, { type BlastResult } from "../components/blast/BlastOverScreen";
import { useI18n } from "../lib/i18n";
import { handState } from "../lib/blast/gestures";
import { ROUND_SECONDS } from "../lib/blast/scoring";
import { createGameState, stepFrame, BEAM_MS, type GameState } from "../lib/blast/engine";
import { loadLeaderboard, saveScore, type BlastEntry } from "../lib/blast/leaderboard";
import { estimateKcal, EFFORT } from "../lib/calories";
import { addCalories } from "../lib/calorieStore";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { createPoseLandmarker, drawScene } from "../components/blast/blastDetector";

type Phase = "intro" | "countdown" | "playing" | "over";

// Meme Blaster — charge a "Kamehameha" by bringing your hands together, then throw your
// arms apart to fire an energy beam that wipes out drifting meme orbs. React state drives
// the UI; a ref-backed rAF loop drives detection, physics, and scoring. All the mechanic
// reasoning lives in lib/blast/* (unit-tested) — this file wires it to the camera and owns
// only the impure edges: reading the frame, drawing the canvas, and throttling React state.
export default function MemeBlast() {
  const { t } = useI18n();

  const [phase, setPhase] = useState<Phase>("intro");
  const [leaderboard, setLeaderboard] = useState<BlastEntry[]>(() => loadLeaderboard());
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState(3);

  const [score, setScore] = useState(0);
  const [combo, setCombo] = useState(0);
  const [timeLeft, setTimeLeft] = useState(ROUND_SECONDS);
  const [charge, setCharge] = useState(0);
  const [armed, setArmed] = useState(false);
  const [flash, setFlash] = useState<{ hits: number; points: number } | null>(null);

  const [result, setResult] = useState<BlastResult>({ score: 0, hits: 0, bestCombo: 0, kcal: 0 });
  const [submitted, setSubmitted] = useState(false);
  const [rank, setRank] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);

  // Loop bookkeeping the rAF callback mutates without re-rendering; `engine` holds all the
  // pure game state advanced by stepFrame.
  const g = useRef({
    running: false,
    roundEnd: 0,
    last: 0,
    lastVideoTime: -1,
    lastUi: 0,
    engine: createGameState(0) as GameState,
  });

  const teardown = useCallback(() => {
    g.current.running = false;
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((tr) => tr.stop());
    streamRef.current = null;
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
  }, []);

  useEffect(() => teardown, [teardown]);

  const endRound = useCallback(() => {
    const e = g.current.engine;
    teardown();
    // The round always runs the full clock; orbs blasted are the movement signal. Record once here.
    const kcal = estimateKcal({ durationSec: ROUND_SECONDS, moves: e.hits, effort: EFFORT.blast });
    addCalories("blast", kcal);
    setResult({ score: e.score, hits: e.hits, bestCombo: e.bestCombo, kcal });
    setLeaderboard(loadLeaderboard());
    setSubmitted(false);
    setRank(null);
    setPhase("over");
  }, [teardown]);

  const tick = useCallback(() => {
    const s = g.current;
    if (!s.running) return;
    rafRef.current = requestAnimationFrame(tick);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker || video.readyState < 2) return;

    const now = performance.now();
    const dtMs = s.last ? now - s.last : 16;
    s.last = now;
    const secLeft = Math.max(0, Math.ceil((s.roundEnd - now) / 1000));

    // Detection only on a fresh camera frame (MediaPipe needs rising timestamps).
    let landmarks = null as ReturnType<PoseLandmarker["detectForVideo"]>["landmarks"][0] | null;
    if (video.currentTime !== s.lastVideoTime) {
      s.lastVideoTime = video.currentTime;
      landmarks = landmarker.detectForVideo(video, now).landmarks?.[0] ?? null;
    }

    // Advance all game state with one pure step.
    const hand = landmarks ? handState(landmarks) : { valid: false, gap: 0, aimY: 0.5 };
    s.engine = stepFrame(s.engine, { hand, dtMs, now, rng: Math.random });
    const e = s.engine;

    // Render.
    const ctx = canvas.getContext("2d");
    if (ctx) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
      }
      drawScene(
        ctx,
        {
          landmarks,
          targets: e.targets,
          charge: e.charge.charge,
          armed: e.charge.charge >= 1,
          beam: e.beam ? { y: e.beam.y, life: (e.beam.until - now) / BEAM_MS } : null,
        },
        canvas.width,
        canvas.height
      );
    }

    // Throttle React updates to ~20/s.
    if (now - s.lastUi > 50) {
      s.lastUi = now;
      setScore(e.score);
      setCombo(e.combo);
      setTimeLeft(secLeft);
      setCharge(e.charge.charge);
      setArmed(e.charge.charge >= 1);
      setFlash(e.flash);
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
        const now = performance.now();
        const s = g.current;
        s.running = true;
        s.roundEnd = now + ROUND_SECONDS * 1000;
        s.last = 0;
        s.lastVideoTime = -1;
        s.lastUi = 0;
        s.engine = createGameState(now);
        setScore(0);
        setCombo(0);
        setTimeLeft(ROUND_SECONDS);
        setCharge(0);
        setArmed(false);
        setFlash(null);
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
      setError(e instanceof Error ? e.message : t("blast.error"));
    }
  }, [beginCountdown, teardown, t]);

  const submit = useCallback((name: string) => {
    const entry: BlastEntry = {
      name,
      score: result.score,
      hits: result.hits,
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
    <AppLayout title={t("blast.title")}>
      {phase === "intro" && (
        <BlastStartScreen
          leaderboard={leaderboard}
          onStart={start}
          starting={starting}
          error={error}
        />
      )}

      {phase === "over" && (
        <BlastOverScreen
          result={result}
          leaderboard={leaderboard}
          rank={rank}
          submitted={submitted}
          onSubmit={submit}
          onReplay={replay}
        />
      )}

      <div className={`relative flex-1 bg-black ${showCamera ? "" : "hidden"}`}>
        <video
          ref={videoRef}
          muted
          playsInline
          className="absolute inset-0 h-full w-full -scale-x-100 object-contain"
        />
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full object-contain" />

        {phase === "countdown" && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/50">
            <span className="font-display text-8xl font-black text-white drop-shadow-lg">
              {countdown}
            </span>
          </div>
        )}

        {phase === "playing" && (
          <BlastHud
            score={score}
            combo={combo}
            timeLeft={timeLeft}
            charge={charge}
            armed={armed}
            flash={flash}
          />
        )}
      </div>
    </AppLayout>
  );
}
