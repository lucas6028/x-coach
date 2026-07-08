import { useCallback, useEffect, useRef, useState } from "react";
import AppLayout from "../components/AppLayout";
import BlastStartScreen from "../components/blast/BlastStartScreen";
import BlastHud from "../components/blast/BlastHud";
import BlastOverScreen, { type BlastResult } from "../components/blast/BlastOverScreen";
import { useI18n } from "../lib/i18n";
import { handState } from "../lib/blast/gestures";
import {
  stepCharge,
  initialCharge,
  DECAY_MS,
  type ChargeState,
} from "../lib/blast/charger";
import {
  advanceTargets,
  beamHits,
  makeTarget,
  MEME_EMOJIS,
  type Target,
} from "../lib/blast/targets";
import { blastPoints, difficulty, ROUND_SECONDS } from "../lib/blast/scoring";
import { loadLeaderboard, saveScore, type BlastEntry } from "../lib/blast/leaderboard";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { createPoseLandmarker, drawScene } from "../components/blast/blastDetector";

type Phase = "intro" | "countdown" | "playing" | "over";

const BEAM_MS = 220;
const FLASH_MS = 700;

// Meme Blaster — charge a "Kamehameha" by bringing your hands together, then throw your
// arms apart to fire an energy beam that wipes out drifting meme orbs. React state drives
// the UI; a ref-backed rAF loop drives detection, physics, and scoring. All the mechanic
// reasoning lives in lib/blast/* (unit-tested); this file wires it to the camera.
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

  const [result, setResult] = useState<BlastResult>({ score: 0, hits: 0, bestCombo: 0 });
  const [submitted, setSubmitted] = useState(false);
  const [rank, setRank] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);

  // Mutable game state the loop mutates without re-rendering.
  const g = useRef({
    running: false,
    roundStart: 0,
    roundEnd: 0,
    last: 0,
    lastVideoTime: -1,
    lastUi: 0,
    charge: initialCharge as ChargeState,
    targets: [] as Target[],
    nextId: 1,
    spawnAt: 0,
    score: 0,
    combo: 0,
    bestCombo: 0,
    hits: 0,
    beam: null as { y: number; until: number } | null,
    flash: null as { hits: number; points: number } | null,
    flashUntil: 0,
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
    const s = g.current;
    teardown();
    setResult({ score: s.score, hits: s.hits, bestCombo: s.bestCombo });
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
    const dt = dtMs / 1000;
    s.last = now;
    const secLeft = Math.max(0, Math.ceil((s.roundEnd - now) / 1000));

    // Detection only on a fresh camera frame (MediaPipe needs rising timestamps).
    let landmarks = null as ReturnType<PoseLandmarker["detectForVideo"]>["landmarks"][0] | null;
    if (video.currentTime !== s.lastVideoTime) {
      s.lastVideoTime = video.currentTime;
      landmarks = landmarker.detectForVideo(video, now).landmarks?.[0] ?? null;
    }

    // Charge / fire.
    const hs = landmarks ? handState(landmarks) : { valid: false, gap: 0, aimY: 0.5 };
    if (hs.valid) {
      const stepped = stepCharge(s.charge, hs.gap, dtMs);
      s.charge = stepped.state;
      if (stepped.fired) {
        const { hit, remaining } = beamHits(s.targets, hs.aimY);
        if (hit.length > 0) {
          s.combo += 1;
          s.bestCombo = Math.max(s.bestCombo, s.combo);
          s.hits += hit.length;
          const pts = blastPoints(hit.length, s.combo);
          s.score += pts;
          s.targets = remaining;
          s.flash = { hits: hit.length, points: pts };
        } else {
          s.combo = 0;
          s.flash = { hits: 0, points: 0 };
        }
        s.beam = { y: hs.aimY, until: now + BEAM_MS };
        s.flashUntil = now + FLASH_MS;
      }
    } else {
      s.charge = { charge: Math.max(0, s.charge.charge - dtMs / DECAY_MS) };
    }

    // Spawn + advance orbs, ramping with round progress.
    const frac = (now - s.roundStart) / (ROUND_SECONDS * 1000);
    const diff = difficulty(frac);
    if (now >= s.spawnAt) {
      const y = 0.15 + Math.random() * 0.7;
      const emoji = MEME_EMOJIS[Math.floor(Math.random() * MEME_EMOJIS.length)];
      s.targets.push(makeTarget(s.nextId++, y, diff.speed, emoji));
      s.spawnAt = now + diff.spawnMs;
    }
    s.targets = advanceTargets(s.targets, dt).targets;

    if (s.beam && now >= s.beam.until) s.beam = null;

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
          targets: s.targets,
          charge: s.charge.charge,
          armed: s.charge.charge >= 1,
          beam: s.beam ? { y: s.beam.y, life: (s.beam.until - now) / BEAM_MS } : null,
        },
        canvas.width,
        canvas.height
      );
    }

    // Throttle React updates to ~20/s.
    if (now - s.lastUi > 50) {
      s.lastUi = now;
      setScore(s.score);
      setCombo(s.combo);
      setTimeLeft(secLeft);
      setCharge(s.charge.charge);
      setArmed(s.charge.charge >= 1);
      setFlash(s.flashUntil && now < s.flashUntil ? s.flash : null);
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
        s.roundStart = now;
        s.roundEnd = now + ROUND_SECONDS * 1000;
        s.last = 0;
        s.lastVideoTime = -1;
        s.lastUi = 0;
        s.charge = initialCharge;
        s.targets = [];
        s.nextId = 1;
        s.spawnAt = now;
        s.score = 0;
        s.combo = 0;
        s.bestCombo = 0;
        s.hits = 0;
        s.beam = null;
        s.flash = null;
        s.flashUntil = 0;
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
