import { useCallback, useEffect, useRef, useState } from "react";
import AppLayout from "../components/AppLayout";
import NinjaStartScreen from "../components/ninja/NinjaStartScreen";
import NinjaHud from "../components/ninja/NinjaHud";
import NinjaOverScreen, { type NinjaResult } from "../components/ninja/NinjaOverScreen";
import { useI18n } from "../lib/i18n";
import { LM } from "../lib/pose";
import { createGameState, stepGame, type GameState } from "../lib/ninja/engine";
import { isSwipe, type Blade } from "../lib/ninja/slice";
import { spawnPieces, advancePieces, type Piece } from "../lib/ninja/pieces";
import { loadLeaderboard, saveScore, type NinjaEntry } from "../lib/ninja/leaderboard";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { createPoseLandmarker, drawScene, type Point } from "../components/ninja/ninjaDetector";

type Phase = "intro" | "countdown" | "playing" | "over";

const WRISTS = [LM.LEFT_WRIST, LM.RIGHT_WRIST];
const TRAIL_LEN = 7;
// Exponential smoothing on the wrist position (fraction of the way toward the raw landmark each
// frame). Damps MediaPipe jitter before it can masquerade as a swipe, without lagging a real one.
const SMOOTH = 0.6;
const BOMB_MS = 600;
// Linger on the frozen board briefly after game over so the last cut / bomb blast reads.
const OVER_HOLD_MS = 600;

// Fruit Ninja — slice flying fruit with your hands via MediaPipe pose (both wrists are blades).
// Miss three fruits or hit a bomb and it's over. React state drives the UI; a ref-backed rAF loop
// drives detection, blade building, and physics. All the game reasoning lives in lib/ninja/*
// (unit-tested); this file wires it to the camera and owns only the impure edges.
export default function FruitNinja() {
  const { t } = useI18n();

  const [phase, setPhase] = useState<Phase>("intro");
  const [leaderboard, setLeaderboard] = useState<NinjaEntry[]>(() => loadLeaderboard());
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState(3);

  const [score, setScore] = useState(0);
  const [combo, setCombo] = useState(0);
  const [lives, setLives] = useState(3);
  const [pop, setPop] = useState(0);
  const [bombFlash, setBombFlash] = useState(false);

  const [result, setResult] = useState<NinjaResult>({ score: 0, bestCombo: 0, bombed: false });
  const [submitted, setSubmitted] = useState(false);
  const [rank, setRank] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);

  // Loop bookkeeping the rAF callback mutates without re-rendering; `game` is the pure state.
  const g = useRef({
    running: false,
    last: 0,
    lastVideoTime: -1,
    lastUi: 0,
    game: createGameState(0) as GameState,
    prev: [null, null] as (Point | null)[],
    trails: [[], []] as Point[][],
    pieces: [] as Piece[],
    nextPieceId: 1,
    popCount: 0,
    bombUntil: 0,
    bombed: false,
    overAt: 0,
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
    setResult({ score: s.game.score, bestCombo: s.game.bestCombo, bombed: s.bombed });
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

    if (video.currentTime !== s.lastVideoTime) {
      s.lastVideoTime = video.currentTime;
      // dt spans the interval between *detection* frames (camera fps), not rAF ticks. Measuring it
      // here — where the wrist positions actually update — keeps wrist speed and physics on the
      // same clock; measuring across rAF inflated speed ~2× and let a still hand's jitter slice.
      const dtMs = s.last ? now - s.last : 16;
      s.last = now;
      const dt = dtMs / 1000;

      const lm = landmarker.detectForVideo(video, now).landmarks?.[0] ?? null;

      // Build a blade for each wrist that genuinely swiped since last frame; keep a short trail.
      const blades: Blade[] = [];
      WRISTS.forEach((idx, hand) => {
        const p = lm?.[idx];
        const raw = p && (p.visibility ?? 1) >= 0.5 ? { x: p.x, y: p.y } : null;
        const prev = s.prev[hand];
        // Smooth toward the raw landmark to damp jitter before speed/distance are judged.
        const cur = raw
          ? prev
            ? { x: prev.x + SMOOTH * (raw.x - prev.x), y: prev.y + SMOOTH * (raw.y - prev.y) }
            : raw
          : null;
        if (cur && prev && isSwipe(prev, cur, dt)) {
          blades.push({ x1: prev.x, y1: prev.y, x2: cur.x, y2: cur.y });
        }
        s.prev[hand] = cur;
        if (cur) {
          s.trails[hand].push(cur);
          if (s.trails[hand].length > TRAIL_LEN) s.trails[hand].shift();
        } else {
          s.trails[hand] = [];
        }
      });

      if (!s.game.over) {
        const stepped = stepGame(s.game, { blades, dtMs, now, rng: Math.random });
        s.game = stepped.state;
        if (stepped.sliceFlash > 0) s.popCount += 1;
        // Burst each cut fruit into two halves flying apart along the swing.
        if (stepped.slicedFruits.length > 0) {
          let dx = 0;
          let dy = 0;
          blades.forEach((b) => {
            dx += b.x2 - b.x1;
            dy += b.y2 - b.y1;
          });
          for (const f of stepped.slicedFruits) {
            const burst = spawnPieces(s.nextPieceId, f, dx, dy, Math.random);
            s.pieces.push(...burst.pieces);
            s.nextPieceId = burst.nextId;
          }
        }
        if (stepped.bombFlash) {
          s.bombUntil = now + BOMB_MS;
          s.bombed = true;
        }
        if (s.game.over && s.overAt === 0) s.overAt = now;
      }

      // Advance the flying halves every detection frame, regardless of game-over, so the last
      // slice's pieces finish their arc.
      s.pieces = advancePieces(s.pieces, dt);

      const ctx = canvas.getContext("2d");
      if (ctx) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
        }
        drawScene(
          ctx,
          {
            entities: s.game.entities,
            pieces: s.pieces,
            trails: s.trails,
            bombFlash: s.bombUntil > now ? (s.bombUntil - now) / BOMB_MS : null,
          },
          canvas.width,
          canvas.height
        );
      }
    }

    if (now - s.lastUi > 50) {
      s.lastUi = now;
      setScore(s.game.score);
      setCombo(s.game.combo);
      setLives(s.game.lives);
      setPop(s.popCount);
      setBombFlash(s.bombUntil > now);
    }

    if (s.game.over && now - s.overAt >= OVER_HOLD_MS) endRound();
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
        s.last = 0;
        s.lastVideoTime = -1;
        s.lastUi = 0;
        s.game = createGameState(now);
        s.prev = [null, null];
        s.trails = [[], []];
        s.pieces = [];
        s.nextPieceId = 1;
        s.popCount = 0;
        s.bombUntil = 0;
        s.bombed = false;
        s.overAt = 0;
        setScore(0);
        setCombo(0);
        setLives(3);
        setPop(0);
        setBombFlash(false);
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
      setError(e instanceof Error ? e.message : t("ninja.error"));
    }
  }, [beginCountdown, teardown, t]);

  const submit = useCallback(
    (name: string) => {
      const entry: NinjaEntry = {
        name,
        score: result.score,
        bestCombo: result.bestCombo,
        ts: Date.now(),
      };
      const { board, rank: r } = saveScore(entry);
      setLeaderboard(board);
      setRank(r > 0 ? r : null);
      setSubmitted(true);
    },
    [result]
  );

  const replay = useCallback(() => {
    setLeaderboard(loadLeaderboard());
    setPhase("intro");
  }, []);

  const showCamera = phase === "playing" || phase === "countdown";

  return (
    <AppLayout title={t("ninja.title")}>
      {phase === "intro" && (
        <NinjaStartScreen
          leaderboard={leaderboard}
          onStart={start}
          starting={starting}
          error={error}
        />
      )}

      {phase === "over" && (
        <NinjaOverScreen
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
          <NinjaHud score={score} combo={combo} lives={lives} pop={pop} bombFlash={bombFlash} />
        )}
      </div>
    </AppLayout>
  );
}
