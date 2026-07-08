import { useCallback, useEffect, useRef, useState } from "react";
import AppLayout from "../components/AppLayout";
import DuelStartScreen from "../components/duel/DuelStartScreen";
import DuelHud from "../components/duel/DuelHud";
import DuelOverScreen, { type DuelResult } from "../components/duel/DuelOverScreen";
import { useI18n } from "../lib/i18n";
import { assignPlayers } from "../lib/duel/assign";
import { poseSignature } from "../lib/duel/angles";
import { poseById, pickPose, scorePose, POSES } from "../lib/duel/poses";
import {
  advanceHold,
  roundWinner,
  matchWinner,
  HOLD_MS,
  MATCH_THRESHOLD,
  ROUND_BREAK_MS,
  type Side,
} from "../lib/duel/match";
import { loadResults, saveResult, type DuelEntry } from "../lib/duel/leaderboard";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { createPoseLandmarker, drawScene } from "../components/duel/duelDetector";

type Phase = "intro" | "countdown" | "playing" | "over";

// Pose Duel — two players share one camera (numPoses: 2). Each round shows a target pose; the
// first to hold it long enough takes the round, and first to MATCH_POINTS takes the match. React
// state drives the UI; a ref-backed rAF loop drives detection, assignment, and scoring. All the
// rules live in lib/duel/* (unit-tested); this file wires them to the camera.
export default function PoseDuel() {
  const { t } = useI18n();

  const [phase, setPhase] = useState<Phase>("intro");
  const [results, setResults] = useState<DuelEntry[]>(() => loadResults());
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState(3);

  const [poseEmoji, setPoseEmoji] = useState(POSES[0].emoji);
  const [poseNameKey, setPoseNameKey] = useState(POSES[0].nameKey);
  const [aWins, setAWins] = useState(0);
  const [bWins, setBWins] = useState(0);
  const [aHold, setAHold] = useState(0);
  const [bHold, setBHold] = useState(0);
  const [aPresent, setAPresent] = useState(false);
  const [bPresent, setBPresent] = useState(false);
  const [roundFlash, setRoundFlash] = useState<Side | null>(null);

  const [result, setResult] = useState<DuelResult>({ winner: "a", aWins: 0, bWins: 0 });
  const [submitted, setSubmitted] = useState(false);
  const [savedTs, setSavedTs] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);

  // Mutable game state the loop mutates without re-rendering.
  const g = useRef({
    running: false,
    last: 0,
    lastVideoTime: -1,
    lastUi: 0,
    poseId: POSES[0].id,
    winsA: 0,
    winsB: 0,
    holdA: 0, // ms toward HOLD_MS
    holdB: 0,
    breakUntil: 0, // scoring frozen while now < breakUntil
    roundFlash: null as Side | null,
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

  const endMatch = useCallback(() => {
    const s = g.current;
    const winner = matchWinner(s.winsA, s.winsB) ?? "a";
    teardown();
    setResult({ winner, aWins: s.winsA, bWins: s.winsB });
    setResults(loadResults());
    setSubmitted(false);
    setSavedTs(null);
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
    const inBreak = now < s.breakUntil;

    // Detect (up to two bodies) only on a fresh camera frame.
    let poses: ReturnType<PoseLandmarker["detectForVideo"]>["landmarks"] = [];
    if (video.currentTime !== s.lastVideoTime) {
      s.lastVideoTime = video.currentTime;
      poses = landmarker.detectForVideo(video, now).landmarks ?? [];
    }
    const { a: la, b: lb } = assignPlayers(poses);

    let matchedA = false;
    let matchedB = false;
    if (!inBreak) {
      const pose = poseById(s.poseId) ?? POSES[0];
      if (la) matchedA = scorePose(poseSignature(la), pose).score >= MATCH_THRESHOLD;
      if (lb) matchedB = scorePose(poseSignature(lb), pose).score >= MATCH_THRESHOLD;
      s.holdA = advanceHold(s.holdA, matchedA, dtMs);
      s.holdB = advanceHold(s.holdB, matchedB, dtMs);

      const w = roundWinner(s.holdA, s.holdB);
      if (w) {
        if (w === "a") s.winsA += 1;
        else s.winsB += 1;
        s.holdA = 0;
        s.holdB = 0;
        s.roundFlash = w;
        if (matchWinner(s.winsA, s.winsB)) {
          endMatch();
          return;
        }
        s.poseId = pickPose(s.poseId, Math.random()).id;
        s.breakUntil = now + ROUND_BREAK_MS;
      }
    }

    const showFlash = now < s.breakUntil ? s.roundFlash : null;

    // Render both skeletons + hold rings.
    const ctx = canvas.getContext("2d");
    if (ctx) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
      }
      drawScene(
        ctx,
        {
          players: { a: la, b: lb },
          a: { hold: s.holdA / HOLD_MS, matched: matchedA },
          b: { hold: s.holdB / HOLD_MS, matched: matchedB },
        },
        canvas.width,
        canvas.height
      );
    }

    // Throttle React updates to ~20/s.
    if (now - s.lastUi > 50) {
      s.lastUi = now;
      const pose = poseById(s.poseId) ?? POSES[0];
      setPoseEmoji(pose.emoji);
      setPoseNameKey(pose.nameKey);
      setAWins(s.winsA);
      setBWins(s.winsB);
      setAHold(s.holdA / HOLD_MS);
      setBHold(s.holdB / HOLD_MS);
      setAPresent(!!la);
      setBPresent(!!lb);
      setRoundFlash(showFlash);
    }
  }, [endMatch]);

  const beginCountdown = useCallback(() => {
    setPhase("countdown");
    let n = 3;
    setCountdown(n);
    const iv = setInterval(() => {
      n -= 1;
      if (n <= 0) {
        clearInterval(iv);
        const s = g.current;
        const first = pickPose(null, Math.random());
        s.running = true;
        s.last = 0;
        s.lastVideoTime = -1;
        s.lastUi = 0;
        s.poseId = first.id;
        s.winsA = 0;
        s.winsB = 0;
        s.holdA = 0;
        s.holdB = 0;
        s.breakUntil = 0;
        s.roundFlash = null;
        setPoseEmoji(first.emoji);
        setPoseNameKey(first.nameKey);
        setAWins(0);
        setBWins(0);
        setAHold(0);
        setBHold(0);
        setRoundFlash(null);
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
        video: { facingMode: "user", width: 1280, height: 480 },
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
      setError(e instanceof Error ? e.message : t("duel.error"));
    }
  }, [beginCountdown, teardown, t]);

  const submit = useCallback(
    (winner: string, loser: string) => {
      const winnerIsA = result.winner === "a";
      const entry: DuelEntry = {
        winner,
        loser,
        winnerPoints: winnerIsA ? result.aWins : result.bWins,
        loserPoints: winnerIsA ? result.bWins : result.aWins,
        ts: Date.now(),
      };
      setResults(saveResult(entry));
      setSavedTs(entry.ts);
      setSubmitted(true);
    },
    [result]
  );

  const replay = useCallback(() => {
    setResults(loadResults());
    setPhase("intro");
  }, []);

  const showCamera = phase === "playing" || phase === "countdown";

  return (
    <AppLayout title={t("duel.title")}>
      {phase === "intro" && (
        <DuelStartScreen results={results} onStart={start} starting={starting} error={error} />
      )}

      {phase === "over" && (
        <DuelOverScreen
          result={result}
          results={results}
          submitted={submitted}
          savedTs={savedTs}
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
            <div className="text-center">
              <span className="font-display text-8xl font-black text-white drop-shadow-lg">
                {countdown}
              </span>
              <p className="mt-2 font-mono text-sm uppercase tracking-wider text-zinc-300">
                {t("duel.getReady")}
              </p>
            </div>
          </div>
        )}

        {phase === "playing" && (
          <DuelHud
            poseEmoji={poseEmoji}
            poseNameKey={poseNameKey}
            a={{ wins: aWins, hold: aHold, present: aPresent }}
            b={{ wins: bWins, hold: bHold, present: bPresent }}
            roundFlash={roundFlash}
          />
        )}
      </div>
    </AppLayout>
  );
}
