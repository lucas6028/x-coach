import { useCallback, useEffect, useRef, useState } from "react";
import AppLayout from "../components/AppLayout";
import SixSevenStartScreen from "../components/sixseven/SixSevenStartScreen";
import SixSevenHud from "../components/sixseven/SixSevenHud";
import SixSevenOverScreen, { type SixSevenResult } from "../components/sixseven/SixSevenOverScreen";
import { useI18n } from "../lib/i18n";
import { handLead, type Lead } from "../lib/sixseven/gesture";
import { stepCount, initialCount, ROUND_SECONDS, type CountState } from "../lib/sixseven/counter";
import { loadLeaderboard, saveScore, type SixSevenEntry } from "../lib/sixseven/leaderboard";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { createPoseLandmarker, drawScene } from "../components/sixseven/sixSevenDetector";

type Phase = "intro" | "countdown" | "playing" | "over";

const POP_MS = 350;

// 67 — the brainrot mini-game. Do the "6-7" bob (alternate raising each hand) and every switch
// counts one 67; keep the rhythm for a combo. React state drives the UI; a ref-backed rAF loop
// drives detection + counting. The gesture + counter logic lives in lib/sixseven/* (unit-tested)
// — this file wires it to the camera and owns only the impure edges.
export default function SixSeven() {
  const { t } = useI18n();

  const [phase, setPhase] = useState<Phase>("intro");
  const [leaderboard, setLeaderboard] = useState<SixSevenEntry[]>(() => loadLeaderboard());
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState(3);

  const [count, setCount] = useState(0);
  const [combo, setCombo] = useState(0);
  const [timeLeft, setTimeLeft] = useState(ROUND_SECONDS);
  const [lead, setLead] = useState<Lead>("neutral");
  const [pop, setPop] = useState(0);

  const [result, setResult] = useState<SixSevenResult>({ count: 0, bestCombo: 0 });
  const [submitted, setSubmitted] = useState(false);
  const [rank, setRank] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);

  // Loop bookkeeping the rAF callback mutates without re-rendering; `counter` holds the pure
  // rep-counting state advanced by stepCount.
  const g = useRef({
    running: false,
    roundEnd: 0,
    lastVideoTime: -1,
    lastUi: 0,
    counter: initialCount as CountState,
    lead: "neutral" as Lead,
    popCount: 0,
    popUntil: 0,
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
    const c = g.current.counter;
    teardown();
    setResult({ count: c.count, bestCombo: c.bestCombo });
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
    const secLeft = Math.max(0, Math.ceil((s.roundEnd - now) / 1000));

    // Detection + counting run only on a fresh camera frame (MediaPipe needs rising timestamps).
    if (video.currentTime !== s.lastVideoTime) {
      s.lastVideoTime = video.currentTime;
      const lm = landmarker.detectForVideo(video, now).landmarks?.[0] ?? null;
      const hs = lm ? handLead(lm) : null;
      const lead = hs?.valid ? hs.lead : "neutral";
      s.lead = lead;

      const stepped = stepCount(s.counter, lead, now);
      s.counter = stepped.state;
      if (stepped.scored) {
        s.popCount += 1;
        s.popUntil = now + POP_MS;
      }

      const ctx = canvas.getContext("2d");
      if (ctx) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
        }
        drawScene(
          ctx,
          { landmarks: lm, lead, pop: s.popUntil > now ? (s.popUntil - now) / POP_MS : null },
          canvas.width,
          canvas.height
        );
      }
    }

    // Throttle React updates to ~20/s.
    if (now - s.lastUi > 50) {
      s.lastUi = now;
      setCount(s.counter.count);
      setCombo(s.counter.combo);
      setTimeLeft(secLeft);
      setLead(s.lead);
      setPop(s.popCount);
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
        const s = g.current;
        s.running = true;
        s.roundEnd = performance.now() + ROUND_SECONDS * 1000;
        s.lastVideoTime = -1;
        s.lastUi = 0;
        s.counter = initialCount;
        s.lead = "neutral";
        s.popCount = 0;
        s.popUntil = 0;
        setCount(0);
        setCombo(0);
        setTimeLeft(ROUND_SECONDS);
        setLead("neutral");
        setPop(0);
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
      setError(e instanceof Error ? e.message : t("six.error"));
    }
  }, [beginCountdown, teardown, t]);

  const submit = useCallback(
    (name: string) => {
      const entry: SixSevenEntry = {
        name,
        count: result.count,
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
    <AppLayout title={t("six.title")}>
      {phase === "intro" && (
        <SixSevenStartScreen
          leaderboard={leaderboard}
          onStart={start}
          starting={starting}
          error={error}
        />
      )}

      {phase === "over" && (
        <SixSevenOverScreen
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
          <SixSevenHud count={count} combo={combo} timeLeft={timeLeft} lead={lead} pop={pop} />
        )}
      </div>
    </AppLayout>
  );
}
