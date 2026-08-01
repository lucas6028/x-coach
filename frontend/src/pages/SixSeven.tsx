import { useCallback, useEffect, useRef, useState } from "react";
import AppLayout from "../components/AppLayout";
import SixSevenStartScreen from "../components/sixseven/SixSevenStartScreen";
import SixSevenHud from "../components/sixseven/SixSevenHud";
import SixSevenOverScreen, { type SixSevenResult } from "../components/sixseven/SixSevenOverScreen";
import { useI18n } from "../lib/i18n";
import { handLead, type Lead } from "../lib/sixseven/gesture";
import { stepCount, initialCount, ROUND_SECONDS, type CountState } from "../lib/sixseven/counter";
import { loadLeaderboard, saveScore, type SixSevenEntry } from "../lib/sixseven/leaderboard";
import { estimateKcal, EFFORT } from "../lib/calories";
import { addCalories } from "../lib/calorieStore";
import { waitForVideoFrame } from "../lib/videoFrame";
import { CameraError, getCameraStream } from "../lib/camera";
import { isInLiffClient } from "../lib/liff";
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

  const [result, setResult] = useState<SixSevenResult>({ count: 0, bestCombo: 0, kcal: 0 });
  const [submitted, setSubmitted] = useState(false);
  const [rank, setRank] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

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
    if (countdownRef.current !== null) {
      clearInterval(countdownRef.current);
      countdownRef.current = null;
    }
    streamRef.current?.getTracks().forEach((tr) => tr.stop());
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
    const c = g.current.counter;
    teardown();
    // The round always runs the full clock; 67s completed are the movement signal. Record once here.
    const kcal = estimateKcal({ durationSec: ROUND_SECONDS, moves: c.count, effort: EFFORT.sixseven });
    addCalories("sixseven", kcal);
    setResult({ count: c.count, bestCombo: c.bestCombo, kcal });
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

      // Cache the context once — it never changes for a given canvas, and this runs every frame.
      const ctx = (ctxRef.current ??= canvas.getContext("2d"));
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
        countdownRef.current = null;
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
    countdownRef.current = iv;
  }, [tick]);

  const start = useCallback(async () => {
    setError("");
    setStarting(true);
    try {
      // The model download (WASM + .task, the bulk of the wait on mobile) doesn't depend on the
      // camera — kick it off in parallel with the getUserMedia permission/stream so the two waits
      // overlap instead of stacking.
      const landmarkerPromise = createPoseLandmarker();
      // Timeout-wrapped: inside the LINE (LIFF) in-app browser on iOS getUserMedia can hang
      // forever — the wrapper turns that into a catchable error (see lib/camera).
      const stream = await getCameraStream({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      // getUserMedia resolved after an unmount — stop the orphaned track and bail. Also close the
      // model once its download settles so it doesn't leak.
      if (!mountedRef.current) {
        stream.getTracks().forEach((tr) => tr.stop());
        landmarkerPromise.then((l) => l.close()).catch(() => {});
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }
      const landmarker = await landmarkerPromise;
      // Unmounted while loading the model — close it and release the stream teardown owns.
      if (!mountedRef.current) {
        landmarker.close();
        teardown();
        return;
      }
      landmarkerRef.current = landmarker;

      // Warm up the GPU delegate NOW, while the loading spinner is still up. The first
      // detectForVideo compiles shaders and can freeze the main thread for tens of seconds on
      // mobile; run it here so that stall hides in the load the user is already waiting on rather
      // than freezing the 2.4s countdown (where it can't possibly fit). Needs a decoded frame, and
      // a warmup that throws must not abort the game.
      if (video) {
        await waitForVideoFrame(video);
        if (!mountedRef.current) {
          teardown();
          return;
        }
        if (video.readyState >= 2) {
          try {
            landmarker.detectForVideo(video, performance.now());
          } catch (e) {
            console.error("six-seven: warmup inference failed", e);
          }
        }
      }

      setStarting(false);
      beginCountdown();
    } catch (e) {
      teardown();
      setStarting(false);
      // Show the localized message; keep the raw DOMException in the console for debugging.
      console.error("six-seven: failed to start", e);
      // A hung/missing camera inside the LINE in-app browser gets a specific escape-hatch
      // hint (open in a real browser) instead of the generic failure text.
      const liffCameraDead = e instanceof CameraError && (await isInLiffClient());
      setError(liffCameraDead ? t("camera.liffHint") : t("six.error"));
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
    <AppLayout initialSidebarOpen={false}>
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
