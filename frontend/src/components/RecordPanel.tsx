/* c8 ignore start — camera + MediaRecorder + WASM overlay glue, unrunnable under jsdom */
import { useEffect, useRef, useState } from "react";
import { CameraError, getCameraStream } from "../lib/camera";
import { LIVE_OVERLAY_TIER } from "../lib/poseTier";
import { POSE_CONNECTIONS } from "../lib/pose";
import { waitForVideoFrame } from "../lib/videoFrame";

function pickMime(): string {
  const prefs = ["video/webm;codecs=vp9", "video/webm", "video/mp4"];
  return prefs.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
}

export default function RecordPanel({
  onRecorded,
  onError,
}: {
  onRecorded: (blob: Blob) => void;
  onError: (msg: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const [recording, setRecording] = useState(false);
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;
  const onRecordedRef = useRef(onRecorded);
  onRecordedRef.current = onRecorded;

  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    let landmarker: { detectForVideo(v: HTMLVideoElement, t: number): { landmarks?: { x: number; y: number }[][] }; close(): void } | null = null;

    (async () => {
      try {
        const stream = await getCameraStream({ video: { facingMode: "user", width: 1280, height: 720 }, audio: false });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play().catch(() => {});
        await waitForVideoFrame(video);
        const { createPoseLandmarker } = await import("./poseLandmarker");
        landmarker = await createPoseLandmarker(LIVE_OVERLAY_TIER);
        if (cancelled) { landmarker.close(); return; }
        const draw = () => {
          if (cancelled) return;
          const canvas = canvasRef.current!;
          const ctx = canvas.getContext("2d")!;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const res = landmarker!.detectForVideo(video, performance.now());
          const pts = res.landmarks?.[0];
          if (pts) {
            ctx.strokeStyle = "#f97316";
            ctx.lineWidth = 3;
            for (const [a, b] of POSE_CONNECTIONS) {
              ctx.beginPath();
              ctx.moveTo(pts[a].x * canvas.width, pts[a].y * canvas.height);
              ctx.lineTo(pts[b].x * canvas.width, pts[b].y * canvas.height);
              ctx.stroke();
            }
          }
          raf = requestAnimationFrame(draw);
        };
        draw();
      } catch (err) {
        if (cancelled) return; // unmounted mid-init: cleanup already stopped the stream
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        const reason = err instanceof CameraError ? err.reason : "error";
        onErrorRef.current(reason === "unsupported" || reason === "timeout"
          ? "此裝置無法在瀏覽器內開啟相機，請改用上傳。"
          : "相機啟動失敗，請改用上傳。");
      }
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      try { landmarker?.close(); } catch { /* noop */ }
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.onstop = null;
        recorderRef.current.stop();
      }
    };
  }, []);

  const start = () => {
    const stream = streamRef.current;
    if (!stream) return;
    chunks.current = [];
    const rec = new MediaRecorder(stream, { mimeType: pickMime() });
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    rec.onstop = () => onRecordedRef.current(new Blob(chunks.current, { type: rec.mimeType }));
    rec.start();
    recorderRef.current = rec;
    setRecording(true);
  };

  const stop = () => {
    recorderRef.current?.stop();
    setRecording(false);
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative w-full max-w-md overflow-hidden rounded-xl bg-black">
        <video ref={videoRef} className="w-full" muted playsInline />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      </div>
      <button
        onClick={recording ? stop : start}
        className="rounded-full bg-primary px-6 py-3 font-medium text-white active:translate-y-px"
      >
        {recording ? "停止並分析" : "開始錄影"}
      </button>
    </div>
  );
}
/* c8 ignore stop */
