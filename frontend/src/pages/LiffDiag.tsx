// LIFF device check (Phase 0 of the LINE rollout — see docs/line-login-liff-evaluation.md).
// Open this page INSIDE LINE on a real phone to answer, per device, the questions the
// integration hinges on: did liff.init succeed, is there an ID token to exchange, did the
// silent Supabase login happen, and — the known iOS risk — does getUserMedia work in the
// LIFF browser. Everything is read-only except the camera probe (opens then immediately
// releases the stream) and the file-capture input (manual tap test).

import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { Liff } from "@line/liff";
import { probeCamera, type CameraProbeResult } from "../lib/camera";
import { probeLivePose, type PoseProbeResult } from "../lib/poseProbe";
import { initLiff, isLiffConfigured } from "../lib/liff";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";

interface Fact {
  label: string;
  value: string;
}

function safe(read: () => unknown): string {
  try {
    const value = read();
    if (value === null || value === undefined || value === "") return "—";
    return String(value);
  } catch {
    return "—";
  }
}

// The environment table: every LIFF fact the rollout decision needs, null-safe whether we
// are outside LINE, outside LIFF, or the SDK failed to init. Exported for unit tests.
export function collectLiffFacts(liff: Liff | null): Fact[] {
  const configured = isLiffConfigured();
  return [
    { label: "VITE_LIFF_ID", value: configured ? "configured" : "not configured" },
    { label: "liff.init", value: liff ? "ok" : configured ? "failed" : "skipped" },
    { label: "isInClient", value: safe(() => liff?.isInClient()) },
    { label: "isLoggedIn", value: safe(() => liff?.isLoggedIn()) },
    { label: "ID token", value: liff?.getIDToken?.() ? "present" : "—" },
    { label: "OS", value: safe(() => liff?.getOS?.()) },
    { label: "LINE version", value: safe(() => liff?.getLineVersion?.()) },
    { label: "LIFF SDK", value: safe(() => liff?.getVersion?.()) },
    { label: "language", value: safe(() => liff?.getLanguage?.()) },
    {
      // lib.dom types say this always exists; real in-app webviews disagree, hence the runtime check.
      label: "mediaDevices",
      value:
        typeof navigator.mediaDevices?.getUserMedia === "function" ? "available" : "missing",
    },
    { label: "userAgent", value: navigator.userAgent },
  ];
}

export default function LiffDiag() {
  const { t } = useI18n();
  const { user } = useAuth();
  const [facts, setFacts] = useState<Fact[]>([]);
  const [probe, setProbe] = useState<CameraProbeResult | null>(null);
  const [probing, setProbing] = useState(false);
  const [pose, setPose] = useState<PoseProbeResult | null>(null);
  const [posing, setPosing] = useState(false);
  const [inClient, setInClient] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let active = true;
    initLiff().then((liff) => {
      if (!active) return;
      setFacts(collectLiffFacts(liff));
      setInClient(Boolean(liff?.isInClient()));
    });
    return () => {
      active = false;
    };
  }, []);

  async function runProbe() {
    setProbing(true);
    setProbe(null);
    const result = await probeCamera();
    setProbe(result);
    setProbing(false);
  }

  async function runPoseProbe() {
    const video = videoRef.current;
    if (!video) return;
    setPosing(true);
    setPose(null);
    const result = await probeLivePose(video);
    setPose(result);
    setPosing(false);
  }

  const showLiffHint = probe && !probe.ok && inClient && probe.reason !== "denied";

  // Camera answering is necessary but not sufficient — the games are playable only when
  // sustained MediaPipe inference keeps a usable frame rate in this browser.
  const poseVerdictKey = !pose?.ok
    ? null
    : !pose.landmarksSeen
      ? "diag.poseNoLandmarks"
      : (pose.avgFps ?? 0) >= 15
        ? "diag.poseGood"
        : (pose.avgFps ?? 0) >= 8
          ? "diag.poseMarginal"
          : "diag.poseBad";

  return (
    <div className="min-h-[100dvh] bg-background-dark px-5 py-10 text-content">
      <div className="mx-auto w-full max-w-lg">
        <h1 className="font-display text-2xl font-bold tracking-tight">{t("diag.title")}</h1>
        <p className="mt-1.5 text-sm text-muted">{t("diag.subtitle")}</p>

        {/* Environment facts */}
        <h2 className="mt-8 text-xs font-semibold uppercase tracking-wider text-faint">
          {t("diag.env")}
        </h2>
        <dl className="mt-2 divide-y divide-border-dark rounded-xl border border-border-dark bg-surface-dark">
          {facts.map((fact) => (
            <div key={fact.label} className="flex gap-3 px-3.5 py-2 text-sm">
              <dt className="w-32 shrink-0 font-mono text-xs leading-5 text-faint">{fact.label}</dt>
              <dd className="min-w-0 break-all">{fact.value}</dd>
            </div>
          ))}
        </dl>

        {/* Supabase session — verifies the silent in-LIFF login end-to-end */}
        <h2 className="mt-8 text-xs font-semibold uppercase tracking-wider text-faint">
          {t("diag.session")}
        </h2>
        <div className="mt-2 rounded-xl border border-border-dark bg-surface-dark px-3.5 py-2.5 text-sm">
          {user ? (
            <span>
              {t("diag.signedIn")}: <span className="font-mono text-xs">{user.email ?? user.id}</span>
            </span>
          ) : (
            <span className="text-muted">{t("diag.signedOut")}</span>
          )}
        </div>

        {/* Live-camera probe — the decisive iOS-in-LIFF question */}
        <h2 className="mt-8 text-xs font-semibold uppercase tracking-wider text-faint">
          {t("diag.camera")}
        </h2>
        <button
          type="button"
          onClick={runProbe}
          disabled={probing}
          className="mt-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 disabled:opacity-60"
        >
          {probing ? t("diag.probing") : t("diag.probeBtn")}
        </button>
        {probe && (
          <div
            className={`mt-3 rounded-xl border p-3.5 text-sm ${
              probe.ok
                ? "border-primary/30 bg-primary/[0.06] text-primary"
                : "border-danger/30 bg-danger/[0.06] text-danger"
            }`}
          >
            <p className="font-mono text-xs uppercase">{probe.reason}</p>
            <p className="mt-1">{probe.ok ? t("diag.cameraOk") : probe.message}</p>
            {showLiffHint && <p className="mt-2">{t("camera.liffHint")}</p>}
          </div>
        )}

        {/* Camera + MediaPipe together — camera alone doesn't prove the games are playable:
            WASM/WebGL performance inside the LINE WebView is its own failure mode. */}
        <h2 className="mt-8 text-xs font-semibold uppercase tracking-wider text-faint">
          {t("diag.pose")}
        </h2>
        <button
          type="button"
          onClick={runPoseProbe}
          disabled={posing}
          className="mt-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 disabled:opacity-60"
        >
          {posing ? t("diag.probing") : t("diag.poseBtn")}
        </button>
        {/* Small live preview so the tester can aim the camera at themselves. */}
        <video
          ref={videoRef}
          muted
          playsInline
          className={`mt-3 w-40 rounded-lg border border-border-dark ${posing ? "" : "hidden"}`}
        />
        {pose && (
          <div
            className={`mt-3 rounded-xl border p-3.5 text-sm ${
              pose.ok
                ? "border-primary/30 bg-primary/[0.06] text-primary"
                : "border-danger/30 bg-danger/[0.06] text-danger"
            }`}
          >
            <p className="font-mono text-xs uppercase">{pose.stage}</p>
            {pose.ok ? (
              <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-xs">
                <dt>model</dt>
                <dd>{pose.modelLoadMs} ms</dd>
                <dt>warmup</dt>
                <dd>{pose.warmupMs} ms</dd>
                <dt>fps</dt>
                <dd>{pose.avgFps}</dd>
                <dt>landmarks</dt>
                <dd>{pose.landmarksSeen ? "yes" : "no"}</dd>
              </dl>
            ) : (
              <p className="mt-1">{pose.message}</p>
            )}
            {poseVerdictKey && <p className="mt-2">{t(poseVerdictKey)}</p>}
            {!pose.ok && inClient && pose.stage === "camera" && (
              <p className="mt-2">{t("camera.liffHint")}</p>
            )}
          </div>
        )}

        {/* File-input capture — the upload-flow alternative when live camera is dead */}
        <label className="mt-6 block text-sm text-muted">
          {t("diag.capture")}
          <input
            type="file"
            accept="video/*"
            capture="environment"
            className="mt-2 block w-full text-xs text-faint file:mr-3 file:rounded-lg file:border-0 file:bg-content/[0.06] file:px-3 file:py-1.5 file:text-xs file:text-content"
          />
        </label>

        <p className="mt-8">
          <Link to="/app" className="text-sm text-faint transition-colors hover:text-muted">
            ← x-coach
          </Link>
        </p>
      </div>
    </div>
  );
}
