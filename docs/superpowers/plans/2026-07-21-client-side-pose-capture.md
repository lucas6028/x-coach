# Client-Side Pose Capture (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move MediaPipe pose extraction into the browser — user records live (with a real-time skeleton overlay) or uploads a video, the client produces pose JSON, and a new movement-aware backend endpoint runs the existing squat rule detector on it.

**Architecture:** The live overlay (Lite, visual only) is decoupled from the offline analysis extraction (tier-selectable, run from the recorded/uploaded blob). Both input modes funnel through one client extraction pipeline that emits pose JSON byte-identical to `src/pose/process_videos.py`. A new `POST /api/analyze/pose` accepts `{movement, pose JSON, video}` and routes by movement to a pluggable analysis strategy — Squat = the untouched `detect_pose_rules_from_payload`. The server no longer runs MediaPipe for this path; the video is still uploaded for storage/replay.

**Tech Stack:** React 18 + Vite + TypeScript + Tailwind (frontend), `@mediapipe/tasks-vision@0.10.35` (already a dep), `MediaRecorder` + `requestVideoFrameCallback` (new usage), FastAPI + Python (backend), Vitest + pytest.

## Global Constraints

- Python interpreter is `.venv\Scripts\python.exe` from repo root. NEVER bare `python`/`pip`, never `source activate`.
- Backend/ML tests: `.venv\Scripts\python.exe -m pytest tests/` (always scope to `tests/`).
- Backend coverage gate (CI enforces 95%): `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- ALL frontend commands run with cwd = `frontend/`: `yarn test` (vitest run), `yarn test:coverage`, `yarn build`.
- Pose JSON schema is fixed and shared: `{metadata:{fps,width,height,total_frames}, frames:[{frame_index, landmarks:[{x,y,z,visibility}×33]|null, world_landmarks:[…]|null}]}`. The client output MUST match this verbatim — `src/pose/pose_rule_detector.py` is not modified.
- Movement names are canonical spellings from `frontend/src/lib/movements.ts` / `config.DEFAULT_ANALYSIS_MOVEMENT` (`"Squat"`). Do not invent variants.
- Backend tests must NOT import MediaPipe/OpenCV/torch — monkeypatch the detector, mirroring `tests/test_analyze_endpoint.py`.
- MediaPipe/WebGL/camera glue is excluded from frontend coverage (see `vite.config.ts` + `codecov.yml`) — keep impure glue thin and put testable logic in pure helpers.
- No server-side extraction fallback. Device incapability → graceful error, upload mode still offered.

---

## File Structure

**Backend**
- `src/pose/process_videos.py` (modify): expose `model_complexity` param on `process_video` (needed only by the validation task; production analyze path no longer calls it).
- `scripts/pose/validate_complexity_verdicts.py` (create): squat verdict-agreement sweep across complexity 0/1/2; writes result to `notes/`.
- `backend/app/services/analysis.py` (modify): add `_ANALYSIS_STRATEGIES` registry + `analyze_pose_payload(payload, *, movement, video_id)`; reuse existing `build_pose_block_from_payload` / `_strip_frame_metrics` / `save_upload`.
- `backend/app/routers/analyze.py` (modify): add `POST /api/analyze/pose`.

**Frontend** (all under `frontend/src`)
- `lib/poseTier.ts` (create): `PoseTier` type, per-tier `.task` URL map, `DEFAULT_ANALYSIS_TIER`, localStorage get/set.
- `components/poseLandmarker.ts` (modify): `createPoseLandmarker(tier)`.
- `lib/poseExtract.ts` (create): pure `landmarksToFrame(...)` + `extractPoseFromBlob(...)` producing the shared schema.
- `api.ts` (modify): `PoseJson` type + `analyzePose(movement, pose, video)`.
- `components/ComplexitySelector.tsx` (create): Lite/Full/Heavy control, persists via `poseTier`.
- `components/RecordPanel.tsx` (create): camera + live Lite overlay + `MediaRecorder` → blob.
- `components/CaptureStudio.tsx` (create): mode toggle (upload/record) + selector + drives extraction → `analyzePose`.
- `App.tsx` (modify): host `CaptureStudio` in the pre-analysis slot; add `runPoseAnalysis`.
- `pages/Movements.tsx` (modify): Squat card enters the capture studio.

**Tests**
- `tests/test_analyze_pose_service.py`, `tests/test_analyze_pose_endpoint.py` (create).
- `frontend/src/test/lib.poseExtract.test.ts`, `lib.poseTier.test.ts`, `api.pose.test.ts`, `components.ComplexitySelector.test.tsx`, `components.CaptureStudio.test.tsx` (create).

---

## Phase A — Validation gate (sets the default tier)

### Task 1: Squat verdict-agreement across MediaPipe complexity tiers

Measurement spike (not TDD): its output decides `DEFAULT_ANALYSIS_TIER` in Task 4. Requires the local squat dataset under `data/Squat/`.

**Files:**
- Modify: `src/pose/process_videos.py:48-53`
- Create: `scripts/pose/validate_complexity_verdicts.py`
- Create: `notes/mediapipe_complexity_squat_verdicts.md` (output, written by the script run)

**Interfaces:**
- Produces: confirmation of `DEFAULT_ANALYSIS_TIER` value (`"lite"` if verdicts agree, else `"heavy"`), cited in Task 4.

- [ ] **Step 1: Parameterize `process_video` complexity**

In `src/pose/process_videos.py`, change the signature and the `mp_pose.Pose(...)` call:

```python
def process_video(
    input_path: str,
    output_json_path: str,
    output_video_path: str | None = None,
    model_complexity: int = 2,
) -> bool:
    ...
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
```

Default stays `2`, so every existing caller is unchanged.

- [ ] **Step 2: Write the validation script**

Create `scripts/pose/validate_complexity_verdicts.py`:

```python
"""Does MediaPipe complexity (Lite=0 / Full=1 / Heavy=2) change SQUAT rule-detector verdicts?

The shipped "Lite is fine" claim came from a downstream correctness classifier pooled over 6
rehab exercises (notes/rehab24_correctness_experiment_summary.md), NOT the squat fault verdicts.
This measures the thing that actually matters for SP1: for each squat clip, extract pose at each
tier and compare the SET OF DETECTED fault_ids. If Lite and Heavy agree on essentially every
clip, defaulting the client analysis extraction to Lite is defensible.

Run:  .venv\\Scripts\\python.exe scripts/pose/validate_complexity_verdicts.py --limit 40
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pose.pose_rule_detector import detect_pose_rules_from_json  # noqa: E402
from src.pose.process_videos import process_video  # noqa: E402

TIERS = {"lite": 0, "full": 1, "heavy": 2}
SQUAT_VIDEOS = REPO_ROOT / "data" / "Squat" / "Labeled_Dataset" / "videos"


def verdict_set(pose_json: Path) -> frozenset[str]:
    result = detect_pose_rules_from_json(pose_json, include_retrieval=False, movement="Squat")
    return frozenset(d["fault_id"] for d in result.get("detections", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, default=SQUAT_VIDEOS)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "notes" / "mediapipe_complexity_squat_verdicts.md")
    args = ap.parse_args()

    clips = sorted(p for p in args.videos_dir.glob("*.mp4"))[: args.limit]
    if not clips:
        print(f"No .mp4 under {args.videos_dir}", file=sys.stderr)
        return 1

    disagreements = 0
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for clip in clips:
            verdicts = {}
            for tier, cx in TIERS.items():
                out = tmp / f"{clip.stem}_{tier}.json"
                process_video(str(clip), str(out), None, cx)
                verdicts[tier] = verdict_set(out)
            agree = verdicts["lite"] == verdicts["heavy"]
            disagreements += 0 if agree else 1
            rows.append((clip.name, verdicts, agree))

    n = len(clips)
    agree_pct = 100.0 * (n - disagreements) / n
    lines = [
        "# MediaPipe complexity vs squat rule-detector verdicts",
        "",
        f"- clips: {n}",
        f"- Lite==Heavy verdict agreement: {agree_pct:.1f}% ({n - disagreements}/{n})",
        "",
        "| clip | lite | full | heavy | lite==heavy |",
        "|---|---|---|---|---|",
    ]
    for name, v, agree in rows:
        fmt = lambda s: ",".join(sorted(s)) or "—"
        lines.append(f"| {name} | {fmt(v['lite'])} | {fmt(v['full'])} | {fmt(v['heavy'])} | {'yes' if agree else 'NO'} |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} — Lite==Heavy on {agree_pct:.1f}% of {n} clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the validation**

Run: `.venv\Scripts\python.exe scripts/pose/validate_complexity_verdicts.py --limit 40`
Expected: writes `notes/mediapipe_complexity_squat_verdicts.md` and prints the Lite==Heavy agreement %.

- [ ] **Step 4: Record the decision**

Decision rule: **≥95% Lite==Heavy verdict agreement → `DEFAULT_ANALYSIS_TIER = "lite"`** (Task 4). Otherwise `"heavy"`. Note the actual % in the spec's §7 and in the Task 4 comment.

- [ ] **Step 5: Commit**

```bash
git add src/pose/process_videos.py scripts/pose/validate_complexity_verdicts.py notes/mediapipe_complexity_squat_verdicts.md
git commit -m "test(pose): validate Lite vs Heavy squat verdict agreement; parameterize process_video complexity"
```

---

## Phase B — Backend endpoint + strategy seam

### Task 2: Analysis strategy registry + `analyze_pose_payload`

**Files:**
- Modify: `backend/app/services/analysis.py`
- Test: `tests/test_analyze_pose_service.py`

**Interfaces:**
- Consumes: existing `build_pose_block_from_payload(payload)`, `_strip_frame_metrics(result)`, `config.{KG_GRAPH_FILE,RAG_DB_DIR}`; `src.pose.pose_rule_detector.detect_pose_rules_from_payload`.
- Produces: `analyze_pose_payload(payload: dict, *, movement: str, video_id: str | None = None) -> dict[str, Any]`; module dict `_ANALYSIS_STRATEGIES: dict[str, Callable]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_pose_service.py`:

```python
"""analyze_pose_payload: route a client pose payload to a detector strategy (no server extraction)."""
from __future__ import annotations

import unittest

from backend.app.services import analysis as svc


class AnalyzePosePayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = dict(svc._ANALYSIS_STRATEGIES)

    def tearDown(self) -> None:
        svc._ANALYSIS_STRATEGIES.clear()
        svc._ANALYSIS_STRATEGIES.update(self._orig)

    def test_squat_routes_to_strategy_and_attaches_pose_block(self) -> None:
        payload = {
            "metadata": {"fps": 30, "width": 100, "height": 200, "total_frames": 1},
            "frames": [{"frame_index": 0, "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.9}] * 33}],
        }
        svc._ANALYSIS_STRATEGIES["Squat"] = lambda pl, vid: {"detections": [], "video_id": vid}
        result = svc.analyze_pose_payload(payload, movement="Squat", video_id="vid1")
        self.assertEqual(result["source"], "upload")
        self.assertEqual(result["video_id"], "vid1")
        self.assertIn("pose", result)
        self.assertEqual(result["pose"]["fps"], 30.0)
        self.assertEqual(len(result["pose"]["frames"]), 1)

    def test_unknown_movement_returns_coming_soon_without_detector(self) -> None:
        payload = {"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []}
        result = svc.analyze_pose_payload(payload, movement="Deadlift", video_id="v2")
        self.assertEqual(result["analysis_pending"], True)
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["video_id"], "v2")
        self.assertIn("pose", result)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_pose_service.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_ANALYSIS_STRATEGIES'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/app/services/analysis.py`, add after `analyze_video_file` (keep `Callable` import):

```python
from typing import Any, Callable  # extend the existing typing import


def _squat_strategy(payload: dict[str, Any], video_id: str) -> dict[str, Any]:
    # Deferred import: the detector drags in numpy/networkx; keep module import light.
    from src.pose.pose_rule_detector import detect_pose_rules_from_payload

    return detect_pose_rules_from_payload(
        payload,
        video_id=video_id,
        include_retrieval=True,
        graph_file=config.KG_GRAPH_FILE,
        rag_db_dir=config.RAG_DB_DIR,
        movement="Squat",
    )


# The seam SP2 extends: register a per-movement detector strategy under its canonical name.
_ANALYSIS_STRATEGIES: dict[str, Callable[[dict[str, Any], str], dict[str, Any]]] = {
    "Squat": _squat_strategy,
}


def analyze_pose_payload(
    payload: dict[str, Any], *, movement: str, video_id: str | None = None
) -> dict[str, Any]:
    """Analyze a client-supplied pose JSON payload — no server-side MediaPipe.

    Routes by movement to a detector strategy. Movements without a strategy yet return a
    skeleton-only 'analysis pending' result (the video is still stored by the caller).
    """
    vid = video_id or f"upload_{uuid.uuid4().hex[:12]}"
    pose_block = build_pose_block_from_payload(payload)
    strategy = _ANALYSIS_STRATEGIES.get(movement)
    if strategy is None:
        return {
            "video_id": vid,
            "source": "upload",
            "analysis_pending": True,
            "movement": movement,
            "detections": [],
            "retrievals": [],
            "pose": pose_block,
        }
    result = strategy(payload, vid)
    result = _strip_frame_metrics(result)
    result["pose"] = pose_block
    result["source"] = "upload"
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_pose_service.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/analysis.py tests/test_analyze_pose_service.py
git commit -m "feat(analysis): movement-keyed pose-payload analysis strategy seam"
```

### Task 3: `POST /api/analyze/pose` endpoint

**Files:**
- Modify: `backend/app/routers/analyze.py`
- Test: `tests/test_analyze_pose_endpoint.py`

**Interfaces:**
- Consumes: `analysis.save_upload`, `analysis.analyze_pose_payload`, `store.persist_analysis`, `settings.allowed_upload_suffixes`, `_ANALYSIS_SEMAPHORE`.
- Produces: `analyze_pose(movement: str, pose: str, file: UploadFile, user)` coroutine returning the analysis dict.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_pose_endpoint.py`:

```python
"""/api/analyze/pose: accept client pose JSON + video, run the detector off the event loop."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import unittest
from pathlib import Path

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend.app.routers import analyze as analyze_router
from backend.app.services import analysis as analysis_service

_GOOD_POSE = json.dumps({"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": 0}, "frames": []})


def _upload(filename: str = "clip.webm", data: bytes = b"fake") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


class AnalyzePoseEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_save = analysis_service.save_upload
        self._orig_analyze = analysis_service.analyze_pose_payload
        analysis_service.save_upload = lambda data, suffix=".mp4": ("upload_test", Path(f"upload_test{suffix}"))
        analysis_service.analyze_pose_payload = lambda payload, *, movement, video_id=None: {
            "video_id": video_id, "source": "upload", "movement": movement, "detections": [],
        }

    def tearDown(self) -> None:
        analysis_service.save_upload = self._orig_save
        analysis_service.analyze_pose_payload = self._orig_analyze

    def test_happy_path_returns_analysis(self) -> None:
        result = asyncio.run(analyze_router.analyze_pose("Squat", _GOOD_POSE, _upload(), user=None))
        self.assertEqual(result["video_id"], "upload_test")
        self.assertEqual(result["movement"], "Squat")

    def test_rejects_bad_json(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", "{not json", _upload(), user=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_pose_without_frames_list(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", json.dumps({"metadata": {}}), _upload(), user=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_unsupported_suffix(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(analyze_router.analyze_pose("Squat", _GOOD_POSE, _upload("x.txt"), user=None))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_runs_off_the_event_loop(self) -> None:
        seen: dict[str, threading.Thread] = {}

        def record(payload, *, movement, video_id=None):
            seen["t"] = threading.current_thread()
            return {"video_id": video_id, "source": "upload", "detections": []}

        analysis_service.analyze_pose_payload = record
        asyncio.run(analyze_router.analyze_pose("Squat", _GOOD_POSE, _upload(), user=None))
        self.assertIsNot(seen["t"], threading.main_thread())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_pose_endpoint.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'analyze_pose'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/app/routers/analyze.py`, add `Form` to the fastapi import (`from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile`) and add:

```python
import json


@router.post("/analyze/pose")
async def analyze_pose(
    movement: str = Form(...),
    pose: str = Form(...),
    file: UploadFile = File(...),
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict:
    """Analyze a client-extracted pose JSON (no server-side MediaPipe).

    The browser ran MediaPipe on the recorded/uploaded clip and posts the resulting pose JSON
    alongside the raw video (still stored for replay/overlay). Routing/persistence mirror
    ``/api/analyze`` so uploads and library clips still render identically.
    """
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    allowed = await run_in_threadpool(settings.allowed_upload_suffixes)
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    try:
        payload = json.loads(pose)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Malformed pose JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise HTTPException(status_code=400, detail="Pose JSON must have a 'frames' list.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    video_id, _saved_path = await run_in_threadpool(analysis.save_upload, data, suffix=suffix)
    del data
    try:
        async with _ANALYSIS_SEMAPHORE:
            result = await run_in_threadpool(
                analysis.analyze_pose_payload, payload, movement=movement, video_id=video_id
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if user is not None:
        try:
            result["analysis_id"] = await run_in_threadpool(
                store.persist_analysis,
                token=user.token,
                user_id=user.id,
                video_id=video_id,
                source="upload",
                result=result,
                filename=file.filename,
            )
        except Exception:  # noqa: BLE001 — never lose a completed analysis to a storage error
            logger.exception("Failed to persist pose analysis (user=%s video=%s)", user.id, video_id)
            result["analysis_id"] = None
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_pose_endpoint.py -v`
Expected: PASS (all 5).

- [ ] **Step 5: Run the backend coverage gate**

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS (≥95%).

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/analyze.py tests/test_analyze_pose_endpoint.py
git commit -m "feat(api): POST /api/analyze/pose accepts client pose JSON + video"
```

---

## Phase C — Frontend extraction core

### Task 4: Pose tier config + localStorage + landmarker parameterization

**Files:**
- Create: `frontend/src/lib/poseTier.ts`
- Modify: `frontend/src/components/poseLandmarker.ts`
- Test: `frontend/src/test/lib.poseTier.test.ts`

**Interfaces:**
- Produces: `type PoseTier = "lite" | "full" | "heavy"`; `MODEL_URL: Record<PoseTier, string>`; `DEFAULT_ANALYSIS_TIER: PoseTier`; `LIVE_OVERLAY_TIER: PoseTier` (`"lite"`); `loadAnalysisTier(): PoseTier`; `saveAnalysisTier(t: PoseTier): void`. `createPoseLandmarker(tier?: PoseTier)`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/lib.poseTier.test.ts`:

```ts
import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_ANALYSIS_TIER, MODEL_URL, loadAnalysisTier, saveAnalysisTier } from "../lib/poseTier";

afterEach(() => localStorage.clear());

describe("poseTier", () => {
  it("maps every tier to its distinct .task model", () => {
    expect(MODEL_URL.lite).toContain("pose_landmarker_lite");
    expect(MODEL_URL.full).toContain("pose_landmarker_full");
    expect(MODEL_URL.heavy).toContain("pose_landmarker_heavy");
  });

  it("defaults to the validated tier when storage is empty", () => {
    expect(loadAnalysisTier()).toBe(DEFAULT_ANALYSIS_TIER);
  });

  it("round-trips a saved tier and ignores garbage", () => {
    saveAnalysisTier("heavy");
    expect(loadAnalysisTier()).toBe("heavy");
    localStorage.setItem("xcoach.poseTier", "bogus");
    expect(loadAnalysisTier()).toBe(DEFAULT_ANALYSIS_TIER);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (cwd `frontend/`): `yarn test lib.poseTier`
Expected: FAIL — cannot resolve `../lib/poseTier`.

- [ ] **Step 3: Write the implementation**

Create `frontend/src/lib/poseTier.ts`:

```ts
// Which MediaPipe model tier to run. The LIVE overlay is always Lite (perf, visual only);
// the ANALYSIS extraction (offline, from the recorded/uploaded blob) is user-selectable.
export type PoseTier = "lite" | "full" | "heavy";

const MODEL_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker";
export const MODEL_URL: Record<PoseTier, string> = {
  lite: `${MODEL_BASE}/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  full: `${MODEL_BASE}/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
  heavy: `${MODEL_BASE}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
};

// Live overlay: never anything but Lite. Analysis default: set by Task 1's squat verdict-
// agreement measurement (see notes/mediapipe_complexity_squat_verdicts.md for the measured %):
// keep "lite" if Lite==Heavy agreement was ≥95%, otherwise change this to "heavy".
export const LIVE_OVERLAY_TIER: PoseTier = "lite";
export const DEFAULT_ANALYSIS_TIER: PoseTier = "lite";

const KEY = "xcoach.poseTier";
const TIERS: readonly PoseTier[] = ["lite", "full", "heavy"];

export function loadAnalysisTier(): PoseTier {
  const raw = localStorage.getItem(KEY);
  return (TIERS as readonly string[]).includes(raw ?? "") ? (raw as PoseTier) : DEFAULT_ANALYSIS_TIER;
}

export function saveAnalysisTier(tier: PoseTier): void {
  localStorage.setItem(KEY, tier);
}
```

- [ ] **Step 4: Parameterize the landmarker**

In `frontend/src/components/poseLandmarker.ts`, replace the hard-coded `MODEL_URL` with the tier map and add the param (keep the default so game call sites are unchanged):

```ts
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { MODEL_URL, type PoseTier } from "../lib/poseTier";

const VERSION = "0.10.35";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;

export async function createPoseLandmarker(tier: PoseTier = "lite"): Promise<PoseLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL[tier], delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run (cwd `frontend/`): `yarn test lib.poseTier`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/poseTier.ts frontend/src/components/poseLandmarker.ts frontend/src/test/lib.poseTier.test.ts
git commit -m "feat(pose): tier config + selectable model, live overlay pinned to Lite"
```

### Task 5: Client extraction pipeline (blob → pose JSON)

**Files:**
- Create: `frontend/src/lib/poseExtract.ts`
- Test: `frontend/src/test/lib.poseExtract.test.ts`

**Interfaces:**
- Consumes: `PoseTier`, `createPoseLandmarker`; MediaPipe `PoseLandmarkerResult` shape (`landmarks[0]`, `worldLandmarks[0]` = arrays of `{x,y,z,visibility}`).
- Produces:
  - `landmarksToFrame(frameIndex: number, landmarks, worldLandmarks): PoseJsonFrame` (pure).
  - `type PoseJson`, `type PoseJsonFrame` (import-compatible with `api.ts`).
  - `extractPoseFromBlob(blob: Blob, tier: PoseTier, onProgress?: (p: number) => void): Promise<PoseJson>` (impure glue; coverage-excluded).

- [ ] **Step 1: Write the failing test (pure serializer only)**

Create `frontend/src/test/lib.poseExtract.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { landmarksToFrame } from "../lib/poseExtract";

const lm = (n: number) => Array.from({ length: n }, (_, i) => ({ x: i / 100, y: i / 50, z: 0.1, visibility: 0.9 }));

describe("landmarksToFrame", () => {
  it("serializes 33 landmarks + world landmarks into the shared schema", () => {
    const frame = landmarksToFrame(7, lm(33), lm(33));
    expect(frame.frame_index).toBe(7);
    expect(frame.landmarks).toHaveLength(33);
    expect(frame.landmarks![0]).toEqual({ x: 0, y: 0, z: 0.1, visibility: 0.9 });
    expect(frame.world_landmarks).toHaveLength(33);
  });

  it("emits null landmarks when the frame has no full 33-point pose", () => {
    expect(landmarksToFrame(1, undefined, undefined).landmarks).toBeNull();
    expect(landmarksToFrame(2, lm(20), lm(20)).landmarks).toBeNull(); // detector needs >=33
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (cwd `frontend/`): `yarn test lib.poseExtract`
Expected: FAIL — cannot resolve `../lib/poseExtract`.

- [ ] **Step 3: Write the implementation**

Create `frontend/src/lib/poseExtract.ts`:

```ts
// Client-side pose extraction: decode a recorded/uploaded clip frame-by-frame, run MediaPipe,
// and emit pose JSON byte-compatible with src/pose/process_videos.py so the backend detector is
// untouched. The pure serializer (landmarksToFrame) is unit-tested; the <video>/rVFC/WASM glue
// in extractPoseFromBlob is impure and coverage-excluded like the other detector boundaries.
import { createPoseLandmarker } from "../components/poseLandmarker";
import type { PoseTier } from "./poseTier";

const LANDMARK_COUNT = 33;

interface MpLandmark { x: number; y: number; z: number; visibility?: number }
export interface PoseJsonLandmark { x: number; y: number; z: number; visibility: number }
export interface PoseJsonFrame {
  frame_index: number;
  landmarks: PoseJsonLandmark[] | null;
  world_landmarks: PoseJsonLandmark[] | null;
}
export interface PoseJson {
  metadata: { fps: number; width: number; height: number; total_frames: number };
  frames: PoseJsonFrame[];
}

const toPts = (lms?: MpLandmark[]): PoseJsonLandmark[] | null =>
  lms && lms.length >= LANDMARK_COUNT
    ? lms.map((l) => ({ x: l.x, y: l.y, z: l.z, visibility: l.visibility ?? 0 }))
    : null;

export function landmarksToFrame(
  frameIndex: number,
  landmarks?: MpLandmark[],
  worldLandmarks?: MpLandmark[]
): PoseJsonFrame {
  return { frame_index: frameIndex, landmarks: toPts(landmarks), world_landmarks: toPts(worldLandmarks) };
}

/* c8 ignore start — <video>/requestVideoFrameCallback/WASM glue, unrunnable under jsdom */
export async function extractPoseFromBlob(
  blob: Blob,
  tier: PoseTier,
  onProgress?: (p: number) => void
): Promise<PoseJson> {
  const url = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.src = url;
  const landmarker = await createPoseLandmarker(tier);
  const frames: PoseJsonFrame[] = [];
  try {
    await new Promise<void>((res, rej) => {
      video.onloadedmetadata = () => res();
      video.onerror = () => rej(new Error("Could not decode the video."));
    });
    const fps = 30;
    const duration = video.duration || 0;
    let i = 0;
    // Seek-and-detect: step through the clip at a fixed cadence so frame_index is deterministic
    // and aligned to the stored video (rVFC live-rate would drift on drops).
    for (let t = 0; t < duration; t += 1 / fps) {
      video.currentTime = t;
      await new Promise<void>((r) => { video.onseeked = () => r(); });
      const result = landmarker.detectForVideo(video, Math.round(t * 1000));
      frames.push(landmarksToFrame(i, result.landmarks?.[0], result.worldLandmarks?.[0]));
      i += 1;
      onProgress?.(duration ? Math.min(1, t / duration) : 1);
    }
    onProgress?.(1);
    return {
      metadata: { fps, width: video.videoWidth, height: video.videoHeight, total_frames: frames.length },
      frames,
    };
  } finally {
    landmarker.close();
    URL.revokeObjectURL(url);
  }
}
/* c8 ignore stop */
```

- [ ] **Step 4: Run tests to verify they pass**

Run (cwd `frontend/`): `yarn test lib.poseExtract`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/poseExtract.ts frontend/src/test/lib.poseExtract.test.ts
git commit -m "feat(pose): client extraction pipeline emits schema-compatible pose JSON"
```

### Task 6: API client `analyzePose`

**Files:**
- Modify: `frontend/src/api.ts`
- Test: `frontend/src/test/api.pose.test.ts`

**Interfaces:**
- Consumes: `PoseJson` (from `lib/poseExtract`), existing `authHeader()`, `Analysis`.
- Produces: `api.analyzePose(movement: string, pose: PoseJson, video: Blob): Promise<Analysis>`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/api.pose.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";

const pose = { metadata: { fps: 30, width: 1, height: 1, total_frames: 0 }, frames: [] };

afterEach(() => vi.restoreAllMocks());

describe("api.analyzePose", () => {
  it("posts movement + pose JSON + video and returns the analysis", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ video_id: "v1", source: "upload" }), { status: 200 })
    );
    const result = await api.analyzePose("Squat", pose as never, new Blob(["x"], { type: "video/webm" }));
    expect(result.video_id).toBe("v1");
    const [, init] = fetchMock.mock.calls[0];
    const form = init!.body as FormData;
    expect(form.get("movement")).toBe("Squat");
    expect(JSON.parse(form.get("pose") as string).metadata.fps).toBe(30);
    expect(form.get("file")).toBeInstanceOf(Blob);
  });

  it("throws the backend detail on failure", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Pose JSON must have a 'frames' list." }), { status: 400 })
    );
    await expect(api.analyzePose("Squat", pose as never, new Blob(["x"]))).rejects.toThrow("frames");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (cwd `frontend/`): `yarn test api.pose`
Expected: FAIL — `api.analyzePose is not a function`.

- [ ] **Step 3: Write the implementation**

In `frontend/src/api.ts`, add the import at the top (`import type { PoseJson } from "./lib/poseExtract";`) and add this method inside the `api` object, next to `analyzeUpload`:

```ts
  async analyzePose(movement: string, pose: PoseJson, video: Blob): Promise<Analysis> {
    const form = new FormData();
    form.append("movement", movement);
    form.append("pose", JSON.stringify(pose));
    const ext = video.type.includes("mp4") ? "mp4" : "webm";
    form.append("file", video, `capture.${ext}`);
    const res = await fetch("/api/analyze/pose", {
      method: "POST",
      body: form,
      headers: await authHeader(),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error((detail as { detail?: string }).detail || `Analyze failed (${res.status})`);
    }
    return (await res.json()) as Analysis;
  },
```

- [ ] **Step 4: Run tests to verify they pass**

Run (cwd `frontend/`): `yarn test api.pose`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.ts frontend/src/test/api.pose.test.ts
git commit -m "feat(api): analyzePose client posts pose JSON + video to /api/analyze/pose"
```

---

## Phase D — Frontend UI

### Task 7: Complexity selector

**Files:**
- Create: `frontend/src/components/ComplexitySelector.tsx`
- Test: `frontend/src/test/components.ComplexitySelector.test.tsx`

**Interfaces:**
- Consumes: `PoseTier`, `loadAnalysisTier`, `saveAnalysisTier`.
- Produces: `<ComplexitySelector value={PoseTier} onChange={(t: PoseTier) => void} />`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/components.ComplexitySelector.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ComplexitySelector from "../components/ComplexitySelector";

describe("ComplexitySelector", () => {
  it("renders the three tiers and reports the picked one", () => {
    const onChange = vi.fn();
    render(<ComplexitySelector value="lite" onChange={onChange} />);
    fireEvent.click(screen.getByRole("radio", { name: /heavy/i }));
    expect(onChange).toHaveBeenCalledWith("heavy");
  });

  it("marks the current tier as checked", () => {
    render(<ComplexitySelector value="full" onChange={() => {}} />);
    expect(screen.getByRole("radio", { name: /full/i })).toBeChecked();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (cwd `frontend/`): `yarn test components.ComplexitySelector`
Expected: FAIL — cannot resolve the component.

- [ ] **Step 3: Write the implementation**

Create `frontend/src/components/ComplexitySelector.tsx`:

```tsx
import type { PoseTier } from "../lib/poseTier";

const TIERS: { tier: PoseTier; label: string; hint: string }[] = [
  { tier: "lite", label: "Lite", hint: "最快，預設" },
  { tier: "full", label: "Full", hint: "較準" },
  { tier: "heavy", label: "Heavy", hint: "最準，較慢" },
];

// Chooses the model tier for the OFFLINE analysis extraction only. The live recording overlay is
// always Lite regardless of this control.
export default function ComplexitySelector({
  value,
  onChange,
}: {
  value: PoseTier;
  onChange: (tier: PoseTier) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-1.5">
      <legend className="text-xs font-semibold uppercase tracking-wider text-faint">分析精度</legend>
      <div role="radiogroup" className="flex gap-2">
        {TIERS.map(({ tier, label, hint }) => (
          <label
            key={tier}
            className={`flex-1 cursor-pointer rounded-lg border px-3 py-2 text-center text-sm transition-colors ${
              value === tier ? "border-primary bg-primary/[0.08] text-content" : "border-border-dark text-muted hover:bg-content/5"
            }`}
          >
            <input
              type="radio"
              name="pose-tier"
              className="sr-only"
              checked={value === tier}
              onChange={() => onChange(tier)}
              aria-label={`${label} — ${hint}`}
            />
            <span className="block font-medium">{label}</span>
            <span className="block text-xs text-faint">{hint}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run (cwd `frontend/`): `yarn test components.ComplexitySelector`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ComplexitySelector.tsx frontend/src/test/components.ComplexitySelector.test.tsx
git commit -m "feat(ui): analysis-tier ComplexitySelector"
```

### Task 8: Record panel (camera + live Lite overlay + MediaRecorder)

Impure camera/WASM/MediaRecorder glue — coverage-excluded; verified by hand on a real phone.

**Files:**
- Create: `frontend/src/components/RecordPanel.tsx`

**Interfaces:**
- Consumes: `getCameraStream`, `CameraError` (`lib/camera`), `createPoseLandmarker` (Lite via `LIVE_OVERLAY_TIER`), `waitForVideoFrame`.
- Produces: `<RecordPanel onRecorded={(blob: Blob) => void} onError={(msg: string) => void} />`.

- [ ] **Step 1: Write the component**

Create `frontend/src/components/RecordPanel.tsx`:

```tsx
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
        const reason = err instanceof CameraError ? err.reason : "error";
        onError(reason === "unsupported" || reason === "timeout"
          ? "此裝置無法在瀏覽器內開啟相機，請改用上傳。"
          : "相機啟動失敗，請改用上傳。");
      }
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      try { landmarker?.close(); } catch { /* noop */ }
    };
  }, [onError]);

  const start = () => {
    const stream = streamRef.current;
    if (!stream) return;
    chunks.current = [];
    const rec = new MediaRecorder(stream, { mimeType: pickMime() });
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    rec.onstop = () => onRecorded(new Blob(chunks.current, { type: rec.mimeType }));
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
```

- [ ] **Step 2: Verify `POSE_CONNECTIONS` exists**

Run (cwd `frontend/`): `grep -n "POSE_CONNECTIONS" src/lib/pose.ts`
Expected: a connections list export. If it is named differently, import the actual edge-list export from `lib/pose.ts` (used by `SkeletonOverlay.tsx`) and adjust the loop accordingly.

- [ ] **Step 3: Typecheck**

Run (cwd `frontend/`): `yarn build`
Expected: builds without type errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/RecordPanel.tsx
git commit -m "feat(ui): RecordPanel — camera + live Lite skeleton overlay + MediaRecorder"
```

### Task 9: Capture studio (mode toggle → extraction → analyze)

**Files:**
- Create: `frontend/src/components/CaptureStudio.tsx`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/test/components.CaptureStudio.test.tsx`

**Interfaces:**
- Consumes: `UploadDropzone`, `RecordPanel`, `ComplexitySelector`, `extractPoseFromBlob`, `loadAnalysisTier`/`saveAnalysisTier`, a passed `onBlob(blob, tier)` handler.
- Produces: `<CaptureStudio onBlob={(blob: Blob, tier: PoseTier) => void} busy={boolean} progress={number} />`; `App.runPoseAnalysis(blob, tier)`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/components.CaptureStudio.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import CaptureStudio from "../components/CaptureStudio";

// RecordPanel is camera glue — stub it so the studio test stays in jsdom.
vi.mock("../components/RecordPanel", () => ({
  default: ({ onRecorded }: { onRecorded: (b: Blob) => void }) => (
    <button onClick={() => onRecorded(new Blob(["v"], { type: "video/webm" }))}>fake-record</button>
  ),
}));

describe("CaptureStudio", () => {
  it("defaults to upload mode and can switch to record", () => {
    render(<CaptureStudio onBlob={() => {}} busy={false} progress={0} />);
    fireEvent.click(screen.getByRole("tab", { name: /錄影/ }));
    expect(screen.getByText("fake-record")).toBeInTheDocument();
  });

  it("hands a recorded blob + selected tier to onBlob", () => {
    const onBlob = vi.fn();
    render(<CaptureStudio onBlob={onBlob} busy={false} progress={0} />);
    fireEvent.click(screen.getByRole("tab", { name: /錄影/ }));
    fireEvent.click(screen.getByText("fake-record"));
    expect(onBlob).toHaveBeenCalledWith(expect.any(Blob), "lite");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (cwd `frontend/`): `yarn test components.CaptureStudio`
Expected: FAIL — cannot resolve the component.

- [ ] **Step 3: Write the implementation**

Create `frontend/src/components/CaptureStudio.tsx`:

```tsx
import { useState } from "react";
import ComplexitySelector from "./ComplexitySelector";
import RecordPanel from "./RecordPanel";
import UploadDropzone from "./UploadDropzone";
import { loadAnalysisTier, saveAnalysisTier, type PoseTier } from "../lib/poseTier";

type Mode = "upload" | "record";

// The pre-analysis capture screen: pick upload vs live record and the analysis tier, then hand the
// resulting video blob up. Extraction + the API call happen in the parent (App.runPoseAnalysis).
export default function CaptureStudio({
  onBlob,
  busy,
  progress,
}: {
  onBlob: (blob: Blob, tier: PoseTier) => void;
  busy: boolean;
  progress: number;
}) {
  const [mode, setMode] = useState<Mode>("upload");
  const [tier, setTier] = useState<PoseTier>(() => loadAnalysisTier());

  const setTierPersist = (t: PoseTier) => {
    setTier(t);
    saveAnalysisTier(t);
  };

  if (busy) {
    return (
      <div className="flex flex-col items-center gap-3 py-10">
        <p className="text-sm text-muted">分析中… {Math.round(progress * 100)}%</p>
        <div className="h-2 w-64 overflow-hidden rounded-full bg-content/10">
          <div className="h-full bg-primary transition-[width]" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <div role="tablist" className="flex gap-2">
        {(["upload", "record"] as Mode[]).map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            onClick={() => setMode(m)}
            className={`rounded-lg px-4 py-2 text-sm font-medium ${
              mode === m ? "bg-primary text-white" : "bg-content/5 text-muted"
            }`}
          >
            {m === "upload" ? "上傳影片" : "即時錄影"}
          </button>
        ))}
      </div>

      <ComplexitySelector value={tier} onChange={setTierPersist} />

      {mode === "upload" ? (
        <UploadDropzone onFile={(file) => onBlob(file, tier)} />
      ) : (
        <RecordPanel onRecorded={(blob) => onBlob(blob, tier)} onError={(msg) => setMode("upload") || alert(msg)} />
      )}
    </div>
  );
}
```

Note: replace the `|| alert(msg)` error path with the app's existing error surface when wiring in Step 5 (App owns an `error` state); it is inline here only to keep the component self-contained. Do not ship a raw `alert` — see Task 10.

- [ ] **Step 4: Run tests to verify they pass**

Run (cwd `frontend/`): `yarn test components.CaptureStudio`
Expected: PASS.

- [ ] **Step 5: Wire `runPoseAnalysis` into `App.tsx`**

In `frontend/src/App.tsx`, add near `runUpload` (mirrors it, but extracts client-side first). Add a `progress` state (`const [progress, setProgress] = useState(0);`) and import `extractPoseFromBlob` + `PoseTier`:

```tsx
  const runPoseAnalysis = useCallback(async (blob: Blob, tier: PoseTier) => {
    setLoading(true);
    setError("");
    setAnalysis(null);
    setProgress(0);
    setStatusMsg(t("app.analysing"));
    try {
      const pose = await extractPoseFromBlob(blob, tier, setProgress);
      const data = await api.analyzePose("Squat", pose, blob);
      setAnalysis(data);
      if (data.analysis_id) {
        skipReloadId.current = data.analysis_id;
        setSearchParams({ analysis: data.analysis_id }, { replace: true });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setStatusMsg("");
    }
  }, [t, setSearchParams]);
```

Render `<CaptureStudio onBlob={runPoseAnalysis} busy={loading} progress={progress} />` in the pre-analysis slot where `DemoIntro`/`UploadDropzone` currently renders (replacing the `onFile={runUpload}` wiring for the squat flow). Keep `runUpload`/`api.analyzeUpload` in place for any non-client fallback paths that still need them.

- [ ] **Step 6: Run the full frontend suite + build**

Run (cwd `frontend/`): `yarn test && yarn build`
Expected: all tests pass, build succeeds.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/CaptureStudio.tsx frontend/src/App.tsx frontend/src/test/components.CaptureStudio.test.tsx
git commit -m "feat(ui): CaptureStudio — upload/record modes drive client-side pose analysis"
```

### Task 10: Movements entry + capability gate + coverage

**Files:**
- Modify: `frontend/src/pages/Movements.tsx`
- Modify: `frontend/src/components/CaptureStudio.tsx` (replace the placeholder `alert` error path)

**Interfaces:**
- Consumes: the capture flow at `/app`.

- [ ] **Step 1: Confirm the Squat entry lands on the capture flow**

The Squat card already `navigate("/app")` (`Movements.tsx:57`), and `/app` now renders `CaptureStudio`. No routing change needed — verify by reading `main.tsx` that `/app` mounts `App`. If `App` gates on a different pre-analysis condition, ensure `CaptureStudio` shows when `analysis === null && !loading`.

- [ ] **Step 2: Replace the placeholder error path in CaptureStudio**

Lift a `onError?: (msg: string) => void` prop into `CaptureStudio` and pass it to `RecordPanel`; in `App.tsx` pass `onError={setError}`. Remove the inline `|| alert(msg)`:

```tsx
// CaptureStudio signature gains: onError?: (msg: string) => void
<RecordPanel onRecorded={(blob) => onBlob(blob, tier)} onError={(msg) => { setMode("upload"); onError?.(msg); }} />
```

- [ ] **Step 3: Run the full frontend coverage suite**

Run (cwd `frontend/`): `yarn test:coverage`
Expected: PASS. The coverage-excluded glue (`RecordPanel`, `extractPoseFromBlob`, `poseLandmarker`) must carry the `/* c8 ignore */` markers so coverage does not regress.

- [ ] **Step 4: Run the full backend suite + coverage gate**

Run: `.venv\Scripts\python.exe -m pytest tests/` then `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Movements.tsx frontend/src/components/CaptureStudio.tsx frontend/src/App.tsx
git commit -m "feat(ui): Squat entry into capture studio; surface record errors via app state"
```

- [ ] **Step 6: Update graphify**

Run: `graphify update .`
Expected: AST-only refresh, no API cost.

---

## Self-Review notes (coverage of the spec)

- §1 選動作入口 → Task 10. 上傳/錄影雙模式 → Tasks 8, 9. 共用離線抽取管線 → Task 5. complexity 選擇器 → Tasks 4, 7. 新端點 + strategy registry → Tasks 2, 3. 測試 + 覆蓋率 →每個任務 + Tasks 3/10.
- §2 即時 Lite / 分析 tier 解耦 → `LIVE_OVERLAY_TIER` vs `DEFAULT_ANALYSIS_TIER` (Task 4), RecordPanel uses Lite, extraction uses selected tier.
- §4 schema 對齊 → Task 5 `landmarksToFrame` emits `{x,y,z,visibility}` + world; verified against `build_pose_block_from_payload`.
- §5 錯誤處理 / 無 fallback → RecordPanel CameraError branch (Task 8), App error surface (Task 10). Long-video soft warning is a follow-up nicety, not gating.
- §6 後端 → Tasks 2, 3. 舊 `/api/analyze` untouched.
- §7 Lite 預設驗證 → Task 1 gates the `DEFAULT_ANALYSIS_TIER` value in Task 4.
- Open follow-up (not SP1-gating): >30s soft warning UI; MediaPipe init capability pre-gate reusing `probeLivePose` before showing record mode.
