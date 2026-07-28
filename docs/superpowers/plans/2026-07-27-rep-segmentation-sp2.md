# RS-SP2：只抽取選中 rep 區間 — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓瀏覽器只對「會被評分的那幾下」逐幀跑 MediaPipe，而不是整段影片。

**Architecture:** 兩趟抽取。粗掃（每 3 幀取 1、Lite）算出 `avg_knee_angle` → 切 rep → 選首/中/尾
→ 以每個 rep 的**谷底**為錨展開 padded span → 只對這些 span 密集抽取（使用者選的 tier）
→ 在密集訊號上精修邊界 → 送出「全長 frames（未抽取為 null）+ reps 區塊」。後端不再自己切割，
改為接受並**驗證** client 給的區間。

**Tech Stack:** TypeScript + Vitest（前端）、Python + unittest/pytest（後端與 ML）、
MediaPipe tasks-vision 0.10.35、FastAPI。

**Spec:** `docs/superpowers/specs/2026-07-27-rep-segmentation-sp2-design.md`（每個 Task 都標了對應章節）

## Global Constraints

- **Python 直譯器一律 `.venv\Scripts\python.exe`**（此機器 PATH 上沒有 `python`）。
  測試：`.venv\Scripts\python.exe -m pytest tests/`（永遠 scope 到 `tests/`，不要裸跑 `pytest`）。
- **前端所有 yarn/vitest 指令 cwd 必須是 `frontend/`**。跑錯目錄會讓 vitest 大規模失敗。
- 後端覆蓋率門檻：`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`。
- **所有門檻是具名常數，不是行內字面值**（SP1 §7 的可移植性契約，TS 與 Python 兩份實作靠它不漂移）。
- 常數值（spec §2.5、§2.8、§3）：`CANONICAL_FPS = 30`、`COARSE_STRIDE = 3`、
  `COARSE_SMOOTH_WINDOW = 3`、`DENSE_SMOOTH_WINDOW = 5`、`REP_PADDING_FRAMES = 24`。
- 使用者可見字串一律走 `frontend/src/lib/i18n.tsx`，繁體中文 + 英文兩份。
- 新 TS 檔放 `frontend/src/lib/`，測試放 `frontend/src/test/`，命名比照既有的 `lib.<name>.test.ts`。
- 測試是 `unittest.TestCase`（Python，`tests/`）與 vitest（前端）。

---

## File Structure

**新增（前端）**
| 檔案 | 職責 |
|---|---|
| `frontend/src/lib/repSignal.ts` | 從 landmarks 算 rep 訊號：`avgKneeAngle`、`centeredMedian`、`TS_REP_SIGNALS` |
| `frontend/src/lib/repSegmentation.ts` | `segmentReps` / `selectReps`——`src/pose/rep_segmentation.py` 的逐行移植 |
| `frontend/src/lib/repSpans.ts` | 谷底定位、span 展開與合併、邊界精修、`frameIndexAt` |

**修改（前端）**
| 檔案 | 改什麼 |
|---|---|
| `frontend/src/lib/poseExtract.ts` | 拆出 `sampleFrames`；新增 `extractPoseWithReps`（兩趟 + 精修）；`frame_index` 改用 `frameIndexAt` |
| `frontend/src/api.ts` | `analyzePose` 多送 `reps`；`Analysis` 型別加 `reps` / `quality.extracted_frames` |
| `frontend/src/App.tsx:116-141` | `runPoseAnalysis` 改呼叫 `extractPoseWithReps` 並把 `reps` 往下傳 |
| `frontend/src/components/Timeline.tsx` | 未分析區段畫中性斜紋 |
| `frontend/src/components/MetricsCards.tsx:93-97` | 分母改 `extracted_frames` |
| `frontend/src/lib/i18n.tsx` | 新增 4 個 key |

**新增（後端 / ML）**
| 檔案 | 職責 |
|---|---|
| `tests/test_coarse_segmentation_corpus.py` | 真實 pose JSON 上的粗掃迴歸（`skipUnless` 資料存在） |

**修改（後端 / ML）**
| 檔案 | 改什麼 |
|---|---|
| `src/pose/movements/base.py` | `RepPlan` dataclass；`run_detector(..., rep_plan=None)` |
| `src/pose/pose_rule_detector.py:596-663` | `detect_pose_rules_from_payload(..., rep_plan=None)`；`quality` 加 `extracted_frames` / `extracted_frame_ratio` |
| `backend/app/services/analysis.py:155-191` | `analyze_pose_payload(..., rep_plan=None)` |
| `backend/app/routers/analyze.py:164-224` | `/analyze/pose` 收 `reps` Form 欄位 + `_validate_reps` |

---

## Task 1：把 `frame_index` 釘在 30fps 正規格點

**Spec:** §2.5

**為什麼先做**：`poseExtract.ts:140` 目前用自增計數器 `i`，只在步長剛好 `1/30` 時等於
`round(t*30)`。粗掃步長不同，沿用計數器會讓區間落在**錯的索引空間，而且不會報錯**。這是既有隱患，
與其餘工作無相依。

**Files:**
- Create: `frontend/src/lib/repSpans.ts`
- Modify: `frontend/src/lib/poseExtract.ts:134-147`
- Test: `frontend/src/test/lib.repSpans.test.ts`

**Interfaces:**
- Consumes: 無
- Produces: `CANONICAL_FPS: 30`、`frameIndexAt(t: number): number`

- [ ] **Step 1: 寫失敗的測試**

`frontend/src/test/lib.repSpans.test.ts`：

```ts
import { describe, it, expect } from "vitest";
import { CANONICAL_FPS, frameIndexAt } from "../lib/repSpans";

describe("frameIndexAt", () => {
  it("puts every sample on the canonical 30fps grid", () => {
    expect(CANONICAL_FPS).toBe(30);
    expect(frameIndexAt(0)).toBe(0);
    expect(frameIndexAt(1 / 30)).toBe(1);
    expect(frameIndexAt(1)).toBe(30);
  });

  it("gives a coarse pass the SAME indices a dense pass would give", () => {
    // The bug this pins: an incrementing counter makes coarse sample k index k, not 3k.
    const dense = Array.from({ length: 10 }, (_, i) => frameIndexAt(i / 30));
    const coarse = [0, 3, 6, 9].map((i) => frameIndexAt(i / 30));
    expect(coarse).toEqual([0, 3, 6, 9]);
    expect(coarse.every((c) => dense.includes(c))).toBe(true);
  });
});
```

- [ ] **Step 2: 跑測試確認失敗**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSpans.test.ts`
預期：FAIL，`Failed to resolve import "../lib/repSpans"`

- [ ] **Step 3: 寫最小實作**

`frontend/src/lib/repSpans.ts`：

```ts
// Frame bookkeeping shared by the coarse and dense extraction passes (RS-SP2 spec §2.5).
//
// WHY A FUNCTION AND NOT A COUNTER. poseExtract used to number frames with an incrementing
// counter, which equals round(t * 30) only when the sampling step happens to be 1/30. The coarse
// pass steps by 3/30, so a counter would number its samples 0,1,2,… while the video's frames are
// 0,3,6,… — every rep window derived from it would land in the wrong index space, silently. Both
// passes now derive the index from the TIMESTAMP, so they share one coordinate system.

/** The grid every frame_index is expressed on, matching poseExtract's fixed sampling cadence. */
export const CANONICAL_FPS = 30;

/** The frame_index of the sample at `t` seconds. */
export function frameIndexAt(t: number): number {
  return Math.round(t * CANONICAL_FPS);
}
```

- [ ] **Step 4: 跑測試確認通過**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSpans.test.ts`
預期：PASS（2 passed）

- [ ] **Step 5: 讓 `extractPoseFromBlob` 使用它**

`frontend/src/lib/poseExtract.ts`：把 `import type { PoseTier }` 那行下面加上
`import { CANONICAL_FPS, frameIndexAt } from "./repSpans";`，然後把抽取迴圈改成：

```ts
    await metadataReady;
    const fps = CANONICAL_FPS;
    // NOT `video.duration || 0` — a live-recorded clip reports no length and that silently sampled
    // nothing. See resolveDuration.
    const duration = await resolveDuration(video);
    // Seek-and-detect: step through the clip at a fixed cadence so frame_index is deterministic
    // and aligned to the stored video (rVFC live-rate would drift on drops). The index comes from
    // the TIMESTAMP, not a counter — the coarse pass steps differently and must agree. See
    // repSpans.frameIndexAt.
    for (let t = 0; t < duration; t += 1 / fps) {
      video.currentTime = t;
      await new Promise<void>((r) => { video.onseeked = () => r(); });
      const result = landmarker.detectForVideo(video, Math.round(t * 1000));
      frames.push(landmarksToFrame(frameIndexAt(t), result.landmarks?.[0], result.worldLandmarks?.[0]));
      onProgress?.(duration ? Math.min(1, t / duration) : 1);
    }
```

（同時刪掉 `let i = 0;` 與 `i += 1;` 兩行——計數器不再有人用。）

- [ ] **Step 6: 跑全部前端測試**

cwd = `frontend/`，執行：`yarn test`
預期：全部 PASS（`lib.poseExtract.test.ts` 不測那個迴圈，不受影響）

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/repSpans.ts frontend/src/lib/poseExtract.ts frontend/src/test/lib.repSpans.test.ts
git commit -m "fix(pose): derive frame_index from the timestamp, not a counter"
```

---

## Task 2：rep 訊號的 TS 移植（`repSignal.ts`）

**Spec:** §2.6、§2.7

**要照抄的 Python：** `src/pose/geometry.py:42-49`（`visible_point`）、`60-73`（`angle_degrees`，
**用 3 維，含 z**）、`101-105`（`mean_finite`）、`108-119`（`centered_median`）、
`src/pose/pose_rule_detector.py:130-141`（有效性 gating）、`143-144`+`167`（左右膝角平均）。

**Files:**
- Create: `frontend/src/lib/repSignal.ts`
- Test: `frontend/src/test/lib.repSignal.test.ts`

**Interfaces:**
- Consumes: 無
- Produces:
  - `interface SignalLandmark { x: number; y: number; z: number; visibility: number }`
  - `VISIBILITY_THRESHOLD: 0.5`
  - `centeredMedian(values: number[], window: number): number[]`
  - `avgKneeAngle(landmarks: SignalLandmark[] | null | undefined): number`
  - `TS_REP_SIGNALS: Record<string, (lm: SignalLandmark[] | null | undefined) => number>`

- [ ] **Step 1: 寫失敗的測試**

`frontend/src/test/lib.repSignal.test.ts`：

```ts
import { describe, it, expect } from "vitest";
import {
  avgKneeAngle,
  centeredMedian,
  TS_REP_SIGNALS,
  type SignalLandmark,
} from "../lib/repSignal";

// A 33-point skeleton where every landmark is visible and at the origin, so a test only has to
// place the four points avgKneeAngle reads (hips 23/24, knees 25/26, ankles 27/28).
function skeleton(overrides: Record<number, [number, number, number]>): SignalLandmark[] {
  const lms: SignalLandmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));
  for (const [index, [x, y, z]] of Object.entries(overrides)) {
    lms[Number(index)] = { x, y, z, visibility: 1 };
  }
  return lms;
}

// hip directly above knee, ankle directly below knee => a perfectly straight leg => 180 degrees.
const STRAIGHT = skeleton({
  23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],
  24: [1, 0, 0], 26: [1, 1, 0], 28: [1, 2, 0],
});
// hip above knee, ankle horizontally out from knee => a right angle => 90 degrees.
const BENT = skeleton({
  23: [0, 0, 0], 25: [0, 1, 0], 27: [1, 1, 0],
  24: [1, 0, 0], 26: [1, 1, 0], 28: [2, 1, 0],
});

describe("avgKneeAngle", () => {
  it("measures a straight leg as 180 degrees", () => {
    expect(avgKneeAngle(STRAIGHT)).toBeCloseTo(180, 4);
  });

  it("measures a right-angled knee as 90 degrees", () => {
    expect(avgKneeAngle(BENT)).toBeCloseTo(90, 4);
  });

  it("averages the two sides", () => {
    const mixed = skeleton({
      23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],   // left straight  => 180
      24: [1, 0, 0], 26: [1, 1, 0], 28: [2, 1, 0],   // right bent     => 90
    });
    expect(avgKneeAngle(mixed)).toBeCloseTo(135, 4);
  });

  it("is NaN when a required point is below the visibility threshold", () => {
    const hidden = skeleton({
      23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 2, 0],
      24: [1, 0, 0], 26: [1, 1, 0], 28: [1, 2, 0],
    });
    hidden[25] = { ...hidden[25], visibility: 0.49 };
    expect(Number.isNaN(avgKneeAngle(hidden))).toBe(true);
  });

  it("is NaN for a missing or short landmark list", () => {
    expect(Number.isNaN(avgKneeAngle(null))).toBe(true);
    expect(Number.isNaN(avgKneeAngle([]))).toBe(true);
  });

  it("uses z, matching Python's 3-D angle_degrees", () => {
    // Same x/y as STRAIGHT but the ankle pushed out in z: a 3-D measure must see the bend, a
    // 2-D one would still report 180.
    const inZ = skeleton({
      23: [0, 0, 0], 25: [0, 1, 0], 27: [0, 1, 1],
      24: [1, 0, 0], 26: [1, 1, 0], 28: [1, 1, 1],
    });
    expect(avgKneeAngle(inZ)).toBeCloseTo(90, 4);
  });
});

describe("centeredMedian", () => {
  it("smooths with a centred window and shrinks it at the edges", () => {
    expect(centeredMedian([1, 100, 1, 1, 1], 5)).toEqual([1, 1, 1, 1, 1]);
  });

  it("skips non-finite values instead of poisoning the window", () => {
    // The hole matters: RS-SP2 sends null frames, and geometry.py:117 skips them the same way.
    const out = centeredMedian([2, NaN, 4, NaN, 6], 3);
    expect(out[0]).toBe(2);
    expect(out[2]).toBe(4);
  });

  it("returns NaN where the whole window is non-finite", () => {
    expect(Number.isNaN(centeredMedian([NaN], 3)[0])).toBe(true);
  });

  it("returns an empty array for empty input", () => {
    expect(centeredMedian([], 5)).toEqual([]);
  });
});

describe("TS_REP_SIGNALS", () => {
  it("covers Squat and nothing else in SP2", () => {
    expect(Object.keys(TS_REP_SIGNALS)).toEqual(["Squat"]);
    expect(TS_REP_SIGNALS.Squat(STRAIGHT)).toBeCloseTo(180, 4);
  });
});
```

- [ ] **Step 2: 跑測試確認失敗**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSignal.test.ts`
預期：FAIL，`Failed to resolve import "../lib/repSignal"`

- [ ] **Step 3: 寫實作**

`frontend/src/lib/repSignal.ts`：

```ts
// The 1-D signal a movement's repetitions are found in, computed in the browser (RS-SP2 spec §2.6).
//
// PORTED, NOT REINVENTED. Every function here mirrors a specific Python one, because the backend
// trusts the rep windows this signal produces and would never see a disagreement (spec §2.3, §2.7):
//   avgKneeAngle   <- pose_rule_detector.py:143-144,167 + geometry.py:60-73 (angle_degrees, 3-D)
//   visibility gate<- geometry.py:42-49 (visible_point), threshold 0.50
//   centeredMedian <- geometry.py:108-119
// The shared fixture pins signal->windows; it does NOT pin that both sides compute the same signal.
// This file is the only thing that does, so change it only alongside its Python twin.

const LANDMARK_COUNT = 33;
/** geometry.py:7 — a landmark at or above this is trusted; below it the point does not exist. */
export const VISIBILITY_THRESHOLD = 0.5;

const LEFT_HIP = 23;
const RIGHT_HIP = 24;
const LEFT_KNEE = 25;
const RIGHT_KNEE = 26;
const LEFT_ANKLE = 27;
const RIGHT_ANKLE = 28;

export interface SignalLandmark { x: number; y: number; z: number; visibility: number }

type Point = [number, number, number];

/** geometry.py:42-49. Returns null for an absent, non-finite, or insufficiently visible point. */
function visiblePoint(lms: SignalLandmark[], index: number): Point | null {
  const lm = lms[index];
  if (!lm) return null;
  const { x, y, z, visibility } = lm;
  if (![x, y, z, visibility].every(Number.isFinite)) return null;
  if (visibility < VISIBILITY_THRESHOLD) return null;
  return [x, y, z];
}

/** geometry.py:60-73. The angle at `b` in degrees, in 3-D. NaN when any point is unusable. */
function angleDegrees(lms: SignalLandmark[], a: number, b: number, c: number): number {
  const pa = visiblePoint(lms, a);
  const pb = visiblePoint(lms, b);
  const pc = visiblePoint(lms, c);
  if (!pa || !pb || !pc) return NaN;
  const ba: Point = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
  const bc: Point = [pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]];
  const norm = (v: Point) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const denominator = norm(ba) * norm(bc);
  if (denominator <= 1e-8) return NaN;
  const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
  const cosine = Math.min(1, Math.max(-1, dot / denominator));
  return (Math.acos(cosine) * 180) / Math.PI;
}

/** geometry.py:101-105. Mean of the finite entries; NaN when there are none. */
function meanFinite(values: number[]): number {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return NaN;
  return finite.reduce((sum, v) => sum + v, 0) / finite.length;
}

/**
 * geometry.py:108-119. Median over a centred window, NON-FINITE ENTRIES SKIPPED.
 *
 * Skipping rather than propagating is what lets a padded span be smoothed at all: RS-SP2 leaves
 * holes in the frame sequence, and a NaN-propagating median would spread each hole by the window
 * radius. The window simply shrinks instead.
 */
export function centeredMedian(values: number[], window: number): number[] {
  if (values.length === 0) return [];
  const radius = Math.max(0, Math.floor(window / 2));
  return values.map((_, index) => {
    const start = Math.max(0, index - radius);
    const end = Math.min(values.length, index + radius + 1);
    const finite = values.slice(start, end).filter(Number.isFinite).sort((a, b) => a - b);
    if (finite.length === 0) return NaN;
    const mid = Math.floor(finite.length / 2);
    return finite.length % 2 === 1 ? finite[mid] : (finite[mid - 1] + finite[mid]) / 2;
  });
}

/**
 * The squat rep signal: the mean of the two knee angles.
 *
 * Mirrors raw_frame_metrics' validity rule (pose_rule_detector.py:133-141) — hips, knees and
 * ankles must all be visible or the frame has no metrics at all — and then its
 * `mean_finite([left_knee_angle, right_knee_angle])`.
 */
export function avgKneeAngle(landmarks: SignalLandmark[] | null | undefined): number {
  if (!landmarks || landmarks.length < LANDMARK_COUNT) return NaN;
  const required = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE];
  if (required.some((index) => visiblePoint(landmarks, index) === null)) return NaN;
  return meanFinite([
    angleDegrees(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    angleDegrees(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
  ]);
}

/**
 * Which movements can be segmented in the browser, keyed by the registry's canonical name.
 *
 * A movement ABSENT here is not broken — it takes the whole-clip fallback (spec §4.1,
 * `segmentation_disabled`) and behaves exactly as it does today. That is why SP2 can ship with
 * only Squat without blocking any other movement.
 */
export const TS_REP_SIGNALS: Record<string, (lm: SignalLandmark[] | null | undefined) => number> = {
  Squat: avgKneeAngle,
};
```

- [ ] **Step 4: 跑測試確認通過**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSignal.test.ts`
預期：PASS（11 passed）

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/repSignal.ts frontend/src/test/lib.repSignal.test.ts
git commit -m "feat(pose): port the squat rep signal to the browser"
```

---

## Task 3：切割器的 TS 移植（`repSegmentation.ts`）

**Spec:** §2.6、§2.7、§6

**要照抄的 Python：** `src/pose/rep_segmentation.py` 全檔。**先讀完那個檔案的模組註解與每個函式的
docstring 再動手**——裡面記錄了四次失敗的雜訊估計嘗試、以及 `_climb_backward` 為什麼必須用嚴格
大於，那些都是這份移植必須保留的語意。

**兩個會讓兩邊分歧的陷阱（都已實測確認）：**

1. **百分位**：numpy 的 `percentile` 用線性插值。驗證過 `np.percentile([1,2,3,4], 5) = 1.15`、
   `95 = 3.85`。TS 必須實作同一種插值，不能用「取第 k 個元素」。
2. **`select_reps` 的捨入**：Python `int(round(np.float64(2.5)))` 是**銀行家捨入**（→ 2），
   JS `Math.round(2.5)` → 3。實測分歧點：`n=6,k=3` Python 給 `[0,2,5]`、`Math.round` 給
   `[0,3,5]`；`n=10,k=3` Python `[0,4,9]`、`Math.round` `[0,5,9]`。必須實作 half-to-even。

**Files:**
- Create: `frontend/src/lib/repSegmentation.ts`
- Test: `frontend/src/test/lib.repSegmentation.test.ts`
- Read-only: `tests/fixtures/rep_segmentation_cases.json`（既有，SP1 產出）

**Interfaces:**
- Consumes: 無
- Produces:
  - `interface RepWindow { index: number; start: number; end: number; partial: boolean }`
  - `interface SegmentOptions { fps: number; polarity?: "min" | "max"; rectify?: boolean; repStart?: "extended" | "flexed"; minRepSeconds?: number }`
  - `segmentReps(signal: number[], options: SegmentOptions): RepWindow[]`
  - `selectReps(reps: RepWindow[], maxReps: number | null): RepWindow[]`
  - 常數 `PERCENTILE_LOW`、`PERCENTILE_HIGH`、`ENTER_FRACTION`、`EXIT_FRACTION`、`DEFAULT_MIN_REP_SECONDS`

- [ ] **Step 1: 寫失敗的測試（fixture 驅動 + 兩個陷阱）**

`frontend/src/test/lib.repSegmentation.test.ts`：

```ts
import { readFileSync } from "node:fs";
import { describe, it, expect } from "vitest";
import {
  DEFAULT_MIN_REP_SECONDS,
  ENTER_FRACTION,
  EXIT_FRACTION,
  PERCENTILE_HIGH,
  PERCENTILE_LOW,
  segmentReps,
  selectReps,
  type RepWindow,
} from "../lib/repSegmentation";

// The SAME file tests/test_rep_segmentation.py reads. Either implementation changing a threshold
// turns both suites red — that is the whole point of SP1 §7 having produced it. Resolved from
// import.meta.url, not cwd, so the test does not care where vitest was launched from.
interface FixtureCase {
  name: string;
  signal: number[];
  fps: number;
  polarity: "min" | "max";
  rectify: boolean;
  rep_start: "extended" | "flexed";
  min_rep_seconds: number;
  expected: { index: number; start: number; end: number; partial: boolean }[];
}
const fixture = JSON.parse(
  readFileSync(new URL("../../../tests/fixtures/rep_segmentation_cases.json", import.meta.url), "utf-8")
) as { cases: FixtureCase[] };

describe("segmentReps against the shared Python fixture", () => {
  it("has the fixture and it is not empty", () => {
    expect(fixture.cases.length).toBeGreaterThan(0);
  });

  for (const c of fixture.cases) {
    it(`matches Python on "${c.name}"`, () => {
      const got = segmentReps(c.signal, {
        fps: c.fps,
        polarity: c.polarity,
        rectify: c.rectify,
        repStart: c.rep_start,
        minRepSeconds: c.min_rep_seconds,
      });
      expect(got).toEqual(c.expected.map((e) => ({ ...e })));
    });
  }
});

describe("thresholds are named constants, matching rep_segmentation.py", () => {
  it("carries the Python values", () => {
    expect(PERCENTILE_LOW).toBe(5);
    expect(PERCENTILE_HIGH).toBe(95);
    expect(ENTER_FRACTION).toBe(0.35);
    expect(EXIT_FRACTION).toBe(0.65);
    expect(DEFAULT_MIN_REP_SECONDS).toBe(0.4);
  });
});

describe("segmentReps degenerate inputs", () => {
  it("returns [] for a flat signal (span == 0)", () => {
    expect(segmentReps(new Array(90).fill(5), { fps: 30 })).toEqual([]);
  });

  it("returns [] when there are fewer samples than two minimum reps", () => {
    expect(segmentReps([1, 2, 3], { fps: 30 })).toEqual([]);
  });

  it("rejects an unknown polarity rather than guessing", () => {
    // @ts-expect-error deliberately wrong, mirroring rep_segmentation.py:175-178
    expect(() => segmentReps([1, 2], { fps: 30, polarity: "sideways" })).toThrow(/polarity/);
  });
});

const win = (index: number, start: number, end: number, partial = false): RepWindow =>
  ({ index, start, end, partial });

describe("selectReps", () => {
  const five = [win(1, 0, 9), win(2, 10, 19), win(3, 20, 29), win(4, 30, 39), win(5, 40, 49)];

  it("takes first / middle / last from five reps", () => {
    expect(selectReps(five, 3).map((r) => r.index)).toEqual([1, 3, 5]);
  });

  it("takes everything when there are fewer than the cap", () => {
    expect(selectReps(five.slice(0, 2), 3).map((r) => r.index)).toEqual([1, 2]);
  });

  it("treats 0 and null as 'every rep'", () => {
    expect(selectReps(five, 0)).toHaveLength(5);
    expect(selectReps(five, null)).toHaveLength(5);
  });

  it("skips partial reps when complete ones exist", () => {
    const mixed = [win(1, 0, 9, true), win(2, 10, 19), win(3, 20, 29)];
    expect(selectReps(mixed, 3).map((r) => r.index)).toEqual([2, 3]);
  });

  it("keeps partial reps when they are all there is", () => {
    const allPartial = [win(1, 0, 9, true), win(2, 10, 19, true)];
    expect(selectReps(allPartial, 3).map((r) => r.index)).toEqual([1, 2]);
  });

  // THE TRAP. Python's int(round(...)) is banker's rounding; Math.round is not. Measured:
  // n=6,k=3 -> Python [0,2,5] but Math.round gives [0,3,5]; n=10,k=3 -> [0,4,9] vs [0,5,9].
  // Without half-to-even, a 6-rep and a 10-rep clip analyse DIFFERENT reps in the two languages.
  it("rounds half to even, like Python, on six reps", () => {
    const six = [...five, win(6, 50, 59)];
    expect(selectReps(six, 3).map((r) => r.index)).toEqual([1, 3, 6]);
  });

  it("rounds half to even, like Python, on ten reps", () => {
    const ten = Array.from({ length: 10 }, (_, i) => win(i + 1, i * 10, i * 10 + 9));
    expect(selectReps(ten, 3).map((r) => r.index)).toEqual([1, 5, 10]);
  });

  it("agrees with numpy.linspace on a tie that rounds UP", () => {
    // n=8,k=3 puts the middle at 3.5, which half-to-even sends to 4 -- the same way Math.round
    // would. Included because the two rules only differ in one direction, and a port that got
    // the direction backwards would still pass the two cases above.
    const eight = Array.from({ length: 8 }, (_, i) => win(i + 1, i * 10, i * 10 + 9));
    expect(selectReps(eight, 3).map((r) => r.index)).toEqual([1, 5, 8]);
  });

  it("returns [] for no reps", () => {
    expect(selectReps([], 3)).toEqual([]);
  });
});
```

- [ ] **Step 2: 跑測試確認失敗**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSegmentation.test.ts`
預期：FAIL，`Failed to resolve import "../lib/repSegmentation"`

- [ ] **Step 3: 寫實作**

`frontend/src/lib/repSegmentation.ts`：

```ts
// Split a movement clip into repetitions from a single 1-D metric series.
//
// A LINE-BY-LINE PORT of src/pose/rep_segmentation.py, which is the authority: read its module
// docstring for why there is deliberately NO noise-vs-range gate (four attempts at one each
// false-rejected ordinary training signal), and why _climbBackward's strict `>` is load-bearing.
// tests/fixtures/rep_segmentation_cases.json pins both implementations to the same outputs.
//
// TWO PORTING TRAPS, both measured, both covered by tests:
//   - numpy's percentile interpolates linearly (np.percentile([1,2,3,4], 5) === 1.15). Picking the
//     k-th element instead shifts the hysteresis band and moves every boundary.
//   - Python's int(round(x)) is banker's rounding. Math.round is not, and the two disagree on
//     6- and 10-rep clips, which would make the languages analyse different repetitions.

/** Robust bounds for the signal's dynamic range, so one bad frame cannot define it. */
export const PERCENTILE_LOW = 5;
export const PERCENTILE_HIGH = 95;
/** Hysteresis band, as fractions of the dynamic range from the effort-peak end. */
export const ENTER_FRACTION = 0.35;
export const EXIT_FRACTION = 0.65;
/** Floor on repetition duration — the ONLY thing separating a real excursion from a blip. */
export const DEFAULT_MIN_REP_SECONDS = 0.4;

const POLARITIES = ["min", "max"] as const;
const REP_STARTS = ["extended", "flexed"] as const;
export type Polarity = (typeof POLARITIES)[number];
export type RepStart = (typeof REP_STARTS)[number];

export interface RepWindow {
  /** 1-based: it is what a user is told ("your 3rd rep"). */
  index: number;
  /** Inclusive POSITION IN THE PASSED SEQUENCE, not a frame_index. */
  start: number;
  end: number;
  partial: boolean;
}

export interface SegmentOptions {
  fps: number;
  polarity?: Polarity;
  rectify?: boolean;
  repStart?: RepStart;
  minRepSeconds?: number;
}

/** numpy.percentile's default linear interpolation, over values that are already sorted. */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 1) return sorted[0];
  const position = ((sorted.length - 1) * p) / 100;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/** Python's round(): ties go to the even integer, unlike Math.round which always goes up. */
function roundHalfToEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Normalise any movement's signal to the convention "the effort peak is a LOW value". */
function oriented(signal: number[], polarity: Polarity, rectify: boolean): number[] {
  return signal.map((value) => {
    // A bipolar signal (torso twist: centre -> A -> centre -> B) has two excursions in opposite
    // directions. Rectifying makes each swing its own excursion from zero.
    const v = rectify ? Math.abs(value) : value;
    return polarity === "max" ? -v : v;
  });
}

/** Maximal inclusive runs at/below `threshold`, skipping non-finite samples. */
function runsAtOrBelow(values: number[], threshold: number): [number, number][] {
  const runs: [number, number][] = [];
  let start: number | null = null;
  values.forEach((value, index) => {
    if (!Number.isFinite(value)) return; // an unmeasurable frame neither opens nor closes a run
    if (value <= threshold && start === null) start = index;
    else if (value > threshold && start !== null) {
      runs.push([start, index - 1]);
      start = null;
    }
  });
  if (start !== null) runs.push([start, values.length - 1]);
  return runs;
}

function lastAtOrAbove(values: number[], threshold: number, before: number): number | null {
  for (let i = before - 1; i >= 0; i -= 1) {
    if (Number.isFinite(values[i]) && values[i] >= threshold) return i;
  }
  return null;
}

function firstAtOrAbove(values: number[], threshold: number, after: number): number | null {
  for (let i = after + 1; i < values.length; i += 1) {
    if (Number.isFinite(values[i]) && values[i] >= threshold) return i;
  }
  return null;
}

/**
 * Walk back from an exit crossing to the top of the excursion, STOPPING AT A PLATEAU.
 *
 * The strict `>` is what makes a window's length equal the excursion's length rather than the
 * clip's, which in turn is what lets `finalize`'s min-frames filter reject a blip on duration
 * alone. See the Python docstring for the full argument.
 */
function climbBackward(values: number[], index: number): number {
  let i = index;
  while (i > 0 && Number.isFinite(values[i - 1]) && values[i - 1] > values[i]) i -= 1;
  return i;
}

function climbForward(values: number[], index: number): number {
  let i = index;
  const last = values.length - 1;
  while (i < last && Number.isFinite(values[i + 1]) && values[i + 1] > values[i]) i += 1;
  return i;
}

/** The full extent of the excursion a deep run belongs to: top, through the bottom, to top. */
function excursionBounds(
  values: number[], deepStart: number, deepEnd: number, exit: number
): [number, number, boolean] {
  const before = lastAtOrAbove(values, exit, deepStart);
  const after = firstAtOrAbove(values, exit, deepEnd);
  const start = before === null ? 0 : climbBackward(values, before);
  const end = after === null ? values.length - 1 : climbForward(values, after);
  return [start, end, before === null || after === null];
}

/** Boundaries at the EXTENDED end: a rep runs standing -> bottom -> standing. */
function windowsFromPlateaus(
  values: number[], deepRuns: [number, number][], exit: number, minFrames: number
): RepWindow[] {
  return finalize(deepRuns.map(([s, e]) => excursionBounds(values, s, e, exit)), minFrames);
}

/**
 * Boundaries at the FLEXED end: a rep runs floor -> lockout -> floor (deadlift).
 *
 * Filters each deep run on the duration of the excursion it belongs to, because a valley-to-valley
 * window's length is the rep PERIOD, not any one excursion — without this the flexed path has no
 * anomaly rejection at all. See the Python docstring for the boundary-run trade this accepts.
 */
function windowsFromValleys(
  values: number[], deepRuns: [number, number][], exit: number, minFrames: number
): RepWindow[] {
  const realRuns = deepRuns.filter(([s, e]) => {
    const [start, end] = excursionBounds(values, s, e, exit);
    return end - start + 1 >= minFrames;
  });
  if (realRuns.length === 0) return [];

  const valleys = realRuns.map(([s, e]) => {
    let best = s;
    let bestValue = Infinity;
    for (let i = s; i <= e; i += 1) {
      if (Number.isFinite(values[i]) && values[i] < bestValue) { bestValue = values[i]; best = i; }
    }
    return best;
  });

  const spans: [number, number, boolean][] = [];
  if (valleys[0] > 0) spans.push([0, valleys[0] - 1, true]);
  for (let i = 0; i + 1 < valleys.length; i += 1) spans.push([valleys[i], valleys[i + 1] - 1, false]);
  if (valleys[valleys.length - 1] < values.length - 1) {
    spans.push([valleys[valleys.length - 1], values.length - 1, true]);
  }
  return finalize(spans, minFrames);
}

/** De-duplicate, resolve shared boundaries, drop too-short spans, and number the rest. */
function finalize(spans: [number, number, boolean][], minFrames: number): RepWindow[] {
  const sorted = [...spans].sort((a, b) =>
    a[0] - b[0] || a[1] - b[1] || Number(a[2]) - Number(b[2]));

  const unique: [number, number, boolean][] = [];
  const seen = new Set<string>();
  for (const span of sorted) {
    const key = `${span[0]}:${span[1]}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(span);
  }

  const windows: RepWindow[] = [];
  unique.forEach(([start, rawEnd, partial], position) => {
    // Adjacent reps meet at a single frame — the peak between them belongs to the rep that STARTS
    // there — so the earlier window gives it up, or one frame is phased and scored twice.
    const end = position + 1 < unique.length ? Math.min(rawEnd, unique[position + 1][0] - 1) : rawEnd;
    if (end - start + 1 < minFrames) return;
    windows.push({ index: windows.length + 1, start, end, partial });
  });
  return windows;
}

/**
 * Segment `signal` into repetitions.
 *
 * Returns `[]` — never a guess — when the signal carries no repetition structure. The caller is
 * required to fall back to whole-clip analysis in that case, NOT to report no faults.
 */
export function segmentReps(signal: number[], options: SegmentOptions): RepWindow[] {
  const polarity = options.polarity ?? "min";
  const repStart = options.repStart ?? "extended";
  const minRepSeconds = options.minRepSeconds ?? DEFAULT_MIN_REP_SECONDS;
  if (!POLARITIES.includes(polarity)) throw new Error(`polarity must be min or max, got ${polarity}`);
  if (!REP_STARTS.includes(repStart)) throw new Error(`repStart must be extended or flexed, got ${repStart}`);

  const values = oriented(signal, polarity, options.rectify ?? false);
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  const minFrames = Math.max(3, roundHalfToEven(minRepSeconds * Math.max(options.fps, 1)));
  if (finite.length < 2 * minFrames) return [];

  const low = percentile(finite, PERCENTILE_LOW);
  const high = percentile(finite, PERCENTILE_HIGH);
  const span = high - low;
  if (span <= 0) return [];

  const enter = low + ENTER_FRACTION * span;
  const exit = low + EXIT_FRACTION * span;
  const deepRuns = runsAtOrBelow(values, enter);
  if (deepRuns.length === 0) return [];

  return repStart === "flexed"
    ? windowsFromValleys(values, deepRuns, exit, minFrames)
    : windowsFromPlateaus(values, deepRuns, exit, minFrames);
}

/**
 * Choose which repetitions to actually analyze: first / middle / last.
 *
 * Not "the first N": the first rep carries warm-up errors, the middle one steady state, the last
 * one fatigue breakdown, and sampling only the middle systematically hides the fault a lifter most
 * needs told. Partial reps are skipped when complete ones exist, but kept when they are all there
 * is — analyzing a truncated rep beats analyzing nothing.
 */
export function selectReps(reps: RepWindow[], maxReps: number | null): RepWindow[] {
  const complete = reps.filter((rep) => !rep.partial);
  const candidates = complete.length > 0 ? complete : [...reps];
  if (candidates.length === 0) return [];
  if (!maxReps || maxReps <= 0 || candidates.length <= maxReps) return candidates;

  const last = candidates.length - 1;
  const positions = new Set<number>();
  for (let i = 0; i < maxReps; i += 1) {
    positions.add(roundHalfToEven((last * i) / (maxReps - 1)));
  }
  return [...positions].sort((a, b) => a - b).map((position) => candidates[position]);
}
```

- [ ] **Step 4: 跑測試確認通過**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSegmentation.test.ts`
預期：PASS（fixture 11 個 case + 其餘共 24 passed）

若某個 fixture case 不過：**不要改門檻去湊**。回去讀 `src/pose/rep_segmentation.py` 對應的函式，
差異幾乎必定在 `percentile` 插值、`finalize` 的排序，或 `climbBackward` 的嚴格大於。

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/repSegmentation.ts frontend/src/test/lib.repSegmentation.test.ts
git commit -m "feat(pose): port the rep segmenter to TypeScript against the shared fixture"
```

---

## Task 4：Span 規劃與邊界精修（`repSpans.ts`）

**Spec:** §2.1.1、§2.8、§3

**這個 Task 是 SP2 的核心決定**：span 錨在**谷底**而不是粗掃邊界。實測（46 clips / 70 reps）
粗掃邊界誤差 p95 = 15、max = 45 幀，谷底誤差 max = 5 幀——差兩個數量級，且加密粗掃救不了邊界誤差。

**Files:**
- Modify: `frontend/src/lib/repSpans.ts`（Task 1 建立）
- Test: `frontend/src/test/lib.repSpans.test.ts`（Task 1 建立）

**Interfaces:**
- Consumes: `segmentReps`、`RepWindow`（Task 3）；`centeredMedian`（Task 2）
- Produces:
  - `COARSE_STRIDE: 3`、`REP_PADDING_FRAMES: 24`、`COARSE_SMOOTH_WINDOW: 3`、`DENSE_SMOOTH_WINDOW: 5`
  - `valleyPosition(signal: number[], window: RepWindow): number`
  - `interface FrameSpan { start: number; end: number }`
  - `spanForRep(coarseSignal: number[], rep: RepWindow, lastFrameIndex: number): FrameSpan`
  - `mergeSpans(spans: FrameSpan[]): FrameSpan[]`
  - `spanFrameIndices(spans: FrameSpan[]): number[]`
  - `type Refinement = true | false | "clipped"`
  - `refineWindow(denseSignal: (number | undefined)[], span: FrameSpan, coarse: FrameSpan, fps: number, lastFrameIndex: number): { start: number; end: number; refined: Refinement }`

**精修的效果已實測**（同一組 46 clips / 70 reps）：精修後的邊界與「整片密集抽取」的邊界
**95.7% 完全相同**（0 幀誤差），p95 = 0、max = 27（就是那唯一一個被切到的 rep）。粗掃邊界則是
p50 2 / p95 21 / max 45。換句話說，精修不是理論上的改進，它把誤差消到 0。

**`clipped` 的判定有一個容易寫錯的地方**：window 貼齊 span 邊緣時，若那個邊緣**同時是片子的
邊緣**，就不是被切到——外面根本沒有東西可抽。用錯的定義量出 43% 被切；排除片緣後是 **1.4%**，
與 §2.8 的 98.6% 涵蓋率吻合。所以 `refineWindow` 必須知道 `lastFrameIndex`。

- [ ] **Step 1: 寫失敗的測試（附加到 `lib.repSpans.test.ts`）**

```ts
import {
  COARSE_STRIDE,
  REP_PADDING_FRAMES,
  mergeSpans,
  refineWindow,
  spanForRep,
  spanFrameIndices,
  valleyPosition,
} from "../lib/repSpans";
import { segmentReps } from "../lib/repSegmentation";

// A cosine rep: `count` excursions from 170 degrees down to 60 and back, `period` samples each.
function repSignal(count: number, period: number): number[] {
  return Array.from({ length: count * period }, (_, i) =>
    115 + 55 * Math.cos((2 * Math.PI * (i % period)) / period));
}

describe("valleyPosition", () => {
  it("finds the deepest sample inside the window", () => {
    const signal = [10, 8, 3, 8, 10];
    expect(valleyPosition(signal, { index: 1, start: 0, end: 4, partial: false })).toBe(2);
  });

  it("ignores non-finite samples rather than returning one", () => {
    const signal = [10, NaN, 3, NaN, 10];
    expect(valleyPosition(signal, { index: 1, start: 0, end: 4, partial: false })).toBe(2);
  });
});

describe("spanForRep", () => {
  it("anchors on the valley and spans the coarse half-width plus the padding", () => {
    // 30 coarse samples, valley at 15 => frame 45 on the canonical grid; half-width 15 samples
    // => 45 frames; span = 45 +/- (45 + 24).
    const coarse = repSignal(1, 30);
    const [rep] = segmentReps(coarse, { fps: 10 });
    const span = spanForRep(coarse, rep, 89);
    const valleyFrame = valleyPosition(coarse, rep) * COARSE_STRIDE;
    expect(span.start).toBeLessThan(valleyFrame);
    expect(span.end).toBeGreaterThan(valleyFrame);
    expect(valleyFrame - span.start).toBe(span.end - valleyFrame);
  });

  it("clamps to the clip", () => {
    const coarse = repSignal(1, 30);
    const [rep] = segmentReps(coarse, { fps: 10 });
    const span = spanForRep(coarse, rep, 89);
    expect(span.start).toBeGreaterThanOrEqual(0);
    expect(span.end).toBeLessThanOrEqual(89);
  });

  it("pads by the measured constant", () => {
    expect(REP_PADDING_FRAMES).toBe(24);
    expect(COARSE_STRIDE).toBe(3);
  });
});

describe("mergeSpans", () => {
  it("merges overlapping spans so no frame is extracted twice", () => {
    expect(mergeSpans([{ start: 0, end: 50 }, { start: 40, end: 90 }])).toEqual([{ start: 0, end: 90 }]);
  });

  it("merges spans that only touch", () => {
    expect(mergeSpans([{ start: 0, end: 10 }, { start: 11, end: 20 }])).toEqual([{ start: 0, end: 20 }]);
  });

  it("keeps disjoint spans apart and sorts them", () => {
    expect(mergeSpans([{ start: 60, end: 80 }, { start: 0, end: 10 }]))
      .toEqual([{ start: 0, end: 10 }, { start: 60, end: 80 }]);
  });

  it("returns [] for no spans", () => {
    expect(mergeSpans([])).toEqual([]);
  });
});

describe("spanFrameIndices", () => {
  it("enumerates every frame in every span, once, in order", () => {
    expect(spanFrameIndices([{ start: 0, end: 2 }, { start: 5, end: 6 }])).toEqual([0, 1, 2, 5, 6]);
  });
});

describe("refineWindow", () => {
  // A dense 90-frame rep sitting inside a 200-frame clip, spanned from frame 0 to 119.
  const LAST = 199;
  const dense: (number | undefined)[] = new Array(LAST + 1).fill(undefined);
  repSignal(1, 90).forEach((v, i) => { dense[i + 15] = v; });

  it("recovers the dense boundary from a coarse one that is 10 frames late", () => {
    const out = refineWindow(dense, { start: 0, end: 119 }, { start: 25, end: 100 }, 30, LAST);
    expect(out.refined).toBe(true);
    expect(Math.abs(out.start - 15)).toBeLessThanOrEqual(2);
  });

  it("reports 'clipped' when the span cut the rep off mid-clip", () => {
    // The span ends at 89 while the clip runs to 199, so there WAS more to extract and the
    // padding was too small — the one case that must stay visible.
    const out = refineWindow(dense, { start: 15, end: 89 }, { start: 20, end: 85 }, 30, LAST);
    expect(out.refined).toBe("clipped");
  });

  it("does NOT report 'clipped' when the span edge is the clip's own edge", () => {
    // Nothing exists beyond the clip, so touching that edge is not a padding failure. Measured:
    // conflating the two reported 43% of real reps as clipped instead of the true 1.4%.
    const flush: (number | undefined)[] = new Array(90).fill(undefined);
    repSignal(1, 90).forEach((v, i) => { flush[i] = v; });
    const out = refineWindow(flush, { start: 0, end: 89 }, { start: 0, end: 89 }, 30, 89);
    expect(out.refined).toBe(true);
  });

  it("picks the window overlapping the coarse one when padding caught a neighbour", () => {
    // mergeSpans can fuse two adjacent reps into one span, so the span legitimately holds two
    // windows and the overlap tiebreak decides — it is load-bearing, not a safety net.
    const two: (number | undefined)[] = new Array(LAST + 1).fill(undefined);
    repSignal(2, 90).forEach((v, i) => { two[i] = v; });
    const out = refineWindow(two, { start: 0, end: 179 }, { start: 95, end: 175 }, 30, LAST);
    expect(out.start).toBeGreaterThanOrEqual(80);
  });

  it("falls back to the coarse boundary when the span holds no window", () => {
    const flat: (number | undefined)[] = new Array(LAST + 1).fill(5);
    const out = refineWindow(flat, { start: 0, end: 119 }, { start: 20, end: 100 }, 30, LAST);
    expect(out).toEqual({ start: 20, end: 100, refined: false });
  });
});
```

- [ ] **Step 2: 跑測試確認失敗**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSpans.test.ts`
預期：FAIL，`valleyPosition is not a function`（或同類的匯出缺失）

- [ ] **Step 3: 寫實作（附加到 `repSpans.ts`）**

```ts
import { centeredMedian } from "./repSignal";
import { segmentReps, type RepWindow } from "./repSegmentation";

/** Coarse pass samples every Nth frame of the canonical grid, so sample k IS frame k*N. */
export const COARSE_STRIDE = 3;
/** Smoothing windows. Aligned by TIME, not frame count: SP1's 5 frames at 30fps is 0.17s, and
 *  reusing 5 at the coarse 10fps would smooth over 0.5s and flatten a squat's bottom. */
export const DENSE_SMOOTH_WINDOW = 5;
export const COARSE_SMOOTH_WINDOW = 3;

/**
 * How far past the coarse window's own half-width a span must reach (spec §2.8).
 *
 * MEASURED, not chosen: across 46 real squat pose JSONs (70 reps), a span of
 * `valley +/- (coarseHalf + 24)` contained the dense-derived window for 98.6% of reps; 8 frames
 * covered 95.7% and 32 covered all 70. The remaining 1.4% surface as refined:"clipped" rather than
 * silently losing part of a rep.
 *
 * WHY THE SPAN IS ANCHORED ON THE VALLEY. The same measurement put the coarse-vs-dense BOUNDARY
 * error at p95 15 frames and max 45, and sweeping the stride from 2 to 6 barely moved it — the
 * error comes from the hysteresis band's percentiles shifting with the sample distribution, not
 * from resolution, so a denser coarse pass cannot fix it. The VALLEY, an argmin rather than a
 * threshold crossing, landed within 5 frames every time. Anchoring there is what turns the padding
 * constant from "absorb a 36-frame tail" into "absorb a 7-frame one".
 */
export const REP_PADDING_FRAMES = 24;

export interface FrameSpan { start: number; end: number }

/** The position of the deepest sample in `window`, skipping non-finite samples. */
export function valleyPosition(signal: number[], window: RepWindow): number {
  let best = window.start;
  let bestValue = Infinity;
  for (let i = window.start; i <= window.end; i += 1) {
    if (Number.isFinite(signal[i]) && signal[i] < bestValue) { bestValue = signal[i]; best = i; }
  }
  return best;
}

/** The frames to extract densely for one coarse-detected rep, in canonical frame_index space. */
export function spanForRep(
  coarseSignal: number[], rep: RepWindow, lastFrameIndex: number
): FrameSpan {
  const valleyFrame = valleyPosition(coarseSignal, rep) * COARSE_STRIDE;
  // floor, not ceil: the 98.6% coverage figure REP_PADDING_FRAMES is set from was measured with
  // Python's `(end - start + 1) * STRIDE // 2`, and tests/test_coarse_segmentation_corpus.py
  // re-measures it the same way. A one-frame difference is immaterial next to a 24-frame pad, but
  // the constant's justification is only reproducible if both sides compute the span identically.
  const coarseHalf = Math.floor(((rep.end - rep.start + 1) * COARSE_STRIDE) / 2);
  const half = coarseHalf + REP_PADDING_FRAMES;
  return {
    start: Math.max(0, valleyFrame - half),
    end: Math.min(lastFrameIndex, valleyFrame + half),
  };
}

/** Union of overlapping or touching spans, sorted — so no frame is ever extracted twice. */
export function mergeSpans(spans: FrameSpan[]): FrameSpan[] {
  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: FrameSpan[] = [];
  for (const span of sorted) {
    const last = merged[merged.length - 1];
    if (last && span.start <= last.end + 1) last.end = Math.max(last.end, span.end);
    else merged.push({ ...span });
  }
  return merged;
}

/** Every frame_index covered by `spans`, ascending. */
export function spanFrameIndices(spans: FrameSpan[]): number[] {
  const indices: number[] = [];
  for (const span of spans) {
    for (let i = span.start; i <= span.end; i += 1) indices.push(i);
  }
  return indices;
}

export type Refinement = true | false | "clipped";

/**
 * Re-derive a rep's boundary from the DENSE signal inside its extracted span (spec §2.1.1).
 *
 * The coarse boundary can be tens of frames off (see REP_PADDING_FRAMES), and `assign_phases`
 * takes a window's first 15% as setup — so a start that lands late puts "setup" mid-descent,
 * which is exactly the bug SP1 exists to fix, arriving by a new route. Padding cannot correct
 * that; only measuring the boundary on data that exists at full rate can.
 *
 * MEASURED, on the same 46 clips REP_PADDING_FRAMES came from: the refined boundary equals the
 * whole-clip dense boundary EXACTLY for 95.7% of reps (p95 0 frames, max 27), against the coarse
 * boundary's p50 2 / p95 21 / max 45. Re-deriving the hysteresis band from one span's percentiles
 * rather than the whole clip's was the obvious worry here, and it does not materialise.
 *
 * `denseSignal` is indexed by frame_index, with `undefined` wherever nothing was extracted.
 * `lastFrameIndex` is the clip's own end, and it is not optional bookkeeping: a window touching a
 * span edge that IS the clip edge has not been clipped -- nothing exists beyond it to extract.
 * Conflating the two reported 43% of real reps as clipped instead of the true 1.4%.
 */
export function refineWindow(
  denseSignal: (number | undefined)[], span: FrameSpan, coarse: FrameSpan,
  fps: number, lastFrameIndex: number
): { start: number; end: number; refined: Refinement } {
  const slice = [];
  for (let i = span.start; i <= span.end; i += 1) {
    const value = denseSignal[i];
    slice.push(value === undefined ? NaN : value);
  }
  const windows = segmentReps(centeredMedian(slice, DENSE_SMOOTH_WINDOW), { fps });
  if (windows.length === 0) return { start: coarse.start, end: coarse.end, refined: false };

  // The coarse window says WHICH rep this span is about; pick the refined window that overlaps it
  // most, so a neighbouring rep caught by the padding cannot steal the boundary.
  const best = windows.reduce((a, b) => (overlap(b, span, coarse) > overlap(a, span, coarse) ? b : a));
  const start = span.start + best.start;
  const end = span.start + best.end;
  const clipped =
    (start === span.start && span.start > 0) ||
    (end === span.end && span.end < lastFrameIndex);
  return { start, end, refined: clipped ? "clipped" : true };
}

function overlap(window: RepWindow, span: FrameSpan, coarse: FrameSpan): number {
  const start = span.start + window.start;
  const end = span.start + window.end;
  return Math.max(0, Math.min(end, coarse.end) - Math.max(start, coarse.start) + 1);
}
```

- [ ] **Step 4: 跑測試確認通過**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSpans.test.ts`
預期：PASS（14 passed）

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/repSpans.ts frontend/src/test/lib.repSpans.test.ts
git commit -m "feat(pose): anchor extraction spans on the rep valley and refine on dense data"
```

---

## Task 5：真實語料的粗掃迴歸測試（Python）

**Spec:** §2.8、§6

**為什麼是 Python 而不是 vitest**：真實 pose JSON 在 `data/` 下且被 gitignore，Python 這側已經有
`compute_raw`。這個測試釘住的是**演算法的行為**，不是 TS 移植——它會在有人改切割器或改
`REP_PADDING_FRAMES` 時變紅。比照 `tests/test_view_regression_corpus.py` 的 `skipUnless` 作法。

**合成 fixture 已經證明它會低估**（§2.8 第一輪說 ≤3 幀，真實資料是 p95 15 / max 45），所以這個
測試不可省略。

**Files:**
- Create: `tests/test_coarse_segmentation_corpus.py`

**Interfaces:**
- Consumes: `src.pose.movements.registry`、`src.pose.geometry.centered_median`、
  `src.pose.rep_segmentation.segment_reps`
- Produces: 無（純測試）

- [ ] **Step 1: 寫測試**

```python
"""Pin what the coarse pass costs, measured on real clips rather than synthetic ones.

RS-SP2 samples every third frame to find repetitions, then extracts only the selected ones
densely. This file pins the two quantities that design rests on -- and exists because the
SYNTHETIC fixture badly underestimated both: decimating tests/fixtures/rep_segmentation_cases.json
suggested boundary error stayed within 3 frames, while real footage puts p95 at 15 and max at 45.

The measurements below (46 clips, 70 reps) are what REP_PADDING_FRAMES = 24 and the valley anchor
in frontend/src/lib/repSpans.ts are derived from. If this test goes red, that constant is wrong.

Data lives under data/ and is gitignored, so this skips in CI and bites locally -- the same
arrangement as tests/test_view_regression_corpus.py.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.pose.geometry import centered_median
from src.pose.movements import registry
from src.pose.rep_segmentation import RepWindow, segment_reps

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIRS = (
    REPO_ROOT / "data" / "runtime" / "pose_json",
    REPO_ROOT / "data" / "Fitness-AQA" / "Squat" / "Labeled_Dataset" / "pose_json",
)

# Kept in sync with frontend/src/lib/repSpans.ts -- these ARE those constants.
COARSE_STRIDE = 3
COARSE_SMOOTH_WINDOW = 3
DENSE_SMOOTH_WINDOW = 5
REP_PADDING_FRAMES = 24

MIN_CLIPS = 20            # below this the percentiles below mean nothing
MAX_VALLEY_ERROR = 5      # measured max across 70 reps
MIN_SPAN_COVERAGE = 0.98  # measured 98.6%
MAX_COUNT_MISMATCHES = 2
MIN_REFINED_EXACT = 0.95  # measured 95.7% of reps refine to the whole-clip boundary exactly
MAX_REFINED_P95 = 0.0     # measured: p95 of the refined error is zero frames
MAX_CLIPPED_SHARE = 0.03  # measured 1.4%


def _clips() -> list[tuple[str, float, list[float]]]:
    detector = registry.get_detector("Squat")
    out: list[tuple[str, float, list[float]]] = []
    for directory in CORPUS_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            frames = payload.get("frames") or []
            if len(frames) < 40:
                continue
            fps = float((payload.get("metadata") or {}).get("fps", 30.0) or 30.0)
            raw = detector.compute_raw(frames, fps)
            out.append((path.name, fps, [float(r.get("avg_knee_angle", np.nan)) for r in raw]))
    return out


def _valley(signal, window: RepWindow) -> int:
    chunk = np.asarray(signal[window.start : window.end + 1], dtype=float)
    return window.start + int(np.nanargmin(np.where(np.isfinite(chunk), chunk, np.inf)))


CLIPS = _clips()


@unittest.skipUnless(len(CLIPS) >= MIN_CLIPS, f"needs >= {MIN_CLIPS} local squat pose JSONs")
class CoarseSegmentationCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valley_errors: list[int] = []
        self.refined_errors: list[int] = []
        self.covered = 0
        self.clipped = 0
        self.total_reps = 0
        self.mismatches = 0
        for _name, fps, values in CLIPS:
            dense_signal = centered_median(values, window=DENSE_SMOOTH_WINDOW)
            coarse_signal = centered_median(values[::COARSE_STRIDE], window=COARSE_SMOOTH_WINDOW)
            dense = segment_reps(dense_signal, fps=fps)
            coarse = segment_reps(coarse_signal, fps=fps / COARSE_STRIDE)
            if len(dense) != len(coarse):
                self.mismatches += 1
                continue
            last = len(values) - 1
            for d, c in zip(dense, coarse):
                self.total_reps += 1
                valley = _valley(coarse_signal, c) * COARSE_STRIDE
                self.valley_errors.append(abs(_valley(dense_signal, d) - valley))
                half = (c.end - c.start + 1) * COARSE_STRIDE // 2 + REP_PADDING_FRAMES
                if valley - half <= d.start and d.end <= valley + half:
                    self.covered += 1

                # Refinement: re-segment the DENSE signal restricted to the padded span, then take
                # the window overlapping the coarse one most (padding can catch a neighbour).
                span_start, span_end = max(0, valley - half), min(last, valley + half)
                windows = segment_reps(dense_signal[span_start : span_end + 1], fps=fps)
                if not windows:
                    continue
                coarse_start, coarse_end = c.start * COARSE_STRIDE, c.end * COARSE_STRIDE
                best = max(windows, key=lambda w: max(
                    0, min(span_start + w.end, coarse_end) - max(span_start + w.start, coarse_start) + 1))
                start, end = span_start + best.start, span_start + best.end
                self.refined_errors.append(max(abs(start - d.start), abs(end - d.end)))
                if (start == span_start and span_start > 0) or (end == span_end and span_end < last):
                    self.clipped += 1

    def test_valley_location_survives_decimation(self) -> None:
        """The span's anchor. If this drifts, anchoring on the valley stops being the cheap option."""
        self.assertLessEqual(max(self.valley_errors), MAX_VALLEY_ERROR)

    def test_padding_contains_the_dense_window(self) -> None:
        """REP_PADDING_FRAMES = 24 is justified by THIS number and nothing else."""
        coverage = self.covered / self.total_reps
        self.assertGreaterEqual(coverage, MIN_SPAN_COVERAGE, f"coverage {coverage:.3f}")

    def test_coarse_rep_count_rarely_disagrees(self) -> None:
        """A miscount is the one error neither padding nor refinement can repair: a missed rep is
        never extracted, and the user is told a rep count that is simply wrong (spec §8)."""
        self.assertLessEqual(self.mismatches, MAX_COUNT_MISMATCHES)

    def test_refinement_recovers_the_whole_clip_boundary(self) -> None:
        """The claim refinement rests on, and the obvious way it could fail.

        Refining re-derives the hysteresis band from ONE span's percentiles rather than the whole
        clip's, which is narrower and could plausibly shift the boundary -- the very thing
        refinement exists to get right, since assign_phases takes a window's first 15% as setup.
        Measured on this corpus it does not: the refined boundary is EXACTLY the whole-clip one for
        95.7% of reps, against the coarse boundary's p50 2 / p95 21 / max 45 frames. If this goes
        red, refinement needs the whole-clip band passed in rather than re-derived per span.
        """
        exact = sum(1 for error in self.refined_errors if error == 0)
        self.assertGreaterEqual(exact / len(self.refined_errors), MIN_REFINED_EXACT)
        self.assertLessEqual(float(np.percentile(self.refined_errors, 95)), MAX_REFINED_P95)

    def test_spans_rarely_cut_a_rep_short(self) -> None:
        """`clipped` counts ONLY a span edge that is not also the clip's edge -- nothing exists
        beyond the clip to extract, so touching that is not a padding failure. Conflating the two
        reported 43% of these reps as clipped instead of the true 1.4%."""
        self.assertLessEqual(self.clipped / self.total_reps, MAX_CLIPPED_SHARE)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑測試**

`.venv\Scripts\python.exe -m pytest tests/test_coarse_segmentation_corpus.py -v`
預期：5 passed（本機有資料）。若 skip，先確認 `data/runtime/pose_json` 存在。

**這五個斷言的數字全部已經量過**（46 clips / 70 reps）：谷底誤差 max 5、span 涵蓋 98.6%、
數量不一致 2 個 clip、精修完全命中 95.7%（p95 = 0 幀）、真正被切到 1.4%。所以這個測試應該
**第一次跑就綠**。若不綠，是實作與量測時的參數不一致，不是門檻訂太嚴。

- [ ] **Step 3: 確認整套沒被影響**

`.venv\Scripts\python.exe -m pytest tests/`
預期：全部 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_coarse_segmentation_corpus.py
git commit -m "test(pose): pin the coarse pass's cost against the real squat corpus"
```

---

## Task 6：兩趟抽取的組裝（`poseExtract.ts`）

**Spec:** §2.1、§2.1.1、§4.1

**Files:**
- Modify: `frontend/src/lib/poseExtract.ts`
- Test: `frontend/src/test/lib.poseExtract.test.ts`（既有，附加）

**Interfaces:**
- Consumes: Task 2/3/4 的全部匯出
- Produces:
  - `type RepsFallback = "no_reps_detected" | "only_partial_reps" | "segmentation_disabled" | null`
  - `interface RepSegment { index: number; start_frame: number; end_frame: number; partial: boolean; analyzed: boolean; refined: Refinement }`
  - `interface RepsPlan { max_reps: number; fallback: RepsFallback; segments: RepSegment[] }`
  - `planReps(coarseSignal: number[], maxReps: number, lastFrameIndex: number, movement: string): { plan: RepsPlan; spans: FrameSpan[] }`
  - `extractPoseWithReps(blob, tier, movement, maxReps, onProgress?) => Promise<{ pose: PoseJson; reps: RepsPlan }>`

**注意**：`planReps` 是**純函式**（測得到）；`extractPoseWithReps` 是 `<video>`/WASM glue，包在
既有的 `/* c8 ignore */` 區段裡，維持薄。所有決策都在 `planReps` 與 Task 4 的函式裡。

- [ ] **Step 1: 寫失敗的測試（附加到 `lib.poseExtract.test.ts`）**

```ts
import { planReps } from "../lib/poseExtract";
import { COARSE_STRIDE } from "../lib/repSpans";

function coarseRepSignal(count: number, period: number): number[] {
  return Array.from({ length: count * period }, (_, i) =>
    115 + 55 * Math.cos((2 * Math.PI * (i % period)) / period));
}

describe("planReps", () => {
  const LAST = 5 * 30 * COARSE_STRIDE - 1; // five 30-sample coarse reps on the canonical grid

  it("marks the first / middle / last of five reps as analyzed", () => {
    const { plan } = planReps(coarseRepSignal(5, 30), 3, LAST, "Squat");
    expect(plan.fallback).toBeNull();
    expect(plan.segments).toHaveLength(5);
    expect(plan.segments.filter((s) => s.analyzed).map((s) => s.index)).toEqual([1, 3, 5]);
  });

  it("returns spans only for the analyzed reps", () => {
    const { spans } = planReps(coarseRepSignal(5, 30), 3, LAST, "Squat");
    expect(spans.length).toBeGreaterThan(0);
    expect(spans.length).toBeLessThanOrEqual(3);
  });

  it("reports frame_index, not coarse positions", () => {
    const { plan } = planReps(coarseRepSignal(5, 30), 3, LAST, "Squat");
    // Rep 2 of 5 cannot start before frame 30 if each rep is 90 canonical frames long.
    expect(plan.segments[1].start_frame).toBeGreaterThanOrEqual(COARSE_STRIDE * 20);
  });

  it("falls back to the whole clip when nothing segments", () => {
    const { plan, spans } = planReps(new Array(150).fill(5), 3, LAST, "Squat");
    expect(plan.fallback).toBe("no_reps_detected");
    expect(plan.segments).toEqual([]);
    expect(spans).toEqual([{ start: 0, end: LAST }]);
  });

  it("falls back for a movement with no browser-side signal", () => {
    const { plan, spans } = planReps(coarseRepSignal(5, 30), 3, LAST, "Deadlift");
    expect(plan.fallback).toBe("segmentation_disabled");
    expect(spans).toEqual([{ start: 0, end: LAST }]);
  });

  it("falls back when every rep is partial", () => {
    // A clip that STARTS at the bottom and only rises: the single window has no crossing to climb
    // from on its left, so it is partial. Verified against Python — segment_reps returns exactly
    // one window with partial=True — so this asserts unconditionally.
    const rising = Array.from({ length: 30 }, (_, i) =>
      115 - 55 * Math.cos((2 * Math.PI * i) / 60));
    const { plan, spans } = planReps(rising, 3, 89, "Squat");
    expect(plan.fallback).toBe("only_partial_reps");
    expect(plan.segments).toEqual([]);
    expect(spans).toEqual([{ start: 0, end: 89 }]);
  });

  it("NEVER returns an empty span list — a fallback still extracts everything", () => {
    for (const signal of [new Array(150).fill(5), coarseRepSignal(5, 30)]) {
      expect(planReps(signal, 3, LAST, "Squat").spans.length).toBeGreaterThan(0);
    }
  });
});
```

- [ ] **Step 2: 跑測試確認失敗**

cwd = `frontend/`，執行：`yarn test src/test/lib.poseExtract.test.ts`
預期：FAIL，`planReps is not a function`

- [ ] **Step 3: 寫 `planReps`（純函式，放在 `poseExtract.ts` 的 c8-ignore 區段之前）**

```ts
import { TS_REP_SIGNALS, centeredMedian, type SignalLandmark } from "./repSignal";
import { segmentReps, selectReps } from "./repSegmentation";
import {
  COARSE_SMOOTH_WINDOW, COARSE_STRIDE, CANONICAL_FPS, frameIndexAt, mergeSpans, refineWindow,
  spanForRep, spanFrameIndices, valleyPosition, type FrameSpan, type Refinement,
} from "./repSpans";

export type RepsFallback =
  | "no_reps_detected" | "only_partial_reps" | "segmentation_disabled" | null;

export interface RepSegment {
  index: number;
  start_frame: number;
  end_frame: number;
  partial: boolean;
  analyzed: boolean;
  refined: Refinement;
}

export interface RepsPlan {
  max_reps: number;
  fallback: RepsFallback;
  segments: RepSegment[];
}

/**
 * Decide which frames to extract densely (spec §2.1, §4.1).
 *
 * EVERY fallback returns the WHOLE clip as one span. Sending a sparse frame list with no windows
 * would leave the backend to segment data that does not exist, and reporting a segmentation
 * failure as "no faults found" is the failure mode this codebase refuses (see resolveDuration's
 * comment below for the same rule applied to decoding). Not saving time is the correct trade.
 */
export function planReps(
  coarseSignal: number[], maxReps: number, lastFrameIndex: number, movement: string
): { plan: RepsPlan; spans: FrameSpan[] } {
  const wholeClip: FrameSpan[] = [{ start: 0, end: lastFrameIndex }];
  const fallbackPlan = (fallback: RepsFallback, segments: RepSegment[] = []) =>
    ({ plan: { max_reps: maxReps, fallback, segments }, spans: wholeClip });

  if (!(movement in TS_REP_SIGNALS)) return fallbackPlan("segmentation_disabled");

  const smoothed = centeredMedian(coarseSignal, COARSE_SMOOTH_WINDOW);
  const reps = segmentReps(smoothed, { fps: CANONICAL_FPS / COARSE_STRIDE });
  if (reps.length === 0) return fallbackPlan("no_reps_detected");
  if (reps.every((rep) => rep.partial)) {
    // A tightly-trimmed single-rep clip looks like this; analysing it whole is correct for it.
    return fallbackPlan("only_partial_reps");
  }

  const analyzed = new Set(selectReps(reps, maxReps).map((rep) => rep.index));
  const segments: RepSegment[] = reps.map((rep) => ({
    index: rep.index,
    start_frame: rep.start * COARSE_STRIDE,
    end_frame: Math.min(lastFrameIndex, rep.end * COARSE_STRIDE),
    partial: rep.partial,
    analyzed: analyzed.has(rep.index),
    refined: false, // upgraded by refineSegments once the dense signal exists
  }));

  const spans = mergeSpans(
    reps.filter((rep) => analyzed.has(rep.index))
        .map((rep) => spanForRep(smoothed, rep, lastFrameIndex))
  );
  return { plan: { max_reps: maxReps, fallback: null, segments }, spans };
}

/** Replace each analyzed segment's coarse boundary with the one the dense signal gives (§2.1.1). */
export function refineSegments(
  plan: RepsPlan, spans: FrameSpan[], denseSignal: (number | undefined)[], lastFrameIndex: number
): RepsPlan {
  if (plan.fallback !== null) return plan;
  const segments = plan.segments.map((segment) => {
    if (!segment.analyzed) return segment;
    const coarse = { start: segment.start_frame, end: segment.end_frame };
    const span = spans.find((s) => s.start <= coarse.start && coarse.end <= s.end)
      ?? spans.find((s) => s.start <= coarse.end && coarse.start <= s.end);
    if (!span) return segment;
    const { start, end, refined } =
      refineWindow(denseSignal, span, coarse, CANONICAL_FPS, lastFrameIndex);
    return { ...segment, start_frame: start, end_frame: end, refined };
  });
  return { ...plan, segments };
}
```

- [ ] **Step 4: 跑測試確認通過**

cwd = `frontend/`，執行：`yarn test src/test/lib.poseExtract.test.ts`
預期：PASS

- [ ] **Step 5: 寫 `extractPoseWithReps`（glue，放進 c8-ignore 區段）**

在 `poseExtract.ts` 的 `/* c8 ignore start */` 之後、`extractPoseFromBlob` 之前，先把逐幀取樣抽成
共用函式，再加兩趟的組裝：

```ts
/** Seek to each frame_index in turn and run the landmarker. Shared by both passes. */
async function sampleFrames(
  video: HTMLVideoElement,
  landmarker: { detectForVideo(v: HTMLVideoElement, t: number): {
    landmarks?: MpLandmark[][]; worldLandmarks?: MpLandmark[][] } },
  frameIndices: number[],
  onProgress?: (p: number) => void
): Promise<PoseJsonFrame[]> {
  const out: PoseJsonFrame[] = [];
  for (let n = 0; n < frameIndices.length; n += 1) {
    const index = frameIndices[n];
    const t = index / CANONICAL_FPS;
    video.currentTime = t;
    await new Promise<void>((r) => { video.onseeked = () => r(); });
    const result = landmarker.detectForVideo(video, Math.round(t * 1000));
    out.push(landmarksToFrame(index, result.landmarks?.[0], result.worldLandmarks?.[0]));
    onProgress?.((n + 1) / frameIndices.length);
  }
  return out;
}

/**
 * Two-pass extraction: find the reps cheaply, then measure only the selected ones (spec §2.1).
 *
 * Two passes are FORCED, not chosen: selectReps takes first/middle/last, so the total rep count
 * must be known before selecting, and no single streaming pass can know it. "Just take the first
 * three" is the failure SP1 rejected by name — fatigue breakdown shows up in the LAST rep.
 *
 * The returned pose JSON is FULL LENGTH with `landmarks: null` outside the extracted spans, which
 * keeps RepWindow positions equal to frame_index and frame_metrics one row per frame (spec §2.2).
 */
export async function extractPoseWithReps(
  blob: Blob,
  tier: PoseTier,
  movement: string,
  maxReps: number,
  onProgress?: (p: number) => void
): Promise<{ pose: PoseJson; reps: RepsPlan }> {
  const url = URL.createObjectURL(blob);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  const metadataReady = new Promise<void>((res, rej) => {
    video.onloadedmetadata = () => res();
    video.onerror = () => rej(new Error("Could not decode the video."));
  });
  metadataReady.catch(() => undefined);
  video.src = url;

  const coarseLandmarker = await createPoseLandmarker(LIVE_OVERLAY_TIER);
  try {
    await metadataReady;
    const duration = await resolveDuration(video);
    const lastFrameIndex = Math.max(0, frameIndexAt(duration) - 1);

    // Pass 1 — coarse. Lite, every COARSE_STRIDE-th frame, only to locate repetitions.
    const coarseIndices: number[] = [];
    for (let i = 0; i <= lastFrameIndex; i += COARSE_STRIDE) coarseIndices.push(i);
    const coarseFrames = await sampleFrames(video, coarseLandmarker, coarseIndices,
      (p) => onProgress?.(p * 0.3));
    const signal = TS_REP_SIGNALS[movement];
    const coarseSignal = coarseFrames.map((f) =>
      signal ? signal(f.landmarks as SignalLandmark[] | null) : NaN);
    const { plan, spans } = planReps(coarseSignal, maxReps, lastFrameIndex, movement);

    // Pass 2 — dense, at the user's tier, over the padded spans only.
    const denseIndices = spanFrameIndices(spans);
    const denseLandmarker = tier === LIVE_OVERLAY_TIER
      ? coarseLandmarker
      : await createPoseLandmarker(tier);
    let denseFrames: PoseJsonFrame[];
    try {
      denseFrames = await sampleFrames(video, denseLandmarker, denseIndices,
        (p) => onProgress?.(0.3 + p * 0.7));
    } finally {
      if (denseLandmarker !== coarseLandmarker) denseLandmarker.close();
    }

    // Full-length frame list: extracted frames in place, `null` landmarks everywhere else.
    const byIndex = new Map(denseFrames.map((f) => [f.frame_index, f]));
    const frames: PoseJsonFrame[] = [];
    for (let i = 0; i <= lastFrameIndex; i += 1) {
      frames.push(byIndex.get(i) ?? landmarksToFrame(i, undefined, undefined));
    }

    const denseSignal: (number | undefined)[] = new Array(lastFrameIndex + 1).fill(undefined);
    if (signal) {
      for (const frame of denseFrames) {
        denseSignal[frame.frame_index] = signal(frame.landmarks as SignalLandmark[] | null);
      }
    }
    onProgress?.(1);
    return {
      pose: {
        metadata: {
          fps: CANONICAL_FPS, width: video.videoWidth, height: video.videoHeight,
          total_frames: frames.length,
        },
        frames,
      },
      reps: refineSegments(plan, spans, denseSignal, lastFrameIndex),
    };
  } finally {
    coarseLandmarker.close();
    URL.revokeObjectURL(url);
  }
}
```

同時在檔案頂端補上 `import { LIVE_OVERLAY_TIER } from "./poseTier";`。

- [ ] **Step 6: 跑全部前端測試 + build**

cwd = `frontend/`，執行：`yarn test` 然後 `yarn build`
預期：測試全 PASS；build 成功（`yarn build` 會做 TypeScript 檢查，抓出型別錯誤）

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/poseExtract.ts frontend/src/test/lib.poseExtract.test.ts
git commit -m "feat(pose): extract only the repetitions that will be scored"
```

---

## Task 7：後端接受外部給的 rep 區間

**Spec:** §2.3、§4.3

**Files:**
- Modify: `src/pose/movements/base.py:114-220`
- Test: `tests/test_movement_registry.py`（既有，附加）

**Interfaces:**
- Consumes: 無（Python 側自成一格）
- Produces:
  - `@dataclass(frozen=True) class RepPlan: reps: tuple[RepWindow, ...]; analyzed: tuple[RepWindow, ...]; fallback: str | None`
  - `run_detector(..., rep_plan: RepPlan | None = None)`

- [ ] **Step 1: 寫失敗的測試**

附加到 `tests/test_movement_registry.py`：

```python
class RunDetectorWithRepPlanTest(unittest.TestCase):
    """RS-SP2: the browser extracts only the selected reps, so it -- not run_detector -- owns the
    rep boundaries. Frames outside those windows do not exist here, which is why segment_reps
    cannot simply be re-run (spec §2.3)."""

    def _frames(self, count: int) -> list[dict]:
        from tests.test_pose_rule_detector import frame  # the existing 33-landmark builder
        return [frame(i) for i in range(count)]

    def test_supplied_windows_replace_segmentation(self) -> None:
        from src.pose.movements import registry
        from src.pose.movements.base import RepPlan, run_detector
        from src.pose.rep_segmentation import RepWindow

        detector = registry.get_detector("Squat")
        frames = self._frames(90)
        window = RepWindow(index=1, start=10, end=49, partial=False)
        plan = RepPlan(reps=(window,), analyzed=(window,), fallback=None)
        run = run_detector(detector, frames, 30.0, "side", 0.9, rep_plan=plan)

        self.assertEqual([r.index for r in run.reps], [1])
        self.assertEqual([r.index for r in run.analyzed], [1])
        self.assertIsNone(run.fallback)
        # Frames outside the supplied window belong to no rep, so they are never scored.
        self.assertTrue(all(c.phase == "rest" for c in run.core[:10]))
        self.assertTrue(all(c.phase == "rest" for c in run.core[50:]))
        self.assertFalse(all(c.phase == "rest" for c in run.core[10:50]))

    def test_supplied_fallback_analyses_the_whole_clip(self) -> None:
        from src.pose.movements import registry
        from src.pose.movements.base import RepPlan, run_detector

        detector = registry.get_detector("Squat")
        run = run_detector(
            detector, self._frames(90), 30.0, "side", 0.9,
            rep_plan=RepPlan(reps=(), analyzed=(), fallback="no_reps_detected"),
        )
        self.assertEqual(run.fallback, "no_reps_detected")
        self.assertEqual(run.analyzed, [])
        self.assertTrue(all(c.phase != "rest" for c in run.core))

    def test_rep_plan_wins_over_max_reps(self) -> None:
        """The client already applied its cap; re-selecting here could only analyse FEWER reps
        than were actually extracted (spec §4.3)."""
        from src.pose.movements import registry
        from src.pose.movements.base import RepPlan, run_detector
        from src.pose.rep_segmentation import RepWindow

        detector = registry.get_detector("Squat")
        windows = tuple(
            RepWindow(index=i + 1, start=i * 30, end=i * 30 + 29, partial=False) for i in range(3)
        )
        run = run_detector(
            detector, self._frames(90), 30.0, "side", 0.9,
            max_reps=1, rep_plan=RepPlan(reps=windows, analyzed=windows, fallback=None),
        )
        self.assertEqual([r.index for r in run.analyzed], [1, 2, 3])

    def test_no_rep_plan_is_byte_for_byte_sp1(self) -> None:
        from src.pose.movements import registry
        from src.pose.movements.base import run_detector

        detector = registry.get_detector("Squat")
        frames = self._frames(90)
        a = run_detector(detector, frames, 30.0, "side", 0.9)
        b = run_detector(detector, frames, 30.0, "side", 0.9, rep_plan=None)
        self.assertEqual([d.fault_id for d in a.detections], [d.fault_id for d in b.detections])
        self.assertEqual([c.phase for c in a.core], [c.phase for c in b.core])
```

- [ ] **Step 2: 跑測試確認失敗**

`.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py::RunDetectorWithRepPlanTest -v`
預期：FAIL，`ImportError: cannot import name 'RepPlan'`

- [ ] **Step 3: 實作**

`src/pose/movements/base.py`，在 `RunResult` 之前加入：

```python
@dataclass(frozen=True)
class RepPlan:
    """Repetition boundaries decided elsewhere, for `run_detector` to use instead of its own.

    RS-SP2 extracts only the selected repetitions in the browser, so the frames between them do
    not exist by the time this module sees the clip. `segment_reps` cannot re-derive windows from
    data that was never captured -- it is underdetermined, not merely unreliable -- so the client
    supplies them and is validated at the API boundary instead (see the SP2 spec §2.3, §4.3).

    `fallback` carries the client's own reason string (the same three values this module produces)
    so a clip the browser could not segment is analysed whole here, exactly as SP1 would have.
    """

    reps: tuple[RepWindow, ...]
    analyzed: tuple[RepWindow, ...]
    fallback: str | None
```

然後把 `run_detector` 的簽名與切割段落改成：

```python
def run_detector(
    detector: MovementDetector,
    frames: Sequence[object],
    fps: float,
    view_type: str,
    view_confidence: float,
    *,
    max_reps: int | None = DEFAULT_MAX_REPS,
    rep_plan: RepPlan | None = None,
) -> RunResult:
    """Compute metrics over the whole clip, then phase and score one repetition at a time.

    Smoothing stays GLOBAL. When `rep_plan` is supplied the clip is SPARSE -- only the planned
    windows carry landmarks -- and `centered_median` skips the non-finite gaps, so a window padded
    by at least the smoothing radius still sees a full window at every frame inside it.

    `rep_plan` OVERRIDES `max_reps`: the client already applied its own cap when deciding what to
    extract, so re-selecting here could only score fewer reps than were actually measured.
    """
    raw = detector.compute_raw(frames, fps)
    smoothed = {
        key: centered_median([float(item.get(key, np.nan)) for item in raw], window=5)
        for key in detector.metric_keys
    }

    if rep_plan is not None:
        reps = list(rep_plan.reps)
        fallback = rep_plan.fallback
    elif detector.rep_signal is None:
        reps, fallback = [], "segmentation_disabled"
    else:
        reps = segment_reps(
            smoothed[detector.rep_signal],
            fps=fps,
            polarity=detector.rep_polarity,
            rectify=detector.rep_rectify,
            rep_start=detector.rep_start,
            min_rep_seconds=detector.min_rep_seconds,
        )
        fallback = None
        if not reps:
            fallback = "no_reps_detected"
        elif all(rep.partial for rep in reps):
            fallback = "only_partial_reps"
```

`segmented` 那行以下維持不變，只把 `analyzed` 的計算改成：

```python
    analyzed = list(rep_plan.analyzed) if rep_plan is not None else select_reps(segmented, max_reps)
```

（`segmented = reps if fallback is None else []` 這行不動——fallback 時 `analyzed` 也會是空的，
因為 client 在 fallback 時送的 `analyzed` 就是空的。）

- [ ] **Step 4: 跑測試確認通過**

`.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py -v`
預期：全部 PASS（含新的 4 個）

- [ ] **Step 5: 跑整套**

`.venv\Scripts\python.exe -m pytest tests/`
預期：全部 PASS——特別是 `test_squat_via_registry_matches_legacy` 必須仍然過。

- [ ] **Step 6: Commit**

```bash
git add src/pose/movements/base.py tests/test_movement_registry.py
git commit -m "feat(pose): let the caller supply rep windows the detector cannot re-derive"
```

---

## Task 8：`quality` 附加已抽取幀數

**Spec:** §4.3、§4.4

**為什麼需要**：SP2 之後 `valid_frame_ratio` 會合理地掉到 ~30%（分母仍是全片），而
`MetricsCards.tsx:57,96` 直接顯示它——使用者讀到的是「追蹤品質很差」，實際是設計如此。
`wasMeasured`（`quality.ts:29`）與 `chat.py:182` 都是類別式的 `> 0`，不受影響。

**Files:**
- Modify: `src/pose/pose_rule_detector.py:596-663`
- Test: `tests/test_pose_rule_detector.py`（既有，附加）

**Interfaces:**
- Consumes: Task 7 的 `RepPlan`
- Produces: `detect_pose_rules_from_payload(..., rep_plan=None)`；`quality.extracted_frames`、
  `quality.extracted_frame_ratio`

- [ ] **Step 1: 寫失敗的測試**

附加到 `tests/test_pose_rule_detector.py`：

```python
class SparsePayloadQualityTest(unittest.TestCase):
    """RS-SP2 sends a full-length frame list with null landmarks outside the extracted spans."""

    def _payload(self, extracted: range, total: int = 60) -> dict:
        frames = []
        for i in range(total):
            frames.append(frame(i) if i in extracted else
                          {"frame_index": i, "landmarks": None, "world_landmarks": None})
        return {"metadata": {"fps": 30, "width": 640, "height": 480, "total_frames": total},
                "frames": frames}

    def test_reports_how_many_frames_were_extracted(self) -> None:
        result = detect_pose_rules_from_payload(self._payload(range(20, 40)), movement="Squat")
        quality = result["quality"]
        self.assertEqual(quality["extracted_frames"], 20)
        self.assertAlmostEqual(quality["extracted_frame_ratio"], 20 / 60, places=4)

    def test_existing_denominators_stay_whole_clip(self) -> None:
        """SP1 §5: quality is a compatibility surface for analysis.py, the frontend and
        perception_to_graph.py. Adding fields is safe; changing these is not."""
        result = detect_pose_rules_from_payload(self._payload(range(20, 40)), movement="Squat")
        self.assertEqual(result["quality"]["total_frames"], 60)
        self.assertEqual(len(result["frame_metrics"]), 60)

    def test_a_dense_clip_reports_every_frame_extracted(self) -> None:
        result = detect_pose_rules_from_payload(self._payload(range(60)), movement="Squat")
        self.assertEqual(result["quality"]["extracted_frames"], 60)
        self.assertAlmostEqual(result["quality"]["extracted_frame_ratio"], 1.0, places=4)
```

- [ ] **Step 2: 跑測試確認失敗**

`.venv\Scripts\python.exe -m pytest tests/test_pose_rule_detector.py::SparsePayloadQualityTest -v`
預期：FAIL，`KeyError: 'extracted_frames'`

- [ ] **Step 3: 實作**

`src/pose/pose_rule_detector.py`：`detect_pose_rules_from_payload` 的簽名加上
`rep_plan: object | None = None`，往下傳給 `run_detector(..., rep_plan=rep_plan)`；
並在 `analyzed_frames` 那段之後加入：

```python
    # ADDITIVE (SP2 §4.4). Under RS-SP2 the frames outside the extracted spans carry no landmarks,
    # so valid_frame_ratio legitimately falls -- its denominator stays whole-clip by SP1's rule.
    # This is the denominator that answers "was tracking good WHERE WE LOOKED", which is what
    # MetricsCards must show instead, or a deliberate design reads to the user as bad tracking.
    extracted_frames = sum(
        1 for f in frames if isinstance(f, dict) and f.get("landmarks")
    )
```

`quality` 字典裡加兩行：

```python
            "extracted_frames": extracted_frames,
            "extracted_frame_ratio": round(extracted_frames / len(frames), 4) if frames else 0.0,
```

- [ ] **Step 4: 跑測試確認通過**

`.venv\Scripts\python.exe -m pytest tests/test_pose_rule_detector.py -v`
預期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/pose/pose_rule_detector.py tests/test_pose_rule_detector.py
git commit -m "feat(pose): report how many frames were actually extracted"
```

---

## Task 9：端點收 `reps` 並驗證

**Spec:** §4.2、§4.3

**驗證是這個 Task 的重點**，因為 §2.3 之後後端信任 client。**默默忽略非法的 `reps` 是最糟的選項**
——它會讓後端拿有洞的訊號重新切割，產出看起來正常但錯誤的區間。

**Files:**
- Modify: `backend/app/routers/analyze.py:164-224`
- Modify: `backend/app/services/analysis.py:155-191`
- Test: `tests/test_analyze_pose_endpoint.py`（既有；若不存在則建立同名檔）

**Interfaces:**
- Consumes: Task 7 的 `RepPlan`、Task 8 的 `detect_pose_rules_from_payload(rep_plan=...)`
- Produces: `_validate_reps(raw: str | None, frame_count: int, frames: list) -> RepPlan | None`

- [ ] **Step 1: 寫失敗的測試**

附加到 `tests/test_analyze_pose_endpoint.py`。**注意該檔案的既有慣例**：測試**直接呼叫**
`analyze_router.analyze_pose(...)`，不走 FastAPI/TestClient，所以每個 `Form` 參數都要明確傳值
（未解析的 `Form(...)` sentinel 會原封不動到達驗證器）。`setUp` 的 `analyze_pose_payload` 樁也要
補上新的 `rep_plan` 參數，否則呼叫會 `TypeError`。

```python
# 先把 setUp 裡的樁改成接受 rep_plan（新增的關鍵字參數）：
#     analysis_service.analyze_pose_payload = (
#         lambda payload, *, movement, video_id=None, max_reps=-1, rep_plan=None: {
#             "video_id": video_id, "source": "upload", "movement": movement,
#             "detections": [], "rep_plan": rep_plan,
#         }
#     )

_LANDMARKS = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]


def _pose(extracted: range, total: int = 60) -> str:
    """A full-length frame list with landmarks only inside `extracted` — the RS-SP2 shape."""
    frames = [
        {"frame_index": i,
         "landmarks": _LANDMARKS if i in extracted else None,
         "world_landmarks": _LANDMARKS if i in extracted else None}
        for i in range(total)
    ]
    return json.dumps({"metadata": {"fps": 30, "width": 1, "height": 1, "total_frames": total},
                       "frames": frames})


def _segment(index: int, start: int, end: int, *, analyzed: bool = True) -> dict:
    return {"index": index, "start_frame": start, "end_frame": end,
            "partial": False, "analyzed": analyzed, "refined": True}


def _reps(segments: list[dict], fallback: str | None = None) -> str:
    return json.dumps({"max_reps": 3, "fallback": fallback, "segments": segments})


class AnalyzePoseRepsValidationTests(unittest.TestCase):
    """`reps` is client-supplied and the backend now trusts it for rep boundaries, so every
    violation must be a 400 -- silently ignoring one would leave the backend re-segmenting a
    signal full of holes and emitting plausible-looking, wrong windows (spec §4.3)."""

    def setUp(self) -> None:
        self._orig_save = analysis_service.save_upload
        self._orig_analyze = analysis_service.analyze_pose_payload
        analysis_service.save_upload = lambda data, suffix=".mp4": ("upload_test", Path(f"upload_test{suffix}"))
        analysis_service.analyze_pose_payload = (
            lambda payload, *, movement, video_id=None, max_reps=-1, rep_plan=None: {
                "video_id": video_id, "source": "upload", "movement": movement,
                "detections": [], "rep_plan": rep_plan,
            }
        )

    def tearDown(self) -> None:
        analysis_service.save_upload = self._orig_save
        analysis_service.analyze_pose_payload = self._orig_analyze

    def _run(self, pose: str, reps: str | None):
        return asyncio.run(
            analyze_router.analyze_pose("Squat", pose, _upload(), max_reps=None, reps=reps, user=None)
        )

    def _assert_400(self, pose: str, reps: str | None) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self._run(pose, reps)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_accepts_a_well_formed_plan_and_forwards_it(self) -> None:
        result = self._run(_pose(range(0, 30)), _reps([_segment(1, 0, 29)]))
        plan = result["rep_plan"]
        self.assertEqual([w.index for w in plan.reps], [1])
        self.assertEqual([w.index for w in plan.analyzed], [1])
        self.assertIsNone(plan.fallback)

    def test_rejects_a_window_past_the_end_of_the_clip(self) -> None:
        self._assert_400(_pose(range(0, 30)), _reps([_segment(1, 0, 9999)]))

    def test_rejects_start_after_end(self) -> None:
        self._assert_400(_pose(range(0, 30)), _reps([_segment(1, 20, 5)]))

    def test_rejects_overlapping_windows(self) -> None:
        self._assert_400(
            _pose(range(0, 60)), _reps([_segment(1, 0, 29), _segment(2, 20, 49)])
        )

    def test_rejects_indices_that_do_not_run_1_to_n(self) -> None:
        self._assert_400(_pose(range(0, 30)), _reps([_segment(2, 0, 29)]))

    def test_rejects_an_unknown_fallback_value(self) -> None:
        self._assert_400(_pose(range(0, 30)), _reps([], fallback="because_i_said_so"))

    def test_rejects_more_segments_than_the_cap(self) -> None:
        segments = [_segment(i + 1, i, i) for i in range(analyze_router.MAX_REP_SEGMENTS + 1)]
        self._assert_400(_pose(range(0, 300), total=300), _reps(segments))

    def test_rejects_an_analyzed_window_over_unextracted_frames(self) -> None:
        """The violation every ordering/range/overlap check passes. Scoring all-invalid frames
        produces an EMPTY detection list -- a clean verdict from data nothing measured, which is
        the exact failure frontend/src/lib/quality.ts exists to prevent."""
        self._assert_400(_pose(range(40, 60)), _reps([_segment(1, 0, 29)]))

    def test_allows_an_UNanalyzed_window_over_unextracted_frames(self) -> None:
        """Reps that were found but not scored legitimately have no landmarks — that is the whole
        point of SP2, and `segments[].analyzed=False` is how the payload says so."""
        result = self._run(
            _pose(range(0, 30), total=60),
            _reps([_segment(1, 0, 29), _segment(2, 30, 59, analyzed=False)]),
        )
        self.assertEqual([w.index for w in result["rep_plan"].analyzed], [1])

    def test_rejects_malformed_reps_json(self) -> None:
        self._assert_400(_pose(range(0, 30)), "{not json")

    def test_omitting_reps_keeps_todays_behaviour(self) -> None:
        """The CLI, the research datasets and old clients send no `reps` and must be unaffected."""
        result = self._run(_pose(range(0, 30)), None)
        self.assertIsNone(result["rep_plan"])
```

- [ ] **Step 2: 跑測試確認失敗**

`.venv\Scripts\python.exe -m pytest tests/test_analyze_pose_endpoint.py -v`
預期：FAIL

- [ ] **Step 3: 實作驗證器**

`backend/app/routers/analyze.py`，在 `_validate_pose_landmarks` 之後：

```python
MAX_REP_SEGMENTS = 200
_FALLBACKS = {None, "no_reps_detected", "only_partial_reps", "segmentation_disabled"}


def _validate_reps(raw: str | None, frames: list) -> "RepPlan | None":
    """Turn a client-supplied rep plan into a RepPlan, rejecting anything malformed.

    EVERY violation is a 400 rather than a silent fall-through to the backend's own segmentation.
    Under RS-SP2 the frames outside the extracted spans carry no landmarks, so re-segmenting that
    signal does not fail loudly -- it produces plausible windows over data that was never
    measured, and the result reads like a normal analysis (spec §4.3).
    """
    from src.pose.movements.base import RepPlan
    from src.pose.rep_segmentation import RepWindow

    if raw is None:
        return None
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Malformed reps JSON.") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("segments"), list):
        raise HTTPException(status_code=400, detail="reps must have a 'segments' list.")
    if plan.get("fallback") not in _FALLBACKS:
        raise HTTPException(status_code=400, detail="Unknown reps.fallback value.")

    segments = plan["segments"]
    if len(segments) > MAX_REP_SEGMENTS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_REP_SEGMENTS} rep segments.")

    windows: list[RepWindow] = []
    analyzed: list[RepWindow] = []
    previous_end = -1
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise HTTPException(status_code=400, detail="Malformed rep segment.")
        try:
            index = int(segment["index"])
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Malformed rep segment.") from exc
        if index != position + 1:
            raise HTTPException(status_code=400, detail="rep indices must run 1..N in order.")
        if start < 0 or end >= len(frames) or start > end:
            raise HTTPException(status_code=400, detail="rep window out of range.")
        if start <= previous_end:
            raise HTTPException(status_code=400, detail="rep windows must not overlap.")
        previous_end = end

        window = RepWindow(index=index, start=start, end=end, partial=bool(segment.get("partial")))
        windows.append(window)
        if segment.get("analyzed"):
            # The check the others miss: a window over frames that were never extracted scores
            # all-invalid data and yields an empty detection list, i.e. a clean verdict from
            # nothing. See the test that pins this.
            if not any(
                isinstance(frames[i], dict) and frames[i].get("landmarks")
                for i in range(start, end + 1)
            ):
                raise HTTPException(
                    status_code=400, detail="An analyzed rep window contains no extracted frames."
                )
            analyzed.append(window)

    return RepPlan(reps=tuple(windows), analyzed=tuple(analyzed), fallback=plan.get("fallback"))
```

- [ ] **Step 4: 接進端點**

`analyze_pose` 的簽名加 `reps: str | None = Form(None)`，在 `_validate_pose_landmarks(payload)`
之後加 `rep_plan = _validate_reps(reps, payload["frames"])`，並把 `analysis.analyze_pose_payload`
的呼叫加上 `rep_plan=rep_plan`。

`backend/app/services/analysis.py` 的 `analyze_pose_payload` 加 `rep_plan: object | None = None`
參數，往下傳給 `_run_detector(..., rep_plan=rep_plan)`，`_run_detector` 再傳給
`detect_pose_rules_from_payload(..., rep_plan=rep_plan)`。

- [ ] **Step 5: 跑測試 + 覆蓋率**

```
.venv\Scripts\python.exe -m pytest tests/
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```
預期：全部 PASS，覆蓋率 ≥ 95%

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/analyze.py backend/app/services/analysis.py tests/test_analyze_pose_endpoint.py
git commit -m "feat(api): accept and validate client-supplied rep windows"
```

---

## Task 10：前端把 `reps` 送出去

**Spec:** §4.2

**Files:**
- Modify: `frontend/src/api.ts:573-589`（`analyzePose`）與 `Analysis` 型別
- Modify: `frontend/src/App.tsx:116-141`
- Test: `frontend/src/test/api.pose.test.ts`（既有，附加）

**Interfaces:**
- Consumes: Task 6 的 `RepsPlan`、`extractPoseWithReps`
- Produces: `analyzePose(movement, pose, video, reps?)`；`Analysis["reps"]`

- [ ] **Step 1: 寫失敗的測試**

附加到 `frontend/src/test/api.pose.test.ts`：

```ts
it("posts the rep plan alongside the pose JSON", async () => {
  const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
  vi.stubGlobal("fetch", fetchMock);
  const reps = { max_reps: 3, fallback: null, segments: [
    { index: 1, start_frame: 0, end_frame: 29, partial: false, analyzed: true, refined: true as const },
  ] };
  await api.analyzePose("Squat", { metadata: { fps: 30, width: 1, height: 1, total_frames: 1 }, frames: [] },
    new Blob([], { type: "video/webm" }), reps);
  const form = fetchMock.mock.calls[0][1].body as FormData;
  expect(JSON.parse(form.get("reps") as string)).toEqual(reps);
});

it("omits reps entirely when there is no plan, so old behaviour is unchanged", async () => {
  const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
  vi.stubGlobal("fetch", fetchMock);
  await api.analyzePose("Squat", { metadata: { fps: 30, width: 1, height: 1, total_frames: 1 }, frames: [] },
    new Blob([], { type: "video/webm" }));
  expect((fetchMock.mock.calls[0][1].body as FormData).get("reps")).toBeNull();
});
```

- [ ] **Step 2: 跑測試確認失敗**

cwd = `frontend/`，執行：`yarn test src/test/api.pose.test.ts`
預期：FAIL（`reps` 是 `null`）

- [ ] **Step 3: 實作**

`frontend/src/api.ts`：`Analysis` 型別加上（放在 `quality` 附近）

```ts
  reps?: {
    detected: number;
    analyzed: number[];
    max_reps: number | null;
    fallback: string | null;
    segments: {
      index: number; start_frame: number; end_frame: number;
      start_time: number; end_time: number; analyzed: boolean; partial: boolean;
    }[];
  };
```

`analyzePose` 改成：

```ts
  async analyzePose(movement: string, pose: PoseJson, video: Blob, reps?: RepsPlan): Promise<Analysis> {
    const form = new FormData();
    form.append("movement", movement);
    form.append("pose", JSON.stringify(pose));
    // Only sent when the browser actually planned the extraction. Omitting it keeps the endpoint
    // on its pre-SP2 path, which is what the CLI and any old client rely on.
    if (reps) form.append("reps", JSON.stringify(reps));
    const ext = video.type.includes("mp4") ? "mp4" : "webm";
    form.append("file", video, `capture.${ext}`);
    // ... unchanged
```

`frontend/src/App.tsx` 的 `runPoseAnalysis`：

```ts
      const { pose, reps } = await extractPoseWithReps(blob, tier, canonicalMovement, DEFAULT_MAX_REPS);
      const data = await api.analyzePose(canonicalMovement, pose, blob, reps);
```

並在 `frontend/src/lib/poseTier.ts` 旁新增（或放在 `repSpans.ts`）`export const DEFAULT_MAX_REPS = 3;`
——與 `backend/app/config.py` 的 `DEFAULT_MAX_REPS` 同值，註解交叉引用。

- [ ] **Step 4: 跑測試 + build**

cwd = `frontend/`，執行：`yarn test` 然後 `yarn build`
預期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.ts frontend/src/App.tsx frontend/src/lib/repSpans.ts frontend/src/test/api.pose.test.ts
git commit -m "feat(pose): send the browser's rep plan with the pose payload"
```

---

## Task 11：未分析區段的 UI

**Spec:** §5

**這一節不是裝飾。** 這是第一次會有大段影片真的沒有姿態資料；骨架在片中消失、timeline 空白，
在使用者眼裡就是 bug 或「這幾下沒問題」。這個 codebase 明文拒絕讓管線的限制被呈現成教練判定
（見 `poseExtract.ts:59-66`、`quality.ts:1-27`）。

**Files:**
- Modify: `frontend/src/components/Timeline.tsx`
- Modify: `frontend/src/components/MetricsCards.tsx:93-97`
- Modify: `frontend/src/lib/i18n.tsx`
- Test: `frontend/src/test/components.Timeline.test.tsx`、`components.MetricsCards.test.tsx`（既有）

**Interfaces:**
- Consumes: Task 10 的 `Analysis["reps"]`、Task 8 的 `quality.extracted_frames`
- Produces: 無

- [ ] **Step 1: 寫失敗的測試**

附加到 `components.Timeline.test.tsx`：

```tsx
it("marks spans that were never analyzed", () => {
  const analysis = { ...baseAnalysis, reps: {
    detected: 3, analyzed: [1, 3], max_reps: 3, fallback: null,
    segments: [
      { index: 1, start_frame: 0, end_frame: 29, start_time: 0, end_time: 1, analyzed: true, partial: false },
      { index: 2, start_frame: 30, end_frame: 59, start_time: 1, end_time: 2, analyzed: false, partial: false },
      { index: 3, start_frame: 60, end_frame: 89, start_time: 2, end_time: 3, analyzed: true, partial: false },
    ],
  }};
  render(<Timeline analysis={analysis} duration={3} currentTime={0} onSeek={() => {}} />);
  expect(screen.getAllByTestId("unanalyzed-span")).toHaveLength(1);
});

it("says how many reps were found and how many were scored", () => {
  // ... same analysis
  expect(screen.getByText(/3/)).toBeInTheDocument();
});

it("shows nothing extra when the whole clip was analyzed as one unit", () => {
  const analysis = { ...baseAnalysis, reps: {
    detected: 0, analyzed: [], max_reps: 3, fallback: "no_reps_detected", segments: [],
  }};
  render(<Timeline analysis={analysis} duration={3} currentTime={0} onSeek={() => {}} />);
  expect(screen.queryAllByTestId("unanalyzed-span")).toHaveLength(0);
});
```

附加到 `components.MetricsCards.test.tsx`：

```tsx
it("counts valid frames against the frames that were EXTRACTED, not the whole clip", () => {
  // Under RS-SP2 only the scored reps carry landmarks, so a whole-clip denominator would show
  // "30%" for a deliberately partial extraction and read as bad tracking.
  const analysis = { ...baseAnalysis, quality: {
    ...baseAnalysis.quality, total_frames: 900, valid_frames: 260,
    extracted_frames: 270, extracted_frame_ratio: 0.3, valid_frame_ratio: 0.289,
  }};
  render(<MetricsCards analysis={analysis} />);
  expect(screen.getByText("96%")).toBeInTheDocument();
});

it("falls back to the whole-clip denominator for analyses with no extracted_frames", () => {
  render(<MetricsCards analysis={baseAnalysis} />);
  // baseAnalysis has no extracted_frames — old stored analyses and CLI output look like this.
  expect(screen.getByText(`${Math.round((baseAnalysis.quality.valid_frame_ratio ?? 0) * 100)}%`))
    .toBeInTheDocument();
});
```

- [ ] **Step 2: 跑測試確認失敗**

cwd = `frontend/`，執行：`yarn test src/test/components.Timeline.test.tsx src/test/components.MetricsCards.test.tsx`
預期：FAIL

- [ ] **Step 3: i18n key**

`frontend/src/lib/i18n.tsx` 的 zh-TW 與 en 兩份都加：

```
"timeline.unanalyzed": "未分析" / "Not analyzed"
"timeline.repsSummary": "共 {{detected}} 下，分析了第 {{list}} 下" /
                        "{{detected}} reps found, analyzed #{{list}}"
"timeline.wholeClip": "整段分析" / "Whole clip analyzed"
"metric.framesRatioExtracted": "{{valid}} / {{extracted}} 已抽取" /
                               "{{valid}} / {{extracted}} extracted"
```

- [ ] **Step 4: Timeline 實作**

在 fault segments 那個 `map` **之前**（讓錯誤標記畫在上層）插入：

```tsx
        {/* Spans that carry no pose data because RS-SP2 never extracted them. NEUTRAL, never a
            warning colour: they are not a problem, they are unexamined — and an empty timeline
            must not read as "these reps were fine". */}
        {(analysis.reps?.segments ?? [])
          .filter((s) => !s.analyzed)
          .map((s) => (
            <div
              key={`un-${s.index}`}
              data-testid="unanalyzed-span"
              title={t("timeline.unanalyzed")}
              className="absolute h-2 rounded-full bg-track opacity-70
                         [background-image:repeating-linear-gradient(45deg,transparent,transparent_3px,rgba(255,255,255,0.12)_3px,rgba(255,255,255,0.12)_6px)]"
              style={{
                left: pct(s.start_time),
                width: `${Math.max(1.5, ((s.end_time - s.start_time) / dur) * 100)}%`,
              }}
            />
          ))}
```

並在底部的 legend 那一行後面加上摘要：

```tsx
        {analysis.reps && (
          <span className="text-muted">
            {analysis.reps.fallback
              ? t("timeline.wholeClip")
              : t("timeline.repsSummary", {
                  detected: analysis.reps.detected,
                  list: analysis.reps.analyzed.join("、"),
                })}
          </span>
        )}
```

- [ ] **Step 5: MetricsCards 實作**

把 `valid frames` 那張卡改成：

```tsx
      {/* Denominator: the frames that were EXTRACTED, when the payload says. Under RS-SP2 only the
          scored reps carry landmarks, so the whole-clip ratio legitimately falls to ~30% and would
          read as bad tracking — a pipeline decision presented as a measurement problem. Analyses
          predating SP2 (and CLI output) have no extracted_frames and keep the old denominator. */}
      <Stat
        label={t("metric.validFrames")}
        value={`${(extracted > 0 ? (q.valid_frames ?? 0) / extracted * 100 : validRatio).toFixed(0)}%`}
        sub={extracted > 0
          ? t("metric.framesRatioExtracted", { valid: q.valid_frames ?? 0, extracted })
          : t("metric.framesRatio", { valid: q.valid_frames ?? 0, total: q.total_frames ?? 0 })}
      />
```

並在 `const validRatio = ...` 旁邊加 `const extracted = q.extracted_frames ?? 0;`。

- [ ] **Step 6: 跑全部前端測試 + build**

cwd = `frontend/`，執行：`yarn test` 然後 `yarn build`
預期：全部 PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/Timeline.tsx frontend/src/components/MetricsCards.tsx frontend/src/lib/i18n.tsx frontend/src/test/
git commit -m "feat(ui): show which spans of a clip were never examined"
```

---

## Task 12：T1 訊號 parity 與 T3 效能實測

**Spec:** §3

**T2 已在寫 spec 時完成**（§2.8，46 clips / 70 reps，`REP_PADDING_FRAMES = 24`）。這個 Task 收尾
剩下的兩項，並把結果寫進 `notes/`。

**T1 用 golden file 而不是跨語言執行**：Python 產生「真實 landmarks → 期望角度」的檔案，vitest
讀它比對。這樣不需要 node 跑 TS，而且比對變成**常設測試**，不是一次性腳本——`repSignal.ts` 之後
任何改動都會被擋下。

**Files:**
- Create: `scripts/pose/capture_rep_signal_golden.py`
- Create: `tests/fixtures/rep_signal_golden.json`（由上面的腳本產生，**要進版控**：只有幾十幀，
  且沒有它 vitest 那側就沒有 ground truth）
- Create: `frontend/src/test/lib.repSignal.golden.test.ts`
- Create: `notes/rep_segmentation_sp2_measurements.md`

**Interfaces:**
- Consumes: Task 2 的 `avgKneeAngle`；Python 的 `raw_frame_metrics`
- Produces: `tests/fixtures/rep_signal_golden.json`

- [ ] **Step 1: 寫 golden file 產生腳本**

`scripts/pose/capture_rep_signal_golden.py`：

```python
"""Freeze real landmarks and the angle Python computes from them, for the TypeScript port to match.

RS-SP2 recomputes the squat rep signal in the browser, and the backend then TRUSTS the rep windows
that signal produces -- so a divergence between the two implementations would never surface on its
own (spec §2.3, §2.7). tests/fixtures/rep_segmentation_cases.json pins signal->windows; this file
pins landmarks->signal, which is the other half and the one nothing else covers.

Regenerate only when the Python formula deliberately changes:
    .venv\\Scripts\\python.exe scripts/pose/capture_rep_signal_golden.py <pose.json>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pose.pose_rule_detector import raw_frame_metrics  # noqa: E402

OUTPUT = REPO_ROOT / "tests" / "fixtures" / "rep_signal_golden.json"
# Enough frames to cover a full rep's range of angles without bloating the repo.
STRIDE = 3
MAX_FRAMES = 60


def main(source: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    fps = float((payload.get("metadata") or {}).get("fps", 30.0) or 30.0)
    cases = []
    for frame in payload.get("frames", [])[::STRIDE][:MAX_FRAMES]:
        metrics = raw_frame_metrics(frame, fps)
        angle = metrics.get("avg_knee_angle")
        cases.append({
            "landmarks": frame.get("landmarks"),
            # null encodes "no measurable angle" -- JSON has no NaN, and the TS side asserts
            # Number.isNaN for these rather than an equality that would silently pass on undefined.
            "avg_knee_angle": None if angle is None or not math.isfinite(angle) else float(angle),
        })
    OUTPUT.write_text(json.dumps({"source": source.name, "cases": cases}), encoding="utf-8")
    print(f"wrote {len(cases)} cases from {source.name} to {OUTPUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
```

- [ ] **Step 2: 產生 golden file**

`.venv\Scripts\python.exe scripts/pose/capture_rep_signal_golden.py data/runtime/pose_json/<某支>.json`

挑一支**有完整深蹲**的（角度範圍夠寬）。確認輸出的 `avg_knee_angle` 不是清一色 `null`。

- [ ] **Step 3: 寫 vitest 比對（T1 本體）**

`frontend/src/test/lib.repSignal.golden.test.ts`：

```ts
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import { avgKneeAngle, type SignalLandmark } from "../lib/repSignal";

// Ground truth generated by scripts/pose/capture_rep_signal_golden.py from a REAL clip. The shared
// segmentation fixture pins signal->windows; this pins landmarks->signal, and nothing else does.
//
// NOT `new URL("../../../tests/...", import.meta.url)` — Vite 6's asset plugin statically rewrites
// that form into an http://.../@fs/... URL, which readFileSync rejects. Resolve from the test
// file's own directory instead, so the path stays independent of where vitest was launched from.
// (Same reason and same shape as lib.repSegmentation.test.ts.)
const FIXTURE = resolve(
  dirname(fileURLToPath(import.meta.url)), "../../../tests/fixtures/rep_signal_golden.json"
);
const golden = JSON.parse(readFileSync(FIXTURE, "utf-8")) as {
  source: string;
  cases: { landmarks: SignalLandmark[] | null; avg_knee_angle: number | null }[];
};

describe("avgKneeAngle matches Python on real landmarks", () => {
  it("has a golden file with measurable frames in it", () => {
    expect(golden.cases.length).toBeGreaterThan(0);
    expect(golden.cases.some((c) => c.avg_knee_angle !== null)).toBe(true);
  });

  it("agrees to floating-point precision on every frame", () => {
    for (const [i, testCase] of golden.cases.entries()) {
      const got = avgKneeAngle(testCase.landmarks);
      if (testCase.avg_knee_angle === null) {
        expect(Number.isNaN(got), `frame ${i} should be unmeasurable`).toBe(true);
      } else {
        // A real port difference (2-D instead of 3-D, a different visibility gate) shows up far
        // above this; anything at 1e-6 is float32-vs-float64 in the Python side's arithmetic.
        expect(Math.abs(got - testCase.avg_knee_angle), `frame ${i}`).toBeLessThan(1e-6);
      }
    }
  });
});
```

- [ ] **Step 4: 跑 T1**

cwd = `frontend/`，執行：`yarn test src/test/lib.repSignal.golden.test.ts`
預期：PASS。**若不過，是 Task 2 的移植有 bug，回去修，不要放寬容差。**

- [ ] **Step 5: 跑 T3（真機或桌機瀏覽器）**

用 `yarn dev` 開起來，錄一段 ~30 秒 5 下的深蹲，在 devtools console 量：
(a) 粗掃牆鐘時間、(b) 密集牆鐘時間、(c) 兩次模型載入時間、(d) SP2 之前的總時間（切到 main 對照）。

- [ ] **Step 6: 記錄結果**

`notes/rep_segmentation_sp2_measurements.md` 寫下 T1、T2（引用 §2.8）、T3 的數字，
以及 T3 對「要不要接即時 landmarks」（spec §2.4）的結論。**如實記錄，包含整體沒有變快的情況。**

- [ ] **Step 7: 跑完整驗證**

```
.venv\Scripts\python.exe -m pytest tests/
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```
cwd = `frontend/`：`yarn test:coverage` 然後 `yarn build`

- [ ] **Step 8: Commit**

```bash
git add scripts/pose/capture_rep_signal_golden.py tests/fixtures/rep_signal_golden.json \
        frontend/src/test/lib.repSignal.golden.test.ts notes/rep_segmentation_sp2_measurements.md
git commit -m "test(pose): pin the TS rep signal against Python on real landmarks"
```

---

## Self-Review 結果

**Spec 覆蓋**：§2.1→Task 6；§2.1.1→Task 4；§2.2→Task 6 Step 5；§2.3→Task 7；§2.5→Task 1；
§2.6→Task 2/3/6；§2.7→Task 3/12；§2.8→Task 4/5；§3→Task 12（T2 已完成）；§4.1→Task 6；
§4.2→Task 9/10；§4.3→Task 7/9；§4.4→Task 8/11；§5→Task 11；§6→散在各 Task 的測試步驟；
§7→本計畫的 Task 順序；§8→Task 5 的三個斷言把三個風險釘成測試。

**未涵蓋且刻意如此**：§2.4 的即時 landmarks 重用（延後，由 T3 的數字決定）。

**Placeholder 掃描**：無。Task 9 的測試 helper 原本留白，已改寫成該檔案既有的
「直接呼叫 `analyze_pose`、不走 TestClient」慣例，並補上 `setUp` 樁必須新增的 `rep_plan` 參數
（漏掉會 `TypeError`）。

**型別一致性檢查**：
- `RepWindow` 在 TS（`repSegmentation.ts`）與 Python（`rep_segmentation.py`）都是
  `{index, start, end, partial}`，`start`/`end` 都是**序列位置**。Task 6 在寫進 payload 時才
  乘 `COARSE_STRIDE` 轉成 `start_frame`/`end_frame`，Task 9 讀回來時因為 payload 是全長陣列而
  直接等於位置——這條等式是方案 A（spec §2.2）的全部理由，改動任一端都會破壞它。
- `Refinement` = `true | false | "clipped"`，Task 4 產生、Task 6 寫進 segment、Task 9 不驗證它
  （診斷欄位，後端不因它改變行為，spec §4.2）。
- `RepsPlan`（TS，送出去的）與 `Analysis["reps"]`（TS，收回來的）**形狀不同**：前者用
  `segments[].analyzed`，後者另有 `detected` 與 `analyzed` 索引清單。Task 10 的型別分開定義，
  不要合併——spec §4.2 明文警告過這一點。
- `DEFAULT_MAX_REPS = 3` 在 `frontend/src/lib/repSpans.ts` 與 `backend/app/config.py` 各一份，
  註解互相引用。

**沒有涵蓋且刻意如此**：spec §2.4 的即時 landmarks 重用（延後，由 Task 12 的 T3 數字決定）。
