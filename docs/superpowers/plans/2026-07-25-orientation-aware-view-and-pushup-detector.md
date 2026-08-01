# Orientation-Aware View Estimation + Push-up Rule Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make view estimation valid for horizontal bodies (and stop it fabricating `side` from missing evidence), then ship the Push-up detector — the third movement — on top of trustworthy view gating.

**Architecture:** Phase 1 replaces the vertical-only body-extent denominator in `src/pose/view_estimation.py` with an extent measured along the body's own long axis, and removes the `mean_finite(default=0.0)` path that turns "no width evidence" into "maximally narrow → side". Phase 2 adds `src/pose/movements/pushup.py` following the existing `MovementDetector` contract (raw metrics → phases → cited rules), registered alongside squat and overhead press.

**Tech Stack:** Python 3.11/3.12, numpy, stdlib. Tests are `unittest.TestCase` under `tests/`. No new dependencies.

## Global Constraints

- **Interpreter:** `.venv\Scripts\python.exe` from the repo root. NEVER bare `python`/`pip`, never `source .venv/bin/activate`. This machine has NO `python` on PATH.
- **Tests:** `.venv\Scripts\python.exe -m pytest tests/` — always scope to `tests/`, never bare `pytest`.
- **Coverage gate (CI-enforced):** `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`.
- **Source of truth for every rule:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` (Push-up rules at lines ~296–430). If this plan and the spec disagree, **the spec wins** — stop and report the conflict.
- **Anti-hallucination (the project's whole premise):** every rule carries a real `citation` + `citation_support` copied from the spec. NEVER invent a threshold, an author, or an anthropometric constant. If a spec heuristic is not implementable with MediaPipe's 33 landmarks, say so and substitute explicitly — a **substitution is not a unit conversion**.
- **Never bend production code to make a test pass.** If a fixture cannot produce a condition, fix the fixture.
- **Squat is production.** `backend/app/services/analysis.py` and `library.py` hardcode `movement="Squat"`. Any change that moves a squat verdict is a regression unless explicitly approved.
- **MediaPipe coords:** normalized image space, `x,y ∈ [0,1]`, **y grows DOWNWARD**. Landmarks: 0 nose, 7/8 ears, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles.
- **All pose data under `data/` is gitignored.** Committed tests must not depend on it; data-backed checks must `skipUnless` the files exist.
- Commit messages end with: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

## Evidence this plan is built on (already measured — do not re-derive)

Measured on the 45 real pose JSONs in `data/runtime/pose_json/` + `data/Fitness-AQA/Squat/Labeled_Dataset/pose_json/`:

| Finding | Evidence |
|---|---|
| Current corpus verdicts | 30 `rear_oblique`, 13 `rear`, 1 `side`, 1 `unknown` |
| The **only** `side` verdict is fabricated | `vid1.json` — a degenerate fixture whose 33 landmarks are all identical, so `torso_width_ratio` is NaN in every frame; `mean_finite(default=0.0)` makes `narrow_body_signal` read **maximally narrow** → `side` @ conf **0.9** with zero width evidence |
| Vertical-only extent collapses on horizontal bodies | demo clips: squat y-extent **0.661** vs pushups **0.292** (2.3×), inflating `torso_width_ratio` 0.046 → 0.354 |
| Axis-relative extent is safe for upright bodies | **0 verdict flips** across all 45 files; max \|Δtorso_width_ratio\| 0.110, max \|Δconfidence\| 0.043 |
| Axis-relative extent fixes horizontal bodies | synthetic sagittal push-up: current code flips to `rear_oblique` once residual L/R separation `dx ≥ 0.02`; axis-relative holds `side` to `dx ≈ 0.09` (**4× wider margin**) |

**Known residual limitation (must be documented, NOT silently fixed):** `estimate_view_for_pose` is called with `allow_front=False` (`pose_rule_detector.py:564`), so production can only emit `side`, `rear_oblique`, `rear`, `unknown`. And `signed_orientation` (`view_estimation.py:169-174`) is `sign(left.x - right.x)`, whose front/rear meaning is **not validated for a horizontal body**. Phase 2 therefore must NOT gate any push-up rule on `front`/`rear` labels.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pose/view_estimation.py` (modify) | Add `body_axis_extent`; make `body_height` delegate to it; stop defaulting NaN width evidence to 0.0 |
| `tests/test_view_estimation.py` (modify) | Synthetic upright-vs-horizontal tests + the fabricated-`side` regression test |
| `tests/fixtures/view_baseline.json` (create) | Frozen verdicts for the 45-file corpus |
| `tests/test_view_regression_corpus.py` (create) | Data-conditional gate: re-runs the corpus, diffs against the frozen fixture, `skipUnless` data present |
| `scripts/pose/capture_view_baseline.py` (create) | Regenerates the fixture; documents how the gate is refreshed |
| `src/pose/movements/pushup.py` (create) | Push-up raw metrics, phase segmentation, 4 cited rules + 1 registered-but-silent rule |
| `src/pose/movements/registry.py` (modify) | Side-effect import of `pushup` |
| `tests/test_pushup.py` (create) | Fixture + metric, phase, rule firing/non-firing, boundary, and severity tests |
| Spec + `scripts/pose/README.md` (modify) | Record deviations and the new `--movement "Push-up"` value |

---

# PHASE 1 — Orientation-aware view estimation

### Task 1: Freeze the regression baseline BEFORE any behavior change

**Files:**
- Create: `scripts/pose/capture_view_baseline.py`
- Create: `tests/fixtures/view_baseline.json`
- Create: `tests/test_view_regression_corpus.py`

**Interfaces:**
- Consumes: `src.pose.view_estimation.estimate_view_for_pose(Path) -> ViewEstimate`
- Produces: `tests/fixtures/view_baseline.json` — `{relative_posix_path: {view_type, view_confidence, side_score, front_score, rear_score, oblique_score, torso_width_ratio_mean, orientation_score_mean, valid_frame_ratio, total_frames}}`, all floats rounded to 6dp.

- [ ] **Step 1: Write the capture script**

```python
"""Capture view-estimation verdicts over every real pose JSON in the repo.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/pose/capture_view_baseline.py

Writes tests/fixtures/view_baseline.json, the frozen corpus the
tests/test_view_regression_corpus.py gate diffs against. Regenerate ONLY when a
verdict change is intentional and reviewed -- this file is the record that a
refactor did not move production squat behavior.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pose.view_estimation import estimate_view_for_pose  # noqa: E402

CORPUS_ROOTS = (
    Path("data/runtime/pose_json"),
    Path("data/Fitness-AQA/Squat/Labeled_Dataset/pose_json"),
)
FIXTURE_PATH = Path("tests/fixtures/view_baseline.json")


def capture(repo_root: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for root in CORPUS_ROOTS:
        for path in sorted((repo_root / root).rglob("*.json")):
            estimate = estimate_view_for_pose(path)
            key = path.relative_to(repo_root).as_posix()
            records[key] = {
                "view_type": estimate.view_type,
                "view_confidence": round(estimate.view_confidence, 6),
                "side_score": round(estimate.side_score, 6),
                "front_score": round(estimate.front_score, 6),
                "rear_score": round(estimate.rear_score, 6),
                "oblique_score": round(estimate.oblique_score, 6),
                "torso_width_ratio_mean": round(estimate.torso_width_ratio_mean, 6),
                "orientation_score_mean": round(estimate.orientation_score_mean, 6),
                "valid_frame_ratio": round(estimate.valid_frame_ratio, 6),
                "total_frames": estimate.total_frames,
            }
    return records


def main() -> None:
    records = capture(REPO_ROOT)
    out = REPO_ROOT / FIXTURE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    counts: dict[str, int] = {}
    for record in records.values():
        counts[record["view_type"]] = counts.get(record["view_type"], 0) + 1
    print(f"{len(records)} files -> {FIXTURE_PATH}")
    for key in sorted(counts):
        print(f"  {key:15s} {counts[key]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the fixture and confirm the expected corpus shape**

Run: `.venv\Scripts\python.exe scripts/pose/capture_view_baseline.py`

Expected output (this exact distribution — if it differs, STOP and report; the corpus changed under you):
```
45 files -> tests/fixtures/view_baseline.json
  rear            13
  rear_oblique    30
  side             1
  unknown          1
```

- [ ] **Step 3: Write the data-conditional regression gate**

```python
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "view_baseline.json"


def _load_baseline() -> dict:
    if not FIXTURE.exists():
        return {}
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


BASELINE = _load_baseline()
AVAILABLE = [key for key in BASELINE if (REPO_ROOT / key).exists()]


class ViewRegressionCorpusTests(unittest.TestCase):
    """Pose corpora live under gitignored data/, so this gate is local-only: it
    skips in CI and on fresh clones, and bites on the machine that has the data.
    A verdict move here means production squat gating changed."""

    @unittest.skipUnless(AVAILABLE, "view baseline corpus not present (data/ is gitignored)")
    def test_view_verdicts_match_frozen_baseline(self) -> None:
        from src.pose.view_estimation import estimate_view_for_pose

        drifted = []
        for key in AVAILABLE:
            expected = BASELINE[key]
            actual = estimate_view_for_pose(REPO_ROOT / key)
            if actual.view_type != expected["view_type"]:
                drifted.append(
                    f"{key}: {expected['view_type']} -> {actual.view_type}"
                )
        self.assertEqual(drifted, [], f"view verdicts moved for {len(drifted)} file(s)")

    @unittest.skipUnless(AVAILABLE, "view baseline corpus not present (data/ is gitignored)")
    def test_confidence_does_not_drift_far(self) -> None:
        from src.pose.view_estimation import estimate_view_for_pose

        for key in AVAILABLE:
            expected = BASELINE[key]
            actual = estimate_view_for_pose(REPO_ROOT / key)
            self.assertLess(
                abs(actual.view_confidence - expected["view_confidence"]),
                0.10,
                f"{key}: confidence moved from {expected['view_confidence']} to {actual.view_confidence}",
            )
```

- [ ] **Step 4: Run the gate against UNCHANGED source — it must PASS**

Run: `.venv\Scripts\python.exe -m pytest tests/test_view_regression_corpus.py -v`
Expected: 2 passed (not skipped — the data is present on this machine). A skip here means the corpus paths are wrong; fix before continuing, or Phase 1 has no safety net.

- [ ] **Step 5: Commit**

```bash
git add scripts/pose/capture_view_baseline.py tests/fixtures/view_baseline.json tests/test_view_regression_corpus.py
git commit -m "test(pose): freeze view-estimation verdict baseline over the 45-file corpus"
```

---

### Task 2: Stop fabricating `side` from absent width evidence

**Files:**
- Modify: `src/pose/view_estimation.py:299` (the `torso_width_ratio` aggregation) and `score_view`'s NaN handling
- Test: `tests/test_view_estimation.py`

**Interfaces:**
- Consumes: `mean_finite(values, default) -> float`, `score_view(...) -> tuple[str, float, float, float, float, float]`
- Produces: no signature changes. Behavior change only: absent width evidence must no longer read as "narrow".

**Why:** `estimate_view_for_pose` aggregates with `mean_finite([...], default=0.0)`. `score_view` then computes `narrow_body_signal = clip01((0.24 - 0.0) / 0.16) = 1.0` — **maximally narrow** — so a clip with no measurable torso width scores `side` at 0.9. That is the classifier inventing evidence, which this project forbids. `orientation_score` and `z_asymmetry` defaulting to 0.0 are fine (0.0 genuinely means "no left/right bias" / "no depth split"); only the width ratio has a poisoned default.

- [ ] **Step 1: Write the failing test**

```python
    def test_absent_width_evidence_does_not_score_as_side(self) -> None:
        # A clip where torso width is unmeasurable in every frame must NOT be called
        # "side": narrow_body_signal previously read the 0.0 default as maximally narrow
        # and returned side @ 0.9 with no width evidence at all.
        view_type, confidence, _front, _rear, side_score, _oblique = score_view(
            orientation_score=0.0,
            face_visibility=0.5,
            torso_width_ratio=float("nan"),
            z_asymmetry_value=0.0,
            valid_frame_ratio=1.0,
        )
        self.assertNotEqual(view_type, "side")
        self.assertLess(side_score, 0.62)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_view_estimation.py::ViewEstimationTests::test_absent_width_evidence_does_not_score_as_side -v`
Expected: FAIL. (`score_view` already guards `np.isfinite(torso_width_ratio)`, so this test passes at the `score_view` level — the poisoned default is introduced by the *caller*. If it PASSES, that confirms the bug lives in `estimate_view_for_pose`; keep the test as a guard and move to Step 3, noting this in your report.)

- [ ] **Step 3: Propagate NaN instead of 0.0 at the aggregation site**

In `estimate_view_for_pose`, change ONLY the width aggregation (line ~299):

```python
    # NaN, not 0.0: a 0.0 width ratio reads as "maximally narrow" in
    # narrow_body_signal and manufactures a high-confidence `side` verdict from
    # clips that carry no width evidence at all. score_view already treats a
    # non-finite ratio as "no evidence" (both width signals fall to 0.0).
    torso_width_ratio = mean_finite(
        [signal["torso_width_ratio"] for signal in valid_signals], default=np.nan
    )
```

Leave `orientation_score`, `face_visibility`, and `z_asymmetry_value` defaults at `0.0` — those zeros are meaningful.

- [ ] **Step 4: Add the end-to-end test for the real failing file shape**

```python
    def test_degenerate_all_coincident_landmarks_is_not_side(self) -> None:
        # Mirrors data/runtime/pose_json/vid1.json: every landmark at the same point,
        # so shoulder/hip widths are 0 and body extent is 0 -> torso_width_ratio NaN in
        # every frame. This produced `side` @ conf 0.9 before the fix.
        import json, tempfile
        from pathlib import Path
        from src.pose.view_estimation import estimate_view_for_pose

        landmarks = [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.9} for _ in range(33)]
        payload = {"metadata": {}, "frames": [{"frame_index": 0, "landmarks": landmarks}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "degenerate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            estimate = estimate_view_for_pose(path)
        self.assertNotEqual(estimate.view_type, "side")
```

- [ ] **Step 5: Run both tests plus the corpus gate**

Run: `.venv\Scripts\python.exe -m pytest tests/test_view_estimation.py tests/test_view_regression_corpus.py -v`

Expected: the two new tests PASS. **The corpus gate will now FAIL on `vid1.json`** (`side -> unknown`) — that is the intended fix landing, and it is the ONLY file permitted to move. Verify no other file drifted; if any real upload moved, STOP and report.

- [ ] **Step 6: Re-freeze the baseline, recording the intended change**

Run: `.venv\Scripts\python.exe scripts/pose/capture_view_baseline.py`
Expected new distribution: `rear 13, rear_oblique 30, unknown 2` (the fabricated `side` is gone).

Run: `.venv\Scripts\python.exe -m pytest tests/test_view_regression_corpus.py -v` → 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/pose/view_estimation.py tests/test_view_estimation.py tests/fixtures/view_baseline.json
git commit -m "fix(pose): stop scoring absent torso-width evidence as a side view"
```

---

### Task 3: Measure body extent along the body's own long axis

**Files:**
- Modify: `src/pose/view_estimation.py:156-166`
- Test: `tests/test_view_estimation.py`

**Interfaces:**
- Produces: `body_axis_extent(points: np.ndarray | None) -> float` — extent of the visible body-landmark cloud projected onto the unit vector from shoulder-midpoint to ankle-midpoint (falling back to hip-midpoint, then to vertical). `body_height` is kept as a thin alias so existing callers and tests keep working.

**Why:** `body_height` returns the **y-extent only**, i.e. it assumes the subject is upright. For a horizontal body it measures thickness off the floor, not body length — a 2.3× collapse on the real demo clips — which inflates `torso_width_ratio` by the same factor and pushes sagittal push-ups out of the `side` band. Measured: axis-relative extent produces **0 verdict flips** on all 45 real upright files while widening the sagittal push-up margin 4×.

- [ ] **Step 1: Write the failing tests**

```python
    def test_axis_extent_matches_y_extent_for_an_upright_body(self) -> None:
        # For a vertical body axis the projection reduces to the y-extent, so the
        # upright behaviour this classifier was tuned on must be preserved exactly.
        from src.pose.view_estimation import body_axis_extent, landmarks_to_array

        points = landmarks_to_array(_upright_landmarks())
        self.assertAlmostEqual(body_axis_extent(points), 0.60, delta=0.02)

    def test_axis_extent_recovers_body_length_when_horizontal(self) -> None:
        # Same body rotated 90 degrees: the y-extent collapses to the body's thickness,
        # but the axis-relative extent must still recover its full length.
        from src.pose.view_estimation import body_axis_extent, landmarks_to_array

        points = landmarks_to_array(_horizontal_landmarks())
        self.assertGreater(body_axis_extent(points), 0.45)
```

with these fixture helpers at module level:

```python
def _landmark(x: float, y: float, visibility: float = 0.95) -> dict:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def _upright_landmarks() -> list[dict]:
    """Standing subject: shoulders high, ankles low, body axis vertical (extent ~0.60)."""
    lm = [_landmark(0.5, 0.5) for _ in range(33)]
    lm[11], lm[12] = _landmark(0.46, 0.20), _landmark(0.54, 0.20)
    lm[23], lm[24] = _landmark(0.47, 0.50), _landmark(0.53, 0.50)
    lm[25], lm[26] = _landmark(0.47, 0.65), _landmark(0.53, 0.65)
    lm[27], lm[28] = _landmark(0.47, 0.80), _landmark(0.53, 0.80)
    return lm


def _horizontal_landmarks() -> list[dict]:
    """The same body rotated 90 degrees (push-up/plank): axis runs along image x."""
    lm = [_landmark(0.5, 0.5) for _ in range(33)]
    lm[11], lm[12] = _landmark(0.20, 0.46), _landmark(0.20, 0.54)
    lm[23], lm[24] = _landmark(0.50, 0.47), _landmark(0.50, 0.53)
    lm[25], lm[26] = _landmark(0.65, 0.47), _landmark(0.65, 0.53)
    lm[27], lm[28] = _landmark(0.80, 0.47), _landmark(0.80, 0.53)
    return lm
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_view_estimation.py -k axis_extent -v`
Expected: FAIL — `cannot import name 'body_axis_extent'`.

- [ ] **Step 3: Implement `body_axis_extent`**

Replace `body_height` (lines 156-166) with:

```python
def _visible_midpoint(
    points: np.ndarray | None, left_index: int, right_index: int, min_visibility: float = 0.35
) -> np.ndarray | None:
    left = visible_point(points, left_index, min_visibility=min_visibility)
    right = visible_point(points, right_index, min_visibility=min_visibility)
    if left is None or right is None:
        return None
    return (left[:2] + right[:2]) / 2.0


def body_axis_extent(points: np.ndarray | None) -> float:
    """Extent of the visible body-landmark cloud measured ALONG the body's long axis.

    The axis runs shoulder-midpoint -> ankle-midpoint, falling back to the hip
    midpoint when ankles are not visible and finally to vertical. For an upright
    subject the axis IS vertical, so this reduces exactly to the y-extent the
    original implementation used -- verified as 0 verdict flips across the 45-file
    corpus. For a horizontal subject (push-up, plank) the y-extent measures only the
    body's thickness off the floor, which inflates torso_width_ratio and pushes a
    true sagittal view out of the `side` band; measuring along the body's own axis
    recovers its length instead.
    """
    if points is None:
        return np.nan

    coords: list[np.ndarray] = []
    for index in BODY_LANDMARKS:
        point = visible_point(points, index, min_visibility=0.35)
        if point is not None:
            coords.append(point[:2])
    if len(coords) < 4:
        return np.nan

    shoulder_mid = _visible_midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER)
    far_mid = _visible_midpoint(points, LEFT_ANKLE, RIGHT_ANKLE)
    if far_mid is None:
        far_mid = _visible_midpoint(points, LEFT_HIP, RIGHT_HIP)

    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    if shoulder_mid is not None and far_mid is not None:
        vector = np.asarray(far_mid, dtype=np.float64) - np.asarray(shoulder_mid, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            axis = vector / norm

    projections = [float(np.dot(np.asarray(point, dtype=np.float64), axis)) for point in coords]
    return float(max(projections) - min(projections))


def body_height(points: np.ndarray | None) -> float:
    """Backwards-compatible alias. Body "height" is now measured along the body's own
    axis rather than the image's vertical, so the name is a slight misnomer kept for
    existing callers; prefer `body_axis_extent` in new code."""
    return body_axis_extent(points)
```

Confirm `LEFT_ANKLE` / `RIGHT_ANKLE` are imported/defined in this module; add them next to the existing landmark constants if not.

- [ ] **Step 4: Run the new tests, the module's suite, and the corpus gate**

Run: `.venv\Scripts\python.exe -m pytest tests/test_view_estimation.py tests/test_view_regression_corpus.py -v`
Expected: ALL PASS, **including the corpus gate with zero verdict drift**. If any file's verdict moved, STOP — the measured expectation is 0 flips; investigate rather than re-freezing.

- [ ] **Step 5: Add the horizontal-body end-to-end discrimination test**

```python
    def test_sagittal_horizontal_body_scores_side_not_oblique(self) -> None:
        # A sagittal push-up with a realistic residual left/right separation was
        # misclassified `rear_oblique` when body extent was measured vertically,
        # because the collapsed denominator inflated torso_width_ratio ~2.3x.
        from src.pose.view_estimation import (
            body_axis_extent, landmarks_to_array, mean_finite, score_view, xy_distance,
        )

        lm = _horizontal_landmarks()
        for index in (12, 24, 26, 28):          # nudge the far side to a 0.04 residual gap
            lm[index] = _landmark(lm[index]["x"] + 0.04, lm[index]["y"])
        points = landmarks_to_array(lm)
        width = mean_finite([xy_distance(points, 11, 12), xy_distance(points, 23, 24)],
                            default=float("nan"))
        ratio = width / body_axis_extent(points)
        view_type, _confidence, _f, _r, _s, _o = score_view(
            orientation_score=0.0, face_visibility=0.5, torso_width_ratio=ratio,
            z_asymmetry_value=0.0, valid_frame_ratio=1.0,
        )
        self.assertEqual(view_type, "side")
```

- [ ] **Step 6: Run the full suite and the coverage gate**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: all pass. Known pre-existing flake: `tests/test_analyze_endpoint.py::test_concurrent_analyses_are_bounded` — if ONLY that fails, note it and move on.

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`

- [ ] **Step 7: Commit**

```bash
git add src/pose/view_estimation.py tests/test_view_estimation.py
git commit -m "fix(pose): measure body extent along the body axis so horizontal poses classify correctly"
```

---

### Task 4: Document the orientation limits that remain

**Files:**
- Modify: `src/pose/view_estimation.py` (module docstring)
- Modify: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` (§7 limitations) and its `.zh-TW.md` mirror

- [ ] **Step 1: Add a module-level note stating exactly what is and is not fixed**

```python
"""...existing docstring...

Orientation support (2026-07-25)
--------------------------------
Body extent is measured along the body's own long axis (`body_axis_extent`), so the
narrow/broad torso signal is valid for horizontal subjects (push-up, plank) as well as
upright ones. Two limits remain and are NOT fixed here:

1. `signed_orientation` is `sign(left.x - right.x)`, an image-space left/right ordering.
   Its front/rear meaning is validated only for UPRIGHT subjects; for a horizontal body
   the frontal axis no longer maps onto image x, so `front`/`rear`/`*_oblique` labels
   carry no validated meaning. Do not gate a horizontal-movement rule on them.
2. `estimate_view_for_pose` is called with `allow_front=False` in the production path
   (`pose_rule_detector.py`), so `front` and `front_oblique` are unreachable there.
"""
```

- [ ] **Step 2: Mirror both limits into the spec's §7 limitations section (EN and zh-TW)**

- [ ] **Step 3: Commit**

```bash
git add src/pose/view_estimation.py docs/superpowers/specs/
git commit -m "docs(pose): document the view-estimation orientation limits that remain"
```

---

# PHASE 2 — Push-up detector

Spec rules: `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` lines ~296–430. **Five rules; four implementable.**

| fault_id | Implementable | Primary citation |
|---|---|---|
| `pushup_hip_sag` | yes | Freeman/McGill Med Sci Sports Exerc (2006) DOI 10.1249/01.mss.0000189317.08635.1b |
| `pushup_shallow_depth` | yes | San Juan JG et al. BMC Musculoskelet Disord (2015) PMC4327800 |
| `pushup_elbow_flare` | yes (self-gating, see Task 6) | Donkers MJ et al. J Biomech (1993) DOI 10.1016/0021-9290(93)90026-b |
| `pushup_head_drop` | yes | Lee S et al. J Phys Ther Sci (2013) PMC3820220 + Al Hammadi PMC12514857 |
| `pushup_scapular_winging` | **NO — observability `none`** | Lee S et al. PMC3820220 + Abdollahi S et al. PMC12366113 |

**Decision already made (do not relitigate):** `pushup_scapular_winging` is **registered but never emits**. MediaPipe's 33 landmarks contain no scapular border points, so it is not measurable from any view. It stays in the detector as a permanently-silent, cited, documented rule so the spec and code remain in 1:1 correspondence and nobody later "implements" it by inventing a proxy.

### Task 5: Push-up raw metrics and phase segmentation

**Files:**
- Create: `src/pose/movements/pushup.py`
- Create: `tests/test_pushup.py`

**Interfaces:**
- Consumes: `src.pose.geometry` (`landmarks_to_array`, `visible_point`, `distance`, `angle_degrees`, `midpoint`, `mean_visibility`, `mean_finite`, `contiguous_true_segments`, `severity_from_range`), `src.pose.movements.base` (`CoreFrame`, `RuleContext`, `MovementDetector`, `run_detector`)
- Produces: `PUSHUP_METRIC_KEYS: tuple[str, ...]`, `pushup_compute_raw(frames, fps) -> list[dict]`, `pushup_assign_phases(raw) -> list[str]` with phases in `{"setup", "descent", "bottom", "ascent", "unknown"}`

**Metrics to compute** (all normalized so they are scale-free; y grows DOWNWARD):

| key | definition |
|---|---|
| `left_elbow_angle` / `right_elbow_angle` / `min_elbow_angle` | `angle_degrees` at elbow (shoulder→elbow→wrist); `min_` = the more-flexed (smaller) of the two finite sides |
| `hip_offset_ratio` | **signed** perpendicular offset of hip-midpoint from the shoulder-mid→ankle-mid line, normalized by that line's length. **Positive = hip toward the ground (sag); negative = pike.** Sign is resolved from the body-axis normal, not from image y, so it is independent of which way the subject faces |
| `plank_angle_deviation_deg` | `abs(180 - angle_degrees(shoulder_mid, hip_mid, ankle_mid))` — the spec's stated equivalent to the offset criterion |
| `hand_width_ratio` | `distance(wrist15, wrist16) / distance(shoulder11, shoulder12)` |
| `neck_line_angle_deg` | angle at the shoulder between the ear→shoulder vector and the shoulder→hip torso vector, minus its own per-clip baseline (see Task 7) |
| `body_axis_tilt_deg` | angle of the shoulder-mid→ankle-mid vector from image horizontal — a diagnostic that lets tests assert the subject really is horizontal |

- [ ] **Step 1: Write the failing metric tests** — a `pushup_frame(...)` fixture with named knobs (`elbow_angle`, `hip_offset`, `hand_width_ratio`, `ear_offset`), built so **each metric is controlled by construction, not hardcoded**. Reuse the perpendicular-bisector elbow construction from `tests/test_overhead_press.py::_elbow_xy` (`d = h / tan(radians(angle)/2)`) — it is the proven way to make `elbow_angle` actually track the requested value. Assert: `min_elbow_angle` tracks the requested angle within 3°; `hip_offset_ratio` is POSITIVE for a sag and NEGATIVE for a pike; `hand_width_ratio` equals the requested ratio; `body_axis_tilt_deg` is near 0 for a horizontal body.

- [ ] **Step 2: Run to verify failure** — `.venv\Scripts\python.exe -m pytest tests/test_pushup.py -v` → FAIL, module does not exist.

- [ ] **Step 3: Implement `pushup_compute_raw`** following the exact structure of `ohp_compute_raw` (`src/pose/movements/overhead_press.py:59-158`): per-frame validity via a `required` landmark tuple, `visible_point` gating, NaN for unmeasurable metrics, one dict per frame. `required` must include shoulders, elbows, wrists, hips **and ankles** (the plank line needs them). Document at module level that requiring ankles means a clip cropped at the knees silences ALL push-up rules — the same validity-gate effect documented for OHP.

- [ ] **Step 4: Implement `pushup_assign_phases`** — segment on `min_elbow_angle`: first ~15% of frames `setup`; `bottom` where elbow angle is at/below its 30th percentile; `descent` before the deepest frame; `ascent` after. Mirror `ohp_assign_phases` (`overhead_press.py:161-208`) including the `unknown` fallback when no valid frames exist.

- [ ] **Step 5: Run tests → PASS. Step 6: Commit**

```bash
git add src/pose/movements/pushup.py tests/test_pushup.py
git commit -m "feat(pose): push-up raw metrics and phase segmentation"
```

---

### Task 6: The two sagittal rules — hip sag and shallow depth

**Files:**
- Modify: `src/pose/movements/pushup.py`, `tests/test_pushup.py`

**Interfaces:**
- Produces: `rule_hip_sag(core, ctx) -> list[PoseRuleDetection]`, `rule_shallow_depth(core, ctx) -> list[PoseRuleDetection]`

**Numbers — TWO CATEGORIES, label them differently in the code. Do NOT call the ramps "spec-copied".**

*(Corrected after the Task 6 review. This heading previously read "Thresholds — copied from the spec, NOT invented" and covered the severity ramps too, which is false: the spec's Push-up section, lines 292–433, contains no `Severity ramp` line and neither the string `0.15` nor `140`. `grep -n "pushup_hip_sag\|pushup_shallow_depth"` over the spec returns exactly two lines — the two `fault_id` lines. The Squat section DOES state ramps explicitly ("Severity ramp 0.82 → 0.70"), so the absence is meaningful, not a formatting quirk.)*

**(a) Fire thresholds — copied from the spec, NOT invented:**
- `pushup_hip_sag`: `hip_offset_ratio > 0.06` (sag) or `< -0.06` (pike); equivalently `plank_angle_deviation_deg > 12`. Observability `high` on `side`, near-`none` from `front`/`rear`.
- `pushup_shallow_depth`: at the bottom phase, `min_elbow_angle > 100` is the spec's lower bound of its "~100–110°" band. **Use 100 and state in a comment that the spec gives a band and 100 is its conservative (fewer false positives) end.** Observability `high` on `side`/`front_oblique`.

**(b) Severity ramps — RULE-LEVEL CHOICES, not in the spec. Keep the values, but document the reasoning rather than claiming provenance:**
- `pushup_hip_sag`: ramp 0.06 → 0.15 (0.15 ≈ 2.5× the fire threshold; no source fixes that multiple).
- `pushup_shallow_depth`: ramp 100 → 140 (140° is well past the spec's own "a full rep reaches roughly ≤90°").

- [ ] **Step 1: Write failing firing AND non-firing tests for both rules**, plus **boundary tests just inside and just outside each threshold** and **one exact severity-value assertion per rule**. (The prior OHP review found 5 of 10 threshold mutants surviving because every fixture sat at an extreme — do not repeat that.) Build boundary fixtures as constant-value frames so the median smoothing in `run_detector` is a no-op and the asserted severity is exact.
- [ ] **Step 2: Run → FAIL. Step 3: Implement both rules** following `rule_excessive_back_lean` (`overhead_press.py:319-365`) for structure: build a phase-scoped mask, walk `contiguous_true_segments(mask, ctx.min_frames)`, compute severity via `severity_from_range`, emit through `build_detection` with `citation` + `citation_support` **copied verbatim from the spec**.
- [ ] **Step 4: Distinguish sag from pike in the SAME rule** — both are `pushup_hip_sag` per the spec, but the evidence dict must record which (`"direction": "sag"` or `"pike"`) so feedback is not inverted. Add a test asserting a pike is reported as a pike.
- [ ] **Step 5: Run → PASS. Step 6: Commit**

```bash
git commit -m "feat(pose): push-up hip-sag and shallow-depth cited rules"
```

---

### Task 7: Head drop, self-gating elbow flare, and the silent winging rule

**Files:**
- Modify: `src/pose/movements/pushup.py`, `tests/test_pushup.py`

**Interfaces:**
- Produces: `rule_head_drop`, `rule_elbow_flare`, `rule_scapular_winging` (always returns `[]`)

- [ ] **Step 1: `rule_head_drop`** — fires when `neck_line_angle_deg` deviates by `> 15` (spec's number — verified present as "by > ~15°"; note the spec states NO severity ramp for this fault either, so whatever ramp is chosen must be labelled a rule-level choice, as in Task 6) during descent/bottom. **Per-clip baseline:** measure the neck-line angle over the `setup` phase and flag deviation FROM that baseline, not from an absolute value — absolute neck angle varies with individual anatomy and camera height, and the spec gives no absolute reference. Document this as a spec deviation. Observability `medium` on `side`/`front_oblique`.

- [ ] **Step 2: `rule_elbow_flare` — self-gating on measurability, NOT on view label.** The spec asks for a `front`/`rear` view, but Task 4 established those labels have no validated meaning for a horizontal body, and `front` is unreachable in production anyway. So gate on whether the metric is *physically measurable*: fire only when the wrists are genuinely separated in image space (`distance(wrist15, wrist16) > 0.25 * shoulder_width` — i.e. the camera is looking down the body's long axis; from a true sagittal view the wrists overlap and this is near zero). Fire threshold `hand_width_ratio > 1.6` **is from the spec** ("flag when ratio > ~1.6"); the severity ramp 1.6 → 2.2 **is NOT — it is a rule-level choice**, verified: `2.2` appears nowhere in the spec's Push-up section (lines 292–433). Label the two differently in the docstring, exactly as Task 6 does. Observability `medium`.
  Write a comment explaining WHY this deviates from the spec's view gating, and a test proving a sagittal fixture (overlapping wrists) does NOT fire it.

- [ ] **Step 3: `rule_scapular_winging` — registered, never emits.**

```python
def rule_scapular_winging(core: list[CoreFrame], ctx: RuleContext) -> list[PoseRuleDetection]:
    """Registered but PERMANENTLY SILENT -- always returns [].

    Scapular winging is a real, well-cited push-up fault (serratus anterior weakness
    lets the scapula wing, narrowing the subacromial space), but MediaPipe's 33
    landmarks contain NO scapular border points, so it cannot be measured from any
    view. The spec marks it observability `none`.

    It is registered rather than omitted so the spec and the code stay in 1:1
    correspondence: anyone auditing "are all 5 push-up rules present?" finds it here
    with its citation and this explanation, instead of finding a gap and "fixing" it
    by inventing an unvalidated proxy (e.g. upper-back rounding from a rear view,
    which the spec explicitly calls untrustworthy).

    Citation: Lee S, Lee D, Park J. J Phys Ther Sci (2013) PMC3820220; corroborated by
    Abdollahi S et al. J Orthop Surg Res (2025) PMC12366113.
    """
    return []
```

- [ ] **Step 4: Write a test that pins the silence deliberately**

```python
    def test_scapular_winging_never_emits(self) -> None:
        # Pinned on purpose: MediaPipe 33 has no scapular landmarks, so this rule must
        # stay silent. If someone implements a proxy, this test should fail and force a
        # spec conversation rather than shipping an unvalidated verdict.
        from src.pose.movements.pushup import rule_scapular_winging
        frames = [pushup_frame(elbow_angle=90, frame_index=i) for i in range(12)]
        core, _ = run_detector(PUSHUP_DETECTOR, frames, 30.0, "side", 0.8)
        self.assertEqual(rule_scapular_winging(core, _ctx()), [])
```

- [ ] **Step 5: Run → PASS. Step 6: Commit**

```bash
git commit -m "feat(pose): push-up head-drop, self-gating elbow-flare, silent winging rule"
```

---

### Task 8: Register, wire the CLI, and document

**Files:**
- Modify: `src/pose/movements/pushup.py` (detector assembly), `src/pose/movements/registry.py`, `scripts/pose/README.md`, spec §8 (EN + zh-TW)
- Test: `tests/test_movement_registry.py`

- [ ] **Step 1: Assemble and register the detector**

```python
PUSHUP_DETECTOR = MovementDetector(
    "Push-up",
    PUSHUP_METRIC_KEYS,
    pushup_compute_raw,
    pushup_assign_phases,
    (rule_hip_sag, rule_shallow_depth, rule_elbow_flare, rule_head_drop, rule_scapular_winging),
)

registry.register(PUSHUP_DETECTOR)
```

- [ ] **Step 2: Add the side-effect import in `registry.py`** next to the existing `squat` / `overhead_press` imports.
- [ ] **Step 3: Test registry resolution** — `get_detector("Push-up")` and `get_detector("push-up")` both resolve; an unknown movement still raises `KeyError` (no silent squat fallback); **the squat byte-for-byte gate test still passes.**
- [ ] **Step 4: Run the FULL verification set**

```
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
.venv\Scripts\python.exe -m pytest tests/test_view_regression_corpus.py -v
```

- [ ] **Step 5: Document** — add `Push-up` to the `--movement` values in `scripts/pose/README.md`; add a spec §8 status block (EN + zh-TW) recording: Push-up 4/5 rules live + 1 permanently silent; the head-drop per-clip-baseline deviation; the elbow-flare measurability gate replacing the spec's view gate; and that **push-up thresholds are spec-derived and UNVALIDATED** — no labeled push-up data exists in the repo (REHAB24-6 Ex3 is standing table push-ups; EgoExo push-up frames are an unextracted 3 GB archive).
- [ ] **Step 6: Confirm the product surface is unchanged** — `backend/app/services/analysis.py` and `library.py` still hardcode `movement="Squat"`, and `frontend/src/lib/movements.ts` still lists `ANALYZABLE_MOVEMENTS = ["Squat"]`. Push-up must remain CLI-only. State this explicitly in the report.
- [ ] **Step 7: Commit**

```bash
git commit -m "feat(pose): register the push-up detector and expose it via --movement"
```

---

## Self-Review

**Spec coverage.** All 5 push-up spec rules are accounted for: 4 implemented (Tasks 6–7), 1 registered-but-silent with its citation (Task 7 Step 3). No rule is silently dropped. Phase 1 is not spec'd rule work — it is the prerequisite that makes the spec's `side`-gated push-up heuristics meaningful, justified by the measured evidence table above.

**Deviations from the spec, all deliberate and all documented in-code + in the spec:**
1. `pushup_shallow_depth` uses 100° from the spec's "~100–110°" band (conservative end).
2. `pushup_head_drop` uses a per-clip setup-phase baseline instead of an absolute neck angle — the spec supplies a deviation threshold (15°) but no absolute reference.
3. `pushup_elbow_flare` gates on metric measurability (wrist separation) instead of the spec's `front`/`rear` view label, because Task 4 establishes those labels are not validated for horizontal bodies and `front` is unreachable in production.

**Placeholder scan.** No TBDs. Every code step carries real code; every threshold is either copied from the spec or explicitly flagged as a documented deviation. Citations are copied from the spec at implementation time, never recalled from memory.

**Type consistency.** `CoreFrame.m(key)`, `RuleContext(fps, view_type, view_confidence, min_frames)`, `MovementDetector(name, metric_keys, compute_raw, assign_phases, rules)`, `run_detector(...) -> (core, detections)`, and `registry.register/get_detector` are used identically to `squat.py` and `overhead_press.py`. `body_axis_extent` is introduced in Task 3 and `body_height` retained as an alias so no existing caller breaks.

**Honesty.** Push-up thresholds are spec-derived and unvalidated — there is no labeled push-up data in this repo, and the plan says so in Task 8. Synthetic fixtures prove geometry, not real-world fault detection. Phase 1's safety claim rests on a measured 45-file corpus, not on judgement.

**Risk register.**
- Task 2 intentionally moves exactly ONE corpus verdict (`vid1.json`, a degenerate all-coincident-landmark fixture). Any other movement is a STOP condition.
- Task 3 is expected to move ZERO verdicts. Any movement is a STOP condition.
- The corpus gate skips on machines without `data/` — it protects this machine, not CI. Synthetic tests in `tests/test_view_estimation.py` are the CI-visible protection.
