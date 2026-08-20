# RS-SP2 measurement tasks (spec §3)

Spec: `docs/superpowers/specs/2026-07-27-rep-segmentation-sp2-design.md`. Three measurement tasks
decide constants the design depends on. T1 and T2 are done; T3 needs a real device and is
recorded honestly as not yet measured.

## T1 — TS/Python signal parity (DONE)

**Question:** does `frontend/src/lib/repSignal.ts`'s `avgKneeAngle` compute the same angle as
`src/pose/pose_rule_detector.py`'s `raw_frame_metrics` on the same real landmarks?

**Method:** `scripts/pose/capture_rep_signal_golden.py` runs `raw_frame_metrics` on every 3rd frame
(`STRIDE = 3`, `MAX_FRAMES = 60`) of a real clip and freezes `{landmarks, avg_knee_angle}` pairs
into `tests/fixtures/rep_signal_golden.json` (committed — only a few dozen frames, and without it
the vitest side has no ground truth). `frontend/src/test/lib.repSignal.golden.test.ts` replays
each frame's landmarks through `avgKneeAngle` and diffs against the frozen Python value. This is a
standing test, not a one-off script: any future change to either implementation that breaks parity
turns it red.

**Source clip:** `data/runtime/pose_json/upload_14ca3c958efa.json` (gitignored, not committed —
recorded via the app's own capture path; 86 frames total, fps 30). Chosen because it's a single
real squat with a wide angle range, not a static or degenerate clip.

**Golden file statistics:**

| | |
|---|---|
| cases | 29 |
| measurable (`avg_knee_angle` not null) | 27 |
| unmeasurable (null) | 2 |
| angle range (measurable cases) | 65.26° – 163.72° |

The 2 null cases are not the `landmarks: null` early-return path — they're genuine low-visibility
frames (both ankles below the 0.5 visibility threshold, e.g. frame 0: left ankle visibility 0.462,
right ankle 0.420), so the visibility gate itself is exercised, not bypassed.

**Result: PASS.** Max absolute difference across all 27 measurable cases: **1.4472295958967152e-05
degrees** (frame 28), measured by a temporary `console.log` inside the test (removed after
recording this number).

**Deviation from the brief, declared:** the brief's draft test used tolerance `1e-6` with a comment
that the residual would be "float32-vs-float64 in the Python side's arithmetic." Running against
this fixture, 9 of the 27 measurable cases exceeded `1e-6` (max 1.4472e-5), so the test failed
as first written. Investigation (recomputing the angle in float64 by hand from the same landmarks,
independent of both implementations) confirmed this is exactly that predicted mechanism, not a
port bug:

- Python's numeric path is float32 end to end — `landmarks_to_array` (`src/pose/geometry.py:28`),
  `visible_point` (`:49`), and `mean_finite` (`:109`, which re-quantizes the already-float32 angles
  when averaging left/right) all use `dtype=np.float32`.
- JS numbers are float64.
- float32 epsilon is ~1.19e-7; at angle magnitudes of ~100°, that predicts absolute noise on the
  order of 1e-5 — matching the measured 1.4472e-5 max almost exactly.
- A hand-computed float64 replica of Python's own formula (same landmarks, same angle_degrees
  algorithm, no float32 casts) reproduces the TS output, not the float32 golden value — i.e. the TS
  port and the algorithm agree; only Python's storage precision differs.
- A genuine port bug (2-D gate collapsed into 3-D, wrong landmark index, the historical NaN-`z`
  divergence) shows up as multiple degrees of error or a NaN/number mismatch, not a fraction of a
  thousandth of a degree.

The tolerance was changed to a named constant, `FLOAT32_TOLERANCE_DEGREES = 1e-4`
(`frontend/src/test/lib.repSignal.golden.test.ts`), ~7x the measured max and ~4 orders of magnitude
below what a real divergence would produce — chosen to not be tuned to this one fixture's exact
max. The brief's own "don't loosen tolerance to hide a bug" instruction is an anti-cheating
guardrail; this is a correction of a mis-set constant given real data, not an evasion of it — the
port itself changed nothing.

**What this golden file does NOT prove.** Real MediaPipe landmarks always populate `z` (finite,
even if low-visibility), so this fixture cannot exercise the specific divergence a July 2026 review
caught: collapsing Python's 2-D validity gate (`dims=2`, allows NaN `z`) and 3-D angle computation
(`dims=3`) into a single 3-D check, which made a frame with one ankle's `z` as `NaN` return `NaN`
instead of the other leg's angle alone. That case is covered separately, by a deliberately
constructed unit test: `frontend/src/test/lib.repSignal.test.ts`, "allows NaN z during the validity
gate, computing the metric from the other side" (asserts `avgKneeAngle` returns 180, the left leg
alone, when the right ankle's `z` is `NaN`). The golden file and that constructed test are
complementary, not redundant: real data proves everyday parity, the constructed case proves the
dims=2/dims=3 split survives.

**To regenerate** (only when the Python formula deliberately changes):
```
.venv\Scripts\python.exe scripts/pose/capture_rep_signal_golden.py <path-to-pose.json>
```

## T2 — coarse-pass span width (DONE, cited)

Already measured while writing the spec — see spec §2.8 and `tests/test_coarse_segmentation_corpus.py`
(the standing test that keeps the constant honest; skips in CI because `data/` is gitignored, runs
locally against real pose JSON). Not re-derived here.

- Corpus: 46 real clips, 70 reps (`data/runtime/pose_json` + `data/Fitness-AQA/Squat/Labeled_Dataset/pose_json`).
- Coarse-pass **boundary** error is large and does not shrink with denser sampling (p95 15 frames,
  p99 36, max 45 @ 30fps) — this is why spans anchor on the coarse valley, not the coarse boundary.
- Coarse-pass **valley** (argmin) error is small and stable: p99 5 frames, max 5.
- `REP_PADDING_FRAMES = 24` gives 98.6% span coverage (span contains the true dense-derived
  interval) at a median span width of 4.4s; this is the constant actually adopted
  (`frontend/src/lib/repSpans.ts`, `tests/test_coarse_segmentation_corpus.py`).
- Known residual, honestly recorded in the spec: the coarse pass can miss or invent a whole
  repetition in ~4% of clips (2/46); this is not fixable by padding or refinement, since both only
  operate inside a span that already exists.

## T3 — coarse-vs-dense wall-clock share (NOT YET MEASURED)

**Question (spec §2.4):** what fraction of total client-side extraction time does the coarse pass
(plus its model load) cost, relative to the dense pass and the pre-SP2 single-pass baseline? This
number, not a guess, decides whether it's worth reusing live recording-time landmarks as the coarse
signal instead of running a dedicated coarse pass (deferred by design in §2.4 specifically pending
this measurement).

**This requires a real browser, a camera, and a person performing a squat** — it is not something
this task fabricates or simulates. It has not been run. The instrumentation to make the
measurement possible has been added; the actual numbers are outstanding.

### Instrumentation added

`frontend/src/lib/poseExtract.ts`, inside `extractPoseWithReps` (the two-pass extractor; the whole
function is `/* c8 ignore */`d as browser/WASM glue unrunnable under jsdom, and is mocked out in
every existing test, so this adds no test surface):

- A module-level `timingEnabled()` gate, read from `localStorage.getItem("xcoach.repSignalTiming") === "1"`.
  Off by default. One `localStorage.getItem` per extraction call — negligible — and when off, zero
  `performance.now()` calls and zero console output. This does not degrade normal use.
- When enabled, logs four `performance.now()` deltas via `console.log`, one line each, prefixed
  `[repSignalTiming]`:
  - `coarse model load` — time to construct the Lite landmarker for the coarse pass.
  - `coarse pass` — wall clock of `sampleFrames` over every `COARSE_STRIDE`-th frame.
  - `dense model load` — time to construct the analysis-tier landmarker for the dense pass.
  - `dense pass` — wall clock of `sampleFrames` over the padded, merged spans.
  - `total` — start-to-finish of `extractPoseWithReps`, for comparing against the pre-SP2
    single-pass baseline on `main`.

### Steps to collect T3 (for whoever runs this on a real device)

1. `cd frontend && yarn dev`, open the app in a desktop or phone browser, camera-enabled.
2. In devtools console, run: `localStorage.setItem("xcoach.repSignalTiming", "1")`.
3. Record (or upload) a ~30 second clip of 5 squats with a full range of motion.
4. Run analysis. Read the five `[repSignalTiming]` lines from the console:
   (a) coarse pass wall clock, (b) dense pass wall clock, (c) coarse model load,
   (d) dense model load, and total.
5. For the "did this get faster overall" comparison, checkout `main` (pre-SP2, single-pass
   `extractPoseFromBlob`), repeat the same clip, and time total extraction (no flag needed there —
   that path has no coarse/dense split to instrument).
6. Record all five SP2 numbers plus the `main` baseline total here, and revisit the §2.4 decision
   (reuse live landmarks vs. keep the dedicated coarse pass) using the actual coarse-pass share of
   total time, not an estimate.
7. `localStorage.removeItem("xcoach.repSignalTiming")` when done, to stop console output.

**Numbers to record here once measured:** coarse pass ms, dense pass ms, coarse model load ms,
dense model load ms, SP2 total ms, `main` baseline total ms, and the resulting §2.4 recommendation
(keep dedicated coarse pass / reuse live landmarks / inconclusive). None of these are filled in —
an honest "not yet measured" stands until a real run produces them.
