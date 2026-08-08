# Band Pull Apart Rule Detector — Design Spec

**Status:** design spec (approved, pending implementation plan) · **Date:** 2026-08-09
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §Band Pull Apart (lines 713–769)
**Sibling precedents:** `docs/superpowers/specs/2026-08-01-row-detector-design.md`, `src/pose/movements/{row,pushup,lunge,overhead_press}.py`

---

## 1. Purpose and why Band Pull Apart is next

Ship a cited, unvalidated (Beta) rule detector for the standing **Band Pull Apart** — the second
and last movement of the parent spec's Group C, and the seventh of sixteen.

"Next" is meant literally. The six shipped detectors are exactly the first six movements in
parent-spec order — Group A (Squat, Lunge, Deadlift), Group B (Push-up, Overhead Press), Group C
(Row) — so Band Pull Apart is the next entry with no interpretation required.

The parent spec lists **four** faults. This detector ships **three that can fire** and **one
registered-but-permanently-silent**, for the reasons in §3. `BAND_PULL_APART_DETECTOR.validated`
stays `False` (§2).

### 1.1 What makes this movement different from the six already shipped

Every prior detector measured a **sagittal** excursion — knee angle, elbow angle, hip height,
trunk pitch. Band Pull Apart is the first whose defining excursion is **frontal**: the hands
travel apart in the image plane, and nothing about the movement is legible from the side.

That single fact drives most of this document. It inverts which views work (§4.7), it means the
*rep signal itself* is view-bound rather than merely the rules, and it creates a facing-sign
problem for the one sagittal rule in the set (§4.8) that no previously shipped movement had to
solve, because no previously shipped movement mixed planes this way.

---

## 2. There is no labeled ground truth for Band Pull Apart, and `validated` stays `False`

No dataset in this repository carries band-pull-apart video, labeled or otherwise:

- **REHAB24-6** ships six exercises; none is a band pull apart. (`Ex5` — the forward lunge — is
  the only labeled-correctness source the project has ever used for detector validation, in
  `notes/lunge-rule-validation.md`.)
- **Fit3D does contain the movement** — `band pull apart` is one of its ~30 named exercises
  (`docs/movement-kg-expansion-plan.md:33,48`, which lists Band Pull Apart as Fit3D-only,
  1 dataset, General tier) — and it ships 3D ground truth and rep boundaries. But it ships **no
  fault labels**, which is precisely why the KG stub for this movement reads
  `"textbook (Fit3D, no fault labels)"` (`scripts/knowledge/stub_general_movements_v3.py:81`).
  Footage without correct/incorrect labels cannot support a REHAB24-6-style check — the same
  constraint Row's design doc §2 recorded for its own Fit3D row video.
- **The Squat dataset** is squat-only.

So this detector ships **spec-derived and UNVALIDATED**, surfaced with a Beta tag through
`/api/movements`'s `validated` field, exactly like Squat / OHP / Push-up / Deadlift / Row.

**The standing consequence: no threshold tuning.** Every cited number ships as the parent spec
states it. Where a rule is expected to behave poorly, that expectation is written down (§4.5,
§4.8) rather than repaired by moving a number, because with no ground truth a moved number is
not a fix — it is an unmeasurable preference.

---

## 3. `loss_of_scapular_retraction` cannot be implemented, and ships permanently silent

The parent spec's third Band Pull Apart rule (lines 749–756) prescribes:

> from a REAR/rear_oblique view track inter-shoulder width `dist(11,12)`; genuine retraction
> slightly narrows the posterior shoulder points as scapulae adduct […] Flag `no_retraction` if
> wrist spread increases > threshold while `dist(11,12)` change < 0.01

Two independent defects, either of which is disqualifying.

**(a) The fire condition is a null-detection.** It fires when `dist(11,12)` *fails to change*.
A steady frame, a partially occluded frame, and a frame where the lifter genuinely does not
retract are indistinguishable to it — all three satisfy "change < 0.01". Every correctly
performed rep that happens to hold the shoulders stable fires the fault. A rule whose positive
class is "nothing measurable happened" cannot separate the fault from the absence of evidence.

**(b) The measured quantity is confounded with the thing it must be independent of.**
MediaPipe's shoulder landmark is a **glenohumeral** point, not a scapular border point. It moves
with the humerus. During horizontal abduction the humerus is exactly what is moving, so
`dist(11,12)` changes for reasons that have nothing to do with scapular adduction, and the
metric cannot attribute an observed narrowing to retraction rather than to arm position. This is
the same root cause as `pushup.rule_scapular_winging` and `row.rounded_thoracolumbar_spine`:
**MediaPipe Pose has no scapular landmarks**, so no construction over its 33 points measures
scapular position.

Separately, the `0.01` figure carries no citation. Fukunaga (PMC8975561) backs *retraction as the
training mechanism* and supplies no landmark-displacement magnitude in any units.

### 3.1 Silent, not withdrawn — and the distinction is load-bearing

This project has two established treatments for a rule it will not fire, and they say different
things:

| Treatment | Says | Precedent |
|---|---|---|
| **Registered, permanently silent** | "Real, well-cited fault; the sensor cannot see it." | `pushup.rule_scapular_winging`, `row` rule 5 |
| **Withdrawn from the parent spec** | "No citation supports the rule as written." | OHP bar-path (2026-07-25), Deadlift bar-drift (2026-08-01) |

`loss_of_scapular_retraction` takes the **silent** treatment. Fukunaga genuinely does establish
that middle-trapezius recruitment is driven by the retraction-oriented directions, so the fault
is real and cited; what fails is the *sensing*, not the literature. The parent spec is annotated
with a `NOTE`, not a `WITHDRAWN` blockquote.

Concretely: `rule_loss_of_scapular_retraction` exists in `BAND_PULL_APART_DETECTOR.rules`,
always returns `[]`, and carries the argument above in its docstring. Registering it costs one
no-op call per clip and keeps spec↔code in 1:1 correspondence at four rules, so an auditor
comparing the two documents gets "accounted for, and here is why it says nothing" rather than
silence.

**Deliberately not substituted.** Two signals do carry *some* retraction information — the
scapular contour from a rear view, and shoulder-to-spine distance — and neither is recoverable
from 33 landmarks. Shipping a different metric under this `fault_id` would attach Fukunaga's
citation to a quantity Fukunaga says nothing about.

**The KG is not the gap.** `Band Pull Apart:Insufficient Scapular Retraction` resolves with a
non-empty `causes` bucket (`Limited Scapular Retraction`) — verified, §5. The metric is the gap.

---

## 4. Detector design

New module `src/pose/movements/band_pull_apart.py`, following `row.py` exactly: threshold-free
raw metrics → phase assignment → cited rule functions → an assembled `MovementDetector`
registered by side-effect import. `src/pose/movements/base.py` is **not modified**;
`run_detector` already does segmentation, global smoothing, per-rep slicing and merging.

Registry name: **`"Band Pull Apart"`**, matching `frontend/src/lib/movements.ts:14`, the i18n key
`movement.Band Pull Apart` (`彈力帶擴胸`), and the KG `Action` node of the same name. `get_detector`
keys on `name.lower()`, so a spelling drift here makes the movement unselectable in the studio;
the name is pinned by a test.

### 4.1 Rep segmentation

`rep_signal = "wrist_spread_shoulder_norm"`, `rep_polarity = "max"`, all other knobs default.

This is not a fresh choice. `docs/superpowers/specs/2026-07-26-rep-segmentation-sp1-design.md`
§3.4 audited all sixteen movements against the segmentation interface and assigned Band Pull
Apart the signal `雙手間距 / 肩寬` with polarity `max`, in the "clean unipolar excursion, all
defaults" group of eleven. The rep runs hands-together → maximally spread → hands-together, which
is a textbook single-peaked excursion for `segment_reps`'s hysteresis.

**Normalized, not raw, and this matters more here than elsewhere.** `segment_reps` thresholds at
0.35 / 0.65 of the signal's *dynamic range*, so a monotone rescaling of the signal cannot change
where reps are cut — raw wrist spread would segment identically. Shoulder-width normalization is
chosen anyway because rule 2 fires on the same key against an absolute threshold (`1.6`), and
having one key serve both means the number a rule tests is the number a reader sees in
`metric_keys`.

**The failure mode to test for, not assume away.** SP1 §3.4 closes by stating outright that its
table is *interface-design inference, not verified fact*, and that each movement must confirm
segmentation on real footage. There is no real footage (§2). The mitigation is therefore a
**synthetic three-rep clip driven end-to-end through `run_detector`**, asserting three reps are
found — not merely unit tests on the rule functions, which would pass happily while segmentation
returned zero reps and every rule silently ran on the whole-clip fallback. This is the same class
of silent-zero failure as the SP1 live-record bug (`be85d1fd`).

**A second gap the synthetic clip cannot rule out: whether the segmentation floor itself is
compatible with this movement's cadence.** `rep_segmentation.py`'s `DEFAULT_MIN_REP_SECONDS = 0.4`
discards any candidate rep shorter than 0.4s (12 frames at 30fps) as duration-anomaly noise —
"the segmentation itself doing exactly what `rep_segmentation.py` documents it doing", per SP1
§3.4. `BAND_PULL_APART_DETECTOR` takes this default unchanged, on the strength of SP1's "clean
unipolar excursion, all defaults" placement, which §3.4 itself calls interface-design inference
rather than verified fact.

**This is cheaply checkable, and was checked, because §2 already names the source.** §2
establishes that — unlike REHAB24-6 or the Squat dataset — Fit3D **does** carry band-pull-apart
footage with rep boundaries (`rep_ann.json`); what it lacks is fault labels, which is the reason
`validated` stays `False`, not a reason to skip a cadence check that needs no labels at all. Rep
boundaries are exactly the input a cadence question needs. Pulled from
`data/Fit3D/train/{s03,s04,s05,s07,s08}/rep_ann.json` — the five subjects with a
`band_pull_apart` key — each file's boundary list is six frame indices bounding five reps (25
reps total). Every `band_pull_apart.mp4` in this repo runs at **50fps**, confirmed directly with
`ffprobe` (`r_frame_rate=50/1`) rather than assumed — Fit3D is not the 30fps this floor's frame
arithmetic uses elsewhere. Converting each subject's boundary-frame deltas to seconds at 50fps:

| subject | rep durations (s) |
|---|---|
| s03 | 2.36, 1.78, 2.10, 1.80, 2.02 |
| s04 | 2.34, 2.18, 2.30, 2.30, 2.86 |
| s05 | 2.58, 2.18, 2.66, 2.48, 2.28 |
| s07 | 2.64, 2.30, 2.40, 2.38, 2.92 |
| s08 | 2.58, 2.08, 2.60, 2.20, 2.28 |

Across all 25 measured reps the range is **1.78s–2.92s**. The fastest rep measured (s03, 1.78s)
is still 4.45× the 0.4s floor; the slowest (s07, 2.92s) is 7.3×. Five independent subjects, none
within a factor of four of tripping the floor.

**Directional evidence, not a validation — and it should not be read as more than that.** Five
subjects is not a cadence distribution, and Fit3D subjects perform a coached, deliberate protocol
for mocap capture, which is not the same population as a real app user rushing through a set with
imperfect form. This measures rep *duration* from mocap-derived rep boundaries, not this
project's own `segment_reps` running on this project's own pose extraction. What it establishes:
there is no evidence, from the one real source available, that the 0.4s floor is likely to bite
typical band-pull-apart execution — replacing what was previously an unsupported claim that no
such evidence exists at all. What it does **not** establish: that a rushed real user cannot
produce a sub-0.4s rep, or that Fit3D's coached cadence is representative of this app's users.
The residual risk is bounded, not eliminated.

The end-to-end fixture above is built at 18 frames/rep (0.6s at 30fps, comfortably above the
floor and consistent with the measured Fit3D range) — by construction it cannot exercise a
floor problem, because building it any faster fails for a reason that has nothing to do with
wrist-spread segmentation: at 9 frames/rep (0.3s) the same fixture's excursions are still found
on their correct boundaries, and `_finalize` discards all three anyway, purely because `9 < 12`.
That is the floor working as designed, not a segmentation defect — but it is also a preview of
what happens to a genuinely fast real clip. The symptom, if a real clip ever is that fast: every
rep in it falls under 12 frames, `segment_reps` returns `[]`, and `run_detector` falls back to
scoring the whole clip as one window instead of per-rep — silently, with no error and no log line
distinguishing it from a clean single-rep clip. This is the same class of gap SP1 §3.4 already
measured for High Knee (~3Hz alternating-leg cadence, ~10 frames/rep at 30fps, below the same
12-frame floor, which is why High Knee ships with segmentation disabled rather than a guessed
override) — the difference being that here there is now a measurement, and it points the other
way. Not fixed here regardless: overriding `min_rep_seconds` on `BAND_PULL_APART_DETECTOR` would
be real threshold tuning against a five-subject, coached-cadence sample with no cited production
cadence behind it, and would silently weaken the anomaly floor for every real clip on the strength
of a directional signal — the same fabrication this project's threshold rules forbid elsewhere.
Flagged and now measured, not patched; recorded in `TODO.md`.

### 4.2 Phases

`band_pull_apart_assign_phases` labels a rep window `setup → pull → peak → return`, mirroring
`row_assign_phases`: the first 15% of the window is `setup` (subject to frame validity), the peak
of `wrist_spread_shoulder_norm` anchors `peak`, and the frames either side are `pull` and
`return`. Rules gate on the phases the parent spec names — both baseline-comparing rules read
`setup` and fire on `peak`, per the spec's own wording ("at setup baseline and at peak").

`assign_phases` must return exactly one label per input frame; `run_detector` raises loudly
otherwise (`base.py:174`).

### 4.3 The setup baseline

Two of the three firing rules compare against "setup", which is a **per-rep reduction**.
`_setup_baseline` is copied from `row.py:403` verbatim in behavior: the **median** of a key over
this window's valid `setup` frames, `NaN` when there are none. Median rather than mean so one bad
frame in a short setup slice cannot move the reference.

Its NaN policy is inherited unchanged, and the two branches are different on purpose:

- A rule whose fire condition depends **only** on the baseline (rule 1, rule 4) has nothing left
  to evaluate and returns `[]`.
- A rule with a **disjunctive** non-baseline term (rule 2's spread-ratio term) must not
  early-return; it drops only the baseline-dependent term, which happens for free because
  `nan > threshold` is `False`.

Row's measured limitation carries over and is restated rather than rediscovered: on a short rep
the 15% `setup` slice can be 1–2 frames and may already overlap loaded frames, biasing the
baseline toward the loaded state. Because every comparison is `peak − baseline`, that bias makes
the rules **under**-fire. Accepted, not repaired — repairing it needs a second threshold the
parent spec does not supply.

### 4.4 Metrics

`band_pull_apart_compute_raw(frames, fps) -> list[dict]`. **The metric layer contains no
thresholds.** The only constant it defines is `_DEGENERATE_LENGTH = 1e-6`, a division-by-zero
guard, matching `pushup.py` / `overhead_press.py` / `lunge.py` / `row.py`.

`BAND_PULL_APART_METRIC_KEYS`:

| key | meaning | consumed by |
|---|---|---|
| `wrist_spread` | `dist(15, 16)`, raw image units | diagnostic |
| `shoulder_width` | `dist(11, 12)` | normalizer |
| `wrist_spread_shoulder_norm` | `wrist_spread / shoulder_width` | rep signal, rule 2 |
| `left_shoulder_ear_gap` | `y(11) − y(7)` | rule 1 |
| `right_shoulder_ear_gap` | `y(12) − y(8)` | rule 1 |
| `shoulder_ear_gap_shoulder_norm` | mean gap / `shoulder_width` | scale-free diagnostic (§4.5) |
| `left_elbow_angle` | `angle(11, 13, 15)` | rule 2 |
| `right_elbow_angle` | `angle(12, 14, 16)` | rule 2 |
| `min_elbow_angle` | worse of the two | rule 2 |
| `trunk_lean_signed_deg` | signed torso pitch from vertical, facing-corrected (§4.8) | rule 4 |
| `trunk_angle_speed_deg_s` | derivative of the above | rule 4 |
| `wrist_depth_offset` | mean wrist `z` − mean shoulder `z` | facing derivation (§4.8) |

`BAND_PULL_APART_METRIC_KEYS` must stay a two-way match with what `compute_raw` emits — a key the
tuple omits is silently dropped by `run_detector` (which builds each `CoreFrame.metrics` *from*
this tuple), and a key the tuple names but `compute_raw` never emits reads as `NaN` forever.
Pinned by a test, mirroring `test_lunge_metric_keys_match_the_emitted_metrics`.

**Frame validity.** `required` lists both shoulders, both ears, both elbows, both wrists and both
hips. If `visible_point` drops any one, the frame is `valid=False` and carries no metrics, so
**every** rule goes silent for that frame — not only the one whose input vanished. This is the
house convention (`pushup`, `ohp`, `lunge`, `row` all do it): an unmeasurable frame is refused
wholesale rather than degraded, because a silently-wrong verdict is worse than no verdict.

`trunk_angle_speed_deg_s` is a **central difference computed inside `compute_raw` from raw
per-frame positions**, first and last frames `NaN` (never one-sided). Reason, taken from
`row.py` §4.6: `run_detector` applies `centered_median(window=5)` to every key in `metric_keys`,
and a 5-frame median over a *position* series would flatten the velocity transient before rule 4
ever saw it. Differentiating first means the framework's filter low-passes the derivative, which
is a defensible smoothing of the quantity of interest rather than an erasure of it.

### 4.5 Threshold units: one is normalized, one is raw, and the difference is real

The parent spec normalizes one Band Pull Apart threshold explicitly and leaves the other bare:

- **`1.6` is explicitly shoulder-width-normalized** — "wrist_spread […] normalized by shoulder
  width `dist(11,12)`". Scale-free. (Row's design doc §4.5 cites this very line as internal
  evidence that the parent spec says "normalized by" when it means it.)
- **`0.03` (shoulder-ear gap) carries no normalizer**, so by that same internal evidence it is
  **raw MediaPipe image units** — and therefore **camera-distance dependent**. The identical
  shrug filmed from further away yields a smaller gap change and fires less.

Implemented as written, with the same mitigation Row used: `shoulder_ear_gap_shoulder_norm` is
emitted alongside as a **scale-free diagnostic that no rule fires on**, so a future validation can
compare the scale-dependent threshold against a scale-free one without any threshold having been
moved in the interim.

`10°` (rule 4) and `150°` (rule 2's elbow term) are angles — scale-free by construction.

### 4.6 Rules

Every threshold is labeled in-code as exactly one of **`FROM THE SPEC`** or **`RULE-LEVEL CHOICE
MADE HERE`**, in the style of `pushup.py`. The two are never blurred.

Severity ramps are all **RULE-LEVEL**: the parent spec states no ramp for any Band Pull Apart
fault. Convention from `pushup.rule_hip_sag` — ramp endpoint = 2.5× the fire threshold,
documented as a display/ranking curve rather than a cited quantity. The elbow ramp is the one
exception and is pinned to `pushup.rule_shallow_depth`'s so the two elbow ramps cannot drift.

**Ramp direction is expressed through `severity_from_range`'s flag, never by swapping the
endpoints.** Its signature is `severity_from_range(value, mild, severe, *, lower_is_worse)`
(`geometry.py:165`), so the two descending ramps here — rule 2's spread `1.6 → 1.0` and elbow
`150° → 110°` — pass `mild=1.6, severe=1.0, lower_is_worse=True`, matching
`pushup.rule_shallow_depth`'s call form. Rule 1's ascending `0.03 → 0.075` and rule 4's
`10° → 25°` pass `lower_is_worse=False`. Reversing `mild`/`severe` to fake a direction would
silently invert severity, so the tests assert the endpoint values map to severity 0 and 1 in the
intended direction.

#### Rule 1 — `bpa_shrugging`

| | |
|---|---|
| **Fires when** | shoulder-ear gap at `peak` closes by `> 0.03` vs this rep's `setup` baseline, on **either** side |
| **Threshold** | `0.03` — **FROM THE SPEC** ("flag shrug if `gap_peak < gap_setup - 0.03`") |
| **Ramp** | `0.03 → 0.075` — **RULE-LEVEL** (2.5×) |
| **Phase scope** | `peak` |
| **Observability** | `high` on `rear`; `high` on `rear_oblique` (§4.7 — the metric is facing-free) |
| **KG query** | `"Shoulder Shrugging"` → `Band Pull Apart:Shoulder Shrugging` ✅ non-empty (§5) |
| **Citations** | Fukunaga PMC8975561; Camargo & Neumann PMC6849087 — copied verbatim from the parent spec at implementation time, never recalled from memory |

Either-side rather than mean: a unilateral shrug is the common presentation, and averaging the
two gaps would halve it toward the threshold.

#### Rule 2 — `bpa_incomplete_rom`

| | |
|---|---|
| **Fires when** | peak `wrist_spread_shoulder_norm < 1.6` **OR** `min_elbow_angle < 150°` at `peak` |
| **Thresholds** | `1.6` and `150°` — both **FROM THE SPEC**, the second with a direction correction (§4.9) |
| **Ramps** | spread `1.6 → 1.0`, elbow `150° → 110°` — **RULE-LEVEL**; the elbow ramp width matches `pushup.rule_shallow_depth`'s |
| **Phase scope** | `peak` |
| **Observability** | `high` on `rear`; `medium` on `rear_oblique` (spread foreshortens off-axis) |
| **KG query** | `"Bent Elbows"` → resolves, but **connectivity 0** — an empty card (§5) |
| **Citation** | Fukunaga PMC8975561 |

A genuine disjunction, unlike Row's momentum rule whose second condition was a strict subset of
its first. These two cues are independent failure modes: a lifter can reach full spread with bent
elbows (short lever cheat), or keep arms straight and stop short. The evidence payload records
which term fired.

#### Rule 3 — `bpa_loss_of_scapular_retraction`

Registered, **permanently silent**, always returns `[]`. Full argument in §3; the docstring
carries it. No threshold, no KG query, no citation attached to a metric.

#### Rule 4 — `bpa_trunk_extension_compensation`

| | |
|---|---|
| **Fires when** | backward trunk lean exceeds `10°` beyond this rep's `setup` baseline |
| **Threshold** | `10°` — **FROM THE SPEC** ("Flag if `trunk_lean_backward > 10deg` beyond setup baseline") |
| **Ramp** | `10° → 25°` — **RULE-LEVEL** (2.5×) |
| **Phase scope** | `pull` and `peak` (the spec's "synchronized with the pull") |
| **View** | **hard-gated**, negative form: silent on a confident `rear` label and on `unknown` (§4.7) |
| **Facing** | derived per clip from wrist depth (§4.8) |
| **Observability** | `medium` — downgraded from the spec's `high` because the facing derivation is an unvalidated precondition |
| **KG query** | `"No Compensatory Trunk Movement"` — resolves as a shared node but **bare** (§5) |
| **Citation** | Fukunaga PMC8975561, whose harm claim the parent spec itself flags as partly inferential — restated in the docstring, not quietly upgraded |

The spec's second cue ("or a trunk-angle velocity spike co-occurs with the concentric") is
implemented as **evidence, not as a fire condition**: when the rule fires,
`trunk_angle_speed_deg_s` over the same frames is recorded as `evidence["trunk_whip"]`, which
distinguishes a slow lean from a whip for the coaching cue without changing what fires. This
follows Row's treatment of its own co-occurrence clause.

### 4.7 View handling — and why rule 4 gates where every Row rule downgrades

**The reachability facts, verified in `src/pose/view_estimation.py`, not assumed:**

1. Production calls `estimate_view_for_pose(allow_front=False)`, so **`front` and `front_oblique`
   are never emitted** (`view_estimation.py:14–16`, `:365–370`). Reachable labels are exactly
   `{side, rear, rear_oblique, unknown}`. **A rule gated positively on a whitelist containing
   unreachable labels is a latent silence bug**, which is why the gate in §4.7 below is written
   in the negative.
2. On that path, a **genuinely front-facing subject is relabeled `rear_oblique`**
   (`view_estimation.py:368–370`: the `front_score >= rear_score` tie takes the non-`allow_front`
   branch and unconditionally assigns `rear_oblique`).
3. Measured corpus, recorded in the Lunge design doc: across this repo's 45 real pose JSONs the
   estimator emitted 30 `rear_oblique`, 13 `rear`, 2 `unknown`, and `side` effectively never.

**Consequence for segmentation — favorable.** `rear_oblique` is the modal label and wrist spread
survives there, foreshortened but present, so reps segment on the view production actually
produces. A pure `side` view would collapse the signal entirely and yield zero reps, but `side`
is effectively never emitted. The frontal rep signal is safe in practice.

**Consequence for rules 1 and 2.** Both follow the house **downgrade, never gate** convention:

- Rule 1's metric is a **vertical image-y difference**, a magnitude that reads identically from in
  front of or behind the subject. It is facing-free by construction, exactly as
  `row_asymmetric_pull` and `lunge.rule_knee_valgus` argue for theirs, so `rear` and
  `rear_oblique` both earn the spec's `high` rating with no discount.
- Rule 2's spread foreshortens off-axis, so `rear_oblique` takes `high → medium` and confidence
  ×`VIEW_UNAVAILABLE_CONFIDENCE_SCALE`, **imported** from `pose_rule_detector` rather than
  re-typed so a change to the shared constant cannot silently skip this module.

**Consequence for rule 4 — a hard gate, departing from the Row convention.** Rule 4 measures a
*sagittal* quantity. From a pure `rear` view the sagittal axis is perpendicular to the image
plane, so a signed torso lean computed there reads **lateral sway in the frontal plane**, which is
a different fault (or no fault). Firing it on `rear` would not be a low-confidence reading of the
right quantity — the case the ×0.65 discount exists for — it would be a confident reading of the
wrong plane. Row's §4.7 argues against gating on the grounds that gated rules ship silent; that
argument does not apply here, because the view the gate leaves standing is `rear_oblique`, the
modal label.

**Precedent, and the form the gate takes.** `pushup.rule_elbow_flare` is the shipped example of a
hard-gated rule: it is *gated to silence on a confident `side` label*, emitted at `low`
observability with the 0.65 discount otherwise, and `run_detector`'s sort key
(`observability == "low"`, …) keeps it behind any fault seen from a validated view. Note the
shape — the gate is **negative** (silence on a named bad view), not a positive whitelist.

Rule 4 follows that form: **silent on a confident `rear` label, and silent on `unknown`;
otherwise it fires.** Written negatively rather than as a `{side, rear_oblique, front_oblique}`
whitelist for two reasons. First, a whitelist containing `front_oblique` — unreachable under
`allow_front=False` — is dead weight that reads as coverage; the negative form needs no edit
whatsoever if `allow_front` is ever enabled, and admits `front`/`front_oblique` automatically and
correctly. Second, it fails in the safer direction: an unanticipated future label is scored rather
than silently dropped. `unknown` is named explicitly because it means *the view estimator failed*,
not *a confirmed view* — Row's rule 4 discussion draws exactly this distinction.

### 4.8 Rule 4's facing problem, and how it is solved

Fact (2) above has a sharper consequence than the gate. Because `rear_oblique` conflates a true
rear-oblique with a true front-oblique, **the view label cannot tell you which way the lifter
faces**. The sign of a sagittal offset therefore cannot be recovered from `ctx.view_type`, and a
backward lean is indistinguishable from a forward one — which is fatal for a rule whose entire
content is *backward*.

`overhead_press.py:131–139` handles this by **assuming** a facing ("posterior is +x") and
documenting that a subject facing the other way inverts every sagittal reading in the module.
That is a coin flip per clip, and on the losing side rule 4 would confidently report the opposite
fault. It is not adopted here.

**Two rejected alternatives, for the record.** Firing on the *magnitude* of trunk-pitch change
would make the rule sign-free, but it would then fire on forward lean too — relabeling a
different quantity under this rule's `fault_id`, which is exactly the defect that killed Row's
`rounded_thoracolumbar_spine` construction 2. Making rule 4 silent as well would leave two firing
rules out of four, and the trunk whip is the most visible fault in the set.

**Adopted: derive facing per clip from wrist depth.** A band pull apart holds the band *in front
of the torso* by definition — this is what the movement is, from setup through peak. So the sign
of `wrist_depth_offset = mean(z(15), z(16)) − mean(z(11), z(12))` tells you which way the lifter
faces: MediaPipe `z` is negative toward the camera, so wrists nearer the camera than the shoulders
means the lifter faces the camera.

Why this is defensible where a metric depth claim would not be:

- It is a **binary, large-margin** decision, not a metric-depth measurement. Extended arms put the
  wrists tens of centimetres anterior to the shoulders; the sign of that offset does not require
  the depth accuracy this project's Fit3D line found MediaPipe to lack. The depth-bottleneck
  findings are about *magnitudes* of depth-derived cues, not about the sign of a large separation.
- It is a **measurement precondition, not a fault threshold** — it decides which direction counts
  as backward, and it is never compared against a cited number. Nothing in Fukunaga is being
  stretched to cover it.
- It is a **geometric fact about this specific movement**, not a general-purpose facing detector.

Implementation: the facing sign is a **per-clip** reduction (the median of `wrist_depth_offset`
over valid `peak` frames, where the arms are most extended and the margin is largest), not
per-frame, so per-frame `z` jitter cannot flip the sign mid-rep. When the median is not finite, or
its magnitude sits under a degeneracy floor — a clip where the wrists are not resolvably in front
of the torso, which is not a band pull apart — the facing is **undetermined and rule 4 returns
`[]`**. This is the "can only ever SILENCE" guard category `pushup.py` documents.

**Stated limitation:** this derivation is unvalidated on band pull apart footage specifically,
like every other number here. It is reflected in rule 4's observability being **`medium`** rather
than the spec's `high` — the fault is highly visible to a human, but this detector's reading of it
rests on a precondition no band-pull-apart clip has confirmed, and the observability field should
say so.

#### 4.8.1 The `z` plumbing is measured, not assumed, and the degeneracy floor comes from it

The derivation above is worthless if `z` is absent or constant in the payloads production
actually produces. Both writers and the 49 real pose JSONs under `data/runtime/pose_json/` were
checked while writing this spec.

**Writers.** The offline MediaPipe path (`src/pose/process_videos.py:85,97`) writes `lm.z`. The
browser capture path (`frontend/src/lib/poseExtract.ts:40,43`) writes real `z` and **rejects any
frame whose `z` is non-finite**, so on that path `z` is present by construction. But
`src/pose/rtmpose_pose_extraction.py:121,131` writes **`"z": 0.0` for every landmark** — a 2-D
backbone with no depth to report. A z-degenerate runtime therefore exists and is not
hypothetical.

**Measured, on the 49 runtime clips** (35 carry usable landmarks; 6 are distinct, the rest are
re-uploads of the same clip):

| distinct clip | frames | median wrist−shoulder z offset | sign stability |
|---|---|---|---|
| 1 | 111 | −1.3958 | 100.0% |
| 2 | 79 | −0.5070 | 96.2% |
| 3 | 61 | −0.4876 | 98.4% |
| 4 | 2304 | −0.2486 | 99.9% |
| 5 | 30 | −0.1688 | 96.7% |
| 6 | 43 | +0.1295 | 81.4% |
| **3 further clips** | — | **exactly 0.0000** (z identically zero) | — |

Three conclusions, each load-bearing:

1. **`z` is real and non-degenerate on the production path.** The offset is not noise around zero;
   its magnitude runs 0.13–1.40.
2. **The two populations separate cleanly with no overlap.** Non-degenerate clips floor at
   `0.1295`; degenerate clips sit at exactly `0.0`. The degeneracy floor is therefore set at
   **`0.02`** — roughly 6× below the smallest observed real value and far above zero. It is a
   **plumbing test that distinguishes "this runtime reports depth" from "this runtime reports
   zeros"**, not a tuned fault threshold, and the measurement above is why it is that number
   rather than a guess.
3. **Sign stability is 81–100% on clips whose arms are *not* held forward.** These are squats and
   push-ups — the worst case for this cue, since the wrists spend much of the clip beside or under
   the torso. A band pull apart holds the arms extended anteriorly throughout by definition, so
   its margin should exceed every row above. Taking the median over `peak` frames (§4.8), where
   extension is greatest, is what converts even the 81.4% worst case into a stable per-clip sign.

The RTMPose path consequently gets rule 4 silent, automatically and correctly, with no
runtime-specific branch anywhere in this module.

### 4.9 The parent spec's elbow cue has its direction inverted

Parent spec line 739 reads:

> and/or elbow-extension check `elbow_angle > ~150deg` maintained (bent-elbow curl-style cheat =
> fault)

Read literally, `elbow_angle > 150°` — nearly straight arms — is the fault, and the parenthetical
says the opposite in the same sentence. The parenthetical is right and the inequality is a slip:
a bent-elbow cheat means a *smaller* elbow angle. **Implemented as `min_elbow_angle < 150°`.**

Corroboration rather than inference alone: the KG's Band Pull Apart stub names this fault
`Bent Elbows` (`stub_general_movements_v3.py:85`), and Fukunaga's rationale — more range covered
against the band drives higher activation — is a range argument that bending the elbows shortens.

`150°` itself is unchanged and stays **FROM THE SPEC**; only the comparison direction is
corrected. The correction is annotated in the parent spec (§7) and in the rule's docstring, so it
cannot be silently re-flipped later by someone reading line 739 alone.

---

## 5. KG queries — resolved before being written, not after

Every query below was executed against `data/kg/sports_kg_v3.graphml` via
`retrieve_graph_context(query, movement="Band Pull Apart")` **while writing this spec**. Results
are observed, not predicted.

The movement scopes to exactly three `Fault` nodes:

| node | connectivity |
|---|---|
| `Band Pull Apart:Shoulder Shrugging` | 1 |
| `Band Pull Apart:Insufficient Scapular Retraction` | 1 |
| `Band Pull Apart:Bent Elbows` | **0** |

| Rule | Query | Resolves to | Buckets returned |
|---|---|---|---|
| 1 `bpa_shrugging` | `"Shoulder Shrugging"` | `Band Pull Apart:Shoulder Shrugging` | `causes`: Weak Scapular Stabilizers · `quality_impacts`: Shoulder Depression ✅ |
| 2 `bpa_incomplete_rom` | `"Bent Elbows"` | `Band Pull Apart:Bent Elbows` | **none** — only the `HAS_FAULT` backlink ⚠️ |
| 3 (silent) | — | — | — |
| 4 `bpa_trunk_extension_compensation` | `"No Compensatory Trunk Movement"` | shared `QualityDimension` | **none** ⚠️ |

**Two honest gaps, recorded rather than masked.**

Rule 2's node names the fault correctly but has no causes, risks or corrections, so its FaultCard
will be thin. The alternative — pointing the query at the shared `Range Of Motion`
`QualityDimension`, which returns a rich bucket set — was **rejected**: its `corrections` bucket
is `Wrapping Surface Adjustment`, which is meaningless for this movement, and swapping in a
generic node to make a card look full would hide the gap behind plausible-looking content. A
semantically correct thin card is preferable to a semantically wrong rich one.

Rule 4 has **no** Band-Pull-Apart-scoped node at all; `Trunk Extension` and `Loss Of Neutral Body
Position` (Row's queries) do not resolve under this movement's scoping, and the two shared
candidates that do resolve are bare.

Both gaps are one-line fixes in `scripts/knowledge/stub_general_movements_v3.py:80–87` — adding
correction/risk targets to `Bent Elbows` and a trunk-compensation fault to the stub. They are
**out of scope here** (§8) because the graphml is gitignored and regenerating it is a deploy step,
not a code change. They are logged against TODO.md's existing item
「許多錯誤沒有對應到 Knowledge Graph 的節點」.

`retrieval_mode="kg"` for all firing rules, consistent with every other detector.

---

## 6. Testing

`tests/test_band_pull_apart.py`, `unittest.TestCase`, mirroring `tests/test_row.py`'s discipline.

**Fixture builder.** A `bpa_frame(...)` helper whose knobs each control **exactly one metric by
construction** — the property that makes a boundary fixture genuinely sit one step either side of
a threshold. Knobs: `wrist_spread_ratio`, `shoulder_ear_gap` (per side), `elbow_angle_deg`,
`trunk_lean_deg`, `wrist_depth_offset`, `visibility`. The exact-angle elbow placement helper
(`_elbow_xy`, chord/perpendicular-bisector construction) is reused from `test_row.py` so a
requested elbow angle *is* the measured one.

Coverage:

1. **Metric layer** — each key equals its knob; `NaN` propagation on a dropped landmark; the
   whole-frame invalidation rule; central-difference edges are `NaN`.
2. **`BAND_PULL_APART_METRIC_KEYS` two-way match** with what `compute_raw` emits.
3. **Phase assignment** — one label per frame; the `setup`/`peak` anchors; the length-mismatch
   guard in `run_detector` is not tripped.
4. **End-to-end segmentation** — a synthetic **three-rep** clip through `run_detector` asserting
   three reps found, `fallback is None`, and rules scored per rep. This is the §4.1 silent-zero
   guard and is not optional.
5. **Each rule at its boundary** — one fixture just inside and one just outside every threshold,
   including rule 2's **corrected** `< 150°` direction (a test that would fail under the spec's
   literal `>`).
6. **Rule 3 is silent** — asserts `[]` for inputs that would otherwise look like the fault, so a
   future edit cannot un-silence it unnoticed.
7. **Rule 4's gate and facing** — silent on a confident `rear` label and on `unknown`; fires on
   `rear_oblique`; the verdict inverts with the sign of `wrist_depth_offset`; silent when the
   facing median is `NaN` or under the `0.02` floor, including the **all-zero-`z` case** that the
   RTMPose writer produces (§4.8.1), which gets its own fixture.
8. **Setup-baseline NaN policy** — rules 1 and 4 return `[]` on a `NaN` baseline; rule 2 drops
   only its baseline term and still fires on the spread term.
9. **Registry** — `get_detector("band pull apart")` resolves and the name matches
   `frontend/src/lib/movements.ts` exactly.

**One existing test must be rotated.** `tests/test_analyze_pose_service.py:118` asserts
`"Band Pull Apart"` is *not* registered, and its own comment says to move the example onward when
the movement lands (it has already moved Deadlift → Row → Band Pull Apart). It moves to a
still-unimplemented movement — `Bicep Curl`, the next in spec order.

**Commands** (repo root, always scoped to `tests/`):

```
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```

Two backend flakes are known-unrelated on this machine; check against a baseline before
attributing a failure to this change.

---

## 7. Honesty constraints

- **Citations are copied verbatim from the parent spec at implementation time, never recalled
  from memory.** This is the anti-hallucination guarantee the 16-movement programme rests on.
- **No threshold tuning.** With no ground truth (§2), a moved number is an unmeasurable
  preference. Weak behavior is written up.
- `validated = False`.
- **The parent spec gets edited, and that is part of the deliverable, not optional.** Two
  annotations to `2026-07-18-16-movement-rule-detector-design.md` §Band Pull Apart:
  - a `NOTE` on `loss_of_scapular_retraction` recording the null-detection and confounded-landmark
    argument and its permanently-silent status (§3) — a `NOTE`, not a `WITHDRAWN` blockquote,
    because the failure is sensing, not citation;
  - a `NOTE` on `incomplete_horizontal_abduction_rom` recording the inverted inequality (§4.9).
- Rule 4's harm claim is **partly inferential** — the parent spec says so itself, and Fukunaga
  even notes trunk extension can be deliberately engaged. The docstring restates this; it is not
  quietly upgraded to a firm injury claim.
- Rule 4's facing derivation is an **unvalidated precondition**, reflected in its `medium`
  observability.
- The two KG gaps (§5) are recorded, not papered over with a semantically wrong node.

---

## 8. Out of scope

- **Any frontend change.** `/api/movements` is registry-derived, `movements.ts` already lists the
  movement, the i18n key and card art already exist. Adding the detector flips it from "Soon" to
  analyzable with no frontend edit.
- **Any `base.py` / `rep_segmentation.py` change.** All knobs this movement needs already exist.
- **KG enrichment** for the two gaps in §5 — one-line stub edits, but the graphml is gitignored
  and regenerating is a deploy step.
- **Validation against labeled data.** None exists (§2).
- **Enabling `allow_front`** in view estimation. It would improve rules 2 and 4 materially, but it
  changes view labels for *every* movement and every shipped detector's tuning, so it is its own
  piece of work.
- **The remaining nine movements.** Bicep Curl (Group D) is next in spec order.
