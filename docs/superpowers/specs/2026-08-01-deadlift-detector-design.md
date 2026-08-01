# Deadlift Rule Detector — Design

**Status:** design spec · **Date:** 2026-08-01 · **Movement:** Deadlift (5th of 16)
**Parent spec:** `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` §Deadlift (lines 213–263)
**Branch:** `feat/deadlift-detector` (worktree, based on `origin/main` @ a89d46ed)

---

## 1. Why Deadlift, and what this pass is not

Deadlift is next in the parent spec's own order — Group A is Squat → Lunge → Deadlift, and
Lunge landed on 2026-07-30. That is the whole reason. Unlike Lunge, Deadlift does **not**
satisfy the two corroborating criteria that made Lunge an easy pick:

| Criterion | Lunge | Deadlift |
|---|---|---|
| Next in parent-spec order | yes | **yes** |
| Labeled ground truth in-repo | 174 reps (REHAB24-6 `Ex5`) | **none** |
| KG flagship coverage | 665 `Lunge:` nodes | **13 `Deadlift:` nodes (stub)** |
| RAG documents | 8 | 2 |

REHAB24-6 covers arm abduction, arm VW, push-up, leg abduction, lunge and squat. There is no
deadlift data anywhere in this repository, and no deadlift dataset is being acquired in this
pass. **Therefore the Lunge pass's Phase 2 has no analogue here.** `DEADLIFT_DETECTOR.validated`
stays `False`, the parent spec's §8.4 ("validate thresholds against labeled data per movement
before shipping analysis for that movement") remains unsatisfied for this movement, and every
threshold below is spec-derived. This is stated once, here, rather than hedged repeatedly.

**Scope (user-approved):** detector plus tests plus spec amendments. No validation harness, no
dataset acquisition, no KG expansion.

---

## 2. What ships

Three emitting rules and one withdrawal. The parent spec lists four Deadlift rules; one does
not survive review (§5).

| Rule | Status | Observability | Retrieval |
|---|---|---|---|
| `deadlift_hips_shoot_up` | implemented | high `side`; degrades to medium ×0.65 | `rag` |
| `deadlift_incomplete_lockout` | implemented | high `side`; medium oblique | `rag` |
| `deadlift_lumbar_flexion` | implemented | **low**, hard-gated sagittal | `kg` |
| `deadlift_bar_drift` | **WITHDRAWN** | — | — |

Three rules sits at the floor of the parent spec's stated method ("3–6 core faults per
movement"). That is the honest count after the withdrawal, not a target.

### 2.1 Registration surfaces the movement in the web app

`backend/app/routers/movements.py` derives `GET /api/movements` from the detector registry
specifically "so registering a fourth detector surfaces it in the UI with no backend or
frontend edit." **Registration is promotion.** Lunge went live this way. Deadlift therefore
becomes a live, Beta-tagged card on the Movements page and an accepted `movement` value on
`/api/analyze` the moment `registry.register(DEADLIFT_DETECTOR)` runs.

This was raised explicitly and the user chose to follow the Lunge precedent: the
`validated=False` Beta tag is the mechanism built for exactly this situation. Users can
therefore upload deadlift videos to a detector no labeled data has ever checked. Recorded here
so it is a decision, not a side effect.

**Correction (2026-08-01, found while planning):** an earlier draft of this section also
required a frontend edit. It does not. `frontend/src/lib/movements.ts` already lists Deadlift
in `LOWER_BODY`, and the file states outright that analyzability "comes from GET /api/movements
… so registering a fourth detector surfaces it in the UI with no frontend edit." Nor does any
frontend test break: `pages.Movements.test.tsx:48` stubs `getMovements` with a hardcoded
three-entry fixture that was not even updated when Lunge shipped, so the frontend suite is
decoupled from the registry. **The only test requiring an update is
`tests/test_movements_endpoint.py`.**

---

## 3. Module design

New module `src/pose/movements/deadlift.py`, following `lunge.py`: raw metrics → phase
assignment → cited rules → assembled `DEADLIFT_DETECTOR`, registered under `"Deadlift"` via a
side-effect import in `src/pose/movements/registry.py`. Reachable from
`scripts/pose/run_pose_rule_detection.py --movement "Deadlift"`. No new dependencies.

### 3.1 Rep segmentation — the flexed-start path

`base.py:55` already names deadlift as the motivating case for `rep_start="flexed"`: the rep
starts at the bottom with the bar on the floor, not at an extended standing position. That
path is implemented and tested in `src/pose/rep_segmentation.py` (`_REP_STARTS`, the
`rep_start == "flexed"` branch at line 198, and
`test_flexed_still_segments_real_reps_and_truncated_partials`).

```
rep_signal   = "hip_angle_deg"
rep_polarity = "min"
rep_start    = "flexed"
```

Two consequences worth stating because they shape the rules:

1. **A flexed-start window contains the eccentric.** The rep runs floor → lockout → floor. The
   parent spec's four phases cover only the concentric, so a fifth phase `lowering` is added;
   without it, return-to-floor frames would be labeled `lockout` and
   `deadlift_incomplete_lockout` would score the descent.
2. **The window's opening frames genuinely are the floor setup.** This is what makes a
   setup-referenced baseline meaningful for Deadlift, where it would be dubious for a movement
   whose rep starts standing. `deadlift_lumbar_flexion` depends on this.

### 3.2 Phases

`setup → lift_off → mid_pull → lockout → lowering`, assigned per rep by `run_detector`.
Structure mirrors `lunge_assign_phases`: a proportional `setup` prefix, then splits on the rep
signal's excursion between its minimum and its peak, with the peak-extension frame separating
the concentric phases from `lowering`. Frames outside any rep take the shared `REST_PHASE`.

`DEADLIFT_ACTIVE_PHASES = {"lift_off", "mid_pull", "lockout"}`. `lowering` is deliberately
excluded from every rule: no rule below has literature backing for a claim about the eccentric.

### 3.3 Metrics emitted by `deadlift_compute_raw`

| Key | Definition |
|---|---|
| `hip_angle_deg` | `angle(shoulder_mid(11,12), hip_mid(23,24), knee_mid(25,26))` — rep signal + lockout |
| `knee_angle_deg` | `angle(hip_mid, knee_mid, ankle_mid(27,28))` — lockout |
| `torso_pitch_deg` | `angle_from_vertical(shoulder_mid → hip_mid)`, image plane |
| `hip_y` | image-y of the hip midpoint — §4.3's hips-near-stationary term |
| `torso_len` | `‖shoulder_mid − hip_mid‖`, image plane |

`shoulder_y` is deliberately absent: the only rule that would have consumed it turned out to be
computing trunk pitch by another route (§4.1).

`DEADLIFT_METRIC_KEYS` must stay a two-way match with what `deadlift_compute_raw` emits — a key
the tuple omits is dropped by `run_detector` (which builds each `CoreFrame.metrics` **from**
this tuple) and read back as NaN, silently. Pinned by a dedicated test, mirroring
`test_lunge_metric_keys_match_the_emitted_metrics`.

Baselines (`torso_len` at setup, `hip_y`/`shoulder_y` at setup) are **not** metrics. They are
derived inside the rules from the window's `setup`-phase frames, following squat's `heel_rise`
precedent, because `compute_raw` runs globally over the clip while a baseline is per-rep.

**Midpoint visibility caveat.** Parent spec §7 item 3 records that `_visible_midpoint` requires
both landmarks of a pair above 0.35 visibility, and that one occluded shoulder silently reverts
`body_axis_extent` to a vertical fallback — "exactly in the view most likely to trigger it: a
sagittal (side) view is precisely where far-side landmarks are most often occluded." Every
metric above is built from midpoints and every Deadlift rule wants a sagittal view, so this
detector sits squarely in that failure mode. Each rule therefore requires finite metrics in its
mask and the module documents the exposure rather than assuming it away.

---

## 4. The three rules

### 4.1 `deadlift_hips_shoot_up`

**Criterion.** During `lift_off`/`mid_pull`, flag when the trunk has flattened relative to the
rep's own setup **and** is flat in absolute terms:

```
torso_pitch_deg > torso_pitch_deg₀            (flattened vs this rep's setup)
torso_pitch_deg > 55°                         (flat in absolute terms)
```

subscript-0 = the rep's `setup` baseline. Severity ramps on peak pitch 55° → 75°.

**Why this is not written as a hip-vs-shoulder rise differential.** The parent spec phrases the
signal as "`Δ(hip_y)` rises faster than `Δ(shoulder_y)`," and an earlier draft of this design
implemented that literally as
`hip_lead_ratio = ((hip_y₀ − hip_y) − (shoulder_y₀ − shoulder_y)) / torso_len₀ > 0`. That term
was checked numerically before being written into code, and it is **algebraically identical to a
trunk-pitch change**. Since `shoulder_y − hip_y = −torso_len·cos(pitch)`, a rigid torso gives

```
hip_lead_ratio ≡ cos(pitch₀) − cos(pitch_t)
```

verified to machine precision on a sagittal stick model. It depends **only** on pitch and carries
no information whatever about how far the hips travelled — two landmarks dressing up a
single-angle test. Writing it as a differential would have implied the rule corroborates trunk
pitch with an independent kinematic signal, which is false. The parent spec's own "i.e." linking
the two phrasings turns out to be exactly right, so stating the rule in terms of pitch is
faithful to it, not a deviation.

The term does discriminate — on the stick model a good hinge (pitch 60°→0°) gives −0.21/−0.41/
−0.50 while a hips-shoot-up rep (60°→75°) gives +0.13/+0.24/+0.19 — which is why the
relative-to-setup clause is kept. It is what separates the *sequencing* fault the spec describes
from a lifter who merely sets up flat and stays there; the absolute 55° gate alone cannot.

**Consequence for metrics:** `shoulder_y` is not needed and is not emitted. `hip_y` is still
required by §4.3.

**Thresholds.** 55°/75° are the parent spec's numbers and have **no numeric backing** — neither
deadlift RAG document reports a trunk-inclination angle in degrees (verified: the only degree
value in PMC12148905 is an unrelated 8° knee adduction). What *is* backed is the mechanism and
the direction, which is what the conjunction encodes. Labeled as spec-derived in-code.

**View handling — degrade, do not gate.** Head-on, a forward-pitched trunk projects as a short,
near-vertical segment, so `torso_pitch_deg` *under*-reads. The failure mode off-view is
therefore silence, not a wrong claim, and the rule follows squat's `excessive_forward_lean`
precedent: emit at medium observability with the ×0.65 confidence discount rather than hard-gate.
(Contrast §4.3, where the failure mode is inverted and the decision flips.) Note the rise
differential itself is view-robust — vertical is vertical in any projection.

**Citation.** Moreira VM, et al. PMC12225233 (2023); cross-support Hanen NC, et al. PMC12148905
(2025). Support: "leaning the trunk forward results in higher spinal flexion torque generated by
the barbell," requiring greater erector-spinae force to resist flexion; PMC12148905 frames a
"significantly reduced trunk inclination angle" as the low-back-sparing state.

### 4.2 `deadlift_incomplete_lockout`

**Criterion.** At `lockout`, flag `hip_angle_deg < 165°` **OR** `knee_angle_deg < 165°`. Ramp
165° → 140° on **both** axes, take the worse.

**Score both ramps.** The parent spec's §8 status note records that `ohp_incomplete_lockout`
originally selected its ramp by asking which reading was finite, which mis-attributed severity
when a segment fired on one criterion alone. This rule copies the fix, not the bug.

**Thresholds.** The best-grounded rule of the three. Moreira PMC12225233 measured the three key
positions at lift-off ≈ **95°**, mid-pull ≈ **126°** and lock-out ≈ **180°**, with "180° …
equivalent to full extension" — so the 180° triple-extension target is a measured quantity, not
an assumption. Hanen PMC12148905 independently defines completion as "a fully upright position
with extended hips and knees, with scapular retraction." The 165° flag point is a tolerance
neither source states; labeled spec-derived in-code.

**Observability.** High on `side`, medium on oblique/front (hip extension is partly foreshortened
head-on). Degrades with the ×0.65 discount; no hard gate — an angle magnitude, like §4.1,
under-reads rather than inverts.

### 4.3 `deadlift_lumbar_flexion`

The clinically most important deadlift fault and the weakest detection. Parent spec §7 already
lists it as low-observability; this design does not upgrade that.

**Criterion (proxy).** In a sagittal view a rigid hip hinge holds the projected shoulder→hip
segment length constant, because the trunk rotates *within* the image plane. Shortening against
the rep's own setup baseline, while the hips are not themselves travelling, is therefore
consistent with the trunk curling:

```
torso_len / torso_len₀  <  0.95               (unsourced — see below)
|hip_y − hip_y₀| / torso_len₀  <  0.10        (hips near-stationary)
```

evaluated over `lift_off`/`mid_pull`. Severity ramps 0.95 → 0.85 on the length ratio
(more shortening = worse).

**Both numbers are pinned here so they are reviewed rather than chosen at the keyboard, and
both are arbitrary.** 0.95 says "5% shortening," picked to sit above frame-to-frame landmark
jitter without any measurement of what that jitter actually is; 0.85 is a doubling of it. 0.10
of a torso length is a loose "the hips have not really moved yet" band. None of the three is
derived from a source or from data.

**The threshold is unsourced and will be named as such.** No source gives a
segment-shortening-to-lumbar-flexion figure. The user chose to ship the rule at low
observability with this cost accepted rather than silence it. Mitigations: the constant carries
`UNSOURCED` in its name and an in-code comment stating no literature backs it; the rule emits at
observability `low` with the ×0.65 discount (`run_detector` already sorts `low` last, so it
cannot outrank a grounded fault); and the gap is added to parent-spec §7. An alternative —
calibrating the gate to the measured landmark-jitter floor — was offered and declined as out of
scope for a CLI-only pass; it remains the obvious upgrade path.

**View handling — hard gate, unlike §4.1.** Off-view, trunk pitch alone shortens the projected
segment, so the proxy produces false **positives** rather than silence. Where the failure mode is
a wrong claim, the OHP precedent (`ohp_forward_head`) hard-gates. This rule therefore returns no
detections at all outside `{side, front_oblique}` and additionally requires
`view_confidence >= 0.20` (`SIDE_VIEW_CONF_THRESHOLD`, shared with squat's `rule_knees_forward`
and OHP — no new number introduced). Per parent-spec §7 item 2, `front_oblique` is unreachable
in the production path, so in practice this is `side` only.

**Citation.** Moreira VM, et al. PMC12225233 (2023). Support: "The lift-off position in DL,
using the powerlift posture, generates greater lumbar spine shear force," and erector-spinae
activation peaks at lift-off/mid-pull because "ERE requires higher activation and higher strength
to avoid trunk flexion, reducing shear." Note what this does and does not do: it establishes the
**fault** is real, loaded and mechanistically understood. It says nothing about detecting it from
pose. The weakness here is in the measurement, not the evidence — which is precisely what
distinguishes this rule from the one withdrawn in §5.

---

## 5. `deadlift_bar_drift` is withdrawn

The parent spec prescribes a wrist-as-bar proxy referenced to
`midfoot_x = mean(ankle_x, foot_index_x)`, flagged past `0.5·foot_len`. Withdrawn for three
reasons, mirroring the OHP bar-path withdrawal at parent-spec lines 498–521:

1. **The citation contains no bar-path measurement.** Hanen PMC12148905 was read in full for
   this: the only bar-position statement is qualitative — "keeping the barbell closer to the body
   during the SDL reduces the lever arm stress." No distance, no threshold, no units.
2. **The citation explicitly disclaims it.** Line 243 of the RAG document: *"Analyzing the bar
   path would be valuable to validate this hypothesis."* The paper did not analyze bar path and
   says so. A rule cannot cite a source for a measurement that source declares un-performed.
3. **The mid-foot reference is the precise construct already forbidden.** The OHP withdrawal
   rejected bar-path partly because referencing to mid-foot "would require an invented mid-foot
   proxy — forbidden by this project's every-threshold-literature-backed premise." The Deadlift
   section then prescribes exactly that construct.

Reason 3 alone makes it inconsistent with a decision already taken; reasons 1 and 2 make it
unsupported on its own terms. A boxed `WITHDRAWN` note in the parent spec's Deadlift section will
record this, in the same form as OHP's.

**Open spec question (recorded, not resolved):** does the Deadlift rule set want a genuine
bar-path fault? It would need (a) a base-of-support reference MediaPipe can actually resolve and
(b) a citation that measures bar displacement with a number. Neither exists today. This is a
withdrawal pending a decision, not a silent deletion.

---

## 6. KG retrieval — measured, not assumed

Following the Lunge Task-3 Step-0 protocol, every candidate query was resolved against the live
graph **before** any rule was written, via
`graph_retrieval.resolve_nodes(g, q, movement="Deadlift")` on `data/kg/sports_kg_v3.graphml`:

| Query | Resolves to |
|---|---|
| `Lumbar Flexion` | `Deadlift:Lumbar Flexion` |
| `Insufficient Hip Hinge` | `Deadlift:Insufficient Hip Hinge` |
| `Hyperextension At Lockout` | `Deadlift:Hyperextension At Lockout` |
| `Bar Drift From Body` | `Deadlift:Bar Drift From Body` |
| `Incomplete Lockout`, `Rounded Lower Back`, `Spinal Flexion`, `Trunk Over Inclination`, `Anterior Trunk Tilt`, `Excessive Forward Lean` | *(nothing)* |
| `Hips Rise Before Shoulders`, `Insufficient Hip Extension` | `Hip` (generic shared-layer anatomy node) |
| `Incomplete Range Of Motion` | `Range Of Motion` (generic) |

**Assignments.** `deadlift_lumbar_flexion` → `retrieval_mode="kg"`, query `"Lumbar Flexion"`.
The other two → `retrieval_mode="rag"` (the codebase's existing no-KG value, already used at
`pose_rule_detector.py:541`). The two rejections are deliberate:

- **`Insufficient Hip Hinge` is a near-miss in the wrong direction.** Insufficient hinge means
  failing to push the hips back — a knee-dominant, squat-like pull. `deadlift_hips_shoot_up` is
  the opposite failure: excessive hip dominance with the trunk flattening. Grounding the coaching
  chat on it would retrieve advice for the inverse fault.
- **`Hyperextension At Lockout` is the literal opposite of `deadlift_incomplete_lockout`** — too
  much extension versus too little.
- The bare `Hip` and `Range Of Motion` hits are shared-layer anatomy nodes, not faults, and would
  ground explanations on an anatomical concept rather than on an error.

The Lunge brief's rule applies: *"If no node exists for a fault, do not invent a near-miss."*

**Divergence worth recording in the parent spec.** The 4-node Deadlift stub carries nodes for two
faults the spec has no rule for (`Hyperextension At Lockout`, `Insufficient Hip Hinge`), lacks
nodes for two rules that do exist, and its one perfectly-matching fault node
(`Bar Drift From Body`) belongs to the rule being withdrawn. The stub and the rule catalog were
authored independently and do not agree. Fixing that is KG work, explicitly out of scope here;
this pass records it.

---

## 7. Testing

`tests/test_deadlift.py`, `unittest.TestCase`, mirroring `tests/test_lunge.py`'s fixture style
(windows of synthetic `CoreFrame`s carrying chosen metrics):

- `DEADLIFT_METRIC_KEYS` is a two-way match with `deadlift_compute_raw`'s output.
- Phase assignment: the five phases in order; the flexed-start boundary; `lowering` is not
  scored; `assign_phases` returns exactly one phase per input frame (the length contract
  `run_detector` raises on).
- Per rule: fires on a clear positive; stays silent on a clean rep; severity ramp endpoints;
  NaN guards.
- `deadlift_incomplete_lockout`: a segment firing on the hip criterion alone still scores the
  hip ramp (the OHP mis-attribution regression).
- **`deadlift_incomplete_lockout` fires on a rep that never locks out.** The phase split keys on
  peak extension, and the fault *is* failing to reach extension — so a rep peaking at, say, 150°
  must not collapse `lockout` to fewer frames than
  `min_frames = max(3, ceil(fps·0.2))`, which would make `contiguous_true_segments` return
  nothing and silence the rule on exactly the reps it exists to catch. Pinned with an explicit
  shallow-finish fixture. This is the same class as Lunge's discarded-opening-15% finding, which
  this codebase has already been bitten by once.
- View handling, both directions: §4.1 and §4.2 emit off-view at ×0.65; §4.3 emits **nothing**
  off-view and nothing below `view_confidence` 0.20.
- Registry round-trip; `tests/test_movements_endpoint.py` updated for the five-detector list and
  `validated` map. No frontend test is affected — see the correction in §2.1. `yarn test:coverage`
  is still run, to confirm that rather than assume it.

Gates, per CLAUDE.md: `.venv\Scripts\python.exe -m pytest tests/`,
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`, and
`yarn test:coverage` with cwd = `frontend/`.

---

## 8. Deliverables

1. `src/pose/movements/deadlift.py` — 3 rules, registered.
2. `src/pose/movements/registry.py` — side-effect import.
3. `tests/test_deadlift.py`; updates to `tests/test_movements_endpoint.py`.
4. Parent-spec amendments: boxed `WITHDRAWN` note on `deadlift_bar_drift`; §7 gains the unsourced
   `lumbar_flexion` threshold; §6/§7 gain the KG-stub divergence and the two `rag` fallbacks.

**Not delivered, and why:** threshold validation (no labeled data exists), KG expansion (out of
scope), a bar-path rule (no citation supports one).
