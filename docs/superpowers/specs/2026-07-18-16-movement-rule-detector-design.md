# 16-Movement Cited Rule-Detector Specification

**Status:** design spec (foundation for detector implementation) · **Date:** 2026-07-18
**Author:** Claude Fable 5 (x-coach) · **Rules:** 70 across 16 canonical movements

---

## 1. Purpose

This document catalogs the **core fault rules** for every one of x-coach's 16 canonical
movements. It is the vetted, literature-grounded foundation that the pose rule-detector
(`src/pose/pose_rule_detector.py`) will implement per movement. Today the detector ships
squat only; the squat rules here **re-state the five already-coded rules** (`knees_inward`,
`knees_forward`, `shallow_depth`, `excessive_forward_lean`, `heel_rise`) with their exact
geometry/thresholds and now attach explicit literature citations (the code previously carried
only KG-query strings). Every other movement's rules are new.

**Every rule carries at least one literature citation whose specific finding backs it** — no
rule rests on common sense or unsupported assertion. Each citation is accompanied by a
`citation_support` line quoting or paraphrasing the exact finding, extracted from a source that
was actually read (a RAG document under `data/rag/docs/`, or a web source fetched during
research). This is the anti-hallucination guarantee the task required.

## 2. Method

- **Depth:** 3–6 core, biomechanically important, monocular-detectable faults per movement
  (quality over quantity), mirroring the existing squat set.
- **Citations, RAG-first:** the existing RAG corpus (`data/rag/docs/`, 81 documents mapped to
  movements via `data/paper_metadata.json`) grounds 15 of the 16 movements. Sub-researchers read
  the relevant papers per movement and extracted the specific finding for each rule.
- **Web for gaps:** three movements had thin or zero peer-reviewed RAG coverage — **High Knee**
  (no RAG doc), **Torso Twist** and **Jumping Jacks** (Wikipedia-only). For these, peer-reviewed
  sources were found via web search and verified by fetching the source page. Wikipedia is used
  only as supplementary *descriptive* support, never as sole backing for an injury-risk claim.
- **Authoritative references:** for RAG-sourced papers, the canonical author/title/journal is the
  entry in `data/paper_metadata.json` and the References index (§6); inline citations key on the
  PMCID/PMID/DOI, which is the reliable anchor.

## 3. Detection model (applies to every rule)

Detection runs on **MediaPipe Pose, 33 landmarks**, normalized image coordinates
(x, y ∈ [0,1], y increasing downward; plus z depth and a visibility score), **monocular single
camera**. Landmark indices referenced throughout:

| idx | landmark | idx | landmark | idx | landmark |
|----|----------|----|----------|----|----------|
| 0 | nose | 13 | L elbow | 25 | L knee |
| 7 | L ear | 14 | R elbow | 26 | R knee |
| 8 | R ear | 15 | L wrist | 27 | L ankle |
| 11 | L shoulder | 16 | R wrist | 28 | R ankle |
| 12 | R shoulder | 23 | L hip | 29/30 | L/R heel |
| | | 24 | R hip | 31/32 | L/R foot index |

A per-rep `view_type` ∈ {side, front, rear, front_oblique, rear_oblique} is estimated
(`src/pose/view_estimation.py`). Many faults are only observable from certain views; each rule
states its required view and an **observability** rating (high / medium / low / none). Confidence
is scaled down when the required view is unavailable (the coded squat detector multiplies by
~0.65), matching existing convention.

## 4. Per-rule schema

Each fault is specified with: **fault_id** (snake_case), **fault_name**, **description**,
**detection_heuristic** (concrete pose-geometry signal + threshold + direction),
**observability** (rating + required view), **biomechanical_rationale** (injury or performance
reason), **citation** (reference + PMCID/PMID/DOI/URL), and **citation_support** (the specific
finding backing the rule). Where a well-known fault is **not** reliably recoverable from
monocular pose (e.g. scapular winging, true lumbar flexion), it is listed with observability
`low`/`none` and an explicit proxy rather than a fabricated precise measure.

## 5. Rule catalog

| Group | Movements | Rules |
|---|---|---|
| A | Squat, Lunge, Deadlift | 13 |
| B | Push-up, Overhead Press | 10 |
| C | Row, Band Pull Apart | 9 |
| D | Bicep Curl, Arm Abduction, Arm VW | 12 |
| E | Sit-up, Shoulder Bridge, Leg Abduction | 12 |
| F | Torso Twist, Jumping Jacks, High Knee | 14 |
| | **Total** | **70** |



---

## Group A — Lower-body compound — Squat, Lunge, Deadlift

Movements: **Squat**, **Lunge**, **Deadlift**. Detection runs on MediaPipe Pose (33
landmarks, normalized image coords x,y ∈ [0,1] + z depth + visibility), monocular single
camera. Landmark indices used below: 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24
hips, 25/26 knees, 27/28 ankles, 29/30 heels, 31/32 foot index. A `view_type` is estimated
per rep (side / front / rear / front_oblique / rear_oblique).

Severity ramps are stated as `mild → severe` on the driving metric; confidence is scaled
down when the required `view_type` is not available (matching the existing squat detector,
which multiplies confidence by ~0.65 when the fault's view is unavailable).

---

### Squat

Rep phases: setup → descent → bottom → ascent → lockout.
(These 5 faults RE-STATE the already-coded rules in `src/pose/pose_rule_detector.py`, which
previously carried only KG-query strings; each now has a specific literature citation.)

#### Knees Inward / Knee Valgus

- **fault_id**: knees_inward
- **fault_name**: Knees Inward / Knee Valgus
- **description**: The knees collapse medially so knee separation becomes narrower than ankle separation during the loaded portion of the rep.
- **detection_heuristic**: In the frontal projection, `knee_width = ||L_knee(25) − R_knee(26)||` and `ankle_width = ||L_ankle(27) − R_ankle(28)||` (2-D, image plane). Flag when `knee_width / ankle_width < 0.82` during descent/bottom/ascent. Severity ramp 0.82 → 0.70 (ratio falling = worse).
- **observability**: high on **front / rear / front_oblique / rear_oblique**; medium on side (medial travel is foreshortened, confidence ×0.65).
- **biomechanical_rationale**: Dynamic knee abduction (valgus) is a leading non-contact ACL-injury and patellofemoral-pain mechanism; it loads the ACL and lateral patellofemoral joint the movement is not built to resist.
- **citation**: Ford KR, et al. "An evidence-based review of hip-focused neuromuscular exercise interventions to address dynamic lower extremity valgus." Open Access J Sports Med (2015). PMC4556293.
- **citation_support**: "knee abduction moment, which directly contributes to dynamic lower extremity valgus, was a significant predictor for future ACL injury risk with 73% sensitivity and 78% specificity in a prospective study of young female athletes"; dynamic lower extremity valgus is defined as "hip adduction and internal rotation, knee abduction, tibial external rotation and anterior translation, and ankle eversion" and "high knee abduction moment was predictive of both PFP and ACL injury risk." Verified in RAG doc.

#### Knees Forward / Anterior Knee Translation

- **fault_id**: knees_forward
- **fault_name**: Knees Forward / Anterior Knee Translation
- **description**: The knee travels excessively past the toes in the sagittal plane, driving the shank forward under the descending torso.
- **detection_heuristic**: Per leg, project the knee onto the foot vector `toe − ankle` and normalise by foot length: `knee_forward_ratio = (proj(knee−ankle onto foot) − foot_len)/foot_len` (2-D). Flag when `knee_forward_ratio > 0.10` during active phases; severe ≥ 0.30. Only computed when the shank is seen from the side.
- **observability**: high on **side** (view_confidence ≥ 0.20); low otherwise (knee-to-toe projection is unreliable head-on — the detector already emits a low-observability placeholder in that case).
- **biomechanical_rationale**: Letting the knee translate in front of the toes sharply raises knee-extensor and patellar-tendon loading, which is the mechanism behind anterior-knee/patellar-tendon overload.
- **citation**: Zellmer M, et al. "Patellar tendon stress between two variations of the forward step lunge." J Sport Health Sci (2019). PMC6523035. [Lunge study; mechanism transfers directly to the squat's knee-over-toe question.]
- **citation_support**: With standardized step length, moving the knee in front of the toes (FSL-FT) vs behind (FSL-BT) produced "peak patellar tendon stress … 11.1% greater," stress impulse "18.8% greater," peak quadriceps force 12.6% greater, and peak knee extension moment 25.8% greater (all p < 0.001; Table 1). Verified in RAG doc.

#### Shallow Depth

- **fault_id**: shallow_depth
- **fault_name**: Shallow Depth
- **description**: The lifter fails to reach parallel — the hip crease does not descend to knee level and the knee stays too extended at the turnaround.
- **detection_heuristic**: At the `bottom` phase, using hip-midpoint(23,24) and knee-midpoint(25,26): flag when `hip_y − knee_y < −0.02` (hip stays above knee in image-y, y grows downward) **OR** `avg_knee_angle > 105°`. Severity ramps: hip axis −0.02 → −0.10; knee-angle axis 105° → 125°; take the max.
- **observability**: high on **side / front / front_oblique**; medium on rear/rear_oblique (hip-crease occluded).
- **biomechanical_rationale**: Habitually training only partial (above-parallel) squats under heavy load is associated with long-term degeneration of the knee and spinal joints, and truncates the muscular effort/ROM stimulus that makes the squat effective. Depth to at least parallel is the accepted training and competition standard.
- **citation**: Hartmann H, Wirth K, Klusemann M. "Analysis of the load on the knee joint and vertebral column with changes in squatting depth and weight load." Sports Medicine 43(10):993–1008 (2013). DOI 10.1007/s40279-013-0073-6, PMID 23821469. Supplemented descriptively by the Wikipedia "Squat (exercise)" article (parallel-depth standard).
- **citation_support**: PubMed abstract (fetched): "With the same load configuration as in the deep squat, half and quarter squat training with comparatively supra-maximal loads will favour degenerative changes in the knee joints and spinal joints in the long term," and "the deep squat presents an effective training exercise for protection against injuries and strengthening of the lower extremity." Wikipedia adds that competition standard is hip crease below the top of the knee and that "incomplete squats … are both less effective and more likely to cause injury." Verified via WebFetch + RAG doc.

#### Excessive Forward Lean

- **fault_id**: excessive_forward_lean
- **fault_name**: Excessive Forward Lean
- **description**: The torso folds toward horizontal (the "good-morning" squat) so the shoulders travel forward of the hips and the back angle flattens.
- **detection_heuristic**: `torso_lean_deg = angle_from_vertical(shoulder_mid(11,12) → hip_mid(23,24))` in the image plane. Flag when `torso_lean_deg > 35°`; severity ramp 35° → 55°.
- **observability**: high on **side / front_oblique / rear_oblique**; medium head-on (confidence ×0.65 — trunk pitch is foreshortened in a pure front/rear view).
- **biomechanical_rationale**: A more horizontal trunk lengthens the moment arm of the load about the lumbar spine, raising spinal flexion torque and shear and increasing lower-back injury risk, while shifting work off the quads onto the low back.
- **citation**: Moreira VM, et al. "Analysis of Muscle Strength and Electromyographic Activity during Different Deadlift Positions." Muscles (2023). PMC12225233. Supplemented by Ross S (Starting Strength), "The Good Morning Squat" (coaching description), and Wikipedia "Squat (exercise)."
- **citation_support**: PMC12225233: "leaning the trunk forward results in higher spinal flexion torque generated by the barbell. Therefore, ERE [erector spinae] requires higher activation and higher strength to avoid trunk flexion, reducing shear," and greater lumbar spine shear force accompanies the more forward-inclined positions. Wikipedia: "Over-flexing the torso greatly increases the forces exerted on the lower back, risking a spinal disc herniation." Verified in RAG docs.

#### Heel Rise

- **fault_id**: heel_rise
- **fault_name**: Heel Rise
- **description**: The heels lift off the floor at depth so weight rolls onto the forefoot, usually a work-around for limited ankle dorsiflexion.
- **detection_heuristic**: Per foot, `heel_height_delta = heel_y(29/30) − toe_y(31/32)` (image-y). Establish a `setup` baseline (mean over setup frames); at `bottom`, flag when `heel_height_delta − baseline > 0.015`. Severity ramp 0.015 → 0.055.
- **observability**: medium on **side / oblique** (heel-vs-toe height needs a lateral or oblique view; nearly invisible head-on).
- **biomechanical_rationale**: Heel rise signals exhausted/limited ankle dorsiflexion, which forces compensatory joint moments up the kinetic chain (knee, hip, spine) and is associated with ankle and knee injury risk; it also shifts load off the posterior chain (glutes) onto the forefoot/quads.
- **citation**: Mata AJ, Hayashi H, Moreno PA, Dudley RI, Sorenson EA. "Hip Flexion Angles During Supine Range of Motion and Bodyweight Squats." Int J Exerc Sci 14(1):912–918 (2021). Supplemented by Tumminello N, Human Kinetics, "Heel-raised squats aren't bad" (dorsiflexion-limitation context) and Wikipedia "Squat (exercise)."
- **citation_support**: Mata 2021: heel elevation increased ankle excursion (25.9°→34.7° / 24.6°→33.2°, p<0.001) and squat depth (30.9%→55.0% leg length, p<0.001), and "reduced dorsiflexion mobility can lead to compensatory joint moments up the kinetic chain, potentially leading to injury." Human Kinetics: restricted ankle dorsiflexion "has been associated with … ankle injuries and knee injuries … abnormal lower extremity biomechanics." Wikipedia: raising the heels "reduces the contribution of the gluteus muscles." Verified via RAG docs.

---

### Lunge

Rep phases (lead leg): stance/setup → descent → bottom (deep lunge) → ascent → recovery.

#### Lead Knee Past Toes

- **fault_id**: lunge_knee_past_toes
- **fault_name**: Lead Knee Past Toes / Anterior Knee Translation
- **description**: The lead knee drifts well in front of the toes as the lifter lowers, over-loading the front knee.
- **detection_heuristic**: On the lead leg, `knee_forward_ratio = (proj(knee−ankle onto (toe−ankle)) − foot_len)/foot_len` (2-D, as in the squat). Flag when `> 0.10` during descent/bottom/ascent; severe ≥ 0.30. Lead leg = the more flexed / more anterior foot.
- **observability**: high on **side** (view_confidence ≥ 0.20); low head-on (sagittal knee travel not resolvable).
- **biomechanical_rationale**: Allowing the lead knee to translate in front of the toes materially increases patellar-tendon stress and knee-extensor demand — a progression lever in patellar-tendinopathy rehab, but a load spike to control in general training.
- **citation**: Zellmer M, et al. "Patellar tendon stress between two variations of the forward step lunge." J Sport Health Sci (2019). PMC6523035.
- **citation_support**: Knee-in-front-of-toes lunges (FSL-FT) vs knee-behind-toes (FSL-BT) gave "peak patellar tendon stress … 11.1% greater," stress impulse "18.8% greater," peak quadriceps force 12.6% greater, peak knee-extension moment 25.8% greater, and peak knee flexion 110.2°→124.7° (all p<0.001; Table 1). Verified in RAG doc.

#### Lead Knee Valgus

- **fault_id**: lunge_knee_valgus
- **fault_name**: Lead Knee Valgus / Medial Collapse
- **description**: The lead knee caves medially relative to the hip–ankle line (knee drifts inside the foot) during the loaded phase.
- **detection_heuristic**: In the frontal projection, take the lead-leg hip(23/24)→ankle(27/28) line; measure the signed medial offset of the knee(25/26) x-coordinate from that line, normalised by hip width. Flag when medial offset > ~0.10·hip_width toward the midline; ramp 0.10 → 0.25. (Frontal-plane knee-abduction proxy — no true 3-D abduction angle from monocular pose.)
- **observability**: high on **front / front_oblique**; low on side (frontal collapse invisible in the sagittal view).
- **biomechanical_rationale**: Dynamic lower-extremity valgus (hip adduction/IR + knee abduction) at the knee is a documented predictor of ACL rupture and patellofemoral pain; the single-leg loaded lunge is exactly where hip-abductor/rotator control failures surface.
- **citation**: Ford KR, et al. "An evidence-based review of hip-focused neuromuscular exercise interventions to address dynamic lower extremity valgus." PMC4556293 (2015).
- **citation_support**: "knee abduction moment … was a significant predictor for future ACL injury risk with 73% sensitivity and 78% specificity"; "the inability to eccentrically control hip adduction and internal rotation may lead to greater dynamic lower extremity valgus commonly seen during landing, squatting, and running." Verified in RAG doc.

#### Insufficient Lunge Depth

- **fault_id**: lunge_insufficient_depth
- **fault_name**: Insufficient Depth
- **description**: The lead knee never reaches roughly a right angle at the bottom, so the working range and quadriceps demand are truncated.
- **detection_heuristic**: Lead-knee angle = `angle(hip(23/24), knee(25/26), ankle(27/28))`. Flag when the minimum lead-knee angle across the rep `> 100°` (i.e., < ~80° of flexion). Severity ramp 100° → 130° (more extended = worse). The canonical target is ~90° knee flexion.
- **observability**: high on **side / front_oblique**; medium head-on.
- **biomechanical_rationale**: Patellofemoral and quadriceps loading rise monotonically with lead-knee flexion through the mid-range; a chronically shallow lunge forfeits that strengthening stimulus (the standard clinical/lab target is 90° flexion).
- **citation**: Alkjær T, et al. "Forward lunge before and after anterior cruciate ligament reconstruction." PLoS One (2020), PMC6980669. Supplemented by Escamilla R, et al. "Patellofemoral Joint Loading During the Performance of the Forward and Side Lunge with Step Height Variations." IJSPT (2022), PMC8805090.
- **citation_support**: PMC6980669 defines the protocol as "flexing the knee to 90°" as the target depth, and reduced knee flexion/extensor moment marks impaired (non-coper) function. PMC8805090: "patellofemoral joint force and stress generally increased progressively as knee flexion increased during the descent phase" — i.e., depth is what produces the loading/strengthening stimulus. Verified in RAG docs.

#### Pelvic Drop / Trunk Lateral Lean

- **fault_id**: lunge_pelvic_drop
- **fault_name**: Pelvic Drop / Contralateral Trunk Lean (Trendelenburg)
- **description**: The non-stance-side pelvis drops and/or the trunk leans over the lead leg, signalling weak lead-hip abductor control.
- **detection_heuristic**: Frontal projection: `pelvis_tilt_deg = angle_from_horizontal(L_hip(23) → R_hip(24))`; also `trunk_lateral_lean = angle_from_vertical(shoulder_mid → hip_mid)` in the x–y plane. Flag when `pelvis_tilt_deg > 8°` (contralateral hip lower) sustained through bottom/ascent; ramp 8° → 20°.
- **observability**: medium on **front / rear**; low head-on-only ambiguity resolved by hip-landmark visibility; not observable from a pure side view.
- **biomechanical_rationale**: Contralateral pelvic drop / ipsilateral trunk lean is the visible signature of hip-abductor (gluteus medius) insufficiency, part of the dynamic-valgus chain that raises ACL and patellofemoral-pain risk and destabilises single-leg loading.
- **citation**: Ford KR, et al. PMC4556293 (2015). Cross-support: Alkjær T, et al. PMC6980669 (2020).
- **citation_support**: PMC4556293: "Failure to produce the abduction force is observed as a Trendelenburg posture, with the contralateral pelvis dropping," and hip-focused training reduced "ipsilateral trunk inclination, and contralateral pelvis depression during a single leg squat." PMC6980669 found gluteus medius EMG "significantly higher for the ACL injured participants … possibly a compensatory mechanism to control the trunk and pelvis in the frontal plane." Verified in RAG docs.

---

### Deadlift

Rep phases: setup → lift-off → knee-passing (mid-pull) → lockout. **Monocular caveat:** the
deadlift is filmed from the side and the single most important fault — lumbar (lower-back)
flexion under load — is only weakly observable from MediaPipe landmarks (no spine markers
between shoulders and hips), so it is marked honestly below.

#### Lumbar Flexion Under Load

- **fault_id**: deadlift_lumbar_flexion
- **fault_name**: Rounded Lower Back / Lumbar Flexion
- **description**: The lower back loses its neutral/extended posture and rounds into flexion during lift-off, the highest-shear moment of the pull.
- **detection_heuristic**: **Proxy only.** MediaPipe has no lumbar landmarks; the shoulder(11/12)→hip(23/24) segment cannot separate true lumbar flexion from hip hinge or thoracic curvature. Best available proxy: track the shoulder–hip segment angle and flag a *change* in back angle between setup and lift-off that is inconsistent with a rigid hip hinge (e.g., shoulder–hip segment shortening in the image with the hips near-stationary). Report low confidence and defer to view quality.
- **observability**: **low**, needs **side / side_oblique**; genuinely rounded-vs-neutral spine is not reliably resolvable from 33-landmark monocular pose. Do NOT assert precision here.
- **biomechanical_rationale**: Lumbar flexion under a heavy load concentrates shear on the intervertebral discs/posterior structures and is the classic deadlift lower-back injury mechanism; the erector spinae must fire hard specifically to prevent trunk/spinal flexion.
- **citation**: Moreira VM, et al. "Analysis of Muscle Strength and Electromyographic Activity during Different Deadlift Positions." Muscles (2023). PMC12225233.
- **citation_support**: "The lift-off position in DL, using the powerlift posture, generates greater lumbar spine shear force," and erector-spinae activation was highest at lift-off/mid-pull because "leaning the trunk forward results in higher spinal flexion torque … ERE requires higher activation and higher strength to avoid trunk flexion, reducing shear." Verified in RAG doc. (Rule is retained for coaching value but flagged low-observability per the honest-reporting rule.)

#### Bar Drifting Away From the Body

- **fault_id**: deadlift_bar_drift
- **fault_name**: Bar Drift / Bar Away From Shins
- **description**: The bar/hands travel forward of the mid-foot instead of staying tucked against the legs, lengthening the load's lever arm on the back.
- **detection_heuristic**: No bar landmark exists; use the **wrist** as a bar proxy. Side view: `bar_offset = wrist_x(15/16) − midfoot_x` where `midfoot_x = mean(ankle(27/28)_x, foot_index(31/32)_x)`, normalised by foot length. Flag when `bar_offset` (forward of mid-foot) exceeds ~0.5·foot_len during lift-off/mid-pull; ramp 0.5 → 1.2.
- **observability**: medium on **side** (requires a lateral view; the fore–aft axis collapses head-on).
- **biomechanical_rationale**: Keeping the bar close minimises the horizontal moment arm about the lumbar spine; letting it drift forward increases lower-back lever-arm stress and back-extensor demand, degrading both safety and efficiency.
- **citation**: Hanen NC, et al. "Biomechanical analysis of conventional and sumo deadlift." Front Bioeng Biotechnol (2025). PMC12148905, DOI 10.3389/fbioe.2025.1597209.
- **citation_support**: "keeping the barbell closer to the body during the SDL reduces the lever arm stress, thereby decreasing mechanical stress on the lower back"; a more upright posture with reduced trunk inclination is what "facilitated" lower back-loading in the wider-stance pull. Verified in RAG doc.

> **WITHDRAWN — bar drift.** This rule is **withdrawn** (2026-08-01) and is NOT implemented in
> `src/pose/movements/deadlift.py`, for three reasons:
>
> 1. **The citation contains no bar-path measurement.** Hanen PMC12148905 was read in full. Its
>    only bar-position statement is qualitative — "keeping the barbell closer to the body during
>    the SDL reduces the lever arm stress." No distance, no threshold, no units. The
>    `0.5·foot_len` figure above has no source.
> 2. **The citation explicitly disclaims it.** The paper states: *"Analyzing the bar path would
>    be valuable to validate this hypothesis."* It did not analyze bar path and says so. A rule
>    cannot cite a source for a measurement that source declares un-performed.
> 3. **The mid-foot reference is the construct already forbidden.** The OHP bar-path withdrawal
>    (above) rejected referencing the bar to mid-foot because it "would require an invented
>    mid-foot proxy — forbidden by this project's every-threshold-literature-backed premise."
>    This rule prescribes exactly that construct.
>
> **Open spec question:** does the Deadlift rule set want a genuine bar-path fault? It would need
> (a) a base-of-support reference MediaPipe can resolve and (b) a citation that measures bar
> displacement with a number. Neither exists today. This is a withdrawal pending a decision, not
> a silent deletion.

#### Hips Rise Faster Than Shoulders (Stiff-Leg / Segment Split)

- **fault_id**: deadlift_hips_shoot_up
- **fault_name**: Hips Rise Before Shoulders / Trunk Over-Inclination
- **description**: Off the floor the hips shoot up while the shoulders lag, so the trunk pitches toward horizontal and the pull becomes a back-dominant grind.
- **detection_heuristic**: Side view. Track `torso_pitch = angle_from_vertical(shoulder_mid(11,12) → hip_mid(23,24))` and hip-height vs shoulder-height rate over lift-off. Flag when `torso_pitch` *increases* (trunk flattens) early in the pull, i.e., `Δ(hip_y) rises faster than Δ(shoulder_y)` while `torso_pitch > 55°` from vertical. Ramp on peak pitch 55° → 75°.
- **observability**: high–medium on **side / side_oblique**; not observable head-on.
- **biomechanical_rationale**: When the hips out-run the shoulders, the trunk becomes more horizontal, increasing the spinal-flexion torque and back-extensor load the abstract ties to greater lumbar demand; it also strips the quads/hip-extensors of leverage, hurting the lift.
- **citation**: Moreira VM, et al. PMC12225233 (2023). Cross-support: Hanen NC, et al. PMC12148905 (2025).
- **citation_support**: PMC12225233: "In these two positions [lift-off, mid-pull], leaning the trunk forward results in higher spinal flexion torque generated by the barbell," requiring greater erector-spinae force to resist flexion. PMC12148905: a wider-stance pull that "maintain[s] a more upright posture … with a significantly reduced trunk inclination angle" reduces low-back lever-arm stress — i.e., excessive trunk inclination is the loaded state to avoid. Verified in RAG docs.

#### Incomplete Lockout

- **fault_id**: deadlift_incomplete_lockout
- **fault_name**: Incomplete Lockout
- **description**: The rep ends without full, upright hip and knee extension (hips still flexed / trunk not vertical).
- **detection_heuristic**: At the top phase, `hip_angle = angle(shoulder(11/12), hip(23/24), knee(25/26))` and `knee_angle = angle(hip, knee, ankle(27/28))`. Flag when `hip_angle < 165°` OR `knee_angle < 165°` at rep end (target ≈ 180° triple extension). Severity ramp 165° → 140°.
- **observability**: high on **side**; medium on oblique/front (hip extension partly foreshortened head-on).
- **biomechanical_rationale**: A completed deadlift is defined by triple extension — fully extended hips and knees with the trunk upright and scapulae retracted; stopping short means the rep's terminal ROM (and the glute/erector lockout demand) is never achieved.
- **citation**: Hanen NC, et al. PMC12148905 (2025). Cross-support: Moreira VM, et al. PMC12225233 (2023).
- **citation_support**: PMC12148905: "the third event, lift completion, is achieved when the athlete assumes a fully upright position with extended hips and knees, with scapular retraction," and "A trial was considered successful if, at the end of the concentric phase, the participant stood upright with fully extended knees and hips, a straight torso, and retracted shoulders." PMC12225233 defines lockout as "the lifter's trunk reaches the vertical position … with the bar positioned at its highest point." Verified in RAG docs.

---

### Notes / honest gaps

- **Deadlift lumbar flexion (deadlift_lumbar_flexion)** is the clinically most important
  deadlift fault but is only **low-observability** from monocular 33-landmark MediaPipe pose
  (no spine landmarks); the heuristic above is an explicit proxy, not a precise measure. This
  is flagged rather than papered over.
- The squat **knees_forward** citation (PMC6523035) is a *forward-step-lunge* study; its
  knee-in-front-vs-behind-toes contrast is the cleanest available quantification of the
  anterior-knee-translation load mechanism and transfers directly, but it is not a pure
  back-squat paper. Noted for transparency.
- All citation_support strings above were extracted from sources actually read in this
  session (RAG docs under `data/rag/docs/`, plus the Hartmann 2013 PubMed abstract fetched
  via WebFetch). No UNVERIFIED entries.


---

## Group B — Upper-body push — Push-up, Overhead Press

Pose model: MediaPipe Pose, 33 landmarks, monocular, normalized image coords (x,y in
[0,1] + z depth + visibility). Landmark indices per shared_context.md.

Covers: **Push-up**, **Overhead Press**.

---

### Push-up

Rep phases: **setup/top plank** → **descent (eccentric)** → **bottom** → **ascent
(concentric)** → **lockout/top**. Detection assumes the subject is horizontal; the
`side` view (camera in the sagittal plane, perpendicular to the body's long axis) is the
primary useful view. Hand/elbow-width faults need a `front`/`rear` view down the body's
long axis (camera at head or foot end, slightly elevated).

#### Hip sag / lost plank alignment (lumbar hyperextension)

- **fault_id**: `pushup_hip_sag`
- **fault_name**: Hip sag / broken plank line
- **description**: The hips drop below (or pike above) the straight shoulder-hip-ankle
  line so the torso and legs no longer form one rigid plank.
- **detection_heuristic**: Side view. Fit the line through shoulder midpoint
  (mid 11/12) and ankle midpoint (mid 27/28); measure signed perpendicular offset of the
  hip midpoint (mid 23/24) from that line, normalized by the shoulder→ankle length. Flag
  `sag` when the hip sits toward the ground (larger `y`) by offset > ~0.06 of body length;
  flag `pike` when hip rises above the line by the same margin. Equivalent: hip angle
  (shoulder–hip–ankle) departs from 180° by > ~12°.
- **observability**: high — `side` (sagittal). Near-`none` from `front`/`rear` (offset is
  in-plane and collapses).
- **biomechanical_rationale**: Loss of the neutral plank shifts load from the abdominal
  wall to passive lumbar structures, raising L4–L5 spine load; excessive or repeated
  lumbar loading is an injury concern in push-up variants.
- **citation**: Freeman S, Karpowicz A, Gray J, McGill S. Med Sci Sports Exerc (2006).
  DOI 10.1249/01.mss.0000189317.08635.1b.
- **citation_support**: The study "quantify[ied] the normalized amplitudes of the
  abdominal wall and back extensor musculature" and "their impact on spinal loading by
  calculating spinal compression and torque generation in the L4-5 area," finding push-up
  form drives large differences in L4–L5 spine compression (the one-arm push-up produced
  "the highest spine compression"). This establishes that push-up trunk posture governs
  lumbar load; a sagging (hyperextended) trunk is the posture that raises passive lumbar
  loading. Note: the paper measured spine load by variant, not sag angle directly, so the
  sag→load link is inferred from the loading mechanism it quantifies.

#### Shallow depth / incomplete elbow ROM

- **fault_id**: `pushup_shallow_depth`
- **fault_name**: Shallow depth (partial rep)
- **description**: The elbows do not bend far enough at the bottom, so the chest never
  approaches the floor and the rep uses only part of the range.
- **detection_heuristic**: Side view. Elbow angle = angle at elbow (shoulder 11 → elbow
  13 → wrist 15), take left/right whichever is more visible. At the bottom frame (wrist–
  shoulder vertical distance minimized), flag `shallow` when min elbow flexion angle
  > ~100–110° (a full rep reaches roughly ≤90°). Optionally corroborate with small
  shoulder-to-ground travel between top and bottom.
- **observability**: high — `side` / `front_oblique`; medium from a true head-on
  `front`/`rear` where the elbow angle foreshortens.
- **biomechanical_rationale**: Cutting the range short reduces the mechanical work and the
  target-muscle stimulus, because both external load and scapular-stabilizer demand rise
  as the elbow travels through deeper flexion.
- **citation**: San Juan JG, Suprak DN, Roach SM, Lyda M. BMC Musculoskelet Disord (2015)
  PMC4327800.
- **citation_support**: Measuring elbow kinematics in 5° increments across the push-up
  range, vertical ground-reaction force "displayed a significant linear decrease across
  the ROM" and was "highest during the traditional PUP at 90° … of elbow flexion and
  lowest at 20°," while serratus anterior and other muscle EMG rose across elbow
  extension. Deeper elbow flexion = higher force/demand, so a shallow rep that never
  reaches the deep-flexion positions forfeits the largest portion of the stimulus.

#### Elbows flared / hands too wide

- **fault_id**: `pushup_elbow_flare`
- **fault_name**: Flared elbows / excessive hand width
- **description**: The hands are placed well outside the shoulders and the upper arms
  abduct far from the torso, so the elbows point outward rather than tracking back.
- **detection_heuristic**: Best proxy uses a `front`/`rear` view down the body: hand-width
  ratio = wrist-to-wrist distance (15↔16) / shoulder width (11↔12); flag when ratio
  > ~1.6. If the upper arm is visible, corroborate with the trunk-to-upper-arm angle
  (torso vector shoulder→hip vs upper-arm vector shoulder→elbow) exceeding ~65°. From a
  pure `side` view this is largely unobservable (both wrists overlap).
- **observability**: medium — `front` / `rear` (down the long axis); low/`none` from
  `side`.
- **biomechanical_rationale**: Hand placement measurably changes elbow intersegmental
  loads — including the valgus (medial-ligament) torque — so an abnormally wide or
  displaced hand position shifts joint loading away from the trained pattern and can raise
  medial-elbow stress.
- **citation**: Donkers MJ, An KN, Chao EY, Morrey BF. J Biomech (1993).
  DOI 10.1016/0021-9290(93)90026-b.
- **citation_support**: Recording elbow forces in six hand positions, "peak forces exerted
  on the elbow joint along the forearm axis averaged 45% of the body weight for the
  'normal' hand position and were significantly decreased if hands were positioned either
  'apart' or 'superior'," while "the maximum valgus torque at the elbow opposed by the
  medial ligamentous structure … was significantly increased if the hand was positioned
  superiorly" (and rose 42% one-handed). Hand position therefore strongly modulates elbow
  joint loading, justifying a rule that flags hand placement deviating from a
  shoulder-width baseline.

#### Scapular winging / absent scapular control

- **fault_id**: `pushup_scapular_winging`
- **fault_name**: Scapular winging / incomplete scapular control
- **description**: The medial border/inferior angle of the shoulder blade lifts off the
  ribcage (winging) instead of the serratus anterior holding the scapula flat/protracted.
- **detection_heuristic**: No reliable monocular signal — MediaPipe's 33 landmarks do not
  include scapular border points, so scapular tilt/rotation/winging cannot be measured
  directly. Weak indirect proxy only: gross upper-back rounding from a `rear` view via
  relative shoulder-blade region shape, which is not trustworthy. Recommend NOT emitting a
  confident verdict; surface as informational.
- **observability**: low/`none` — not resolvable with the 33-landmark model from any view.
- **biomechanical_rationale**: Serratus anterior weakness lets the scapula wing and
  over-internally-rotate/anteriorly tilt, reducing subacromial space and predisposing to
  shoulder impingement — which is exactly why push-up/push-up-plus is prescribed to train
  the serratus.
- **citation**: Lee S, Lee D, Park J. J Phys Ther Sci (2013) PMC3820220;
  corroborated by Abdollahi S et al. J Orthop Surg Res (2025) PMC12366113.
- **citation_support**: PMC3820220 states "Weakening of the serratus anterior muscle leads
  to excessive activation of the upper trapezius … reducing the dynamic stability of the
  scapula," which drives "a clash between the subacromion and the head of the humerus";
  PMC12366113 similarly notes fatigue of the serratus anterior yields "increased internal
  rotation and decreased posterior tilt of the scapula." The fault is biomechanically real
  and important, but honestly not monocular-observable, hence observability `none`.

#### Forward head / neck drop

- **fault_id**: `pushup_head_drop`
- **fault_name**: Forward head / neck drop
- **description**: The head juts forward or drops so the neck leaves the straight line of
  the spine, often as the chin reaches for the floor ahead of the chest.
- **detection_heuristic**: Side view. Neck-line angle = angle at shoulder between the
  ear→shoulder vector (ear 7/8 → shoulder 11/12) and the shoulder→hip torso vector; flag
  `head_drop` when the head deviates below the torso line (nose/ear `y` well beneath the
  shoulder–hip line) by > ~15°, or when nose 0 sits clearly ahead of the shoulder along
  the body axis.
- **observability**: medium — `side` / `front_oblique`; low from `front`/`rear`.
- **biomechanical_rationale**: Correct push-up form keeps "the head, spine and pelvis …
  in a straight line, in a neutral state"; a dropped/forward head places the cervical
  spine in sustained non-neutral loading and is a marker of the same protraction/forward-
  posture pattern linked to shoulder impingement.
- **citation**: Lee S et al. J Phys Ther Sci (2013) PMC3820220 (form/neutral-alignment
  standard); mechanism corroborated by Al Hammadi MI et al. Cureus (2025) PMC12514857.
- **citation_support**: PMC3820220's protocol required that "the head, spine, and pelvis
  were positioned in a straight line, in a neutral state" with "the cervical vertebrae in
  a neutral position," defining neutral cervical alignment as correct form; PMC12514857
  lists "forward head posture" among the postural factors that "interfere with scapular
  movement … leading to a reduction in subacromial space," supplying the injury rationale.
  Direct push-up-specific cervical-injury evidence is thin, so this rule leans on the
  alignment standard plus the general forward-posture→impingement mechanism.

---

### Overhead Press

Rep phases: **rack/start (bar at shoulders)** → **press (concentric)** →
**lockout/top (bar overhead)** → **lowering (eccentric)**. Performed standing or seated.
The `side` view (sagittal) is primary for spine/lockout/bar-path faults; a `front` view is
needed for left/right asymmetry.

#### Excessive lumbar hyperextension / back-lean

- **fault_id**: `ohp_lumbar_hyperextension`
- **fault_name**: Excessive back-lean / lumbar hyperextension (rib flare)
- **description**: The lifter leans the trunk backward and arches the lower back to press
  the bar, driving the ribcage up and the shoulders behind the hips.
- **detection_heuristic**: Side view. Torso-lean angle = angle of the hip→shoulder vector
  (mid 23/24 → mid 11/12) relative to true vertical; flag when the shoulders travel
  posterior to the hips by > ~10–15° at/after mid-press (a small forward torso is normal;
  a backward lean is the fault). Corroborate with increased hip-forward translation
  (hip `x` ahead of ankle `x`).
- **observability**: high — `side` (sagittal); low from `front`/`rear`.
- **biomechanical_rationale**: Substituting a lumbar-extension backbend for shoulder ROM
  concentrates load on the lower back; historically this exact compensation caused a
  cluster of lower-back injuries in the pressed lift.
- **citation**: Soriano MA, Suchomel TJ, Comfort P. "Weightlifting Overhead Pressing
  Derivatives: A Review of the Literature." Sports Med (2019) PMC6548056.
- **citation_support**: The review recounts that the competition press degenerated into
  the "continental press," "characterised by a considerable quick backbend before the
  lift," and that "a long list of lower back injuries due to the accentuated backbend
  drove the IWF to eliminate the press from all future competitions." An accentuated
  backbend (lumbar hyperextension) is directly named as the mechanism of lower-back injury
  in overhead pressing.

#### Incomplete elbow lockout

- **fault_id**: `ohp_incomplete_lockout`
- **fault_name**: Incomplete lockout at the top
- **description**: At the top the elbows are not fully straightened, so the rep stops
  short of a stable overhead lockout.
- **detection_heuristic**: Side or front view. Elbow angle (shoulder 11/12 → elbow 13/14
  → wrist 15/16) at the highest bar position; flag `incomplete_lockout` when peak elbow
  extension < ~160° (full lockout ≈ 175–180°). Take the worse of the two arms.
- **observability**: high — `side` / `front`.
- **biomechanical_rationale**: The elbow extensors are what finish the lift, becoming
  dominant near full extension; a rep that never reaches elbow extension omits the
  lockout that defines a completed press and leaves the load unsupported over the joint.
- **citation**: Evangelista P, Rum L, Picerno P, Biscarini A. "Decoding the Contribution of Shoulder and Elbow
  Mechanics … Sticking Region in Bench and Overhead Press." J Funct Morphol Kinesiol (2025) PMC12372072, DOI 10.3390/jfmk10030322.
- **citation_support**: The link-chain model found "elbow extensors contributed minimally
  during early lift phases but became dominant near full extension," and the lift is
  defined as complete only "when the elbow is fully extended … and the barbell reaches its
  final position," motivating training strategies that "target … elbow strength near
  lockout." Full elbow extension is therefore the mechanical definition of a finished rep,
  and stopping short is a genuine ROM failure.

#### Forward head at lockout

- **fault_id**: `ohp_forward_head`
- **fault_name**: Forward head posture at lockout
- **description**: The head juts forward of the shoulder line at/near lockout.
- **detection_heuristic**: Side view. Horizontal offset of ear (7/8) ahead of the shoulder
  (11/12) along the anterior axis, normalized by shoulder width; flag when the ear is
  anterior by > ~0.3 shoulder-widths.
- **observability**: medium–high — `side` (the cue is a sagittal-plane offset); low from
  `front`.

> **WITHDRAWN sub-criterion — bar path ahead of midline.** This rule was originally written
> with a second cue: *"(b) Bar-forward: at lockout, wrist (15/16) horizontal offset anterior
> to shoulder; flag when the wrist is not stacked roughly vertically over the shoulder
> (offset > ~0.3 shoulder-widths)."* That sub-criterion is **withdrawn** (2026-07-25) and its
> `wrist_forward_offset` metric deleted from the implementation, for three reasons:
>
> 1. **It re-describes back lean.** Referencing bar path to the *shoulder* conflates it with
>    trunk lean: a back lean moves the shoulders posterior while the bar stays over the base
>    of support, so "bar anterior of the shoulders" **is** the mechanical signature of a back
>    lean. Empirically, a pure back-lean rep emitted this fault at severity 1.0 / confidence
>    1.0, outranking `ohp_lumbar_hyperextension` (0.41) — the fault that actually occurred.
> 2. **The correct reference frame is not measurable.** The description's own wording
>    ("stacked over the shoulder **and mid-foot**") means the bar should be referenced to
>    mid-foot, which would require an invented mid-foot proxy — forbidden by this project's
>    every-threshold-literature-backed premise.
> 3. **The citation does not support it.** Abdelraouf et al. (PMC13116542) defines forward
>    head posture by **craniovertebral angle**, an ear-relative-to-shoulder measure — so
>    ear-vs-shoulder referencing is exactly what the literature measures. Nothing in that
>    source, or in PMC13086636 / PMC12514857, speaks to bar position at all.
>
> **Open spec question:** does the OHP rule set want a genuine bar-path fault? If so it needs
> (a) a base-of-support reference that MediaPipe can actually resolve and (b) its own
> citation. Until both exist, the rule is a forward-head cue only. This is a withdrawal
> pending a decision, not a silent reinterpretation.
- **biomechanical_rationale**: A forward-head / kyphotic overhead posture reduces the
  scapular upward rotation and shoulder flexion available and narrows the subacromial
  space, both cutting achievable overhead ROM and raising impingement risk; fatigue in
  pressing is shown to push the head into exactly this forward posture.
- **citation**: Abdelraouf OR et al. J Clin Med (2026) PMC13116542; mechanism from Gregori
  P et al. J Exp Orthop (2026) PMC13086636 and Al Hammadi MI et al. Cureus (2025)
  PMC12514857.
- **citation_support**: PMC13116542 found high-load overhead-press training to failure
  significantly reduced the craniovertebral angle (defining "a craniovertebral angle …
  less than 48 degrees … as forward head posture"), i.e. pressing drives measurable
  forward head. PMC13086636 reports "greater thoracic kyphosis is associated with …
  reduced shoulder abduction … and flexion," and PMC12514857 notes forward head / thoracic
  kyphosis "reduce[s] the subacromial space," giving both the performance (ROM) and injury
  (impingement) rationale.

#### Asymmetric press

- **fault_id**: `ohp_asymmetric_press`
- **fault_name**: Asymmetric press (one side leading)
- **description**: One arm presses higher or faster than the other, so the bar/hands
  finish at uneven heights and the shoulder girdle is tilted.
- **detection_heuristic**: Front view. Vertical wrist-height difference
  |y(15) − y(16)| normalized by shoulder width; flag `asymmetric` when > ~0.15 at/near
  lockout, and/or left-vs-right elbow-extension-angle difference > ~15°. Corroborate with
  a shoulder-line tilt (11↔12 not horizontal).
- **observability**: medium — `front` / `rear` (needs the frontal plane); low from `side`
  (arms overlap). Underlying scapular contribution is not directly trackable.
- **biomechanical_rationale**: A persistent side-to-side height/timing difference reflects
  shoulder-girdle asymmetry (scapular dyskinesis), which is associated with impaired
  scapular stability and elevated shoulder-injury risk.
- **citation**: Abdelraouf OR et al. J Clin Med (2026) PMC13116542.
- **citation_support**: The study operationalizes shoulder-girdle asymmetry via the
  scapular balance angle and lateral scapular slide, defining scapular dyskinesis as "a
  difference between the two sides of the body of more than 7 degrees in the scapular angle
  or more than 1.5 cm in the lateral shift distance," and found high-load press training to
  failure "resulted in a more protracted scapular position and shoulder girdle asymmetry
  (scapular dyskinesis)." Measurable left/right asymmetry is thus a validated marker of a
  clinically meaningful fault (the wrist-height proxy stands in for the scapular measure,
  which MediaPipe cannot capture directly).

#### Insufficient overhead elevation (bar not fully overhead)

- **fault_id**: `ohp_insufficient_elevation`
- **fault_name**: Insufficient overhead elevation / short press
- **description**: The bar/hands never travel to a true overhead position above the head —
  the press ends around forehead/eye level (often a shrug-and-stall) instead of arms
  overhead.
- **detection_heuristic**: Side or front view. At the top frame, flag
  `insufficient_elevation` when the wrist (15/16) does not clear the head — i.e. wrist `y`
  is not above nose `y` (0) by at least ~0.5 head-heights — despite the rep being counted.
  Distinguish from `ohp_incomplete_lockout`: here the wrist height itself is low even if
  the elbow angle looks partly extended.
- **observability**: medium–high — `side` / `front`.
- **biomechanical_rationale**: A complete overhead press requires simultaneous scapular
  upward rotation, humeral abduction/flexion, and elbow extension to bring the load
  genuinely overhead; ending short of overhead means the target motion (and its shoulder
  musculature) was never fully loaded.
- **citation**: Coratella G et al. Front Physiol (2022) PMC9354811; end-position
  corroborated by Evangelista P et al. J Funct Morphol Kinesiol (2025) PMC12372072.
- **citation_support**: PMC9354811 states that "the simultaneous scapular upward rotation
  …, together with the humerus abduction and elbow extension … makes the overhead press
  suitable to stimulate upper trapezius, deltoids and triceps," defining the full overhead
  end position as the combination of scapular upward rotation + humeral elevation + elbow
  extension; PMC12372072 defines the lift as complete only when "the barbell reaches its
  final position" at full extension. A press that stalls below overhead has not achieved
  this end position.

---

### Notes / gaps

- **Push-up scapular winging (`pushup_scapular_winging`)** is marked observability
  `none`: it is a real, well-cited fault, but MediaPipe's 33 landmarks contain no scapular
  border points, so it cannot be measured monocularly. Listed honestly rather than faked.
- **Hip-sag citation** (McGill 2006) measures L4–L5 spine load by push-up variant, not by
  sag angle directly; the sag→lumbar-load link is inferred from the loading mechanism the
  paper quantifies (stated in citation_support). All other citation_support items are
  direct findings from sources read in full.
- Author-name attributions for PMC6548056 (WOPD review) and PMC12372072 (sticking-region
  model) were originally approximate first-author guesses from the RAG text (the docs did
  not expose a clean author line). **Both have since been resolved against
  `data/paper_metadata.json`** and the §6 reference index now carries the authoritative
  strings: PMC6548056 = Soriano MA, Suchomel TJ, Comfort P; PMC12372072 = Evangelista P,
  Rum L, Picerno P, Biscarini A. The §6 index is the source of truth for author strings —
  where an inline rule entry abbreviates, it must abbreviate the *indexed* first author.
- No web search was needed: RAG coverage for both movements was sufficient to back every
  emitted rule.


---

## Group C — Upper-body pull — Row, Band Pull Apart

Detection model: MediaPipe Pose, 33 landmarks, normalized image coords (x,y in [0,1], z depth,
visibility). Monocular single camera. view_type estimated per rep (side / front / rear /
front_oblique / rear_oblique). Landmark indices per shared_context.md.

---

### Row (bent-over / barbell row)

Rep phases: **setup (hip-hinge, torso fixed) → concentric pull (bar/hands toward torso, scapular
retraction) → peak hold (bar at abdomen, scapulae retracted) → eccentric lower → return.**

#### Torso rising / loss of hip-hinge

| field | value |
|---|---|
| **fault_id** | `torso_rising_hip_hinge_loss` |
| **fault_name** | Torso rising (loss of hip-hinge) |
| **description** | The trunk drifts from its hinged (near-horizontal) setup angle toward upright across the concentric pull, using hip extension to help move the load. |
| **detection_heuristic** | Torso vector = midpoint(shoulders 11,12) → midpoint(hips 23,24). Compute `trunk_angle_from_horizontal` at setup baseline and at peak pull. Flag if `trunk_angle_peak - trunk_angle_setup > 15deg` (torso becoming more upright). Direction: increasing angle = fault. |
| **observability** | high — side / front_oblique / rear_oblique (needs a lateral component to read trunk pitch). Low from pure front/rear. |
| **biomechanical_rationale** | The bent-over row is a trunk-stabilization task in which the erector spinae must isometrically hold the hinged position; letting the torso rise substitutes hip/lumbar momentum for back-muscle work, reducing target loading and creating a kinetic-chain "catch-up" pattern linked to injury when proximal stability breaks down. |
| **citation** | Saeterbakken A et al. Int J Sports Med (2015) PMID 26134664; Owens LP et al. IJSPT / Int J Sports Phys Ther (2026) PMC13232157. |
| **citation_support** | Saeterbakken: the free-weight bent-over row produced greater erector spinae EMG than the machine row both bilaterally and unilaterally — i.e. the hinged free-weight row imposes a high, sustained trunk-extensor stabilizing demand that a rising torso abandons. Owens: describes kinetic-chain (KC) sequencing where "breaks in efficient KC sequencing require distal segments to increase functional capacity... described as the 'catch-up' phenomenon," and uses a trunk-parallel-to-floor prone position specifically to control trunk posture during rowing. |

#### Rounded (flexed) thoracolumbar spine

| field | value |
|---|---|
| **fault_id** | `rounded_thoracolumbar_spine` |
| **fault_name** | Rounded (flexed) spine during the pull/hold |
| **description** | The back loses its neutral slight-arch and flexes (rounds) forward under load instead of staying flat through the hinge. |
| **detection_heuristic** | Proxy spinal-line curvature: three-point angle at mid-spine using shoulder-midpoint(11,12), a synthesized mid-trunk point = 0.5·(shoulder_mid + hip_mid), and hip-midpoint(23,24); alternatively track shoulder→hip line vs a straight setup reference. Flag flexion if the shoulder-midpoint drops below the straight shoulder–hip line by a normalized sag > 0.04 (i.e. upper back bows toward the floor). Monocular proxy only — true spinal flexion is not directly measurable. |
| **observability** | medium — side / front_oblique / rear_oblique; the sag proxy is coarse. Low from pure front/rear. |
| **biomechanical_rationale** | Because the hinged row loads the lumbar erector spinae heavily, rowing with a flexed spine shifts load from the actively-contracting extensors onto passive spinal structures (discs/ligaments), the classic mechanism for lower-back strain under a loaded hinge. |
| **citation** | Saeterbakken A et al. Int J Sports Med (2015) PMID 26134664; bent-over row Wikipedia (descriptive, supplementary) — `data/rag/docs/row_wiki.txt`. |
| **citation_support** | Saeterbakken: free-weight bent-over row elicited the greatest erector spinae EMG activity, establishing that the lumbar spine is under high extensor load in the hinged position (the peer-reviewed load claim). Wiki (descriptive only): recommends "maintaining an arch (a slight concavity) in the spine for a healthy lower back," and notes contraction of rectus abdominis "would cause the back to round and de-activate the lower back." NOTE: no RAG source experimentally tests spinal-flexion injury under row load; the injury inference rests on the documented high extensor load plus the descriptive neutral-spine cue. |

#### Incomplete ROM / insufficient retraction

| field | value |
|---|---|
| **fault_id** | `incomplete_retraction_rom` |
| **fault_name** | Incomplete ROM (bar/hand not reaching torso; retraction insufficient) |
| **description** | The pull stops short — the bar/hands never reach the torso and the scapulae never fully retract, so the top (peak-contraction) portion of the range is skipped. |
| **detection_heuristic** | (a) Pull depth: minimum normalized distance from wrist(15/16) to hip(23/24) or to torso line across the rep; flag if `min_wrist_to_torso_dist > 0.12` (hand never approaches the abdomen). (b) Elbow flexion at peak: `elbow_angle (11-13-15 / 12-14-16) > 100deg` at the top = pull not completed. Direction: larger residual distance / larger elbow angle = fault. |
| **observability** | high — side / oblique for pull depth; front/rear also usable for elbow travel. |
| **biomechanical_rationale** | The upper half of the row ROM (bar approaching the torso) is where the latissimus dorsi is most excited, and full coordinated scapular retraction is what loads the mid-back and optimizes glenohumeral force transmission; cutting the pull short forfeits peak lat and scapular-retractor loading. |
| **citation** | Fischer J et al. J Electromyogr Kinesiol (2025) PMID 40513198; Padovan R et al. J Funct Morphol Kinesiol (2025) PMC12821611. |
| **citation_support** | Fischer (prone barbell row, 3 ROMs): "The LD showed significantly higher mean muscle excitation in the upper-half ROM compared to both the lower-half ROM (p < 0.001) and full ROM (p < 0.001)" — the top of the pull (bar near torso) drives peak lat excitation. Padovan: describes the row as driven by "scapular retraction, external rotation, and posterior tilt [which] contributes to optimizing glenohumeral alignment and force transmission," and the concentric endpoint "defined when the handle reached the abdominal target." |

#### Momentum / jerk (body English)

| field | value |
|---|---|
| **fault_id** | `momentum_jerk_body_english` |
| **fault_name** | Momentum / jerk (using body English) |
| **description** | The bar is yanked with an explosive jerk / whole-body heave rather than a controlled pull, creating a velocity spike and slack at the bottom. |
| **detection_heuristic** | Frame-to-frame wrist(15/16) velocity and acceleration along the pull axis; flag if peak concentric wrist acceleration exceeds ~3× the rep's median concentric acceleration, OR if a simultaneous trunk-angle velocity spike co-occurs with the wrist spike (heave). Direction: sharp acceleration transient = fault. |
| **observability** | medium — any view with the pulling wrist visible; requires a stable frame rate. |
| **biomechanical_rationale** | Accelerating the load unloads the muscle: a ballistic concentric raises peak concentric force briefly but sheds mechanical tension elsewhere (notably reduced eccentric demand), and the jerk creates momentary slack/weightlessness that removes controlled tension from the target muscles and can shock-load the spine at the bottom. |
| **citation** | Padovan R et al. J Funct Morphol Kinesiol (2025) PMC12821611; bent-over row Wikipedia (descriptive, supplementary) — `data/rag/docs/row_wiki.txt`. |
| **citation_support** | Padovan: "Accelerating a given load during dynamic contractions increases force requirements during the concentric phase, whereas the same load imposes lower mechanical demands during the eccentric phase" — momentum redistributes/reduces loading away from the controlled tension the exercise intends; their protocol standardizes a controlled 2 s concentric / 2 s eccentric tempo. Wiki (descriptive): advises "a slow tempo and avoiding jerking... prevents momentum from creating momentary weightlessness or slack in the muscles during the ascent, or... a jerking catch on the bottom of the lift." |

#### Asymmetric pull

| field | value |
|---|---|
| **fault_id** | `asymmetric_pull` |
| **fault_name** | Asymmetric pull (one side higher / leading) |
| **description** | One arm pulls higher or farther than the other, tilting the shoulder line and introducing trunk rotation. |
| **detection_heuristic** | At peak: compare left vs right elbow height `|y13 - y14|` and wrist-to-hip travel `| dist(15,23) - dist(16,24) |`; also shoulder-line tilt `|y11 - y12|` relative to setup. Flag if elbow-height asymmetry > 0.05 normalized OR shoulder-line tilt increases > 0.04 vs setup. Direction: growing left/right difference = fault. |
| **observability** | high — front / rear (both shoulders and elbows visible); low from pure side view. |
| **biomechanical_rationale** | An asymmetric pull rotates the trunk and shifts the row toward a unilateral pattern, which markedly raises anti-rotation core (external oblique) demand and uneven spinal loading, and prevents balanced bilateral scapular retraction. |
| **citation** | Saeterbakken A et al. Int J Sports Med (2015) PMID 26134664; Padovan R et al. J Funct Morphol Kinesiol (2025) PMC12821611. |
| **citation_support** | Saeterbakken: "unilateral performance of exercises activated the external oblique more than bilateral performance, regardless of exercise" — an unintended one-sided (asymmetric) pull imposes the higher anti-rotation/oblique load characteristic of unilateral rowing. Padovan: frames correct rowing as "coordinated scapulothoracic motion" and bilateral scapular adduction to the abdominal target — asymmetry breaks that coordinated bilateral retraction. |

---

### Band Pull Apart

Rep phases: **start (arms extended forward, band at chest height, hands together) → concentric
horizontal abduction (pull apart + scapular retraction) → peak (band to chest, hands maximally
spread) → eccentric return.**

#### Shrugging (upper-trap dominance)

| field | value |
|---|---|
| **fault_id** | `shrugging_upper_trap_dominance` |
| **fault_name** | Shrugging (shoulders rise toward ears) |
| **description** | The shoulders elevate toward the ears during the pull-apart, signaling upper-trapezius over-activation instead of mid/lower-trap retraction. |
| **detection_heuristic** | Shoulder-to-ear vertical gap: `gap = y_shoulder(11/12) - y_ear(7/8)` (image y grows downward, so a smaller gap = shoulder risen). Compute at setup baseline and at peak; flag shrug if `gap_peak < gap_setup - 0.03` (shoulders elevate) on either side. Direction: shrinking shoulder-ear gap = fault. |
| **observability** | high — front / rear (both shoulders + ears visible). |
| **biomechanical_rationale** | Band pull-apart is meant to preferentially load the middle/lower trapezius and posterior rotator cuff with LOW upper-trap contribution; upper-trapezius dominance is counterproductive in shoulder-pain/impingement populations because it increases anterior scapular tilt and can compromise subacromial space. |
| **citation** | Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI 10.26603/001c.33026; Camargo PR & Neumann DA, Braz J Phys Ther (2019) 23(6):467–475, PMC6849087, DOI 10.1016/j.bjpt.2019.01.011. |
| **citation_support** | Fukunaga: "it has been suggested that exercises should aim to preferentially target the middle trapezius, lower trapezius, and posterior RTC, with lower contributions from the upper trapezius and deltoid muscles" — a shrug inverts the intended UT-low pattern. Camargo & Neumann: "Exercises that increase the strength or relative activation of the upper trapezius may be counterproductive in many patients with shoulder pain, especially those with symptoms of impingement," because "the upper trapezius naturally causes an increased anterior tilt of the scapula, which may compromise the volume within the subacromial space." |

#### Incomplete horizontal-abduction ROM

| field | value |
|---|---|
| **fault_id** | `incomplete_horizontal_abduction_rom` |
| **fault_name** | Incomplete ROM (hands don't reach full spread / band to chest) |
| **description** | The hands stop short of full horizontal abduction — the band never reaches the chest and the arms never fully spread. |
| **detection_heuristic** | Peak wrist separation: `wrist_spread = dist(wrist15, wrist16)`, normalized by shoulder width `dist(11,12)`. Flag if `wrist_spread_peak / shoulder_width < 1.6` (arms not carried past the torso line), and/or elbow-extension check `elbow_angle > ~150deg` maintained (bent-elbow curl-style cheat = fault). Direction: smaller spread ratio = fault. |
| **observability** | high — front / rear. |
| **biomechanical_rationale** | Muscle activity in the pull-apart rises with the range covered against resistance; stopping short of full horizontal abduction forfeits the higher scapular-muscle activation seen at greater excursion and fails to reach end-range scapular retraction. |
| **citation** | Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI 10.26603/001c.33026. |
| **citation_support** | Fukunaga: peak muscle activity spanned "15.3% to 72.6% of MVC across muscles and exercise conditions," and the diagonal-up (largest-excursion, against-gravity) direction produced the highest trapezius activity — "the diagonal up movement showing the highest shoulder-girdle muscle activity is understandable as the arm is moving against gravity, resulting in higher overall load" — i.e. covering more range against the band drives higher target activation, which a truncated pull loses. |

> **NOTE — direction inversion in this rule's elbow cue, corrected in implementation
> (2026-08-09).** The `detection_heuristic` above reads "elbow-extension check `elbow_angle >
> ~150deg` maintained (bent-elbow curl-style cheat = fault)". Read literally, `> 150°` — nearly
> *straight* arms — is the fault, contradicting the parenthetical in the same sentence. The
> parenthetical is right: a bent-elbow cheat means a *smaller* elbow angle.
> `src/pose/movements/band_pull_apart.py` implements **`min_elbow_angle < 150°`**. The number
> `150` is unchanged and remains FROM THE SPEC; only the comparison direction is corrected.
> Corroboration beyond the parenthetical: the KG names this fault `Bent Elbows`
> (`scripts/knowledge/stub_general_movements_v3.py:85`), and Fukunaga's rationale — more range
> covered against the band drives higher activation — is a range argument that bending the elbows
> shortens.

#### Loss of scapular retraction

| field | value |
|---|---|
| **fault_id** | `loss_of_scapular_retraction` |
| **fault_name** | Loss of scapular retraction (arms-only pull) |
| **description** | The band is spread with the arms while the scapulae stay protracted (shoulders don't draw back/together), making it a glenohumeral-only movement. |
| **detection_heuristic** | Monocular proxy — scapular retraction is not directly visible from the front. Proxy: from a REAR/rear_oblique view track inter-shoulder width `dist(11,12)`; genuine retraction slightly narrows the posterior shoulder points as scapulae adduct, whereas pure horizontal abduction without retraction keeps them unchanged. Flag `no_retraction` if wrist spread increases > threshold while `dist(11,12)` change < 0.01 (arms move, scapulae don't). Coarse proxy only. |
| **observability** | low–medium — rear / rear_oblique preferred; essentially none from a pure front view (scapulae occluded). |
| **biomechanical_rationale** | The therapeutic target of the pull-apart is middle/lower-trapezius scapular retraction; if the scapulae never retract, the periscapular retractors are bypassed and the exercise loses its scapular-stabilizer training effect. |
| **citation** | Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI 10.26603/001c.33026. |
| **citation_support** | Fukunaga: middle-trapezius activity was significantly driven by the retraction-oriented directions (highest in diagonal-up/horizontal vs diagonal-down), and the exercise is framed around recruiting "periscapular muscles" for "scapular stabilization" — retraction is the mechanism; an arms-only pull removes it. Honest limitation: scapular position itself is not reliably recoverable from monocular front-view pose, hence low observability. |

> **NOTE — implemented as a permanently-silent rule (2026-08-09).** `rule_loss_of_scapular_retraction`
> is registered in `src/pose/movements/band_pull_apart.py` and always returns `[]`, following
> `pushup.rule_scapular_winging`. Two independent defects in the heuristic above, either
> disqualifying:
>
> 1. **The fire condition is a null-detection.** It fires when `dist(11,12)` *fails to change*
>    ("change < 0.01"), so a steady frame, a partially occluded frame, and a genuine non-retraction
>    are indistinguishable. Every correct rep that holds the shoulders stable would fire it.
> 2. **The metric is confounded with what it must be independent of.** MediaPipe's shoulder
>    landmark is a *glenohumeral* point that moves with the humerus, and horizontal abduction is
>    exactly the humeral motion in question, so `dist(11,12)` cannot attribute a narrowing to
>    scapular adduction rather than arm position. Root cause: MediaPipe Pose has no scapular
>    landmarks.
>
> Separately, `0.01` carries no citation; Fukunaga supplies no landmark-displacement magnitude.
>
> **A NOTE and not a WITHDRAWN blockquote, deliberately.** Fukunaga genuinely backs retraction as the
> training mechanism, so the fault is real and cited and it is the *sensing* that fails — the
> `pushup.rule_scapular_winging` case, not the OHP-bar-path / deadlift-bar-drift case. The KG is
> not the gap either: `Band Pull Apart:Insufficient Scapular Retraction` resolves with a non-empty
> `causes` bucket. The metric is the gap.

#### Trunk-extension compensation (leaning back)

| field | value |
|---|---|
| **fault_id** | `trunk_extension_compensation` |
| **fault_name** | Trunk-extension compensation (leaning back) |
| **description** | The lifter leans/whips the trunk backward into lumbar extension to fling the band apart instead of using the shoulder-girdle muscles. |
| **detection_heuristic** | Torso vector midpoint(shoulders 11,12) → midpoint(hips 23,24); compute lean from vertical. Flag if `trunk_lean_backward > 10deg` beyond setup baseline OR a trunk-angle velocity spike co-occurs with the concentric pull (whip). Direction: increasing backward lean synchronized with the pull = fault. |
| **observability** | high — side / oblique (needs lateral component to read trunk pitch); low from pure front/rear. |
| **biomechanical_rationale** | A standing pull-apart should be driven by horizontal abduction / scapular retraction; substituting trunk extension recruits lumbar momentum to move the band, unloading the intended scapular muscles and adding uncontrolled lumbar-extension load. |
| **citation** | Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI 10.26603/001c.33026. |
| **citation_support** | Fukunaga establishes the pull-apart as a standing horizontal-abduction / diagonal scapular exercise whose load should come from the shoulder girdle; the paper notes trunk/hip extension can be *engaged* deliberately but the target muscles are the periscapular/RTC group — so a backward trunk whip that replaces (rather than stabilizes for) horizontal abduction diverts the movement off its intended muscles. NOTE: no RAG/EMG source directly quantifies a "lean-back cheat" injury; the rule is a controlled-execution/performance-loss rule grounded in the exercise's intended horizontal-abduction mechanics, so the compensation framing is partly inferential (observability of the fault is high, but its harm claim is supported indirectly). |

---

### Verification / gaps summary

- All Row rules are anchored to peer-reviewed RAG docs (PMID 26134664, PMC12821611, PMID 40513198,
  PMC13232157); `row_wiki.txt` is used only as supplementary descriptive support, never as sole
  backing for an injury claim.
- `rounded_thoracolumbar_spine`: the *load magnitude* is peer-reviewed (Saeterbakken erector spinae),
  but no RAG source experimentally tests spinal-flexion injury under row load — flagged inline.
- Band Pull Apart is backed by Fukunaga PMC8975561 plus one verified supplementary peer-reviewed
  source, Camargo & Neumann PMC6849087 (fetched and quoted), for the upper-trap-dominance harm claim.
- `trunk_extension_compensation`: harm claim is partly inferential (Fukunaga even notes trunk
  extension can be beneficial) — flagged inline; the fault itself is highly observable.


---

## Group D — Arm / scapular isolation — Bicep Curl, Arm Abduction, Arm VW

Movements: **Bicep Curl**, **Arm Abduction (standing lateral / shoulder-abduction raise)**,
**Arm VW (scapular V-to-W protraction/retraction)**.

Detection model: MediaPipe Pose, 33 landmarks, normalized image coords (x,y in [0,1], z depth,
visibility), monocular single camera. Landmark indices used below: 0 nose; 7/8 L/R ear;
11/12 L/R shoulder; 13/14 L/R elbow; 15/16 L/R wrist; 19/20 L/R index; 23/24 L/R hip.
`view_type` ∈ {side, front, rear, front_oblique, rear_oblique}, estimated per rep.

Angle conventions (match squat detector style):
- `elbow_angle` = angle at elbow vertex (shoulder–elbow–wrist), 180° = straight arm.
- `arm_elevation_angle` = angle between torso vector (shoulder→hip) and upper-arm vector
  (shoulder→elbow); ~0° = arm at side, ~90° = horizontal, ~180° = fully overhead.
- `torso_lean_deg` = angle of the mid-shoulder→mid-hip vector from image vertical
  (signed: sagittal for fwd/back lean, frontal for lateral lean depending on view).
- `neck_gap` = (ear_y − shoulder_y) in normalized units; shrinks when shoulders elevate
  (y increases downward). Compared against the rep's setup baseline.

---

### Bicep Curl

Rep phases: **setup/bottom** (arms extended at sides, `elbow_angle`≈170–180°) →
**concentric** (elbow flexion, lifting) → **top** (peak flexion, `elbow_angle`≈40–55°) →
**eccentric** (controlled lowering) → return to bottom.

> **NOTE (2026-08-09) — two section-wide corrections made at implementation time.** Recorded
> here so a reader who arrives at the original wording alone cannot silently re-introduce
> either. Neither changes a threshold or a citation. Implementation:
> `src/pose/movements/bicep_curl.py`; design spec:
> `docs/superpowers/specs/2026-08-09-bicep-curl-detector-design.md`.
>
> 1. **The `fault_id`s below are unprefixed and ship prefixed `curl_*`.** Every movement after
>    Squat prefixes, and the collision is not hypothetical: `row_incomplete_rom` and
>    `bpa_incomplete_rom` both already exist, and `merge_by_fault`, the analyses table and the
>    frontend's `byFault` map all key on `fault_id` with **no movement qualifier**, so a third
>    bare `incomplete_rom` would be indistinguishable from either. Shipped ids:
>    `curl_elbow_drift_forward`, `curl_trunk_swing_momentum`, `curl_incomplete_rom`.
> 2. **The directional qualifiers are dropped and both metrics are taken unsigned** — "toward
>    the anterior (wrist) side" in `elbow_drift_forward`, and "backward lean" in
>    `trunk_swing_momentum`'s second term. Parpa's protocol is "the elbows kept close to the
>    torso" and "avoiding trunk movements", **neither of which names a direction**, so the
>    undirected reading is what the citation actually supports and the signed one asserts more
>    than the source does. Recovering "anterior" would need a facing proxy whose threshold no
>    cited source supplies — the construct the OHP bar-path and deadlift bar-drift withdrawals
>    both rejected. Band Pull Apart's `wrist_depth_offset` facing sign does **not** transfer: it
>    works only because that movement holds the band in front of the torso by definition,
>    whereas a curl's wrists change depth sign within the rep. The cost is a wider net (a
>    backward "drag-curl" drift also fires), in the direction the citation supports.

#### Elbow drift forward
- **fault_id**: `elbow_drift_forward`
- **fault_name**: Elbow drifts forward (loss of elbow fixation)
- **description**: The elbow travels forward/up and away from the torso during the lift, adding shoulder flexion instead of keeping the upper arm fixed at the side.
- **detection_heuristic**: `upper_arm_lean = angle(shoulder→elbow vector, image-vertical-down)`. In setup the upper arm hangs ~vertical (≈0–10°). Flag if `upper_arm_lean > 25°` toward the anterior (wrist) side at any frame during concentric, or if elbow x-displacement anterior of the shoulder–hip vertical line exceeds `0.5 × upper_arm_length`. Direction: elbow moving anterior/superior relative to the shoulder.
- **observability**: medium — needs **side** or **front_oblique** (forward drift is largely in the sagittal plane; from a pure **front** view it collapses to depth-z and is low/unreliable).
- **biomechanical_rationale**: Forward elbow drift converts the curl into partial shoulder flexion, shifting load from biceps brachii to the anterior deltoid and reducing the target-muscle stimulus (performance loss).
- **citation**: Parpa K et al., *Muscles* (2025), PMC12550948, DOI 10.3390/muscles4040045.
- **citation_support**: The paper's validated proper-execution protocol states the arms were "fully extended at the sides, with the elbows kept close to the torso throughout the whole movement," with two investigators visually monitoring execution — i.e., the elbow staying fixed at the torso is the defined correct form, so anterior drift is a deviation from it. (Verified — read in RAG doc.)

> **NOTE (2026-08-09) — the second detection cue is UNREACHABLE and is not implemented.** The
> heuristic above offers "or if elbow x-displacement anterior of the shoulder–hip vertical line
> exceeds `0.5 × upper_arm_length`". That is **the first cue restated in different units, and
> strictly weaker**: displacement `= upper_arm_length · sin(lean)`, so the `0.5` threshold is
> `lean > arcsin(0.5) = 30°` — always satisfied when the angular term's `25°` already is. Every
> frame it could catch, the angular term has caught. Implementing it would add a metric, a
> threshold and a branch that can never change a verdict, which is the exact defect
> `row.rule_momentum_jerk`'s second condition had: a strict subset of its first, and therefore
> dead code that *read* as coverage. `upper_arm_length` is still emitted as a diagnostic so the
> equivalence stays checkable, and the arithmetic is pinned by
> `tests/test_bicep_curl.py::ElbowDriftRuleTest::test_the_displacement_disjunct_is_unreachable`.
> (The cue would also need the anterior direction the section-wide NOTE above rejects.)

#### Trunk swing / momentum
- **fault_id**: `trunk_swing_momentum`
- **fault_name**: Trunk swing / back-lean momentum
- **description**: The lifter leans the trunk back and uses hip/trunk momentum to heave the weight up rather than isolating elbow flexion.
- **detection_heuristic**: Track `torso_lean_deg` (mid-shoulder 11/12 → mid-hip 23/24 vs vertical) across the rep. Flag if within-rep oscillation `max(torso_lean_deg) − min(torso_lean_deg) > 12°`, or backward lean during concentric exceeds the setup baseline by `> 10°`. Direction: shoulders moving posterior/superior relative to hips at the start of the concentric.
- **observability**: high — best from **side**/**front_oblique** (sagittal back-lean). From **front** view, medium (sagittal lean projects to depth; only gross oscillation is visible).
- **biomechanical_rationale**: Trunk momentum offloads the biceps (reducing the intended stimulus) and imposes repeated lumbar extension/shear loading, a low-back injury risk.
- **citation**: Parpa K et al., *Muscles* (2025), PMC12550948, DOI 10.3390/muscles4040045.
- **citation_support**: Participants performed the curl "avoiding trunk movements and jerky motions," and "two experienced investigators visually monitored trunk movements and knee flexion to ensure the proper execution" — trunk movement is explicitly treated as a cheating/compensation deviation to be excluded. (Verified — read in RAG doc.)

#### Incomplete range of motion
- **fault_id**: `incomplete_rom`
- **fault_name**: Incomplete range of motion (partial curl)
- **description**: The rep fails to fully extend the elbow at the bottom and/or fully flex at the top.
- **detection_heuristic**: `elbow_angle` = angle(shoulder 11/12 – elbow 13/14 – wrist 15/16). Flag **incomplete extension** if `max(elbow_angle)` over the rep `< 150°`; flag **incomplete flexion** if `min(elbow_angle) > 60°`.
- **observability**: medium–high — best from **side**/**front_oblique** (forearm lies in the image plane; from **front** view forearm foreshortening degrades `elbow_angle` accuracy → medium).
- **biomechanical_rationale**: Training through the full joint range yields superior strength adaptation across arm angles; chronic partial reps forfeit strength gains available at the extended/flexed ends of the range (performance loss).
- **citation**: Havers et al., *European Journal of Sport Science* (2025), DOI 10.1002/ejsc.70087 (PubMed 41247250); supported by Parpa K et al., *Muscles* (2025), PMC12550948.
- **citation_support**: Havers et al. found full ROM (0–140°) produced greater strength gains than initial partial ROM — larger 1RM (SMD≈0.17) and greater MVC at the 100° elbow angle (SMD≈0.24). The RAG doc (Parpa) prescribes "a slow, controlled lowering of the dumbbells back to the starting position through the full range of motion." (Verified — fetched Havers PubMed abstract + read RAG doc.)

#### Wrist flexion
- **fault_id**: `wrist_flexion_curl`
- **fault_name**: Wrist flexion (curling with the wrist)
- **description**: The wrist bends into flexion to help hoist the load instead of staying neutral and rigid.
- **detection_heuristic**: `wrist_angle` = angle(elbow 13/14 → wrist 15/16 vector, wrist → index 19/20 vector); ~180° = straight. Flag deviation into flexion `> 30°`. **Best-available proxy only** — see observability.
- **observability**: low (any view) — the hand landmarks (index/thumb) are small, frequently occluded by the dumbbell, and wrist flexion is a small-joint motion poorly resolved by monocular pose; treat detections as low-confidence.
- **biomechanical_rationale**: Wrist flexion recruits wrist flexors and can strain the wrist joint, and diverts effort from the elbow flexors, reducing biceps loading.
- **citation**: Parpa K et al., *Muscles* (2025), PMC12550948, DOI 10.3390/muscles4040045.
- **citation_support**: The paper notes the curl "involves elbow flexion accompanied by … wrist supination or pronation" and that grip/wrist positioning influences flexor recruitment; a supinated, controlled grip is the prescribed form. Support for wrist *flexion* as a fault is indirect (grip/wrist position matters), and it is not monocular-observable — flagged low/UNVERIFIED for the specific injury magnitude. (Verified that the source discusses wrist/grip influence; the injury-risk magnitude of wrist flexion is UNVERIFIED in this source.)

> **WITHDRAWN — wrist flexion.** This rule is **withdrawn** (2026-08-09) and is NOT implemented
> in `src/pose/movements/bicep_curl.py`. It presents two failure modes at once — observability
> `low` on every view, *and* a `citation_support` that already self-reports "the injury-risk
> magnitude of wrist flexion is UNVERIFIED in this source" — so which treatment it gets is
> decided by reading the source rather than this paraphrase of it.
>
> **Parpa PMC12550948 was read in full. It never discusses wrist flexion.** Every wrist- and
> grip-related statement in the paper concerns **forearm rotation** (supination / pronation) or
> **grip type** — a different degree of freedom from flexion/extension:
>
> - "It primarily involves elbow flexion accompanied by either dynamic or mostly isometric
>   shoulder flexion and **wrist supination or pronation**." (line 18)
> - "biceps brachii and brachioradialis activation were the highest with the **supinated grip**
>   during the ascending phase" (line 33, citing Coratella 2023)
> - The protocol prescribes "holding a dumbbell in each hand in a **supinated grip**." (line 75)
>
> Nowhere does the paper state that the wrist bending into flexion is a fault, a cheat, or a
> loading risk. The rule's asserted mechanism ("Wrist flexion recruits wrist flexors and can
> strain the wrist joint") and its `30°` threshold **appear nowhere in the cited source** — the
> OHP bar-path (2026-07-25) and deadlift bar-drift (2026-08-01) pattern exactly: a threshold
> with no provenance attached to a citation that does not measure it.
>
> **Withdrawn, not registered-silent, and the distinction is load-bearing.** A
> registered-but-permanently-silent rule (`pushup.rule_scapular_winging`,
> `band_pull_apart.rule_loss_of_scapular_retraction`) is this project's way of saying "real,
> well-cited fault; the sensor cannot see it". Had the citation held, this rule would have
> shipped silent — the dumbbell does occlude landmarks 19/20 for much of the rep. It does not
> hold, so the rule is **absent**; a silent stub would assert the wrong diagnosis.
>
> **Open spec question:** does the Bicep Curl rule set want a genuine wrist rule? It would need
> (a) a source that measures wrist *flexion* under curl load with a number, and (b) a
> hand-landmark reading that survives dumbbell occlusion. Neither exists today. This is a
> withdrawal pending a decision, not a silent deletion.
>
> **The KG node is not the gap.** `Bicep Curl:Wrist Flexion Under Load` resolves with a
> non-empty `corrections` bucket (`Wrists In Line With Forearms`). The node stays; nothing in
> the detector points at it.

---

### Arm Abduction (standing lateral / shoulder-abduction raise)

Rep phases: **setup/bottom** (arms adducted at sides, `arm_elevation_angle`≈0°) →
**concentric** (abduction/raise) → **top** (target ≈90°) → **eccentric** (controlled lower).

> **NOTE (2026-08-09) — three section-wide corrections made at implementation time.** Recorded
> here so a reader who arrives at the original wording alone cannot silently re-introduce any of
> them. None changes a threshold or a citation. Implementation:
> `src/pose/movements/arm_abduction.py`; design spec:
> `docs/superpowers/specs/2026-08-09-arm-abduction-detector-design.md`.
>
> 1. **The `fault_id`s below are unprefixed and ship prefixed `arm_abd_*`.** Every movement after
>    Squat prefixes, because `merge_by_fault`, the analyses table and the frontend's `byFault` map
>    all key on `fault_id` with **no movement qualifier**. The prefix is `arm_abd_` rather than
>    `abduction_` as a deliberate collision guard: Group E's **Leg Abduction** is also coming.
>    Shipped ids: `arm_abd_shoulder_shrug`, `arm_abd_contralateral_trunk_lean`,
>    `arm_abd_lr_asymmetry`.
> 2. **"target ≈90°" above is not a value this pipeline has.** Grepped across `src/pose/` and
>    `backend/app/`: there is no prescribed target angle, no per-user ROM goal, nothing. The
>    two datasets for this movement disagree about the height anyway — REHAB24-6 Ex1's median
>    peak elevation is **130.2°** and Fit3D `side_lateral_raise`'s is **97.1°**, both performed
>    as instructed. This is why `excessive_elevation_impingement_arc` is withdrawn (below).
> 3. **LABELED CORRECT/INCORRECT GROUND TRUTH EXISTS FOR THIS MOVEMENT, and it is the first.**
>    REHAB24-6 `Ex1` **is** arm abduction (`src/rehab24/dataset.py` `EXERCISE_NAMES["1"]`): 178
>    repetitions, 9 subjects each contributing both classes, **90 correct / 88 incorrect**, 0
>    flagged mocap-erroneous, with marker-driven 3-D and cached MediaPipe landmarks for all 13
>    videos. §8.4's standing "no labeled data" caveat therefore does **not** apply here, and the
>    Deadlift / Row / Band Pull Apart / Bicep Curl docstrings that assert it must not be copied
>    forward. **Lunge got there first** — REHAB24-6 `Ex5` is lunge and
>    `notes/lunge-rule-validation.md` is the 174-rep validation actually run against it — so this
>    is the **second** such movement and the **first whose data exists while the check has not been
>    run**.
>    `validated` is still `False` because **nothing has run the check** — and Ex1 is
>    **unilateral on 178/178 reps** (`exercise_subtype == "right arm"`), a variant this rule set
>    does not model, which by itself makes `lr_abduction_asymmetry` unvalidatable there in either
>    direction. The bilateral variant comes from Fit3D `side_lateral_raise` (8 subjects × 5 reps
>    of mocap 3-D, no correctness label) instead.

#### Shoulder shrug (upper-trap dominance)
- **fault_id**: `shoulder_shrug_elevation`
- **fault_name**: Shrugging / scapular elevation
- **description**: The shoulders hike up toward the ears (upper-trapezius dominance) during the raise, especially as the arm passes ~90°.
- **detection_heuristic**: `neck_gap = ear_y (7/8) − shoulder_y (11/12)`, per side, relative to the setup-baseline `neck_gap`. Flag if `neck_gap` shrinks `> 18%` below baseline (shoulders rise toward ears) during the raise; escalate severity when it co-occurs with `arm_elevation_angle > 90°`. **Confound**: some acromion/shoulder elevation is normal scapulohumeral rhythm at high elevation — the discriminating fault signal is *early or disproportionate* shrug (`neck_gap` collapse while `arm_elevation_angle < 90°`), so weight early-phase shrug more heavily to avoid firing on clean high-elevation reps.
- **observability**: high — **front** or **rear** view (vertical shoulder elevation is in-plane and clearly resolved).
- **biomechanical_rationale**: Persistent upper-trapezius overactivation with under-active lower scapular stabilizers drives scapular dyskinesis and raises the risk of subacromial impingement and glenohumeral instability.
- **citation**: Mun WL, Jung EY, Lei S, Roh SY, *Medicina* (2025), PMC12029123, DOI 10.3390/medicina61040645.
- **citation_support**: "Persistent overactivity of the UT can lead to scapular dysfunction (or dyskinesia), such as subacromial impingement or glenohumeral instability," and UT activation "consistently increases as the shoulder abduction angle surpasses 120°" so "care should be taken to avoid the excessive activation of the UT" at higher angles. (Verified — read in RAG doc.)

> **NOTE (2026-08-09) — REGISTERED BUT PERMANENTLY SILENT.** Implemented as
> `arm_abduction.rule_shoulder_shrug`, which always returns `[]`. Mun genuinely backs the fault,
> so this is a **sensing** failure, not a citation failure, and it takes the silent treatment
> (`pushup.rule_scapular_winging`, `band_pull_apart.rule_loss_of_scapular_retraction`) rather
> than withdrawal. **It is the first silent rule in this project justified by a measurement
> rather than an argument**, and the measurement has two independent halves:
>
> - **The metric collapses during abduction as a matter of anatomy, in 3-D ground truth, on the
>   BILATERAL variant, with no pose estimator in the path.** On Fit3D `side_lateral_raise` the
>   within-clip Spearman correlation between the head→shoulder vertical gap and the arm's own
>   elevation is **−0.699 to −0.954 across all 8 subjects**, the gap travels **27%–94% of its own
>   baseline**, and the `18%` threshold fires on **34 of 40** reps performed deliberately for a
>   mocap capture. Both endpoints move the same way — the shoulder joint rises (ρ +0.00 to +0.94)
>   *and* the head drops (ρ −0.32 to −0.86). The glenohumeral joint rising during abduction is
>   scapulohumeral rhythm: it is the movement, not a fault. Because it is measured on the
>   bilateral variant, the confound is **variant-independent** — on a bilateral raise both
>   shoulders ride their own humerus, so nothing is left to read the shrug against.
> - **MediaPipe reports the glenohumeral joint, not the acromion**, so the one reading that could
>   rescue the metric is unavailable. On REHAB24-6 Ex1, as a fraction of its own baseline height
>   above the mid-hip: the marker **clavicle** travels **1.0%**, the marker **glenohumeral** joint
>   **13.9%**, and **MediaPipe's landmark 11/12 travels 11.2%**. The fire rates split on the same
>   line — `18%` fires on **1/178** reps read off the marker clavicle and **172/178 = 96.6%** read
>   off MediaPipe, with the working-side gap at **ρ = −0.957** against true arm elevation *on
>   correct reps alone*, against **ρ = +0.068** for the resting arm (the control that makes this a
>   statement about the arm rather than the head or the framing).
>
> **The heuristic's own prescribed mitigation was measured and does not rescue it**: restricting
> to frames below 90° of elevation (its "early or disproportionate shrug") takes the MediaPipe
> fire rate only from 96.6% to **49.4%** — half of every rep in a dataset that is half correct.
>
> Two further notes on the citation, neither of which changes the treatment but both of which
> would matter a great deal to anyone who repairs the sensing. (i) **The `18%` carries no
> provenance**: Mun measures **EMG** during a Pilates Reformer arm-work movement at four
> abduction angles and supplies no landmark-displacement magnitude in any units. (ii) The
> citation_support's "consistently increases as the shoulder abduction angle surpasses 120°" is a
> **loose attribution** — in the source UT was highest at **160°** across all phases (the measured
> conditions were 0°/90°/135°/160°), and `120°` appears there only as a citation to a different
> study on elastic-band scapular retraction.
>
> This measurement also converts `band_pull_apart.rule_loss_of_scapular_retraction`'s **asserted**
> premise ("MediaPipe's shoulder landmark is a GLENOHUMERAL point … and it moves with the
> humerus") into a measured one. Whether the same confound reaches `band_pull_apart
> .rule_shrugging` — which ships **live** on this construction — is **not** claimed: that
> movement's excursion is horizontal abduction at roughly fixed elevation, so it plausibly
> differs in kind. Logged in TODO.md as a check to run, not a defect found.
>
> **Open, recorded, not resolved:** a working shrug rule needs shoulder height read *at matched
> arm elevation*, comparing like with like across the rep rather than against a setup baseline
> taken at a different arm position. Novel construction, no citation, no validation — not
> invented here.

#### Excessive elevation through the impingement arc
- **fault_id**: `excessive_elevation_impingement_arc`
- **fault_name**: Raising past safe/target ROM (impingement arc)
- **description**: The arm is driven high into (and through) the painful mid-abduction arc, or well past the prescribed target height, with poor scapular control.
- **detection_heuristic**: `arm_elevation_angle` = angle(shoulder→hip vs shoulder→elbow) in the frontal plane. Flag sustained `arm_elevation_angle` in ~70–120° performed with a concurrent shrug (`shoulder_shrug_elevation` true), or raising `> target + 15°` (e.g., `>105°` when the prescribed target is 90°).
- **observability**: high — **front** view (frontal-plane elevation is well measured). From a **side** view the arm overlaps the torso → low.
- **biomechanical_rationale**: Between ~70–120° of abduction the subacromial space narrows and the supraspinatus/long-head-biceps tendons and subacromial bursa are compressed (the "painful arc"); repeatedly loading through this arc with inadequate scapular upward rotation risks impingement.
- **citation**: Creech JA, Busse A, Li D, et al. *Shoulder Impingement Syndrome*, StatPearls (NCBI Bookshelf NBK554518, updated 2026); supported by Mun WL et al., *Medicina* (2025), PMC12029123.
- **citation_support**: StatPearls: the painful arc occurs "between approximately 70° and 120° of active shoulder abduction," where the subacromial space (normally 1–1.5 cm) "narrows physiologically with abduction," compressing the supraspinatus tendon, long head of biceps, and subacromial–subdeltoid bursa. Mun et al. corroborate elevated UT/impingement risk above 120°. (Verified — fetched StatPearls + read RAG doc.)

> **WITHDRAWN — excessive elevation through the impingement arc.** This rule is **withdrawn**
> (2026-08-09) and is **NOT implemented** in `src/pose/movements/arm_abduction.py` — absent, not
> silent. It fails three independent ways, and any one of them would be sufficient.
>
> 1. **The citation does not say what the rule says.** StatPearls NBK554518 was re-fetched and
>    read. It describes the painful arc as a **diagnostic sign**: *"Pain is reproduced between
>    approximately 70° and 120° of active shoulder abduction, with relative relief beyond 120°,
>    which is supportive of subacromial pathology."* Asked directly for any statement that raising
>    the arm through the arc, or above a specific angle, is itself a fault, an error, or a thing
>    to avoid during exercise, the source yields **nothing**. The arc is where a person who
>    *already has* subacromial pathology hurts; the rule's rationale reads that sign as a cause,
>    a step no source read here takes.
> 2. **The first disjunct is vacuous and is a restatement of another rule.** "Sustained
>    `arm_elevation_angle` in ~70–120° **with a concurrent shrug**" — measured on REHAB24-6 Ex1's
>    marker 3-D, **178 of 178 reps enter that band**, spending a median 30% of their frames there.
>    Passing through 70–120° *is* what an abduction is, so the arc conjunct is always true and the
>    cue reduces to "`shoulder_shrug_elevation` fired" — which, per the NOTE above, is never. Same
>    defect as `row.rule_momentum_jerk`'s second condition and the Bicep Curl elbow-displacement
>    disjunct: a branch that reads as coverage and can never change a verdict.
> 3. **The second disjunct has no referent.** "`> target + 15°`" needs a prescribed target, and
>    **this pipeline has none** (grepped across `src/pose/` and `backend/app/`). Fixing 90° would
>    be an uncited rule-level number, and the two datasets show it cannot be fixed at all: `>105°`
>    fires on **168/178 = 94.4%** of REHAB24-6 Ex1 reps but **8/40 = 20%** of Fit3D
>    `side_lateral_raise` reps. A threshold whose fire rate swings ~5× between two datasets of the
>    same named movement measures which variant was performed, not a fault. And on Ex1 the
>    direction is wrong too: **correct reps go higher than incorrect ones** (median 132.4° vs
>    125.2°; AUC that incorrect reps rank high = **0.333**, an inversion).
>
> **Withdrawn, not registered-silent, and the distinction is load-bearing.** A silent rule says
> "real, well-cited fault; the sensor cannot see it". The sensor reads elevation angles perfectly
> well — MediaPipe's arm elevation tracks the markers at within-rep ρ = +0.995. It is the
> citation and the arithmetic that fail, so the rule is **absent**; a silent stub would assert the
> wrong diagnosis.
>
> **Open spec question, recorded not resolved.** The KG's own Arm Abduction fault list contains
> **`Arm Abduction:Incomplete Elevation`** — the *opposite* fault, and the richest of the three
> nodes (`quality_impacts: Humerus Abduction`; `causes: Limited Shoulder ROM`). Every other
> movement in this spec got an incomplete-ROM rule; Arm Abduction got "raised too high" instead
> and now has no ROM rule at all. Filling that gap needs a source that puts a number on
> insufficient abduction. **No rule was invented to fill it.**

#### Contralateral trunk lean
- **fault_id**: `contralateral_trunk_lean`
- **fault_name**: Trunk lean to the opposite side
- **description**: The torso side-bends away from the working arm to help hoist it (frontal-plane compensation).
- **detection_heuristic**: `lateral_trunk_lean` = angle of mid-shoulder→mid-hip vector from image vertical in the frontal plane (uses the x-offset between mid-shoulder and mid-hip). Flag if lateral lean `> 12°` away from the raising arm during concentric, or if it grows with load across a set. (For a single-arm raise, sign the lean relative to the working side.)
- **observability**: high — **front**/**rear** view (lateral lean is in-plane).
- **biomechanical_rationale**: Contralateral lean substitutes trunk lateral flexors for deltoid/scapular work, reducing target loading and indicating insufficient shoulder strength/control; the accompanying poor scapular mechanics is part of the impingement-risk pattern.
- **citation**: Creech JA, Busse A, Li D, et al. *Shoulder Impingement Syndrome*, StatPearls (NCBI Bookshelf NBK554518, updated 2026).
- **citation_support**: StatPearls attributes impingement in part to "inadequate scapular upward rotation and posterior tilt" — i.e., compensation that fails to control the scapula during elevation, which contralateral trunk lean is a gross form of. The injury mechanism (impingement from poor scapular control during elevation) is verified via StatPearls. The specific frontal-plane trunk-lean substitution during abduction is **UNVERIFIED** in a peer-reviewed source (no read source isolated trunk lateral flexion during abduction; only fitness-coaching sources describe it, which do not qualify as injury-risk support). (Partially verified — injury mechanism verified; trunk-lean-specific EMG/kinematic finding UNVERIFIED.)

> **NOTE (2026-08-09) — SHIPS despite the UNVERIFIED line, and two sub-criteria are dropped.**
> Implemented as `arm_abduction.rule_contralateral_trunk_lean`, threshold unchanged at 12°.
>
> **Why an UNVERIFIED citation line did not withdraw this rule, when it withdrew curl
> wrist-flexion.** Re-fetching StatPearls NBK554518 confirms the paraphrase — asked for any
> mention of trunk lean, lateral trunk flexion, side-bending or contralateral compensation during
> abduction, it yields **nothing**. That is where wrist-flexion started too, and it lands
> differently for three **measured** reasons: (a) the cue **orders incorrect reps above correct
> ones** — per-subject median **AUC 0.800** across 9 subjects on REHAB24-6 Ex1's 178 human-labeled
> reps (pooled 0.647; 0.760 against the rep's own setup baseline), where wrist-flexion had no such
> measurement and no way to obtain one; (b) observability is `high` on front/rear and **those
> views are reachable**, where wrist-flexion was `low` on every view; (c) the injury **mechanism**
> is verified — only the specific frontal-plane substitution finding is not. That is the
> `lunge_insufficient_depth` shape — real cue, cited cut in the tail — whose settled treatment
> (`notes/lunge-rule-validation.md` §5.4) is *"Neither threshold moves."*
>
> **The threshold's placement, recorded rather than repaired.** 12° fires on **0/178** REHAB24-6
> Ex1 reps (max lean observed **7.6°**) and **1/40** Fit3D reps (max 14.1°). As shipped this rule
> will almost never fire, and when it does the lean is gross. Both figures are 3-D ground truth;
> image-plane obliquity foreshortens a frontal lean, so the projection error runs in the same
> direction as the threshold placement — a missed fault, never a false one.
>
> **Two sub-criteria are dropped, both unimplementable rather than unwanted.** (i) *"away from the
> raising arm"* — on a **bilateral** raise there is no raising arm, so the qualifier is undefined
> for the variant this app models, and on the unilateral variant it would need a working-side
> determination the detector cannot make. The metric is taken **unsigned** (the same construction
> the Bicep Curl NOTE records): an unsigned departure is what the verified mechanism describes,
> and the cost is that a lean *toward* the working arm also fires. (ii) *"or if it grows with load
> across a set"* — this pipeline has no load, and `run_detector` scores one rep at a time with no
> cross-rep state anywhere. Absent rather than approximated.

#### Left/right asymmetry
- **fault_id**: `lr_abduction_asymmetry`
- **fault_name**: Left vs right asymmetry
- **description**: One arm lags, rises less, or is timed differently from the other during a bilateral raise.
- **detection_heuristic**: Compare sides: `asym = |arm_elevation_angle_L − arm_elevation_angle_R|`. Flag if `asym > 12°` at the top-hold, or if peak wrist heights differ by `> 0.05` normalized units, sustained across reps.
- **observability**: high — **front**/**rear** view (both arms visible, elevation in-plane).
- **biomechanical_rationale**: Inter-limb asymmetry reflects unbalanced strength/scapular control; asymmetries in the ~10–15% range are associated with elevated injury risk and reduced performance.
- **citation**: Terré M, Solana-Tramunt M, *Healthcare (Basel)* (2025), 13(10):1153, PMC12110944, DOI 10.3390/healthcare13101153.
- **citation_support**: The paper states "asymmetries between 10% and 15% are often associated with a higher risk of injury and reduced performance," and uses a limb-symmetry scale (asymmetry 0–79%, limit 80–89%, normal/symmetrical 90–100%). (Verified — fetched PMC article.)

> **NOTE (2026-08-09) — SHIPS with the spec's 12°, the second cue is dropped, and the citation's
> units need stating.** Implemented as `arm_abduction.rule_lr_asymmetry`, scoped to `peak` (the
> spec's own "top-hold").
>
> **The 12° has no provenance in its citation, and the mismatch is a category one, not just a
> unit one.** Terré & Solana-Tramunt PMC12110944 was re-fetched: it measures **middle- and
> lower-trapezius EMG symmetry** during bilateral scapular retraction at 45° and 90° of shoulder
> abduction, and every threshold in it is a **percentage** — "10% and 15%", on a limb-symmetry
> scale of 0–79 / 80–89 / 90–100. **No angular threshold appears anywhere in the paper.** A 12°
> difference on a 90° raise is ~13%, inside the cited band — but that correspondence is a
> **reconstruction, not a provenance**, it silently assumes the 90° target the section-wide NOTE
> above shows the pipeline does not have, and this spec never states it. Shipped unchanged anyway,
> following `ohp_asymmetric_press` (cited at 7° of scapular angle / 1.5 cm of lateral shift,
> shipped as 0.15 of normalized wrist height): the mismatch is written at the constant.
> Re-expressing the rule as a percentage was considered and **rejected** — changing units changes
> what fires, which the no-tuning rule covers, and it would still transfer an EMG figure to a
> kinematic quantity.
>
> **The wrist-height disjunct is NOT implemented, and for a reason no previously dropped disjunct
> had.** "peak wrist heights differ by `> 0.05` normalized units" is **not** redundant with the
> angular cue the way the Bicep Curl elbow-displacement disjunct was. It is dropped because
> **`0.05` in raw normalized image units is not a well-defined criterion**: normalized coordinates
> scale with how much of the frame the subject occupies. Measured across the 43 production pose
> JSONs under `data/runtime/pose_json` carrying a usable shoulder width, the per-clip median
> `shoulder_width` runs **0.0591 to 0.4923** — an **8.3× spread** — so 0.05 units is **0.102
> shoulder-widths** on the widest-framed clip and **0.846** on the narrowest. The same physical
> asymmetry fires or does not depending on how far the phone was from the lifter. `shoulder_width`
> is emitted as a diagnostic so this stays checkable; the arithmetic is pinned by
> `tests/test_arm_abduction.py::AsymmetryRuleTest::test_the_wrist_height_disjunct_is_frame_scale_dependent`.
> The trailing *"sustained across reps"* is dropped too: no rule in this codebase carries cross-rep
> state.
>
> **This rule is the one the only labeled dataset cannot check, and the reason is a variant
> mismatch rather than a defect** — see the section-wide NOTE: REHAB24-6 Ex1 is unilateral on
> 178/178 reps, where the two arms' elevations differ by 64.3–132.2° (median 104.2°), so the
> threshold is exceeded on every rep of both classes. That **is not a false-positive rate**: a
> rule reporting "your two arms did completely different things" is correct about a one-armed
> raise. On the bilateral variant (Fit3D `side_lateral_raise`) the same threshold at the same
> phase fires on **2/40** reps — median asymmetry at the peak 4.4°, max 16.8°. **No bilateral
> precondition is implemented**: gating on "both arms are actually raising" needs an elevation
> floor no cited source supplies.

> **ADDENDUM (2026-08-09) — THIS RULE'S "no view gate, only a discount" RATIONALE IS REFUTED BY A
> MEASUREMENT TAKEN ON ARM VW, AND THE RULE IS NEVERTHELESS UNCHANGED.** `arm_abduction
> .rule_lr_asymmetry` ships live on every view because "obliquity foreshortens both arms together,
> so a real asymmetry reads smaller — a missed fault, never a false one." On REHAB24-6 **Ex2**
> (arm VW, bilateral, 208 reps), split by camera orientation, MediaPipe's `|L − R|` sits at a
> median **5.9°** against the markers' 4.6° on `front` clips — where the argument holds — and at
> **16.0°** against **4.1°** on `half-profile` clips, where the same 12° cut fires on **66 of 99**
> reps the 3-D truth calls symmetric. **Obliquity does not foreshorten the asymmetry; it
> fabricates it.** `arm_vw.rule_lr_asymmetry` therefore gates to `{front, rear}`.
>
> **It is NOT applied here, and the reason is evidence rather than scope.** The measurement is
> three inferential steps from this rule's operating conditions: it is on **Ex2, not Ex1** (whose
> unilateral variant makes this rule's own false-positive rate unmeasurable — see above); on the
> **`image` 2-D cache** while production runs `angle_degrees(dims=3)`, which the cache cannot
> reproduce because it stores no image-z; and on **front-hemisphere** obliquity while production
> is 37/49 `rear_oblique`. Changing a shipped rule's firing behaviour across inferential steps is
> the same move the no-tuning rule forbids for thresholds. **Logged in TODO.md as a scoped check
> to run against Arm Abduction's own data**, at which point either this rule gates too or the
> discount is justified with a measurement instead of an argument. `ohp_asymmetric_press` uses a
> different metric (normalized wrist height) and inherits the same open question.

---

### Arm VW (scapular V-to-W protraction/retraction)

Open-chain scapular drill: arms overhead/wide in a **V (Y)** with the scapulae elevated/upwardly
rotated → pull the elbows down and back into a **W** with scapular retraction + depression →
brief isometric hold → return to V. Rep phases: **V/protraction-elevation** →
**pull-down/retraction** → **W hold (isometric)** → **return to V**.

> **NOTE (2026-08-09) — four section-wide corrections made at implementation time.** Recorded
> here so a reader who arrives at the original wording alone cannot silently re-introduce any of
> them. Implementation: `src/pose/movements/arm_vw.py`; design spec:
> `docs/superpowers/specs/2026-08-09-arm-vw-detector-design.md`.
>
> 1. **The `fault_id`s below are unprefixed and ship prefixed `vw_*`.** Every movement after
>    Squat prefixes, because `merge_by_fault`, the analyses table and the frontend's `byFault`
>    map all key on `fault_id` with **no movement qualifier**. Shipped ids:
>    `vw_incomplete_excursion`, `vw_shrug_substitution`, `vw_loss_of_elevation`,
>    `vw_lr_asymmetry`.
> 2. **LABELED CORRECT/INCORRECT GROUND TRUTH EXISTS, AND FOR THE FIRST TIME IT IS THE VARIANT
>    THE APP MODELS.** REHAB24-6 `Ex2` **is** arm VW (`src/rehab24/dataset.py`
>    `EXERCISE_NAMES["2"]`): **208 repetitions — the largest labeled set of any non-squat
>    movement** (Lunge 174, Arm Abduction 178) — **94 correct / 114 incorrect**, 9 subjects each
>    contributing both classes, 0 flagged mocap-erroneous, marker 3-D and cached MediaPipe
>    landmarks for all 12 videos. It is **BILATERAL**, established by measurement rather than by
>    the blank `exercise_subtype` field: per-rep left/right excursion ratio median **0.954** (min
>    0.791), within-rep r(L,R) elevation median **0.9977** (min 0.9628). Arm Abduction had to
>    reach for Fit3D because Ex1 was unilateral on 178/178 reps; **nothing here does**.
>    `validated` is still `False` because **nothing has run the check** — the second movement
>    carrying that debt. One caveat for anyone using Ex2: **person 8 contributes 2 correct
>    against 20 incorrect**, so every per-subject AUC below is reported with and without them.
> 3. **ALL FOUR CITED SOURCES STUDY A DIFFERENT EXERCISE THAN THIS ONE, AND ALL FOUR ARE EMG.**
>    Jung PMC12734928 is quadruped / single-leg **push-up-plus** and sternum-drop; Abiara
>    PMC12335237 is prone cobra / wall slide / scapula setting / prone trapezius exercise; Mun
>    PMC12029123 is a **Pilates Reformer** "arm work" movement; Terré PMC12110944 is bilateral
>    scapular retraction at 45° and 90°. **None reports a kinematic threshold in any landmark
>    unit**, so every number in this section is this spec's rather than a source's. That is the
>    generalised form of the lesson the Arm Abduction impingement-arc withdrawal drew, and it is
>    now four for four.
> 4. **The rep signal is `avg_arm_elevation_deg` with polarity `min`** — the inverse of Arm
>    Abduction's — because the effort peak is the **W**, where elevation is lowest. Measured on
>    Ex2's 208 annotated reps: median start **140.4°**, median trough **54.7°** at position
>    **0.508** of the rep, median end 141.1°. `rule_loss_of_elevation` is consequently the first
>    shipped rule in the project scoped to the 15% `setup` window — the phase-fraction trap that
>    silenced Bicep Curl's extension term — and it clears at **1.65×** on the reps the rules
>    actually score (234 segmented over Ex2's 12 videos, **217 complete and analyzed**, 2.20–19.73 s;
>    `setup` min 9 frames against `min_frames` 6), or **1.25×** on the shortest **partial** rep
>    (1.67 s), which `select_reps` analyzes when no complete rep exists.

#### Incomplete scapular / arm excursion
- **fault_id**: `incomplete_scapular_rom`
- **fault_name**: Incomplete protraction/retraction excursion
- **description**: The movement is shallow — the arms/scapulae don't reach full retraction+depression in the W (or full elevation in the V).
- **detection_heuristic**: Use the visible arm-excursion proxy for the (non-observable) scapular travel: vertical travel of wrist/elbow between phases, `excursion = wrist_y(V) − elbow_y(W)`, and elbow descent to shoulder line at W. Flag if `arm_elevation_angle` swing between V and W phases `< 40°`, or elbow fails to descend to within `0.05` (normalized y) of the shoulder line at the W. True A-P scapular retraction is not directly measured (see observability).
- **observability**: medium for the **arm-elevation excursion** (front view); **low** for true scapular protraction/retraction, which is an anterior–posterior depth motion not resolvable from a monocular front view.
- **biomechanical_rationale**: Greater scapular excursion increases trapezius recruitment; a truncated excursion under-loads the middle/lower trapezius the drill is meant to train (performance loss).
- **citation**: Jung EY, Roh SY, Mun WL, *Life* (2025), PMC12734928, DOI 10.3390/life15121840.
- **citation_support**: The study found the larger-excursion variation (sternum-drop, STD) "elicited higher trapezius activation, especially during large scapular excursions," and that "greater scapular excursion is known to increase muscle activation" (end-range positions were marker-verified). (Verified — read in RAG doc.)

> **NOTE (2026-08-09) — SHIPS on the first disjunct only, and its warrant is weaker than
> `contralateral_trunk_lean`'s.** Implemented as `arm_vw.rule_incomplete_excursion`, scoped to
> the whole rep because an excursion is a property of the rep.
>
> **The second disjunct is dropped**, for the reason the Arm Abduction spec §6.7 established and
> pinned: `0.05` in raw normalized image units is not a well-defined criterion, because
> normalized coordinates scale with how much of the frame the subject occupies. Per-clip median
> `shoulder_width` across the 43 usable production pose JSONs runs **0.0591–0.4923** — an
> **8.3× spread** — so `0.05` units is 0.102 shoulder-widths on the widest-framed clip and 0.846
> on the narrowest. `shoulder_width` is emitted as a diagnostic so this stays checkable.
>
> **Where the 40° sits, recorded rather than repaired:** it fires on **0/208** REHAB24-6 Ex2 reps
> on the marker 3-D, **0/208** on the same reps through MediaPipe, and **0/41** on Fit3D
> `overhead_trap_raises`. The smallest swing observed anywhere is **47.0°**.
>
> **It is NOT logically dominated by `loss_of_elevation`, and the tempting claim that it is would
> be wrong.** That rule is silent when V ≥ 120 and W ≥ 75; a rep with V = 120 and W = 85 satisfies
> both and still swings only 35°. So this is not the vacuous-branch defect that killed
> `row.rule_momentum_jerk`'s second condition, Bicep Curl's elbow-displacement disjunct and the
> impingement arc's first conjunct — it is a live branch that simply never fires on anything
> measured.
>
> **Say plainly what it ships on.** `arm_abd_contralateral_trunk_lean` shipped past an UNVERIFIED
> citation because its cue scored a per-subject median AUC of **0.800**. This cue scores **0.452
> (pooled 0.476)** over Ex2's 9 subjects and **0.494 (pooled 0.502)** over the eight
> non-degenerate ones — *exactly at chance*. It ships on **semantic correctness** (a sub-40°
> swing really is an incomplete V-to-W, so firing on one is never wrong) plus a **background-cited
> mechanism**, and NOT on measured discrimination. The at-chance AUC is evidence about **Ex2's
> error type** — REHAB24-6 does not record which error each incorrect rep contains — not about
> the rule.
>
> **Attaching an ARM metric to `Arm VW:Insufficient Scapular Retraction` is not the metric
> substitution this project forbids**, and a reader will trip on it unless it is written down:
> this section itself declares the rule a proxy ("Use the visible arm-excursion proxy for the
> (non-observable) scapular travel … True A-P scapular retraction is not directly measured").
> The forbidden move is shipping metric B under a fault_id whose citation is about metric A
> *without saying so*.

#### Shrug substitution
- **fault_id**: `shrug_substitution`
- **fault_name**: Upper-trap shrug substitution
- **description**: The upper trapezius takes over (shoulders shrug up toward the ears) instead of the lower trapezius/serratus performing scapular depression and retraction.
- **detection_heuristic**: `neck_gap = ear_y − shoulder_y` vs setup baseline. During the pull-down/retraction and W-hold — where the shoulders should stay depressed — flag if `neck_gap` shrinks `> 18%` below baseline (shoulders rising). **Confound**: the V phase legitimately elevates the shoulders (arms overhead), so restrict this flag to the pull-down/W-hold phases where depression is expected; the discriminating signal is shoulders *rising when they should be depressing*, not absolute elevation.
- **observability**: high — **front**/**rear** view (vertical shoulder elevation in-plane).
- **biomechanical_rationale**: Upper-trap dominance (a high UT/LT and UT/SA activation ratio) is the maladaptive scapular-dyskinesis pattern and defeats the lower-trap/serratus training aim.
- **citation**: Abiara S et al., *PeerJ* (2025), PMC12335237, DOI 10.7717/peerj.19861; supported by Jung EY et al., *Life* (2025), PMC12734928.
- **citation_support**: Abiara et al.: "ratios lower than 1.0 for the UT/LT ratio are preferred … although lower than 0.6 are ideal," and shoulder pain is "characterized by increased activation of the upper trapezius and decreased activation of the lower trapezius and serratus anterior." Jung et al.: "excessive UT dominance is linked to scapular dyskinesis," and lower UT/SA ratios reflect "a more favorable stabilization pattern." (Verified — read both RAG docs.)

> **NOTE (2026-08-09) — REGISTERED BUT PERMANENTLY SILENT.** Implemented as
> `arm_vw.rule_shrug_substitution`, which always returns `[]`. Abiara genuinely backs the fault,
> so this is a **sensing** failure and takes the silent treatment
> (`pushup.rule_scapular_winging`, `band_pull_apart.rule_loss_of_scapular_retraction`,
> `arm_abduction.rule_shoulder_shrug`) rather than withdrawal.
>
> **This is the SECOND movement on which the `neck_gap = ear_y − shoulder_y` construction has
> been measured, and it fails for a NEW reason.** The heuristic's confound mitigation above is
> structurally sounder than Arm Abduction's — this movement's pull-down runs the arm DOWN, where
> scapulohumeral rhythm predicts the shoulder should descend — so the rule was **re-measured
> rather than silenced by inheritance**. On REHAB24-6 Ex2's 208 reps, each candidate "shoulder"
> taken as height above the mid-hip and the gap referenced to the rep's opening frame:
>
> | point | ρ(gap, elevation) over the pull-down | 18% shrink fires | gap travel, % of baseline |
> |---|---|---|---|
> | marker **clavicle** (acromion) | med **−0.305** | **0/208** | 1.2% |
> | marker **glenohumeral** | med **−0.998** | **0/208** | 36.3% |
> | **MediaPipe** `\|ear − shoulder\|` | med **−0.957** | **0/208** | — |
>
> Shoulder-height travel as a fraction of its own baseline: marker clavicle **0.6%**, marker
> glenohumeral **9.8%** — MediaPipe reports the **glenohumeral joint, not the acromion**, the Arm
> Abduction finding reproduced under a **reversed** elevation direction, which is what makes it a
> property of the landmark rather than of that movement. Two independent failures follow:
> **(a)** the metric is an arm-elevation readout, not a shrug readout (ρ = −0.957); **(b)** the
> 18% threshold **can never fire on this movement's baseline convention**, because the rep opens
> at the **V** — the most-shrugged position in the whole movement — so "shrink below baseline" is
> negative throughout the pull-down and the W hold. **0/208 on all three instruments**, the exact
> inverse of Arm Abduction's 96.6%, and just as unusable. The cue also carries no label
> information: pooled AUC **0.484**, per-subject median 0.549.
>
> **The 18% carries no citation** (Abiara reports EMG ratios and no landmark displacement in any
> units), recorded so a future reader who repairs the sensing does not inherit it as cited.
>
> **One honesty note that does not change the treatment.** Abiara's Exercise C — "Participants
> stood against the wall and began with their arm abducted to 90°, their elbows bent to 90°, and
> their palms facing forward" — is the closest thing in any cited source to the **W position**,
> and the paper reports its UT/LT ratio as **over 1.0**, concluding "only the Modified Prone Cobra
> (Exercise B) can be recommended." The cited literature is lukewarm about the exercise this rule
> set is built around.
>
> **What this says about `band_pull_apart.rule_shrugging`, which ships LIVE on this construction:
> it NARROWS the open item without discharging it.** The gap now measurably tracks arm elevation
> on two movements whose elevation runs in **opposite** directions, so the confound scales with
> the *magnitude* of the elevation excursion rather than its sign. Band Pull Apart's excursion is
> horizontal abduction at roughly fixed elevation, so its confound should be **small** — an
> argument **for** that rule being sound, not against it. Still not measured on its own data.
>
> **Open, recorded, not resolved:** the same requirement Arm Abduction recorded — a working shrug
> rule needs shoulder height read *at matched arm elevation*, not against a setup baseline taken
> at a different arm position. Novel construction, no citation, no validation; not invented here.

#### Loss of arm-elevation angle
- **fault_id**: `loss_of_elevation_angle`
- **fault_name**: Loss of target V/W elevation angle
- **description**: The arms fall below the prescribed elevation in the V (or the elbows drop too low in the W), moving off the lower-trap-optimal position.
- **detection_heuristic**: `arm_elevation_angle` per side. Flag if V-phase peak `< 120°` (arms not raised high enough) or W-phase abduction `< 75°` (elbows collapsed toward the body). Thumbs-up/forearm orientation is not reliably measured monocularly and is not required for the flag.
- **observability**: high — **front** view (frontal-plane elevation well measured).
- **biomechanical_rationale**: Lower-trapezius activation is maximized near ~135° of shoulder abduction (aligned with its fiber direction); losing the elevation angle moves the scapula off the LT-optimal position and reduces the exercise's targeted effect.
- **citation**: Mun WL et al., *Medicina* (2025), PMC12029123, DOI 10.3390/medicina61040645; supported by Abiara S et al., *PeerJ* (2025), PMC12335237.
- **citation_support**: Mun et al.: "the LT activation was the highest at a 135° shoulder abduction angle, with excessively high angles leading to a decrease," and researchers "recommend shoulder abduction near 145°, aligning with the muscle fiber direction, for maximum LT activation." Abiara et al. describe the LT-targeting exercise performed with "arms abducted above 90°, thumbs up." (Verified — read both RAG docs.)

> **NOTE (2026-08-09) — the V disjunct SHIPS; the W disjunct is WITHDRAWN.** Implemented as
> `arm_vw.rule_loss_of_elevation`, scoped to `setup` — the **opening** V.
>
> **The 120° is the low end of a cited OPTIMUM, never a stated fault threshold.** Mun's own
> finding is 135° (p < 0.001) during a Pilates Reformer arm-work movement at 0/90/135/160°; the
> 145° and the 120° in the citation_support above are both **Mun citing other studies in its
> discussion**, and the 120° appears there as another study's LT *optimum*, not as a floor.
> Reading 120° as a floor is a defensible rendering of "stay in the 120–145° band", but no source
> states it as a failure threshold. Not moved. Measured, it lands just under the observed
> distribution — Ex2's median V peak on the markers is **143.8°**, inside the cited optimum — so
> this is the `lunge_insufficient_depth` shape.
>
> **It is the best-discriminating cue measured on this movement**, which is why it ships rather
> than merely being recorded: ranking Ex2's incorrect reps above its correct ones on the V peak
> gives pooled **0.596** / per-subject median **0.660** over all 9 subjects, and pooled **0.713**
> / per-subject median **0.735** over the eight non-degenerate ones. 0.735 is comparable to the
> 0.800 that carried `arm_abd_contralateral_trunk_lean`.
>
> **The shipped fire rate, measured through the REAL pipeline on the windows the rule actually
> sees.** `run_detector(ARM_VW_DETECTOR, ...)` over Ex2's 12 cached MediaPipe videos — which
> smooths, segments, trims and phases exactly as production does — fires this rule on **34 of the
> 217 analyzed reps (15.7%)**: 10/57 on the `front` clips, 24/160 on the `half-profile` ones.
> **Read over the annotation windows instead, the same rule fires on only 9/208** — `segment_reps`
> trims the V plateau away, so a segmented window's first 15% sits further down the descent.
> 15.7% is the shipped number; the 4.3% was measuring a window the rule never sees.
>
> **One semantic note on the reading.** "V-phase **peak** < 120°" strictly means the maximum over
> the V window is below 120; the codebase idiom is a per-frame mask plus
> `contiguous_true_segments`, which fires on any **sustained run** below 120 — strictly weaker, so
> it fires more. Over annotation windows the two read 6/208 against 31/208 on the marker 3-D; the
> shipped reading is the codebase idiom. Also note the **closing** V falls in `eccentric` and is
> not read, and the rep's global maximum sits near the end on most reps (median argmax position
> 0.918) — so the rule under-reads the movement's best moment, in the conservative direction.
>
> **WITHDRAWN — "or W-phase abduction < 75°".** Absent, not silent, and it fails two ways, either
> of which would be sufficient. **(i) The 75° appears in no cited source**: Mun measures
> 0/90/135/160°, Abiara's wall slide begins "abducted to 90°" and its prone exercise is "above
> 90°", Terré tests 45° and 90°, and no source read describes a *floor* on the W at all.
> **(ii) The computable quantity puts the entire observed distribution below the cut**:
> `angle(hip, shoulder, elbow)` is a frontal-plane reading and in the W the elbow travels down
> **and back**, an A-P component this section itself rates non-observable from a monocular frontal
> view. Median W elevation **58.4°** on the Ex2 markers (fires **187/208**), **24.6°** through
> MediaPipe (fires **206/208**), **67.9°** on Fit3D `overhead_trap_raises` (fires **39/41**). A
> criterion firing on 90–99% of reps in a dataset that is 45% correct is not measuring a fault,
> and its discrimination confirms it: per-subject AUC 0.360 over 9 subjects, **0.510** over the
> eight non-degenerate ones — at chance, the apparent inversion being a person-8 artifact.
>
> Withdrawn rather than registered-silent because the sensor reads frontal-plane elevation angles
> perfectly well; it is the **number** that has no source and the **quantity** that does not
> capture what this section meant by the W. Same treatment as the Arm Abduction impingement arc.

#### Left/right asymmetry
- **fault_id**: `lr_vw_asymmetry`
- **fault_name**: Left vs right scapular asymmetry
- **description**: One arm/scapula lags, sits lower, or retracts less than the other through the V→W cycle.
- **detection_heuristic**: `asym = |arm_elevation_angle_L − arm_elevation_angle_R|` at the V peak and at the W hold. Flag if `asym > 12°`, or if `|wrist_y_L − wrist_y_R| > 0.05` (normalized), sustained across reps.
- **observability**: high — **front**/**rear** view (both arms visible, elevation in-plane; A-P retraction asymmetry itself remains low-observability).
- **biomechanical_rationale**: Asymmetric scapular control reflects side-to-side stabilizer imbalance; inter-limb asymmetries of ~10–15% are associated with higher injury risk and reduced performance.
- **citation**: Terré M, Solana-Tramunt M, *Healthcare (Basel)* (2025), 13(10):1153, PMC12110944, DOI 10.3390/healthcare13101153; scapular-dyskinesis context from Jung EY et al., *Life* (2025), PMC12734928.
- **citation_support**: Terré & Solana-Tramunt: "asymmetries between 10% and 15% are often associated with a higher risk of injury and reduced performance" (limb-symmetry scale: normal 90–100%). Jung et al. tie unbalanced scapular muscle activation to scapular dyskinesis. (Verified — fetched PMC article + read RAG doc.)

> **NOTE (2026-08-09) — SHIPS with the spec's 12°, and is THE FIRST ASYMMETRY RULE IN THIS
> PROJECT TO GATE ON VIEW.** Implemented as `arm_vw.rule_lr_asymmetry`, scoped to `setup` ∪
> `peak` (this section's own "at the V peak and at the W hold"), **gated to `{front, rear}`**.
>
> **The 12° has the same non-provenance it has for `lr_abduction_asymmetry`**: Terré measures
> middle- and lower-trapezius **EMG symmetry** at 45° and 90° of abduction, and every threshold in
> it is a **percentage**. No angular threshold appears anywhere in the paper. Shipped unchanged,
> mismatch written at the constant; re-expressing it as a percentage was rejected because changing
> units changes what fires.
>
> **THE GATE, AND THE MEASUREMENT THAT FORCES IT — WHICH ALSO REFUTES THE REASON
> `lr_abduction_asymmetry` GIVES FOR NOT GATING.** `arm_abduction.rule_lr_asymmetry` ships live on
> every view on the argument that "obliquity foreshortens both arms together, so a real asymmetry
> reads smaller — a missed fault, never a false one." Measured on REHAB24-6 Ex2, split by
> `cam17_orientation`, taking max `|L − R|` over each window against the 12° cut:
>
> | cam17 | instrument | V window: median (fires) | W window: median (fires) |
> |---|---|---|---|
> | **front** (109) | marker 3-D | 4.6° (3/109) | 6.4° (12/109) |
> | | **MediaPipe image 2-D** | **5.9° (13/109)** | **7.4° (20/109)** |
> | | MediaPipe `world` 3-D | 27.0° (107/109) | 27.8° (104/109) |
> | **half-profile** (99) | marker 3-D | 4.1° (0/99) | 5.8° (5/99) |
> | | **MediaPipe image 2-D** | **16.0° (66/99)** | **22.2° (88/99)** |
> | | MediaPipe `world` 3-D | 28.8° (96/99) | 20.4° (86/99) |
>
> From a **true frontal** view the difference metric behaves (5.9° against 4.6°) and the
> common-mode-cancellation argument holds. From an **oblique** view it is **fabricated** (16.0°
> against 4.1°), and the shipped threshold fires on **66 of 99** reps the 3-D truth calls
> symmetric. The near arm and the far arm foreshorten by *different* amounts, so obliquity does
> not shrink the asymmetry — it manufactures one. MediaPipe's own 3-D does not rescue it (`world`
> is worse on both views), though `world` is a metric hip-centred output and **not** the image-z
> that `angle_degrees(dims=3)` consumes, so that row is a proxy in both directions.
>
> **State the ceiling, because it is severe.** Production is `rear_oblique` 37, `rear` 9,
> `unknown` 3, `side` 0 over 49 pose JSONs (re-measured 2026-08-09), and `front` is unreachable
> under `allow_front=False`. So this rule is **live on 9 of 49 clips and silent on the other 40** —
> the price of not firing falsely on two thirds of them. **What the gate buys and what it does
> not, measured through the real pipeline:** over Ex2's 217 analyzed reps it fires on **20/217 =
> 9.2%** (20/57 `front`, 0/160 `half-profile`); forced through as `front` everywhere it fires on
> **121/217 = 56%**, so the gate suppresses 101 firings — 63% of the oblique reps. **The residual
> is still high:** 20/57 = **35%** on truly-frontal clips against a marker-3-D exceedance of 3/109
> and 12/109. The gate removes the asymmetry **obliquity adds**; it does not make the metric agree
> with 3-D truth. Two inferential steps also sit underneath
> the gate: Ex2's cameras are front-hemisphere, so the gate excludes the views where fabrication
> was **measured** and admits one (`rear`) where it was **not**; those 9 clips earn `high` on a
> geometric argument (a frontal-plane difference reads the same mirrored, and `|L − R|` is
> sign-invariant), not on a measurement.
>
> **`src/pose/movements/arm_abduction.py` is deliberately NOT edited**, for evidence rather than
> scope — see the annotation on `lr_abduction_asymmetry` above.
>
> **Both sub-criteria are dropped.** "`|wrist_y_L − wrist_y_R| > 0.05` (normalized)" is the same
> frame-scale-dependent criterion as `incomplete_scapular_rom`'s second disjunct (8.3× spread in
> per-clip median `shoulder_width`). "sustained across reps" needs cross-rep state no rule in this
> codebase carries.
>
> **What Ex2 says about this rule is very little, and it is entitled to say it.** `|L − R|` on the
> marker 3-D scores per-subject AUC 0.378 (V) and 0.536 (W) — 0.375 / 0.513 without person 8 — and
> exceeds 12° on 3/208 and 17/208 reps. Ex2's incorrect reps are not asymmetric ones. **Unlike Arm
> Abduction**, where Ex1's unilateral variant made the rule unvalidatable in either direction, Ex2
> is bilateral, so this is a real if uninformative reading rather than a variant artifact.

---

### Verification / honesty notes
- All citation_support strings above were taken from sources actually read (four RAG docs read
  in full; Havers 2025, StatPearls NBK554518, and Terré 2025 fetched and quoted).
- **UNVERIFIED / partial**: (1) `wrist_flexion_curl` — the RAG source discusses grip/wrist
  influence but not wrist-flexion injury magnitude; also not monocular-observable (observability
  low). (2) `contralateral_trunk_lean` — the impingement/scapular-control injury mechanism is
  verified (StatPearls), but no peer-reviewed source read isolated frontal-plane trunk lateral
  flexion during abduction (web search surfaced only fitness-coaching sources, which do not
  qualify as injury-risk support and were not used as citations).
- Depth-dependent scapular protraction/retraction (A-P motion) is inherently low-observability
  from monocular front-view pose; the VW heuristics fall back to visible arm-elevation proxies
  and say so.

> **UPDATE (2026-08-09) — how the two UNVERIFIED items above actually resolved at implementation
> time, and one this section did not anticipate.** Both were re-checked against the sources
> rather than against this paraphrase, and they resolved in **opposite** directions, which is the
> point: an "UNVERIFIED" line is a prompt to read the source, not a verdict.
>
> - `wrist_flexion_curl` → **WITHDRAWN** (Bicep Curl, above). Parpa never discusses wrist
>   flexion at all; every wrist statement in it is about forearm rotation or grip.
> - `contralateral_trunk_lean` → **SHIPS**. StatPearls does contain nothing about trunk lean, but
>   the cue orders REHAB24-6 Ex1's incorrect reps above its correct ones at a per-subject median
>   AUC of **0.800**, and the injury mechanism is verified. Real cue, cited cut in the tail.
> - **Not anticipated here:** `excessive_elevation_impingement_arc` → **WITHDRAWN**, and this
>   section rated it fully verified. The quote was accurate; the **inference** was not. StatPearls
>   describes the 70–120° arc as a **diagnostic sign of existing pathology**, never as a fault to
>   avoid during exercise — so "Verified — fetched StatPearls" was true of the quotation and false
>   of the rule built on it. The lesson generalises past this movement: verifying that a source
>   contains a quoted string is not verifying that it supports the claim the quote is attached to.
>
> Two further corrections to the ratings above, recorded where a reader will meet them:
> `shoulder_shrug_elevation` is rated `high` on front/rear here and is **registered permanently
> silent** — the *view* rating was right and the *metric* is unusable (see its NOTE); and
> `lr_abduction_asymmetry`'s Terré citation is an **EMG-symmetry** finding, not a kinematic one,
> so its `12°` has no provenance in the cited source.

> **UPDATE (2026-08-09, second pass) — GROUP D IS COMPLETE, and the Arm VW pass generalised the
> lesson above rather than adding a new one.** All three Group D detectors now ship
> (`bicep_curl.py`, `arm_abduction.py`, `arm_vw.py`), 10 of 16 movements.
>
> - **The third UNVERIFIED-style resolution.** This section's first bullet says all
>   citation_support strings "were taken from sources actually read". True of the *strings*, and
>   the Arm VW pass shows it is not the property that matters: **all four Arm VW sources study a
>   different exercise than the one they are cited for, and all four are EMG.** Jung PMC12734928
>   is quadruped / single-leg **push-up-plus**; Abiara PMC12335237 is prone cobra / wall slide /
>   scapula setting; Mun PMC12029123 is a **Pilates Reformer** arm-work movement; Terré
>   PMC12110944 is bilateral scapular retraction at 45°/90°. None reports a kinematic threshold in
>   any landmark unit. Verifying a quotation is not verifying the claim it is attached to — now
>   demonstrated on the impingement arc (inference), on `wrist_flexion_curl` (absence), and here
>   on **exercise identity**.
> - **A source that is lukewarm about its own movement, recorded once.** Abiara's Exercise C (wall
>   slide, "arm abducted to 90°, elbows bent to 90°") is the nearest thing in any cited source to
>   the **W position**, and the paper reports its UT/LT ratio as **over 1.0**, concluding "only the
>   Modified Prone Cobra (Exercise B) can be recommended."
> - **The third bullet above is confirmed and sharpened.** A-P scapular motion is indeed not
>   resolvable monocularly — and the price is now measured, not just asserted: the frontal reading
>   `angle(hip, shoulder, elbow)` puts the W at 58.4° on 3-D markers and **24.6° through
>   MediaPipe**, which is what withdrew `loss_of_elevation_angle`'s W disjunct.
> - **One rating correction, and it belongs to Arm Abduction rather than Arm VW.** The
>   `lr_abduction_asymmetry` NOTE's "no view gate, only a discount" is refuted by an Ex2
>   measurement (see its ADDENDUM). `lr_vw_asymmetry` gates; `lr_abduction_asymmetry` is unchanged
>   pending a check on its own data.
> - **A new structural finding for the framework, not for this movement.** `loss_of_elevation_angle`
>   is the first shipped rule scoped to the 15% `setup` window, and the Bicep Curl phase-fraction
>   trap (`phase_fraction · T ≥ min_frames / fps`) **binds** there rather than being dodged. It
>   clears at 1.65× on analyzed reps and 1.25× on the shortest partial one, and both sides of the
>   cliff are pinned end-to-end. Any future rule reading a movement's *opening* position inherits
>   this constraint.
> - **A method note worth carrying forward.** Fire rates for phase-scoped rules must be measured
>   on **segmented** windows through `run_detector`, not on a dataset's annotation windows: the two
>   differ by **3.7×** for `loss_of_elevation_angle` here, because `segment_reps` trims the
>   plateau the annotations include. This document originally carried the annotation-window figure.

---

## Group E — Core / rehab — Sit-up, Shoulder Bridge, Leg Abduction

Movements: **Sit-up (curl-up)**, **Shoulder Bridge (supine bridge)**, **Leg Abduction (side-lying / standing hip abduction)**.

Detection model: MediaPipe Pose, 33 landmarks, normalized image coords (x, y, z, visibility), monocular single camera. Landmark indices per shared context (0 nose; 7/8 ears; 11/12 shoulders; 13/14 elbows; 15/16 wrists; 23/24 hips; 25/26 knees; 27/28 ankles; 29/30 heels; 31/32 foot index).

Convention notes used below:
- **Trunk-flexion angle** = angle of the shoulder-midpoint→hip-midpoint vector relative to the floor/horizontal (0deg = lying flat, 90deg = fully upright seated).
- **Hip angle** = angle at the hip landmark formed by shoulder→hip→knee (≈180deg = trunk/thigh straight line; <180deg = hip flexed; >180deg = hip hyperextended/arched).
- **Pelvic-tilt (frontal)** = signed angle of the left-hip(23)→right-hip(24) line relative to horizontal.
- **Trunk lateral-lean** = horizontal offset of shoulder midpoint from hip midpoint, normalized by trunk length.

---

### Sit-up (curl-up)

Rep phases: **setup (supine)** → **concentric trunk flexion (curl up)** → **top** → **eccentric return (lower)** → **rest**.

#### excessive_speed_trunk_control_loss

- **fault_id**: `situp_excessive_speed`
- **fault_name**: Excessive speed / loss of trunk control
- **description**: The curl is thrown up with a fast, jerky, ballistic motion instead of a slow controlled lift, with the trunk wobbling off the sagittal line.
- **detection_heuristic**: Concentric phase duration (frames from lift-off to top) and peak angular velocity of the trunk-flexion angle. Flag if concentric phase < ~1.0 s (roughly the fastest cadence tested) OR peak |d(trunk_angle)/dt| exceeds a per-user baseline by a large margin; secondary signal = high frame-to-frame acceleration (jerk) spikes and increased medial-lateral wobble of the shoulder midpoint (x-variance about the sagittal path).
- **observability**: medium — needs **side** view for velocity/ROM; medial-lateral wobble needs **front**/**front_oblique**. Absolute speed is measurable; "control loss" is a proxy.
- **biomechanical_rationale**: Fast curl-ups increase trunk angular momentum and spinal load/intradiscal pressure and reduce the time available for neuromuscular correction, so they should be used with caution in people with motor-control deficits or low-back disorders.
- **citation**: Barbado D, Moreno-Navarro P, Vera-Garcia FJ, et al. "Effect of Performance Speed on Trunk Movement Control During the Curl-Up Exercise." J Hum Kinet (2015). PMC4519219, DOI 10.1515/hukin-2015-0031.
- **citation_support**: "the linear variability of COP_ML significantly increased as curl-up exercise speed increased" and participants "performed a greater neuromuscular effort to control trunk motion during the fastest curl-up exercises"; the paper states that "due to the effect of performance speed on the spinal loads and intradiscal pressure, fast curl-up exercises should be used with caution in people with motor control deficits or low-back disorders, as well as in novice, untrained or unfit individuals," and that higher speed increases angular momentum and "greater difficulty to slow down the trunk flexion motion." VERIFIED (read RAG doc).

#### hip_flexor_dominance_anchored

- **fault_id**: `situp_hip_flexor_dominance`
- **fault_name**: Hip-flexor dominance (anchored feet / rigid straight-body pull)
- **description**: The trunk is lifted as a rigid straight segment rotating about the hip (spine not curling) with feet anchored, so the movement is driven by the hip flexors rather than by segmental trunk flexion of the abdominals.
- **detection_heuristic**: Measure trunk "curl" as the change in the shoulder→hip→knee collinearity during the concentric phase. Flag if the spine stays near-straight (shoulder–hip–knee remain close to collinear, trunk_curl change < ~10-15deg) while the trunk-thigh (hip) angle closes rapidly — i.e. rigid-body rotation about a fixed pelvis. Supporting proxy: feet/heels (29/30, 31/32) remain fixed (low displacement) and knees do not lift, indicating anchored feet.
- **observability**: medium — needs **side** view. True muscle recruitment is not visible; rigid-vs-segmental trunk motion is the observable proxy.
- **biomechanical_rationale**: Anchoring the feet and pulling with straight legs recruits the iliopsoas/rectus femoris and increases lumbar lordosis and uneven lumbar-disc loading while under-activating the abdominals, defeating the purpose of the exercise and raising low-back stress.
- **citation**: Mandroukas A, Michailidis Y, Metaxas T. "Surface Electromyographic Activity of the Rectus Abdominis and External Oblique during Isometric and Dynamic Exercises." J Funct Morphol Kinesiol (2022). PMC9505236, DOI 10.3390/jfmk7030067.
- **citation_support**: "support on the feet activates the hip flexors and reduces the activity of the abdominal muscles"; the curl-up is recommended "with flexed unsupported knees, without holding the knees or feet ... to isolate the activity of the hip flexors," and the movements performed by "the hip flexors, particularly by the iliopsoas, rectus femoris, and sartorius ... increases lordosis in the lumbar spine." The quick, rigid start also "did not [give] enough time for the abdominal muscles to contract." VERIFIED (read RAG doc).

#### excessive_rom_full_situp

- **fault_id**: `situp_excessive_rom`
- **fault_name**: Excessive ROM (full sit-up past the curl-up range)
- **description**: The trunk continues past a partial curl into a full upright sit-up, lifting the whole lumbar spine off the floor rather than stopping once the scapulae clear.
- **detection_heuristic**: Peak trunk-flexion angle at top. In a correct curl-up the scapulae just lift and the trunk reaches only ~35-40deg from the floor; flag `excessive_rom` if peak trunk-flexion angle > ~50-60deg (approaching a full seated position) OR the hip angle (shoulder–hip–knee) closes below ~110deg, indicating a full trunk-on-thigh sit-up.
- **observability**: high — **side** view; trunk-flexion angle is directly measurable in the sagittal plane.
- **biomechanical_rationale**: Full sit-ups markedly raise lumbar intervertebral-disc pressure (reported at L3), whereas limiting trunk flexion to ~35-40deg keeps the lumbar spine on the floor, lowers disc load, and still develops the abdominals — so over-ranging trades safety for little added benefit.
- **citation**: Mandroukas A, Michailidis Y, Metaxas T. J Funct Morphol Kinesiol (2022). PMC9505236, DOI 10.3390/jfmk7030067.
- **citation_support**: "The stress placed on the lumbar spine decreases by limiting the amount of trunk flexion to 35-40deg ... curl-ups performed through a partial range may be an effective method of gaining abdominal muscle strength, while protecting the lumbar spine," and "Nachemson reported increased pressure on the intervertebral disc at the level of L3 during the execution of full sit-ups." VERIFIED (read RAG doc).

#### incomplete_rom_scapula

- **fault_id**: `situp_incomplete_rom`
- **fault_name**: Incomplete ROM (scapulae not lifted)
- **description**: The head/neck lifts but the shoulders/scapulae barely clear the floor, so little real trunk flexion occurs.
- **detection_heuristic**: Peak trunk-flexion angle at top. Flag `incomplete_rom` if peak trunk-flexion angle < ~20deg (shoulder midpoint y barely rises relative to hip; scapula not lifted). Distinguish from head-only motion by checking that shoulder-midpoint vertical displacement — not just nose(0) displacement — stays small.
- **observability**: high — **side** view; sagittal-plane displacement of shoulders vs hips is directly measurable.
- **biomechanical_rationale**: The curl-up is defined as a lift "to the point where the scapula was lifted"; if the scapulae never clear, the target abdominal range is not reached and the exercise stimulus is lost (a performance/effectiveness fault rather than an injury one).
- **citation**: Barbado D et al. J Hum Kinet (2015). PMC4519219, DOI 10.1515/hukin-2015-0031; corroborated by Mandroukas A et al. PMC9505236.
- **citation_support**: Barbado defines the curl-up as "a head, arms and upper trunk lift to the point where the scapula was lifted from the force plate, then returning to the starting position"; Mandroukas: curl-up is a lift "with a rounded back to approximately 35-40deg from the floor." The lifted-scapula endpoint gives the observable ROM target. VERIFIED (read RAG docs).

---

### Shoulder Bridge (supine bridge)

Rep phases: **setup (supine, hips/knees flexed, feet flat)** → **concentric hip extension (lift pelvis)** → **top (isometric hold)** → **eccentric lower** → **rest**.

#### incomplete_hip_extension_top

- **fault_id**: `bridge_incomplete_hip_extension`
- **fault_name**: Incomplete hip extension at top
- **description**: At the top the pelvis is not lifted enough to bring shoulders, hips, and knees into a straight line, so the hips stay flexed and sagging.
- **detection_heuristic**: Hip angle (shoulder→hip→knee) at the top of the lift. Target is a straight line (~170-180deg, 0deg hip flexion); flag `incomplete_extension` if peak hip angle < ~160deg (hips remain visibly flexed / pelvis low). Use averaged left/right; peak taken over the top-hold frames.
- **observability**: high — **side** view; the shoulder–hip–knee angle is directly measurable in the sagittal plane.
- **biomechanical_rationale**: Gluteus-maximus recruitment and hip-extension torque are greatest near full hip extension, so stopping short under-loads the glutes and defeats the exercise's purpose.
- **citation**: Colonna S, D'Alessandro A, Tarozzi R, Casacci F. "Supine Bridge Exercise: A Narrative Review of the Literature (Part I)." Cureus (2025). PMC11981018, DOI 10.7759/cureus.80349; endpoint corroborated by Escamilla RF et al. Bioengineering (2024). PMC11048684, DOI 10.3390/bioengineering11040356.
- **citation_support**: Colonna: "the pelvis is lifted from the floor until it reaches the neutral angular position of the hip," and "The greatest hip extension torque during the SBE occurs when the hip is nearly fully extended. In this position, the GM is recruited more than at any other angle within the range of motion." Escamilla defines the endpoint as lifting "until the hips were in a neutral position with 0deg hip flexion, with the knees, hips, and shoulders approximately in a straight line." VERIFIED (read RAG docs).

#### lumbar_hyperextension_overarch

- **fault_id**: `bridge_lumbar_hyperextension`
- **fault_name**: Lumbar hyperextension / overarching
- **description**: The pelvis is pushed too high so the low back arches (anterior pelvic tilt / lumbar lordosis) instead of the shoulder-hip-thigh staying in a straight line — hip extension is replaced by back extension.
- **detection_heuristic**: Hip angle (shoulder→hip→knee) at top; flag `lumbar_hyperextension` if peak hip angle overshoots the straight line into extension (> ~190deg, i.e. hips rise above the shoulder–knee line forming an arch). Supporting proxy: hip-midpoint y at top rises above the straight line interpolated between shoulder-midpoint and knee-midpoint.
- **observability**: medium — **side** view. Global hip/back extension is visible; isolating lumbar vs hip contribution from surface landmarks is approximate, so this is a proxy for the arch.
- **biomechanical_rationale**: Excessive, uncontrolled lumbar lordosis and anterior pelvic tilt from dominant erector-spinae activity increase compression stress on the lumbar and pelvic regions and can lead to secondary dysfunction with repetition.
- **citation**: Colonna S, D'Alessandro A, Tarozzi R, Casacci F. Cureus (2025). PMC11981018, DOI 10.7759/cureus.80349.
- **citation_support**: "In patients performing bridging exercises, excessive and uncontrolled lumbar lordosis and anterior pelvic tilt (APT) are frequently observed due to the dominant hyperactivity of the ES. The repetitive motion associated with this activity could increase compression stress on the lumbar and pelvic regions." Some authors "recommend maintaining a straight alignment of the shoulders, hips, and thighs during bridging to prevent excessive APT caused by dominant ES activity." VERIFIED (read RAG doc).

#### asymmetric_pelvic_drop

- **fault_id**: `bridge_asymmetric_pelvic_drop`
- **fault_name**: Asymmetric pelvic drop (esp. single-leg bridge)
- **description**: One side of the pelvis sags/drops relative to the other during the hold — a Trendelenburg-like frontal-plane tilt, most common in the single-leg variant.
- **detection_heuristic**: Frontal-plane pelvic-tilt angle = angle of the left-hip(23)→right-hip(24) line vs horizontal, measured over the top-hold frames. Flag `pelvic_drop` if |pelvic-tilt| exceeds ~8-10deg (or a large asymmetry vs the setup baseline). For single-leg, the pelvis typically drops on the unsupported (swing-leg) side.
- **observability**: medium — needs **front** or **rear** view (frontal-plane tilt); nearly invisible from a pure side view. Front/rear_oblique partially usable.
- **biomechanical_rationale**: The gluteus medius must generate hip-abduction force to keep the pelvis level during single-leg support; when it fails, the pelvis drops on the opposite side (Trendelenburg), reducing stability and, over time, raising low-back and lower-limb load.
- **citation**: Colonna S, D'Alessandro A, Tarozzi R, Casacci F. Cureus (2025). PMC11981018, DOI 10.7759/cureus.80349.
- **citation_support**: "In a Trendelenburg gait, the Gmed is unable to maintain the pelvis on the opposite side during single-leg support, causing the pelvis to drop when the swing leg is in the air. This pelvic drop occurs when the Gmed fails to generate enough of an internal hip abduction force to counteract the external hip adduction force that happens during single-leg stance." VERIFIED (read RAG doc).

#### knee_valgus_bridge

- **fault_id**: `bridge_knee_valgus`
- **fault_name**: Knee valgus (knees collapse inward)
- **description**: During the lift the knees drift medially (toward each other) relative to the feet instead of tracking over them.
- **detection_heuristic**: Compare knee separation to ankle/foot separation in the frontal plane: `knee_width/ankle_width` where knee_width = |x(25)-x(26)| and ankle_width = |x(27)-x(28)|. Flag `knee_valgus` if this ratio drops below ~0.85 during the lift (knees closer together than feet) — mirrors the squat knees-inward heuristic.
- **observability**: medium — needs **front**/**front_oblique** view; not observable from side.
- **biomechanical_rationale**: Hip-abductor / external-rotator weakness allows excessive hip adduction and internal rotation, producing knee valgus that increases stress on the knee's ligamentous structures.
- **citation**: Colonna S, D'Alessandro A, Tarozzi R, Casacci F. Cureus (2025). PMC11981018, DOI 10.7759/cureus.80349.
- **citation_support**: "Powers theorized that hip abductor and external rotator weakness may lead to excessive hip adduction and internal rotation, resulting in increased knee valgus. This position can place excessive stress on the knee's ligamentous structures." VERIFIED (read RAG doc).

---

### Leg Abduction (side-lying / standing hip abduction)

Rep phases: **setup (neutral hip)** → **concentric abduction (lift/step leg out)** → **peak abduction** → **eccentric return (adduct)** → **rest**.

#### pelvic_drop_trunk_lean_compensation

- **fault_id**: `abd_pelvic_drop_trunk_lean`
- **fault_name**: Pelvic drop / trunk lateral-lean compensation (Trendelenburg-like)
- **description**: Instead of a clean isolated abduction, the pelvis drops or the trunk leans laterally to hike/tilt the pelvis and cheat the leg outward.
- **detection_heuristic**: Two coupled signals: (1) frontal-plane pelvic-tilt = angle of left-hip(23)→right-hip(24) line vs horizontal; (2) trunk lateral-lean = horizontal offset of shoulder midpoint (11,12) from hip midpoint (23,24), normalized by trunk length. Flag if pelvic-tilt change from setup > ~8-10deg OR lateral-lean exceeds ~0.10-0.15 of trunk length during the abduction phase.
- **observability**: medium/high for **standing** hip abduction from a **front**/**rear** view (frontal plane faces camera); medium for **side-lying** where trunk lean is visible from **side** view but pelvic drop is largely out-of-plane.
- **biomechanical_rationale**: When the hip abductors are weak, the body compensates with contralateral pelvic drop and/or ipsilateral trunk lean, which offloads the abductors and defeats the exercise while reflecting the same mechanism that produces Trendelenburg gait.
- **citation**: González-de-la-Flor Á. "Optimizing Hip Abductor Strengthening ... Monster Walk and Lateral Band Walk." J Funct Morphol Kinesiol (2025). PMC12372021, DOI 10.3390/jfmk10030294; corroborated by Rodrigues R et al. PLoS One (2025). PMC12416692, DOI 10.1371/journal.pone.0331553.
- **citation_support**: González-de-la-Flor: "weakness leads to a characteristic Trendelenburg gait or compensatory trunk lean," and "excessive sway or lateral trunk lean may reduce abductor demand by mechanically offloading the stance limb ... The optimal technique involves maintaining frontal plane neutrality." Rodrigues: hip-abductor weakness is compensated "by increasing ipsilateral trunk lean," and greater pelvic drop is associated with greater hip adduction in single-leg tasks. VERIFIED (read RAG docs).

#### hip_flexion_external_rotation_substitution

- **fault_id**: `abd_hip_flexion_er_substitution`
- **fault_name**: Hip flexion + external rotation substitution
- **description**: The leg drifts forward (hip flexion) and/or the toes turn up (external rotation) — recruiting TFL/hip flexors — instead of a pure frontal-plane abduction with a neutral hip.
- **detection_heuristic**: Forward-drift proxy (side view): during abduction the foot index(31/32)/ankle(27/28) x-position moves anteriorly past the hip's sagittal line by more than a small tolerance (foot ahead of hip), rather than moving purely laterally/vertically. External-rotation proxy: change in the foot-vector orientation (ankle→foot-index) indicating the toes rotating upward/outward. Flag when forward drift is large relative to the pure-abduction excursion.
- **observability**: low/medium — the forward-drift (hip-flexion) component needs a **side** view; external rotation from the foot-index landmark is a weak, noisy proxy under monocular pose, so do not over-trust it. Not reliably separable from true abduction on a **front** view alone.
- **biomechanical_rationale**: Substituting hip flexion and rotation shifts load from the gluteus medius to the TFL and hip flexors, reducing the targeted gluteal recruitment and reinforcing a faulty movement pattern; keeping frontal-plane neutrality is required for selective gluteal loading.
- **citation**: González-de-la-Flor Á. J Funct Morphol Kinesiol (2025). PMC12372021, DOI 10.3390/jfmk10030294.
- **citation_support**: The review stresses "maintaining frontal plane neutrality" and cueing to promote "gluteal over TFL recruitment," noting the TFL (a hip flexor/abductor) as the compensating muscle and that femoral torsion / rotation and posture "influence movement quality"; distal band placement "introduces a slight external rotation torque." The specific forward-drift/toes-up substitution is the clinical expression of losing frontal-plane neutrality. VERIFIED for the neutrality/TFL principle; the exact toes-up cue is inferred clinical description (mark support MODERATE).

#### insufficient_abduction_rom

- **fault_id**: `abd_insufficient_rom`
- **fault_name**: Insufficient abduction ROM
- **description**: The leg is lifted / stepped only a small amount, well short of a full abduction range.
- **detection_heuristic**: Peak abduction angle = angle of the thigh vector (hip 23/24 → knee 25/26) relative to the pelvis midline / vertical in the frontal plane. Flag `insufficient_rom` if peak abduction angle < ~25-30deg for standing/side-lying abduction (target range commonly ~30-45deg). For band walks, use step-width (ankle separation) staying below a per-user setup threshold.
- **observability**: medium/high — **front**/**rear** view for standing abduction; **side** view for side-lying (leg lifts in a plane visible edge-on).
- **biomechanical_rationale**: Side-lying hip abduction generates high gluteus-medius activation (~80% MVIC) through its range; abbreviating the range reduces the training stimulus and the strengthening effect that underpins pelvic stability and injury prevention.
- **citation**: González-de-la-Flor Á. J Funct Morphol Kinesiol (2025). PMC12372021, DOI 10.3390/jfmk10030294.
- **citation_support**: "side-lying hip abduction ... generates high levels of gluteus medius activation, reaching approximately 80% of maximal voluntary isometric contraction (MVIC) ... greater muscle activation than other closed-chain or multi-joint exercises such as clamshells, lunges, and hops"; the review emphasizes "optimal squat depth" and adequate excursion for effective loading. VERIFIED (read RAG doc). Note: the specific degree threshold is a practical target, not a value stated in the source.

#### momentum_uncontrolled

- **fault_id**: `abd_momentum`
- **fault_name**: Using momentum / uncontrolled swing
- **description**: The leg is swung ballistically and bounced back rather than lifted and lowered under control.
- **detection_heuristic**: Peak angular velocity of the thigh (hip→knee) abduction angle and its symmetry between concentric and eccentric phases; jerk (frame-to-frame acceleration) spikes at the top. Flag `momentum` if peak angular velocity greatly exceeds a per-user baseline OR the eccentric (return) phase is much faster than the concentric (a dropped/swung leg).
- **observability**: medium — **front**/**rear** for standing, **side** for side-lying; velocity is measurable but the "momentum" judgment is a proxy from kinematics.
- **biomechanical_rationale**: Swinging with momentum lets non-target tissues and gravity do the work instead of the gluteals, reducing the strengthening stimulus and the control component the exercise is meant to train.
- **citation**: González-de-la-Flor Á. J Funct Morphol Kinesiol (2025). PMC12372021, DOI 10.3390/jfmk10030294.
- **citation_support**: "Proper execution requires control of the trunk and pelvis, optimal squat depth, and consistent band tension," and technique guidance emphasizes "controlled steps" to ensure "the targeted muscles are effectively engaged." Support for the control requirement is explicit; the specific velocity thresholds are practical proxies (MODERATE support). VERIFIED (read RAG doc).

---

### Verification note

All citation_support paraphrases/quotes above were taken from the six RAG docs actually read in this session (PMC4519219, PMC9505236, PMC11981018, PMC11048684, PMC12372021, PMC12416692). Two items are honestly down-graded to MODERATE support where the exact clinical cue exceeds the literal wording of the source: the "toes-up external-rotation" component of `abd_hip_flexion_er_substitution`, and the velocity thresholds of `abd_momentum` (the sources support the frontal-plane-neutrality and movement-control principles, respectively, but do not state the specific pose thresholds). No fault rests on an unsupported injury-risk claim.

> **UPDATE (2026-08-09, Sit-up implemented — 11/16) — THREE OF THIS SECTION'S FOUR SIT-UP RULES DO
> NOT SHIP, AND THE CENTRAL FINDING IS ABOUT THIS GROUP AS A WHOLE, NOT ABOUT SIT-UP.**
> Design spec: `docs/superpowers/specs/2026-08-09-situp-detector-design.md`. Module:
> `src/pose/movements/situp.py`. `situp_incomplete_rom` ships; `situp_hip_flexor_dominance` is
> registered permanently silent; `excessive_speed` and `excessive_rom` are withdrawn.
>
> - **GROUP E'S MEASUREMENT CONVENTION IS NOT RECOVERABLE FROM AN IMAGE, AND THAT APPLIES TO ALL
>   THREE MOVEMENTS.** The convention block above defines trunk-flexion, pelvic-tilt and trunk
>   lateral-lean against "the floor/horizontal". The image horizontal is not the floor: EgoExo-
>   Fitness, the only dataset with labeled sit-ups, ships its two near-sagittal views (`exo_l`,
>   `exo_r`) **rotated a quarter turn with no EXIF orientation tag** (PIL `getexif()` is empty on
>   all three exo views). Every rule shipped in this project before Sit-up was immune by accident —
>   they all read joint-relative `angle_degrees(a, b, c)`, invariant under camera roll. Sit-up
>   re-anchors to the body (hip-angle excursion) deliberately. **Shoulder Bridge and Leg Abduction
>   must do the same**; their `hip angle`, `pelvic-tilt` and `knee_width/ankle_width` quantities are
>   already body-relative, but `lumbar_hyperextension_overarch`'s "hip-midpoint y rises above the
>   shoulder–knee line" proxy is not, and neither is any reading of pelvic tilt "vs horizontal".
> - **RE-ANCHORING FIXES THE REPRESENTATION, NOT THE ESTIMATOR — MEASURED.** Rotating 300 real
>   `zOfbr6/exo_l` frames by 90° and re-running MediaPipe moves the same frame's hip angle by a
>   **median 9.8° (p90 18.6, max 32.5)**, with detection succeeding 300/300 either way. MediaPipe is
>   not roll-equivariant. That residue is half the shipped 20° threshold, produced by camera roll
>   alone, and no landmark convention removes it.
> - **THE `side` RATINGS IN THIS SECTION ARE FICTION, AND THE ESTIMATOR IS NOT MERELY SILENT ON A
>   SUPINE SUBJECT — IT IS INVERTED.** Three of four Sit-up rules are rated on `side`, which
>   production has emitted on 0 of 49 clips. Run over the six real sit-up clips in all three
>   exocentric views: the **near-sagittal** `exo_l`/`exo_r` come back **`rear`** and the **head-on**
>   `exo_m` comes back **`rear_oblique`**, deterministically, with `side` and `unknown` never
>   emitted. `view_estimation.py`'s docstring limit 1 already forbids gating a horizontal-movement
>   rule on these labels; this is the measurement behind it. `situp_incomplete_rom` is accordingly
>   **the first shipped rule in the project with neither a view gate nor a view discount** — a
>   discount keyed on a meaningless label is arbitrary, not conservative.
> - **A FOURTH AND A FIFTH CITATION FAILURE MODE, after inference (impingement arc), absence (curl
>   wrist flexion) and exercise identity (all four Arm VW sources).**
>   **(4) SECONDARY SOURCING** — right paper, right exercise, but the paper is quoting someone else.
>   Both numbers this section draws from Mandroukas PMC9505236 are things he reports from other
>   literature behind reference markers: "limiting the amount of trunk flexion to 35–40° [ ]" and
>   "Nachemson [ ] reported increased pressure on the intervertebral disc". His own result is the
>   EMG finding that RA activity "decreased as the range of motion became greater, more than
>   35–40°".
>   **(5) SOURCE-MEASURED NULL ON THE PROPOSED PROXY** — `excessive_speed`'s secondary signal
>   ("medial-lateral wobble of the shoulder midpoint") *is* Barbado PMC4519219's `SG_ML`, and
>   Barbado's headline result is that it **did not change significantly with speed**. The quantity
>   that did rise, `COP_ML`, is force-plate centre of pressure and is not observable from video.
>   This is the only one of the five that survives checking what a source *says* and falls only to
>   checking what it *found*. Its `~1.0 s` threshold is separately just the fastest metronome
>   cadence tested, and its primary signal ("exceeds a per-user baseline") does not exist in this
>   architecture at all.
> - **THE FIRST RULE WITHDRAWN BECAUSE ITS KNOWLEDGE-GRAPH SEED WOULD BE SEMANTICALLY INVERTED.**
>   The graph's four `Sit-up:` fault nodes are the EgoExo TKV criteria — Feet Not Together, Arms Not
>   Extended Overhead (both dangling), Incomplete Forward Reach, Abdominal Disengagement. There is
>   no excessive-ROM node and the only ROM-adjacent one means the **opposite**. Band Pull Apart,
>   Bicep Curl, Arm Abduction and Arm VW all accepted **thin** seeds and Arm VW accepted a **shared**
>   one; none accepted an **inverted** one.
> - **THE SPEC AND THE APP MODEL DIFFERENT EXERCISES, and this section is the odd one out.** This
>   section says curl-up. The knowledge graph, EgoExo-Fitness's canonical guidance ("touch your feet
>   with your hands", faulted on 28/82 judged actions when not achieved), the frontend's Traditional
>   Chinese string (**仰臥起坐**, not 捲腹) and the shipped card artwork all say **full sit-up**. Two
>   of the three non-shipping outcomes turn on that disagreement. It is a product decision, recorded
>   in TODO.md, not taken here.
> - **A FOURTH VACUOUS-BRANCH DEFECT, AND THE FIRST CAUGHT BEFORE IMPLEMENTATION.**
>   `hip_flexor_dominance_anchored`'s heuristic asks that "shoulder–hip–knee remain close to
>   collinear" *while* "the trunk-thigh (hip) angle closes rapidly" — and this section's own
>   convention block defines the hip angle AS shoulder→hip→knee. Both clauses name the same
>   quantity, so the rule can never fire. Same class as `row.rule_momentum_jerk`'s second condition,
>   Bicep Curl's elbow-displacement disjunct and the impingement arc's first conjunct.
> - **A NEW REASON FOR `validated=False`, THE THIRD IN THIS REGISTRY.** Not "no labeled data exists"
>   (Deadlift, Row, Band Pull Apart, Bicep Curl) and not "nobody ran the check" (Arm Abduction, Arm
>   VW), but **the labeled data that exists describes a different variant**. REHAB24-6 has no sit-up
>   and Fit3D has no supine action among its 47 activity types, so there was no escape hatch of the
>   kind Arm Abduction used.
> - **ONE LIVE RULE IS THE HONEST OUTCOME.** Padding the detector to look comparable to Squat's five
>   would mean inventing thresholds.
>
> **What Group E still has going for it:** Leg Abduction is REHAB24-6 `Ex4` — **210 reps, 120
> correct / 90 incorrect, 12 videos** — the largest labeled non-squat set after Arm VW's 208 and the
> only Group E movement with matching-variant ground truth. It should be the best-evidenced detector
> in this group by a wide margin, and the Sit-up findings about reference frames and view labels
> apply to it directly.

> **UPDATE (2026-08-09, Shoulder Bridge implemented — 12/16) — ONE OF THIS SECTION'S FOUR
> SHOULDER BRIDGE RULES SHIPS, AND THE CENTRAL FINDING IS THAT `angle_degrees` IS UNSIGNED.**
> Design spec: `docs/superpowers/specs/2026-08-09-shoulder-bridge-detector-design.md`. Module:
> `src/pose/movements/shoulder_bridge.py`. `bridge_incomplete_hip_extension` ships;
> `bridge_lumbar_hyperextension` is registered permanently silent; `asymmetric_pelvic_drop` and
> `knee_valgus` are withdrawn.
>
> - **`angle_degrees` RETURNS `arccos`, RANGE [0, 180], AND THAT BREAKS TWO OF THESE FOUR RULES AT
>   ONCE.** `lumbar_hyperextension`'s "peak hip angle > ~190deg" **can never fire** — the FIFTH
>   vacuous-branch defect in this registry (after `row.rule_momentum_jerk`, Bicep Curl's elbow
>   disjunct, the impingement arc's first conjunct and `situp_hip_flexor_dominance`), and the second
>   caught BEFORE implementation. Worse, and NOT anticipated anywhere in this section: the function
>   is **exactly symmetric about 180deg**, so `incomplete_hip_extension`'s "< 160deg" also fires on
>   a bridge arched 20deg PAST neutral and reports it as one that never got there. Measured on a
>   synthetic fixture: +20deg and −20deg from the straight line both read **140.00deg**. The shipped
>   rule carries that mislabel, stated at its definition site and pinned by a test, because in the
>   only labeled data the direction it assumes is the direction annotators fault (16/77) and the
>   other direction is not among the twelve criteria at all.
> - **THE SIGN THAT WOULD REPAIR BOTH IS NOT RECOVERABLE — TWO CONSTRUCTIONS, BOTH MEASURED, BOTH
>   REFUTED.** (A) hip vs ANKLE about the shoulder→knee line is provably invariant under rotation
>   AND mirroring, recovers 120/160/180/200/240 on a synthetic fixture where the unsigned angle
>   gives 120/160/180/160/120 — and on real footage reads "arched" on **57.0%** and **62.3%** of
>   frames of repetitions annotators marked correct. (B) hip vs KNEE about the shoulder→ankle line
>   (the mat, being the two contact points) **disagrees with the subject's own other side** on 21 of
>   24 sampled frames. Near the straight line, where the rule must decide, both cross products go to
>   zero and the sign is noise. This is Sit-up's lesson recurring: re-anchoring fixes the
>   REPRESENTATION, not the ESTIMATOR.
> - **THE VIEW LABELS ARE NOT MERELY INVERTED ON A SUPINE SUBJECT — THEY ARE UNSTABLE.** Sit-up
>   measured a deterministic per-camera inversion. Re-measured on a different record and movement:
>   `rear` three times, `rear_oblique` three times, `side` and `unknown` never — and the SAME
>   CAMERA disagrees with itself between two clips of the same person in the same room (`exo_l` →
>   `rear` on one, `rear_oblique` on the other), at confidences from 0.02 to 0.72. So this is the
>   second shipped rule with neither a view gate nor a view discount.
> - **THE SHIPPED RULE FIRES ON 5 OF THE 6 REAL CLIP-VIEWS, ALL OF THEM CORRECT REPETITIONS, AND
>   THE CENSUS SPLITS BY CAMERA GEOMETRY.** On the four near-sagittal clip-views it is silent once
>   and otherwise fires at 0.02, 0.08, 0.15. On the two AXIAL (head-on, down the body's long axis)
>   clip-views it fires at **0.95 and 1.00** — because that camera foreshortens the sagittal hip
>   angle into meaninglessness (median 90deg vs 128–134deg on the sagittal cameras, same
>   repetitions). The same repetitions read **110.6 to 167.9deg** across three SIMULTANEOUS cameras:
>   a ~50–58deg spread against the 20deg margin this threshold relies on. That establishes the
>   MAGNITUDE of the measurement error and NOT a fire rate (n = 2 actions, 1 subject) and NOT a bias
>   direction — the tempting "MediaPipe under-reads a straight-line bridge by about 20deg" is
>   consistent with the data and is deliberately not claimed.
> - **TWO RULES WITHDRAWN ON EXERCISE IDENTITY, WHICH IS NOW THIS PROGRAMME'S MOST COMMON CITATION
>   FAILURE.** `asymmetric_pelvic_drop`'s citation_support quotes a passage that announces its own
>   subject in its first words — "**In a Trendelenburg gait** …" — and Escamilla, who studies
>   UNIPEDAL bridging directly, never mentions pelvic drop at all (checked). `knee_valgus`'s quotes
>   "**Powers [ ] theorized** …" from Colonna's section on hip dysfunction and knee pathology, whose
>   surrounding sentences are about ACL injury during LANDING. Neither fault is observed in a bridge
>   by either bridge source.
> - **AND `knee_valgus` FAILS INDEPENDENTLY ON MEASUREMENT.** Median `knee_width/ankle_width` across
>   the six clip-views: **0.726, 0.895, 0.911, 0.927, 1.020, 1.027**, per-clip minimum **0.043**,
>   against this section's 0.85 cut (squat's shipped one is 0.82). Two of six sit below the cut on
>   their MEDIAN frame, on repetitions judged correct. Its `|x(25)-x(26)|` form is also not
>   roll-invariant, and the codebase's own precedent (`pose_feature_extraction.py:296`) already uses
>   the full 2-D distance — the measurement above used the INVARIANT form, so fixing the form does
>   not rescue the rule.
> - **A FOURTH REASON FOR `validated=False`, AND THE FIRST THAT A DOWNLOAD FIXES.** Not "no labeled
>   data" (Deadlift, Row, Band Pull Apart, Bicep Curl), not "nobody ran the check" (Arm Abduction,
>   Arm VW), not "the labeled data describes a different variant" (Sit-up), but **the labels exist
>   and match and the PIXELS are missing**. EgoExo-Fitness has **77 human-judged Shoulder Bridge
>   actions / 130 annotator records**, guidance identical across all 77 and naming this rule's
>   endpoint verbatim, and one of its twelve criteria IS this rule ("Progressively raise your body
>   until your knees, hips, and shoulders align in a straight line", faulted **16/77**). The
>   `frames_open` archive is missing part `.ac`, so **2 of the 77** are recoverable. This is the most
>   actionable finding in the movement and is recorded in TODO.md as an action, not a limitation.
> - **THE GOOD NEWS THIS GROUP HAS NOT HAD: AN EXACT KG SEED AND A DOUBLY-PRIMARY ENDPOINT.**
>   `Shoulder Bridge:Incomplete Hip Extension` resolves with THREE non-empty buckets (causes: Poor
>   Hip Extension, Weak Gluteus Maximus; corrections: Squeeze Glutes) — the first seed in the whole
>   programme that is neither thin, shared, nor inverted. And the endpoint is stated in the OWN
>   WORDS of both sources with no reference marker: Escamilla's Methods give BOTH ends ("hips flexed
>   approximately 50deg" → "0deg hip flexion, with the knees, hips, and shoulders approximately in a
>   straight line"), which no rule in this programme has previously had. What is secondary here is
>   the RATIONALE, not the definition: Colonna's hip-extension-torque and GM-recruitment sentences
>   both carry reference markers, so this section's "VERIFIED" is true of the strings and not of
>   their authorship.
> - **`only_partial_reps` BITES HARDEST ON THE BEST FOOTAGE.** On the one clip-view with 100%
>   landmark detection, `segment_reps` found 2 repetitions and marked both partial, so the rule was
>   handed the entire 16-second clip as a single window. Same gap as the Deadlift setup-baseline
>   defect: `RunResult.fallback` is not threaded into `RuleContext`, so a rule cannot decline a
>   window handed to it by the whole-clip path. Recorded, not fixed.
>
> **What this leaves for Leg Abduction:** the note above still stands — REHAB24-6 `Ex4` is 210 reps
> of the matching variant and should make it the best-evidenced detector in Group E. Two findings
> here transfer directly: `pelvic-tilt vs horizontal` and `trunk lateral-lean` are BOTH specified
> against the image horizontal and both need re-anchoring, and `abd_insufficient_rom`'s "thigh
> vector relative to the pelvis midline / vertical" is a mixed case — the pelvis-midline reading is
> body-relative and survives, the vertical reading does not.

> **UPDATE (2026-08-09, Leg Abduction implemented — 13/16, GROUP E COMPLETE) — ONE OF THIS
> SECTION'S FOUR LEG ABDUCTION RULES SHIPS, AND FOR THE FIRST TIME IN THIS PROGRAMME THE LABELED
> DATA DECIDED THE ROSTER RATHER THAN COMMENTING ON IT.**
> Design spec: `docs/superpowers/specs/2026-08-09-leg-abduction-detector-design.md`. Module:
> `src/pose/movements/leg_abduction.py`. Validation: `notes/leg-abduction-rule-validation.md`.
> `abd_pelvic_drop_trunk_lean` ships — **its trunk-lean disjunct only**;
> `abd_insufficient_rom` is registered permanently silent; `abd_hip_flexion_er_substitution` and
> `abd_momentum` are withdrawn.
>
> - **THE CHECK RAN DURING DESIGN AND HAD AUTHORITY TO CHANGE THE ANSWER, WHICH IS NEW.** REHAB24-6
>   `Ex4` is standing unilateral hip abduction — 210 human-labeled repetitions, 120 correct / 90
>   incorrect, 9 subjects, 12 videos, two orthogonal cameras, and the variant matches the app's own
>   card art (verified by looking at the frames, not by reading the exercise name). Lunge was
>   checked after the fact and nothing changed. Here the run **silenced a rule**, **settled a
>   sub-clause**, and **corrected the working-side resolver's construction**. No threshold was
>   tuned: the shipped cut is this section's own ratio re-expressed by an identity, and the
>   silenced rule's cut is this section's own number, left where it is.
> - **THE SUPPORT LIMB IS THE VERTICAL THIS GROUP HAS BEEN MISSING, AND ONLY A STANDING MOVEMENT
>   HAS ONE.** The note above asked for `pelvic-tilt vs horizontal` and `trunk lateral-lean` to be
>   re-anchored. `hip_stance → ankle_stance` does it: in a standing unilateral exercise the stance
>   leg is planted and load-bearing, so it is a body-internal stand-in for gravity. Measured on the
>   frontal camera the stance limb sits a **median 2.3° from the image vertical (p90 4.5°)**, so
>   re-anchoring costs essentially nothing on this corpus and buys roll-invariance for production
>   video. Neither Sit-up nor Shoulder Bridge could take this route — both subjects are lying down.
> - **THE SIGN IS RECOVERABLE HERE, AND THE RULE IS "DOT PRODUCTS, NOT CROSS PRODUCTS".** Shoulder
>   Bridge's two refuted sign constructions were both cross products, which are roll-invariant but
>   ANTI-invariant under mirroring — the sign flips when the subject faces away from the camera,
>   which monocular pose cannot tell. Every signed quantity here is a projection onto a body axis,
>   invariant under both, pinned by a test asserting byte-identical detections under a 90° roll AND
>   under mirroring. Narrow claim, transferable: **prefer a projection onto a body axis; it needs no
>   argument about mirroring at all.**
> - **A NEW CITATION FAILURE MODE, THE SIXTH: THE CITATION AND THE MEASUREMENT DISAGREE ABOUT THE
>   SIGN OF THE FAULT.** This section's shipped rule is a DISJUNCTION of pelvic tilt and trunk lean.
>   The pelvic disjunct is not implemented, because three sources point two ways: the citation says
>   pelvic **DROP** (and its sentence opens "…a characteristic Trendelenburg **gait**"), the
>   knowledge graph has `Leg Abduction:Pelvic Hiking` and matches **zero** nodes for `Pelvic Drop`,
>   and 210 labeled repetitions separate on **HIKING**. Firing as written would fire on the
>   direction the data calls correct; firing the observed direction would be a rule with no
>   citation. Sit-up withdrew a rule for an inverted KG seed; this is the mirror case — graph and
>   data agree, the citation is the odd one out — resolved the same way. **It is only visible
>   because the sign is recoverable at all**; on Shoulder Bridge the same question was unanswerable.
>   And it is NOT free: ρ(trunk lean, pelvic hike) = **0.713**, so a real detection opportunity is
>   declined.
> - **STANDING UP DOES NOT FIX THE VIEW ESTIMATOR, AND THAT BREAKS THE REGIME BOUNDARY
>   `view_estimation.py` CLAIMS.** Limit 1 voids the front/rear/oblique labels for HORIZONTAL
>   subjects; Sit-up (inverted) and Shoulder Bridge (unstable) both measured failures inside that
>   regime. This subject is upright and `Ex4` records the true orientation per repetition, so the
>   labels could finally be checked: `front` → `rear_oblique` **116/116**, `half-profile` → `side`
>   **92/94**, and a `FRONTAL_OBSERVABLE_VIEWS` label emitted on **0 of 210**. Systematically
>   inverted, outside the documented regime. The shipped rule's confidence discount is therefore a
>   CONSTANT here and proves nothing about gating. Logged in TODO.md as an unscoped audit —
>   `squat.rule_knees_inward` and both `arm_abduction` frontal rules read these labels.
> - **THE SILENCED RULE HAS THE BEST KG SEED IN THE SECTION AND STILL DOES NOT SHIP.**
>   `Leg Abduction:Insufficient Abduction Range` resolves with two non-empty buckets, and the
>   metric is clean. What is missing is a number: no source states a range of motion (this section
>   admits as much), and this section's practical ~30° cut fires on **39/93 (42%) of repetitions
>   humans judged CORRECT** against 8/70 (11%) judged incorrect, with the cue's AUC at **0.206
>   pooled and below chance in all 9 subjects**. Silent rather than moved, because moving it is
>   fitting a threshold to labels.
> - **THE TWO RULES WITH NO GRAPH NODE ARE ALSO THE TWO WITHDRAWN ON CITATION GROUNDS** — but that
>   is the whole of the agreement, and it is weaker than it first looks. This movement has exactly
>   three `Leg Abduction:` fault nodes. Node-presence predicts the outcome in 2 of the 5 decisions
>   taken here: the other three all HAVE nodes and are a ship, a permanent silence, and a
>   not-implemented sub-clause. So the graph is a useful negative filter and **not** a predictor of
>   which rules survive. Recorded, not offered as a method.
> - **THE FIRST SIDE RESOLVER IN THE REGISTRY WITH GROUND TRUTH — AND IT CAUGHT A DESIGN ERROR.**
>   `exercise_subtype` names the working leg on all 210 repetitions. Final: **163 correct, 1 wrong,
>   11 declined** of the 175 that reached it (accuracy 0.994, coverage 0.937). The FIRST
>   construction referenced each thigh to the other leg, making both quantities approximately the
>   angle *between* the legs — it scored **7 correct / 14 wrong / 30 refused**, worse than a coin
>   flip, and the data is what exposed it.
> - **A FIFTH REASON FOR `validated=False`, AND THE FIRST THAT IS NOT A GAP.** Not "no labeled
>   data" (Deadlift, Row, Band Pull Apart, Bicep Curl), not "nobody ran the check" (Arm Abduction,
>   Arm VW), not "different variant" (Sit-up), not "the pixels are missing" (Shoulder Bridge), but
>   **the check ran, it changed the roster, and rep-level labels still cannot confirm a FAULT-level
>   claim.** REHAB24-6 never names which fault occurred.
> - **THE EIGHT-LANDMARK GATE IS EXPENSIVE AND THE COST IS STATED.** Requiring both ankles for the
>   support limb — two more than any other Group E module — yields a median validity rate of
>   **0.600 with p10 0.000**, and **35 of 210 repetitions (17%) end on a `segment_reps` fallback
>   path**. Those are excluded from every AUC, which are computed over 163/210 and say so.
> - **THE CAMERA CENSUS GOES THE OTHER WAY FROM SHOULDER BRIDGE'S, WHICH IS THE BENIGN DIRECTION.**
>   `front` 30 tp / 5 fp / 23 fn; `half-profile` 9 tp / **0 fp** / 28 fn. An oblique camera costs
>   SENSITIVITY, not precision — a lean projected obliquely reads smaller than it is, so the rule
>   goes quiet rather than wrong. Shoulder Bridge's axial views produced near-full-severity false
>   alarms instead.
>
> **Group E is complete: Sit-up 1 live, Shoulder Bridge 1 live, Leg Abduction 1 live.** Three
> movements, twelve spec rules, three shipped. That ratio is the honest one for this group and the
> reasons are per-rule rather than systemic — although the recurring theme is real: this section's
> measurement conventions were written against the image frame, and only the movement with a
> planted limb could recover one.


---

## Group F — Dynamic / rotational — Torso Twist, Jumping Jacks, High Knee

Movements: **Torso Twist (seated Russian twist / standing ab twist)**, **Jumping Jacks (side-straddle hop)**, **High Knee (running drill / march)**.

Detection model: MediaPipe Pose, 33 landmarks, normalized image coords (x,y ∈ [0,1], y increases downward), monocular. Landmark indices per shared context. Where a fault is not cleanly monocular-observable, the best geometric proxy is given and observability is downgraded honestly.

Sourcing note: these three movements have thin/zero peer-reviewed RAG coverage, so every citation below was found via web search and the specific finding was verified by fetching the source page (PubMed / PMC / journal). Wikipedia is used only as a supplementary *descriptive* cite, never as the sole support for an injury-risk claim.

---

### Torso Twist

Rep phases: **center (braced setup) → rotate to side A (peak) → return through center → rotate to side B (peak) → center**. Each side-swing = one rep. Seated Russian-twist geometry: hips fixed on floor, knees bent, torso held ~45° off the ground; rotation should come from the thoracic spine while the lumbar stays braced. A standing variant keeps the torso vertical.

#### tt_lumbar_rotation_dominant
- **fault_id**: `tt_lumbar_rotation_dominant`
- **fault_name**: Rotating from the lower back instead of the thoracic spine
- **description**: The pelvis/hip line swings around with the shoulders (whole-trunk twist) or the back rounds under the rotational load, so the twist is driven through the lumbar spine rather than the upper trunk.
- **detection_heuristic**: Compare rotation of the shoulder line (11→12 vector) against the hip line (23→24 vector) across the swing. In front/oblique view, twist is read as the change in projected horizontal separation of the paired landmarks (|x11−x12| and |x23−x24|) plus their left–right x-ordering flip. Flag when the hip-line rotation magnitude ≥ ~0.6 × the shoulder-line rotation magnitude (pelvis turning with the trunk instead of staying fixed), OR when the shoulder-midpoint drops/forwards relative to the hip midpoint by a rising margin through the rep (spinal rounding). Proxy only — true thoracic-vs-lumbar segmentation is not resolvable from 33 sparse landmarks.
- **observability**: low–medium; needs front or front_oblique / rear_oblique view. Not reliable from a pure side view (rotation is into/out of the image plane).
- **biomechanical_rationale**: Under axial rotation the trunk musculature co-activates mainly to *stabilize* the lumbar segment rather than to produce torque; letting the rotation collapse into a rounded/twisting lumbar spine removes that protective bracing and directs torsional load onto the passive disc and facet structures (a recognized torsional-injury pathway).
- **citation**: McGill, S.M. (1991). "Electromyographic activity of the abdominal and low back musculature during the generation of isometric and dynamic axial trunk torque: implications for lumbar mechanics." *Journal of Orthopaedic Research* 9(1):91–103. PMID 1824571. https://pubmed.ncbi.nlm.nih.gov/1824571/
- **citation_support**: VERIFIED (fetched). During maximal axial-torque efforts the obliques were the dominant abdominal actors (external oblique 52%, internal oblique 55% MVC vs rectus abdominis 22%), and the paper concludes "stabilization of the joints during twisting is far more important to the lumbar spine than production of large levels of axial torque," with the analytic model under-predicting torque (14 Nm predicted vs 91 Nm measured) — i.e. torsion is a stability problem and torsional loading is an injury mechanism for the annulus/facets.

#### tt_trunk_not_braced
- **fault_id**: `tt_trunk_not_braced`
- **fault_name**: Losing the braced upright torso (collapsing / rounding back)
- **description**: The torso drifts out of its held ~45° (seated) or vertical (standing) position — usually rounding the spine or sagging back toward the floor between swings.
- **detection_heuristic**: Track the trunk vector hip-midpoint→shoulder-midpoint angle relative to vertical. Establish a setup baseline in the first braced frame. Flag when the trunk angle deviates from baseline by > ~15°, or (seated, side view) when the shoulder-midpoint y falls toward the floor past the setup band, indicating the brace was dropped. Combine with a spinal-rounding proxy (shoulder-midpoint moving forward of the hip-midpoint in x on a side view).
- **observability**: medium; best from side view for the seated 45° hold, front/side for the standing variant.
- **biomechanical_rationale**: A maintained trunk brace keeps the co-activation that stabilizes the lumbar segments during rotation (see McGill above); a rounded or collapsed spine under a rotational/inertial load shifts torsional stress onto passive disc/ligament tissue.
- **citation**: McGill, S.M. (1991), *J Orthop Res* 9(1):91–103, PMID 1824571 (as above). Supplementary descriptive: Wikipedia, "Russian twist" (CC BY-SA) — `data/rag/docs/torso_twist_russian_wiki.txt`.
- **citation_support**: VERIFIED. McGill (fetched) supports the stabilization rationale. The RAG Wikipedia doc (read) supplies the technique target only: "the torso is kept straight with the back kept off the ground at a 45-degree angle" — descriptive support for the upright-brace geometry, not the injury claim.

#### tt_insufficient_rotation_rom
- **fault_id**: `tt_insufficient_rotation_rom`
- **fault_name**: Insufficient rotation range of motion
- **description**: The twist is shallow — the hands/shoulders barely cross past the body midline, so the obliques are not taken through a meaningful rotational range.
- **detection_heuristic**: Measure peak horizontal excursion of the wrist midpoint (15,16) relative to the hip-midline x (mean of 23,24) at each side-peak, and/or peak shoulder-line rotation. Flag a rep when the wrist midpoint fails to travel past the hip-midline x by more than a small band (e.g. |x_wrist_mid − x_hip_mid| < ~0.08 of shoulder width) on that side, indicating minimal rotation. Front / front_oblique view.
- **observability**: medium; front or oblique view (rotation projects poorly from the side).
- **biomechanical_rationale**: The Russian twist is prescribed specifically to load the internal/external obliques through trunk rotation; a truncated ROM under-recruits the very muscles that are the dominant rotators and trunk stabilizers, reducing the training stimulus (performance loss rather than injury).
- **citation**: McGill, S.M. (1991), *J Orthop Res* 9(1):91–103, PMID 1824571. https://pubmed.ncbi.nlm.nih.gov/1824571/
- **citation_support**: VERIFIED (fetched). McGill established the obliques as the dominant abdominal rotators during axial-torque generation (external oblique 52%, internal oblique 55% MVC) — a truncated twist ROM under-loads exactly these muscles. (Escamilla et al. 2006, *Phys Ther* 86(5):656–671, PMID 16649890, was fetched and checked but concerns abdominal *flexion* exercises, not rotation, so it is intentionally NOT cited here.)

#### tt_momentum_over_control
- **fault_id**: `tt_momentum_over_control`
- **fault_name**: Swinging with momentum instead of controlled rotation
- **description**: The arms/weight are flung side to side ballistically with no pause, so momentum rather than muscular control drives the twist.
- **detection_heuristic**: Temporal signal. Compute angular velocity of the wrist-midpoint about the hip-midpoint per frame; flag reps whose peak angular speed exceeds a tempo threshold and/or that show no near-zero-velocity dwell at the side-peaks (no control pause). Combine with rep cadence (reps/sec) above a set ceiling.
- **observability**: medium; any view that resolves the swing (front/oblique). This is a tempo heuristic, not a single-frame geometry.
- **biomechanical_rationale**: Ballistic momentum spikes the peak torsional load and shifts it onto passive tissue at end-range while reducing time-under-tension for the obliques; controlled tempo keeps the stabilizing co-activation engaged. Framed as combined control/injury concern.
- **citation**: McGill, S.M. (1991), *J Orthop Res* 9(1):91–103, PMID 1824571; supplementary descriptive: Wikipedia "Russian twist" (`data/rag/docs/torso_twist_russian_wiki.txt`).
- **citation_support**: VERIFIED (descriptive) + VERIFIED (mechanism). The RAG doc (read) explicitly states "The slower one moves the arms from side to side, the harder the exercise becomes" and warns not to rely on between-rep momentum — direct descriptive support for controlled tempo. McGill supplies the stabilization/torsion rationale.

---

> **UPDATE (2026-08-10, Torso Twist implemented — 14/16, GROUP F OPENS) — ONE OF THIS SECTION'S
> FOUR TORSO TWIST RULES SHIPS, AND THE TWO WITHDRAWALS WERE DECIDED BY EVIDENCE RATHER THAN
> JUDGEMENT: A PROJECTION MEASUREMENT AGAINST 3-D GROUND TRUTH, AND A SOURCE THAT PRESCRIBES THE
> BEHAVIOUR ITS OWN RULE FLAGS.**
> Design spec: `docs/superpowers/specs/2026-08-10-torso-twist-detector-design.md`. Module:
> `src/pose/movements/torso_twist.py`. Tests: `tests/test_torso_twist.py` (37 cases).
> `tt_trunk_not_braced` ships — **its brace disjunct only**; `tt_insufficient_rotation_rom` is
> registered permanently silent; `tt_lumbar_rotation_dominant` and `tt_momentum_over_control` are
> withdrawn.
>
> - **FOUR ARTIFACTS IN THIS PROJECT NAME "TORSO TWIST" AND THEY DESCRIBE FOUR DIFFERENT
>   EXERCISES.** This section's own rep phases, the RAG doc and the app card art all say **seated
>   Russian twist**, and that is the contract the module implements. The app's icon
>   (`MovementIcon.tsx:148`) draws a **standing** figure in both its comment and its strokes — an
>   asset defect, recorded and not fixed. Fit3D's `standing_ab_twists` is a **standing cross-body
>   knee-to-elbow twist** (looked at, not inferred from the name). EgoExo-Fitness's 95 judged
>   `Kneeling Side Torso Twist` actions are, by their own criteria text, a **prone lateral
>   flexion**. Nothing in this repository films the exercise the app depicts, so `validated=False`
>   for **Sit-up's** reason — the labeled data describes a different variant — held three times
>   over, and **not** a sixth distinct reason.
>
> - **THE KNOWLEDGE GRAPH'S THREE TORSO TWIST FAULTS ARE SEEDED FROM THE WRONG EXERCISE, AND THE
>   SEEDING SCRIPT SAYS SO IN ITS OWN WORDS.** `scripts/knowledge/stub_general_movements_v3.py:152`
>   records this movement's grounding as *"EgoExo-Fitness TKV (Kneeling Side Torso Twist:
>   pause-at-bottom 23%, lateral-flexion depth 21%, base 13%, abs)"*. That is PRIMARY provenance,
>   not an inference from node names, and it explains why a graph backing four **axial rotation**
>   rules contains `Torso Twist:Insufficient Lateral Flexion Depth`. Leg Abduction §7.3 established
>   that a MISSING node reliably predicts a rule should not exist while a PRESENT node predicts
>   nothing; this movement adds the sharper case — **a present node can be actively misleading**,
>   because it faithfully describes a different movement pattern. Sit-up refused an INVERTED seed;
>   this module refuses a WRONG-AXIS one.
>
> - **THIS SECTION'S PROJECTED-WIDTH ROTATION PROXY IS UNFIT, AND IT IS NOW MEASURED RATHER THAN
>   ARGUED.** The heuristic reads axial rotation as the change in `|x11−x12|` / `|x23−x24|`, i.e.
>   `width · |cos θ|` — a quantity whose derivative is **zero at the braced centre** and which is
>   **even in θ**, so it is blind exactly where the rule must discriminate and cannot tell one side
>   from the other. This section's remedy for the second defect, the left–right x-ordering flip,
>   requires **>90°** of rotation and the true relative trunk twist measured here peaks at a
>   **median 44.9° per repetition** (p90 54.1, max 58.8), so the flip never happens.
>   **Measured with a PERFECT detector** — Fit3D mocap ground truth projected through the real
>   calibration, 8 subjects × 4 cameras × 45 repetitions of `standing_ab_twists`, so every error is
>   projection alone: per-frame MAE **20.4°** on the shoulder line and **17.2°** on the hip line
>   (against a true hip peak of only 19.7°), and on the hip line — the ratio's decisive term — the
>   proxy is **anti-correlated with the truth on 35% of repetitions**. Carried to the decision the
>   rule makes, at this section's own 0.6 cut: truth fires 64/180, proxy fires 86/180, **disagreeing
>   on 30/180 = 16.7%**, of which **26 are the proxy firing where the truth does not**. The rank
>   correlation is 0.876, so the honest reading is that **the proxy is biased, not noisy** — and the
>   bias runs toward false positives. Small-angle resolution against a real floor: one degree of
>   rotation moves the shoulder width by 0.00016 of the image width at 0–15° and 0.00109 at 45–75°,
>   while MediaPipe's own frame-to-frame width movement over all 130 REHAB24-6 cached-landmark
>   videos is 0.000323 — **one frame of that is worth ~2.0° near the centre and ~0.30° near the
>   peak**. Harness: `scripts/fit3d/run_rotation_proxy_fidelity.py --jitter`.
>   *Variant caveat, stated:* `standing_ab_twists` has a FREE pelvis, so the truth *distribution*
>   of the ratio does not transfer to a seated twist with the hips pinned. What transfers is the
>   projection geometry, and no threshold was taken from this corpus.
>   **This also pays the debt the Row status note left open** — that Fit3D can support a
>   2-D-cue-vs-3-D-truth fidelity comparison even though it carries no correctness labels.
>
> - **A SEVENTH CITATION FAILURE MODE: THE PARAPHRASE INVERTS THE SOURCE'S INSTRUCTION.**
>   `tt_momentum_over_control` flags repetitions showing "no near-zero-velocity dwell at the
>   side-peaks (no control pause)", and its `citation_support` claims the RAG doc "warns not to rely
>   on between-rep momentum". Read in place, the doc says it is *"crucial to **not stop** between
>   repetitions or else one will lose the effect of working the abdomen"* — an instruction to keep
>   moving. The rule would fault a user for obeying its own source. This is sharper than Leg
>   Abduction's citation/observation sign disagreement, because **the contradiction is inside the
>   quoted document**.
>
> - **ALL FOUR RULES REST ON ONE PAPER THAT NEVER MENTIONS THE EXERCISE.** McGill 1991 (PMID
>   1824571), re-fetched: 25 adults, isometric plus dynamic axial twists at 30 and 60 °/s, EMG +
>   kinematics + torque. It supports the *mechanism* primarily and in his own words ("stabilization
>   of the joints during twisting is far more important to the lumbar spine than production of
>   large levels of axial torque"; obliques 52/55% MVC vs rectus abdominis 22%) and supplies **no
>   range of motion, no tempo cut, no thoracic-vs-lumbar contribution claim, and no exercise**. Its
>   30/60 °/s are protocol conditions performed by healthy subjects; adopting either as a fault cut
>   would convert a condition into a fault. This is a new shading of the exercise-identity mode —
>   Arm VW's sources were about *adjacent* exercises, whereas McGill is not about an exercise at
>   all.
>
> - **THE THIRD PER-USER-BASELINE WALL, AND IT IS NOW THE MOST COMMON SINGLE BLOCKER.**
>   `tt_insufficient_rotation_rom` has a working, roll- and mirror-invariant metric and a real
>   fault, and is silenced only because no source states a range; the obvious repair — "this swing
>   is shorter than your own usual" — needs cross-clip state this architecture does not have. Same
>   wall as `situp_excessive_speed` and `abd_momentum`.
>
> - **NO VIEW GATE AND NO VIEW DISCOUNT, FOR THE FOURTH TIME, WITH A NEW REASON.** `view_estimation`
>   limit 1 voids the labels for a HORIZONTAL subject and Leg Abduction measured them systematically
>   inverted on an UPRIGHT one. A seated twister's trunk is held at ~45°, i.e. **between two regimes
>   in both of which the labels have been measured wrong**, and no seated-twist footage exists here
>   to settle it.
>
> - **WHAT CAMERA PLACEMENT ALONE COSTS THE SHIPPED RULE, MEASURED.** Four simultaneous Fit3D
>   cameras, mocap-2D: the **absolute** trunk-thigh angle is robust (cross-camera spread of the
>   per-rep median **4.5°**, p90 10.6) while the **signed sag the rule scores** — median value only
>   6.3° on that corpus — has a spread of **5.1°, p90 15.7°**, so a maximum over a window is less
>   camera-robust than the angle it is built from and its p90 disagreement is the size of the 15°
>   cut. Caveat that binds hardest: `standing_ab_twists` moves the trunk mostly in FORWARD FLEXION,
>   the direction this rule does not score, so these figures bound the sag direction only weakly.
>
> - **AN UNSIGNED DEVIATION FROM A SETUP BASELINE IS ACTIVELY INVERTED, FOR THE SECOND TIME IN THIS
>   REGISTRY.** This section says the trunk angle "deviates from baseline by > ~15°", and the first
>   implementation took that literally. `trunk_thigh_angle_deg` is monotone in sag, so an `abs()`
>   fires on a twister who TIGHTENED: measured on the shipped path, a subject setting up loose at
>   95° and tightening to 50° was reported "Braced Torso Lost" at **severity 1.0**. The baseline
>   makes that the ordinary case rather than an edge one, because `setup` is the frames BEFORE the
>   subject braces. `pushup_head_drop` records the identical finding in §8 of this document; the
>   shipped rule is directional, which introduces no new number and fires on a strictly smaller
>   set. Every fixture ramped in the sag direction, so no green test could have caught it.
>
> - **THE SETUP-BASELINE DEFECT, MEASURED AND ATTRIBUTED.** Effective threshold **18.0°** against a
>   nominal 15.0 (1.20×) through the real `run_detector` — and **none of it is Row's trimming**,
>   because on the fixture `segment_reps` returns the windows untrimmed; the whole residual is the
>   3-frame `setup` median already carrying part of the ramp, `15 / (1 − f) = 17.36°`. Stated as a
>   property of that fixture, not a proof the trimming cannot bite on a swing that does not start
>   from rest.
>
> - **FIRST USER OF `rep_rectify`.** `base.py:55` declared the flag for this movement by name in
>   RS-SP1 and it had no user until the fourteenth detector. One swing is one repetition, which is
>   the RAG doc's own definition.

---

### Jumping Jacks

Rep phases: **closed (feet together, arms at sides) → open (feet spread wide, arms overhead) → landing back to closed**. Impact/landing events occur at each touchdown (both the open-stance touchdown and the return-to-closed touchdown). Landmarks: shoulders 11/12, wrists 15/16, hips 23/24, knees 25/26, ankles 27/28, nose 0.

#### jj_knee_valgus_landing
- **fault_id**: `jj_knee_valgus_landing`
- **fault_name**: Knee valgus (inward collapse) on landing
- **description**: On touchdown the knees cave inward relative to the ankles/feet (frontal-plane knee collapse).
- **detection_heuristic**: At the landing frame compute knee-width/ankle-width = |x25−x26| / |x27−x28|. Flag valgus when the ratio < ~0.82 (mirrors the coded squat knees-inward rule), i.e. knees are drawn inside the ankle base. Optionally confirm each knee_x sits medial to its same-side ankle_x. Front view.
- **observability**: high; requires front (or front_oblique) view. Not observable from a pure side view.
- **biomechanical_rationale**: Landing in dynamic knee valgus shifts impact absorption away from the hip and onto the knee, raising knee load and the associated (ACL/patellofemoral) injury risk.
- **citation**: Tamura, A. et al. (2017). "Dynamic knee valgus alignment influences impact attenuation in the lower extremity during the deceleration phase of a single-leg landing." *PLoS ONE* 12(6):e0179810. https://pmc.ncbi.nlm.nih.gov/articles/PMC5478135/
- **citation_support**: VERIFIED (fetched). Valgus landers had significantly greater knee angular impulse (0.093 vs 0.045 Nms/kg·m, p<0.01) and lower hip angular impulse (0.019 vs 0.067, p<0.01) than varus/neutral landers; the paper concludes valgus "may increase the impact the knee joint needs to attenuate," shifting load off the hip and toward the knee — a documented injury-risk pattern.

#### jj_stiff_landing
- **fault_id**: `jj_stiff_landing`
- **fault_name**: Stiff (hard) landing with insufficient knee flexion
- **description**: The athlete lands on near-straight legs with little knee bend, producing a hard, high-impact touchdown instead of absorbing through flexion.
- **detection_heuristic**: At/just after each touchdown compute knee angle (23→25→27 and 24→26→28). Flag a stiff landing when peak knee flexion stays shallow — knee angle remaining > ~160° (i.e. maximum flexion < ~20° of bend) through the impact window, or knee-flexion excursion from touchdown to trough below a set band. Side or front_oblique view gives the cleanest knee angle; front view is an approximation.
- **observability**: medium–high; best from side / oblique. Frontal-only view under-reads sagittal knee flexion.
- **biomechanical_rationale**: Stiff landings generate substantially larger ground reaction forces than soft (deeper-flexion) landings, concentrating impact on joints and passive tissue; greater knee/hip flexion lets the large hip and knee musculature dissipate landing energy.
- **citation**: DeVita, P. & Skelly, W.A. (1992). "Effect of landing stiffness on joint kinetics and energetics in the lower extremity." *Medicine & Science in Sports & Exercise* 24(1):108–115. PMID 1548984. https://pubmed.ncbi.nlm.nih.gov/1548984/
- **citation_support**: VERIFIED (fetched). Soft vs stiff landings averaged 117° vs 77° knee flexion, and "the stiff landing had larger GRFs"; in soft landings the hip and knee muscles were the major energy absorbers while in stiff landings the ankle muscles dominated. The ≥/<90° knee-flexion soft/stiff convention originates here.

#### jj_incomplete_arm_rom
- **fault_id**: `jj_incomplete_arm_rom`
- **fault_name**: Incomplete arm range of motion
- **description**: The arms fail to reach fully overhead at the open phase (hands stop around shoulder/head height).
- **detection_heuristic**: At the open-phase peak, check wrist height vs head: flag when both wrists fail to rise above the nose (y15 and y16 > y0, remembering y increases downward) or above a shoulder-referenced line. Front view.
- **observability**: high; front or front_oblique view.
- **biomechanical_rationale**: Full overhead arm travel is the defining ROM of the movement (side-straddle hop); truncating it reduces the shoulder/cardio work — largely a performance/completeness fault. (Note: the RAG doc also records that repetitive full-overhead jacks have been linked to rotator-cuff irritation, which is why reduced-ROM "half-jacks" exist — so extreme forced ROM is not automatically safer; this rule targets clearly incomplete reps.)
- **citation**: Wikipedia, "Jumping jack" (CC BY-SA) — `data/rag/docs/jumping_jacks_wiki.txt`.
- **citation_support**: VERIFIED (descriptive, read). "The hands go overhead, sometimes in a clap, and then return to a position with the feet together and the arms at the sides" defines the full-ROM target; the half-jack passage notes half-jacks "were created to prevent rotator cuff injuries, which have been linked to the repetitive movements of the exercise." No peer-reviewed source found for this specific ROM fault — descriptive support only.

#### jj_incomplete_leg_rom
- **fault_id**: `jj_incomplete_leg_rom`
- **fault_name**: Incomplete leg abduction (narrow stance) at open phase
- **description**: The feet do not spread to a full wide stance at the open phase — a shuffled, narrow jack.
- **detection_heuristic**: At the open-phase peak compute ankle-width/shoulder-width = |x27−x28| / |x11−x12|. Flag when the ratio stays below ~1.3 (feet barely wider than the shoulders) instead of the wide side-straddle. Front view.
- **observability**: high; front / front_oblique view.
- **biomechanical_rationale**: The wide-legged spread is the defining lower-body ROM of the exercise; a narrow stance reduces the intended abductor/adductor and cardio load. Performance/completeness fault.
- **citation**: Wikipedia, "Jumping jack" (CC BY-SA) — `data/rag/docs/jumping_jacks_wiki.txt`.
- **citation_support**: VERIFIED (descriptive, read). Defined as "jumping to a position with the legs spread wide" then back "with the feet together." No peer-reviewed source located for the narrow-stance fault specifically — descriptive support only.

#### jj_landing_asymmetry
- **fault_id**: `jj_landing_asymmetry`
- **fault_name**: Left/right asymmetry
- **description**: One side consistently reaches, lands, or absorbs differently from the other (uneven arm reach, foot spread, or single-side knee collapse).
- **detection_heuristic**: Per rep compare paired quantities L vs R: wrist peak height (y15 vs y16), ankle lateral excursion from hip-midline, and per-side knee-valgus ratio. Flag when a normalized L–R difference exceeds ~15–20% consistently across reps. Front view.
- **observability**: medium; front view; requires multi-rep consistency to separate true asymmetry from noise.
- **biomechanical_rationale**: Consistent side-to-side asymmetry in landing mechanics concentrates load on one limb; the valgus-side limb carries the elevated knee load documented by Tamura et al., making persistent single-side collapse the higher-risk pattern.
- **citation**: Tamura, A. et al. (2017), *PLoS ONE* 12(6):e0179810, PMC5478135 (as above).
- **citation_support**: VERIFIED. Same fetched Tamura finding — the limb landing in valgus bears the greater knee impulse; asymmetry means one limb is repeatedly that limb. Asymmetry-as-fault is an application of that finding, not a separately measured asymmetry study (noted honestly).

---

### High Knee

Rep phases (running-drill / march): **drive (rapid hip flexion to peak knee-up) → foot strike (stance) → alternate to opposite knee-up**. Single-leg support alternates each stride. Landmarks: nose 0, shoulders 11/12, hips 23/24, knees 25/26, ankles 27/28.

#### hk_insufficient_knee_lift
- **fault_id**: `hk_insufficient_knee_lift`
- **fault_name**: Insufficient knee-lift height (limited hip flexion ROM)
- **description**: The driving thigh stays low — it does not reach the target elevation (thigh at least ~45° above horizontal, approaching parallel in the higher-lift variants), so the hip-flexion ROM that defines the drill is not reached.
- **detection_heuristic**: At the drive peak measure vertical relation of the driving knee to the same-side hip: in image coords (y down) flag when the knee never rises to near hip height, e.g. (y_knee − y_hip) stays > +0.05 (knee well below the hip). Equivalently estimate thigh-to-horizontal angle from the hip→knee vector and flag when peak thigh elevation stays below the ~45° target. Side view is most accurate; front view is a usable proxy.
- **observability**: high; side view preferred, front acceptable.
- **biomechanical_rationale**: The drill's coaching target is a high knee drive with the thigh reaching at least ~45° above horizontal (higher in the more advanced variants); failing to reach it removes the hip-flexion swing-phase stimulus the drill exists to train (performance loss).
- **citation**: Matijašević, P. et al. (2025). "Development and validation of a running drill test battery to predict 5 m and 20 m sprint performance." *International Journal of Exercise Science* 18(8):1269–1285. https://pmc.ncbi.nlm.nih.gov/articles/PMC12591607/ (DOI 10.70252/LYKE8231)
- **citation_support**: VERIFIED (fetched). The A-skip scoring criterion required "the thigh to reach approximately 45° relative to the ground" during the swing (vs the B-skip's higher lift), and the drill's purpose is described as "promoting knee elevation and optimizing kinematics" — direct support for a knee-lift-height target. (The paper also found A-skip had only trivial correlation with sprint outcome — so frame this as a technique/ROM target, not a performance guarantee.)

#### hk_trunk_lean_back
- **fault_id**: `hk_trunk_lean_back`
- **fault_name**: Leaning the trunk backward to hoist the knee
- **description**: The athlete throws the torso backward (lumbar hyperextension) to swing the knee up rather than driving it with the hip flexors.
- **detection_heuristic**: Track the trunk vector hip-midpoint→shoulder-midpoint angle vs vertical, side view; flag when the shoulder-midpoint x moves *behind* the hip-midpoint x and the backward trunk-lean angle exceeds ~10–15° at the knee-drive peak. Establish an upright baseline at stance. Side / side_oblique view.
- **observability**: medium; side or oblique view (backward lean is a sagittal cue). Not resolvable from a pure front view.
- **biomechanical_rationale**: Leaning back substitutes trunk/lumbar extension for genuine hip flexion, cheating the knee height while loading the lumbar spine into repeated hyperextension; the drill requires an upright, controlled body position for the intended sprint-swing mechanics.
- **citation**: Matijašević, P. et al. (2025), *Int J Exerc Sci* 18(8):1269–1285, PMC12591607 (as above).
- **citation_support**: VERIFIED (posture target) / UNVERIFIED (injury magnitude). The fetched paper frames these drills as training "proper body position" and swing-phase kinematics, supporting an upright-trunk criterion. No source specifically quantifying lumbar-hyperextension injury risk in the high-knee drill was found — the injury framing is mechanistic inference, marked here as a gap.

#### hk_forward_trunk_collapse
- **fault_id**: `hk_forward_trunk_collapse`
- **fault_name**: Forward trunk collapse
- **description**: The trunk pitches forward at stance/mid-support instead of staying tall, folding the athlete over the support leg.
- **detection_heuristic**: Trunk vector (hip-mid→shoulder-mid) angle vs vertical, side view; flag when the shoulder-midpoint moves forward of the hip-midpoint and the forward-lean angle at midstance exceeds ~15–20° beyond the upright baseline. Side / side_oblique view.
- **observability**: medium; side or oblique view.
- **biomechanical_rationale**: Greater forward trunk lean at midstance is one of the kinematic patterns that distinguishes injured runners from healthy runners, alongside contralateral pelvic drop — a control fault linked to running-related soft-tissue injury.
- **citation**: Bramah, C., Preece, S.J., Gill, N., Herrington, L. (2018). "Is There a Pathological Gait Associated With Common Soft Tissue Running Injuries?" *American Journal of Sports Medicine* 46(12):3023–3031. PMID 30193080. https://pubmed.ncbi.nlm.nih.gov/30193080/
- **citation_support**: VERIFIED (fetched, PubMed abstract). Injured runners "demonstrated greater contralateral pelvic drop (CPD) and forward trunk lean at midstance" than healthy runners, consistent across all four injury subgroups (PFP, ITBS, MTSS, Achilles tendinopathy) — direct support for forward trunk lean as an injury-associated gait fault.

#### hk_contralateral_pelvic_drop
- **fault_id**: `hk_contralateral_pelvic_drop`
- **fault_name**: Contralateral pelvic drop
- **description**: During single-leg support the pelvis tilts so the swing-leg-side hip drops below the stance-side hip (frontal-plane hip-hike failure / Trendelenburg-type pattern).
- **detection_heuristic**: During single-leg stance measure the hip-line tilt = signed (y23 − y24) normalized by hip width |x23−x24|; identify the stance vs swing side from which foot is grounded (lower ankle y). Flag when the swing-side hip drops relative to the stance-side hip beyond a threshold (e.g. pelvic-obliquity angle > ~5–8°, tuned to landmark noise). Front / front_oblique or rear view.
- **observability**: high; front or rear view. A pure side view cannot see frontal-plane pelvic obliquity.
- **biomechanical_rationale**: Contralateral pelvic drop is the single kinematic variable most strongly associated with common running-related soft-tissue injuries; each additional degree markedly raises injury odds, reflecting loss of frontal-plane pelvic (hip-abductor) control.
- **citation**: Bramah, C. et al. (2018), *Am J Sports Med* 46(12):3023–3031, PMID 30193080 (as above).
- **citation_support**: VERIFIED (fetched, PubMed abstract). "CPD was found to be the most important variable predicting the classification of participants as healthy or injured," and "for every 1° increase in pelvic drop, there was an 80% increase in the odds of being classified as injured." (Caveat noted: McCarney et al. 2020, *Chiropr Man Therap* 28:53, PMC7570029 — fetched — found only ~4.6° average drop and NO correlation between the observed Trendelenburg pelvic drop and measured hip-abductor strength in healthy adults, so the *drop* is the observable injury-associated signal here, not a direct readout of abductor weakness.)

#### hk_stride_asymmetry
- **fault_id**: `hk_stride_asymmetry`
- **fault_name**: Left/right stride asymmetry
- **description**: One leg consistently drives to a lower knee height, or one side shows more pelvic drop / trunk shift than the other.
- **detection_heuristic**: Compare peak knee-lift height (knee-to-hip y relation) and per-side pelvic-drop angle between left and right strides across reps; flag when the normalized L–R difference exceeds ~15–20% consistently. Front view (knee height + pelvic drop) with multi-rep aggregation.
- **observability**: medium; front / oblique view, needs several strides to be reliable.
- **biomechanical_rationale**: Persistent side-to-side asymmetry concentrates the injury-associated patterns (pelvic drop, reduced hip flexion) on one limb; contralateral pelvic drop is the strongest injury-associated running variable, so a habitually dropping/under-driving side is the elevated-risk limb.
- **citation**: Bramah, C. et al. (2018), *Am J Sports Med* 46(12):3023–3031, PMID 30193080 (as above).
- **citation_support**: VERIFIED (application). Bramah's CPD finding supports singling out the worse side; the asymmetry framing is an application of that result, not a dedicated asymmetry study (stated honestly).

---

### Summary of citations used

| Movement | Primary peer-reviewed source(s) | Status |
|---|---|---|
| Torso Twist | McGill 1991, *J Orthop Res* (PMID 1824571) | fetched & verified (Escamilla 2006 checked, found off-topic, dropped) |
| Jumping Jacks | Tamura 2017, *PLoS ONE* (PMC5478135); DeVita & Skelly 1992, *MSSE* (PMID 1548984) | both fetched & verified |
| High Knee | Matijašević 2025, *Int J Exerc Sci* (PMC12591607); Bramah 2018, *Am J Sports Med* (PMID 30193080) | both fetched & verified |

Wikipedia RAG docs (`torso_twist_russian_wiki.txt`, `jumping_jacks_wiki.txt`) used as supplementary descriptive support only. High Knee had NO RAG doc (known gap) — fully covered by web-sourced peer-reviewed literature.

Honest gaps / UNVERIFIED items:
- `jj_incomplete_arm_rom`, `jj_incomplete_leg_rom`: only descriptive (Wikipedia) support; no peer-reviewed source found for these completeness faults.
- `hk_trunk_lean_back`: upright-posture target verified; the lumbar-hyperextension *injury* magnitude is mechanistic inference, no dedicated source found.
- `hk_contralateral_pelvic_drop`: pelvic-drop→injury link strongly verified (Bramah); pelvic-drop→abductor-weakness link is contested (McCarney 2020) and deliberately not asserted.


---

## 6. References index

RAG-corpus sources (authoritative author/title/journal from `data/paper_metadata.json`), keyed by PMCID:

- **PMC11048684** — Escamilla RF, Thompson IS, Carinci J, et al. (2024). Effects of Ankle Position While Performing One- and Two-Leg Floor Bridging Exercises on Core and Lower Extremity Muscle Recruitment. Bioengineering (Basel, Switzerland). PMC11048684.
- **PMC11981018** — Colonna S, D'Alessandro A, Tarozzi R, Casacci F. (2025). Supine Bridge Exercise: A Narrative Review of the Literature (Part I). Cureus. PMC11981018.
- **PMC12029123** — Mun WL, Jung EY, Lei S, Roh SY. (2025). Scapular Muscle Activation at Different Shoulder Abduction Angles During Pilates Reformer Arm Work Exercise. Medicina (Kaunas, Lithuania). PMC12029123
- **PMC12148905** — Hanen NC et al. Frontiers in bioengineering and biotechnology (2025). PMC12148905
- **PMC12225233** — Moreira VM et al. Muscles (Basel, Switzerland) (2023). PMC12225233
- **PMC12335237** — Abiara S, Heinrichs V, Chorneyko A, Lang AE. (2025). Acute effects of lower trapezius activation exercises on shoulder muscle activation during overhead functional tasks in symptomatic and asymptomatic adults. PeerJ. PMC12335237
- **PMC12366113** — Abdollahi S, Sheikhhoseini R, Salsali M, Piri H, Hides JA. (2025). The influence of hand position on scapular kinematics in push-ups: comparing athletes with chronic shoulder pain and healthy controls. Journal of orthopaedic surgery and research. PMC12366113.
- **PMC12372021** — González-de-la-Flor Á. (2025). Optimizing Hip Abductor Strengthening for Lower Extremity Rehabilitation: A Narrative Review on the Role of Monster Walk and Lateral Band Walk. Journal of functional morphology and kinesiology. PMC12372021.
- **PMC12372072** — Evangelista P, Rum L, Picerno P, Biscarini A. (2025). Decoding the Contribution of Shoulder and Elbow Mechanics to Barbell Kinematics and the Sticking Region in Bench and Overhead Press Exercises: A Link-Chain Model with Single- and Two-Joint Muscles. Journal of functional morphology and kinesiology
- **PMC12416692** — Rodrigues R, Sonda FC, Frigotto MF, et al. (2025). Sex as a moderator of the relationship between hip abduction strength and muscle activation during single-leg stance. PloS one. PMC12416692.
- **PMC12514857** — Al Hammadi MI, Shah ZA, Rathod RK, Seddik MA. (2025). Shoulder Impingement Pain Syndrome: Pathophysiology, Diagnosis, and a Review of Current Treatment Strategies. Cureus
- **PMC12550948** — Parpa K, Vasiliou A, Michaelides M, et al. (2025). An Exploratory Study of Biceps Brachii Electromyographic Activity During Traditional Dumbbell Versus Bayesian Cable Curls. Muscles (Basel, Switzerland). PMC12550948
- **PMC12734928** — Jung EY, Roh SY, Mun WL. (2025). Electromyographic Patterns of Scapular Muscles During Four Variations of Protraction-Retraction Exercises. Life (Basel, Switzerland). PMC12734928
- **PMC12821611** — Padovan R, Cè E, Longo S, et al. (2025). High-Density Surface Electromyography Excitation of Prime Movers Across Scapular Positions in the Seated Row. Journal of functional morphology and kinesiology.
- **PMC13086636** — Gregori P, La Bruna M, Papalia GF, Giurazza G, Caria C, Paciotti M, Russo F, Franceschetti E, Longo UG, Papalia R. (2026). Spine alignment influences shoulder range of motion and scapular orientation: A systematic review from the FP-UCBM Shoulder Study Group. Journal of experimental orthopaedics
- **PMC13116542** — Abdelraouf OR, Abdel-Aziem AA, Alkhamees NH, Ibrahim ZM, Aboelela EM, Dawood RS, Ashour AA. (2026). Acute Effects of High-Load Training to Failure vs. Non-Failure on Posture and Core Endurance in Collegiate Weightlifters: A Crossover Study. Journal of clinical medicine
- **PMC13232157** — Owens LP, Coyles G, Khaiyat O (2026). Whole Body Kinetic Chain Muscle Activity during selected Rehabilitation Exercises in Healthy and Injured Overhead Throwing Athletes. International journal of sports physical therapy.
- **PMC3820220** — Lee S, Lee D, Park J. (2013). The Effect of Hand Position Changes on Electromyographic Activity of Shoulder Stabilizers during Push-up Plus Exercise on Stable and Unstable Surfaces. Journal of physical therapy science. PMC3820220.
- **PMC4327800** — San Juan JG, Suprak DN, Roach SM, Lyda M. (2015). The effects of exercise type and elbow angle on vertical ground reaction force and muscle activity during a push-up plus exercise. BMC musculoskeletal disorders. PMC4327800.
- **PMC4519219** — Barbado D, Elvira JL, Moreno FJ, Vera-Garcia FJ. (2015). Effect of Performance Speed on Trunk Movement Control During the Curl-Up Exercise. Journal of human kinetics. PMC4519219.
- **PMC4556293** — Ford KR, Nguyen AD, Dischiavi SL, Hegedus EJ, Zuk EF, Taylor JB. (2015). An evidence-based review of hip-focused neuromuscular exercise interventions to address dynamic lower extremity valgus. Open access journal of sports medicine. PMC4556293.
- **PMC6523035** — Zellmer M, Kernozek TW, Gheidi N, Hove J, Torry M. (2019). Patellar tendon stress between two variations of the forward step lunge. Journal of sport and health science. PMC6523035.
- **PMC6548056** — Soriano MA, Suchomel TJ, Comfort P. (2019). Weightlifting Overhead Pressing Derivatives: A Review of the Literature. Sports medicine (Auckland, N.Z.)
- **PMC6980669** — Alkjær T, Smale KB, Flaxman TE, Marker IF, Simonsen EB, Benoit DL, Krogsgaard MR. (2020). Forward lunge before and after anterior cruciate ligament reconstruction: Faster movement but unchanged knee joint biomechanics. PloS one. PMC6980669.
- **PMC8805090** — Escamilla R, Zheng N, MacLeod TD, Imamura R, Wilk KE, Wang S, Rubenstein I, Yamashiro K, Fleisig GS. (2022). Patellofemoral Joint Loading During the Performance of the Forward and Side Lunge with Step Height Variations. International journal of sports physical therapy. PMC8805090.
- **PMC8975561** — Fukunaga T et al. International journal of sports physical therapy (2022). PMC8975561
- **PMC9354811** — Coratella G, Tornatore G, Longo S, Esposito F, Cè E. (2022). Front vs Back and Barbell vs Machine Overhead Press: An Electromyographic Analysis and Implications For Resistance Training. Frontiers in physiology
- **PMC9505236** — Mandroukas A, Michailidis Y, Kyranoudis AE, Christoulas K, Metaxas T. (2022). Surface Electromyographic Activity of the Rectus Abdominis and External Oblique during Isometric and Dynamic Exercises. Journal of functional morphology and kinesiology. PMC9505236.

Web-fetched / external peer-reviewed sources. **Each was independently re-fetched and confirmed to resolve to the stated title, authors, year, and finding** (these have no local metadata to diff against, so they were verified one by one):

- Hartmann H, Wirth K, Klusemann M. (2013). Analysis of the load on the knee joint and vertebral column with changes in squatting depth and weight load. *Sports Medicine* 43(10):993–1008. DOI 10.1007/s40279-013-0073-6, PMID 23821469. (WebFetched — squat depth)
- McGill SM. (1991). Electromyographic activity of the abdominal and low back musculature during the generation of isometric and dynamic axial trunk torque. *J Orthop Res* 9(1):91–103. PMID 1824571. (WebFetched — torso twist)
- Tamura A, et al. (2017). Dynamic knee valgus alignment influences impact attenuation in the lower extremity during the deceleration phase of a single-leg landing. *PLoS ONE* 12(6):e0179810. PMC5478135. (WebFetched — jumping jacks)
- DeVita P, Skelly WA. (1992). Effect of landing stiffness on joint kinetics and energetics in the lower extremity. *Med Sci Sports Exerc* 24(1):108–115. PMID 1548984. (WebFetched — jumping jacks)
- Matijašević P, et al. (2025). Development and validation of a running drill test battery to predict 5 m and 20 m sprint performance. *Int J Exerc Sci* 18(8):1269–1285. PMC12591607, DOI 10.70252/LYKE8231. (WebFetched — high knee)
- Bramah C, Preece SJ, Gill N, Herrington L. (2018). Is There a Pathological Gait Associated With Common Soft Tissue Running Injuries? *Am J Sports Med* 46(12):3023–3031. PMID 30193080. (WebFetched — high knee)
- McCarney L, Andrews A, Henry P, Fazalbhoy A, Selva Raj I, Lythgo N, Kendall JC. (2020). Determining Trendelenburg test validity and reliability using 3-dimensional motion analysis and muscle dynamometry. *Chiropr Man Therap* 28:53. PMC7570029. (WebFetched — high knee, contested pelvic-drop/abductor-strength link)
- Camargo PR, Neumann DA. (2019). Kinesiologic considerations for targeting activation of scapulothoracic muscles – part 2: trapezius. *Braz J Phys Ther* 23(6):467–475. PMC6849087, DOI 10.1016/j.bjpt.2019.01.011. (WebFetched — band pull apart / upper-trap)
- Havers T, Wagner N, Held S, Geisler S, Wiewelhove T. (2025). Partial Range, Full Gains? The Effect of 8 Weeks of Partial Range of Motion Training at Long Muscle Lengths on Elbow Flexor Hypertrophy and Strength in Trained Individuals. *European Journal of Sport Science*. DOI 10.1002/ejsc.70087, PMID 41247250. (WebFetched — bicep curl ROM)
- Creech JA, Busse A, Li D, et al. Shoulder Impingement Syndrome. *StatPearls* (NCBI Bookshelf NBK554518, updated 2026). (WebFetched — arm abduction painful arc / impingement)
- Terré M, Solana-Tramunt M. (2025). Muscle Recruitment and Asymmetry in Bilateral Shoulder Injury Prevention Exercises: A Cross-Sectional Comparison Between Tennis Players and Non-Tennis Players. *Healthcare (Basel)* 13(10):1153. PMC12110944, DOI 10.3390/healthcare13101153. (WebFetched — arm VW asymmetry)

Supplementary descriptive (Wikipedia, CC BY-SA — used only for movement definitions/technique targets, never as sole injury-risk backing): `squat_wiki.txt`, `lunge_wiki.txt`, `pushup_wiki.txt`, `ohp_wiki.txt`, `row_wiki.txt`, `torso_twist_russian_wiki.txt`, `jumping_jacks_wiki.txt`.

*The following PMCIDs are web-fetched external sources listed in the block above, not RAG-corpus entries: PMC12110944, PMC12591607, PMC5478135, PMC6849087, PMC7570029.*

---

## 7. Honest limitations & gaps

These are stated rather than papered over, per the spec's honesty requirement:

- **Deadlift lumbar flexion** (`deadlift_lumbar_flexion`) — the clinically most important
  deadlift fault — is **low observability**: MediaPipe has no spine landmarks between shoulders
  and hips, so the heuristic is an explicit proxy, not a true rounded-vs-neutral spine measure.
- **Deadlift lumbar-flexion detection thresholds are UNSOURCED** (2026-08-01). The implemented
  proxy — projected torso shortening against the rep's own setup baseline while the hips stay
  stationary — uses `0.95` / `0.85` ratio endpoints and a `0.10` hip-stationary band. No source
  gives a segment-shortening-to-lumbar-flexion figure; 0.95 was chosen to sit above landmark
  jitter *without any measurement of what that jitter is*. The constants carry `UNSOURCED` in
  their names. The fault is cited; the detection is not. Calibrating against a measured jitter
  floor is the known upgrade path.
- **Deadlift `hips_shoot_up` ramp endpoints are unsourced** (2026-08-01): neither deadlift RAG
  document reports a trunk inclination in degrees, so 55°/75° rest on the spec alone. The
  mechanism and direction are cited; the numbers are not.
- **Two of three Deadlift rules have no KG node** (2026-08-01) and take the `rag` fallback. The
  5-node `Deadlift:` stub (9 nodes counting its shared 1-hop neighbours, e.g.
  `Lumbar Spine Injury`, `Hip Hinge`) was authored independently of this rule catalog and does
  not agree with it: it carries nodes for two faults the catalog has no rule for (`Hyperextension At
  Lockout`, `Insufficient Hip Hinge`), lacks nodes for `deadlift_hips_shoot_up` and
  `deadlift_incomplete_lockout`, and its one exactly-matching fault node (`Bar Drift From Body`)
  belongs to the rule withdrawn above. Only `Deadlift:Lumbar Flexion` grounds a shipped rule.
  Near-misses were rejected rather than used: `Insufficient Hip Hinge` describes a
  knee-dominant pull where `hips_shoot_up` is hip-dominant, and `Hyperextension At Lockout` is
  the literal opposite of `incomplete_lockout`.
- **Deadlift and OHP lockout evidence cannot distinguish "not measured" from "fully flexed"**
  (2026-08-01). When one axis is entirely unmeasurable across a flagged segment,
  `deadlift_incomplete_lockout` reports `peak_hip_angle_deg` / `peak_knee_angle_deg` as **0.0**,
  following `overhead_press.py`'s established `round(x, 2) if np.isfinite(x) else 0.0`
  convention. 0.0° is a physically meaningful angle — a maximally flexed joint — so a reader of
  the evidence cannot tell a missing measurement from a catastrophic one. The fallback exists
  because a bare NaN survives `dataclasses.asdict()` into a postgrest write with
  `allow_nan=False`, whose `ValueError` this codebase documents as silently swallowed, dropping
  the analysis from the user's history entirely. Choosing a misleading number over a vanished
  analysis is the lesser evil, not a good outcome; a `None`-with-explicit-reason evidence shape
  would fix both movements at once and is not attempted here.
- **Setup-relative rules are silently corrupted on `run_detector`'s whole-clip fallback**
  (2026-08-01). `run_detector` falls back to analyzing the clip as one unit on
  `segmentation_disabled`, `no_reps_detected` or `only_partial_reps`
  (`src/pose/movements/base.py:159`), phasing the whole clip in one pass (`:182`) and running
  every rule over it (`:214`). Deadlift's `deadlift_assign_phases` labels the first 10% of
  whatever it is handed `setup` **positionally**, without inspecting the signal — so on a
  fallback run `setup` is the first 10% of the *clip*, which may be the lifter standing around
  before walking up to the bar, and `setup_baseline` returns a **standing** torso.
  `rule_hips_shoot_up` is the casualty: with a ~7° baseline instead of ~60°, its
  `torso_pitch_deg > baseline` clause is satisfied by every loaded frame and contributes
  nothing, so the rule degenerates to its bare 55° absolute gate — losing exactly the
  discriminator the deadlift design spec's §4.1 says it exists for, and firing on a clean rep.
  Reproduced with `DEADLIFT_DETECTOR` unmodified on a trimmed clip yielding
  `fallback=only_partial_reps`: severity 0.2821 with `setup_torso_pitch_deg: 6.84` on a rep
  whose trunk pitch decreased monotonically. `rule_lumbar_flexion` escapes the same corrupted
  baseline only incidentally, because its `_hips_still` term happens to reject travelling hips.
  **The user gets no signal this happened**: `RunResult.fallback` *is* carried in the API
  payload (`src/pose/pose_rule_detector.py:688`) but is rendered **nowhere** in `frontend/src`,
  so a whole-clip analysis is presented exactly like a per-rep one. The same fallback path is
  shared by squat and push-up, whose setup/rest-relative baselines were not tested for this.
  **Not fixed here**: threading `fallback` into `RuleContext` so setup-relative rules can
  abstain is a framework change touching all three movements at once, and a plausibility gate on
  the baseline would introduce exactly the unsourced threshold the deadlift module forbids.
  Threading `fallback` is the known upgrade path and is **not** attempted.
- **Push-up scapular winging** (`pushup_scapular_winging`) is real and cited but **observability
  none** from monocular pose (no scapular landmarks); listed for completeness, not detection.
- **Band-pull-apart / row loss of scapular retraction** is **low observability** — scapular
  position is not reliably recoverable from a monocular front view; a rear-view width proxy is given.
- **Row rounded-spine injury** and **band-pull-apart lean-back**: the *load magnitude* is
  peer-reviewed but the flexion-under-load *injury* link is inferential (flagged inline).
- **Torso-twist lumbar-vs-thoracic rotation** is **low observability** — sparse landmarks cannot
  segment thoracic from lumbar rotation; a hip-line-vs-shoulder-line proxy is used.
- **Jumping-jack ROM-completeness faults** (`jj_incomplete_arm_rom`, `jj_incomplete_leg_rom`)
  rest on Wikipedia descriptive support only — no peer-reviewed source exists for them specifically.
- **High-knee trunk-lean-back** injury magnitude is mechanistic inference (upright-posture target
  is sourced; the lumbar-hyperextension harm is not separately quantified).
- **Sit-up / leg-abduction** cue thresholds (toes-up external rotation, velocity limits) exceed
  the sources' literal wording; the underlying principles are supported, the exact numbers are
  tuning targets to validate empirically.
- **Contralateral pelvic drop** (high knee, lunge) is a strong *injury-association* signal
  (Bramah 2018); it is **not** asserted as a direct readout of hip-abductor weakness, which is
  contested (McCarney 2020).
- **`rounded_thoracolumbar_spine` (Row) is not implementable from this document's own detection
  model, and was not implemented.** Its `detection_heuristic` offers **three** constructions,
  and none of them measures spinal curvature. The "three-point angle at mid-spine" places its
  middle point at `0.5·(shoulder_mid + hip_mid)`, which is by construction the midpoint of the
  segment joining the other two, so the angle is exactly 180° on every frame — a constant. The
  sag alternative measures the distance from `shoulder_mid` to a line of which `shoulder_mid` is
  an endpoint, which is identically zero — also a constant. The third, "alternatively track
  shoulder→hip line vs a straight setup reference," is **not** degenerate — it is perfectly
  computable and nonzero. `row_torso_rising`'s own metric,
  `trunk_angle_from_horizontal_deg = arctan2(|dy|, |dx|)` between hip_mid and shoulder_mid, is a
  pure angle — invariant to whole-body translation and to camera-distance scaling — so neither
  confound applies here. The construction is rejected on narrower grounds: it is
  `row_torso_rising`'s own signal (that same pitch, compared against the same setup baseline),
  relabeled as spinal shape, which would attach this rule's citation (Saeterbakken PMID
  26134664, an EMG magnitude result) to a quantity that citation says nothing about. All three
  constructions fail for the same root cause: MediaPipe Pose (§3)
  has no thoracic or lumbar landmark, so no point exists between the shoulders and the hips, and
  nothing between them can be measured — two of the three routes collapse to constants and the
  third measures a different quantity than the one the rule names. Found during the Row
  implementation (2026-08-01, `docs/superpowers/specs/2026-08-01-row-detector-design.md` §3).
  Row therefore ships **four** rules, not five. Two further monocular substitutes were
  considered and rejected — trunk-length foreshortening and ear-drop relative to the trunk line
  — because both are confounded by camera distance and by the hinge angle, and neither is what
  this rule's citation supports; either would need its own `fault_id` and an explicitly-invented
  threshold. The KG target `Row:Trunk Flexion` exists and is non-empty, so the gap is the
  metric, not the knowledge.

**View-estimation orientation limits (2026-07-25, added when `body_axis_extent` made body-extent
measurement orientation-aware; see `src/pose/view_estimation.py` module docstring for the
authoritative version of these four):**

1. `signed_orientation` (`sign(left.x - right.x)`) is an image-space left/right ordering whose
   front/rear meaning is validated only for UPRIGHT subjects. For a horizontal body the frontal
   axis no longer maps onto image x, so `front`/`rear`/`*_oblique` labels carry no validated
   meaning there. Do not gate a horizontal-movement rule (e.g. push-up) on them.
2. `estimate_view_for_pose` is called with `allow_front=False` in the production path
   (`src/pose/pose_rule_detector.py`), so `front` and `front_oblique` are unreachable there;
   only `side`, `rear`, `rear_oblique`, and `unknown` are ever emitted downstream.
3. `_visible_midpoint` requires BOTH the left and right landmark of a pair above 0.35
   visibility to contribute to the body axis. One occluded shoulder — or an incomplete ankle
   AND hip pair — silently reverts `body_axis_extent` to the pre-fix vertical fallback instead
   of the true body axis, with no NaN and no other signal raised. Measured: on a horizontal
   fixture with landmark 12 (right shoulder) forced to visibility 0.1, the axis extent returned
   0.070 instead of ~0.60 (8.6x low). Not a regression — the fallback is the pre-2026-07-25
   behavior, correct for upright squats — but it silently undoes the orientation-aware fix
   exactly in the view most likely to trigger it: a sagittal (side) view is precisely where
   far-side landmarks are most often occluded.
4. When a clip carries no orientation evidence at all (`front_score == rear_score == 0.0`) but
   still clears the evidence floor on torso-width alone, `score_view`'s branch ladder resolves
   it to `rear_oblique` rather than `unknown`, because with `allow_front=False` the
   `front_score >= rear_score` branch is taken unconditionally on the tie. Downstream,
   `rear_oblique` sits inside `rule_knees_inward`'s `observable_alignment` gate in
   `src/pose/movements/squat.py` (the old `side` verdict did not), and that gate has no
   confidence floor — so an evidence-free clip can score `knees_inward` at **confidence 1.000 /
   observability "high"** instead of being excluded, versus **confidence 0.650 / "medium"**
   before this change. **Deliberately NOT fixed here**: a confidence floor on that gate would
   change squat rule output, and `tests/test_movement_registry.py` pins a field-by-field
   comparison of the registry path against the legacy oracle in `pose_rule_detector.py`, which
   would need the identical change in lockstep or the gate test fails. This is a known, measured
   defect awaiting a scoped follow-up, not an oversight.

   *(Precision note, 2026-07-25: that gate is often called "byte-for-byte", which overstates it.
   Its `comparable()` helper compares exactly eight fields — `fault_id`, `severity` and
   `confidence` to 4 dp, `observability`, `start_frame`, `end_frame`, `peak_frame`, `phase` — and
   does NOT compare `evidence`, `fault_name`, `kg_query`, `retrieval_mode`, `start_time` or
   `end_time`. A legacy-vs-registry divergence confined to the evidence dict would pass it. The
   confidence-floor argument above is unaffected, since `confidence` is one of the eight.)*

## 8. Next steps

1. User review of this spec.
2. Implement per-movement detectors in `src/pose/pose_rule_detector.py`, extending the frame-metric
   and segment-detection machinery beyond squat (requires per-movement phase segmentation and the
   new geometry signals defined here).
3. Wire each fault's `citation`/`citation_support` into the KG/RAG retrieval layer so the coaching
   chat can surface the grounding source, as the squat faults already do via `kg_query`.
4. Validate thresholds against labeled data per movement before shipping analysis for that movement.

**Status (2026-07-18):** Foundation shipped (movement registry + citations + behavior-preserving
squat migration + Overhead Press) on branch `feat/movement-rule-detector-spec`. **OHP thresholds
are spec-derived and unvalidated** — no labeled OHP data yet (§8.4). Remaining 14 movements follow
as per-movement plans reusing this framework.

**Status (2026-07-25):** All **5 of 5** OHP rules are now implemented in
`src/pose/movements/overhead_press.py` (`ohp_incomplete_lockout`, `ohp_lumbar_hyperextension`,
`ohp_asymmetric_press`, `ohp_insufficient_elevation`, `ohp_forward_head`). Deviations from the
detection heuristics written above, all deliberate and documented in-code:

- `ohp_insufficient_elevation` — the "~0.5 head-heights" wording above is **not implementable**:
  MediaPipe's 33 landmarks contain no head-height measure (nose, eyes, ears and mouth all lie
  *within* the face, so no pair of them spans the head). The implementation **substitutes** a
  shoulder-width-normalized nose-clearance criterion (fires when
  `(wrist_mean_y − nose_y) / shoulder_width > −0.15`). This is a **substitution, not a unit
  conversion** — no head-height-to-biacromial-width anthropometric constant was assumed.
- `ohp_forward_head` (renamed from `ohp_forward_head_barpath`) — implemented as a **hard view
  gate** rather than the observability downgrade implied by "medium–high `side`; low from
  `front`". The cue is a pure horizontal offset whose direction is unresolvable without knowing
  the subject's facing, so the rule returns **no detections at all** outside
  `{side, front_oblique}` instead of low-confidence ones; a wrong direction claim is worse than
  silence. Its shoulder-width normalizer is also weakly conditioned in exactly the sagittal views
  it is gated to.
- `ohp_forward_head` — the gate additionally requires `view_confidence >= 0.20`
  (`SIDE_VIEW_CONF_THRESHOLD` in `src/pose/pose_rule_detector.py`), following the squat precedent
  in `rule_knees_forward`. **This extends that precedent**: squat gates only `side`, whereas here
  the same floor is applied to `front_oblique` too, on the grounds that a weakly-classified
  oblique view authorizes a directional claim just as little as a weakly-classified side view
  does. No new number was introduced — the constant is shared with squat.
- `ohp_forward_head` — the spec's **bar-path sub-criterion is WITHDRAWN**; see the boxed note in
  the rule entry above for the full rationale and the open spec question it leaves.
- `ohp_incomplete_lockout` — the mask fires on `elbow_flag OR wrist_flag`, so severity is scored
  from **both** spec'd ramps (160→140° elbow, 0.0→0.15 wrist) and the worse taken. Selecting the
  ramp by "is the elbow reading finite?" mis-attributed severity when a segment fired on the
  wrist criterion alone: a rep whose bar never left shoulder height with the elbows locked
  straight was emitted at **severity 0.0 / confidence 0.0**, carrying evidence ("peak elbow 178°
  vs 160° threshold") that contradicted the fault it named. No new threshold was introduced.

All five OHP thresholds remain unvalidated against labeled data (§8.4).

**Status (2026-07-25) — Push-up detector registered:** `src/pose/movements/pushup.py` is now
assembled as `PUSHUP_DETECTOR` and registered under `"Push-up"`, reachable from
`scripts/pose/run_pose_rule_detection.py --movement "Push-up"`. All **5 of 5** push-up rules are
present: **4 can fire** (`pushup_hip_sag`, `pushup_shallow_depth`, `pushup_head_drop`,
`pushup_elbow_flare`) and **1 is permanently silent by design** (`rule_scapular_winging`, which
always returns `[]` — MediaPipe's 33 landmarks contain no scapular border points, and this spec
rates the fault observability `low`/`none` and recommends not emitting a confident verdict. It is
registered rather than omitted so the spec and the code stay 1:1 and an auditor finds it with its
citation instead of finding a gap and closing it with an invented proxy).

> **PUSH-UP THRESHOLDS ARE SPEC-DERIVED AND UNVALIDATED (§8.4).** There is **no labeled push-up
> data in this repository at all** — REHAB24-6 Ex3 is a *standing table push-up*, not the floor
> push-up this detector models, and the EgoExo push-up frames are an unextracted ~3 GB archive.
> Nothing in this detector has been checked against a human-labeled push-up fault. The synthetic
> fixtures in `tests/test_pushup.py` prove geometry and sign conventions, not real-world fault
> detection. Push-up is therefore **CLI-only**: `backend/app/config.py`'s
> `DEFAULT_ANALYSIS_MOVEMENT` remains `"Squat"` and `frontend/src/lib/movements.ts` still lists
> `ANALYZABLE_MOVEMENTS = ["Squat"]`, so these numbers do not reach end users.

Deviations from the detection heuristics written above, all deliberate and documented in-code:

- `pushup_shallow_depth` — fires at **100°**, the conservative (strictest-to-fire) end of the
  spec's "~100–110°" band. No number outside the band is used. It also reads `min_elbow_angle`
  (the **more-flexed** arm) where the spec says "whichever is more visible"; on an asymmetric rep
  that is the more generous reading, so the rule under-reports rather than over-reports.
- `pushup_head_drop` — **per-clip `setup` baseline instead of an absolute reading, on both axes.**
  The spec's wording is absolute but supplies no absolute reference for "neutral", and none
  exists: resting ear-to-shoulder angle varies with neck length, ear position and camera height.
  The nose axis needs it even more concretely — measured on the neutral fixture,
  `nose_ahead_ratio = 0.1833` with no fault present (3× an absolute 0.06 cut), so read absolutely
  the cue fires on every correct push-up. The rule therefore measures each axis's deviation from
  its mean over the clip's own `setup` frames, mirroring this spec's own construction for the
  squat's heel rise. **Honest cost:** it measures *change*, not *posture* — a lifter who sets up
  with the head already dropped and holds it there is never flagged (pinned by
  `test_a_head_drop_held_from_setup_is_invisible`).
- `pushup_head_drop` — the nose cue's **0.06 of body length is a RULE-LEVEL CHOICE, not this
  spec's number.** The spec ORs the cue in but quantifies it only as "clearly ahead". The
  magnitude is borrowed **by analogy** from `pushup_hip_sag`'s spec'd "> ~0.06 of body length" —
  a defensible reading of "clearly", but an analogy, and it must not later be cited as spec
  provenance.
- `pushup_head_drop` — a **signed** metric (`neck_line_signed_deg`) was added to the metric layer
  to support it. A baseline on the unsigned angle is not merely non-directional but *actively
  inverted*: with a +5° baseline, a head **lifted** to −15° reads deviation +10 and fires as a
  head drop. The neck angle is also referenced to the **body axis** (shoulder-mid → ankle-mid)
  rather than the spec's shoulder→hip chord, because the chord rotates when the hips drop and so
  manufactures neck deviation out of hip sag (measured: `hip_offset_ratio` 0.100 → 11.31° of
  spurious neck signal, 55% of a full head-drop reading). The swap trades one modeling assumption
  for another and the residual is documented in-code, not corrected.
- `pushup_head_drop` — scoped to `{descent, bottom}` (the spec scopes it to no phase), so a head
  that drops only on the way back up is missed. Stated rather than hidden.
- `pushup_elbow_flare` — **a measurability gate replaces the spec's view gate.** The spec asks for
  a `front`/`rear` view down the body's long axis. Task 4 establishes that `signed_orientation`'s
  front/rear meaning is validated only for *upright* subjects, so for a horizontal body those
  labels carry no validated meaning, and the production path calls
  `estimate_view_for_pose(allow_front=False)`, so `front` is never emitted at all — a positive
  gate on those labels would be either meaningless or permanently false. The rule instead requires
  `shoulder_axis_ratio > 0.15` (a **rule-level** measurability threshold, not in this spec: enough
  transverse extent survived the projection for the hand-width question to be answerable) and
  hard-gates to silence on a *confident* `side` label, which is the answerable negative claim.
  Without that guard a sagittal collapse forges a verdict: a measured shoulder separation of
  0.0020 against a wrist separation of 0.0050 — both sub-pixel, both meaningless — yields
  `hand_width_ratio = 2.500`, a full-severity flare out of pure noise.
- `pushup_elbow_flare` — emitted at a flat **observability `low` with the 0.65 confidence
  discount**, below this spec's `medium` ceiling for the fault, because that ceiling is attached
  to `front`/`rear` labels that are not validated for a horizontal body. `run_detector` sorts
  `low` detections behind every other detection regardless of severity (key
  `(observability == "low", −severity, start_frame)`; `False < True`), so a flare verdict can
  never outrank a fault observed from a view the pipeline actually validated. The rule's second
  gate, `hand_width_ratio > 0.25`, is **arithmetically inert** while the fire threshold sits at
  1.6 and is kept explicit only so the measurability condition stays legible.
- `pushup_elbow_flare` — the spec's optional corroborating signal (trunk-to-upper-arm angle
  > ~65°) is **not implemented**; the metric layer emits no such angle.
- `pushup_hip_sag` and `pushup_elbow_flare` — both scoped to `{descent, bottom, ascent}`, a
  rule-level call following the squat detector's `ACTIVE_PHASES`; this spec scopes neither.
- `pushup_hip_sag` — hard-gates `front`/`rear` to **silence** rather than a confidence discount,
  because the offset normalizer *inflates* head-on (the shoulder→ankle projection shortens), which
  is a false-positive amplifier rather than a weak signal. It also requires
  `hand_offset_ratio > 0.0` as a **camera-inversion guard**: the hands are planted on the floor, so
  a negative value means "groundward" resolved backwards and every sag would be reported as a
  confident pike. The plank-angle form ("hip angle departs from 180° by > ~12°") is reported as
  corroboration but does **not** gate firing, because it is unsigned and would reintroduce the
  direction-free verdict the guard exists to prevent.
- **Severity ramps for every push-up rule are rule-level choices, not spec quantities.** This
  spec's Push-up section carries no `Severity ramp` line at all (the Squat/Lunge/Deadlift sections
  do, so the absence is meaningful). The endpoints chosen — 0.15 hip offset, 140° elbow, 35° neck,
  0.15 nose, 2.2 hand-width — are ranking curves with in-code reasoning, not cited numbers.

**Module-wide behavior worth recording, not a deviation:** `pushup_compute_raw` requires both
shoulders, elbows, wrists, hips **and both ankles**, because the plank line is shoulder-mid →
ankle-mid. One dropped landmark marks the frame invalid and silences *every* push-up rule for it,
so a clip framed from the knees up produces no push-up verdict at all. That refusal is deliberate
— `view_estimation`'s degrading alternative was measured to read a body-axis extent of 0.070
instead of ~0.60 (8.6× low) with no NaN and no other signal, and a silently-wrong verdict is worse
than none. The sagittal view the spec calls primary is also the view most likely to trip it.

**Status (2026-07-26) — Squat view gating brought in line with §3.** A compliance audit of
`src/pose/movements/squat.py` against the Squat rules above found every threshold, severity ramp,
phase scope and citation matching, and all four `retrieval_mode="kg"` queries resolving against
`data/kg/sports_kg_v3.graphml` (`Knee Valgus`, `Anterior Knee Translation`, `Shallow Depth`,
`Excessive Forward Lean` → their `Squat:`-scoped nodes — no squat analogue of the dangling OHP
queries below). Two rules ignored §3's "confidence is scaled down when the required view is
unavailable" convention and now honour it (mirrored in `pose_rule_detector.detect_rule_segments`,
the legacy oracle, and pinned across all six view labels by `tests/test_movement_registry.py`):

- `heel_rise` — was `observability="medium"` at undiscounted confidence for **every** view,
  including head-on, where this spec calls the cue "nearly invisible". Now gated to
  `{side, front_oblique, rear_oblique}`; outside that set (and on `unknown`) it emits
  observability `low` with the ×0.65 discount. **The `low` rating is a rule-level downgrade, not
  this spec's number** — the entry above rates the fault `medium` and names no rating for the
  views where it is unavailable. `low` additionally demotes the verdict behind every observed
  fault via `run_detector`'s sort key, which is the intended consequence.
- `shallow_depth` — `unknown` (the view estimator's *evidence-floor failure* verdict, not a view)
  fell through to the rule's `else` branch and so earned its **best** rating, observability `high`
  at full confidence, on precisely the clips whose camera geometry could not be established. It
  now takes medium/×0.65, matching how `knees_inward` and `excessive_forward_lean` already
  resolve `unknown`. Behaviour on the views this spec does enumerate is unchanged: side → high,
  rear/rear_oblique → medium, both undiscounted.

> **DEFECT IN THIS SPEC'S `heel_rise` HEURISTIC (2026-07-26, found by that audit, NOT fixed).**
> The heuristic above — `heel_height_delta = heel_y(29/30) − toe_y(31/32)`, flag when
> `heel_height_delta − baseline > 0.015` — is **inverted against §3's own coordinate convention**
> ("y increasing downward"). A heel lifting off the floor travels UP the image, so `heel_y`
> *decreases* and the delta goes *negative*; the stated condition can only be met when the heel
> drops **below** the toe line, i.e. on a toe rise. `src/pose/geometry.py:heel_height_delta`
> implements this text faithfully, so code and spec agree and are both wrong: measured on a
> fixture with the heels raised 0.05 above the toe line, `rule_heel_rise` emits nothing.
> Consistent with that, `heel_rise` appears in **zero** stored detections — no file under
> `data/` contains the string at all (5 detection JSONs are present locally; the rest of the
> labeled set is gitignored, so this is corroboration, not proof).
> Pinned as `tests/test_squat_view_gating.py::test_a_real_heel_rise_never_fires`
> (`@unittest.expectedFailure`), so whoever flips the sign is told by an unexpected-success
> failure to amend this spec entry and drop the marker. Fixing it changes the meaning of a
> metric already written into stored analyses, so it is left as an explicit decision.

**Known open item (not a deviation, a gap):** three OHP `kg_query` strings resolve to **no KG
node at all** against `data/kg/sports_kg_v3.graphml` — `"Incomplete Elbow Lockout"`,
`"Lumbar Hyperextension"` and `"Asymmetric Press"` (verified via
`graph_retrieval.resolve_nodes`, both movement-scoped and unscoped). `"Forward Head Posture"`
and `"Limited Shoulder Elevation"` do resolve. There is no near-miss node to re-point them at
(the closest OHP-scoped nodes are `Near Lockout`, `Thoracolumbar Extension`,
`Elbow Extensor Torque`), so this needs KG content work, not a string tweak. Those three faults
currently reach the chat layer with citations but no retrieved grounding.

**Status (2026-07-30) — Lunge detector registered.** `src/pose/movements/lunge.py` is now
assembled as `LUNGE_DETECTOR` and registered under `"Lunge"`, reachable from
`scripts/pose/run_pose_rule_detection.py --movement "Lunge"`. All **4 of 4** Lunge rules are
present and can fire: `rule_knee_past_toes`, `rule_knee_valgus`, `rule_insufficient_depth`,
`rule_pelvic_drop`. No Lunge fault is permanently silent by design, unlike push-up's
`rule_scapular_winging`.

- **The lead-leg substitution, and why it lives in the RULES, not `lunge_compute_raw`.** This
  spec's Lunge entries define the lead leg as "the more flexed / more anterior foot". The
  `more anterior` half is exactly the axis that collapses in a frontal view, where two of the
  four rules live, so the implementation (`resolve_lead_side`) uses the more-flexed half only,
  evaluated at a window's bottom frame. It cannot live in `lunge_compute_raw`: `run_detector`
  calls `compute_raw` over the whole clip before `segment_reps`, so at metric time there is no
  rep boundary and therefore no bottom frame to resolve "which leg is loaded THIS rep" against.
  A per-frame heuristic would flicker through `setup`/`recovery`, where both knees sit near
  extension within landmark noise of each other, corrupting every lead-relative metric and
  `centered_median`'s smoothing across the swap. `lunge_compute_raw` therefore emits every
  side-specific metric for BOTH legs (`left_*`/`right_*`), and `resolve_lead_side` chooses
  between them only once a per-rep window exists.
- **The two rule-level numbers, both labeled as such in-code, neither from this spec:**
  `LEAD_SIDE_MIN_SEPARATION_DEG = 5.0` (the minimum left/right knee-angle gap at the bottom
  before a lead leg is claimed — this spec names no such floor, and the constant can only
  *silence*: an unresolved lead side emits nothing rather than a guessed, mis-attributed one)
  and `LUNGE_ACTIVE_PHASES = {descent, bottom, ascent}` (this spec scopes only
  `lunge_knee_past_toes` to phases; applying the same set to the other rules follows the squat
  detector's `ACTIVE_PHASES` precedent rather than a spec requirement — cost: a fault visible
  only during `setup`/`recovery` is missed). Every other threshold and severity ramp in the four
  rules is this spec's own number, verbatim (re-confirmed by Step 1's audit below).
- **`KNEE_FORWARD_MILD`/`KNEE_FORWARD_SEVERE` reuse.** `rule_knee_past_toes`'s fire/ramp
  (0.10 → 0.30) is worded identically to the Squat entry's, so the implementation imports
  Squat's existing constants from `src/pose/pose_rule_detector.py` rather than re-typing the
  literals, so the two movements cannot drift apart independently.
- **`rule_pelvic_drop`'s split-stance foreshortening bias, documented not corrected.** In a
  frontal view of a split stance the L-hip→R-hip vector is rotated in the transverse plane, so
  its image projection shortens and `atan2(dy, |dx|)` *inflates* the apparent tilt — the deeper
  the lunge, the worse. The expected failure mode is therefore **false positives on deep,
  correctly-performed reps**, not silence. Correcting it needs a depth estimate this pipeline
  does not have, so Phase 2 reads specificity on correct reps first for exactly this reason.
- **`rule_knee_valgus`'s known contamination, the mirror-image bias.** Obliquely, anterior knee
  travel and medial knee travel project onto the same perpendicular axis, so a deep,
  perfectly-tracked lunge can read as valgus in every view this pipeline reaches (`front`,
  which would separate the two cleanly, is never emitted downstream). Pinned by
  `test_anterior_knee_travel_contaminates_the_valgus_proxy` in `tests/test_lunge.py`. Phase 2
  checks whether firing tracks step depth rather than correctness.
- **Task 3's depth-scope correction.** The brief's example KG candidate for insufficient depth,
  "Excessive Knee Flexion", resolves to a real node but the wrong one — its only edge is
  `INCREASES_RISK_OF → Achilles Tendon Injury`, the wrong fault direction (this spec's own
  `citation_support` describes *reduced* flexion marking impaired function). `LUNGE_DEPTH_KG_QUERY`
  is `"Decreased Knee Flexion"` instead, the node whose edges (`CAUSED_BY ← Weak Quadriceps`,
  `INCREASES_RISK_OF → ACL Injury`) match the cited sentence.
- **Step 1 audit, re-run for this status entry, not assumed:** `resolve_nodes` against
  `data/kg/sports_kg_v3.graphml` confirms all four `*_KG_QUERY` constants still resolve to
  exactly one `Lunge:`-scoped node apiece (`Knee Anterior To Toes`, `Knee Valgus`,
  `Decreased Knee Flexion`, `Trendelenburg Posture`) — no drift since Task 3. Every
  `citation`/`citation_support` string in `src/pose/movements/lunge.py` was extracted
  programmatically from a live-fired `PoseRuleDetection` (not read by eye) and confirmed to be
  a byte-exact substring of this spec's text for all four rules. **No fault is left without a
  resolving KG node** — unlike OHP's three-of-five gap above, Lunge has no open KG item.
  `tests/test_kg_query_resolution.py::test_every_kg_query_resolves` ran locally (the graph is
  present in this checkout) and passed, corroborating the audit independently of it.
- **A test-infrastructure fix, generalized rather than special-cased.** Unlike squat/push-up/OHP,
  which pass `kg_query=` as an inline string literal at each `build_detection` call,
  `lunge.py` passes it as a reference to a module-level constant
  (`kg_query=LUNGE_PAST_TOES_KG_QUERY`, etc.) — deliberate, so the Step 0 provenance comment
  attached to each constant stays a single source of truth instead of being re-typed at every
  call site. `tests/test_kg_query_resolution.py`'s AST-based `_kg_queries` scanner only
  recognized literal `ast.Constant` values, so it read zero queries out of `lunge.py` and its
  own `test_queries_were_actually_found` gate failed. Fixed by teaching the scanner to also
  resolve `ast.Name` references against the module's top-level string-constant assignments,
  which generalizes the helper for any future module rather than adding a lunge-shaped
  exception to it.
- **Product-surface consequence of registration — read, not assumed.** `backend/app/config.py`'s
  `DEFAULT_ANALYSIS_MOVEMENT` is unchanged, still `"Squat"` (its own comment already calls it
  "the FALLBACK movement, not a pin"). But unlike when the Push-up status block above was
  written, the frontend's `ANALYZABLE_MOVEMENTS` constant this spec used to name **no longer
  exists** — `frontend/src/lib/movements.ts` now derives which movements are analyzable
  entirely from `GET /api/movements`, which in turn is generated live from
  `src/pose/movements/registry.py` (`backend/app/routers/movements.py`'s own docstring states
  this as the design: *"registering a fourth detector surfaces it in the UI with no backend or
  frontend edit"*). Confirmed by reading `frontend/src/App.tsx`: it reads `?movement=` from the
  URL, validates it against the fetched catalog, and passes the resolved name — not a hardcoded
  `"Squat"` — to the analyze call. So registering `LUNGE_DETECTOR` here makes Lunge appear in
  `GET /api/movements`, makes its `/movements` menu card clickable instead of inert "Soon", and
  makes it genuinely analyzable end-to-end through `/api/analyze` and `/api/analyze/pose` for any
  visitor of the public `/app` demo. This is **not** a defect introduced by this task and this
  task does not add code to suppress it (that would be new production policy against an explicit
  "do not change" instruction, and the router's documented design is to auto-surface every
  registered detector). The mitigating fact: it surfaces with `validated=False`, so the frontend
  renders it with a Beta tag — the same signal Push-up and Overhead Press already carry, which,
  by the same mechanism, are *also* already live in the product today despite the "CLI-only"
  framing of their own status blocks above (written before the `ANALYZABLE_MOVEMENTS` frontend
  refactor). `tests/test_movements_endpoint.py`'s two exhaustive-list assertions were updated to
  include Lunge/Beta accordingly; nothing was added to gate it out.
- **The Task 1 view-gate finding — PENDING, stated as such, not answered either way.**
  `lunge_knee_past_toes` is hard-gated on a confidently-classified `side` view
  (`SIDE_VIEW_CONF_THRESHOLD`), mirroring squat's `rule_knees_forward`. Whether a true sagittal
  Lunge clip is actually classified `side` by `estimate_view_for_pose` in production is an open
  question Task 1 (REHAB24-6 Ex5 pose extraction + view-gate reconnaissance) was measuring at the
  time this detector was registered, and Task 1 had **not completed** — extraction for the 18 Ex5
  clips was still running. What is known so far: across the 45 real pose JSONs already in this
  repository, the view estimator emitted `side` exactly once, and that single verdict was a
  fabricated degenerate case since removed. Whether the gate ever opens on real sagittal footage
  is therefore **being measured, not settled** — this status block does not assert an answer in
  either direction, and Task 7/8's validation work is what will.
- **All four Lunge thresholds are spec-derived and UNVALIDATED against labeled data at this
  point.** `tests/test_lunge.py` proves geometry, sign conventions and the fault-attribution
  contract (including the alternating-lead multi-rep regression the Phase 2 harness structurally
  cannot see, since that harness feeds one rep per clip), not real-world fault detection accuracy.
  Phase 2 (REHAB24-6 Ex5) is what changes that; flipping `validated` to `True` is a separate,
  evidence-backed decision made after Phase 2 produces numbers, not part of this task.

**Status (2026-07-30) — Lunge validated against REHAB24-6 Ex5. This is the FIRST movement in
this repository ever checked against human-labeled ground truth, and it closes §8.4 for Lunge
only.** All four rules were replayed over **174 labeled repetitions** (8 subjects, 96 incorrect
/ 78 correct, two orthogonal cameras) via `scripts/rehab24/validate_lunge_rules.py`. Full
numbers, method and caveats: **`notes/lunge-rule-validation.md`**. Read that before quoting any
figure from here.

**What the labels can and cannot support.** REHAB24-6 says a rep was correct or incorrect and
**never names the fault**, so a rule firing on an incorrect rep is not evidence it found that
rep's error. Everything below measures whether a rule's signal **carries information about rep
correctness** — not per-fault precision. Headline statistics are **per-subject** (median and
range across the 7 of 8 subjects that carry both classes; person 3 has 21 incorrect reps and
zero correct ones); pooled figures are secondary and **no p-value is computed on pooled reps**.
Whatever separates here is validated **on this dataset** — a lab recording with fixed cameras,
controlled lighting and instructed errors.

**No threshold or severity ramp was changed in response to any of it, and
`LUNGE_DETECTOR.validated` stays `False`.** Tuning a cited number to a measured metric would
make its citation a false provenance claim; flipping the flag is a product claim about "checked
against labeled ground truth" and is the user's decision with these numbers in hand.

- **The lead-side substitution is what failed validation, ahead of any threshold — and it bounds
  all four rules,** since every one of them reads `f"{lead}_..."`. `resolve_lead_side` agrees
  with `exercise_subtype` on **96/154 = 0.623** of resolved cam17 reps and **72/152 = 0.474** of
  cam18 reps (**below chance**), leaving 11.5%/12.6% unresolved. The failure is in the *premise*
  it substitutes for this spec's "the more flexed / **more anterior** foot" definition (the
  anterior axis collapses in a frontal view, so only the more-flexed half is used): the labeled
  lead knee is the more flexed knee at the rep's bottom on only **101/169 = 59.8%** (cam17) and
  **77/161 = 47.8%** (cam18) of reps, measured over the **full labeled window** with
  `segment_reps` and smoothing out of the picture. On the reps it gets wrong the left-right
  separation is a median 19.4°/25.4°, far outside the 5° band
  `LEAD_SIDE_MIN_SEPARATION_DEG` refuses on, so that guard cannot catch it. The guard is also
  quieter than the unresolved rate implies: decomposed by `lead_unresolved_reason`, only
  **15/174 = 8.6%** (cam17) and **9/174 = 5.2%** (cam18) of reps are the 5° guard proper; the
  rest is missing data (no valid frame carrying a finite `min_knee_angle`: 5 and 13; a bottom
  frame with a non-finite knee angle: 0 and 0).
  **But the failure is one of MEASUREMENT before it is one of anatomy, and the writeup is
  deliberately narrower than "more flexed does not identify a lunge's lead leg".** Three controls
  say so: the two SIMULTANEOUS cameras disagree about which knee is more flexed on **33%** of
  reps — that is measurement error, not anatomy; the two
  premise rates disagree by 12 points across those same views; and recomputing the identical
  angle in the image plane alone, dropping MediaPipe's pseudo-depth `z`, swings cam17 from
  **59.8% to 17.2%** while moving cam18 the other way. (Both controls are reported on two frame
  populations — the `segment_reps`-re-cut scored window and the full labeled window. Same
  geometry, different frame, and the harness computes both: the scored-window variants read
  **58/156 = 37.2%** and **24/169 = 14.2%**. The **full-window figures are the quoted ones
  because they are segmentation-independent**, which is the property this argument must not
  borrow from the harness's own windowing; every conclusion holds on both populations.) The
  supported claim is therefore **"the more-flexed-knee cue, as this pipeline measures it, does
  not identify the labeled lead leg from either view available here"** — which still fully
  condemns `resolve_lead_side` as shipped and still bounds all four rules. **The distinction
  drives the fix: this data cannot separate "the premise is wrong" from "the premise is
  unrecoverable from this projection", so the indicated repair is a DEPTH-ROBUST lead cue before
  a different cue.** §7's risk register predicted the outcome explicitly; it materialized.
  **Recorded, not patched** — a replacement cue is a detector change with its own validation.
- **`lunge_knee_past_toes` — the cue is informative; the rule as shipped cannot reach it.** On
  the 88 genuinely sagittal cam18 reps, the shipped rule's metric **inverts**: per-subject median
  AUC **0.171** (correct reps ordered *above* incorrect). Reading the same metric off the leg
  `exercise_subtype` names, on the same frames, gives per-subject median **0.833** (0.725 over
  all 174) — and **0.171 vs 0.850 restricted to the 80 reps BOTH lead choices score**, so the
  contrast is not a denominator artifact. (Every AUC in the writeup carries its rep count; the
  two columns lose different reps to an unresolved lead side.) The `half-profile` stratum's
  production 0.850 is not a counter-example: its lead-oracle figure is 0.845, i.e. the lead
  choice barely matters there, so the wrong-leg penalty that inverts the sagittal reps is not
  levied on it. The gate is not the problem: the estimator returns `side` on all 88, matching Phase
  0's 88/88, and the production and oracle passes are byte-identical here because neither yields
  `side` off that stratum.
- **`lunge_knee_valgus` — weak, MEDIAN-above-chance separation, and the least
  lead-side-sensitive of the four. Not a validation.** Per-subject median AUC **0.590** (0.629
  excluding the 40 level-2/3 extra-person reps; 0.620 under the lead-oracle) — but the seven
  per-subject values are 0.263, 0.374, 0.486, 0.590, 0.629, 0.810, 0.852, so **only 4 of 7
  subjects are above 0.5**, one inverts substantially, and **no null was tested** (this harness
  computes no permutation null, CI or significance test, and p-values are declined on these reps
  for independence reasons). The claim is "the median is above chance on 4 of 7 subjects", not
  that chance has been excluded. The predicted step-depth contamination is **present but weaker
  than predicted**: Spearman ρ = **−0.325** between the valgus proxy and bottom-phase lead-knee
  angle within the *correct* reps only (vs −0.211 on incorrect) — the predicted sign and shape,
  but a weak association, so step depth explains part of the firing, not most of it. On the 86
  `half-profile` reps the rule fires on **83%** of them (threshold at percentile 6.3), so that
  stratum's sensitivity 0.915 is trivial and the **fire rate is the primary read**.
- **`lunge_insufficient_depth` — not exercised by this dataset, and its apparent signal is a
  selection artifact.** It fires **6 times in 174 reps** (threshold 100° sits at percentile 84.5
  of the observed distribution — `rank_auc`'s documented "informative cue, cited cut in the tail"
  case, and the cut does not move). The sagittal stratum's apparent 0.792 per-subject AUC
  **collapses to 0.320 under the lead-oracle** (0.792 vs 0.300 at matched n=80): `resolve_lead_side` picks the *more flexed* knee
  by construction, so "the maximum angle of the selected knee" is a biased statistic of the pair,
  not a measurement of the lead leg. Read off the labeled leg the metric is at or below chance
  (0.390 overall). Not evidence the rule is wrong — evidence the fault is absent from, or
  invisible in, REHAB24-6's instructed errors.
- **`lunge_pelvic_drop` — barely exercised, and §6.5's predicted failure mode is NOT refuted.**
  It fires **10 times in 174 reps** (4 tp / 6 fp) and under the lead-oracle sits at **0.467 —
  chance**, so per the plan's own rule, near-zero firing on **both** classes means "not exercised
  by this dataset", **not** "the rule works". On the risk §6.5 raised (false positives on deep
  correct reps from split-stance foreshortening), the reassuring-looking numbers are artifacts
  and must not be used to retire it: the half-profile stratum's **1.000 specificity is vacuous**
  — the rule fired zero times there because the view mislabel below gated it off on all 39
  correct reps, and a specificity for a silenced rule measures nothing — while the overall
  **0.923 counts 54 of its 78 correct reps as true negatives on reps where the rule was
  structurally silent** (37 view-gated OR 18 could-not-fire, overlapping on 1 — union 54, not
  55). On the **24 correct reps where the rule could actually act it false-fired on 6:
  specificity 0.750, a 25% false-positive rate**, which is the shape §6.5 predicted rather than
  its refutation. Excluding the 40 level-2/3 extra-person reps moves that conditional specificity
  to 0.895 and the per-subject median AUC to 0.679, so person-locking is not the cause. The
  threshold sits at percentile 54.4 of the observed distribution (64.3 front / 45.6
  half-profile). **The §6.5 risk stays OPEN**: 6 false fires give no power to confirm the
  foreshortening mechanism specifically.
- **The one real production-vs-oracle gap is a GATE failure, and it qualifies Phase 0's
  headline.** `lunge_pelvic_drop` fires 10 in production and 41 in the oracle pass, because the
  view estimator labels **84 of 86 cam17 `half-profile` reps as `side`**, and the rule correctly
  returns `[]` on `side`. The *label* is what is wrong. So the `side` gate has good sensitivity
  (88/88 on genuinely sagittal cam18 reps, per `notes/lunge-view-reconnaissance.md`) and **poor
  specificity** — it also emits `side` on clearly non-sagittal frontal-camera reps. The same
  mislabeling silently downgrades `lunge_knee_valgus` there (`side` ∉ `ALIGNMENT_OBSERVABLE_VIEWS`
  → observability `medium`, confidence ×0.65) without changing whether it fires. No gate,
  threshold or confidence floor was changed in response.
- **Every contingency figure above is quoted with its structural silence, because the raw tables
  are misleading without it.** A rep can be unable to fire because its **view gate** was shut, its
  masked phase was shorter than `min_frames` (6 at 30 fps), or its lead side was unresolved —
  and `contingency` counts every such rep as a true or false NEGATIVE, deflating sensitivity and
  inflating specificity invisibly. **The categories OVERLAP and must never be added** — every
  count below is a union, with its components and their overlap given so a reader's own
  arithmetic reconciles. Production, out of 174 reps each: `lunge_knee_past_toes` **98
  non-actionable** (86 view-gated OR 32 could-not-fire, overlapping on 20) leaving 76;
  `lunge_insufficient_depth` **48** (none view-gated; 26 of them windows whose `bottom` phase is
  shorter than the floor) leaving 126; `lunge_knee_valgus` **32** (none view-gated) leaving 142;
  `lunge_pelvic_drop` **111** (84 view-gated OR 33 could-not-fire, overlapping on 6) leaving 63.
  So "6 fires in 174" for depth and "10 in 174" for pelvic drop are partly statements about
  window length and view labeling, not only about which errors the dataset contains. The
  conditional tables restricted to the reps where each rule could act are in
  `notes/lunge-rule-validation.md` §4 and do not rescue either rule.
- **Two measurement conditions that cap every number above.** (1) **Frame validity**:
  `lunge_compute_raw`'s all-or-nothing landmark gate leaves only **74.0%** of cam17 frames and
  **58.4%** of cam18 frames carrying any metrics — the sagittal camera is worse, exactly as
  `pushup.py`'s equivalent note predicts. (2) **The lunge plan's §4.2 isolation claim is false as
  implemented**: it says ground-truth rep boundaries mean `segment_reps` is bypassed entirely, but
  `run_detector` segments whatever it is handed, and it re-cut the labeled window on **152 of 174**
  cam17 reps and **91 of 174** cam18 reps. Continuous scores are computed over exactly the frames
  the rules saw, so score support matches rule support, and the lead-side finding was reproduced
  independently over the full labeled windows — but the isolation the spec promised was not
  achieved. Forcing it needs `replace(LUNGE_DETECTOR, rep_signal=None)`; not done here.

**Status (2026-08-01):** **Deadlift** implemented in `src/pose/movements/deadlift.py` and
registered as the 5th of 16 — `deadlift_hips_shoot_up`, `deadlift_incomplete_lockout`,
`deadlift_lumbar_flexion`. `deadlift_bar_drift` is WITHDRAWN (see the boxed note in §Deadlift).
**Thresholds are spec-derived and UNVALIDATED**: no labeled deadlift data exists in this
repository, so unlike Lunge there is no validation pass to defer to, and §8.4 remains
unsatisfied for this movement. Deviations from the heuristics written above, deliberate and
documented in-code:

- `deadlift_hips_shoot_up` — the spec's "Δ(hip_y) rises faster than Δ(shoulder_y)" is
  **implemented as a trunk-pitch test**, not a two-landmark differential. The differential was
  checked numerically first and is algebraically identical to a pitch change: since
  `shoulder_y − hip_y = −torso_len·cos(pitch)`, a rigid torso gives
  `hip_lead_ratio ≡ cos(pitch₀) − cos(pitch_t)` exactly. It depends only on pitch and says
  nothing about hip travel, so writing it as a differential would falsely imply independent
  corroboration. The spec's own "i.e." equating the two phrasings is correct.
- Phase cutoffs are **percentiles of each rep's own hip-angle excursion**, not absolute angles.
  `deadlift_incomplete_lockout` scores the `lockout` phase and the fault *is* failing to reach
  extension, so an absolute cutoff would delete the phase on exactly the reps the rule exists
  to catch.
- `deadlift_lumbar_flexion` is **hard-gated** to sagittal views while the other two rules
  **degrade** off-view. The asymmetry is deliberate: an angle magnitude under-reads head-on
  (failure mode = silence), whereas the torso-shortening proxy is corrupted by trunk pitch
  head-on (failure mode = a false positive). Gate where a wrong claim is possible, discount
  where only a missed one is — the OHP `ohp_forward_head` precedent.
- `deadlift_incomplete_lockout` scores **both** the hip and knee ramps unconditionally and takes
  the worse, rather than selecting a ramp by which reading is finite — the mis-attribution bug
  §8's 2026-07-25 block records against `ohp_incomplete_lockout`. Because
  `severity == max(hip_sev, knee_sev)`, the reported `driver` axis cannot disagree with the
  reported severity. Its per-axis aggregate is guarded for the all-NaN case
  (`overhead_press.py`'s shape), which is reachable here precisely because the firing test ORs
  two independently finite-checked clauses.
- `deadlift_incomplete_lockout` scores the rep's **PEAK extension** (`nanmax` per axis), not a
  contiguous run of individually-failing frames — **corrected 2026-08-01 after the whole-branch
  review found the frame-window version emitting a false positive.** Because `lockout` is a
  *rank* cutoff (the 75th percentile of the rep's own hip-angle excursion, see the bullet
  above), a rep that spends under 25% of its frames above 165° gets a `lockout` band reaching
  *below* 165°, and a per-frame `< 165` mask fired on it. Measured on the segmented production
  path: a rep peaking at **178°** reported "incomplete lockout, minimum hip angle 148.5°" at
  severity 0.66 / observability "high". Peak-scoring introduces **no new number** (same 165/140
  ramp, different aggregate), matches the spec's own "at the top phase … at rep end" phrasing,
  and matches `ohp_incomplete_lockout`, which has always aggregated with `nanmax` before
  scoring. The percentile phase is deliberately unchanged — it is what guarantees a
  shallow-finishing rep still has a `lockout` phase to score. **General lesson: a percentile
  phase boundary and an absolute per-frame threshold inside that phase do not compose.** Squat,
  lunge and push-up should be checked for the same pairing.

**Status (2026-08-01) — Row detector registered.**

- **Row — IMPLEMENTED 2026-08-01, UNVALIDATED.** Four of five rules
  (`row_torso_rising`, `row_incomplete_rom`, `row_momentum_jerk`, `row_asymmetric_pull`);
  the fifth is recorded in §7 as a spec defect. `validated=False`: REHAB24-6 contains no row.
  Fit3D **does** contain row video (`barbell_row`, `barbell_dead_row`, `one_arm_row` in
  `data/Fit3D/fit3d_info.json`, 3D mocap ground truth under `train/*/joints3d_25/` and rep
  boundaries in `rep_ann.json`, across all 8 train subjects) — but, unlike REHAB24-6, it
  carries no binary correct/incorrect label, so it cannot support the fire-rate/AUC-against-
  correctness validation §8.4 means and the Lunge pass above ran. What Fit3D's 3D truth
  *can* support — the 2D-cue-vs-3D-truth fidelity comparison this project has already run
  elsewhere (`notes/fit3d_2d_vs_3d_summary.md` and related) — is possible for Row and simply
  was **not done in this pass**; it is future work, not blocked on absent data. (Caveat: Fit3D's
  rig is 4 cameras, all oblique, with no true side view, which bears on any Row rule needing a
  lateral component.) So §8.4's "validate thresholds against labeled data per movement" is
  **not** satisfied for Row — REHAB24-6 has no row at all, and Fit3D's row data has 3D truth but
  no correctness labels — and closing it needs either labeled row video or a fidelity-style pass
  against Fit3D, neither of which is blocked on nonexistent data. All four severity ramps are
  rule-level display curves (the Row section states none), and `row_momentum_jerk`'s
  self-normalizing 3×-median threshold is expected to over-fire. **A fifth limitation is
  measured, and now derived, not just observed:** `row_torso_rising`'s effective fire threshold
  is inflated by setup-baseline contamination, because on the **segmented** path `segment_reps`
  trims the rep window to the excursion and leaves a 2-frame `setup` slice, of which one frame
  is clean and the other is already loaded on an abrupt setup→peak transition. The median of two
  values is their mean, so the baseline lands exactly halfway between the true resting angle and
  the loaded peak value, and the measured rise (`peak − baseline`) is exactly **half** the true
  rise: for a true rise `R`, `peak − baseline = (base + R) − (base + R/2) = R/2`, so the 15° fire
  threshold requires `R > 30°` of real fault — **exactly 2×, derived from the trimming
  mechanism, not an empirical curiosity** — and this is invariant to how many extension frames
  precede the rep because `segment_reps` trims them off either way (measured at both 8 and 20
  setup frames; 40 was not tested and is not claimed). Measured end-to-end through
  `run_detector`, sweeping the true rise in 0.5° steps: an abrupt setup→peak transition fires at
  **30.5°** (the smallest step past the derived 30° boundary, 2.03×, matching the algebra above);
  a realistic 6-frame concentric ramp relaxes it to **19.5°** (1.30×) and a 3-frame ramp to
  **20.5°** (1.37×) — smaller than the abrupt case because a ramp spreads the loaded value across
  more of the setup slice instead of concentrating it in one frame.

  **The inflation applies to the segmented path only.** On the whole-clip fallback
  (`no_reps_detected` / `only_partial_reps` / `segmentation_disabled`) there is no window
  trimming — `setup` is the clip's first 15% by position, not by excursion — and the effective
  threshold measures at **15.5°**, the nominal 15° (the extra 0.5° is the sweep's step size, not
  inflation). The counterintuitive consequence: **the same clip can have a stricter effective
  threshold when rep segmentation *succeeds* than when it *fails*** — a torso-rising fault a
  lifter actually committed can go undetected specifically because the pipeline segmented their
  rep correctly. Direction is always toward MISSED faults, never false ones.

---

**Status (2026-08-09) — Arm Abduction registered, and §8.4 changes meaning for the first time.**

- **Arm Abduction — IMPLEMENTED 2026-08-09, UNVALIDATED.** `src/pose/movements/arm_abduction.py`,
  ninth of sixteen. **Two rules live** (`arm_abd_contralateral_trunk_lean`,
  `arm_abd_lr_asymmetry`), **one registered permanently silent** (`arm_abd_shoulder_shrug`), **one
  withdrawn** (`excessive_elevation_impingement_arc`). Each treatment is argued at its own rule
  above; design spec `docs/superpowers/specs/2026-08-09-arm-abduction-detector-design.md`.

- **§8.4 — "validate thresholds against labeled data per movement" — is now BLOCKED ON WORK
  RATHER THAN ON DATA, for the first time outside Squat and Lunge.** REHAB24-6 `Ex1` **is** arm
  abduction: **178 repetitions, 9 subjects, 90 correct / 88 incorrect**, marker 3-D and cached
  MediaPipe landmarks for all 13 videos. Every detector status note since OHP has said "no labeled
  data exists"; for this movement that is simply false, and the accurate statement is **nothing has
  run the check**. What running it looks like is `notes/lunge-rule-validation.md`.

- **Three things bound that future check in advance, and they are worth reading before it is
  scoped.** (i) Ex1 is **unilateral on 178/178 reps**, a variant this rule set does not model, so
  `arm_abd_lr_asymmetry` is unvalidatable there in either direction. (ii) `arm_abd_shoulder_shrug`
  is silent, so there is nothing to validate. (iii) `arm_abd_contralateral_trunk_lean` is the one
  rule Ex1 can genuinely speak to, and its cue already scores a **per-subject median AUC of 0.800**
  on the marker 3-D while the shipped 12° threshold fires on **0/178** — i.e. the check would
  measure a real cue against a cut sitting past the end of the observed distribution.

- **`validated=False`**, and the Beta tag stands. No threshold moved to produce any number above.


---

**Status (2026-08-10) — Torso Twist registered, 14/16, and Group F opens.**

- **Torso Twist — IMPLEMENTED 2026-08-10, UNVALIDATED.** `src/pose/movements/torso_twist.py`,
  fourteenth of sixteen. **One rule live** (`tt_trunk_not_braced`, its brace disjunct only),
  **one registered permanently silent** (`tt_insufficient_rotation_rom`), **two withdrawn**
  (`tt_lumbar_rotation_dominant`, `tt_momentum_over_control`). Each treatment is argued at the
  Group F update block above; design spec
  `docs/superpowers/specs/2026-08-10-torso-twist-detector-design.md`; 37 tests in
  `tests/test_torso_twist.py`.

- **§8.4 is BLOCKED ON DATA HERE, and it is worth being precise about which kind of gap that is.**
  Three corpora contain something called a torso twist and each contains a different exercise:
  REHAB24-6 has none at all, Fit3D's `standing_ab_twists` is a standing cross-body knee-to-elbow
  twist, and EgoExo-Fitness's **95 judged `Kneeling Side Torso Twist` actions** — the richest
  labelling this programme has met, per-criterion True/False rather than binary correctness — are
  a prone **lateral flexion**. So this is not Arm Abduction's "nobody ran the check" and not
  Shoulder Bridge's "the pixels are missing": the labels exist, they are good, and they are about
  other movements. Closing §8.4 for Torso Twist needs footage of a **seated Russian twist**, which
  no dataset in this repository supplies.

- **What DID run, and what it is allowed to conclude.** Fit3D's twist data was used for a
  **sensing-fidelity** pass — mocap 3-D ground truth projected through the real per-camera
  calibration, i.e. a *perfect detector*, measuring how much true axial rotation survives into
  this section's 2-D proxy. It ships as `src/fit3d/rotation_proxy_fidelity.py` +
  `scripts/fit3d/run_rotation_proxy_fidelity.py`, with its pure helpers unit-tested above the
  corpus banner, so every number quoted above is re-runnable — the Row residual recorded earlier
  in this section is exactly the failure of not doing that. That is **projection geometry**,
  which is about cameras and transfers
  across the variant mismatch; it withdrew `tt_lumbar_rotation_dominant` (16.7% decision
  disagreement at the section's own cut, 26 of 30 flips being false positives) and quantified the
  shipped rule's camera sensitivity. **No threshold was taken from that corpus**, because a
  threshold is about the exercise and does not transfer. This is also the first payment on the
  fidelity-comparison debt the Row status note recorded above as "future work, not blocked on
  absent data".

- **A defect in this document is recorded rather than silently worked around.**
  `tt_momentum_over_control`'s `citation_support` paraphrases the RAG doc as warning against
  between-rep momentum; the doc instructs the opposite ("it is crucial to **not stop** between
  repetitions"). The rule as written would fault a user for following its own source. Seventh
  distinct citation failure mode, and the first in which the contradiction sits inside the quoted
  document.

- **An app asset defect is recorded rather than fixed.**
  `frontend/src/components/movements/MovementIcon.tsx:148` draws a standing figure — comment and
  strokes both — for a movement whose card art, RAG doc and rep phases here are all seated.
  Changing it is a frontend change on a movement this branch is not about.

- **`validated=False`**, and the Beta tag stands. It is **Sit-up's** reason (the labeled data
  describes a different variant), not a sixth one; the count of distinct reasons stays at five.
  No threshold moved to produce any number above.
