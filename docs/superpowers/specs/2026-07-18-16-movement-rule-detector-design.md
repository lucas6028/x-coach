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

#### Forward head / bar path ahead of midline

- **fault_id**: `ohp_forward_head_barpath`
- **fault_name**: Forward head / bar path forward of midline
- **description**: The head juts forward and/or the bar finishes in front of the shoulders
  rather than stacked over the shoulder and mid-foot at lockout.
- **detection_heuristic**: Side view. (a) Forward head: horizontal offset of ear (7/8) or
  nose (0) ahead of the shoulder (11/12) along the anterior axis, normalized by shoulder
  width; flag when ear is anterior by > ~0.3 shoulder-widths. (b) Bar-forward: at lockout,
  wrist (15/16) horizontal offset anterior to shoulder; flag when wrist is not stacked
  roughly vertically over the shoulder (offset > ~0.3 shoulder-widths).
- **observability**: medium–high — `side` (both cues are sagittal-plane offsets); low from
  `front`.
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

#### Elbow drift forward
- **fault_id**: `elbow_drift_forward`
- **fault_name**: Elbow drifts forward (loss of elbow fixation)
- **description**: The elbow travels forward/up and away from the torso during the lift, adding shoulder flexion instead of keeping the upper arm fixed at the side.
- **detection_heuristic**: `upper_arm_lean = angle(shoulder→elbow vector, image-vertical-down)`. In setup the upper arm hangs ~vertical (≈0–10°). Flag if `upper_arm_lean > 25°` toward the anterior (wrist) side at any frame during concentric, or if elbow x-displacement anterior of the shoulder–hip vertical line exceeds `0.5 × upper_arm_length`. Direction: elbow moving anterior/superior relative to the shoulder.
- **observability**: medium — needs **side** or **front_oblique** (forward drift is largely in the sagittal plane; from a pure **front** view it collapses to depth-z and is low/unreliable).
- **biomechanical_rationale**: Forward elbow drift converts the curl into partial shoulder flexion, shifting load from biceps brachii to the anterior deltoid and reducing the target-muscle stimulus (performance loss).
- **citation**: Parpa K et al., *Muscles* (2025), PMC12550948, DOI 10.3390/muscles4040045.
- **citation_support**: The paper's validated proper-execution protocol states the arms were "fully extended at the sides, with the elbows kept close to the torso throughout the whole movement," with two investigators visually monitoring execution — i.e., the elbow staying fixed at the torso is the defined correct form, so anterior drift is a deviation from it. (Verified — read in RAG doc.)

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

---

### Arm Abduction (standing lateral / shoulder-abduction raise)

Rep phases: **setup/bottom** (arms adducted at sides, `arm_elevation_angle`≈0°) →
**concentric** (abduction/raise) → **top** (target ≈90°) → **eccentric** (controlled lower).

#### Shoulder shrug (upper-trap dominance)
- **fault_id**: `shoulder_shrug_elevation`
- **fault_name**: Shrugging / scapular elevation
- **description**: The shoulders hike up toward the ears (upper-trapezius dominance) during the raise, especially as the arm passes ~90°.
- **detection_heuristic**: `neck_gap = ear_y (7/8) − shoulder_y (11/12)`, per side, relative to the setup-baseline `neck_gap`. Flag if `neck_gap` shrinks `> 18%` below baseline (shoulders rise toward ears) during the raise; escalate severity when it co-occurs with `arm_elevation_angle > 90°`. **Confound**: some acromion/shoulder elevation is normal scapulohumeral rhythm at high elevation — the discriminating fault signal is *early or disproportionate* shrug (`neck_gap` collapse while `arm_elevation_angle < 90°`), so weight early-phase shrug more heavily to avoid firing on clean high-elevation reps.
- **observability**: high — **front** or **rear** view (vertical shoulder elevation is in-plane and clearly resolved).
- **biomechanical_rationale**: Persistent upper-trapezius overactivation with under-active lower scapular stabilizers drives scapular dyskinesis and raises the risk of subacromial impingement and glenohumeral instability.
- **citation**: Mun WL, Jung EY, Lei S, Roh SY, *Medicina* (2025), PMC12029123, DOI 10.3390/medicina61040645.
- **citation_support**: "Persistent overactivity of the UT can lead to scapular dysfunction (or dyskinesia), such as subacromial impingement or glenohumeral instability," and UT activation "consistently increases as the shoulder abduction angle surpasses 120°" so "care should be taken to avoid the excessive activation of the UT" at higher angles. (Verified — read in RAG doc.)

#### Excessive elevation through the impingement arc
- **fault_id**: `excessive_elevation_impingement_arc`
- **fault_name**: Raising past safe/target ROM (impingement arc)
- **description**: The arm is driven high into (and through) the painful mid-abduction arc, or well past the prescribed target height, with poor scapular control.
- **detection_heuristic**: `arm_elevation_angle` = angle(shoulder→hip vs shoulder→elbow) in the frontal plane. Flag sustained `arm_elevation_angle` in ~70–120° performed with a concurrent shrug (`shoulder_shrug_elevation` true), or raising `> target + 15°` (e.g., `>105°` when the prescribed target is 90°).
- **observability**: high — **front** view (frontal-plane elevation is well measured). From a **side** view the arm overlaps the torso → low.
- **biomechanical_rationale**: Between ~70–120° of abduction the subacromial space narrows and the supraspinatus/long-head-biceps tendons and subacromial bursa are compressed (the "painful arc"); repeatedly loading through this arc with inadequate scapular upward rotation risks impingement.
- **citation**: Creech JA, Busse A, Li D, et al. *Shoulder Impingement Syndrome*, StatPearls (NCBI Bookshelf NBK554518, updated 2026); supported by Mun WL et al., *Medicina* (2025), PMC12029123.
- **citation_support**: StatPearls: the painful arc occurs "between approximately 70° and 120° of active shoulder abduction," where the subacromial space (normally 1–1.5 cm) "narrows physiologically with abduction," compressing the supraspinatus tendon, long head of biceps, and subacromial–subdeltoid bursa. Mun et al. corroborate elevated UT/impingement risk above 120°. (Verified — fetched StatPearls + read RAG doc.)

#### Contralateral trunk lean
- **fault_id**: `contralateral_trunk_lean`
- **fault_name**: Trunk lean to the opposite side
- **description**: The torso side-bends away from the working arm to help hoist it (frontal-plane compensation).
- **detection_heuristic**: `lateral_trunk_lean` = angle of mid-shoulder→mid-hip vector from image vertical in the frontal plane (uses the x-offset between mid-shoulder and mid-hip). Flag if lateral lean `> 12°` away from the raising arm during concentric, or if it grows with load across a set. (For a single-arm raise, sign the lean relative to the working side.)
- **observability**: high — **front**/**rear** view (lateral lean is in-plane).
- **biomechanical_rationale**: Contralateral lean substitutes trunk lateral flexors for deltoid/scapular work, reducing target loading and indicating insufficient shoulder strength/control; the accompanying poor scapular mechanics is part of the impingement-risk pattern.
- **citation**: Creech JA, Busse A, Li D, et al. *Shoulder Impingement Syndrome*, StatPearls (NCBI Bookshelf NBK554518, updated 2026).
- **citation_support**: StatPearls attributes impingement in part to "inadequate scapular upward rotation and posterior tilt" — i.e., compensation that fails to control the scapula during elevation, which contralateral trunk lean is a gross form of. The injury mechanism (impingement from poor scapular control during elevation) is verified via StatPearls. The specific frontal-plane trunk-lean substitution during abduction is **UNVERIFIED** in a peer-reviewed source (no read source isolated trunk lateral flexion during abduction; only fitness-coaching sources describe it, which do not qualify as injury-risk support). (Partially verified — injury mechanism verified; trunk-lean-specific EMG/kinematic finding UNVERIFIED.)

#### Left/right asymmetry
- **fault_id**: `lr_abduction_asymmetry`
- **fault_name**: Left vs right asymmetry
- **description**: One arm lags, rises less, or is timed differently from the other during a bilateral raise.
- **detection_heuristic**: Compare sides: `asym = |arm_elevation_angle_L − arm_elevation_angle_R|`. Flag if `asym > 12°` at the top-hold, or if peak wrist heights differ by `> 0.05` normalized units, sustained across reps.
- **observability**: high — **front**/**rear** view (both arms visible, elevation in-plane).
- **biomechanical_rationale**: Inter-limb asymmetry reflects unbalanced strength/scapular control; asymmetries in the ~10–15% range are associated with elevated injury risk and reduced performance.
- **citation**: Terré M, Solana-Tramunt M, *Healthcare (Basel)* (2025), 13(10):1153, PMC12110944, DOI 10.3390/healthcare13101153.
- **citation_support**: The paper states "asymmetries between 10% and 15% are often associated with a higher risk of injury and reduced performance," and uses a limb-symmetry scale (asymmetry 0–79%, limit 80–89%, normal/symmetrical 90–100%). (Verified — fetched PMC article.)

---

### Arm VW (scapular V-to-W protraction/retraction)

Open-chain scapular drill: arms overhead/wide in a **V (Y)** with the scapulae elevated/upwardly
rotated → pull the elbows down and back into a **W** with scapular retraction + depression →
brief isometric hold → return to V. Rep phases: **V/protraction-elevation** →
**pull-down/retraction** → **W hold (isometric)** → **return to V**.

#### Incomplete scapular / arm excursion
- **fault_id**: `incomplete_scapular_rom`
- **fault_name**: Incomplete protraction/retraction excursion
- **description**: The movement is shallow — the arms/scapulae don't reach full retraction+depression in the W (or full elevation in the V).
- **detection_heuristic**: Use the visible arm-excursion proxy for the (non-observable) scapular travel: vertical travel of wrist/elbow between phases, `excursion = wrist_y(V) − elbow_y(W)`, and elbow descent to shoulder line at W. Flag if `arm_elevation_angle` swing between V and W phases `< 40°`, or elbow fails to descend to within `0.05` (normalized y) of the shoulder line at the W. True A-P scapular retraction is not directly measured (see observability).
- **observability**: medium for the **arm-elevation excursion** (front view); **low** for true scapular protraction/retraction, which is an anterior–posterior depth motion not resolvable from a monocular front view.
- **biomechanical_rationale**: Greater scapular excursion increases trapezius recruitment; a truncated excursion under-loads the middle/lower trapezius the drill is meant to train (performance loss).
- **citation**: Jung EY, Roh SY, Mun WL, *Life* (2025), PMC12734928, DOI 10.3390/life15121840.
- **citation_support**: The study found the larger-excursion variation (sternum-drop, STD) "elicited higher trapezius activation, especially during large scapular excursions," and that "greater scapular excursion is known to increase muscle activation" (end-range positions were marker-verified). (Verified — read in RAG doc.)

#### Shrug substitution
- **fault_id**: `shrug_substitution`
- **fault_name**: Upper-trap shrug substitution
- **description**: The upper trapezius takes over (shoulders shrug up toward the ears) instead of the lower trapezius/serratus performing scapular depression and retraction.
- **detection_heuristic**: `neck_gap = ear_y − shoulder_y` vs setup baseline. During the pull-down/retraction and W-hold — where the shoulders should stay depressed — flag if `neck_gap` shrinks `> 18%` below baseline (shoulders rising). **Confound**: the V phase legitimately elevates the shoulders (arms overhead), so restrict this flag to the pull-down/W-hold phases where depression is expected; the discriminating signal is shoulders *rising when they should be depressing*, not absolute elevation.
- **observability**: high — **front**/**rear** view (vertical shoulder elevation in-plane).
- **biomechanical_rationale**: Upper-trap dominance (a high UT/LT and UT/SA activation ratio) is the maladaptive scapular-dyskinesis pattern and defeats the lower-trap/serratus training aim.
- **citation**: Abiara S et al., *PeerJ* (2025), PMC12335237, DOI 10.7717/peerj.19861; supported by Jung EY et al., *Life* (2025), PMC12734928.
- **citation_support**: Abiara et al.: "ratios lower than 1.0 for the UT/LT ratio are preferred … although lower than 0.6 are ideal," and shoulder pain is "characterized by increased activation of the upper trapezius and decreased activation of the lower trapezius and serratus anterior." Jung et al.: "excessive UT dominance is linked to scapular dyskinesis," and lower UT/SA ratios reflect "a more favorable stabilization pattern." (Verified — read both RAG docs.)

#### Loss of arm-elevation angle
- **fault_id**: `loss_of_elevation_angle`
- **fault_name**: Loss of target V/W elevation angle
- **description**: The arms fall below the prescribed elevation in the V (or the elbows drop too low in the W), moving off the lower-trap-optimal position.
- **detection_heuristic**: `arm_elevation_angle` per side. Flag if V-phase peak `< 120°` (arms not raised high enough) or W-phase abduction `< 75°` (elbows collapsed toward the body). Thumbs-up/forearm orientation is not reliably measured monocularly and is not required for the flag.
- **observability**: high — **front** view (frontal-plane elevation well measured).
- **biomechanical_rationale**: Lower-trapezius activation is maximized near ~135° of shoulder abduction (aligned with its fiber direction); losing the elevation angle moves the scapula off the LT-optimal position and reduces the exercise's targeted effect.
- **citation**: Mun WL et al., *Medicina* (2025), PMC12029123, DOI 10.3390/medicina61040645; supported by Abiara S et al., *PeerJ* (2025), PMC12335237.
- **citation_support**: Mun et al.: "the LT activation was the highest at a 135° shoulder abduction angle, with excessively high angles leading to a decrease," and researchers "recommend shoulder abduction near 145°, aligning with the muscle fiber direction, for maximum LT activation." Abiara et al. describe the LT-targeting exercise performed with "arms abducted above 90°, thumbs up." (Verified — read both RAG docs.)

#### Left/right asymmetry
- **fault_id**: `lr_vw_asymmetry`
- **fault_name**: Left vs right scapular asymmetry
- **description**: One arm/scapula lags, sits lower, or retracts less than the other through the V→W cycle.
- **detection_heuristic**: `asym = |arm_elevation_angle_L − arm_elevation_angle_R|` at the V peak and at the W hold. Flag if `asym > 12°`, or if `|wrist_y_L − wrist_y_R| > 0.05` (normalized), sustained across reps.
- **observability**: high — **front**/**rear** view (both arms visible, elevation in-plane; A-P retraction asymmetry itself remains low-observability).
- **biomechanical_rationale**: Asymmetric scapular control reflects side-to-side stabilizer imbalance; inter-limb asymmetries of ~10–15% are associated with higher injury risk and reduced performance.
- **citation**: Terré M, Solana-Tramunt M, *Healthcare (Basel)* (2025), 13(10):1153, PMC12110944, DOI 10.3390/healthcare13101153; scapular-dyskinesis context from Jung EY et al., *Life* (2025), PMC12734928.
- **citation_support**: Terré & Solana-Tramunt: "asymmetries between 10% and 15% are often associated with a higher risk of injury and reduced performance" (limb-symmetry scale: normal 90–100%). Jung et al. tie unbalanced scapular muscle activation to scapular dyskinesis. (Verified — fetched PMC article + read RAG doc.)

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
