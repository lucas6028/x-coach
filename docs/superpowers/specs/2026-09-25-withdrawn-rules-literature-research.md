# Withdrawn rules: literature re-search (2026-09-25)

**Outcome: no withdrawn rule is reinstated live; one is rewritten.** Every rule withdrawn under "no
citation supports the rule as written" was searched again, both as written and for a rewritten form
a monocular MediaPipe-33 detector could compute. No rule has all its failures cleared. One rewrite
has a primary source: the Arm Abduction impingement arc became `rule_raised_above_shoulder_height`
(Kolber 2014, lateral raise above 90°). It is registered **permanently silent**, because with the
citation repaired a measured sensing failure remains (§2.2). That moves it from "withdrawn" (no
citation) to "silent" (cited, sensor cannot read it). Several withdrawals got **stronger**, because the re-search found the cited sources arguing
against the rule (Sit-up speed, §3.1; Deadlift bar drift, §1.2).

The rule the verdicts apply: a new paper clears only a **citation** failure. A rule that also carries
a **measured** refutation (M) or an **architectural** blocker (A: no KG node, no view, no per-user
baseline, no cross-rep state, no landmark) stays withdrawn whatever the paper says. Rules with an M/A
blocker were re-checked lightly. The effort went to rules where the citation was the only failure, or
the deciding one.

Every candidate was screened against the nine citation failure modes that
`docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md` and the per-movement specs
record: fabricated authors, wrong inference, absence, exercise identity, secondary sourcing,
source-measured null, sign disagreement, inverted paraphrase, and graded family.

**Provenance.** Five parallel search agents ran the queries against Europe PMC REST, PMC, NCBI
E-utilities and web search; PubMed's own pages were behind a reCAPTCHA. Sources this document leans
on were then re-checked while writing it:
- **Against the PMC or Europe PMC record:** Kolber, Nakamura, Butowicz, Alberton, Edington, Andrade
  (MDC only), Juker, Lee 2013, Uebayashi and Gabis.
- **Against the local corpus (`data/rag/docs`):** Barbado and Mandroukas.

Anything marked **(agent report)** comes from a search agent's reading and was not re-checked here.
"Abstract only" means the full text was paywalled and not read. No source here was
added to `data/rag/docs` or `data/paper_metadata.json`, because nothing now cites any of them.

## Verdict table

| # | rule | verdict | blockers still standing |
|---|---|---|---|
| 1.1 | `ohp_forward_head` bar-path sub-criterion | nothing qualifying | M (re-describes back lean), A (mid-foot proxy) |
| 1.2 | `deadlift_bar_drift` | nothing qualifying; source measured a null | A (mid-foot / MTP proxy) |
| 1.3 | Row `rounded_thoracolumbar_spine` | light check, nothing | A (no spinal landmark) |
| 2.1 | `wrist_flexion_curl` | nothing qualifying | observability (occlusion) |
| 2.1b | curl elbow-displacement cue | nothing (descriptive only) | arithmetic (unreachable) |
| 2.2 | `excessive_elevation_impingement_arc` | as written: nothing. **Rewritten** → `rule_raised_above_shoulder_height`, registered SILENT | M (elevation magnitude over-read by 20.6°, toward firing) |
| 2.3 | Arm VW W-disjunct `< 75°` | nothing; source **contradicts** a floor | M (frontal plane cannot see the W) |
| 3.1 | `situp_excessive_speed` | nothing; cited source **argues against** it | A (per-user baseline) |
| 3.2 | `situp_excessive_rom` | nothing; the 35–40° is **unmeasured** in its source | A (KG node inverted), variant undecided |
| 3.3 | `tt_lumbar_rotation_dominant` | light check, nothing | M (2-D proxy unfit), A (no KG node) |
| 3.4 | `tt_momentum_over_control` | nothing (zero Russian-twist tempo studies) | A (no KG node) |
| 3.5 | `tt_trunk_not_braced` rounding disjunct | light check, nothing | A (image-x, mirroring) |
| 4.1 | `bridge_asymmetric_pelvic_drop` | two-leg: nothing. Single-leg: a number exists | A (variant not performed), no KG node |
| 4.2 | `bridge_knee_valgus` | nothing qualifying | M (correct reps already below the cut) |
| 4.3 | `abd_hip_flexion_er_substitution` | nothing qualifying (ER→TFL direction only) | A (needs side view), no KG node |
| 4.4 | `abd_momentum` | nothing (protocol tempos only) | A (per-user baseline), no KG node |
| 4.5 | Leg Abduction pelvic-tilt disjunct | nothing qualifying | — |
| 5.1 | `jj_knee_valgus_landing` | light check, nothing | M (zero-parameter control) |
| 5.2 | `jj_stiff_landing` | nothing qualifying | M (projection bias) |
| 5.3 | `jj_landing_asymmetry` | nothing qualifying | A (cross-rep, disjunction, no KG node) |
| 5.4 | `hk_trunk_lean_back`, `hk_forward_trunk_collapse` | light check, nothing | M (reference axis) |
| 5.5 | `hk_contralateral_pelvic_drop` | light check, nothing | M (three cameras disagree) |
| 5.6 | `hk_stride_asymmetry` | nothing qualifying | A (cross-rep, disjunction, no KG node) |

## 1. Barbell bar path and spinal shape

### 1.1 OHP bar-path sub-criterion

No peer-reviewed study was found that measures the horizontal bar position at overhead-press lockout
with a number, whether relative to the shoulder, the base of support or mid-foot. "Bar over mid-foot"
appears only in coaching web pages. Two corpus papers were re-checked:

- PMC12372072 (Evangelista 2025): its `d` is a grip-width model parameter, not a front-to-back
  bar-over-base measure (absence).
- PMC9354811 (Coratella 2022) states that "no kinematic data were recorded".

Queries: overhead / military / shoulder / push press combined with bar path, barbell trajectory,
horizontal displacement or barbell position (31 Europe PMC hits, none measuring A-P bar position),
plus jerk receiving-position and strict-press lockout-offset searches.

### 1.2 `deadlift_bar_drift`

Nothing qualifies, and the best source measured a **null** against the rule's own rationale.

- **Edington C et al. 2018**, "The Effect of Set Up Position on EMG Amplitude, Lumbar Spine Kinetics,
  and Total Force Output During Maximal Isometric Conventional-Stance Deadlifts", *Sports (Basel)*,
  PMC6162543, DOI 10.3390/sports6030090. It compares a close bar (over the navicular) with a far bar
  (over the 3rd metatarsophalangeal joint) and reports no statistical difference in lumbar shear or
  compression between the two. That is a source-measured null on the lever-arm mechanism, and the
  pulls were isometric (exercise identity).
- **Hancock S, Wyatt F, Kilgore JL 2012**, "Variation in Barbell Position Relative to Shoulder and Foot
  Anatomical Landmarks Alters Movement Efficiency", *Int J Exerc Sci* 5(3):183–195 **(agent report)**.
  It is not indexed in PubMed/PMC; the full PDF was read by the search agent. It reports posterior bar displacement of
  66.7 ± 12.9 mm (MTP–bar–AC alignment) against 37.5 ± 13.7 mm (navicular–bar–scapular spine), N = 6
  weightlifters. These are the means of two imposed conditions, not a fault cut. The displacement is
  **posterior**, the opposite sign to "drift forward", and the units need a scale MediaPipe lacks.
  Neither reference point is resolvable: `foot_index` (31/32) is the toe tip, not the MTP joint.
- Shoaib 2026, *Sci Rep*, PMC13569836: its expert rubric names "bar drifts forward" with no
  threshold. Its own limitation is that "depth information critical for assessing bar path … is
  absent in monocular video".

Queries: deadlift combined with bar path, barbell trajectory or horizontal displacement (60 Europe PMC
hits screened); the Hancock/Edington close-bar vs far-bar line; Escamilla sumo vs conventional
bar-to-ankle distance; "horizontal distance" from barbell to ankle, toe or metatarsal.

### 1.3 Row `rounded_thoracolumbar_spine`

This one is architectural. The markerless-spine sources found (SpineTrack, VideoPose3D spine nodes)
all need spinal keypoints that MediaPipe-33 does not have. No source validates a proxy for bent-over
row spinal flexion built from shoulder, hip and ear landmarks alone.

## 2. Upper limb

### 2.1 `wrist_flexion_curl` (and 2.1b, the elbow-displacement cue)

Nothing qualifies.

- Chua MX et al. 2024, "Analysis of Fatigue-Induced Compensatory Movements in Bicep Curls", arXiv
  2402.11421 **(agent report)**, is a **preprint**. It reports group-mean initial wrist flexion–extension of about −17°,
  −21° and −29° across the weight-free, standard and fatigue conditions. These are descriptive values
  with no fault claim and an unclear sign convention.
- For the same study, shoulder flexion was about 10° (standard) and about 20° (fatigue). Both are
  descriptive and both sit below the shipped 25° elbow-drift cut, so the displacement cue stays
  unreachable.
- Coratella 2023 (PMC10054060) contains no statement about wrist flexion or extension.

Queries: wrist flexion/extension angle with biceps curl and EMG; wrist position with elbow-flexor
torque or strength; dumbbell-curl wrist kinematics; neutral-wrist technique errors.

### 2.2 `excessive_elevation_impingement_arc`: a rewrite is supported, but for a different variant

As written, the rule is still unsupported. Kolber, below, also uses the painful arc only as a
diagnostic test, so the wrong-inference failure stands. A **rewrite** does have a primary source:

- **Kolber MJ, Cheatham SW, Salamh PA, Hanney WJ 2014**, "Characteristics of shoulder impingement in
  the recreational weight-training population", *J Strength Cond Res* 28(4):1081–1089, DOI
  10.1519/JSC.0000000000000250, PMID 24077379. The abstract was verified from the Europe PMC record:
  > "A significant association existed between clinical characteristics of SIS (p ≤ 0.004) and both
  > lateral deltoid raises and upright rows above 90°. … Avoiding performance of lateral deltoid raises
  > and upright rows beyond an angle of 90° … may serve as a useful means to mitigate characteristics
  > associated with SIS."

The rewrite it supports is `lateral_raise_above_shoulder_height`: fire when peak
`angle(hip, shoulder, elbow)` exceeds 90° during a **weighted dumbbell lateral raise**. The caveats:

- The design is cross-sectional and questionnaire-based; the authors call for prospective work
  before any causal claim.
- The "above 90°" was self-reported, not measured.
- The population is young male recreational weight-trainers.
- The paper's practical applications pair the risk with internal rotation, which MediaPipe cannot see.

**What was done: rewritten and registered silent.** The user authorised rewriting rules to what the
literature states. `rule_raised_above_shoulder_height` in `src/pose/movements/arm_abduction.py` is
the Kolber rule. It is registered **permanently silent**, the `rule_shoulder_shrug` treatment, for a
measured reason. The rule reads an elevation **magnitude** against an absolute 90°. On REHAB24-6 Ex1
cam17 (arm-abduction spec §2.4), MediaPipe's elevation error against the markers is a mean 20.6° per
rep (p90 43.8°), and it reads **high**: median peak 157.4° against 130.1°. An over-reading metric
against an upper bound fires on reps that stayed below it. That was the `front` camera, which
production never emits, so no view gate recovers it. The parent spec carries the rewritten entry
below the original, which stays as history.

KG: "Subacromial Impingement" matches only the shared Risk node, with no Arm Abduction fault node.
The seed is thin, not inverted. `Incomplete Elevation` means the opposite fault and is not used.
No frontend mistakes entry was added, matching the silent shrug rule, so the page does not
advertise a fault the app cannot report.

**Questions a live rule would also have to answer.** The measurement that would license firing is
Open decision 1.

- **Load.** Kolber's population performed *weighted* lateral deltoid raises in gym programmes. Is the
  app's Arm Abduction a weighted lateral deltoid raise, or an unloaded mobility/rehab abduction?
- **Evidence grade.** The association is cross-sectional and questionnaire-based. Does that clear the
  premise that every threshold be literature-backed?
- **Fire rate.** On Fit3D `side_lateral_raise` the median peak is 97.1° (arm-abduction spec §4.3), so
  a > 90° cut fires on more than half of those unlabeled repetitions. That follows from the median
  alone and was not separately measured.
- **Internal rotation.** Kolber pairs the risk with internal rotation above 90°, and MediaPipe cannot
  see humeral rotation.

`Arm Abduction:Incomplete Elevation` (the opposite fault) is still unsourced. Coratella 2020
(PMC7503819) and Larsen 2025 (PMC12277279) both prescribe 90° as the protocol top, but neither calls
a shorter range a fault or gives a tolerance.

### 2.3 Arm VW W-disjunct `< 75°`

Nothing qualifies, and the closest source **contradicts** the idea of a floor:

- **Nakamura Y, Tsuruike M, Ellenbecker TS 2016**, "Electromyographic Activity of Scapular Muscle
  Control in Free-Motion Exercise", *J Athl Train* 51(3):195–204, PMC4852525. It places the W
  position at 20° of abduction ("minimal shoulder abduction"), in the robbery exercise rather than the
  V-to-W (exercise identity), and the 20° sits behind reference marker 27 (secondary sourcing).
- Kara 2021 (PMC8675318) tests retraction at 0/45/90/120° and has no W.

The frontal-plane measurement failure stands regardless of the literature: medians of 24.6–67.9°,
firing on 90–99% of repetitions, AUC about 0.51.

## 3. Trunk: Sit-up and Torso Twist

### 3.1 `situp_excessive_speed`: the withdrawal is stronger

- **Barbado (PMC4519219), verified in the local corpus**, argues against the rule: "the fastest cadence
  in our study (1 repetition/1 s) could be used in young physically active individuals to produce high
  levels of trunk muscle activation without impairing trunk motion control". This is a sign
  disagreement between the citation and the rule.
- Barbado's "1 repetition/1 s" is the **whole up-and-back cycle**, so "concentric < ~1.0 s" is not the
  quantity the paper tested.
- Barbado's caution clause about spinal loads points to Axler & McGill 1997 (*Med Sci Sports Exerc*).
  That abstract never mentions speed (secondary sourcing that ends in absence).
- Vera-Garcia 2008 (*J Strength Cond Res*, abstract only) finds that trunk EMG rises with curl-up speed
  and treats speed as a training variable. Elvira 2014 (*Eur J Sport Sci*, abstract only) finds that
  sit-up speed increases hip and knee ROM. Neither states a fault threshold.

### 3.2 `situp_excessive_rom`: the 35–40° was never measured

- Mandroukas (PMC9505236), verified in the local corpus: the Discussion sentence "Rectus abdominis
  muscle activity was greatest in the early stages of trunk flexion and decreased as the range of
  motion became greater, more than 35–40°" carries no reference marker. However, the Methods use
  35–40° only as the curl-up's **protocol endpoint** ("a slow curl-up followed, with rounded back to
  approximately 35–40°"). They record no measured trunk angle and no angle-resolved EMG, so the
  study never observed activity "decreasing beyond 35–40°" (absence). The Introduction's
  35–40° cites the Nordin & Frankel textbook (secondary sourcing). This **corrects** the sit-up spec
  §6(a), which reads the Discussion figure as Mandroukas's own EMG result.
- Ha & Shin 2020 (*J Back Musculoskelet Rehabil*, abstract only) reports that abdominal EMG decreased
  across 30/60/90° and that "the most effective angle for curl-up was 30°". That is an effectiveness
  claim at three levels, not a fault cut.
- Juker 1998 (*Med Sci Sports Exerc*) reports psoas activity of 15–35% MVC in sit-ups against <10% in
  the curl-up. That is input for the **variant decision** (sit-up spec §10), not a rule.

The KG seed is still inverted and the variant is still undecided, so no source could have cleared this
rule.

### 3.3–3.5 Torso Twist

- `tt_momentum_over_control`: Europe PMC returns **zero** titles for "Russian twist". Marras 1995
  (*Spine*) finds dynamic twisting compression twice that of static, but it studies occupational
  axial twisting (exercise identity) and gives no tempo number. Kumar 1996's protocol ("in one smooth
  motion without stopping anywhere") agrees with the RAG doc's "not stop" and gives nothing for a
  dwell criterion.
- `tt_lumbar_rotation_dominant` and the rounding disjunct: no Russian-twist study quantifies pelvis
  against thorax rotation. The measured proxy failure and the mirroring problem both stand.

## 4. Group E: Shoulder Bridge and Leg Abduction

### 4.1 `bridge_asymmetric_pelvic_drop`

No study quantifies a pelvic fault in the **two-leg** bridge. A number exists only for the
**single-leg** variant:

- **Butowicz CM, Ebaugh DD, Noehren B, Silfies SP 2016**, "Validation of Two Clinical Measures of Core
  Stability", *Int J Sports Phys Ther*, PMC4739044. From the Procedures: "The test was terminated when
  they were no longer able to maintain a neutral pelvic position as noted by 10-degree change in
  transverse or sagittal plane alignment", monitored by an inclinometer belted to the pelvis. This is
  an endurance-test stop point with a gravity reference.
- Andrade 2012 (*Rev Bras Fisioter*, PMID 22899181) reports an MDC95 of 6.59° (abstract, verified)
  and a healthy peak pelvic tilt of 8.65 ± 5.74° in the same single-leg test **(agent report, full
  text)**. A 10° absolute cut would therefore sit inside normal variation.

The rewrite recorded for the day a single-leg bridge is added: a change of more than 10° in the
hip-line angle relative to the shoulder line, measured from the start of the hold, with a foot-end
view. It is untested. It does not apply to the two-leg bridge the app models.

### 4.2 `bridge_knee_valgus`

No primary study measures knee valgus, knee separation or hip adduction as a bridging fault. Kang 2016
(*Man Ther*) and Choi 2015 set hip abduction at 0/15/30° as **conditions**; 0° is a normal condition.
The measured noise floor (correct repetitions already below the cut) stands.

### 4.3 `abd_hip_flexion_er_substitution`

- Lee JH 2013 (*J Sport Rehabil*, abstract only) supports the **direction** of the claim: lateral
  rotation raises TFL activation. The study is side-lying and isometric, and gives no tolerance in
  degrees.
- Nothing found tests the **hip-flexion** half. The search agent cited Uebayashi 2026 (PMC13353230)
  for it, but the abstract's TFL null refers to the hip internal-rotation position, not to flexion, so
  it is not used.

The forward drift is a depth component and is invisible from the front view the shipped rule needs.

### 4.4 `abd_momentum` and 4.5 the pelvic-tilt disjunct

- Tempo: only protocol values exist, such as Madsen 2024 (PMC11577667), which prescribes 2 s
  concentric, 1 s isometric and 2 s eccentric. A protocol value is not a fault threshold.
- Pelvic tilt: Kang MH 2019 (*J Electromyogr Kinesiol*, abstract only) mentions compensatory lateral
  pelvic movement but gives no number and no sign (hike vs drop).

## 5. Unregistered movements: Jumping Jacks and High Knee

The local corpus has nothing on either movement except `jumping_jacks_wiki.txt`.

- `jj_knee_valgus_landing`: 23 Europe PMC hits combine jumping jacks with valgus. In every one the
  jumping jack is a **warm-up** and valgus is measured on a drop jump (exercise identity).
- `jj_stiff_landing`: no study measures knee flexion during jumping jacks. Fassett 2022 reports ground
  reaction force only. Costa 2011 (PMC3761520) is aquatic.
- `jj_landing_asymmetry`: the only left/right jumping-jack quantity found is Gabis 2020 (PMC7284634).
  It measures IMU timing lag averaged over five jumps (cross-rep), in children, as a group comparison
  with no cut-off.
- The High Knee trunk and pelvis rules: no running-in-place source states a trunk-angle or pelvic-drop
  tolerance. `hk_stride_asymmetry`: Peterka 2023 (PMC10157157) computes asymmetry across adjacent steps
  in a vestibular setting.
- **Cadence**, the promising unbuilt High Knee rule: nothing qualifies. Cho 2022 (PMC9090847) reports
  a healthy-control jumping-jack feet cycle time of 1.02 (0.13) s, a descriptive value with no
  cut-off. Gabis 2020 imposes a 1 Hz metronome as a study condition.

**Not a withdrawn rule, recorded for the permanently silent `hk_insufficient_knee_lift`.** Alberton
2015, "Kinesiological Analysis of Stationary Running Performed in Aquatic and Dry Land Environments",
*J Hum Kinet* 49:5–14, PMC4723158, instructs "hip and knee flexion to 90°" for dry-land running in
place. That matches this exercise better than the A-skip source. It is still a protocol instruction
with no tolerance, and the implemented 90° cut already fires on every repetition of every judged
action. It does not unsilence the rule.

## What this does NOT say

- It does not say these faults are unimportant. It says no source found states them in the way the
  rules are written, for the exercise the app models.
- "Nothing qualifying" is bounded by the queries recorded here and in each search agent's report.
  Paywalled full texts (Elvira 2014, Ha & Shin 2020, Vera-Garcia 2008, Lee 2013, Kang 2016/2019,
  Marras 1995) were judged on the abstract only.

## Open decisions (for the user, not taken here)

1. **Making `rule_raised_above_shoulder_height` live.** Three steps:
   - Run MediaPipe on Fit3D `side_lateral_raise`; `data/fit3d/derived/preds/mediapipe` has no
     predictions for it yet.
   - Compare the per-rep peak elevation with `joints3d_25` at the 90° cut. Ship live only if the two
     agree there.
   - Then answer the load and evidence-grade questions above, and add a `kg`-mode query and a
     frontend mistakes entry.
2. **Sit-up variant** (sit-up spec §10). Juker 1998 adds evidence to that decision; it does not
   settle it.
