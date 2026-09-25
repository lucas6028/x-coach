# Fit3D SMPL-X collar elevation moves 29–37° per shrug rep while the arm moves 9–10°; its rep-level independence from the arm during raises (about 1°) is undetermined

*Fit3D train split (8 subjects) · SMPL-X fit analysis, no estimator · run 2026-09-25 · step 1 of the
SMPL shoulder-girdle question; follows the withdrawn-rule re-search in
`docs/superpowers/specs/2026-09-25-withdrawn-rules-literature-research.md`*

**Result.** In the barbell shrug positive control, the fitted SMPL-X collar joint elevates by a
median 28.6° (left) and 37.0° (right) per rep. Over the same reps the arm moves by a median 8.7° and 10.3°, a signed change between the
shrug's lowest and highest collar frames (n = 41 reps per side). The shrug amplitude exceeds the same subject's reverse-lunge
amplitude in 7/8 subjects (left) and 8/8 (right). So the fits are not frozen at an average
collar pose. During side lateral raises and scaptions, the collar-elevation change left
unexplained by the arm-elevation change is 0.87–1.40° (2.0–2.8 mm at the shoulder joint, n = 40–41
reps, within-subject R² 0.17–0.90). Whether that residual is independent motion or fit error is
undetermined.

**Caveat on reading it.** The stop/pass rule was fixed before the rep-level residuals were computed, but
its noise floor is not identifiable. Depending on the smoothing window, the residual is 0.23× to
5.4× the floor, and a window-free estimate puts it at 12–17×. The rule's per-side verdicts
(left STOP, right PASS) are therefore not interpreted. The collar "ground truth" is itself a fit,
and Fit3D has no fault labels.

## Background

MediaPipe's shoulder landmark moves with the humerus, so the shrug and scapular-retraction rules
are registered silent (`arm_abduction.rule_shoulder_shrug`, `arm_vw.rule_shrug_substitution`,
`band_pull_apart.rule_loss_of_scapular_retraction`). SMPL-X has a separate collar joint between
spine3 and each shoulder. Step 1 asks whether Fit3D's SMPL-X fits carry collar motion the arm
does not explain. Step 2, not run here, would ask whether an image-to-mesh model such as NLF
reproduces it.

## Terminology

| term | meaning here |
| --- | --- |
| collar elevation | angle of the collar→shoulder direction above horizontal in the thorax (spine3) frame, minus its rest-pose value; positive = shoulder raised |
| collar protraction | forward angle of the same direction minus rest; negative = retraction |
| arm elevation | humerothoracic elevation: angle between shoulder→elbow and thorax-down (0° = at side) |
| horizontal abduction | arm angle in the thorax transverse plane from straight forward toward its own side |
| driver | the per-rep signal whose min and max frames define a rep's two sample points |
| residual SD | SD of Δcollar after regressing on Δarm with per-subject intercepts |
| jitter floor | √2 × pooled SD of collar elevation around a centred moving average (15 frames in the rule) |
| white-noise floor | √2 × second-difference noise estimate, sd(x[t+1] − 2x[t] + x[t−1]) / √6; needs no window |

## Method

All angles come from Fit3D's SMPL-X `body_pose` rotation matrices and the SMPL-X neutral rest
skeleton, in the thorax frame, so global orientation and trunk posture drop out. Left and right
are mirror-consistent by construction. A unit test pins this: a rotation mirrored across the
sagittal plane gives identical angles on the other side.

Reps come from Fit3D `rep_ann.json` (5 per clip; 40–41 per action and side). Per rep, Δcollar
and Δarm are taken between the driver's minimum and maximum frames. The drivers are:

- arm elevation for `side_lateral_raise` and `dumbbell_scaptions`, the Arm Abduction analogues;
- collar elevation for `barbell_shrug`, the positive control;
- horizontal abduction for `band_pull_apart`.

`overhead_trap_raises` deliberately moves the scapula, so it is descriptive only. `w_raise` was
excluded because it is not the app's V-to-W.

The rule was fixed in `src/fit3d/collar_elevation.py` before the rep-level residuals were
computed. By then two things had been seen: the shrug control, and per-rep collar and arm ranges
plus within-clip correlations on 3 subjects.

| verdict | condition |
| --- | --- |
| STOP | residual SD ≤ 2 × jitter floor |
| NOISY | residual SD > 2 × floor, but shrug median amplitude < 3 × residual SD |
| PASS | residual SD > 2 × floor and shrug median amplitude ≥ 3 × residual SD |

## Gates

| gate | result |
| --- | --- |
| angle helpers: identity reads 0°, a known collar rotation reads its angle, mirrored sides agree, shoulder rotation leaves collar angles unchanged | pass (10 unit tests) |
| positive control: collar moves when the girdle moves and the arm does not | pass; shrug collar amplitude > same-subject reverse-lunge amplitude in 7/8 (L) and 8/8 (R) |
| noise floor identifiable | **fail**; the floor depends on the smoothing window (see Bounds) |

## Results

Collar elevation against arm elevation, per rep, within subject:

| action | side | n reps | slope Δcollar/Δarm | R² within | residual SD | residual at shoulder | median Δcollar | median Δarm | rule verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| side_lateral_raise | L | 40 | +0.346 | 0.90 | 0.87° | 2.0 mm | 24.6° | 71.5° | STOP (not interpreted) |
| side_lateral_raise | R | 40 | +0.468 | 0.88 | 1.30° | 2.6 mm | 31.5° | 75.7° | PASS (not interpreted) |
| dumbbell_scaptions | L | 41 | +0.326 | 0.56 | 1.04° | 2.4 mm | 37.8° | 106.4° | STOP (not interpreted) |
| dumbbell_scaptions | R | 41 | +0.140 | 0.17 | 1.40° | 2.8 mm | 51.8° | 110.8° | PASS (not interpreted) |
| overhead_trap_raises | L | 41 | +0.387 | 0.32 | 2.39° | 5.6 mm | 25.2° | 86.0° | descriptive |
| overhead_trap_raises | R | 41 | +0.652 | 0.58 | 2.45° | 5.0 mm | 25.8° | 90.6° | descriptive |

Per-subject median Δcollar/Δarm ratios:

- side_lateral_raise: 0.32 ± 0.07 (L) and 0.35 ± 0.13 (R); s08 is lowest on both sides (0.15, 0.07).
- dumbbell_scaptions: 0.38 ± 0.04 (L) and 0.48 ± 0.06 (R).

Per-subject residual RMS stays within 0.3–2.0° on every raise action and side, so no single
subject drives the pooled residual.

The collar raises with the arm in every subject. The part the arm does not explain is about 1°.
The arm excursion itself varies little rep to rep (within-subject SD 3.5–7.5°), so a low R²
here reflects a narrow Δarm range, not independence.

## Secondary

Band pull-apart (driver = horizontal abduction; median Δ 66.3° L, 62.6° R):

| pair | side | slope | R² within | residual SD | median Δcollar | jitter floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| protraction vs horizontal abduction | L | −0.104 | 0.08 | 2.50° | −26.6° | 0.53° |
| protraction vs horizontal abduction | R | +0.002 | 0.00 | 3.39° | −24.6° | 0.52° |
| elevation vs horizontal abduction | L | −0.202 | 0.29 | 2.23° | −12.0° | 0.37° |
| elevation vs horizontal abduction | R | −0.070 | 0.06 | 1.96° | −21.4° | 0.48° |

The collar retracts by a median 25–27° as the arms open. Rep-to-rep variation of 2.5–3.4° is
not linearly related to how far the arms opened. Fit3D has no retraction-only action, so the
retraction channel has no positive control and no headroom verdict.

The right collar moves more than the left in the shrug control: 37.0° vs 28.6° median, 8/8
subjects larger on the right. The mirror unit test rules out a side-convention error on a symmetric skeleton. The real SMPL-X
neutral template is asymmetric (rest clavicle 0.133 m L, 0.116 m R), but it does not produce the
gap. Left collar rotations mirrored onto the right side of the template give a median 28.6°,
the same as on the left. The difference is in the fitted rotations. Whether it is anatomy or
fitting is not resolved here.

## Bounds

| floor | what it measures | raise-action floor | residual / floor |
| --- | --- | --- | --- |
| white-noise (second difference) | frame-to-frame fit noise; a lower bound, since the fits are temporally smooth and correlated error is invisible to it | 0.070–0.087° | 12–17× |
| jitter, 15-frame window (the rule's) | high-frequency part of the series | 0.49–0.70° | 1.8–2.3× |
| jitter, 9 / 25 / 51 frames | same, other windows | — | 4.1–5.4× / 0.76–0.95× / 0.23–0.29× |
| reverse lunge, per-rep collar amplitude | collar excursion with the arms hanging; includes real girdle motion | median 12.6° (L), 14.5° (R), max 23.1° | — |

The jitter floor changes with the smoothing window by more than the margin the rule needs. The
white-noise floor needs no window, but it is a lower bound that misses correlated fit error. A floor independent of the fit, such as repeated fits of
the same frames or marker-based scapular data, is not available here. The lunge row also bounds
how specific a collar-elevation excursion is: 12.6–14.5° per rep happens without a shrug.

## Deviations from the plan

| item | plan | what happened | what it changes |
| --- | --- | --- | --- |
| plan file | a `notes/*_validation_plan.md` committed before the run | the rule was written into the module docstring before the rep-level residuals were computed, after per-rep ranges and within-clip correlations on 3 subjects had been seen; not committed first | the pre-registration is not verifiable from git |
| noise floor | 15-frame jitter floor decides STOP/PASS | floor not identifiable; verdicts reported but not interpreted | step 1 does not return STOP or PASS |
| added floor | none | white-noise floor and window sweep added after the first run | reported as bounds only; the rule is unchanged |
| added comparison | none | per-subject shrug vs reverse-lunge amplitude added after the first run | used as the positive-control gate |

## Not supported

- Not a validation of any rule. Fit3D has no fault labels, and no action here contains a
  labelled shrug or loss of retraction.
- Not evidence that the collar joint tracks the scapula. The fits come from 53 VICON markers,
  4-view 2D keypoints and a normalizing-flow body prior (per the Fit3D SMPL-X annotation paper,
  PMC13413046). Marker placement over the shoulder girdle is not stated, so the collar/arm
  split during raises may partly come from the prior.
- Not evidence that the rep-level residual (about 1°) is real motion. It is undetermined.
- Not evidence that a monocular image-to-mesh model recovers any of this; that is step 2.
- Not a re-test of the Arm Abduction neck-gap refutation. That refutation was measured on
  `joints3d_25`, which is regressed from these same SMPL-X fits (per `src/fit3d/axial_rotation.py`).
  Its "collapses as a matter of anatomy" wording rests on the fit, and inherits this caveat.
- Nothing about winging. The SMPL-X collar has no degree of freedom for the medial border lifting
  off the ribcage.

## Reproduce

- Code: `src/fit3d/collar_elevation.py` (angles, rep deltas, within-subject fit, rule),
  `scripts/fit3d/run_collar_elevation.py` (runner), `tests/test_fit3d_collar_elevation.py`.
- Inputs: `data/fit3d/train/<subj>/smplx/<action>.json`, `data/fit3d/train/<subj>/rep_ann.json`,
  SMPL-X neutral model at `.kaggle_tmp/smplx_ds/SMPLX_NEUTRAL.npz`.
- Commands, from the repo root:
  `.venv\Scripts\python.exe -m pytest tests/test_fit3d_collar_elevation.py`, then
  `.venv\Scripts\python.exe scripts/fit3d/run_collar_elevation.py --json data/fit3d/derived/collar_elevation.json`.
- Runtime: under a minute on CPU; no GPU.
- Artifact: `data/fit3d/derived/collar_elevation.json`. It holds every number above, including
  the per-subject shrug-vs-lunge amplitudes, the per-subject Δcollar/Δarm ratios, the window
  sweep and the template mirror check.
