# Fit3D axial rotation — can 3D joint centres see a segment's roll?

**Question.** The depth thread established that a 2D keypoint pipeline is blind to out-of-plane depth
and that direct image→3D fixes it. This is the follow-up for *3D* keypoints. A joint-centre skeleton
places one point per joint, so it fixes where each segment **is** but seemingly not how it is **rolled
about its own long axis**. Femoral internal rotation is the textbook mechanism behind knee valgus, and a
three-point knee angle is invariant to it — so on paper this is a blind spot that survives a *perfect*
3D detector. If true, an explicit rotation channel (SMPL pose parameters) would carry information no
keypoint set can hold, which is a representation-limit argument rather than a sample-size-limited one.

**Method.** Ground truth on both sides, so no estimator quality enters:

| | source | shape |
|---|---|---|
| **target** | Fit3D SMPLX `body_pose`, twist about the rest femur axis (swing-twist) | (F, 21, 3, 3) → (F,) deg |
| **input** | Fit3D `joints3d_25` joint centres = a **perfect** keypoint detector | (F, 25, 3) |

Nested joint sets localise *where* any information comes from; LOSO over the 8 train subjects; ridge and
RBF kernel-ridge, **λ tuned per joint set** (being generous to keypoints keeps the bar honest). Scores are
`R2_within` — against the held-out subject's *own* mean — because between-subject offsets are not the
coaching signal; the within-subject variation is. Code `src/fit3d/axial_rotation.py`,
CLI `scripts/fit3d/run_axial_rotation.py`, tests `tests/test_fit3d_axial_rotation.py` (36 pass).
`swing_twist` recovers known swing/twist pairs to ~1e-14 rad. The quaternion maths is numpy-only
(Shepperd conversion + Rodrigues): scipy is an **optional** dependency here -- absent from both
requirements files and lazily imported by its two other users in `src/` -- so a module-level
import would break CI, which installs `requirements-ci.txt` only. A skipped-if-absent test
cross-checks the numpy path against scipy to 1e-9 across all four Shepperd branches.

## 0a — there is signal to recover

Left-hip femoral twist, squat, 8 subjects:

| quantity | value |
|---|---|
| pooled sd | **8.99°** |
| **within-rep sd** | **5.78°** |
| between-subject sd | 4.06° |
| range p1..p99 | −31.1 .. 10.8° (42° span) |

It varies *inside a single rep*, so it is a movement variable, not a body-shape constant. Per-subject
within-rep sd spans 2.0° (s07) to 10.8° (s05) — a 5× spread in how much subjects rotate.

## 0b — the naive blind-spot claim is REFUTED

Squat, left hip, λ tuned (`R2_within` / MAE°):

| joint set | dims | R2_within | MAE |
|---|---|---|---|
| femur_only (hip+knee) | 9 | +0.349 | 4.79 |
| leg (+ankle) | 12 | +0.415 | 4.57 |
| both_legs | 21 | +0.466 | 4.32 |
| upper_only (no legs) | 27 | +0.104 | 5.50 |
| **h36m17** | 51 | **+0.548** | **3.98** |
| full25 | 75 | +0.458 | 4.16 |

Across 6 action×side conditions (squat/deadlift/lunge × L/R) the best keypoint model reaches
**MAE 2.56–3.98°**, `R2_within` 0.30–0.76. **A full-body keypoint skeleton estimates femoral axial
rotation to ~3–4°.** Keypoints never *observe* the roll geometrically, but they *predict* it through
whole-body posture correlation.

What survives is only the **single-segment** claim, and it survives strongly. `femur_only` — the femur's
own two endpoints — goes sharply negative on other conditions (deadlift R −0.344, reverse-lunge R
**−2.913**). A predictor that far below the held-out subject's own mean is not merely uninformative: the
joint-centre configuration of a segment is **actively misleading** about that segment's roll. Two
endpoints do not determine it, and a model that tries to read it off them does worse than not trying.
Everything above that level comes from the rest of the body, not from the segment.

## 0c/0e — where the whole-body signal comes from (and a retracted claim)

**Retracted.** An earlier pass with λ fixed at 10 showed leg-only sets at `R2_within ≈ 0` and a large
contralateral-leg jump, and was read as "keypoints see only the valgus shadow". Both were **λ artefacts**:
with λ tuned, `femur_only` is +0.349 (not +0.076) and the contralateral gain is +0.052 (not +0.39).

The frame ablation explains why the keypoint-frame numbers were untrustworthy. `canonicalize` builds its
lateral axis from `L_HIP − R_HIP`, so it **leaks bilateral information into every joint set**, including
nominally unilateral ones. Re-canonicalising in the true SMPLX pelvis frame (`canonicalize_gt`, an
**oracle** — same role as the oracle per-view offset removal in `decision_eval`):

| frame | leg_L | both_legs | contralateral gain |
|---|---|---|---|
| keypoint (leaks bilateral) | +0.415 | +0.466 | **+0.052** |
| **GT pelvis (oracle)** | **+0.223** | **+0.762** | **+0.539** |

Under a clean frame the unilateral chain drops to +0.223 and the contralateral leg is worth **+0.539** —
so bilateral information is real and large; the keypoint-frame experiment had simply pre-mixed it in.
Explicit bilateral scalars on top of `leg_L` (knee_width_ratio and two relatives) reproduce the small
keypoint-frame gain (+0.484 vs +0.466), consistent with a valgus-shadow reading *in that frame*, but that
gain is too small to carry the mechanism claim on its own.

## 0f — temporal context does not lower the bar

Hip twist is smooth in time and every model above was per-frame, so the bar could have been an artefact
of a weak baseline. Adding velocity + acceleration (`vel`) and a ±6-frame window (`vel_win`) to `h36m17`:

| action | side | frame | vel | vel_win |
|---|---|---|---|---|
| squat | L | **4.00** | 4.17 | 4.09 |
| squat | R | **3.72** | 3.73 | 3.84 |
| deadlift | L | **2.57** | 2.65 | 2.79 |
| deadlift | R | **3.23** | 3.30 | 3.22 |

(MAE°, λ tuned over ridge and RBF-KRR.) Every temporal variant is flat or slightly worse — the extra
dimensions cost more than the smoothness buys at n≈2000. **The per-frame baseline is the strongest one**,
so the bar below stands.

## What Exp 0 actually delivers: the bar

**A monocular rotation estimator must beat 2.6–4.0° MAE (squat L 4.00°, squat R 3.72°, deadlift L 2.57°,
deadlift R 3.23°) to add anything at all over a perfect keypoint skeleton.** Nothing in the earlier plan
established this number, and it is demanding — published monocular SMPL joint-rotation error is typically
well above it. The line may well die at the estimator step; that is a successful kill test, not a failure.

## 0d — RESULT: the estimator loses in every cell, and the line closes

Full run 2026-07-27 (`fit3d-hmr2-rotation` v2, 96 videos, 53 min, 0 errors, det% 1.000 throughout).
LOSO over 8 subjects, 4 cameras pooled, ~8000 paired frames per cell:

| cell | est. LOSO-debiased | est. oracle | keypoint bar | margin |
|---|---|---|---|---|
| squat L | 5.60 | 4.74 | 3.98 | **−1.62** |
| squat R | 4.67 | 4.25 | 3.72 | **−0.95** |
| deadlift L | 5.70 | 5.28 | 2.57 | **−3.13** |
| deadlift R | 4.19 | 3.60 | 3.23 | **−0.96** |

**Even the oracle loses in all four**, so this is not a calibration problem — no per-subject offset
rescues it. `pearson r = 0.685–0.813`: HMR2.0 genuinely tracks the rotation, it is simply noisier than
what a keypoint skeleton already predicts. Per-camera spread is ~20% (squat L: 5.01 best camera to
6.06 worst; deadlift L: 5.15 to 6.27), so rotation estimation *is* view-dependent — but even the best
camera in every cell still loses to the bar. Per-subject on deadlift L, only **1 of 8** subjects (s07,
1.74°) beats the 2.57° bar.

**Conclusion.** For the fault-relevant quantity — femoral axial rotation — a monocular SMPL rotation
estimate carries *less* usable information than the joint centres it is derived alongside. An explicit
rotation channel does not earn its place here. The proposed "keypoints have a rotation blind spot"
research line is closed by measurement rather than by argument.

**What this does NOT establish.** Only HMR2.0 was tested. Multi-HMR is natively SMPL-X and would avoid
the SMPL-vs-SMPLX parameterisation mismatch entirely; debiasing removes the *constant* part of that
mismatch but a pose-dependent component could still inflate these numbers. With margins from −0.95 to
−3.13, a better model could plausibly close the smallest gap but not the largest. A native-SMPL-X model
is the one remaining check before the line is closed for *all* monocular models rather than for HMR2.0.
The GT rotation itself remains informative (0a: within-rep sd 5.78°) — the failure is in *estimating*
it, not in the quantity being meaningless.

**Methodological lesson worth keeping.** The single-video smoke preview read 2.58° against a 2.57° bar
— essentially a tie. The real number for that same cell is **5.70°, 2.2× worse**. Three effects
compounded: the preview subject (s03, 3.84°) is the easiest of the eight, its camera easier still, and
with one subject the LOSO debias degenerates to the oracle. Single-sequence previews with oracle
calibration can be off by more than a factor of two; `TwistAgreement.loso_is_degenerate` now flags the
third of those automatically.

## How 0d was built

No saved Fit3D prediction contained rotations: all six models under `data/Fit3D/derived/preds/` store
only `smpl3d` joint centres. Everything needed to close the loop is now in place:

| piece | where | state |
|---|---|---|
| rotation kernel | `.kaggle_tmp/fit3d_hmr2_rot/fit3d-hmr2-rotation.py` | run, 96/96 npz, 0 errors |
| eval harness | `scripts/fit3d/run_rotation_estimate_eval.py` | ran all 4 cells |
| agreement metrics | `src.fit3d.axial_rotation.compare_twist` | 36 unit tests pass |
| results | `data/Fit3D/derived/rotation_estimate_{squat,deadlift}_{L,R}.json` | written |

### Smoke run (2026-07-26, `s03__deadlift__50591643`, 7 min)

```
body_pose (1563, 23, 3, 3)  finite%=1.000
rotmat check: max|R R^T - I| = 4.17e-07   det in [1.0000, 1.0000]
L_Hip total rotation: mean 27.9 deg, sd 23.1, range 0.7..67.2      <- not frozen
SMPL rest joints (24,3) | L femur 0.3768 m  (SMPLX: 0.383 m, 1.7% apart)
det% 1.000 | 313 frames in 37 s (117 ms/frame) | full-run estimate 0.8 h
```

**Single-video preview — the answer sits exactly on the bar.** deadlift / L hip, 313 frames:
`pearson r = +0.950`, `bias = -5.00 deg` (the predicted SMPL-vs-SMPLX parameterisation offset,
clean and constant), debiased MAE **2.58 deg vs a bar of 2.57**.

Read that as *optimistic*, for three compounding reasons: (a) with one subject the LOSO offset has
no other subjects to fit on, so it degenerates to the **oracle** — `TwistAgreement.loso_is_degenerate`
now flags this and the CLI prints a warning; (b) deadlift L is the cell where keypoints are
*strongest* (lowest bar of the four); (c) one camera, one subject. The honest LOSO number over 8
subjects will be worse. But `r = 0.95` says the signal is genuinely strong, so the full run is the
only thing that can settle it.

The kernel is the pulled `fit3d-hmr2-extract` with only these changes: capture
`out['pred_smpl_params']['body_pose']` (F, 23, 3, 3) and `betas`; export SMPL's rest joints as
`rest_j_smpl`; `SUBSAMPLE=5` to match this experiment's frame grid (the original was 121 min for 64
videos at every frame; the model now runs on 1/5 of frames but decoding still reads all of them, so
expect roughly 40-60 min for all 96, decode-bound rather than the naive 5x); smoke-mode rotation-validity checks. Install, SMPL loading and detection are byte-identical.

**Why `body_pose` is clean here.** HMR2.0's `global_orient` is crop-frame, the caveat that forced
rotation-robust metrics throughout the depth work. `body_pose` entries are rotations **relative to the
parent joint**, so a rotated crop is absorbed entirely by `global_orient`. This comparison is therefore
cleaner than the depth ones, not dirtier.

**Parameterisation caveat.** HMR2.0 predicts SMPL, the GT is SMPLX. Body joints 0..20 correspond, but
rest femur directions differ, so each side's twist is taken about **its own** rest axis
(`hip_twist_series(..., rest=...)`); any residual constant offset is what `mae_debiased` removes.

Reporting is three-valued like `decision_eval`: raw / LOSO-debiased / per-subject-oracle. **Compare
`mae_debiased` against the bar** — it strips the SMPL-vs-SMPLX offset without leaking subject-specific
information. A per-camera breakdown comes free and measures the view-dependence of rotation estimation.

```
.venv\Scripts\kaggle.exe kernels push -p .kaggle_tmp/fit3d_hmr2_rot     # SMOKE=True first
.venv\Scripts\kaggle.exe kernels output haoping6028/fit3d-hmr2-rotation -p .kaggle_tmp/fit3d_hmr2_rot_out
python scripts/fit3d/run_rotation_estimate_eval.py \
    --pred-dir .kaggle_tmp/fit3d_hmr2_rot_out/fit3d_hmr2_rot --action squat --side L
```

## Caveats (honest)

- **n = 8 subjects.** Per-subject `R2_within` ranges −0.20 to +0.84 on the same model. Any per-subject
  pattern here is suggestive at best — the same underpowering flagged throughout the REHAB24 LOSO work.
- **No clean scaling of error with rotation magnitude.** s05/s08 (highest within-rep sd) do have the
  highest MAE (~6.0–6.5°), but s09 has the third-highest sd and the *second-best* MAE (2.96°). s07's
  negative R² is largely a small-denominator effect (lowest sd 4.90, mid-pack MAE 4.39). Do **not** claim
  "it fails where rotation is largest".
- **Target is not noise-limited.** `joints3d_25` limb segments are rigid to ~1e-5 m across frames
  (hip→knee sd 1.4e-6 m), so it is regressed from the same SMPLX fit as the target. The residual is
  information loss, not fit noise.
- **Model class.** Ridge + RBF kernel-ridge. Velocity/acceleration and a ±6-frame window were tested
  (0f) and did not help, so the bar is not a weak-baseline artefact — but a properly trained sequence
  model (TCN/transformer) was not tried and could still lower it.
- **Sign convention unverified.** Left and right come out mirrored for the same anatomical direction. The
  sign is consistent within a side, but "positive = internal rotation" has **not** been checked.
- **Anatomical approximation.** Twist is taken about the hip-centre→knee-centre line, the standard
  proxy for the femoral mechanical axis, not a true anatomical axis.
- Frames subsampled every 5th; squat n=2169, deadlift n=2018, lunge n=1687.

## Repro

```
python scripts/fit3d/run_axial_rotation.py --action squat --side L \
    --json data/Fit3D/derived/axial_rotation_squat_L.json
python scripts/fit3d/run_axial_rotation.py --action squat --side L --frame gt
```
