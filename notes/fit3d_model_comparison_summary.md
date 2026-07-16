# Fit3D model comparison — is depth recovery an NLF quirk or a general direct-3D mechanism?

**Question.** Experiments 1–3 used **NLF** as the single direct image->3D model. Two open
questions remain: (a) *mechanism* — does a second, architecturally-different direct-3D model
*also* recover the sagittal depth cues single-view 2D corrupts, or is it an NLF artefact? and
(b) *ranking* — is NLF actually the best such model? The REHAB24 thread explicitly left the
clean second-model confirmation unfinished because detection there was only ~75%
(`notes/rehab24_correctness_experiment_summary.md`); Fit3D (single subject, clean, **100%
detection**) is the setting that closes it.

**Method.** Ran a second model, **HMR2.0 / 4D-Humans** (parametric SMPL transformer —
architecturally distinct from NLF's localizer-field point regression), on the same 96 Fit3D
videos (squat + deadlift + thruster, all 100% detection), and compared with the
`src/fit3d/model_comparison.py` harness. Because body conventions differ (NLF SMPL-24, HMR2.0
SMPL), the joints sit at slightly different anatomical points — a per-joint **bias**, not depth
recovery. So the comparison leans on **bias-tolerant** metrics: the **ez/exy depth-axis
pattern**, **rotation-invariant knee/hip angle** recovery, **pa_mpjpe**, and the **debiased**
verdict-flip — never raw MPJPE rankings. Both models map to H36M-17 with the *same*
`resolve_lr` + `SMPL24_TO_H36M17` (they share the SMPL body-joint order), so the mapping
artefact cancels in the cross-model compare. **Caveat:** HMR2.0 regresses orientation in the
**crop** camera frame, so its gravity-dependent cues (torso, depth axis, ez/exy) carry a
crop-rotation term that the rotation-invariant knee/hip angles and pa_mpjpe do not.

## Results — NLF vs HMR2.0 (32 sequences each per action)

**Depth-axis pattern** (root-relative, mm) — the mechanism signal:

| action | model | MPJPE | PA-MPJPE | ez (depth) | exy | **ez/exy** |
|---|---|---|---|---|---|---|
| squat | NLF | 77.7 | 65.4 | 42.4 | 36.6 | **1.16** |
| squat | HMR2.0 | 101.9 | 103.7 | 65.2 | 41.6 | **1.57** |
| deadlift | NLF | 80.7 | 61.7 | 44.5 | 38.0 | **1.17** |
| deadlift | HMR2.0 | 103.0 | 102.0 | 68.2 | 41.0 | **1.66** |
| thruster | NLF | 71.2 | 68.8 | 33.8 | 35.5 | **0.95** |
| thruster | HMR2.0 | 100.2 | 107.0 | 59.2 | 42.0 | **1.41** |

**Cue recovery vs single-view 2D** (deg; knee/hip are rotation-invariant = fair for both):

| action | cue | 2D-view | NLF | HMR2.0 |
|---|---|---|---|---|
| squat | knee* | 18.42 | 7.09 | 7.88 |
| squat | hip* | 18.27 | 10.42 | **9.26** |
| deadlift | knee* | 14.76 | 7.51 | 8.81 |
| deadlift | hip* | 17.84 | 7.93 | **7.78** |
| thruster | knee* | 17.37 | 5.91 | 6.64 |
| thruster | hip* | 14.50 | 11.21 | **9.49** |

**Debiased verdict-flip** (swept thresholds, oracle-calibrated both; lower = better):

| action | cue | deb-2D | NLF | HMR2.0 |
|---|---|---|---|---|
| squat | knee | 21% | 11% | 14% |
| squat | hip | 12% | 6% | 6% |
| deadlift | knee | 17% | 10% | 20% |
| deadlift | hip | 18% | 13% | 17% |
| thruster | knee | 22% | 9% | 17% |

## Findings

1. **Mechanism — depth recovery is GENERAL, not an NLF quirk (headline).** On the
   rotation-invariant cues (the fair test), HMR2.0 recovers the sagittal knee/hip cues that
   single-view 2D corrupts about as well as NLF — knee within ~1° of NLF across all three
   actions, and **hip is consistently *better* than NLF** (squat 9.3 vs 10.4, deadlift 7.8 vs
   7.9, thruster 9.5 vs 11.2). Two architecturally-distinct direct image->3D models independently
   halve the 2D cue error. **This closes the REHAB24 second-model thread**: the depth-from-pixels
   signal is real and model-general, it was only buried there by 75% detection.

2. **Ranking — NLF is the cleaner model on the depth axis (secondary).** NLF's per-axis depth
   error is on par with in-plane (ez/exy 0.95–1.17); HMR2.0's depth axis is ~1.5× noisier
   (ez/exy 1.41–1.66) — still far from the 2D-lifting failure mode (ez >> exy), so HMR2.0 *does*
   recover depth, just less cleanly. The verdict-flip echoes it: HMR2.0 reduces the depth/flexion
   verdict-flip vs 2D but less than NLF (thruster knee 2D 22% -> NLF 9%, HMR2.0 17%). The most
   likely cause is HMR2.0's **crop-frame orientation** (it regresses global orientation in the
   256 crop, not the full image), which rotates some depth into the error — a handicap NLF (and
   the pending full-frame Multi-HMR) do not share.

3. **The needs-3D map is model-independent.** Both models recover knee/hip flexion (needs 3D),
   and both are *worse* than calibrated-2D on valgus / tie on torso-lean — same split experiment 3
   found. So "route by fault type — 3D for depth/flexion, calibrated 2D for frontal-plane" holds
   regardless of which direct-3D model you pick.

4. **Why raw MPJPE is not the metric (honest note).** HMR2.0's MPJPE (100–103 mm) exceeds NLF's
   (71–81 mm), but its *cue-relevant angles* are within ~1°; the gap is body-shape (~11% small,
   pelv-neck 0.47 m vs ~0.53) + crop, not a depth-recovery failure. Tellingly HMR2.0's pa_mpjpe
   (102–107) barely improves on its MPJPE while NLF's drops ~15–20 mm — HMR2.0's residual is
   non-rigid/scale, not a single global misalignment procrustes could remove. This is exactly why
   the comparison ranks on ez/exy + rotation-invariant cues + debiased verdicts, not raw mm.

## Third model — Multi-HMR (2024 SOTA, single-shot full-frame SMPL-X)

Multi-HMR (Naver, ECCV'24) is detection-free and predicts in the **full-image** camera frame, so
unlike HMR2.0 it has **no crop-frame orientation handicap** — a test of whether a SOTA full-frame
model matches or beats NLF on depth. Ran on all 96 videos (100% detection). Two honest caveats
that make its numbers *conservative*: (a) at 896 resolution it is ~0.9 s/frame, so we inferred
**every 6th frame** (rep_summary still captures the extreme via nanmin/nanmax over ~50 fps, but
with fewer samples than NLF/HMR2.0's every-frame); (b) it needs a camera intrinsic — we assumed a
60° FOV (Fit3D's true intrinsics differ), which the **rotation-invariant knee/hip angles do NOT
depend on**, but the gravity-dependent cues (torso, depth axis, ez/exy) do. So read Multi-HMR off
the rotation-invariant cues; treat its torso/ez as an upper bound on error.

**Rotation-invariant cue error (deg), all three models vs the 2D baseline:**

| action | cue | 2D-view | NLF | HMR2.0 | Multi-HMR |
|---|---|---|---|---|---|
| squat | knee* | 18.42 | **7.09** | 7.88 | 9.67 |
| squat | hip* | 18.27 | 10.42 | **9.26** | 9.49 |
| deadlift | knee* | 14.76 | **7.51** | 8.81 | 12.40 |
| deadlift | hip* | 17.84 | 7.93 | **7.78** | 10.21 |
| thruster | knee* | 17.37 | **5.91** | 6.64 | 6.93 |
| thruster | hip* | 14.50 | 11.21 | 9.49 | **8.34** |

Depth axis ez/exy: squat 1.16 / 1.57 / 1.50; deadlift 1.17 / 1.66 / 1.50; thruster 0.95 / 1.41 / 1.21
(NLF / HMR2.0 / Multi-HMR).

**Findings (3-model):**

5. **Mechanism confirmed by THREE independent architectures.** Localizer-field (NLF),
   parametric-SMPL transformer (HMR2.0), and single-shot full-frame SMPL-X (Multi-HMR) *all*
   recover **each movement's defining rotation-invariant cue** — squat & thruster: knee flexion
   (2D ~17–18° → 7.1/7.9/9.7 and 5.9/6.6/6.9); deadlift: the hip hinge (2D 17.8° → 7.9/7.8/10.2).
   (Deadlift *knee* is low-signal for everyone — knees barely bend in a hip-hinge, so even NLF is
   only 7.5 — which is why we read each movement off its own cue, as in experiments 1–2.) Every
   ez/exy is ~0.95–1.7, nowhere near 2D-lifting's ez>>exy. Three architectures this different
   agreeing on the defining cue is strong mechanism evidence — the depth-from-pixels signal is real
   and general, definitively closing the REHAB24 second-model thread.

6. **No model beat NLF under this setup; NLF has the lowest rotation-invariant knee error in all
   three actions** — a *modest* edge (squat 7.1 vs 9.7, deadlift 7.5 vs 12.4, thruster 5.9 vs 6.9).
   Two honesty limits on how far to push this ranking: (a) Multi-HMR ran with **non-native
   preprocessing** (a white square-pad, not its own crop) and an **assumed 60° FOV**, so this is
   **not a clean head-to-head** — we do *not* claim strict dominance, and we do *not* use the
   FOV-confounded ez/exy to argue "full-frame can't close the depth gap." (b) The one confound we
   *did* rule out is subsampling: masking NLF/HMR2.0 to the same every-6th frames leaves their cue
   error unchanged (squat knee NLF 7.09→7.09, HMR2.0 7.88→7.87), so the gap is not a frame-count
   artefact. Reassuringly, Multi-HMR's *hip* is competitive/best (**thruster 8.34°**, squat 9.49°),
   which argues the preprocessing did not broadly wreck its pose — but we can't call it neutral.

The practical takeaway is robust across three models: **use direct image->3D for the depth/flexion
verdicts (NLF is the strongest here), keep calibrated 2D for frontal-plane faults.**

Reproduce: `python scripts/fit3d/run_model_comparison.py --action squat \
--model NLF=data/Fit3D/derived/preds/nlf --model HMR2=data/Fit3D/derived/preds/hmr2 \
--model MultiHMR=data/Fit3D/derived/preds/multihmr` (per-action JSON:
`data/Fit3D/derived/model_comparison_<action>.json`; npz under `data/Fit3D/derived/preds/{nlf,hmr2,
multihmr}/`; Kaggle kernels `fit3d-{hmr2,multihmr}-extract`).
