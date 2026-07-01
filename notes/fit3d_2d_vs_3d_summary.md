# Fit3D 2D-vs-3D-vs-mocap — is the bottleneck the 2D *detector* or the *projection*?

**Question.** The project's thesis is "depth is the bottleneck for squat coaching, not 2D
accuracy." Two prior arguments were *indirect*: a stronger 2D backbone (HRNet) didn't beat
RTMPose on REHAB24 correctness, and 2D->3D lifting couldn't recover depth. This experiment makes
it **direct** on mocap GT by decomposing a single-view 2D pipeline's cue error into its two
components:

    real-2D error  =  detector error (real-2D - mocap-2D)  +  projection error (mocap-2D - GT-3D)

* **real-2D**  -- a genuine 2D detector, RTMPose (`rtmlib` Wholebody, RTMW-x-L), on the Fit3D
  videos, mapped into the Fit3D-25 biomech slots (pixels; `src/fit3d/twod_baseline.py`).
* **mocap-2D** -- the GT 3D *projected* to the image = a **perfect** detector (zero detector error).
* **3D**       -- NLF / HMR2.0 / Multi-HMR, via `depth_eval`.

All are cue-reading errors vs the mocap 3D truth (deg / ratio). real-2D and mocap-2D are computed
on the **same** RTMPose-inferred frames, so their difference is *exactly* the detector term.
86/96 videos (RTMPose balanced, every-15th frame; per-frame cue error is unbiased under
subsampling). `src/fit3d/twod_vs_threed.py`, `scripts/fit3d/run_twod_vs_threed.py`.

## Results (cue-reading error vs mocap-3D truth)

| action | cue | real-2D | mocap-2D (perfect) | best 3D | **detector** | verdict |
|---|---|---|---|---|---|---|
| squat | **knee** | 17.11 | 17.88 | 7.09 | **−0.77** | need-3D |
| squat | hip | 19.10 | 17.89 | 9.26 | +1.21 | need-3D |
| squat | valgus | 0.07 | 0.04 | — | +0.03 | better-2D |
| deadlift | **knee** | 13.75 | 14.20 | 7.51 | **−0.45** | need-3D |
| deadlift | hip | 25.18 | 17.59 | 7.78 | +7.60 | need-3D |
| deadlift | valgus | 0.08 | 0.04 | — | +0.04 | better-2D |
| thruster | **knee** | 17.52 | 17.30 | 5.91 | **+0.22** | need-3D |
| thruster | hip | 12.82 | 14.76 | 8.34 | −1.94 | need-3D |
| thruster | valgus | 0.07 | 0.04 | — | +0.04 | better-2D |

RTMPose keypoint MAE vs GT-projected: 55–59 px on ~900 px images (the raw detector error).

## Findings

1. **On the depth cue the detector error is ~0 — a real 2D detector is already as good as a
   perfect one, and both fail.** The knee-angle detector term is **−0.77 / −0.45 / +0.22 deg**
   across squat / deadlift / thruster (per-subject −0.7±1.5, −0.4±1.5, +0.2±1.2 over 7–8 subjects,
   centred on zero). So the entire ~14–18° knee error is **projection geometry**, not detector
   accuracy: even a *perfect* 2D detector (mocap-2D) reads the squat-depth cue 14–18° off, and
   RTMPose is already at that ceiling. Only direct 3D (~6–8°) removes it. **This is the direct,
   mocap-GT proof of "depth is the bottleneck, not 2D accuracy"** — the thing HRNet-vs-RTMPose on
   REHAB24 could only hint at indirectly. No amount of 2D-detector improvement can help the depth
   cue; you must change modality to 3D.

2. **Valgus is the mirror image — detector-dominated, so a better 2D detector is exactly the fix.**
   The perfect detector reads valgus well (mocap-2D 0.04) and the *real* detector adds the error
   (real-2D 0.07–0.08; detector +0.03–0.04). So here 2D is the right tool and the lever is detector
   quality — and 3D is **not** the answer (experiment 3 showed 3D is *worse* on valgus). This is the
   clean converse of the depth cue.

3. **The two components split exactly along the sagittal/frontal axis**, matching experiments 2–3:
   sagittal depth/flexion → projection-limited → 3D; frontal-plane valgus → detector-limited → 2D.
   The decomposition turns "route by fault type" into a *causal* statement about *why* — and about
   which lever (better detector vs different modality) moves each cue.

## Caveats

- **torso_lean reads "need-3D" here but that is a raw-MAE artefact of a *calibratable* offset.**
  Experiment 2 found torso view-*robust* (r=0.87) with a constant ~−22° projection offset; that
  offset inflates the raw MAE (mocap-2D 12–14°), so 3D "fixes" it in raw terms — but experiment 3
  showed **calibrated** 2D handles torso as well as or better than 3D. So torso is "calibratable
  projection", unlike the knee's non-calibratable projection *scatter*. The clean, load-bearing
  case is the **knee** (detector ~0, projection is scatter no calibration removes).
- **hip on deadlift** has a large detector term (+7.60): RTMPose's hip is notably off on the
  hip-hinge; but mocap-2D is also large (17.59) so 3D is still the fix.
- RTMPose = balanced (RTMW-x-L, a strong detector) — the detector≈0 result is *not* because the
  detector is weak; a strong real detector already saturates the projection ceiling on depth.
- 86/96 videos (s11 partial); per-action n in the table header. `depth_ratio` is low-signal
  per-frame (its corruption is the per-rep extreme; see experiments 1–2), hence "mixed".

## Implications

Completes the perception story: for the **sagittal depth/flexion** verdicts, a better 2D detector
is a dead end (even a perfect one fails) — use direct image->3D; for **frontal-plane valgus**, keep
2D and invest in the detector. Reproduce: `python scripts/fit3d/run_twod_vs_threed.py --action squat
--rtmpose-root data/Fit3D/derived/preds/rtmpose --model NLF=... --model HMR2=... --model MultiHMR=...`
(per-action JSON `data/Fit3D/derived/twod_vs_threed_<action>.json`; RTMPose npz under
`data/Fit3D/derived/preds/rtmpose/`, extracted by `scripts/fit3d/run_rtmpose_fit3d.py`).
