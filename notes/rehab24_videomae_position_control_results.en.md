# Does VideoMAE read the movement, or just *when in the recording* the rep happened?

*REHAB24-6 · position control · 2026-09-06 · English companion to
[`rehab24_videomae_position_control_results.md`](rehab24_videomae_position_control_results.md)*

---

## The short version

We deleted the part of the video representation that encodes *when in the recording a
repetition happened*, retrained, and the model's ability to tell good reps from bad ones
inside a single recording barely moved: **0.8741 → 0.8556** (within-session ROC-AUC,
9 of 9 subjects still above chance). The shortcut we were most afraid of accounts for at
most **4.9%** of the signal.

Two caveats travel with that sentence, and they are not decoration:

- Deleting **one** direction was not enough. It took **sixteen** before the information
  was actually gone. The pre-registered primary analysis assumed one, so it failed its
  own positive control and cannot be read. What we report is the registered *secondary*
  analysis promoted to primary — an explicit plan deviation.
- The deletion is **linear**. A network that encodes position non-linearly would sail
  straight through this control untouched.

---

## Why this experiment had to happen

An earlier control ([identity/appearance](rehab24_videomae_identity_appearance_results.md))
established that VideoMAE can rank reps *within a single recording*: sample one correct
and one incorrect rep from the same video, and the model scores the correct one higher
87.4% of the time. Because everything constant across a recording — the person, their
clothes, the room, the camera — contributes exactly zero to a within-session comparison,
that result killed the "it's just recognising the athlete" story.

It did not kill a smaller, nastier story. In REHAB24-6 the correct reps tend to come
first and the incorrect ones later. How strongly? Take the integer `repetition_number`,
zero fitted parameters, one threshold: as a leave-one-subject-out classifier it reaches
**0.7309 ± 0.0481** balanced accuracy, 9 of 9 subjects above chance. The entire VideoMAE
pipeline reaches 0.6612 on the same protocol. A one-integer rule beats the headline.

So "the signal comes from the person, not the background" was never the same claim as
"the signal comes from the movement". Anything in the pixels that drifts monotonically
through a recording — the athlete edging sideways, drifting toward or away from the
camera, sweating, tiring, the lights warming — is correlated with the label for free.

## What we deleted, and why it isn't the obvious thing

We did **not** remove the direction that predicts `repetition_number`. That variable is
nearly collinear with the label inside a recording, so any feature direction predicting
it also predicts the label. Removing it would drop the AUC even if the model were reading
pure biomechanics, and the drop would be uninterpretable.

Instead we removed **within-class position**: among the correct reps of a recording, is
this the 1st or the 9th? Among the incorrect ones, likewise. Normalised to [0, 1], one
value per rep, shared by both camera views. This holds the label fixed by construction,
so a direction that predicts it is a *drift* direction, first-order orthogonal to the
label. Removing it removes drift, not signal.

The mechanics, per LOSO fold:

1. Fit ridge regression from features to within-class position, **training subjects only**.
   λ chosen by an inner leave-one-training-subject-out sweep over
   {0.1, 1, 10, 100, 1000, 10000}, fixed before any result was looked at (folds picked
   100–10000).
2. Normalise β to unit length, project it out of *every* sample: `x' = x − (x·β̂)β̂`.
3. For k > 1, refit on the residual and repeat.
4. Hand `x'` to the untouched training recipe.

The held-out subject never touches the fit. All 10 folds produced pairwise distinct β̂
hashes (10 for k=1, 40 for k=4, 160 for k=16), which is the check that would catch a
leak. And with k = 0 the pipeline reproduces the earlier out-of-fold probabilities
**byte for byte** across all three seeds, so the plumbing is provably the same plumbing.

## What we found

### 1. The information really is in there — and it is wide

A linear probe trained on other subjects predicts within-class position on the held-out
subject with a subject-macro Spearman ρ of **+0.43** (median; all 9 subjects positive,
+0.17 to +0.69). The permutation null sits in [−0.086, +0.085]. This door was wide open.

The surprise is how wide. Deleting directions one at a time:

| directions removed | probe ρ (median) | probe inside null? |
| ---: | ---: | --- |
| 0 | +0.43 | no |
| 1 | +0.26 | no |
| 4 | +0.29 | no |
| 16 | **+0.05** | **yes** (p = 0.24) |

Position is not a line in feature space. It is a subspace, and the pre-registered
assumption that k = 1 would clear it was simply wrong.

### 2. Removing it costs almost nothing

| arm | within-session AUC (mean ± SD, n = 9) | subjects > 0.5 | bootstrap 95% CI |
| --- | ---: | ---: | --- |
| k = 0 (untouched) | 0.8741 ± 0.0408 | 9/9 | [0.850, 0.899] |
| k = 1 (registered primary — **not readable**) | 0.8720 ± 0.0439 | 9/9 | [0.845, 0.899] |
| k = 4 | 0.8681 ± 0.0436 | 9/9 | [0.843, 0.896] |
| **k = 16 (only arm passing its control)** | **0.8556 ± 0.0435** | 9/9 | [0.830, 0.883] |
| k = 1, naive (raw position) | 0.8696 ± 0.0407 | 9/9 | [0.845, 0.895] |

Every arm has p = 1/10001 against a within-session label permutation null (10,000 draws,
seed 20260906). No subject falls below 0.80 after the k = 16 deletion. Expressed as a
fraction of the signal above chance:

```
share along the linear position subspace = (0.8741 − 0.8556) / (0.8741 − 0.5) = 0.049
```

Downstream, leave-one-subject-out balanced accuracy is untouched in every direction:
0.6612 → 0.6717 at k = 16 (+0.0105, 6 of 9 subjects positive, exact Wilcoxon p = 0.36).
All five paired deltas are **undetermined** — with n = 9 and SD ≈ 0.05 this test cannot
resolve effects this small, which is a statement about power, not about equality.

### 3. One recording, walked through

Most REHAB24-6 recordings look like `11111000001111100000` — five good, five bad, five
good, five bad. There, "earlier is better" scores 0.75 by itself. To see whether the
model leans on that, find a recording where position is useless.

P3's `Ex4_PM_027` is one: 27 reps, labels `000001101111000001111100000`, seven alternating
blocks, and the position rule scores **0.494** — pure chance. On that same recording the
model scores **0.854** before the intervention and **0.803** after removing 16 position
directions. Position carries nothing there; the model still ranks correctly; and the
0.05 it loses is the part that happened to align with drift.

The counter-example matters too. P1's `Ex2_PM_003` (`1111011111000000`) is the opposite:
position scores 0.921 and the model only 0.302. Where the labels split cleanly, the model
can be far *worse* than the trivial rule — which is exactly why the 27 single-boundary
recordings are reported but never used for inference.

### 4. What the drift looks like in pixels

Zero-parameter proxies, same within-session statistic, two-sided p:

| proxy | AUC | \|AUC − 0.5\| | direction consistent | p (raw / Holm) |
| --- | ---: | ---: | ---: | --- |
| person-box left edge `x0` | 0.6465 | 0.1465 | 8/9 | 1/10001 / 0.0004 |
| person-box area | 0.3743 | 0.1257 | 7/9 (reversed) | 1/10001 / 0.0004 |
| person-box top edge `y0` | 0.5177 | 0.0177 | 6/9 | 0.41 / 0.81 |
| mean luminance per rep | 0.5069 | 0.0069 | 6/9 | 0.74 / 0.81 |

Within a recording the athlete drifts sideways and shrinks (moves away from the camera)
as the session goes on, and both are correlated with the label. This is visible, real,
and the most plausible physical carrier of the subspace we deleted. Neither clears the
pre-registered 0.15 margin, so we name no carrier. Lighting is not it.

No contradiction with the framing experiment's "box geometry alone scores 0.5075": that
was cross-subject balanced accuracy, this is within-recording ranking. Box geometry
doesn't generalise across people, but it drifts within one recording.

---

## Where this leaves the claims

**Survives.** "The within-session signal comes from the person in the frame, not from
the recording position" — with the bound that linear position accounts for ≤ 4.9%.

**Broken.** The pre-registered primary. Its positive control failed, and per the frozen
protocol a failed control means the number is not interpretable, however good it looks.
We read k = 16 instead. It is the *lowest*-scoring arm of the four, not a cherry-picked
one, but it is still a post-hoc choice and every conclusion above inherits that label.

**Still open, in order of how much it should worry you:**

1. **Non-linear position encoding.** A projection removes a linear subspace. The
   classifier is an MLP. Nothing here constrains what it could reconstruct non-linearly.
2. **Two subjects were never cleaned.** The k = 16 probe *mean* (+0.12) remains outside
   its null (p = 0.0016), driven by P4 (+0.32) and P8 (+0.38). So 4.9% is a floor.
3. **Drift that jumps at the label boundary** — a coach stopping to adjust the lights
   between the good and bad blocks — is perfectly collinear with the label and invisible
   to this design. In the 27 single-boundary recordings *no* analysis can separate it.
   That is a property of the data collection, and only re-recording fixes it.
4. **Within-clip temporal order** (16-frame shuffle) is still untested. Different question,
   still unanswered.

**Consequence for the next experiment.** The fusion pre-registration may open, under the
deviation label, with the terms already agreed: calibrated late fusion of NLF and
`full_frame_letterbox` only, validation-only calibration, one fusion rule, subject as the
pairing unit, stop if it doesn't beat NLF. Carry the position rule (0.7309) along as the
zero-parameter baseline that any claim has to clear.

---

## Reproduction

Code: `src/rehab24/videomae_position_control.py`, CLI `scripts/rehab24/videomae_position_control.py`,
44 synthetic-data tests in `tests/test_rehab24_videomae_position_control.py`.
`videomae_stage_a.run_arm` gained optional `feature_transform_factory` / `materialize_root`;
the default path is unchanged, and `predict --k 0` reproducing the earlier OOF byte for
byte is the proof.

```powershell
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py replicate-exploratory --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict  --k 0 --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze  --k 0 --permutations 10000 --permutation-seed 20260906   # must print 0.8741
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe    --k 0 --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict  --k 1 4 16 --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe    --k 16 --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze  --k 16 --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py drift-proxies --permutations 10000 --luminance-backend ffmpeg
```

CPU only; roughly 20–40 minutes per arm for three seeds, plus ~80 MB of materialised
features per arm. Artifacts land in `data/REHAB24-6/processed/videomae_position_control/`:
per-arm OOF CSVs and fold JSONs, `fold_betas.json` (λ, β̂ hashes, training subjects per
fold), `probe_summary_k*.json`, `within_session_summary_*.json`, `permutation_null_*.npz`,
`drift_proxies.json`, `luminance_per_sample.csv`.

Two operational traps worth knowing. The OpenCV luminance decode runs at ~11 fps on these
1080p 4:4:4 files — five hours for 130 videos, and the first attempt was killed by the OS
at 1h45 with nothing saved; the ffmpeg backend (now the default) does it in ~15 minutes
and caches per video. And the permutation null's *mean* depends on the order sessions are
iterated, because one RNG draw is consumed per session: 0.5002 via the exploratory
script's order, 0.4999 via the module's sorted order. Same observed statistic, same p.
