# EgoExo-Fitness E3 — Inter-annotator agreement (human ceiling)

**Goal.** Before training E1 (technical-keypoint verification) and E2 (quality-score
regression), measure how reliably *humans* agree on these labels — agreement bounds the
accuracy any model can meaningfully reach. A model scoring at the human-disagreement floor is
not failing; it has hit the label-noise ceiling.

**Run.** `python scripts/egoexo/compute_agreement.py`
→ `data/EgoExo-Fitness/processed/agreement_report.json`.

**Method.** Krippendorff's alpha (own numpy implementation, unit-tested in
`tests/test_egoexo_agreement.py`), on the multi-annotator subset. TKV = *nominal* alpha over
per-(action, criterion) binary pass/fail votes; quality score = *ordinal* alpha over 1–5
votes. Plus raw pairwise agreement (exact / within-1). Annotators per action vary (1–5), so
Krippendorff — not Fleiss' kappa / classic ICC, which assume fixed raters.

## Coverage
913 judged actions; **only 573 (63%) are multi-annotator** (mostly 2 raters). Annotators per
action: 1→340, 2→548, 3→14, 4→8, 5→3. The 340 single-annotator actions carry no reliability
estimate — their labels are taken on faith.

## Results

| Target | Metric | Value | Read as |
| --- | --- | --- | --- |
| Quality score (E2) | Krippendorff α (ordinal) | **0.38** | low |
| | pairwise exact agreement | 0.31 | humans rarely agree on the *exact* 1–5 |
| | pairwise within-1 | 0.81 | but usually within 1 point |
| TKV (E1) | Krippendorff α (nominal) | **0.40** | weak–moderate |
| | raw pairwise agreement | 0.84 | inflated — faults are only ~17% prevalent |

**Per-criterion reliability** (all 97 criteria have ≥5 multi-annotator units):

| α bucket | # criteria |
| --- | ---: |
| < 0 (worse than chance) | 22 |
| 0–0.2 (poor) | 21 |
| 0.2–0.4 (weak) | 26 |
| 0.4–0.667 (moderate) | 19 |
| ≥ 0.667 (acceptable) | 9 |

Only **28 / 97 are reliable (α ≥ 0.4)** — the learnable label space, listed in
`agreement_report.json → reliability_summary.reliable_criteria`.

**The reliability split is configurational vs subjective/geometric:**
- *Most reliable* (discrete, observable): "Bend your arms to lower your body" (α=1.00),
  "Bend both knees, lowering your body…" (0.75), "Extend your arms straight above your head"
  (0.72), "Cross your feet" (0.62).
- *Worse than chance* (subjective / effort / fine-geometric): "Distribute your weight evenly
  across both legs" (−0.16), "Tense your abdominal muscles for stability" (−0.12),
  **"Keep your back straight" (−0.07, n=92)**.

## Implications
1. **E1 trains/reports on the ~28 reliable criteria** (unreliable ones reported separately).
   Predicting α ≤ 0 criteria is chasing noise — no model beats a negative ceiling.
2. **Re-frames the depth/pose thread.** Geometric criteria aren't only hard for CLIP — humans
   can't label them consistently from video either, so better features (pose / 3D) can't be
   *validated* against these labels; the bottleneck there is label / observability quality.
   This tempers the "pose wins on geometric criteria" hypothesis and suggests a stronger angle:
   show pose/rule-based geometric assessment is *more self-consistent* than the noisy human
   labels, rather than trying to match them.
3. **E2 evaluates ordinally / within-1, not exact.** Human exact agreement is only 31%;
   within-1 is 81%. Report Spearman ρ / within-1, not exact accuracy.
4. **Small per-criterion samples** (~37–92 multi-annotator units, mostly 2 raters) → individual
   α values are noisy; the configurational-vs-geometric *pattern* is robust, the point estimates
   are not.

## Caveat for the paper baseline
The original paper's GEV F1 ≈ 0.52–0.55 now reads differently: with TKV α ≈ 0.40, ~0.55 may
sit near the label-noise ceiling for many criteria rather than indicating a weak model. Compare
against per-criterion human agreement once E1 produces per-criterion numbers.
