# EgoExo silent-rule separability: pre-registration (2026-09-26)

**Written before the full `frames_open` archive was decoded.** On 2026-09-26, 19 of 21 parts
were on disk and `.an` and `.au` (~6 GB) were still downloading. No frame of either movement had
been decoded beyond the 11 JJ and 6 HK actions the earlier prefix runs used. Everything below is
fixed now. Anything the run does differently goes in the result note as a deviation.

## Question

Two registered-silent rules are silent because their **number** fails, not their metric:
- `jj_incomplete_leg_rom`: the spec's 1.3 cut fires on 79% of the reps humans judged correct.
- `hk_insufficient_knee_lift`: the implemented 90° cut fires on every rep; the cited 45° cut sorted
  six actions backwards.

Both docstrings name the full EgoExo-Fitness archive as the upgrade path. The question is whether
each rule's own metric ranks the human-flagged actions on the fault side of the unflagged ones,
at full n and with participants held apart.

## Scope (user decisions, 2026-09-26)

- **In:** the two rules above.
- **Report-only.** No rule, cut, registration or pinned test changes on this run. A result that
  supports un-silencing goes to the user as a proposal.
- **Out, with the reason:**
  - **Every withdrawn rule.** Each still carries a citation failure (withdrawn-rules re-search,
    2026-09-25), and data clears only measured blockers.
  - **Silent rules blocked by a missing landmark or source, not data:** `rule_scapular_winging`,
    `rule_hip_flexor_dominance`, `rule_lumbar_hyperextension`, `jj_incomplete_arm_rom` and
    `rule_insufficient_rotation_rom`.
  - **Shoulder Bridge validation of its live rule.** Offered, and not chosen.

## Labels, fixed from annotations only (no pixels read)

| | Jumping Jacks | High Knee |
|---|---|---|
| label | checklist: "Perform the jump by opening and closing your feet." | free-text comment rule `classify_knee_lift_comment` (pre-registered 2026-08) |
| primary positive | strict-majority FALSE (`tkv.json` `fault`): **12 / 121** | comment positive, held-out (6 disclosed actions excluded): **12 pos / 47 neg** (3 unattributable excluded) |
| participants (pos / all) | 8 / 30 | 9 / 25 |
| reliability | α 0.41 (71 multi-annotated), just above the 0.4 bar; **10 of 12 positives had one annotator** | secondary evidence tier; a comment judges a whole action |
| secondary label | any annotator FALSE | all 68 (disclosed included): 15 pos / 48 neg |

## JJ label threats, found from the labels before any pixels

**What the 12 positives complain about.** The criterion reads "opening **and closing**", and
widest stance sees only the first half. Every comment on every action with a FALSE vote was read
and categorised on 2026-09-26. The table is fixed in
`src/egoexo/jumping_jacks_validation.py` `FOOT_SPLIT_COMMENT_CATEGORY`.

| category | majority positives (12) | any-false positives (17) | what widest stance does with it |
|---|---|---|---|
| width: feet not spread far enough | 2 | 2 | the fault the rule models |
| pattern: front-back scissor or crossed feet | 3 | 7 | a different movement, which a frontal width ratio also reads as narrow |
| closing: feet did not come back together | 1 | 2 | invisible |
| ambiguous: "not opening and closing", no direction | 3 | 3 | unknown |
| other: start position, arms, smoothness, "Movement error" | 3 | 3 | unrelated |

**Only 2 of the 12 are unambiguously this rule's fault.** A primary "separates" can therefore
be carried by scissor jumps, which is a real fault but not incomplete range. The coaching message
"spread your feet wider" would be half-right for them at best.

**Annotator count shapes the label.** Under strict majority, one FALSE vote makes a
single-annotator action positive, while a two-annotator action needs both votes. The rates are 10
of 50 single-annotated actions positive (20%) against 2 of 71 multi-annotated ones (2.8%). If
annotation batches track participants or dates, a separation can be batch rather than stance.

**Pre-registered consequences:**
- Three robustness analyses run next to the primary. In each, dropped positives are removed, not
  relabelled negative:
  - (a) scissor/crossed-feet positives dropped (9 positives remain);
  - (b) width and ambiguous positives only (5 remain);
  - (c) the single-annotator stratum only (10 positives against 40 negatives).
- **A JJ un-silencing proposal requires all of the following:**
  - the primary separates;
  - the any-false label's point estimate is above 0.5;
  - the point estimates of (a), (b) and (c) are all above 0.5.
- If the primary separates but (a) or (b) does not, the result is recorded as **"the metric
  detects the scissor pattern"**, not as validation of the width cut.

## Data pipeline

1. **Integrity:**
   - All 21 parts present at the Hugging Face listing sizes: 20 × 3,221,225,472 bytes plus
     `.au` at 2,895,783,140.
   - All parts come from one HF commit (2025-02-18), so there is no mixed-revision risk.
   - `gzip -t` over the concatenation must pass. It checks the CRC, which the streaming tar reader
     does not.
2. **Extraction:** one pass, both movements, the three exo views.

   ```
   extract_action_frames.py --movement "Jumping Jacks" --movement "High Knee" --require-complete
   ```

   A stream error, a part after a gap, or any (action, view) pair short of its window by more
   than one frame fails the run.
3. **Pose:** `run_pose_on_frame_dirs.py`, MediaPipe `model_complexity=2`,
   `static_image_mode=False`, unchanged from the 2026-08 runs.
4. **Harnesses, unchanged:** `run_jumping_jacks_validation.py --json` and
   `run_high_knee_validation.py --json`. Metrics are read through the real `run_detector` on
   segmented windows.
5. **Analysis:** `run_silent_rule_separability.py`. Pose runs once per movement subdirectory
   (`<out>/jumping_jacks`, `<out>/high_knee`). The pose runner pointed at the parent would find
   two directories with no jpgs and write nothing.

## Metrics and aggregation

- **JJ score:**
  - Per camera: the median, over scored reps, of each rep's widest `stance_width_ratio`.
  - Per action: the median over the exo cameras that scored at least one rep. All three cameras
    count, because the ratio is roll-, mirror- and scale-invariant and obliquity-cancelling.
  - Lower means fault.
- **HK score:**
  - Per camera: the median per-rep peak `thigh_elevation`.
  - Per action: the median over the gated side cameras `exo_l` and `exo_r`.
  - Lower means fault.
  - The runner prints both camera groups' `anterior_axis_length` ranges. If they overlap at
    full n, that is reported as a threat, not silently re-gated.
- **Exclusions:** actions with no scored rep on any usable camera are excluded, and counted by
  label class.

## Statistics

- **AUC:** the Mann-Whitney probability, ties counted half, oriented to the fault side.
- **95% CI:** 2,000 bootstrap resamples of **participants** (the name part of `original_actor`),
  seed 20260926. Resamples with an empty class are skipped and counted.
- **Verdict words:** lower bound > 0.5 → **separates**; upper bound < 0.5 → **sorted
  backwards**; otherwise **undetermined**. "No difference" is never written.
- **Primary analyses:** JJ widest stance vs majority label; HK peak elevation vs held-out comment
  label.
- **Secondary analyses:** JJ vs any-false; HK on all 68 actions; and the spec cuts' own per-rep
  fire rates as action scores (JJ 1.3, HK cited 45°).
- **JJ candidate cut, reported and never applied:**
  - A Youden cut fitted without each participant, applied to that participant, pooled.
  - Report held-out sensitivity and specificity, plus the all-data cut.
  - Ties in J break toward firing less.
- **HK:** no cut is proposed. The comment rule's docstring already records that an action-level
  comment cannot license a threshold.

## What each outcome licenses

- **JJ separates and every robustness condition above holds:** a proposal to the user covering
  the candidate cut, its held-out sensitivity and specificity, and the pinned
  `test_the_specs_cut_is_kept_where_it_is_rather_than_moved` that a change would have to amend
  with its reason. The movement stays unregistered either way: one rule would be its only live
  rule.
- **HK separates:** the metric sorts the one human signal available. It is still no licence for a
  number.
- **Undetermined or sorted backwards:** the rule stays silent. The docstrings' "one download
  away" upgrade path is then recorded as tried.

## Threats known now

- **Power.** 12 positives from 8 (JJ) and 9 (HK) people. A CI half-width near 0.15 is expected,
  so undetermined is a likely outcome.
- **Resolution.** EgoExo ships preprocessed 456×256 frames, about 2.8× production's landmark
  error in normalized units.
- **JJ positives are mostly single-annotator (10 of 12).** Most complaints are not about width
  (see "JJ label threats").
- **Criterion laxness.** For negatives the criterion may be laxer than the rule: "did the feet
  open" versus "did they open wide enough". That was the 79% false-fire reading of 2026-08.
- **Comment labels are secondary.** A missing complaint is not a judgement that the knee lift
  was adequate.
