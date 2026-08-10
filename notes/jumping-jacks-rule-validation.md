# Jumping Jacks rule validation against EgoExo-Fitness

Design spec: `docs/superpowers/specs/2026-08-10-jumping-jacks-detector-design.md`. Module:
`src/pose/movements/jumping_jacks.py`. Harness: `src/egoexo/jumping_jacks_validation.py`, runner
`scripts/egoexo/run_jumping_jacks_validation.py`, pure helpers tested in
`tests/test_jumping_jacks_validation.py`.

**This is the second movement in the 16-movement programme where the labeled data ran during
DESIGN and changed the rule roster, and the first where it emptied it.** Leg Abduction
(`notes/leg-abduction-rule-validation.md`) silenced one rule of four. Here the check silenced the
rule that had the most going for it, withdrew a second on a zero-parameter control, and left the
detector with no live rule at all — so `src/pose/movements/jumping_jacks.py` deliberately does not
register, and the app keeps telling Jumping Jacks users "coming soon".

**No threshold was changed in response to anything below.** The parent spec's 1.3 stays 1.3 in the
module and the rule is silenced instead; the withdrawn valgus cut stays where it lives, in
`squat.rule_knees_inward`.

Reproduce — the pose corpus is NOT in the repository and cannot be committed, so the recipe is
part of the record:

```
# 1. The archive. data/EgoExo-Fitness/frames_open is a 3 GiB-split download whose `.ac` part is
#    MISSING, so the concatenation cannot be decompressed. `.aa`+`.ab` IS a contiguous gzip
#    prefix and decodes until it runs out. Stream it with Python's tarfile in "r|gz" mode and
#    write only the manifest's Jumping Jacks frame ranges for exo_l / exo_m / exo_r.
#    (Sit-up's pass used `.aa` alone and reached 3 complete records; `.aa`+`.ab` reaches 6.)
# 2. MediaPipe Pose, model_complexity=2, one JSON per (action, view) in the schema
#    src/pose/process_videos.py writes, named {sample_id}__{view}.json.
# 3. Replay:
.venv\Scripts\python.exe scripts/egoexo/run_jumping_jacks_validation.py ^
  --pose-dir <that dir> --json <out.json>
```

Step 1 takes ~15 min (6.4 GiB of gzip), step 2 ~40 min for 9 601 frames on this machine
(shardable — the script takes `shard`/`shards` arguments), step 3 seconds.

---

## 1. What this measures — and what it cannot

**EgoExo-Fitness judges eight criteria per action, and the parent spec writes five rules. Exactly
one pair overlaps** (`jj_incomplete_leg_rom` ↔ "Perform the jump by opening and closing your
feet"). Design spec §2 has both taxonomies side by side.

**And all 11 reachable actions are judged TRUE on that one criterion.** The only faults in the
reachable set are three "Keep your arms tense" failures in `yT4RK3`, which no rule models. So this
corpus has **no positive class**: every firing is a false positive by the corpus's own judgement,
and no rule's sensitivity can be measured at all.

Three inferential steps sit under every number below and are stated rather than buried:

1. **Resolution.** EgoExo distributes *preprocessed* frames — its README says so, and "currently
   the raw videos are not available" — at **456 × 256**. Production is phone video at 720p+, so
   normalized landmark error here is roughly **2.8× production's**. That inflates variance; it
   does not move a median by 12% (§3).
2. **Obliquity.** Not a confound for these metrics, and that is by construction rather than by
   luck: `stance_width_ratio` and the withdrawn knee ratio are both ratios of two **frontal-plane
   widths**, which an azimuthally oblique camera compresses by nearly the same factor, so the
   factor cancels to first order. This is exactly why the module does not use the parent spec's
   image-x form.
3. **Judgement strictness.** An annotator asked whether the feet "open and close" may be answering
   *did they open at all*, not *did they open wide enough*. Stated in §3 as the alternative
   reading it is.

---

## 2. Corpus

| | |
|---|---|
| judged Jumping Jacks actions in EgoExo-Fitness | 121 (195 annotations) — the dataset's largest judged class |
| reachable from the truncated archive | **11** (10 with three exo cameras, `wNsRwL_action_9` with `exo_r` only) |
| (action, camera) pairs | **31** |
| frames | 9 601 |
| judged CORRECT on the foot-split criterion | **11 of 11** |

Records recovered by `.aa`+`.ab`: `zT0YQO`, `zOfbr6`, `z8RAua`, `yT4RK3`, `Y1t9Ew`, `xYkvB0`
complete, `wNsRwL` partial.

## 3. Pipeline properties — everything except the thresholds works

| | |
|---|---|
| median validity rate (8-landmark gate) | **1.000** |
| pairs on the whole-clip fallback | **0 of 31** |
| repetitions found | **255** |
| repetitions lost to the 0.4 s duration floor | **0** |
| median cadence | **0.93 Hz** |
| fastest cadence | **1.14 Hz** (0.88 s per repetition) |

**`base.py:55` names this movement as one that "must lower" `min_rep_seconds`, and it does not.**
Re-segmenting all 31 pairs at a 0.15 s floor finds **exactly the same 255 repetitions**. The
fastest performer in the corpus holds 1.14 Hz — 0.88 s per repetition, more than twice the floor —
and even the RAG doc's Guinness record (136 in a minute, 2.27 Hz, 0.44 s) clears it. The knob the
framework reserved for this movement by name is not needed by it; the framework comment is left
alone because it also names High Knee, the ~3 Hz movement it was really written for.

The measurement is deliberately not circular: every window `segment_reps` *returns* is at least
`min_rep_seconds` long by construction, so the shortest returned repetition can never show the
floor biting. Differencing the counts at two floors can (`floor_discarded`).

## 4. `jj_incomplete_leg_rom` — the spec's cut fires on the correct population

| | |
|---|---|
| scored repetitions | **91** |
| fire rate, per repetition, at the spec's 1.3 cut | **79.1%** |
| fire rate, per (action, camera) pair | **90.3%** |
| median widest stance of a repetition | **1.163** shoulder widths |

Every one of those firings is on a repetition belonging to an action a human judged correct on
exactly this criterion. **The correct population sits below the cut**, and the gap is not marginal:
the median performance is 1.163 against a 1.3 threshold.

Per action, the widest stance a repetition reached (median over that action's cameras):

```
Y1t9Ew_action_4    1.69     xYkvB0_action_4   1.29     yT4RK3_action_8   0.93
wNsRwL_action_9    1.66     xYkvB0_action_8   1.22     zOfbr6_action_13  1.57
xYkvB0_action_14   1.20     yT4RK3_action_3   1.15     zOfbr6_action_3   1.24
yT4RK3_action_13   1.10     zOfbr6_action_8   1.54
```

Four of eleven actions clear 1.3 comfortably and seven do not — subject-level variation, not one
outlier dragging a median.

**Outcome: the rule is PERMANENTLY SILENT, and the 1.3 is left where it is.** A cut
fitted to this distribution could be manufactured at will, and manufacturing one is what this
programme forbids. `abd_insufficient_rom` was silenced rather than moved for the same reason.

**The upgrade path is concrete, which is new for a silent rule here.** Every earlier one needed a
paper nobody has written or a per-user baseline this architecture lacks. This one needs neither:
EgoExo judges this exact criterion on 121 actions, **12 of them failed**, so a cut separating the
classes could be read off human judgement rather than authored. What blocks it is the missing
`.ac` archive part, which leaves 11 reachable and all 11 negative. **That is a download, not a
research programme.**

## 5. `jj_knee_valgus_landing` — a zero-parameter control withdraws it

The rule read `knee_width / ankle_width < 0.82`, a threshold transferred from
`squat.rule_knees_inward`. In a squat — feet about shoulder width, shanks near vertical — that
ratio is near 1.0 when the knees track the feet, which is what makes 0.82 meaningful there.

**In a wide side-straddle the legs splay from a pelvis that does not widen, so a knee sits partway
along the hip→ankle line and its separation is necessarily smaller than the ankles' — with no
valgus whatsoever.** The control replaces both knees with their projections onto the same-side
hip→ankle line, i.e. a perfectly straight limb, zero valgus by construction, and recomputes the
ratio:

| over 2 353 open-phase frames | median | below the 0.82 cut |
|---|---|---|
| observed knees | 0.769 | **79.4%** |
| **perfectly aligned knees** | **0.810** | **68.5%** |

**Two marginal rates are not a decomposition**, so the joint counts were taken rather than their
difference — the aligned firings are *not* a clean subset of the observed ones:

| of 2 353 open-phase frames | |
|---|---|
| fires with a **perfectly straight limb** too — stance alone explains it | **63.2%** |
| fires only with the **real** knees — needed genuine inward deviation | **16.2%** |
| the straight limb would condemn and the real knees do not (knees bowed *out*) | 5.2% |

**Four firings in five — 63.2 of the 79.4 points — need no inward deviation whatsoever**, on a
population every action of which a human judged correct. The rule reads the movement, not the
fault.

The 16.2% is not nothing and is not claimed to be: on about one open-phase frame in six there is
measurable inward deviation relative to the straight limb. That is a reason to want a metric that
**isolates** it, not a reason to keep one that cannot.

The same mechanism is pinned on synthetic geometry by
`tests/test_jumping_jacks.py::StanceGeometryConfoundTest`: a perfectly aligned knee trips the 0.82
cut at a 1.6 stance and does not at a 1.0 stance, and the confound is monotone in stance width.

**Outcome: WITHDRAWN, and the metric is removed from the module** so nothing can quietly start
reading it. What would work — the deviation of each knee from its own hip→ankle line, normalized by
limb length, which is zero for a straight limb at *any* stance — is recorded and **not built**: no
source states a threshold for it, and inventing one is what this programme forbids.

## 6. Three simultaneous cameras

The exo rig films the same instant three ways, so disagreement on one action is pure projection
error. Because both rules are silent, the verdicts compared are what the parent spec's cuts *would*
have said.

| | cross-camera verdict | median cross-camera spread |
|---|---|---|
| `jj_incomplete_leg_rom` | 8 unanimous, **2 split** of 10 comparable actions | 0.107 shoulder widths |
| `jj_knee_valgus_landing` | 9 unanimous, **1 split** of 10 | 0.067 |

Both spreads are small relative to the distance between the correct population and the cut (1.163
vs 1.3 = 0.137; 0.769 vs 0.82 = 0.051), which is the point: **camera placement is not what is
wrong with these rules.** Two actions out of ten would nonetheless have received a different
verdict depending on which camera filmed them.

`wNsRwL_action_9` has a single camera and is reported as *no* agreement verdict rather than as
unanimous — one camera cannot agree with itself.

## 7. What this note does not claim

- **No rule was validated.** There is no positive class (§1), so no sensitivity, AUC or
  fault-level claim appears anywhere above.
- **No threshold moved.** Both cuts are quoted as the spec's and the codebase's own, and both
  rules were silenced or withdrawn rather than retuned.
- **The resolution caveat is a limit on interpretation, not a footnote.** These numbers bound
  behaviour on 456 × 256 footage. The valgus withdrawal does not depend on them — its control is
  a geometric identity that holds at any resolution and is reproduced synthetically in the tests.
