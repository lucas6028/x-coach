# High Knee rule validation against EgoExo-Fitness

Design spec: `docs/superpowers/specs/2026-08-10-high-knee-detector-design.md`. Module:
`src/pose/movements/high_knee.py`. Harness: `src/egoexo/high_knee_validation.py`, runner
`scripts/egoexo/run_high_knee_validation.py`, pure helpers tested in
`tests/test_high_knee_validation.py`.

**The sixteenth and last movement of the programme, and the second whose detector does not
register.** One rule permanently silent, four withdrawn. Unlike Jumping Jacks — where the roster
emptied because thresholds fired on a corpus with no positive class — **two of the four withdrawals
here are refutations of a CONSTRUCTION**, and they would hold on any corpus:

- the two trunk rules have **no vertical to measure against**, and the substitute is the size of the
  fault;
- the pelvic-drop rule is refuted by **three cameras filming the same instant**.

**No threshold was moved.** Both of the parent spec's disagreeing knee-lift cuts stay where they
are; the withdrawn rules' cuts live in the harness only so their evidence stays re-runnable. The one
number that changed is `min_rep_seconds`, and it comes from `base.py:55`'s own arithmetic (§4).

Reproduce — the pose corpus is NOT in the repository and cannot be committed, so the recipe is part
of the record:

```
# 1. The archive. data/EgoExo-Fitness/frames_open is a 3 GiB-split download whose `.ac` part is
#    MISSING (`.aa`, `.ab`, `.ad` present). `.aa`+`.ab` IS a contiguous gzip prefix and decodes
#    until it runs out; `.ad` sits after the hole and is unusable, which the extractor's
#    `contiguous_prefix` enforces rather than assumes.
.venv\Scripts\python.exe scripts/egoexo/extract_action_frames.py ^
  --movement "High Knee" --out <frames-dir> --report <report.json>
# 2. MediaPipe Pose, model_complexity=2, one JSON per (action, view).
.venv\Scripts\python.exe scripts/egoexo/run_pose_on_frame_dirs.py ^
  --frames-root <frames-dir> --out <pose-dir>        # --shard i --shards 6 to parallelise
# 3. Replay:
.venv\Scripts\python.exe scripts/egoexo/run_high_knee_validation.py ^
  --pose-dir <pose-dir> --json <out.json>
```

Step 1 takes ~10 min, step 2 ~20 min sharded six ways on this machine (4 698 frames), step 3
seconds.

**The reachable set is DISCOVERED, not predicted.** The extractor carries the frame ranges of all
68 judged High Knee actions and writes whatever the stream reaches before it dies. It reached 6
actions × 3 exo cameras = 18 pairs, all complete. (A prediction from the archive's record ordering
would have given the same answer, but a prediction can be wrong and this cannot.)

---

## 1. What this measures — and what it cannot

**EgoExo judges seven criteria per action and the parent spec writes five rules. ZERO pairs
overlap.** Jumping Jacks had one. The corpus judges cadence, arm rhythm, upper-body stability,
alternation, gaze, back-straightness and forefoot contact; the spec writes rules about knee height,
trunk lean in two directions, pelvic drop and stride asymmetry. Design spec §2 has both taxonomies
side by side.

So no rule's sensitivity can be measured against the checklist at all. What the corpus *can* do,
and does:

1. **False-positive rates on judged-correct performances** — two of the six actions were marked true
   on every criterion by every annotator.
2. **A zero-parameter camera control.** The three exo views film the SAME instant, so any
   disagreement between them is pure projection error. This is what refutes the pelvic rule.
3. **A reference-axis measurement.** The trunk-to-support-limb angle needs no vertical, so it can be
   measured on rolled frames and compared with the thresholds it is supposed to support.
4. **A secondary positive class from free text** (§3).

Four inferential caveats sit under every number below and are stated rather than buried:

1. **Resolution.** EgoExo distributes *preprocessed* frames at 456×256; production is phone video at
   720p+, so normalized landmark error here is roughly 2.8× production's. That inflates variance.
2. **Roll.** The two side cameras ship **rolled 90°** with no EXIF — a standing subject lies
   horizontally across the frame. Every metric here is roll-invariant by construction, which is the
   only reason there are numbers. But **MediaPipe is not roll-equivariant** (median 9.8° landmark
   shift measured elsewhere in this project), so the side-camera landmarks are degraded.
3. **n = 6 actions.** Where a conclusion rests on the corpus rather than on a construction (§5), that
   is said plainly.
4. **Cadence bias direction.** Cadence is computed over the span the repetitions occupy, not the
   whole clip — dividing by idle end-frames would report a *slower* cadence, which is the direction
   that would falsely support leaving `min_rep_seconds` alone. The bias runs against §4's
   conclusion, not with it.

## 2. Corpus

| | |
|---|---|
| judged High Knee actions in EgoExo-Fitness | 68 (120 annotations) |
| reachable from the truncated archive | **6** (`xYkvB0`×2, `yT4RK3`×3, `zOfbr6`×1) |
| (action, camera) pairs | **18**, all complete |
| frames | **4 698** |
| actions judged faultless on every criterion | **2** (`xYkvB0_action_9`, `xYkvB0_action_15`) |
| actions failing "Keep your back straight" by majority | **0 of 68** |

Pipeline properties, through the real `run_detector`:

| | |
|---|---|
| median validity rate (6-landmark gate) | **1.000** |
| pairs on the whole-clip fallback | **0 of 18** |
| repetitions segmented | **150** |
| repetitions SCORED (partials dropped by `select_reps`) | **146** |
| cadence | median **1.31 Hz**, range **0.70–2.20 Hz** |

**The view gate separates the cameras with no overlap**: `anterior_axis_length` is 0.156–0.318 on
the two side cameras and 0.027–0.044 on the frontal one. Nothing keys on `view_estimation.py`, which
this programme has measured inverted once (Sit-up) and outside its stated regime once (Leg
Abduction).

## 3. `hk_insufficient_knee_lift` — the spec's two numbers give opposite verdicts

The parent spec's rationale cites Matijašević's A-skip criterion ("the thigh of the swinging leg
reaches **45°** relative to the ground") and its heuristic implements the B-skip's ("**90°**", i.e.
the knee at hip height). Over 146 scored repetitions, on the two gated cameras:

| cut | provenance | fires on |
|---|---|---|
| knee at hip height (elevation 0.0) | the spec's **heuristic** | **100.0%, every action** |
| 45° from hanging (elevation −0.707) | the spec's **citation** | 0.0–71.1% by action (0.0–83.3% by camera) |

Observed peak thigh elevation per action (median over the gated cameras), and the free-text label:

```
                     peak    cited cut   implemented cut   free-text label
xYkvB0_action_15   -0.493      0.0%          100.0%        positive
xYkvB0_action_9    -0.513      0.0%          100.0%        positive
yT4RK3_action_2    -0.526      0.0%          100.0%        positive
yT4RK3_action_14   -0.506      7.1%          100.0%        unattributable
yT4RK3_action_9    -0.603     30.0%          100.0%        unattributable
zOfbr6_action_14   -0.742     71.1%          100.0%        negative
```

That is **40–65° of hip flexion**: real performers land *between* the source's two targets, and the
spec picked the far one. The implemented cut fires on every repetition of both faultless actions.

**And the cited cut separates in exactly the WRONG direction.** It fires on **0.0% of every action
whose comments complain about leg height**, and on 7.1–71.1% of the three whose comments do not.
The only human signal the corpus offers about this fault is *anti*-correlated with the cited
threshold on these six actions.

That is a small sample and the label is secondary, so it is not by itself a refutation — but it
removes the one argument that could have justified shipping the cited number, which was that it
happened to sort this corpus sensibly. It does not. What remains is a provenance four transfers
deep, every step of it stated in the paper: it scores the **A-skip** (a skipping drill), performed
**travelling on a track**, by participants **excluded for athletics experience**, and A-skip had "a
trivial correlation" with the sprint outcome the battery was built to predict.

An earlier draft of this note claimed the opposite direction, from reading the per-camera table
instead of running the classifier over these six actions. The harness corrected it before
publication. That is the Row-residual failure mode caught one step earlier than last time.

**The free-text positive class**, under a rule fixed before the held-out comments were read: 12 of
62 held-out actions positive, 3 of 6 disclosed — 15 of 68 overall. Secondary evidence, reported as a
weaker tier than a checklist label; it establishes the fault is real, not where the cut goes.

## 4. `min_rep_seconds` — the knob reserved fifteen movements ago is finally needed

`base.py:55` has named this movement since RS-SP1 as one that "must lower" the duration floor.
Jumping Jacks measured that *it* did not need it and left the comment alone because it also named
High Knee. Measured the same non-circular way — every returned window is at least
`min_rep_seconds` long by construction, so only differencing counts at two floors can show the floor
biting:

| floor | repetitions found |
|---|---|
| **0.15 s (shipped)** | **150** |
| 0.40 s (framework default) | 52 |

**The default discards 65.3%.** Surviving repetitions are 0.45–1.42 s each — physically ordinary, so
the low floor is not manufacturing noise.

**The corpus makes this stronger, not weaker:** 30 of 68 actions are judged FAILED on "maintain the
fastest speed possible", so this population is one humans considered *too slow*, and the default
still throws away two repetitions in three. The shipped 0.15 s is half the 0.33 s `base.py:55`
itself states — not fitted to the 1.31 Hz observed.

## 5. The two trunk rules — no vertical, and the substitute is the size of the fault

A trunk lean is an angle from the **world vertical**. Group E established that the image vertical is
not the world vertical; this corpus proves it twice over by shipping its side cameras rolled 90°.
Leg Abduction's substitute — take the vertical from the **support limb** — is the only construction
available, and here it does not hold:

> **trunk-to-support-limb angle during normal marching: 8.6–23.6°, median 13.1°** (12 gated pairs)

against thresholds of 10–15° (backward) and 15–20° (forward). Part of it is not even marching: the
stance foot sits under the hip *joint* while the axis is drawn from the pelvis *midpoint*, so the
axis is tilted by atan(half-pelvis / leg length) ≈ 6° on adult proportions before anyone moves.

**And the error runs toward one rule's firing direction:**

| cut | fires on |
|---|---|
| 10° backward (`hk_trunk_lean_back`) | **69.7% of scored frames** — 56–83% on the two faultless actions |
| 15° forward (`hk_forward_trunk_collapse`) | **0.0% of scored frames** |

An unvalidated baseline offset running toward the fault is `pushup_head_drop`'s finding and Torso
Twist's brace finding for the **third** time. What is new: here it sinks both signs at once, one by
false-firing and one by never firing at all.

**The near miss, measured.** EgoExo's "Maintain a stable upper body" (10/68 majority-failed) is the
only criterion close to these rules, and its comments describe **sway** — "excessive upper body
sway", "slightly shaking", "unstable center of gravity". Both rules read a signed **mean**:

| judged on stability | actions | pairs | median trunk lean | median SD |
|---|---|---|---|---|
| FALSE | 2 | 4 | −10.10° | 5.77° |
| TRUE | 4 | 8 | −14.77° | 6.19° |

The mean separates **the wrong way**; the variance does not separate. At n = 6 that is not a powered
null — but it is the *reading of the criterion*, not the n, that decides.

## 6. `hk_contralateral_pelvic_drop` — refuted by three simultaneous cameras

The three exo views film the same instant of the same performance, so disagreement between them is
pure projection. Restricted to the two cameras the gate admits:

| action | exo_l vs exo_r spread | frame-by-frame r |
|---|---|---|
| yT4RK3_action_2 | 0.97° | −0.483 |
| yT4RK3_action_9 | 2.72° | −0.217 |
| yT4RK3_action_14 | 5.49° | +0.116 |
| xYkvB0_action_15 | 7.90° | −0.413 |
| xYkvB0_action_9 | 8.52° | −0.026 |
| zOfbr6_action_14 | 13.68° | −0.114 |

against the spec's "> ~5–8°". **The camera moves the quantity by more than the low end of that threshold on four of
six actions**, and frame by frame the two cameras are **anti-correlated on four of six** — they
disagree about which way the pelvis is tilting at any instant, not merely about the average.

The degenerate frontal camera is excluded rather than pooled: a reading from a camera its own gate
rejects is not a second opinion, and pooling it here would *understate* the spread — i.e. bias the
evidence toward keeping the rule.

Explicitly **not** the reason: Bramah's pelvic-drop→injury association (80% higher odds per degree)
is the strongest single result any citation in this section carries. The withdrawal is about
**measurability**.

## 7. What this movement adds to the programme's method log

1. **A ninth citation failure mode, half-new.** The source states a **graded family** of targets
   (A-skip 45°, B-skip 90°) and the spec **cites one grade while implementing the other**. The
   accompanying inverted paraphrase ("45° above horizontal", when 45° to the ground is below
   horizontal) is Torso Twist's mode 7 recurring and is not claimed as new.
2. **A third variety of misleading-but-present KG node.** After Torso Twist's (faithfully describes a
   *different movement*) and Jumping Jacks' (seeded from a *blend* of two): **seeded from the wrong
   criterion of the right movement.** All four of this movement's grounding percentages reproduce
   exactly from the labels, and the fourth is attached to a criterion about alternation and speed
   while being labelled "knee lift".
3. **The graph's negative filter, perfect in both directions for the first time.** The four rules
   with no scoped node are exactly the four withdrawn; the one rule with a scoped node is exactly
   the one kept. Leg Abduction §7.3's finding, with no exceptions either way.
4. **The support-limb vertical does not transfer.** Leg Abduction established it as Group E's
   missing vertical. It fails in a marching drill, partly for a reason no performer can fix (the
   pelvis-midpoint-to-stance-foot offset).
5. **A construction that measured itself to zero.** The first draft referenced the trunk to an axis
   built by removing the trunk direction and reported a lean of identically 0.000 on all 18 pairs.
   Worth naming because it failed *loudly* — a plausible-looking near-zero would have shipped.
6. **Three simultaneous cameras are a free zero-parameter control.** Cheaper and stronger than a
   synthetic one, and available on every EgoExo movement.
7. **Sit-up's 90° roll is not a supine-filming quirk.** It recurs on a standing movement, so it is a
   property of these cameras. Any image-y heuristic is unusable on this corpus.
