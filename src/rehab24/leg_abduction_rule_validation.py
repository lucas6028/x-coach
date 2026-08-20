"""Pure helpers for replaying REHAB24-6 Ex4's labeled repetitions through the production
Leg Abduction rules.

WHAT THIS CAN AND CANNOT MEASURE -- read before quoting any number it produces. REHAB24-6 labels
each repetition `correct` or `incorrect` and NEVER states which fault occurred, so a rule firing
on an incorrect rep is not evidence it found that rep's actual error. Everything here therefore
measures whether a rule's signal CARRIES INFORMATION ABOUT REP CORRECTNESS -- not per-fault
precision. That limit is why `LEG_ABDUCTION_DETECTOR.validated` stays False even though this
harness runs to completion.

WHAT IS DIFFERENT FROM THE LUNGE HARNESS, AND IT IS NOT THE STATISTICS.
  1. `Ex4` IS THE VARIANT THE APP MODELS. Standing unilateral hip abduction, filmed upright --
     the same exercise the frontend's card art shows. Sit-up's variant mismatch and Shoulder
     Bridge's missing pixels both fail to recur.
  2. THE WORKING-SIDE RESOLVER HAS GROUND TRUTH. `exercise_subtype` records "left leg" /
     "right leg" on all 210 repetitions, so `resolve_moving_side` is scored directly. No other
     side resolver in this registry can be.
  3. ONE RULE IS SILENT AND THIS HARNESS IS WHY. `rule_insufficient_abduction_rom` never fires,
     so it has no contingency table -- but its UNDERLYING metric is scored here anyway, because
     the reason it is silent is a measurement this file produces.

TWO SECTIONS, and the split is load-bearing. Everything above the ORCHESTRATION banner is pure --
it takes frames, records and numbers, never a path -- and is unit-tested in CI while the pose
corpus under `data/` stays gitignored. Everything below the banner reads pose JSON off disk and
is therefore exercised only when the corpus is present.

`contingency`, `rank_auc`, `per_subject`, `slice_rep` and `estimate_view_for_window` are IMPORTED
from `lunge_rule_validation` rather than copied. They are movement-agnostic and that module's
published result (notes/lunge-rule-validation.md) rests on their exact behaviour, so re-deriving
them here would risk two harnesses computing "the same" statistic differently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np

from src.rehab24.lunge_rule_validation import (
    contingency,
    estimate_view_for_window,
    per_subject,
    rank_auc,
    slice_rep,
)

# Which camera affords each rule's required view. Segmentation.txt documents that a rep filmed
# `front` in cam17 is `side` in cam18 (the cameras are orthogonal and simultaneous). A trunk lean
# is a FRONTAL-plane cue, so cam17 is the one that can see it.
RULE_CAMERAS: dict[str, str] = {
    "abd_pelvic_drop_trunk_lean": "cam17",
}

# Dataset orientation -> the view label the ORACLE pass feeds the rules; `view_confidence` is
# pinned at 1.0 alongside it, since the premise of this pass is that the view is known.
#
# `front` DELIBERATELY maps to "front", a label production can NEVER emit: the production path
# calls `estimate_view_for_pose(allow_front=False)`. That is the point of the oracle pass -- it
# asks "would this rule fire if the view label were correct?", which requires bypassing the gate
# rather than reproducing it. Any oracle-pass result on a `front` rep is therefore a statement
# about the RULE, never about what a user would see.
#
# `half-profile` maps to `front_oblique` rather than `rear_oblique` because the dataset does not
# record which way the subject faced, and the two are equivalent for the only rule scored here:
# `FRONTAL_OBSERVABLE_VIEWS` contains neither, so both take the same confidence discount.
ORACLE_VIEWS: dict[str, str] = {
    "front": "front",
    "side": "side",
    "half-profile": "front_oblique",
    "profile": "side",
}
ORACLE_VIEW_CONFIDENCE = 1.0

# `exercise_subtype` -> the side `resolve_moving_side` should return. MediaPipe's landmark names
# are anatomical (LEFT_HIP is the subject's left), so this is the identity on the leg word and
# carries no camera-facing correction.
SUBTYPE_MOVING_SIDE: dict[str, str] = {
    "left leg": "left",
    "right leg": "right",
}


def moving_side_accuracy(records: Sequence[dict]) -> dict[str, int | float]:
    """How often `resolve_moving_side` names the leg `exercise_subtype` records.

    THREE OUTCOMES, NOT TWO, AND CONFLATING THEM WOULD OVERSTATE THE RESOLVER'S FAILURES.
      correct / wrong    -- it answered, on a repetition that really was segmented.
      refused_ambiguous  -- it declined: the two legs sat within
                            `MOVING_SIDE_MIN_SEPARATION_DEG` of each other. This is the
                            resolver's own coverage cost, and it silences every side-relative
                            rule for that repetition.
      refused_no_window  -- `segment_reps` found no complete repetition inside the labeled
                            window, so `run_detector` took a fallback path and the harness
                            handed the resolver NOTHING. That is a SEGMENTATION outcome
                            attributed to the resolver only by accident of ordering, and
                            counting it against the resolver would blame it for a different
                            component's behaviour.
    `coverage` is computed over the repetitions that actually reached the resolver.
    """
    correct = wrong = ambiguous = no_window = 0
    for record in records:
        expected = record.get("expected_side")
        if expected is None:
            continue
        if record.get("fallback") is not None:
            no_window += 1
            continue
        resolved = record.get("resolved_side")
        if resolved is None:
            ambiguous += 1
        elif resolved == expected:
            correct += 1
        else:
            wrong += 1
    reached = correct + wrong + ambiguous
    return {
        "correct": correct,
        "wrong": wrong,
        "refused_ambiguous": ambiguous,
        "refused_no_window": no_window,
        "reached_the_resolver": reached,
        "total": reached + no_window,
        "accuracy_when_resolved": correct / (correct + wrong) if correct + wrong else math.nan,
        "coverage": (correct + wrong) / reached if reached else math.nan,
    }


def fire_rates(scores: Sequence[float], correct: Sequence[bool], cut: float,
               *, fires_below: bool) -> dict[str, float | int]:
    """What fraction of each label class a threshold would fire on.

    THIS IS THE NUMBER THAT DECIDES A "NOT ENOUGH" RULE, and AUC is not. An AUC of 0.65 in the
    unhelpful direction and an AUC of 0.35 are the same statement; what a reader needs is how
    often the cut fires on repetitions humans judged CORRECT.
    """
    ok = [s for s, c in zip(scores, correct) if c and math.isfinite(s)]
    bad = [s for s, c in zip(scores, correct) if not c and math.isfinite(s)]
    hit = (lambda v: v < cut) if fires_below else (lambda v: v > cut)
    return {
        "cut": cut,
        "correct_n": len(ok),
        "correct_fired": sum(1 for v in ok if hit(v)),
        "correct_rate": (sum(1 for v in ok if hit(v)) / len(ok)) if ok else math.nan,
        "incorrect_n": len(bad),
        "incorrect_fired": sum(1 for v in bad if hit(v)),
        "incorrect_rate": (sum(1 for v in bad if hit(v)) / len(bad)) if bad else math.nan,
    }


def window_scores(window: list, moving: str | None) -> dict[str, float]:
    """The continuous quantity behind each rule, read off the CoreFrames the rules see.

    Both are extremes over the ACTIVE phases, matching each rule's own scope -- the trunk-lean
    rule's maximum sustained tilt and the silenced ROM rule's peak abduction. Returned even when
    the rule declines, so a rule that never fires still contributes a score to the AUC.
    """
    from src.pose.movements.leg_abduction import ACTIVE_PHASES

    scores = {"trunk_tilt_deg": math.nan, "abduction_deg": math.nan,
              "pelvic_hike_ratio": math.nan}
    if moving is None:
        return scores
    active = [f for f in window if f.valid and f.phase in ACTIVE_PHASES]
    if not active:
        return scores
    for label, key, reducer in (
        ("trunk_tilt_deg", f"{moving}_trunk_tilt_deg", max),
        ("abduction_deg", f"{moving}_abduction_deg", max),
        ("pelvic_hike_ratio", f"{moving}_pelvic_hike_ratio", max),
    ):
        values = [f.m(key) for f in active if np.isfinite(f.m(key))]
        if values:
            scores[label] = float(reducer(values))
    return scores


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation of two per-repetition signals, NaN unless both are finite.

    THIS DECIDES HOW MUCH THE UNIMPLEMENTED DISJUNCT ACTUALLY COSTS. The parent spec's rule is a
    disjunction of pelvic tilt and trunk lean, and only the second ships. If the two signals are
    near-perfectly rank-correlated then the omitted one carries no independent information and
    the decision is free; if they are not, the omission is a real loss and must be stated as one.
    """
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            shared = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                out[order[index]] = shared
            position = end + 1
        return out

    ra, rb = ranks([x for x, _ in pairs]), ranks([y for _, y in pairs])
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else math.nan


def split_by_orientation(records: Sequence[dict], fault_id: str) -> dict[str, dict[str, int]]:
    """The shipped rule's contingency table, split by the camera geometry the dataset records.

    Shoulder Bridge's census split cleanly by camera and not by repetition, and that turned out
    to be the most important thing about it. Ex4 records `front` and `half-profile` on every
    repetition, so the same question is answerable here without inferring anything.
    """
    out: dict[str, dict[str, int]] = {}
    for orientation in sorted({r["cam17_orientation"] for r in records}):
        subset = [r for r in records if r["cam17_orientation"] == orientation]
        out[orientation] = contingency(
            [fault_id in r["production_fired"] for r in subset],
            [r["correct"] for r in subset],
        )
    return out


def view_label_confusion(records: Sequence[dict]) -> dict[str, dict[str, int]]:
    """Dataset orientation -> what the production estimator actually emitted, counted.

    LEG ABDUCTION IS THE FIRST GROUP E MOVEMENT WHOSE SUBJECT STANDS UP, which puts these labels
    inside the regime `view_estimation.py`'s limit 1 leaves alone -- that limit voids the
    front/rear/oblique labels for HORIZONTAL subjects only, and it is what silenced the view
    logic in both other Group E modules. This table is the test of whether standing up is enough,
    and the answer it returns is the point of running it.
    """
    table: dict[str, dict[str, int]] = {}
    for record in records:
        row = table.setdefault(record["cam17_orientation"], {})
        row[record["view_type"]] = row.get(record["view_type"], 0) + 1
    return table


# =======================================================================================
# ORCHESTRATION -- everything below this banner reads pose JSON off disk.
# =======================================================================================

EX4_EXERCISE_ID = "4"
EX4_FPS = 30.0

# Pinned from Segmentation.csv itself, checked before any rule is run. An input that disagrees
# means the dataset, the exercise filter or the correctness polarity has moved, and every number
# downstream would be silently wrong -- so this raises rather than warns.
EX4_EXPECTED = {
    "reps": 210,
    "incorrect": 90,
    "correct": 120,
    "front": 116,
    "half_profile": 94,
    "subjects": 9,
}

POSE_FILE_SUFFIX = {
    "cam17": "-Camera17-30fps.json",
    "cam18": "-Camera18-30fps-transposed.json",
}


def is_correct(segment) -> bool:
    """`Segment.correctness` as a bool. 1 == performed CORRECTLY, 0 == incorrect."""
    return int(segment.correctness) == 1


def assert_dataset_shape(segments: Sequence) -> None:
    """Raise unless the loaded Ex4 segments match the pinned counts.

    Includes the int->bool correctness conversion, which is where a polarity inversion would
    enter. Asserting 120 correct / 90 incorrect here means an inverted conversion cannot reach
    the writeup as a plausible-looking sensitivity/specificity swap.
    """
    counts = {
        "reps": len(segments),
        "incorrect": sum(1 for s in segments if not is_correct(s)),
        "correct": sum(1 for s in segments if is_correct(s)),
        "front": sum(1 for s in segments if s.cam17_orientation == "front"),
        "half_profile": sum(1 for s in segments if s.cam17_orientation == "half-profile"),
        "subjects": len({s.person_id for s in segments}),
    }
    if counts != EX4_EXPECTED:
        raise SystemExit(f"Ex4 shape changed: expected {EX4_EXPECTED}, loaded {counts}. STOP.")
    unknown = {s.exercise_subtype for s in segments} - set(SUBTYPE_MOVING_SIDE)
    if unknown:
        raise SystemExit(f"Ex4 gained unmapped exercise_subtype values {unknown}. STOP.")


def load_pose_frames(path: Path) -> list[dict]:
    """Frames from a pose JSON, after asserting they are zero-origin and contiguous.

    `slice_rep` indexes by LIST POSITION and `Segmentation.csv`'s bounds are frame numbers.
    Those agree only if `frames[i]["frame_index"] == i` throughout; if they ever diverge, every
    rep window is silently misaligned and every number in the writeup is wrong.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = payload.get("frames") or []
    for position, frame in enumerate(frames):
        if int(frame.get("frame_index", -1)) != position:
            raise SystemExit(
                f"{path.name}: frame_index {frame.get('frame_index')!r} at list position "
                f"{position} -- frames are not zero-origin contiguous, so slice_rep would "
                "misalign every rep window. STOP."
            )
    return frames


def evaluate_rep(window_frames: list[dict], segment) -> dict:
    """One labeled repetition through the real `run_detector`, twice.

    PASS 1 uses the view label the production estimator really produces for this window; PASS 2
    uses the dataset's ground-truth orientation. The gap between them is a GATE effect rather
    than a rule effect, which is the only way to tell those two apart.
    """
    from src.pose.movements.base import run_detector
    from src.pose.movements.leg_abduction import LEG_ABDUCTION_DETECTOR, resolve_moving_side

    view_type, view_confidence = estimate_view_for_window(window_frames)
    oracle_view = ORACLE_VIEWS[segment.cam17_orientation]

    def run(view: str, confidence: float) -> tuple[dict, list, str | None]:
        result = run_detector(
            LEG_ABDUCTION_DETECTOR, window_frames, EX4_FPS, view, confidence, max_reps=None
        )
        # `run_detector` scores whichever windows it segments; the harness hands it ONE labeled
        # repetition, so the union over its detections is this repetition's verdict.
        fired = {d.fault_id: float(d.severity) for d in result.detections}
        # ON A FALLBACK PATH THE SCORE IS REFUSED, NOT APPROXIMATED. When `segment_reps` finds
        # nothing (or only partial reps) the rules are handed the WHOLE window and
        # `assign_phases` labels only its first 15% `setup` -- so ~85% of the window, including
        # the standing frames before the lift, would feed a `max()` that the segmented path
        # would never have seen. Scoring those alongside properly segmented repetitions mixes
        # two different quantities into one AUC. `scored` is therefore empty here, the scores
        # come back NaN, `rank_auc` drops them, and the count of excluded repetitions is
        # reported next to every statistic that lost them.
        scored = [
            f for rep in result.analyzed for f in result.core[rep.start : rep.end + 1]
        ]
        return fired, scored, result.fallback

    production, scored, fallback = run(view_type, view_confidence)
    oracle, _, _ = run(oracle_view, ORACLE_VIEW_CONFIDENCE)

    moving = resolve_moving_side(scored)
    valid = sum(1 for f in scored if f.valid)
    return {
        "video_id": segment.video_id,
        "repetition": segment.repetition_number,
        "person_id": segment.person_id,
        "correct": is_correct(segment),
        "cam17_orientation": segment.cam17_orientation,
        "view_type": view_type,
        "view_confidence": round(float(view_confidence), 4),
        "oracle_view": oracle_view,
        "fallback": fallback,
        "frames": len(window_frames),
        "valid_frames": valid,
        "validity_rate": valid / len(window_frames) if window_frames else 0.0,
        "expected_side": SUBTYPE_MOVING_SIDE[segment.exercise_subtype],
        "resolved_side": moving,
        "production_fired": production,
        "oracle_fired": oracle,
        "scores": window_scores(scored, moving),
    }


def evaluate_dataset(pose_dir: Path, segmentation_path: Path) -> dict:
    from src.rehab24.dataset import read_segmentation

    segments = [
        s for s in read_segmentation(segmentation_path) if s.exercise_id == EX4_EXERCISE_ID
    ]
    assert_dataset_shape(segments)

    cache: dict[str, list[dict]] = {}
    records: list[dict] = []
    skipped: list[dict] = []
    for segment in segments:
        camera = RULE_CAMERAS["abd_pelvic_drop_trunk_lean"]
        name = f"{segment.video_id}{POSE_FILE_SUFFIX[camera]}"
        if name not in cache:
            path = pose_dir / name
            if not path.exists():
                skipped.append({"video_id": segment.video_id, "reason": f"missing {name}"})
                continue
            cache[name] = load_pose_frames(path)
        window = slice_rep(cache[name], segment.first_frame, segment.last_frame)
        if not window:
            skipped.append({"video_id": segment.video_id, "reason": "empty window"})
            continue
        records.append(evaluate_rep(window, segment))

    return {"records": records, "skipped": skipped, "expected": EX4_EXPECTED}


def _auc_block(records: Sequence[dict], score_key: str, *, higher_is_worse: bool) -> dict:
    scores = [
        r["scores"].get(score_key, math.nan) if higher_is_worse
        else -r["scores"].get(score_key, math.nan)
        for r in records
    ]
    positive = [not r["correct"] for r in records]
    per = per_subject(
        list(zip(records, scores)),
        lambda pair: pair[0]["person_id"],
        lambda group: rank_auc([s for _, s in group], [not r["correct"] for r, _ in group]),
    )
    usable = [v for v in per.values() if math.isfinite(v)]
    return {
        # Stated next to every AUC so a reader can see what it was computed over. The gap is the
        # repetitions `run_detector` put on a fallback path, whose scores are deliberately NaN.
        "scored_reps": sum(1 for s in scores if math.isfinite(s)),
        "evaluated_reps": len(scores),
        "pooled_auc": rank_auc(scores, positive),
        "per_subject": {k: (None if not math.isfinite(v) else round(v, 4))
                        for k, v in sorted(per.items())},
        "per_subject_median": median(usable) if usable else math.nan,
        "per_subject_min": min(usable) if usable else math.nan,
        "per_subject_max": max(usable) if usable else math.nan,
        "subjects_scored": len(usable),
    }


def summarize(payload: dict) -> dict:
    from src.pose.movements.leg_abduction import ROM_MILD_DEG, TRUNK_LEAN_MILD_DEG

    records = payload["records"]
    correct = [r["correct"] for r in records]
    summary: dict = {
        "reps_evaluated": len(records),
        "skipped": payload["skipped"],
        "subjects": len({r["person_id"] for r in records}),
        "validity_rate_median": median([r["validity_rate"] for r in records]) if records else 0.0,
        # The price of requiring both ankles for the support limb -- two more landmarks than any
        # other Group E module needs. Reported with a low percentile, not just a median, because
        # the median hides the reps where the gate actually bites.
        "validity_rate_p10": (
            float(np.percentile([r["validity_rate"] for r in records], 10)) if records else 0.0
        ),
        "fallbacks": {},
        "moving_side": moving_side_accuracy(records),
        "view_confusion": view_label_confusion(records),
    }
    for record in records:
        key = str(record["fallback"])
        summary["fallbacks"][key] = summary["fallbacks"].get(key, 0) + 1

    rules: dict[str, dict] = {}
    for fault_id in RULE_CAMERAS:
        for pass_name in ("production", "oracle"):
            fired = [fault_id in r[f"{pass_name}_fired"] for r in records]
            rules.setdefault(fault_id, {})[pass_name] = contingency(fired, correct)
    summary["rules"] = rules
    summary["by_orientation"] = {
        fault_id: split_by_orientation(records, fault_id) for fault_id in RULE_CAMERAS
    }
    summary["signal_correlation"] = {
        "trunk_tilt_vs_pelvic_hike": spearman(
            [r["scores"]["trunk_tilt_deg"] for r in records],
            [r["scores"]["pelvic_hike_ratio"] for r in records],
        )
    }

    summary["signals"] = {
        # The shipped rule: higher tilt should mean a WORSE repetition.
        "trunk_tilt_deg": _auc_block(records, "trunk_tilt_deg", higher_is_worse=True),
        # The SILENCED rule: LOWER peak abduction is what it would fault, so the AUC is scored
        # in that direction. A value below 0.5 says the cue ranks correct repetitions as worse.
        "abduction_deg": _auc_block(records, "abduction_deg", higher_is_worse=False),
        # Not a rule -- the disjunct that is deliberately not implemented, scored so the
        # decision not to ship it rests on a number.
        "pelvic_hike_ratio": _auc_block(records, "pelvic_hike_ratio", higher_is_worse=True),
    }

    summary["fire_rates"] = {
        "trunk_tilt_deg": [
            fire_rates([r["scores"]["trunk_tilt_deg"] for r in records], correct, cut,
                       fires_below=False)
            for cut in (TRUNK_LEAN_MILD_DEG, 12.0, 15.0)
        ],
        "abduction_deg": [
            fire_rates([r["scores"]["abduction_deg"] for r in records], correct, cut,
                       fires_below=True)
            for cut in (25.0, ROM_MILD_DEG, 35.0)
        ],
    }
    return summary


def render_report(summary: dict) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    def num(value: float, places: int = 3) -> str:
        return "n/a" if value is None or not math.isfinite(value) else f"{value:.{places}f}"

    add("REHAB24-6 Ex4 (standing leg abduction) -- Leg Abduction rule validation")
    add("=" * 78)
    add(f"repetitions evaluated : {summary['reps_evaluated']} "
        f"across {summary['subjects']} subjects")
    add(f"validity rate         : median {num(summary['validity_rate_median'])} "
        f"p10 {num(summary['validity_rate_p10'])}")
    add("                        (fraction of frames carrying all 8 required landmarks; the two")
    add("                         ankles are what no other Group E module needs)")
    add(f"rep-segmentation paths: {summary['fallbacks']}")
    add("                        reps on a fallback path contribute NO score to any AUC below --")
    add("                        they were handed the whole window, which is a different quantity")
    if summary["skipped"]:
        add(f"SKIPPED               : {summary['skipped']}")
    add()

    side = summary["moving_side"]
    add("WORKING-SIDE RESOLVER vs the dataset's `exercise_subtype` -- the only side resolver in")
    add("this registry with ground truth to check against.")
    add(f"  correct {side['correct']}   wrong {side['wrong']}   "
        f"refused-ambiguous {side['refused_ambiguous']}   "
        f"of {side['reached_the_resolver']} reps that reached it")
    add(f"  accuracy when it answers {num(side['accuracy_when_resolved'])}"
        f"   coverage {num(side['coverage'])}")
    add(f"  a further {side['refused_no_window']} of {side['total']} repetitions never reached")
    add("  the resolver at all: `segment_reps` found no complete rep inside the labeled window,")
    add("  so run_detector took a fallback path. That is a SEGMENTATION outcome, not a resolver")
    add("  one, and it is counted here rather than folded into the refusals above.")
    add()

    add("VIEW ESTIMATOR vs the dataset's recorded orientation. Leg Abduction is the first Group E")
    add("movement filmed UPRIGHT, so view_estimation.py's limit 1 -- which voids these labels for")
    add("HORIZONTAL subjects -- does not apply. This is the test of whether standing up is enough.")
    for orientation, row in sorted(summary["view_confusion"].items()):
        add(f"  cam17 {orientation:14s} -> " +
            ", ".join(f"{k} x{v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1])))
    frontal = sum(
        count
        for row in summary["view_confusion"].values()
        for label, count in row.items()
        if label in {"front", "rear"}
    )
    total = sum(count for row in summary["view_confusion"].values() for count in row.values())
    add(f"  frontal-observable labels emitted: {frontal}/{total}. The shipped rule's confidence")
    add("  discount therefore applies to every repetition equally -- it is a CONSTANT here, and")
    add("  distinguishes nothing. It is not evidence that the gating works.")
    add()

    add("RULES -- POSITIVE = the repetition is INCORRECT. The dataset never names the fault, so")
    add("these say whether the signal tracks correctness, NOT whether the fault was the one.")
    add("THE TWO PASSES ARE EXPECTED TO BE IDENTICAL, and that is the finding rather than a")
    add("confirmation: the only rule here DISCOUNTS confidence off a frontal view and never gates")
    add("firing, so there is no gate effect to separate from a rule effect. A DIFFERENCE between")
    add("the passes would mean a view gate had appeared that this writeup does not describe.")
    for fault_id, passes in sorted(summary["rules"].items()):
        add(f"  {fault_id}")
        for pass_name, table in sorted(passes.items()):
            total = sum(table.values())
            fired = table["tp"] + table["fp"]
            add(f"    {pass_name:10s} fired {fired:3d}/{total:3d}   "
                f"tp {table['tp']:3d}  fp {table['fp']:3d}  "
                f"fn {table['fn']:3d}  tn {table['tn']:3d}")
    add()

    add("THE SAME RULE, SPLIT BY THE CAMERA GEOMETRY THE DATASET RECORDS.")
    for fault_id, rows in sorted(summary["by_orientation"].items()):
        add(f"  {fault_id}")
        for orientation, table in sorted(rows.items()):
            total = sum(table.values())
            add(f"    {orientation:14s} fired {table['tp'] + table['fp']:3d}/{total:3d}   "
                f"tp {table['tp']:3d}  fp {table['fp']:3d}  "
                f"fn {table['fn']:3d}  tn {table['tn']:3d}")
    add()

    add("HOW MUCH THE UNIMPLEMENTED PELVIC-TILT DISJUNCT COSTS -- rank correlation between the")
    add("shipped signal and the one deliberately left out.")
    add(f"  trunk lean vs pelvic hike: rho "
        f"{num(summary['signal_correlation']['trunk_tilt_vs_pelvic_hike'])}")
    add()

    add("SIGNALS -- threshold-free AUC, pooled AND per subject. The pooled figure is secondary:")
    add("210 repetitions come from 9 people and are not independent observations.")
    for name, block in summary["signals"].items():
        add(f"  {name:20s} pooled {num(block['pooled_auc'])}   per-subject "
            f"median {num(block['per_subject_median'])} "
            f"[{num(block['per_subject_min'])}, {num(block['per_subject_max'])}] "
            f"n={block['subjects_scored']}   "
            f"over {block['scored_reps']}/{block['evaluated_reps']} reps")
        add(f"    {block['per_subject']}")
    add()

    add("FIRE RATES -- what each candidate cut would do to repetitions humans judged CORRECT.")
    add("This, not AUC, is what decides a 'not enough' rule.")
    for name, rows in summary["fire_rates"].items():
        add(f"  {name}")
        for row in rows:
            add(f"    cut {row['cut']:7.2f}   correct-reps fire "
                f"{row['correct_fired']:3d}/{row['correct_n']:3d} "
                f"({num(row['correct_rate'], 2)})   incorrect-reps fire "
                f"{row['incorrect_fired']:3d}/{row['incorrect_n']:3d} "
                f"({num(row['incorrect_rate'], 2)})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path,
                        default=Path("data/REHAB24-6/processed/leg_abduction_pose_json"))
    parser.add_argument("--segmentation", type=Path,
                        default=Path("data/REHAB24-6/Segmentation.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/REHAB24-6/processed/leg_abduction_rule_validation.json"))
    parser.add_argument("--report-only", action="store_true",
                        help="re-print the report from --out without re-running the detector")
    args = parser.parse_args(argv)

    if args.report_only:
        with args.out.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = evaluate_dataset(args.pose_dir, args.segmentation)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)

    print(render_report(summarize(payload)))
    return 0
